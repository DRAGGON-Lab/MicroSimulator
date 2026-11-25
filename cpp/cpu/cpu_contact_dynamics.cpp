#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "core/contact_solver.hpp"
#include "core/parallel.hpp"
#include "microsimulator/contact_dynamics.hpp"

namespace microsimulator {
namespace {

using Wrench = std::array<float, 6>;

// Rows and bodies handed to one thread at a time.
constexpr std::size_t row_grain = 256;
constexpr std::size_t body_grain = 128;

[[nodiscard]] float wrench_dot(const Wrench& left, const Wrench& right) {
  float result = 0.0F;
  for (std::size_t index = 0; index < left.size(); ++index) {
    result += left[index] * right[index];
  }
  return result;
}

void add_scaled(Wrench& destination, const Wrench& source, float scale) {
  for (std::size_t index = 0; index < destination.size(); ++index) {
    destination[index] += scale * source[index];
  }
}

// Velocity and angular velocity of a body under a force and torque.
[[nodiscard]] Wrench apply_mobility(const ContactBody& body, const Wrench& wrench) {
  const Vec3 torque{wrench[3], wrench[4], wrench[5]};
  const auto angular =
      torque * body.transverse_mobility +
      body.axis * ((body.axial_mobility - body.transverse_mobility) * dot(body.axis, torque));
  return {
      wrench[0] * body.translational_mobility,
      wrench[1] * body.translational_mobility,
      wrench[2] * body.translational_mobility,
      angular.x,
      angular.y,
      angular.z,
  };
}

[[nodiscard]] float length_rate(const ContactBody& body, const GrowthLoadResponse& growth,
                                float load) {
  return body.length_rate > 0.0F ? body.length_rate * growth_load_factor(growth, load)
                                 : body.length_rate;
}

// One body's side of one row: what the row's multiplier adds to the body's
// wrench and axial load.
struct IncidenceEntry {
  Wrench jacobian{};
  float length_lever{0.0F};
  std::uint32_t row{0};
};

// The rows that touch each body, in row order, with each body's entries side
// by side. A body sums its own wrench from them, so bodies evaluate
// independently of one another, each sum runs in the same order on any number
// of threads, and a pass over the bodies reads the entries front to back.
struct Incidence {
  std::vector<std::uint32_t> offsets;
  std::vector<IncidenceEntry> entries;
};

[[nodiscard]] Incidence make_incidence(const ContactSystem& system) {
  if (system.rows.size() > std::numeric_limits<std::uint32_t>::max() / 2) {
    throw std::overflow_error("contact system has too many rows");
  }
  Incidence incidence;
  incidence.offsets.assign(system.bodies.size() + 1, 0);
  for (const auto& row : system.rows) {
    ++incidence.offsets[static_cast<std::size_t>(row.first_slot) + 1];
    if (row.second_slot != invalid_slot) {
      ++incidence.offsets[static_cast<std::size_t>(row.second_slot) + 1];
    }
  }
  for (std::size_t index = 1; index < incidence.offsets.size(); ++index) {
    incidence.offsets[index] += incidence.offsets[index - 1];
  }
  incidence.entries.resize(incidence.offsets.back());
  std::vector<std::uint32_t> cursors(incidence.offsets.begin(), incidence.offsets.end() - 1);
  for (std::size_t index = 0; index < system.rows.size(); ++index) {
    const auto& row = system.rows[index];
    const auto number = static_cast<std::uint32_t>(index);
    incidence.entries[cursors[row.first_slot]++] = {row.first_jacobian, row.first_length_lever,
                                                    number};
    if (row.second_slot != invalid_slot) {
      incidence.entries[cursors[row.second_slot]++] = {row.second_jacobian, row.second_length_lever,
                                                       number};
    }
  }
  return incidence;
}

// Scratch storage for evaluating a system, reused across solver iterations.
struct Evaluation {
  std::vector<Wrench> velocities;
  std::vector<float> loads;
  std::vector<float> length_rates;
};

// Evaluates what the multipliers `multiplier_of(row)` do to every body.
template <typename MultiplierOf>
void evaluate(const ContactSystem& system, const Incidence& incidence,
              const MultiplierOf& multiplier_of, Evaluation& evaluation) {
  const auto body_count = system.bodies.size();
  evaluation.velocities.resize(body_count);
  evaluation.loads.resize(body_count);
  evaluation.length_rates.resize(body_count);
  detail::parallel_for(body_count, body_grain, [&](std::size_t begin, std::size_t end) {
    for (std::size_t slot = begin; slot < end; ++slot) {
      const auto& body = system.bodies[slot];
      Wrench wrench{};
      auto load = 0.0F;
      for (auto index = incidence.offsets[slot]; index < incidence.offsets[slot + 1]; ++index) {
        const auto& entry = incidence.entries[index];
        const auto multiplier = multiplier_of(entry.row);
        if (multiplier == 0.0F) {
          continue;
        }
        add_scaled(wrench, entry.jacobian, multiplier);
        load += entry.length_lever * multiplier;
      }
      auto velocity = apply_mobility(body, wrench);
      for (std::size_t component = 0; component < velocity.size(); ++component) {
        velocity[component] += body.drift[component];
      }
      evaluation.velocities[slot] = velocity;
      evaluation.loads[slot] = load;
      evaluation.length_rates[slot] = length_rate(body, system.growth, load);
    }
  });
}

// The gap of one row under an evaluation of the multiplier it was made with.
[[nodiscard]] float row_gap(const ContactSystem& system, const Evaluation& evaluation,
                            std::size_t index, float multiplier) {
  const auto& row = system.rows[index];
  auto rate = wrench_dot(row.first_jacobian, evaluation.velocities[row.first_slot]) -
              row.first_length_lever * evaluation.length_rates[row.first_slot];
  if (row.second_slot != invalid_slot) {
    rate += wrench_dot(row.second_jacobian, evaluation.velocities[row.second_slot]) -
            row.second_length_lever * evaluation.length_rates[row.second_slot];
  }
  return row.offset + system.dt * rate + row.compliance * multiplier;
}

// What one pass over the rows accumulates. The residual is not a number once
// any gap is not finite.
struct RowSums {
  double step_dot_step{0.0};
  double step_dot_change{0.0};
  double change_dot_change{0.0};
  float residual{0.0F};
  bool finite{true};

  void include_residual(float multiplier, float gap) {
    residual = std::max(residual, detail::contact_row_residual(multiplier, gap));
    finite = finite && std::isfinite(gap);
  }

  [[nodiscard]] float reported_residual() const {
    return finite ? residual : std::numeric_limits<float>::quiet_NaN();
  }

  [[nodiscard]] static RowSums combine(RowSums left, const RowSums& right) {
    left.step_dot_step += right.step_dot_step;
    left.step_dot_change += right.step_dot_change;
    left.change_dot_change += right.change_dot_change;
    left.residual = std::max(left.residual, right.residual);
    left.finite = left.finite && right.finite;
    return left;
  }
};

// Evaluates the gaps of `multipliers` and returns their residual.
float evaluate_gaps(const ContactSystem& system, const Incidence& incidence,
                    std::span<const float> multipliers, Evaluation& evaluation,
                    std::vector<float>& gaps) {
  evaluate(system, incidence, [&](std::size_t index) { return multipliers[index]; }, evaluation);
  gaps.resize(system.rows.size());
  const auto sums = detail::parallel_reduce(
      system.rows.size(), row_grain, RowSums{},
      [&](std::size_t begin, std::size_t end) {
        RowSums piece;
        for (std::size_t index = begin; index < end; ++index) {
          gaps[index] = row_gap(system, evaluation, index, multipliers[index]);
          piece.include_residual(multipliers[index], gaps[index]);
        }
        return piece;
      },
      RowSums::combine);
  return sums.reported_residual();
}

void require_multiplier_count(const ContactSystem& system, std::span<const float> multipliers) {
  if (multipliers.size() != system.rows.size()) {
    throw std::invalid_argument("contact multiplier count does not match the contact rows");
  }
  for (const auto multiplier : multipliers) {
    if (!std::isfinite(multiplier) || multiplier < 0.0F) {
      throw std::invalid_argument("contact multipliers must be finite and non-negative");
    }
  }
}

}  // namespace

ContactResponse evaluate_contact_response(const ContactSystem& system,
                                          std::span<const float> multipliers) {
  validate_contact_system(system);
  require_multiplier_count(system, multipliers);
  Evaluation evaluation;
  evaluate(
      system, make_incidence(system), [&](std::size_t index) { return multipliers[index]; },
      evaluation);
  return {
      .velocities = std::move(evaluation.velocities),
      .loads = std::move(evaluation.loads),
      .length_rates = std::move(evaluation.length_rates),
  };
}

std::vector<float> evaluate_contact_gaps(const ContactSystem& system,
                                         std::span<const float> multipliers) {
  validate_contact_system(system);
  require_multiplier_count(system, multipliers);
  Evaluation evaluation;
  std::vector<float> gaps;
  std::ignore = evaluate_gaps(system, make_incidence(system), multipliers, evaluation, gaps);
  return gaps;
}

float contact_system_diagonal_bound(const ContactSystem& system) {
  const auto diagonal = [](const ContactBody& body, const Wrench& jacobian) {
    return wrench_dot(jacobian, apply_mobility(body, jacobian));
  };
  auto bound = 0.0F;
  for (const auto& row : system.rows) {
    auto entry = diagonal(system.bodies[row.first_slot], row.first_jacobian);
    if (row.second_slot != invalid_slot) {
      entry += diagonal(system.bodies[row.second_slot], row.second_jacobian);
    }
    bound = std::max(bound, system.dt * entry + row.compliance);
  }
  return bound;
}

ContactSolveResult solve_contact_system_cpu(const ContactSystem& system,
                                            std::span<const float> initial_multipliers,
                                            const ContactSolveParameters& parameters) {
  validate_contact_system(system);
  require_multiplier_count(system, initial_multipliers);
  if (!std::isfinite(parameters.tolerance) || parameters.tolerance <= 0.0F) {
    throw std::invalid_argument("contact solve tolerance must be finite and positive");
  }

  ContactSolveResult result;
  result.multipliers.assign(initial_multipliers.begin(), initial_multipliers.end());
  if (system.rows.empty()) {
    return result;
  }
  const auto bound = contact_system_diagonal_bound(system);
  if (!std::isfinite(bound) || bound <= 0.0F) {
    throw std::invalid_argument("contact system has no positive diagonal");
  }

  const auto incidence = make_incidence(system);
  const auto row_count = system.rows.size();
  Evaluation evaluation;
  std::vector<float> gaps;
  std::vector<float> next_multipliers(row_count);
  std::vector<float> next_gaps(row_count);
  auto step_length = 0.0F;
  std::uint32_t iteration = 0;
  // One projected step in two passes: the bodies respond to the projected
  // multipliers, then the rows store those multipliers and their gaps.
  const auto step = [&](float length) {
    const auto projected = [&](std::size_t index) {
      return std::max(0.0F, result.multipliers[index] - length * gaps[index]);
    };
    evaluate(system, incidence, projected, evaluation);
    const auto sums = detail::parallel_reduce(
        row_count, row_grain, RowSums{},
        [&](std::size_t begin, std::size_t end) {
          RowSums piece;
          for (std::size_t index = begin; index < end; ++index) {
            const auto multiplier = projected(index);
            const auto gap = row_gap(system, evaluation, index, multiplier);
            const auto moved = static_cast<double>(multiplier) - result.multipliers[index];
            const auto change = static_cast<double>(gap) - gaps[index];
            piece.step_dot_step += moved * moved;
            piece.step_dot_change += moved * change;
            piece.change_dot_change += change * change;
            piece.include_residual(multiplier, gap);
            next_multipliers[index] = multiplier;
            next_gaps[index] = gap;
          }
          return piece;
        },
        RowSums::combine);
    result.multipliers.swap(next_multipliers);
    gaps.swap(next_gaps);
    return detail::ContactSolverSums{
        .step_dot_step = sums.step_dot_step,
        .step_dot_change = sums.step_dot_change,
        .change_dot_change = sums.change_dot_change,
        .residual = sums.reported_residual(),
    };
  };
  const detail::ContactSolverOperations operations{
      .initialize =
          [&](float first_step_length) {
            step_length = first_step_length;
            return evaluate_gaps(system, incidence, result.multipliers, evaluation, gaps);
          },
      .iterate =
          [&](std::uint32_t count) {
            return detail::iterate_contact_steps(count, parameters.tolerance, step_length,
                                                 iteration, step);
          },
  };
  result.report = detail::solve_contact_system(operations, 1.0F / bound, parameters);
  return result;
}

}  // namespace microsimulator
