#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/contact_solver.hpp"
#include "microsimulator/contact_dynamics.hpp"

namespace microsimulator {
namespace {

using Wrench = std::array<float, 6>;

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

// Scratch storage for evaluating a system, reused across solver iterations.
struct Evaluation {
  std::vector<Wrench> wrenches;
  std::vector<Wrench> velocities;
  std::vector<float> loads;
  std::vector<float> length_rates;
};

void evaluate(const ContactSystem& system, std::span<const float> multipliers,
              Evaluation& evaluation) {
  const auto body_count = system.bodies.size();
  evaluation.wrenches.assign(body_count, Wrench{});
  evaluation.loads.assign(body_count, 0.0F);
  for (std::size_t index = 0; index < system.rows.size(); ++index) {
    const auto& row = system.rows[index];
    const auto multiplier = multipliers[index];
    if (multiplier == 0.0F) {
      continue;
    }
    add_scaled(evaluation.wrenches[row.first_slot], row.first_jacobian, multiplier);
    evaluation.loads[row.first_slot] += row.first_length_lever * multiplier;
    if (row.second_slot != invalid_slot) {
      add_scaled(evaluation.wrenches[row.second_slot], row.second_jacobian, multiplier);
      evaluation.loads[row.second_slot] += row.second_length_lever * multiplier;
    }
  }
  evaluation.velocities.resize(body_count);
  evaluation.length_rates.resize(body_count);
  for (std::size_t index = 0; index < body_count; ++index) {
    evaluation.velocities[index] = apply_mobility(system.bodies[index], evaluation.wrenches[index]);
    for (std::size_t component = 0; component < 6; ++component) {
      evaluation.velocities[index][component] += system.bodies[index].drift[component];
    }
    evaluation.length_rates[index] =
        length_rate(system.bodies[index], system.growth, evaluation.loads[index]);
  }
}

void evaluate_gaps(const ContactSystem& system, std::span<const float> multipliers,
                   Evaluation& evaluation, std::vector<float>& gaps) {
  evaluate(system, multipliers, evaluation);
  gaps.resize(system.rows.size());
  for (std::size_t index = 0; index < system.rows.size(); ++index) {
    const auto& row = system.rows[index];
    auto rate = wrench_dot(row.first_jacobian, evaluation.velocities[row.first_slot]) -
                row.first_length_lever * evaluation.length_rates[row.first_slot];
    if (row.second_slot != invalid_slot) {
      rate += wrench_dot(row.second_jacobian, evaluation.velocities[row.second_slot]) -
              row.second_length_lever * evaluation.length_rates[row.second_slot];
    }
    gaps[index] = row.offset + system.dt * rate + row.compliance * multipliers[index];
  }
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
  evaluate(system, multipliers, evaluation);
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
  evaluate_gaps(system, multipliers, evaluation, gaps);
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

  Evaluation evaluation;
  std::vector<float> gaps;
  std::vector<float> next_multipliers(system.rows.size());
  std::vector<float> next_gaps;
  const auto residual_of = [](std::span<const float> multipliers, std::span<const float> values) {
    auto residual = 0.0F;
    for (std::size_t index = 0; index < multipliers.size(); ++index) {
      residual =
          std::max(residual, detail::contact_row_residual(multipliers[index], values[index]));
      if (!std::isfinite(values[index])) {
        return std::numeric_limits<float>::quiet_NaN();
      }
    }
    return residual;
  };
  auto step_length = 0.0F;
  std::uint32_t iteration = 0;
  const auto step = [&](float length) {
    for (std::size_t index = 0; index < next_multipliers.size(); ++index) {
      next_multipliers[index] = std::max(0.0F, result.multipliers[index] - length * gaps[index]);
    }
    evaluate_gaps(system, next_multipliers, evaluation, next_gaps);
    detail::ContactSolverSums sums;
    for (std::size_t index = 0; index < next_multipliers.size(); ++index) {
      const auto moved = static_cast<double>(next_multipliers[index]) - result.multipliers[index];
      const auto change = static_cast<double>(next_gaps[index]) - gaps[index];
      sums.step_dot_step += moved * moved;
      sums.step_dot_change += moved * change;
      sums.change_dot_change += change * change;
    }
    sums.residual = residual_of(next_multipliers, next_gaps);
    result.multipliers.swap(next_multipliers);
    gaps.swap(next_gaps);
    return sums;
  };
  const detail::ContactSolverOperations operations{
      .initialize =
          [&](float first_step_length) {
            step_length = first_step_length;
            evaluate_gaps(system, result.multipliers, evaluation, gaps);
            return residual_of(result.multipliers, gaps);
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
