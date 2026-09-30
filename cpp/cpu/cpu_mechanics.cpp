#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cm/mechanics.hpp"

namespace cm {
namespace {

constexpr std::size_t degrees_of_freedom = 7;
using Dofs = std::array<float, degrees_of_freedom>;
using DofVector = std::vector<Dofs>;

struct ContactRow {
  Slot first_slot;
  Slot second_slot{invalid_slot};
  Dofs first;
  Dofs second{};
  float right_hand_side;
};

[[nodiscard]] bool finite(const Dofs& value) {
  for (const auto component : value) {
    if (!std::isfinite(component)) {
      return false;
    }
  }

  return true;
}

[[nodiscard]] float dof_dot(const Dofs& left, const Dofs& right) {
  float result = 0.0F;

  for (std::size_t index = 0; index < degrees_of_freedom; ++index) {
    result += left[index] * right[index];
  }

  return result;
}

[[nodiscard]] float vector_dot(const DofVector& left, const DofVector& right) {
  float result = 0.0F;

  for (std::size_t index = 0; index < left.size(); ++index) {
    result += dof_dot(left[index], right[index]);
  }

  return result;
}

void add_scaled(Dofs& destination, const Dofs& source, float scale) {
  for (std::size_t index = 0; index < degrees_of_freedom; ++index) {
    destination[index] += scale * source[index];
  }
}

void add_scaled(DofVector& destination, const DofVector& source, float scale) {
  for (std::size_t index = 0; index < destination.size(); ++index) {
    add_scaled(destination[index], source[index], scale);
  }
}

[[nodiscard]] float residual_rms(const DofVector& residual) {
  if (residual.empty()) {
    return 0.0F;
  }

  return std::sqrt(vector_dot(residual, residual) / static_cast<float>(residual.size()));
}

[[nodiscard]] Dofs make_jacobian(const Vec3& normal, const Vec3& arm, const Vec3& axis,
                                 float total_length, float weight) {
  const auto angular = cross(arm, normal);
  const auto length = dot(axis, arm) * dot(axis, normal) / total_length;

  return {
      weight * normal.x,  weight * normal.y,  weight * normal.z, weight * angular.x,
      weight * angular.y, weight * angular.z, weight * length,
  };
}

[[nodiscard]] Dofs flatten(const CellCorrection& correction) {
  return {
      correction.translation.x, correction.translation.y, correction.translation.z,
      correction.rotation.x,    correction.rotation.y,    correction.rotation.z,
      correction.length,
  };
}

[[nodiscard]] CellCorrection unflatten(const Dofs& value) {
  return {
      .translation = {value[0], value[1], value[2]},
      .rotation = {value[3], value[4], value[5]},
      .length = value[6],
  };
}

[[nodiscard]] DofVector flatten(std::span<const CellCorrection> values) {
  DofVector result;
  result.reserve(values.size());

  for (const auto& value : values) {
    const auto flattened = flatten(value);

    if (!finite(flattened)) {
      throw std::invalid_argument("mechanics input correction must be finite");
    }

    result.push_back(flattened);
  }

  return result;
}

[[nodiscard]] std::vector<CellCorrection> unflatten(const DofVector& values) {
  std::vector<CellCorrection> result;
  result.reserve(values.size());

  for (const auto& value : values) {
    result.push_back(unflatten(value));
  }

  return result;
}

class CpuMechanicsSystem {
 public:
  CpuMechanicsSystem(const WorldState& state, const ContactGraph& contacts,
                     const ExternalContactGraph& external_contacts,
                     const MechanicsParameters& parameters)
      : geometry_(state.geometry_state()), parameters_(parameters) {
    fixed_ = state.cell_attributes().fixed;
    state.validate();
    validate_mechanics_parameters(parameters_);

    if (contacts.cell_count() != geometry_.size()) {
      throw std::invalid_argument("contact graph and world state cell counts disagree");
    }

    if (external_contacts.cell_count() != geometry_.size()) {
      throw std::invalid_argument("external contact graph and world state cell counts disagree");
    }

    if (external_contacts.size() > std::numeric_limits<std::size_t>::max() - contacts.size()) {
      throw std::overflow_error("mechanics contact row count overflow");
    }

    rows_.reserve(contacts.size() + external_contacts.size());

    for (const auto& contact : contacts.contacts()) {
      const auto first = static_cast<std::size_t>(contact.first_slot);
      const auto second = static_cast<std::size_t>(contact.second_slot);

      if (geometry_.ids[first] != contact.first_id || geometry_.ids[second] != contact.second_id) {
        throw std::invalid_argument("contact graph identifiers do not match current state slots");
      }

      const Vec3 first_center{geometry_.position_x[first], geometry_.position_y[first],
                              geometry_.position_z[first]};
      const Vec3 second_center{geometry_.position_x[second], geometry_.position_y[second],
                               geometry_.position_z[second]};
      const Vec3 first_axis{geometry_.direction_x[first], geometry_.direction_y[first],
                            geometry_.direction_z[first]};
      const Vec3 second_axis{geometry_.direction_x[second], geometry_.direction_y[second],
                             geometry_.direction_z[second]};
      const auto first_total_length = geometry_.lengths[first] + 2.0F * geometry_.radii[first];
      const auto second_total_length = geometry_.lengths[second] + 2.0F * geometry_.radii[second];

      rows_.push_back({
          .first_slot = contact.first_slot,
          .second_slot = contact.second_slot,
          .first = make_jacobian(contact.normal, contact.point_on_first - first_center, first_axis,
                                 first_total_length, contact.weight),
          .second = make_jacobian(contact.normal, contact.point_on_first - second_center,
                                  second_axis, second_total_length, contact.weight),
          .right_hand_side = contact.weight * contact.signed_separation,
      });
    }

    for (const auto& contact : external_contacts.contacts()) {
      const auto cell = static_cast<std::size_t>(contact.cell_slot);

      if (geometry_.ids[cell] != contact.cell_id) {
        throw std::invalid_argument(
            "external contact graph identifiers do not match current state slots");
      }

      const Vec3 center{geometry_.position_x[cell], geometry_.position_y[cell],
                        geometry_.position_z[cell]};
      const Vec3 axis{geometry_.direction_x[cell], geometry_.direction_y[cell],
                      geometry_.direction_z[cell]};
      const auto total_length = geometry_.lengths[cell] + 2.0F * geometry_.radii[cell];
      rows_.push_back({
          .first_slot = contact.cell_slot,
          .first = make_jacobian(contact.normal, contact.point_on_cell - center, axis, total_length,
                                 contact.weight),
          .right_hand_side = contact.weight * contact.signed_separation,
      });
    }
  }

  [[nodiscard]] std::size_t cell_count() const noexcept {
    return geometry_.size();
  }

  void apply(const DofVector& input, DofVector& output) const {
    auto projected = input;

    for (std::size_t index = 0; index < cell_count(); ++index) {
      if (fixed_[index] != 0) {
        projected[index] = {};
      }
    }

    output.assign(cell_count(), Dofs{});

    for (const auto& row : rows_) {
      const auto first = static_cast<std::size_t>(row.first_slot);
      auto row_value = dof_dot(row.first, projected[first]);

      if (row.second_slot != invalid_slot) {
        row_value -= dof_dot(row.second, projected[static_cast<std::size_t>(row.second_slot)]);
      }

      add_scaled(output[first], row.first, row_value);

      if (row.second_slot != invalid_slot) {
        add_scaled(output[static_cast<std::size_t>(row.second_slot)], row.second, -row_value);
      }
    }

    const auto regularization = 1.0F / parameters_.gamma;

    for (std::size_t index = 0; index < cell_count(); ++index) {
      if (fixed_[index] != 0) {
        output[index] = input[index];
        continue;
      }

      const Vec3 axis{geometry_.direction_x[index], geometry_.direction_y[index],
                      geometry_.direction_z[index]};
      const Vec3 rotation{input[index][3], input[index][4], input[index][5]};
      const auto total_length = geometry_.lengths[index] + 2.0F * geometry_.radii[index];
      const auto mass = parameters_.mu_a * total_length;
      const auto radius = geometry_.radii[index];
      const auto axial_inertia = 0.5F * mass * radius * radius;
      const auto transverse_inertia =
          mass * ((total_length * total_length) + 3.0F * radius * radius) / 12.0F;
      const auto inertia_rotation =
          rotation * transverse_inertia +
          axis * ((axial_inertia - transverse_inertia) * dot(axis, rotation));

      output[index][0] += regularization * mass * input[index][0];
      output[index][1] += regularization * mass * input[index][1];
      output[index][2] += regularization * mass * input[index][2];
      output[index][3] += regularization * inertia_rotation.x;
      output[index][4] += regularization * inertia_rotation.y;
      output[index][5] += regularization * inertia_rotation.z;
      output[index][6] += input[index][6];
    }
  }

  [[nodiscard]] DofVector right_hand_side() const {
    DofVector result(cell_count());

    for (const auto& row : rows_) {
      const auto first = static_cast<std::size_t>(row.first_slot);
      add_scaled(result[first], row.first, row.right_hand_side);

      if (row.second_slot != invalid_slot) {
        add_scaled(result[static_cast<std::size_t>(row.second_slot)], row.second,
                   -row.right_hand_side);
      }
    }

    for (std::size_t index = 0; index < cell_count(); ++index) {
      if (fixed_[index] != 0) {
        result[index] = {};
      }
    }

    return result;
  }

 private:
  CellGeometryView geometry_;
  std::span<const std::uint8_t> fixed_;
  MechanicsParameters parameters_;
  std::vector<ContactRow> rows_;
};

[[nodiscard]] DofVector exact_residual(const CpuMechanicsSystem& system,
                                       const DofVector& right_hand_side,
                                       const DofVector& solution) {
  DofVector applied;
  system.apply(solution, applied);
  auto residual = right_hand_side;
  add_scaled(residual, applied, -1.0F);

  return residual;
}

[[nodiscard]] std::uint32_t iteration_limit(const MechanicsParameters& parameters,
                                            std::size_t cell_count) {
  if (parameters.max_iterations != 0) {
    return parameters.max_iterations;
  }

  if (cell_count > std::numeric_limits<std::uint32_t>::max() / degrees_of_freedom) {
    throw std::overflow_error("default mechanics iteration limit exceeds uint32");
  }

  return static_cast<std::uint32_t>(cell_count * degrees_of_freedom);
}

}  // namespace

void validate_mechanics_parameters(const MechanicsParameters& parameters) {
  if (!std::isfinite(parameters.mu_a) || parameters.mu_a <= 0.0F) {
    throw std::invalid_argument("mechanics mu_a must be finite and positive");
  }

  if (!std::isfinite(parameters.gamma) || parameters.gamma <= 0.0F) {
    throw std::invalid_argument("mechanics gamma must be finite and positive");
  }

  if (!std::isfinite(parameters.residual_rms_tolerance) ||
      parameters.residual_rms_tolerance < 0.0F) {
    throw std::invalid_argument("mechanics residual tolerance must be finite and non-negative");
  }
}

std::vector<CellCorrection> apply_mechanics_operator_cpu(
    const WorldState& state, const ContactGraph& contacts,
    const ExternalContactGraph& external_contacts, std::span<const CellCorrection> input,
    const MechanicsParameters& parameters) {
  const CpuMechanicsSystem system(state, contacts, external_contacts, parameters);

  if (input.size() != system.cell_count()) {
    throw std::invalid_argument("mechanics input size does not match the world state");
  }

  const auto flat_input = flatten(input);
  DofVector output;
  system.apply(flat_input, output);

  return unflatten(output);
}

std::vector<CellCorrection> apply_mechanics_operator_cpu(const WorldState& state,
                                                         const ContactGraph& contacts,
                                                         std::span<const CellCorrection> input,
                                                         const MechanicsParameters& parameters) {
  return apply_mechanics_operator_cpu(state, contacts, ExternalContactGraph(state.size(), {}),
                                      input, parameters);
}

std::vector<CellCorrection> build_mechanics_rhs_cpu(const WorldState& state,
                                                    const ContactGraph& contacts,
                                                    const ExternalContactGraph& external_contacts,
                                                    const MechanicsParameters& parameters) {
  const CpuMechanicsSystem system(state, contacts, external_contacts, parameters);

  return unflatten(system.right_hand_side());
}

std::vector<CellCorrection> build_mechanics_rhs_cpu(const WorldState& state,
                                                    const ContactGraph& contacts,
                                                    const MechanicsParameters& parameters) {
  return build_mechanics_rhs_cpu(state, contacts, ExternalContactGraph(state.size(), {}),
                                 parameters);
}

MechanicsSolveResult solve_cell_mechanics_cpu(const WorldState& state, const ContactGraph& contacts,
                                              const ExternalContactGraph& external_contacts,
                                              const MechanicsParameters& parameters) {
  const CpuMechanicsSystem system(state, contacts, external_contacts, parameters);
  auto right_hand_side = system.right_hand_side();
  DofVector solution(system.cell_count());
  auto residual = right_hand_side;
  auto search_direction = residual;

  MechanicsSolveResult result;
  result.corrections.resize(system.cell_count());
  result.report.initial_residual_rms = residual_rms(residual);
  result.report.final_residual_rms = result.report.initial_residual_rms;

  if (!std::isfinite(result.report.initial_residual_rms)) {
    result.report.status = SolverStatus::breakdown;
    result.report.breakdown = SolverBreakdown::non_finite_residual;

    return result;
  }

  if (result.report.initial_residual_rms <= parameters.residual_rms_tolerance) {
    return result;
  }

  result.report.status = SolverStatus::iteration_limit;
  auto residual_squared = vector_dot(residual, residual);
  const auto maximum_iterations = iteration_limit(parameters, system.cell_count());
  DofVector applied;

  for (std::uint32_t iteration = 0; iteration < maximum_iterations; ++iteration) {
    system.apply(search_direction, applied);
    const auto curvature = vector_dot(search_direction, applied);

    if (!std::isfinite(curvature)) {
      result.report.status = SolverStatus::breakdown;
      result.report.breakdown = SolverBreakdown::non_finite_curvature;
      break;
    }

    if (curvature <= 0.0F) {
      result.report.status = SolverStatus::breakdown;
      result.report.breakdown = SolverBreakdown::non_positive_curvature;
      break;
    }

    const auto alpha = residual_squared / curvature;
    add_scaled(solution, search_direction, alpha);
    add_scaled(residual, applied, -alpha);
    result.report.iterations = iteration + 1;

    const auto next_residual_squared = vector_dot(residual, residual);
    const auto recurrence_rms =
        std::sqrt(next_residual_squared / static_cast<float>(system.cell_count()));

    if (!std::isfinite(recurrence_rms)) {
      result.report.status = SolverStatus::breakdown;
      result.report.breakdown = SolverBreakdown::non_finite_residual;
      break;
    }

    if (recurrence_rms <= parameters.residual_rms_tolerance) {
      residual = exact_residual(system, right_hand_side, solution);
      const auto recomputed_rms = residual_rms(residual);

      if (!std::isfinite(recomputed_rms)) {
        result.report.status = SolverStatus::breakdown;
        result.report.breakdown = SolverBreakdown::non_finite_residual;
        break;
      }

      if (recomputed_rms <= parameters.residual_rms_tolerance) {
        result.report.status = SolverStatus::converged;
        break;
      }

      search_direction = residual;
      residual_squared = vector_dot(residual, residual);
      continue;
    }

    const auto beta = next_residual_squared / residual_squared;

    for (std::size_t index = 0; index < search_direction.size(); ++index) {
      for (std::size_t component = 0; component < degrees_of_freedom; ++component) {
        search_direction[index][component] =
            residual[index][component] + beta * search_direction[index][component];
      }
    }

    residual_squared = next_residual_squared;
  }

  residual = exact_residual(system, right_hand_side, solution);
  result.report.final_residual_rms = residual_rms(residual);

  if (!std::isfinite(result.report.final_residual_rms) &&
      result.report.status != SolverStatus::breakdown) {
    result.report.status = SolverStatus::breakdown;
    result.report.breakdown = SolverBreakdown::non_finite_residual;
  }

  result.corrections = unflatten(solution);

  return result;
}

MechanicsSolveResult solve_cell_mechanics_cpu(const WorldState& state, const ContactGraph& contacts,
                                              const MechanicsParameters& parameters) {
  return solve_cell_mechanics_cpu(state, contacts, ExternalContactGraph(state.size(), {}),
                                  parameters);
}

}  // namespace cm
