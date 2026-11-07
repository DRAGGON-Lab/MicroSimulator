#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "cm/mechanics.hpp"

namespace cm {
namespace {

struct GeometryUpdate {
  Slot slot;
  Vec3 position;
  Vec3 direction;
  float length;
};

bool finite(const Vec3& value) {
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

}  // namespace

Vec3 rotate_axis_angle(Vec3 direction, Vec3 rotation, float max_rotation) {
  const auto magnitude = norm(rotation);
  if (magnitude <= 1.0e-12F || max_rotation == 0.0F) {
    return direction;
  }
  const auto angle = std::min(magnitude, max_rotation);
  const auto axis = rotation * (1.0F / magnitude);
  const auto cosine = std::cos(angle);
  const auto sine = std::sin(angle);
  return normalized(direction * cosine + cross(axis, direction) * sine +
                    axis * (dot(axis, direction) * (1.0F - cosine)));
}

void validate_mechanics_integration_parameters(const MechanicsIntegrationParameters& parameters) {
  if (!std::isfinite(parameters.max_rotation_radians) || parameters.max_rotation_radians < 0.0F) {
    throw std::invalid_argument("mechanics rotation limit must be finite and non-negative");
  }
}

void integrate_mechanics_result(WorldState& state, const MechanicsSolveResult& result,
                                const MechanicsIntegrationParameters& parameters,
                                std::span<const float> desired_length_increments) {
  validate_mechanics_integration_parameters(parameters);
  state.validate();
  if (result.corrections.size() != state.size()) {
    throw std::invalid_argument("mechanics correction count does not match the world state");
  }
  if (!desired_length_increments.empty() && desired_length_increments.size() != state.size()) {
    throw std::invalid_argument("desired length increment count does not match the world state");
  }
  if (parameters.require_convergence && result.report.status != SolverStatus::converged) {
    throw std::runtime_error("mechanics corrections require a converged solver result");
  }

  const auto geometry = state.geometry_state();
  const auto attributes = state.cell_attributes();
  std::vector<GeometryUpdate> updates;
  updates.reserve(geometry.size());
  for (std::size_t index = 0; index < geometry.size(); ++index) {
    const Vec3 position{geometry.position_x[index], geometry.position_y[index],
                        geometry.position_z[index]};
    const Vec3 direction{geometry.direction_x[index], geometry.direction_y[index],
                         geometry.direction_z[index]};
    const bool fixed = attributes.fixed[index] != 0;
    const auto& correction = result.corrections[index];
    if (!finite(correction.translation) || !finite(correction.rotation) ||
        !std::isfinite(correction.length)) {
      throw std::invalid_argument("mechanics correction must be finite");
    }
    const auto desired_increment =
        desired_length_increments.empty() ? 0.0F : desired_length_increments[index];
    if (!std::isfinite(desired_increment)) {
      throw std::invalid_argument("desired length increments must be finite");
    }
    float applied_length_increment = 0.0F;
    if (!desired_length_increments.empty()) {
      applied_length_increment =
          fixed || desired_increment < 0.0F
              ? desired_increment
              : std::max(0.0F, desired_increment + correction.length);
    }
    const auto new_position = fixed ? position : position + correction.translation;
    const auto new_direction = fixed ? direction
                                     : rotate_axis_angle(direction, correction.rotation,
                                                         parameters.max_rotation_radians);
    const auto new_length = geometry.lengths[index] + applied_length_increment;
    if (!finite(new_position) || !finite(new_direction) || !std::isfinite(new_length) ||
        new_length < 0.0F) {
      throw std::overflow_error("mechanics integration produced invalid geometry");
    }
    updates.push_back({static_cast<Slot>(index), new_position, new_direction, new_length});
  }

  for (const auto& update : updates) {
    state.set_cell_geometry(update.slot, update.position, update.direction, update.length);
  }
}

}  // namespace cm
