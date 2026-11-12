#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "microsimulator/mechanics.hpp"

namespace microsimulator {
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

void validate_growth_limit_parameters(const GrowthLimitParameters& parameters) {
  if (!std::isfinite(parameters.free_overlap) || parameters.free_overlap < 0.0F ||
      !std::isfinite(parameters.stall_overlap) ||
      parameters.stall_overlap <= parameters.free_overlap) {
    throw std::invalid_argument(
        "growth limit overlaps must be finite, non-negative, and increasing");
  }
}

std::vector<float> growth_limit_factors(const WorldState& state, const ContactGraph& contacts,
                                        const GrowthLimitParameters& parameters) {
  validate_growth_limit_parameters(parameters);
  if (contacts.cell_count() != state.size()) {
    throw std::invalid_argument("contact graph and world state cell counts disagree");
  }
  std::vector<float> deepest(state.size(), 0.0F);
  for (const auto& contact : contacts.contacts()) {
    const auto overlap = -contact.signed_separation;
    for (const auto slot : {contact.first_slot, contact.second_slot}) {
      auto& depth = deepest[static_cast<std::size_t>(slot)];
      depth = std::max(depth, overlap);
    }
  }
  const auto radii = state.geometry_state().radii;
  const auto band = parameters.stall_overlap - parameters.free_overlap;
  std::vector<float> factors(state.size(), 1.0F);
  for (std::size_t slot = 0; slot < factors.size(); ++slot) {
    if (radii[slot] <= 0.0F) {
      continue;
    }
    const auto excess = (deepest[slot] / radii[slot]) - parameters.free_overlap;
    factors[slot] = std::clamp(1.0F - (excess / band), 0.0F, 1.0F);
  }
  return factors;
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

  const auto cells = state.cells();
  std::vector<GeometryUpdate> updates;
  updates.reserve(cells.size());
  for (std::size_t index = 0; index < cells.size(); ++index) {
    const auto& cell = cells[index];
    const auto& correction = result.corrections[index];
    if (!finite(correction.translation) || !finite(correction.rotation) ||
        !std::isfinite(correction.length)) {
      throw std::invalid_argument("mechanics correction must be finite");
    }
    const auto desired_increment =
        desired_length_increments.empty() ? 0.0F : desired_length_increments[index];
    if (!std::isfinite(desired_increment) || desired_increment < 0.0F) {
      throw std::invalid_argument("desired length increments must be finite and non-negative");
    }
    const auto applied_length_increment =
        desired_length_increments.empty()
            ? 0.0F
            : (cell.fixed ? desired_increment
                          : std::max(0.0F, desired_increment + correction.length));
    const auto new_position = cell.fixed ? cell.position : cell.position + correction.translation;
    const auto new_direction = cell.fixed ? cell.direction
                                          : rotate_axis_angle(cell.direction, correction.rotation,
                                                              parameters.max_rotation_radians);
    const auto new_length = cell.length + applied_length_increment;
    if (!finite(new_position) || !finite(new_direction) || !std::isfinite(new_length)) {
      throw std::overflow_error("mechanics integration produced non-finite geometry");
    }
    updates.push_back({cell.slot, new_position, new_direction, new_length});
  }

  for (const auto& update : updates) {
    state.set_cell_geometry(update.slot, update.position, update.direction, update.length);
  }
}

}  // namespace microsimulator
