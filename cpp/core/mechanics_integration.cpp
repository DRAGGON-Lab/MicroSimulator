#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "microsimulator/mechanics.hpp"

namespace microsimulator {

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

void validate_flow_drift_parameters(const FlowDriftParameters& parameters) {
  if (!std::isfinite(parameters.max_rotation_radians) || parameters.max_rotation_radians < 0.0F) {
    throw std::invalid_argument("flow drift rotation limit must be finite and non-negative");
  }
}

}  // namespace microsimulator
