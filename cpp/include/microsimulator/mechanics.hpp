#pragma once

#include "microsimulator/types.hpp"

namespace microsimulator {

struct FlowDriftParameters {
  // Largest turn of a cell's direction in one substep of the drift.
  float max_rotation_radians{0.0872664626F};
};

void validate_flow_drift_parameters(const FlowDriftParameters& parameters);

// The axis-angle rotation of a unit direction by a rotation vector, with the
// angle capped at ``max_rotation``.
[[nodiscard]] Vec3 rotate_axis_angle(Vec3 direction, Vec3 rotation, float max_rotation);

}  // namespace microsimulator
