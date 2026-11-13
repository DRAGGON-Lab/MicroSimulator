#pragma once

#include "microsimulator/backend.hpp"
#include "microsimulator/constraints.hpp"
#include "microsimulator/contact_dynamics.hpp"
#include "microsimulator/contact_graph.hpp"
#include "microsimulator/world_state.hpp"

namespace microsimulator::detail {

// Advances cell positions, orientations, and lengths over `dt` with growth and
// contact solved together. Each substep finds the cells and walls within the
// candidate margin, solves for the non-negative contact forces that keep every
// gap open while the cells elongate under their loads, and moves the cells.
// Contacts that the move opens are added to the same solve until the geometry
// is free of overlap, and a substep whose solve fails is retried at half its
// length. The activation margins of the contact and constraint parameters are
// replaced by the candidate margin. The state is unchanged if this throws.
[[nodiscard]] ContactDynamicsResult advance_contact_dynamics(
    WorldState& state, const ConstraintSet& constraints, ComputeBackend& backend, float dt,
    const ContactDynamicsParameters& parameters, const ContactParameters& contact_parameters,
    const ConstraintContactParameters& constraint_parameters);

}  // namespace microsimulator::detail
