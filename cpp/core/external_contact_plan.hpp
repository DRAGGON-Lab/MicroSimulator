#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "core/device_types.hpp"
#include "microsimulator/constraints.hpp"
#include "microsimulator/world_state.hpp"

namespace microsimulator::detail {

// What every backend's external contact kernels read, in the layout of
// shared/device_types.h.
struct ExternalContactPlan {
  device::ExternalContactData data{};
  // Each cell's center with its length in w, and its axis with its radius
  // in w, by slot.
  std::vector<device::float4> centers;
  std::vector<device::float4> axes;
  // The constraints in ascending id order.
  std::vector<device::ExternalConstraint> constraints;
  std::vector<CellId> cell_ids;

  [[nodiscard]] std::size_t pair_count() const noexcept {
    return static_cast<std::size_t>(data.cell_count) * data.constraint_count;
  }
};

// Validates a detection and lays it out for the kernels. With `every_face`,
// a cell inside a box or cylinder makes a row for every face it is near.
[[nodiscard]] ExternalContactPlan plan_external_contacts(
    const WorldState& state, const ConstraintSet& constraints,
    const ConstraintContactParameters& parameters, bool every_face);

// The contact graph of the records the fill pass wrote, which follow pair
// order. Contacts are ordered by cell id, constraint id, and location, each
// pair's contacts keeping the order the pair emitted them in.
[[nodiscard]] ExternalContactGraph finish_external_contacts(
    const ExternalContactPlan& plan, std::span<const device::ExternalContactRecord> records);

}  // namespace microsimulator::detail
