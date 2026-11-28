#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "core/device_types.hpp"
#include "microsimulator/contact_graph.hpp"
#include "microsimulator/world_state.hpp"

namespace microsimulator::detail {

// What every backend's cell contact kernels read, in the layout of
// shared/device_types.h. The capacities start at
// zero; a backend sets them to the storage it holds.
struct CellContactPlan {
  device::CellContactData data{};
  // Each cell's center with its length in w, and its axis with its radius
  // in w, by slot.
  std::vector<device::float4> centers;
  std::vector<device::float4> axes;
  std::vector<CellId> ids;
};

// Validates a detection and lays it out for the kernels.
[[nodiscard]] CellContactPlan plan_cell_contacts(const WorldState& state,
                                                 const ContactParameters& parameters);

// The candidate pairs the broad phase wrote, ordered by the ids of the first
// and second cell.
[[nodiscard]] std::vector<ContactCandidate> finish_cell_candidates(
    const CellContactPlan& plan, std::span<const device::uint2> candidates);

// The contact graph of the records the narrow phase wrote, ordered by the ids
// of the first and second cell and the ordinal.
[[nodiscard]] ContactGraph finish_cell_contacts(const CellContactPlan& plan,
                                                std::span<const device::CellContactRecord> records);

}  // namespace microsimulator::detail
