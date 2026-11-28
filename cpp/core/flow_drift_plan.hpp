#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "core/device_types.hpp"
#include "microsimulator/mechanics.hpp"
#include "microsimulator/signals.hpp"
#include "microsimulator/world_state.hpp"

namespace microsimulator::detail {

// What every backend's flow drift kernels read, in the layout of
// shared/device_types.h. Each array holds at least one
// element. The velocity field stays with the grid; `revision` names it, so a
// device backend uploads a field once for every step that uses it.
struct FlowDriftPlan {
  device::FlowDriftData data{};
  std::vector<device::float4> centers;
  std::vector<device::float4> axes;
  std::vector<std::uint8_t> fixed;
  std::vector<std::uint8_t> solid;
  const SignalGridVelocityField* field{nullptr};
  std::uint64_t revision{0};
};

// Lays out a drift of the state's cells over `dt` in the grid's velocity
// field. `max_rotation_radians` is the largest turn of one substep of
// `advance_flow_drift`; flow_drift_rates turns cells without a limit.
[[nodiscard]] FlowDriftPlan plan_flow_drift(const WorldState& state, const SignalGrid& grid,
                                            float dt, float max_rotation_radians);

// Throws the error a cell's drift status names, if any does.
void check_flow_drift_status(std::span<const std::uint32_t> status);

// The rates the kernels wrote, six floats per cell, as velocity then angular
// velocity by slot.
[[nodiscard]] std::vector<std::array<float, 6>> unpack_flow_drift_rates(
    std::span<const float> rates, std::size_t cell_count);

// Moves every free cell of the state to the center and axis the kernels
// wrote for it.
void apply_flow_drift_moves(WorldState& state, std::span<const device::float4> centers,
                            std::span<const device::float4> axes);

}  // namespace microsimulator::detail
