#pragma once

#include <array>
#include <vector>

#include "microsimulator/mechanics.hpp"
#include "microsimulator/signals.hpp"
#include "microsimulator/world_state.hpp"

namespace microsimulator {

[[nodiscard]] std::vector<std::array<float, 6>> flow_drift_rates_cpu(const WorldState& state,
                                                                     const SignalGrid& grid,
                                                                     float dt);

void advance_flow_drift_cpu(WorldState& state, const SignalGrid& grid, float dt,
                            const FlowDriftParameters& parameters);

}  // namespace microsimulator
