#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <span>
#include <vector>

#include "microsimulator/constraints.hpp"
#include "microsimulator/deposition.hpp"
#include "microsimulator/mechanics.hpp"
#include "microsimulator/signals.hpp"
#include "microsimulator/world_state.hpp"

namespace microsimulator::cuda {

[[nodiscard]] BiomassDeposit deposit_biomass(const SignalGridSpec& spec, const DepositCells& cells,
                                             const BiomassDepositParameters& parameters,
                                             cudaStream_t stream);

[[nodiscard]] ShallowFlowCoupling shallow_flow_coupling(const SignalGridSpec& spec,
                                                        std::span<const float> base,
                                                        const BiomassDeposit& deposit,
                                                        const BiomassDragParameters& parameters,
                                                        cudaStream_t stream);

[[nodiscard]] std::vector<float> brinkman_drag(const SignalGridSpec& spec,
                                               const BiomassDeposit& deposit,
                                               const BiomassDragParameters& parameters,
                                               cudaStream_t stream);

[[nodiscard]] ExternalContactGraph external_contacts(const WorldState& state,
                                                     const ConstraintSet& constraints,
                                                     const ConstraintContactParameters& parameters,
                                                     bool every_face, cudaStream_t stream);

[[nodiscard]] std::vector<std::array<float, 6>> flow_drift_rates(const WorldState& state,
                                                                 const SignalGrid& grid, float dt,
                                                                 cudaStream_t stream);

void advance_flow_drift(WorldState& state, const SignalGrid& grid, float dt,
                        const FlowDriftParameters& parameters, cudaStream_t stream);

}  // namespace microsimulator::cuda
