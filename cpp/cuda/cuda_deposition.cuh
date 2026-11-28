#pragma once

#include <cuda_runtime_api.h>

#include <span>
#include <vector>

#include "microsimulator/deposition.hpp"

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

}  // namespace microsimulator::cuda
