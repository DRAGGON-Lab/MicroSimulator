#pragma once

#include <span>
#include <vector>

#include "microsimulator/deposition.hpp"

namespace microsimulator {

[[nodiscard]] BiomassDeposit deposit_biomass_cpu(const SignalGridSpec& spec,
                                                 const DepositCells& cells,
                                                 const BiomassDepositParameters& parameters);

[[nodiscard]] ShallowFlowCoupling shallow_flow_coupling_cpu(
    const SignalGridSpec& spec, std::span<const float> base, const BiomassDeposit& deposit,
    const BiomassDragParameters& parameters);

[[nodiscard]] std::vector<float> brinkman_drag_cpu(const SignalGridSpec& spec,
                                                   const BiomassDeposit& deposit,
                                                   const BiomassDragParameters& parameters);

}  // namespace microsimulator
