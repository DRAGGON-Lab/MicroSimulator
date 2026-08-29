#pragma once

#include <cuda_runtime_api.h>

#include <span>

#include "cm/flow.hpp"

namespace cm::cuda {

[[nodiscard]] DepthAveragedFlowResult solve_depth_averaged_flow(
    const SignalGridSpec& spec, std::span<const float> mobility,
    const DepthAveragedFlowParameters& parameters, cudaStream_t stream);

[[nodiscard]] ResolvedFlowResult solve_resolved_flow(const SignalGridSpec& spec,
                                                     std::span<const float> drag,
                                                     const ResolvedFlowParameters& parameters,
                                                     cudaStream_t stream);

}  // namespace cm::cuda
