#pragma once

#include <cuda_runtime_api.h>

#include <span>

#include "microsimulator/flow.hpp"

namespace microsimulator::cuda {

[[nodiscard]] DepthAveragedFlowResult solve_depth_averaged_flow(
    const SignalGridSpec& spec, std::span<const float> mobility,
    const DepthAveragedFlowParameters& parameters, cudaStream_t stream);

[[nodiscard]] DepthAveragedPortFlowResult solve_depth_averaged_port_flow(
    const SignalGridSpec& spec, std::span<const FlowPort> ports, std::span<const float> mobility,
    const DepthAveragedPortFlowParameters& parameters, cudaStream_t stream);

[[nodiscard]] ResolvedFlowResult solve_resolved_flow(const SignalGridSpec& spec,
                                                     std::span<const float> drag,
                                                     const ResolvedFlowParameters& parameters,
                                                     cudaStream_t stream);

[[nodiscard]] ResolvedPortFlowResult solve_resolved_port_flow(
    const SignalGridSpec& spec, std::span<const FlowPort> ports, std::span<const float> drag,
    const ResolvedPortFlowParameters& parameters, cudaStream_t stream);

}  // namespace microsimulator::cuda
