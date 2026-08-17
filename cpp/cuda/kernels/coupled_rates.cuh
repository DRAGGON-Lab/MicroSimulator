#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

#include "signals.cuh"
#include "species.cuh"

namespace cm::cuda {

cudaError_t launch_advance_coupled(
    float* species_levels, const float* previous_lengths, const float4* centers,
    const float4* geometry, const float* growth_rates, const std::int32_t* cell_types,
    const RateInstructionGpu* instructions, const std::uint32_t* species_outputs,
    const std::uint32_t* signal_outputs, float* workspace, const float* grid_levels,
    float* grid_output, const float* diffusion, const float4* advection, const float* fixed_values,
    const float* reaction_source, const float* reaction_loss, const std::uint8_t* obstacles,
    float* cell_signal_rates, std::uint32_t* error, SignalGridBoundariesGpu boundaries,
    SignalGridShapeGpu shape, float4 origin, float4 spacing, float dt,
    std::uint32_t species_count, std::uint32_t signal_count, std::uint32_t instruction_count,
    std::uint32_t cell_count, std::uint32_t level_count, bool crank_nicolson,
    cudaStream_t stream);

}  // namespace cm::cuda
