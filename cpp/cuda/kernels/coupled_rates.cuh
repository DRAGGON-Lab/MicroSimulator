#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

#include "signals.cuh"
#include "species.cuh"

namespace cm::cuda {

// A trilinear stencil covers at most eight lattice sites.
inline constexpr std::uint32_t scatter_stencil_sites = 8;

// Builds the per-site cell lists the coupled grid kernel gathers from. On
// return `*ends` points at the buffer (`counts`, `scan_a`, or `scan_b`) holding
// each site's inclusive end offset into `cells`; site s owns
// [ends[s - 1], ends[s]). `counts`, `cursors`, `scan_a`, and `scan_b` hold one
// value per site, `stencils` one per cell, and `cells`
// `scatter_stencil_sites` per cell. `cell_count` must be positive.
cudaError_t launch_coupled_scatter_lists(const float4* centers, uint2* stencils,
                                         std::uint32_t* counts, std::uint32_t* cursors,
                                         std::uint32_t* scan_a, std::uint32_t* scan_b,
                                         std::uint32_t* cells, SignalGridShapeGpu shape,
                                         float4 origin, float4 spacing, std::uint32_t cell_count,
                                         cudaStream_t stream, const std::uint32_t** ends);

// `scatter_ends` and `scatter_cells` are the lists built by
// `launch_coupled_scatter_lists`; they are not read when `cell_count` is zero.
cudaError_t launch_advance_coupled(
    float* species_levels, const float* previous_lengths, const float4* centers,
    const float4* geometry, const float* growth_rates, const std::int32_t* cell_types,
    const RateInstructionGpu* instructions, const std::uint32_t* species_outputs,
    const std::uint32_t* signal_outputs, float* workspace, const float* grid_levels,
    float* grid_output, const float* diffusion, const float4* advection, const float* fixed_values,
    const float* reaction_source, const float* reaction_loss, const std::uint8_t* obstacles,
    const float* x_faces, const float* y_faces, const float* z_faces,
    std::uint32_t has_velocity_field, float* cell_signal_rates, std::uint32_t* error,
    SignalGridBoundariesGpu boundaries, SignalGridShapeGpu shape, float4 origin, float4 spacing,
    float dt, std::uint32_t species_count, std::uint32_t signal_count,
    std::uint32_t instruction_count, std::uint32_t cell_count, std::uint32_t level_count,
    std::uint32_t crank_nicolson, const std::uint32_t* scatter_ends,
    const std::uint32_t* scatter_cells, cudaStream_t stream);

}  // namespace cm::cuda
