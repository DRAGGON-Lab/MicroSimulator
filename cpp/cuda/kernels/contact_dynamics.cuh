#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "cuda/kernels/contact_solve.cuh"

namespace microsimulator::cuda {

// The contact solve of kernels/contact_solve.cuh on the device. An
// incidence entry is a row index shifted left by one, with the low bit set
// when the cell is the row's second cell.

// The response of every cell to the multipliers, or with `project` to those
// one projected step from them; a projected response does nothing once the
// solve is done.
void launch_contact_respond(const device::ContactRowData* rows,
                            const device::ContactBodyData* bodies, const float* multipliers,
                            const float* gaps, const std::uint32_t* incidence_offsets,
                            const std::uint32_t* incidence_entries,
                            device::ContactResponse* response,
                            const device::ContactSolveState* state, device::ContactSolveData solve,
                            std::uint32_t project, cudaStream_t stream);

// One evaluation of the rows: with `initial` the gaps of the current
// multipliers, otherwise the projected step from the current pair into the
// next. The terms are summed by contact_reduction_tree, level by level
// through `partials`, and recorded in `state`.
void launch_contact_terms(const device::ContactRowData* rows,
                          const device::ContactResponse* response, float* current_multipliers,
                          float* current_gaps, float* next_multipliers, float* next_gaps,
                          float4* const partials[2], device::ContactSolveState* state,
                          device::ContactSolveData solve, std::uint32_t initial,
                          float first_step_length, std::uint32_t max_iterations,
                          cudaStream_t stream);

// The whole solve in one block of contact_solve_threads threads, for up to
// contact_solve_threads / 4 blocks of rows. The multipliers start in
// `multipliers_a`; the state ends with the iterations taken and the residual
// of the multipliers in array `iterations % 2`, whose response `response`
// holds.
inline constexpr std::uint32_t contact_solve_threads = 1024;
void launch_contact_solve(const device::ContactRowData* rows, const device::ContactBodyData* bodies,
                          float* multipliers_a, float* multipliers_b, float* gaps_a, float* gaps_b,
                          const std::uint32_t* incidence_offsets,
                          const std::uint32_t* incidence_entries, device::ContactResponse* response,
                          device::ContactSolveState* state, device::ContactSolveData solve,
                          std::uint32_t max_iterations, float first_step_length,
                          cudaStream_t stream);

}  // namespace microsimulator::cuda
