#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace microsimulator::cuda {

struct alignas(16) MechanicsDofsGpu {
  float4 linear_length;
  float4 rotation;
};

static_assert(sizeof(MechanicsDofsGpu) == 32);

void launch_build_mechanics_rows(const float4* centers, const float4* axes, const float4* geometry,
                                 const std::uint32_t* first_slots,
                                 const std::uint32_t* second_slots, const float4* points,
                                 const float4* normals, const float* separations,
                                 const float* weights, MechanicsDofsGpu* first_rows,
                                 MechanicsDofsGpu* second_rows, float* right_hand_side,
                                 std::uint32_t contact_count, cudaStream_t stream);

void launch_apply_mechanics_b(const MechanicsDofsGpu* first_rows,
                              const MechanicsDofsGpu* second_rows, const std::uint32_t* first_slots,
                              const std::uint32_t* second_slots, const MechanicsDofsGpu* input,
                              const std::uint8_t* fixed, float* row_values,
                              std::uint32_t contact_count, cudaStream_t stream);

void launch_apply_mechanics_transpose(const MechanicsDofsGpu* first_rows,
                                      const MechanicsDofsGpu* second_rows, const float* row_values,
                                      const std::uint32_t* incidence_offsets,
                                      const std::uint32_t* incidence_indices,
                                      const std::uint32_t* first_slots, MechanicsDofsGpu* output,
                                      std::uint32_t cell_count, cudaStream_t stream);

void launch_add_mechanics_regularizer(const float4* axes, const float4* geometry,
                                      const MechanicsDofsGpu* input, MechanicsDofsGpu* output,
                                      const std::uint8_t* fixed, float mu_a, float gamma,
                                      std::uint32_t cell_count, cudaStream_t stream);

void launch_initialize_mechanics_vectors(MechanicsDofsGpu* right_hand_side,
                                         MechanicsDofsGpu* solution, MechanicsDofsGpu* residual,
                                         MechanicsDofsGpu* search_direction,
                                         const std::uint8_t* fixed, std::uint32_t cell_count,
                                         cudaStream_t stream);

void launch_update_mechanics_solution_residual(MechanicsDofsGpu* solution,
                                               MechanicsDofsGpu* residual,
                                               const MechanicsDofsGpu* search_direction,
                                               const MechanicsDofsGpu* applied, float alpha,
                                               std::uint32_t cell_count, cudaStream_t stream);

void launch_update_mechanics_search_direction(const MechanicsDofsGpu* residual,
                                              MechanicsDofsGpu* search_direction, float beta,
                                              std::uint32_t cell_count, cudaStream_t stream);

void launch_subtract_mechanics_vectors(const MechanicsDofsGpu* left, const MechanicsDofsGpu* right,
                                       MechanicsDofsGpu* output, std::uint32_t cell_count,
                                       cudaStream_t stream);

void launch_mechanics_dot_terms(const MechanicsDofsGpu* left, const MechanicsDofsGpu* right,
                                float* terms, std::uint32_t cell_count, cudaStream_t stream);

void launch_reduce_sum_pairs(const float* input, float* output, std::uint32_t element_count,
                             cudaStream_t stream);

}  // namespace microsimulator::cuda
