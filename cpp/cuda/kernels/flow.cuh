#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace cm::cuda {

struct alignas(16) FlowGridParameters {
  std::uint32_t dimensions[4];
  float spacing[4];
  float inverse_spacing_squared[4];
  std::uint32_t face_offsets[4];
  std::uint32_t face_counts[4];
  std::uint32_t flow_axis;
  std::uint32_t site_count;
  std::uint32_t total_face_count;
  std::uint32_t padding;
};

static_assert(sizeof(FlowGridParameters) == 96);

void launch_depth_flow_operator_double(const double* input, const float* mobility, double* output,
                                       const FlowGridParameters& grid, cudaStream_t stream);
void launch_depth_flow_velocity_double(const double* pressure, const float* mobility,
                                       float* velocity, const FlowGridParameters& grid,
                                       cudaStream_t stream);
void launch_depth_flow_pcg_initialize(const double* right_hand_side, const float* diagonal,
                                      double* solution, double* residual, double* preconditioned,
                                      double* direction, std::uint32_t count, cudaStream_t stream);
void launch_depth_flow_pcg_update(double* solution, double* residual, const double* direction,
                                  const double* transformed, double alpha, std::uint32_t count,
                                  cudaStream_t stream);
void launch_depth_flow_pcg_precondition(const double* residual, const float* diagonal,
                                        double* preconditioned, std::uint32_t count,
                                        cudaStream_t stream);
void launch_depth_flow_pcg_direction(const double* preconditioned, double* direction, double beta,
                                     std::uint32_t count, cudaStream_t stream);
void launch_depth_flow_dot_partial(const double* left, const double* right, double* partials,
                                   std::uint32_t count, cudaStream_t stream);
void launch_resolved_flow_momentum(const float* input, const std::uint8_t* active,
                                   const std::uint8_t* exists, const float* face_drag,
                                   float* output, const FlowGridParameters& grid,
                                   cudaStream_t stream);
void launch_resolved_flow_gradient(const float* pressure, const std::uint8_t* fluid,
                                   const std::uint8_t* active, float* gradient,
                                   const FlowGridParameters& grid, cudaStream_t stream);
void launch_resolved_flow_divergence(const float* velocity, const std::uint8_t* fluid,
                                     float* divergence, const FlowGridParameters& grid,
                                     cudaStream_t stream);
void launch_flow_pcg_initialize(const float* right_hand_side, const float* diagonal,
                                float* solution, float* residual, float* preconditioned,
                                float* direction, std::uint32_t count, cudaStream_t stream);
void launch_flow_pcg_update(float* solution, float* residual, const float* direction,
                            const float* transformed, float alpha, std::uint32_t count,
                            cudaStream_t stream);
void launch_flow_pcg_precondition(const float* residual, const float* diagonal,
                                  float* preconditioned, std::uint32_t count, cudaStream_t stream);
void launch_flow_pcg_direction(const float* preconditioned, float* direction, float beta,
                               std::uint32_t count, cudaStream_t stream);
void launch_flow_vector_negate(const float* input, float* output, std::uint32_t count,
                               cudaStream_t stream);
void launch_flow_vector_combine(const float* source, float* target, float alpha, float beta,
                                std::uint32_t count, cudaStream_t stream);
void launch_flow_vector_subtract(const float* left, const float* right, float* output,
                                 std::uint32_t count, cudaStream_t stream);
void launch_flow_dot_partial(const float* left, const float* right, float* partials,
                             std::uint32_t count, cudaStream_t stream);

inline constexpr std::uint32_t flow_reduction_width = 64;

}  // namespace cm::cuda
