#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace cm::cuda {

struct SignalGridShapeGpu {
  std::uint32_t x;
  std::uint32_t y;
  std::uint32_t z;
  std::uint32_t sites;
};

struct SignalGridBoundariesGpu {
  std::uint32_t x_lower;
  std::uint32_t x_upper;
  std::uint32_t y_lower;
  std::uint32_t y_upper;
  std::uint32_t z_lower;
  std::uint32_t z_upper;
};

void launch_advance_signal_grid(const float* levels, float* output, const float* diffusion,
                                const float4* advection, const float* fixed_values,
                                const float* reaction_source, const float* reaction_loss,
                                const std::uint8_t* obstacles, const float* x_faces,
                                const float* y_faces, const float* z_faces,
                                std::uint32_t has_velocity_field, std::uint32_t* error,
                                SignalGridBoundariesGpu boundaries, SignalGridShapeGpu shape,
                                float4 spacing, float dt, std::uint32_t signal_count,
                                std::uint32_t level_count, bool crank_nicolson,
                                cudaStream_t stream);

void launch_signal_square_terms(const float* input, float* terms, std::uint32_t level_count,
                                cudaStream_t stream);
void launch_signal_crank_nicolson_jacobi(const float* current, float* output,
                                         const float* right_hand_side, const float* diffusion,
                                         const float4* advection, const float* fixed_values,
                                         const float* reaction_source, const float* reaction_loss,
                                         const std::uint8_t* obstacles, const float* x_faces,
                                         const float* y_faces, const float* z_faces,
                                         std::uint32_t has_velocity_field, std::uint32_t* error,
                                         SignalGridBoundariesGpu boundaries,
                                         SignalGridShapeGpu shape, float4 spacing, float half_dt,
                                         std::uint32_t signal_count, std::uint32_t level_count,
                                         cudaStream_t stream);
void launch_signal_crank_nicolson_residual_terms(
    const float* current, const float* right_hand_side, float* terms, const float* diffusion,
    const float4* advection, const float* fixed_values, const float* reaction_source,
    const float* reaction_loss, const std::uint8_t* obstacles, const float* x_faces,
    const float* y_faces, const float* z_faces, std::uint32_t has_velocity_field,
    SignalGridBoundariesGpu boundaries, SignalGridShapeGpu shape, float4 spacing, float half_dt,
    std::uint32_t signal_count, std::uint32_t level_count, cudaStream_t stream);

}  // namespace cm::cuda
