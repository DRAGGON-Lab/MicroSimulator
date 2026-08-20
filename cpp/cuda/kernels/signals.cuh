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

// Grid geometry and transport helpers shared by the signal and coupled-rate
// kernels. Whole-program compilation gives each translation unit its own copy,
// so a device-inline definition in this header needs no device linking.

__device__ inline std::uint32_t site_index(SignalGridShapeGpu shape, std::uint32_t x,
                                           std::uint32_t y, std::uint32_t z) {
  return x * shape.y * shape.z + y * shape.z + z;
}

__device__ inline float grid_level(const float* levels, SignalGridShapeGpu shape,
                                   std::uint32_t signal, std::uint32_t x, std::uint32_t y,
                                   std::uint32_t z) {
  return levels[signal * shape.sites + site_index(shape, x, y, z)];
}

__device__ inline float exterior_value(std::uint32_t kind, const float* fixed_values,
                                       std::uint32_t face, std::uint32_t signal,
                                       std::uint32_t signal_count, float current, float periodic) {
  if (kind == 0) {
    return current;
  }
  if (kind == 1) {
    return periodic;
  }
  return fixed_values[face * signal_count + signal];
}

// Whether each of a site's six faces is closed to transport, and the velocity
// it carries. A face is closed by a no-flux boundary, by a periodic boundary
// that wraps onto a solid site, or by a solid neighbour.
struct GridFaceState {
  bool closed_lower[3];
  bool closed_upper[3];
  float lower[3];
  float upper[3];
};

__device__ inline GridFaceState grid_face_state(SignalGridShapeGpu shape,
                                                SignalGridBoundariesGpu boundaries,
                                                const std::uint8_t* obstacles, const float* x_faces,
                                                const float* y_faces, const float* z_faces,
                                                std::uint32_t has_velocity_field, float4 advection,
                                                std::uint32_t x, std::uint32_t y, std::uint32_t z) {
  GridFaceState faces{};
  faces.closed_lower[0] =
      x == 0 ? (boundaries.x_lower == 0 ||
                (boundaries.x_lower == 1 && obstacles[site_index(shape, shape.x - 1, y, z)] != 0))
             : obstacles[site_index(shape, x - 1, y, z)] != 0;
  faces.closed_upper[0] =
      x + 1 == shape.x ? (boundaries.x_upper == 0 ||
                          (boundaries.x_upper == 1 && obstacles[site_index(shape, 0, y, z)] != 0))
                       : obstacles[site_index(shape, x + 1, y, z)] != 0;
  faces.closed_lower[1] =
      y == 0 ? (boundaries.y_lower == 0 ||
                (boundaries.y_lower == 1 && obstacles[site_index(shape, x, shape.y - 1, z)] != 0))
             : obstacles[site_index(shape, x, y - 1, z)] != 0;
  faces.closed_upper[1] =
      y + 1 == shape.y ? (boundaries.y_upper == 0 ||
                          (boundaries.y_upper == 1 && obstacles[site_index(shape, x, 0, z)] != 0))
                       : obstacles[site_index(shape, x, y + 1, z)] != 0;
  faces.closed_lower[2] =
      z == 0 ? (boundaries.z_lower == 0 ||
                (boundaries.z_lower == 1 && obstacles[site_index(shape, x, y, shape.z - 1)] != 0))
             : obstacles[site_index(shape, x, y, z - 1)] != 0;
  faces.closed_upper[2] =
      z + 1 == shape.z ? (boundaries.z_upper == 0 ||
                          (boundaries.z_upper == 1 && obstacles[site_index(shape, x, y, 0)] != 0))
                       : obstacles[site_index(shape, x, y, z + 1)] != 0;
  if (has_velocity_field != 0) {
    faces.lower[0] = x_faces[x * shape.y * shape.z + y * shape.z + z];
    faces.upper[0] = x_faces[(x + 1) * shape.y * shape.z + y * shape.z + z];
    faces.lower[1] = y_faces[x * (shape.y + 1) * shape.z + y * shape.z + z];
    faces.upper[1] = y_faces[x * (shape.y + 1) * shape.z + (y + 1) * shape.z + z];
    faces.lower[2] = z_faces[x * shape.y * (shape.z + 1) + y * (shape.z + 1) + z];
    faces.upper[2] = z_faces[x * shape.y * (shape.z + 1) + y * (shape.z + 1) + z + 1];
  } else {
    const float velocity[3]{advection.x, advection.y, advection.z};
    for (std::uint32_t axis = 0; axis < 3; ++axis) {
      faces.lower[axis] = velocity[axis];
      faces.upper[axis] = velocity[axis];
    }
  }
  return faces;
}

void launch_advance_signal_grid(
    const float* levels, float* output, const float* diffusion, const float4* advection,
    const float* fixed_values, const float* reaction_source, const float* reaction_loss,
    const std::uint8_t* obstacles, const float* x_faces, const float* y_faces, const float* z_faces,
    std::uint32_t has_velocity_field, std::uint32_t* error, SignalGridBoundariesGpu boundaries,
    SignalGridShapeGpu shape, float4 spacing, float dt, std::uint32_t signal_count,
    std::uint32_t level_count, bool crank_nicolson, cudaStream_t stream);

void launch_signal_square_terms(const float* input, float* terms, std::uint32_t level_count,
                                cudaStream_t stream);
void launch_signal_crank_nicolson_jacobi(
    const float* current, float* output, const float* right_hand_side, const float* diffusion,
    const float4* advection, const float* fixed_values, const float* reaction_source,
    const float* reaction_loss, const std::uint8_t* obstacles, const float* x_faces,
    const float* y_faces, const float* z_faces, std::uint32_t has_velocity_field,
    std::uint32_t* error, SignalGridBoundariesGpu boundaries, SignalGridShapeGpu shape,
    float4 spacing, float half_dt, std::uint32_t signal_count, std::uint32_t level_count,
    cudaStream_t stream);
void launch_signal_crank_nicolson_residual_terms(
    const float* current, const float* right_hand_side, float* terms, const float* diffusion,
    const float4* advection, const float* fixed_values, const float* reaction_source,
    const float* reaction_loss, const std::uint8_t* obstacles, const float* x_faces,
    const float* y_faces, const float* z_faces, std::uint32_t has_velocity_field,
    SignalGridBoundariesGpu boundaries, SignalGridShapeGpu shape, float4 spacing, float half_dt,
    std::uint32_t signal_count, std::uint32_t level_count, cudaStream_t stream);

}  // namespace cm::cuda
