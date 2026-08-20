#include <cmath>

#include "signals.cuh"

namespace cm::cuda {
namespace {

struct TransportPoint {
  float rate;
  float diagonal;
};

__device__ TransportPoint transport_point(
    const float* levels, const float* diffusion, const float4* advection, const float* fixed_values,
    const float* reaction_source, const float* reaction_loss, const std::uint8_t* obstacles,
    const float* x_faces, const float* y_faces, const float* z_faces,
    std::uint32_t has_velocity_field, SignalGridBoundariesGpu boundaries, SignalGridShapeGpu shape,
    float4 spacing, std::uint32_t signal_count, std::uint32_t index) {
  const auto signal = index / shape.sites;
  const auto site = index - signal * shape.sites;
  const auto x = site / (shape.y * shape.z);
  const auto yz = site - x * shape.y * shape.z;
  const auto y = yz / shape.z;
  const auto z = yz - y * shape.z;
  if (obstacles[site] != 0) {
    return {.rate = 0.0F, .diagonal = 0.0F};
  }
  const auto current = levels[index];

  float lower[3];
  float upper[3];
  lower[0] = x == 0 ? exterior_value(boundaries.x_lower, fixed_values, 0, signal, signal_count,
                                     current, grid_level(levels, shape, signal, shape.x - 1, y, z))
                    : grid_level(levels, shape, signal, x - 1, y, z);
  upper[0] = x + 1 == shape.x
                 ? exterior_value(boundaries.x_upper, fixed_values, 1, signal, signal_count,
                                  current, grid_level(levels, shape, signal, 0, y, z))
                 : grid_level(levels, shape, signal, x + 1, y, z);
  lower[1] = y == 0 ? exterior_value(boundaries.y_lower, fixed_values, 2, signal, signal_count,
                                     current, grid_level(levels, shape, signal, x, shape.y - 1, z))
                    : grid_level(levels, shape, signal, x, y - 1, z);
  upper[1] = y + 1 == shape.y
                 ? exterior_value(boundaries.y_upper, fixed_values, 3, signal, signal_count,
                                  current, grid_level(levels, shape, signal, x, 0, z))
                 : grid_level(levels, shape, signal, x, y + 1, z);
  lower[2] = z == 0 ? exterior_value(boundaries.z_lower, fixed_values, 4, signal, signal_count,
                                     current, grid_level(levels, shape, signal, x, y, shape.z - 1))
                    : grid_level(levels, shape, signal, x, y, z - 1);
  upper[2] = z + 1 == shape.z
                 ? exterior_value(boundaries.z_upper, fixed_values, 5, signal, signal_count,
                                  current, grid_level(levels, shape, signal, x, y, 0))
                 : grid_level(levels, shape, signal, x, y, z + 1);

  const std::uint32_t dimensions[3]{shape.x, shape.y, shape.z};
  const auto faces = grid_face_state(shape, boundaries, obstacles, x_faces, y_faces, z_faces,
                                     has_velocity_field, advection[signal], x, y, z);
  const bool* closed_lower = faces.closed_lower;
  const bool* closed_upper = faces.closed_upper;
  const float* face_lower = faces.lower;
  const float* face_upper = faces.upper;
  const float grid_spacing[3]{spacing.x, spacing.y, spacing.z};
  float rate = 0.0F;
  float diagonal = 0.0F;
  for (std::uint32_t axis = 0; axis < 3; ++axis) {
    if (dimensions[axis] == 1) {
      continue;
    }
    if (closed_lower[axis]) {
      lower[axis] = current;
    }
    if (closed_upper[axis]) {
      upper[axis] = current;
    }
    const auto inverse_spacing = 1.0F / grid_spacing[axis];
    const auto diffusion_scale = diffusion[signal] * inverse_spacing * inverse_spacing;
    rate += diffusion_scale * (lower[axis] - 2.0F * current + upper[axis]);
    diagonal -= 2.0F * diffusion_scale;
    if (closed_lower[axis]) {
      diagonal += diffusion_scale;
    }
    if (closed_upper[axis]) {
      diagonal += diffusion_scale;
    }
    auto lower_flux =
        face_lower[axis] >= 0.0F ? face_lower[axis] * lower[axis] : face_lower[axis] * current;
    auto upper_flux =
        face_upper[axis] >= 0.0F ? face_upper[axis] * current : face_upper[axis] * upper[axis];
    if (closed_lower[axis]) {
      lower_flux = 0.0F;
    }
    if (closed_upper[axis]) {
      upper_flux = 0.0F;
    }
    rate -= (upper_flux - lower_flux) * inverse_spacing;
    if (!closed_upper[axis] && face_upper[axis] > 0.0F) {
      diagonal -= face_upper[axis] * inverse_spacing;
    }
    if (!closed_lower[axis] && face_lower[axis] < 0.0F) {
      diagonal += face_lower[axis] * inverse_spacing;
    }
  }
  rate += reaction_source[index] - reaction_loss[index] * current;
  diagonal -= reaction_loss[index];
  return {.rate = rate, .diagonal = diagonal};
}

__global__ void advance_signal_grid(const float* levels, float* output, const float* diffusion,
                                    const float4* advection, const float* fixed_values,
                                    const float* reaction_source, const float* reaction_loss,
                                    const std::uint8_t* obstacles, const float* x_faces,
                                    const float* y_faces, const float* z_faces,
                                    std::uint32_t has_velocity_field, std::uint32_t* error,
                                    SignalGridBoundariesGpu boundaries, SignalGridShapeGpu shape,
                                    float4 spacing, float dt, std::uint32_t signal_count,
                                    std::uint32_t level_count, bool crank_nicolson) {
  const auto index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= level_count) {
    return;
  }

  const auto transport =
      transport_point(levels, diffusion, advection, fixed_values, reaction_source, reaction_loss,
                      obstacles, x_faces, y_faces, z_faces, has_velocity_field, boundaries, shape,
                      spacing, signal_count, index);
  const auto scale = crank_nicolson ? 0.5F * dt : dt;
  const auto candidate = levels[index] + scale * transport.rate;
  output[index] = candidate;
  if (!isfinite(candidate) || (!crank_nicolson && candidate < 0.0F)) {
    atomicOr(error, 1U);
  }
}

__global__ void signal_square_terms(const float* input, float* terms, std::uint32_t level_count) {
  const auto index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < level_count) {
    terms[index] = input[index] * input[index];
  }
}

__global__ void signal_crank_nicolson_jacobi(
    const float* current, float* output, const float* right_hand_side, const float* diffusion,
    const float4* advection, const float* fixed_values, const float* reaction_source,
    const float* reaction_loss, const std::uint8_t* obstacles, const float* x_faces,
    const float* y_faces, const float* z_faces, std::uint32_t has_velocity_field,
    std::uint32_t* error, SignalGridBoundariesGpu boundaries, SignalGridShapeGpu shape,
    float4 spacing, float half_dt, std::uint32_t signal_count, std::uint32_t level_count) {
  const auto index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= level_count) {
    return;
  }
  const auto transport =
      transport_point(current, diffusion, advection, fixed_values, reaction_source, reaction_loss,
                      obstacles, x_faces, y_faces, z_faces, has_velocity_field, boundaries, shape,
                      spacing, signal_count, index);
  const auto remainder = transport.rate - transport.diagonal * current[index];
  const auto candidate =
      (right_hand_side[index] + half_dt * remainder) / (1.0F - half_dt * transport.diagonal);
  output[index] = candidate;
  if (!isfinite(candidate)) {
    atomicOr(error, 1U);
  }
}

__global__ void signal_crank_nicolson_residual_terms(
    const float* current, const float* right_hand_side, float* terms, const float* diffusion,
    const float4* advection, const float* fixed_values, const float* reaction_source,
    const float* reaction_loss, const std::uint8_t* obstacles, const float* x_faces,
    const float* y_faces, const float* z_faces, std::uint32_t has_velocity_field,
    SignalGridBoundariesGpu boundaries, SignalGridShapeGpu shape, float4 spacing, float half_dt,
    std::uint32_t signal_count, std::uint32_t level_count) {
  const auto index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= level_count) {
    return;
  }
  const auto transport =
      transport_point(current, diffusion, advection, fixed_values, reaction_source, reaction_loss,
                      obstacles, x_faces, y_faces, z_faces, has_velocity_field, boundaries, shape,
                      spacing, signal_count, index);
  const auto residual = right_hand_side[index] - current[index] + half_dt * transport.rate;
  terms[index] = residual * residual;
}

}  // namespace

void launch_advance_signal_grid(
    const float* levels, float* output, const float* diffusion, const float4* advection,
    const float* fixed_values, const float* reaction_source, const float* reaction_loss,
    const std::uint8_t* obstacles, const float* x_faces, const float* y_faces, const float* z_faces,
    std::uint32_t has_velocity_field, std::uint32_t* error, SignalGridBoundariesGpu boundaries,
    SignalGridShapeGpu shape, float4 spacing, float dt, std::uint32_t signal_count,
    std::uint32_t level_count, bool crank_nicolson, cudaStream_t stream) {
  constexpr std::uint32_t threads_per_block = 256;
  const auto block_count = ((level_count - 1) / threads_per_block) + 1;
  advance_signal_grid<<<block_count, threads_per_block, 0, stream>>>(
      levels, output, diffusion, advection, fixed_values, reaction_source, reaction_loss, obstacles,
      x_faces, y_faces, z_faces, has_velocity_field, error, boundaries, shape, spacing, dt,
      signal_count, level_count, crank_nicolson);
}

void launch_signal_square_terms(const float* input, float* terms, std::uint32_t level_count,
                                cudaStream_t stream) {
  constexpr std::uint32_t threads_per_block = 256;
  const auto block_count = ((level_count - 1) / threads_per_block) + 1;
  signal_square_terms<<<block_count, threads_per_block, 0, stream>>>(input, terms, level_count);
}

void launch_signal_crank_nicolson_jacobi(
    const float* current, float* output, const float* right_hand_side, const float* diffusion,
    const float4* advection, const float* fixed_values, const float* reaction_source,
    const float* reaction_loss, const std::uint8_t* obstacles, const float* x_faces,
    const float* y_faces, const float* z_faces, std::uint32_t has_velocity_field,
    std::uint32_t* error, SignalGridBoundariesGpu boundaries, SignalGridShapeGpu shape,
    float4 spacing, float half_dt, std::uint32_t signal_count, std::uint32_t level_count,
    cudaStream_t stream) {
  constexpr std::uint32_t threads_per_block = 256;
  const auto block_count = ((level_count - 1) / threads_per_block) + 1;
  signal_crank_nicolson_jacobi<<<block_count, threads_per_block, 0, stream>>>(
      current, output, right_hand_side, diffusion, advection, fixed_values, reaction_source,
      reaction_loss, obstacles, x_faces, y_faces, z_faces, has_velocity_field, error, boundaries,
      shape, spacing, half_dt, signal_count, level_count);
}

void launch_signal_crank_nicolson_residual_terms(
    const float* current, const float* right_hand_side, float* terms, const float* diffusion,
    const float4* advection, const float* fixed_values, const float* reaction_source,
    const float* reaction_loss, const std::uint8_t* obstacles, const float* x_faces,
    const float* y_faces, const float* z_faces, std::uint32_t has_velocity_field,
    SignalGridBoundariesGpu boundaries, SignalGridShapeGpu shape, float4 spacing, float half_dt,
    std::uint32_t signal_count, std::uint32_t level_count, cudaStream_t stream) {
  constexpr std::uint32_t threads_per_block = 256;
  const auto block_count = ((level_count - 1) / threads_per_block) + 1;
  signal_crank_nicolson_residual_terms<<<block_count, threads_per_block, 0, stream>>>(
      current, right_hand_side, terms, diffusion, advection, fixed_values, reaction_source,
      reaction_loss, obstacles, x_faces, y_faces, z_faces, has_velocity_field, boundaries, shape,
      spacing, half_dt, signal_count, level_count);
}

}  // namespace cm::cuda
