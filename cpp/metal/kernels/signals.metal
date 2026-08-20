#include <metal_stdlib>

using namespace metal;

;

struct TransportPoint {
  float rate;
  float diagonal;
};

TransportPoint transport_point(device const float* levels, device const float* diffusion,
                               device const float4* advection, device const float* fixed_values,
                               device const float* reaction_source,
                               device const float* reaction_loss, device const uchar* obstacles,
                               device const float* x_faces, device const float* y_faces,
                               device const float* z_faces, uint has_velocity_field,
                               constant uint* boundary_kinds, GridShape shape, float4 spacing,
                               uint signal_count, uint index) {
  uint signal = index / shape.sites;
  uint site = index - signal * shape.sites;
  uint x = site / (shape.y * shape.z);
  uint yz = site - x * shape.y * shape.z;
  uint y = yz / shape.z;
  uint z = yz - y * shape.z;
  if (obstacles[site] != 0u) {
    return {0.0f, 0.0f};
  }
  float current = levels[index];

  float3 lower;
  float3 upper;
  lower.x = x == 0u ? exterior_value(boundary_kinds[0], fixed_values, 0u, signal, signal_count,
                                     current, grid_level(levels, shape, signal, shape.x - 1u, y, z))
                    : grid_level(levels, shape, signal, x - 1u, y, z);
  upper.x = x + 1u == shape.x
                ? exterior_value(boundary_kinds[1], fixed_values, 1u, signal, signal_count, current,
                                 grid_level(levels, shape, signal, 0u, y, z))
                : grid_level(levels, shape, signal, x + 1u, y, z);
  lower.y = y == 0u ? exterior_value(boundary_kinds[2], fixed_values, 2u, signal, signal_count,
                                     current, grid_level(levels, shape, signal, x, shape.y - 1u, z))
                    : grid_level(levels, shape, signal, x, y - 1u, z);
  upper.y = y + 1u == shape.y
                ? exterior_value(boundary_kinds[3], fixed_values, 3u, signal, signal_count, current,
                                 grid_level(levels, shape, signal, x, 0u, z))
                : grid_level(levels, shape, signal, x, y + 1u, z);
  lower.z = z == 0u ? exterior_value(boundary_kinds[4], fixed_values, 4u, signal, signal_count,
                                     current, grid_level(levels, shape, signal, x, y, shape.z - 1u))
                    : grid_level(levels, shape, signal, x, y, z - 1u);
  upper.z = z + 1u == shape.z
                ? exterior_value(boundary_kinds[5], fixed_values, 5u, signal, signal_count, current,
                                 grid_level(levels, shape, signal, x, y, 0u))
                : grid_level(levels, shape, signal, x, y, z + 1u);

  uint3 dimensions = uint3(shape.x, shape.y, shape.z);
  GridFaceState faces = grid_face_state(shape, boundary_kinds, obstacles, x_faces, y_faces, z_faces,
                                        has_velocity_field, advection[signal], x, y, z);
  bool3 closed_lower = faces.closed_lower;
  bool3 closed_upper = faces.closed_upper;
  float3 grid_spacing = spacing.xyz;
  float rate = 0.0f;
  float diagonal = 0.0f;
  for (uint axis = 0; axis < 3u; ++axis) {
    if (dimensions[axis] == 1u) {
      continue;
    }
    if (closed_lower[axis]) {
      lower[axis] = current;
    }
    if (closed_upper[axis]) {
      upper[axis] = current;
    }
    float inverse_spacing = 1.0f / grid_spacing[axis];
    float diffusion_scale = diffusion[signal] * inverse_spacing * inverse_spacing;
    rate += diffusion_scale * (lower[axis] - 2.0f * current + upper[axis]);
    diagonal -= 2.0f * diffusion_scale;
    if (closed_lower[axis]) {
      diagonal += diffusion_scale;
    }
    if (closed_upper[axis]) {
      diagonal += diffusion_scale;
    }
    float lower_flux =
        faces.lower[axis] >= 0.0f ? faces.lower[axis] * lower[axis] : faces.lower[axis] * current;
    float upper_flux =
        faces.upper[axis] >= 0.0f ? faces.upper[axis] * current : faces.upper[axis] * upper[axis];
    if (closed_lower[axis]) {
      lower_flux = 0.0f;
    }
    if (closed_upper[axis]) {
      upper_flux = 0.0f;
    }
    rate -= (upper_flux - lower_flux) * inverse_spacing;
    if (!closed_upper[axis] && faces.upper[axis] > 0.0f) {
      diagonal -= faces.upper[axis] * inverse_spacing;
    }
    if (!closed_lower[axis] && faces.lower[axis] < 0.0f) {
      diagonal += faces.lower[axis] * inverse_spacing;
    }
  }
  rate += reaction_source[index] - reaction_loss[index] * current;
  diagonal -= reaction_loss[index];
  return {rate, diagonal};
}

kernel void advance_signal_grid(
    device const float* levels [[buffer(0)]], device float* output [[buffer(1)]],
    device const float* diffusion [[buffer(2)]], device const float4* advection [[buffer(3)]],
    device const float* fixed_values [[buffer(4)]], device atomic_uint* error [[buffer(5)]],
    constant uint* boundary_kinds [[buffer(6)]], constant GridShape& shape [[buffer(7)]],
    constant float4& spacing [[buffer(8)]], constant float& dt [[buffer(9)]],
    constant uint& signal_count [[buffer(10)]], constant uint& level_count [[buffer(11)]],
    constant uint& crank_nicolson [[buffer(12)]],
    device const float* reaction_source [[buffer(13)]],
    device const float* reaction_loss [[buffer(14)]], device const uchar* obstacles [[buffer(15)]],
    device const float* x_faces [[buffer(16)]], device const float* y_faces [[buffer(17)]],
    device const float* z_faces [[buffer(18)]], constant uint& has_velocity_field [[buffer(19)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= level_count) {
    return;
  }

  TransportPoint transport =
      transport_point(levels, diffusion, advection, fixed_values, reaction_source, reaction_loss,
                      obstacles, x_faces, y_faces, z_faces, has_velocity_field, boundary_kinds,
                      shape, spacing, signal_count, index);
  float scale = crank_nicolson == 0u ? dt : 0.5f * dt;
  float candidate = levels[index] + scale * transport.rate;

  output[index] = candidate;
  if (!isfinite(candidate) || (crank_nicolson == 0u && candidate < 0.0f)) {
    atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
  }
}

kernel void crank_nicolson_jacobi(
    device const float* current [[buffer(0)]], device float* output [[buffer(1)]],
    device const float* right_hand_side [[buffer(2)]], device const float* diffusion [[buffer(3)]],
    device const float4* advection [[buffer(4)]], device const float* fixed_values [[buffer(5)]],
    device atomic_uint* error [[buffer(6)]], constant uint* boundary_kinds [[buffer(7)]],
    constant GridShape& shape [[buffer(8)]], constant float4& spacing [[buffer(9)]],
    constant float& half_dt [[buffer(10)]], constant uint& signal_count [[buffer(11)]],
    constant uint& level_count [[buffer(12)]], device const float* reaction_source [[buffer(13)]],
    device const float* reaction_loss [[buffer(14)]], device const uchar* obstacles [[buffer(15)]],
    device const float* x_faces [[buffer(16)]], device const float* y_faces [[buffer(17)]],
    device const float* z_faces [[buffer(18)]], constant uint& has_velocity_field [[buffer(19)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= level_count) {
    return;
  }
  TransportPoint transport =
      transport_point(current, diffusion, advection, fixed_values, reaction_source, reaction_loss,
                      obstacles, x_faces, y_faces, z_faces, has_velocity_field, boundary_kinds,
                      shape, spacing, signal_count, index);
  float remainder = transport.rate - transport.diagonal * current[index];
  float candidate =
      (right_hand_side[index] + half_dt * remainder) / (1.0f - half_dt * transport.diagonal);
  output[index] = candidate;
  if (!isfinite(candidate)) {
    atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
  }
}

kernel void crank_nicolson_residual_terms(
    device const float* current [[buffer(0)]], device const float* right_hand_side [[buffer(1)]],
    device float* terms [[buffer(2)]], device const float* diffusion [[buffer(3)]],
    device const float4* advection [[buffer(4)]], device const float* fixed_values [[buffer(5)]],
    constant uint* boundary_kinds [[buffer(6)]], constant GridShape& shape [[buffer(7)]],
    constant float4& spacing [[buffer(8)]], constant float& half_dt [[buffer(9)]],
    constant uint& signal_count [[buffer(10)]], constant uint& level_count [[buffer(11)]],
    device const float* reaction_source [[buffer(12)]],
    device const float* reaction_loss [[buffer(13)]], device const uchar* obstacles [[buffer(14)]],
    device const float* x_faces [[buffer(15)]], device const float* y_faces [[buffer(16)]],
    device const float* z_faces [[buffer(17)]], constant uint& has_velocity_field [[buffer(18)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= level_count) {
    return;
  }
  TransportPoint transport =
      transport_point(current, diffusion, advection, fixed_values, reaction_source, reaction_loss,
                      obstacles, x_faces, y_faces, z_faces, has_velocity_field, boundary_kinds,
                      shape, spacing, signal_count, index);
  float residual = right_hand_side[index] - current[index] + half_dt * transport.rate;
  terms[index] = residual * residual;
}

kernel void signal_square_terms(device const float* input [[buffer(0)]],
                                device float* terms [[buffer(1)]],
                                constant uint& element_count [[buffer(2)]],
                                uint index [[thread_position_in_grid]]) {
  if (index >= element_count) {
    return;
  }
  terms[index] = input[index] * input[index];
}

kernel void reduce_signal_sum_pairs(device const float* input [[buffer(0)]],
                                    device float* output [[buffer(1)]],
                                    constant uint& element_count [[buffer(2)]],
                                    uint index [[thread_position_in_grid]]) {
  uint first = index * 2u;
  if (first >= element_count) {
    return;
  }
  float value = input[first];
  if (first + 1u < element_count) {
    value += input[first + 1u];
  }
  output[index] = value;
}
