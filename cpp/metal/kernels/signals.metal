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
  lower.x = x == 0u ? exterior_value(boundary_kinds, fixed_values, 0u, y * shape.z + z, signal,
                                     signal_count, current,
                                     grid_level(levels, shape, signal, shape.x - 1u, y, z))
                    : grid_level(levels, shape, signal, x - 1u, y, z);
  upper.x = x + 1u == shape.x
                ? exterior_value(boundary_kinds, fixed_values, 1u, y * shape.z + z, signal,
                                 signal_count, current, grid_level(levels, shape, signal, 0u, y, z))
                : grid_level(levels, shape, signal, x + 1u, y, z);
  lower.y = y == 0u ? exterior_value(boundary_kinds, fixed_values, 2u, x * shape.z + z, signal,
                                     signal_count, current,
                                     grid_level(levels, shape, signal, x, shape.y - 1u, z))
                    : grid_level(levels, shape, signal, x, y - 1u, z);
  upper.y = y + 1u == shape.y
                ? exterior_value(boundary_kinds, fixed_values, 3u, x * shape.z + z, signal,
                                 signal_count, current, grid_level(levels, shape, signal, x, 0u, z))
                : grid_level(levels, shape, signal, x, y + 1u, z);
  lower.z = z == 0u ? exterior_value(boundary_kinds, fixed_values, 4u, x * shape.y + y, signal,
                                     signal_count, current,
                                     grid_level(levels, shape, signal, x, y, shape.z - 1u))
                    : grid_level(levels, shape, signal, x, y, z - 1u);
  upper.z = z + 1u == shape.z
                ? exterior_value(boundary_kinds, fixed_values, 5u, x * shape.y + y, signal,
                                 signal_count, current, grid_level(levels, shape, signal, x, y, 0u))
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
    if (boundary_kinds[12] == 1u) {
      rate -= van_leer_correction(levels, obstacles, shape, boundary_kinds, signal, uint3(x, y, z),
                                  axis, current, lower[axis], upper[axis], closed_lower[axis],
                                  closed_upper[axis], faces.lower[axis], faces.upper[axis]) *
              inverse_spacing;
    }
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

// The level a transport thread works on. Threads number the fluid sites of each
// signal in turn, so obstacle sites cost nothing and keep the zero they hold.
uint fluid_level(device const uint* fluid_sites, uint fluid_count, uint sites, uint entry) {
  uint signal = entry / fluid_count;
  return signal * sites + fluid_sites[entry - signal * fluid_count];
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
    device const uint* fluid_sites [[buffer(20)]], constant uint& fluid_count [[buffer(21)]],
    uint entry [[thread_position_in_grid]]) {
  if (entry >= signal_count * fluid_count) {
    return;
  }
  uint index = fluid_level(fluid_sites, fluid_count, shape.sites, entry);

  TransportPoint transport =
      transport_point(levels, diffusion, advection, fixed_values, reaction_source, reaction_loss,
                      obstacles, x_faces, y_faces, z_faces, has_velocity_field, boundary_kinds,
                      shape, spacing, signal_count, index);
  float scale = crank_nicolson == 0u ? dt : (crank_nicolson == 1u ? 0.5f * dt : 0.0f);
  float candidate = levels[index] + scale * transport.rate;

  output[index] = candidate;
  // Under the van Leer scheme an explicit level may fall below zero by
  // rounding; the host judges those when it takes the step's levels.
  if (!isfinite(candidate) ||
      (crank_nicolson == 0u && boundary_kinds[12] == 0u && candidate < 0.0f)) {
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
    device const uint* fluid_sites [[buffer(20)]], constant uint& fluid_count [[buffer(21)]],
    device const uint* gate [[buffer(22)]], uint entry [[thread_position_in_grid]]) {
  if (entry >= signal_count * fluid_count || gate[0] != 0u) {
    return;
  }
  uint index = fluid_level(fluid_sites, fluid_count, shape.sites, entry);
  TransportPoint transport =
      transport_point(current, diffusion, advection, fixed_values, reaction_source, reaction_loss,
                      obstacles, x_faces, y_faces, z_faces, has_velocity_field, boundary_kinds,
                      shape, spacing, signal_count, index);
  float remainder = transport.rate - transport.diagonal * current[index];
  float update =
      (right_hand_side[index] + half_dt * remainder) / (1.0f - half_dt * transport.diagonal);
  float relaxation = boundary_kinds[12] == 1u ? van_leer_relaxation : 1.0f;
  float candidate = current[index] + relaxation * (update - current[index]);
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
    device const uint* fluid_sites [[buffer(19)]], constant uint& fluid_count [[buffer(20)]],
    device const uint* gate [[buffer(21)]], uint entry [[thread_position_in_grid]]) {
  if (entry >= signal_count * fluid_count || gate[0] != 0u) {
    return;
  }
  uint index = fluid_level(fluid_sites, fluid_count, shape.sites, entry);
  TransportPoint transport =
      transport_point(current, diffusion, advection, fixed_values, reaction_source, reaction_loss,
                      obstacles, x_faces, y_faces, z_faces, has_velocity_field, boundary_kinds,
                      shape, spacing, signal_count, index);
  float residual = right_hand_side[index] - current[index] + half_dt * transport.rate;
  terms[entry] = residual * residual;
}

// The squares of a field's fluid levels, one term per fluid level.
kernel void signal_square_terms(device const float* input [[buffer(0)]],
                                device float* terms [[buffer(1)]],
                                constant uint& element_count [[buffer(2)]],
                                device const uint* fluid_sites [[buffer(3)]],
                                constant uint& fluid_count [[buffer(4)]],
                                constant uint& sites [[buffer(5)]],
                                uint entry [[thread_position_in_grid]]) {
  if (entry >= element_count) {
    return;
  }
  float value = input[fluid_level(fluid_sites, fluid_count, sites, entry)];
  terms[entry] = value * value;
}

kernel void reduce_signal_sum_pairs(device const float* input [[buffer(0)]],
                                    device float* output [[buffer(1)]],
                                    constant uint& element_count [[buffer(2)]],
                                    device const uint* gate [[buffer(3)]],
                                    uint index [[thread_position_in_grid]]) {
  uint first = index * 2u;
  if (first >= element_count || gate[0] != 0u) {
    return;
  }
  float value = input[first];
  if (first + 1u < element_count) {
    value += input[first + 1u];
  }
  output[index] = value;
}

// Records the sweep whose residual terms `reduction` sums: counts it, and
// finishes the batch when its residual RMS is within `threshold`, is not
// finite, or a level of the sweep was invalid. Does nothing once the batch is
// finished.
kernel void signal_sweep_check(device const float* reduction [[buffer(0)]],
                               device const uint* error [[buffer(1)]],
                               device SignalSweepState* state [[buffer(2)]],
                               constant float& threshold [[buffer(3)]],
                               constant uint& level_count [[buffer(4)]],
                               constant uint& fluid_level_count [[buffer(5)]],
                               uint index [[thread_position_in_grid]]) {
  if (index != 0u || state[0].done != 0u) {
    return;
  }
  state[0].iterations += 1u;
  float sum = fluid_level_count == 0u ? 0.0f : reduction[0];
  float rms = sqrt(sum / float(level_count));
  state[0].residual_rms = rms;
  if (error[0] != 0u) {
    state[0].invalid = 1u;
    state[0].done = 1u;
  } else if (!isfinite(rms) || rms <= threshold) {
    state[0].done = 1u;
  }
}
