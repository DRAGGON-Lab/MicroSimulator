// Depth-averaged pressure solve in double-word arithmetic.
//
// Metal has no 64-bit floating-point type. Each value here is an unevaluated
// sum of two floats (x: leading word, y: trailing word), which carries about
// 14 significant decimal digits, so the conjugate-gradient iteration follows
// the double-precision iteration of the CPU and CUDA backends. The identities
// below are exact only under IEEE evaluation: this source must be compiled
// with fast math disabled.

inline float2 two_sum(float a, float b) {
  const float sum = a + b;
  const float b_virtual = sum - a;
  return float2(sum, (a - (sum - b_virtual)) + (b - b_virtual));
}

// Requires |a| >= |b| or a == 0.
inline float2 quick_two_sum(float a, float b) {
  const float sum = a + b;
  return float2(sum, b - (sum - a));
}

inline float2 two_product(float a, float b) {
  const float product = a * b;
  return float2(product, fma(a, b, -product));
}

inline float2 pair_add(float2 a, float2 b) {
  float2 sum = two_sum(a.x, b.x);
  const float2 trailing = two_sum(a.y, b.y);
  sum.y += trailing.x;
  sum = quick_two_sum(sum.x, sum.y);
  sum.y += trailing.y;
  return quick_two_sum(sum.x, sum.y);
}

inline float2 pair_subtract(float2 a, float2 b) {
  return pair_add(a, -b);
}

inline float2 pair_multiply(float2 a, float2 b) {
  float2 product = two_product(a.x, b.x);
  product.y += a.x * b.y + a.y * b.x;
  return quick_two_sum(product.x, product.y);
}

inline float2 pair_scale(float2 a, float b) {
  float2 product = two_product(a.x, b);
  product.y += a.y * b;
  return quick_two_sum(product.x, product.y);
}

inline float2 pair_divide(float2 a, float b) {
  const float quotient = a.x / b;
  const float2 product = two_product(quotient, b);
  float2 remainder = two_sum(a.x, -product.x);
  remainder.y -= product.y;
  remainder.y += a.y;
  return quick_two_sum(quotient, (remainder.x + remainder.y) / b);
}

inline float2 inverse_square(constant DepthFlowScales& scales, uint axis) {
  return float2(scales.inverse_square_high[axis], scales.inverse_square_low[axis]);
}

kernel void depth_pair_operator(device const float2* input [[buffer(0)]],
                                device const float* mobility [[buffer(1)]],
                                device const float* diagonal [[buffer(2)]],
                                device float2* output [[buffer(3)]],
                                constant FlowGridParameters& grid [[buffer(4)]],
                                constant DepthFlowScales& scales [[buffer(5)]],
                                uint index [[thread_position_in_grid]]) {
  if (index >= grid.site_count) {
    return;
  }
  if (diagonal[index] == 0.0f) {
    output[index] = float2(0.0f);
    return;
  }
  const uint3 coordinate = site_coordinate(index, grid);
  const uint flow_axis = grid.flow_axis;
  float2 result = float2(0.0f);
  const float2 boundary = pair_multiply(
      pair_scale(inverse_square(scales, flow_axis), 2.0f * mobility[index]), input[index]);
  if (coordinate[flow_axis] == 0) {
    result = pair_add(result, boundary);
  }
  if (coordinate[flow_axis] + 1 == grid.dimensions[flow_axis]) {
    result = pair_add(result, boundary);
  }
  for (uint axis = 0; axis < 3; ++axis) {
    if (coordinate[axis] > 0) {
      uint3 neighbor = coordinate;
      neighbor[axis] -= 1;
      const uint neighbor_index = site_index(neighbor, grid);
      const float2 conductance = pair_scale(
          inverse_square(scales, axis), harmonic_mean(mobility[index], mobility[neighbor_index]));
      result = pair_add(
          result, pair_multiply(conductance, pair_subtract(input[index], input[neighbor_index])));
    }
    if (coordinate[axis] + 1 < grid.dimensions[axis]) {
      uint3 neighbor = coordinate;
      neighbor[axis] += 1;
      const uint neighbor_index = site_index(neighbor, grid);
      const float2 conductance = pair_scale(
          inverse_square(scales, axis), harmonic_mean(mobility[index], mobility[neighbor_index]));
      result = pair_add(
          result, pair_multiply(conductance, pair_subtract(input[index], input[neighbor_index])));
    }
  }
  output[index] = result;
}

kernel void depth_pair_velocity(device const float2* pressure [[buffer(0)]],
                                device const float* mobility [[buffer(1)]],
                                device float* velocity [[buffer(2)]],
                                constant FlowGridParameters& grid [[buffer(3)]],
                                uint index [[thread_position_in_grid]]) {
  if (index >= grid.total_face_count) {
    return;
  }
  const FaceCoordinate face = face_coordinate(index, grid);
  const uint component = face.component;
  const bool has_lower = face.coordinate[component] > 0;
  const bool has_upper = face.coordinate[component] < grid.dimensions[component];
  uint3 lower_coordinate = face.coordinate;
  if (has_lower) {
    lower_coordinate[component] -= 1;
  }
  const uint lower = has_lower ? site_index(lower_coordinate, grid) : 0;
  const uint upper = has_upper ? site_index(face.coordinate, grid) : 0;
  const float spacing = grid.spacing[component];
  float2 value = float2(0.0f);
  if (has_lower && has_upper) {
    value = pair_divide(pair_scale(pair_subtract(pressure[lower], pressure[upper]),
                                   harmonic_mean(mobility[lower], mobility[upper])),
                        spacing);
  } else if (component == grid.flow_axis && has_upper) {
    value = pair_divide(
        pair_scale(pair_subtract(float2(1.0f, 0.0f), pressure[upper]), 2.0f * mobility[upper]),
        spacing);
  } else if (component == grid.flow_axis && has_lower) {
    value = pair_divide(pair_scale(pressure[lower], 2.0f * mobility[lower]), spacing);
  }
  velocity[index] = value.x + value.y;
}

kernel void depth_pair_initialize(
    device const float* right_hand_side [[buffer(0)]], device const float* diagonal [[buffer(1)]],
    device float2* solution [[buffer(2)]], device float2* residual [[buffer(3)]],
    device float2* preconditioned [[buffer(4)]], device float2* direction [[buffer(5)]],
    constant uint& count [[buffer(6)]], uint index [[thread_position_in_grid]]) {
  if (index >= count) {
    return;
  }
  const float2 value = float2(right_hand_side[index], 0.0f);
  const float2 scaled = diagonal[index] > 0.0f ? pair_divide(value, diagonal[index]) : float2(0.0f);
  solution[index] = float2(0.0f);
  residual[index] = value;
  preconditioned[index] = scaled;
  direction[index] = scaled;
}

kernel void depth_pair_update(device float2* solution [[buffer(0)]],
                              device float2* residual [[buffer(1)]],
                              device const float2* direction [[buffer(2)]],
                              device const float2* transformed [[buffer(3)]],
                              constant float2& alpha [[buffer(4)]],
                              constant uint& count [[buffer(5)]],
                              uint index [[thread_position_in_grid]]) {
  if (index >= count) {
    return;
  }
  solution[index] = pair_add(solution[index], pair_multiply(alpha, direction[index]));
  residual[index] = pair_subtract(residual[index], pair_multiply(alpha, transformed[index]));
}

kernel void depth_pair_precondition(device const float2* residual [[buffer(0)]],
                                    device const float* diagonal [[buffer(1)]],
                                    device float2* preconditioned [[buffer(2)]],
                                    constant uint& count [[buffer(3)]],
                                    uint index [[thread_position_in_grid]]) {
  if (index >= count) {
    return;
  }
  preconditioned[index] =
      diagonal[index] > 0.0f ? pair_divide(residual[index], diagonal[index]) : float2(0.0f);
}

kernel void depth_pair_direction(device const float2* preconditioned [[buffer(0)]],
                                 device float2* direction [[buffer(1)]],
                                 constant float2& beta [[buffer(2)]],
                                 constant uint& count [[buffer(3)]],
                                 uint index [[thread_position_in_grid]]) {
  if (index >= count) {
    return;
  }
  direction[index] = pair_add(preconditioned[index], pair_multiply(beta, direction[index]));
}

// One thread sums the products of one fixed-width block in index order; the
// host adds the block partials.
kernel void depth_pair_dot(device const float2* left [[buffer(0)]],
                           device const float2* right [[buffer(1)]],
                           device float2* partials [[buffer(2)]],
                           constant uint& count [[buffer(3)]],
                           constant uint& block_width [[buffer(4)]],
                           uint block [[thread_position_in_grid]]) {
  const uint first = block * block_width;
  if (first >= count) {
    return;
  }
  const uint last = min(count, first + block_width);
  float2 sum = float2(0.0f);
  for (uint index = first; index < last; ++index) {
    sum = pair_add(sum, pair_multiply(left[index], right[index]));
  }
  partials[block] = sum;
}
