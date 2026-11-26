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

inline float2 pair_divide_pair(float2 a, float2 b) {
  const float first = a.x / b.x;
  float2 remainder = pair_subtract(a, pair_scale(b, first));
  const float second = remainder.x / b.x;
  remainder = pair_subtract(remainder, pair_scale(b, second));
  const float third = remainder.x / b.x;
  const float2 quotient = quick_two_sum(first, second);
  return pair_add(quotient, float2(third, 0.0f));
}

inline bool pair_at_most(float2 a, float2 b) {
  return a.x < b.x || (a.x == b.x && a.y <= b.y);
}

// The scalars of a solve, kept on the device so that a batch of iterations
// runs without the host: double-word values at these places in one buffer,
// and counters at these places in another.
constant uint depth_rho = 0u;
constant uint depth_curvature = 1u;
constant uint depth_residual_squared = 2u;
constant uint depth_next_rho = 3u;
constant uint depth_alpha = 4u;
constant uint depth_beta = 5u;

constant uint depth_iterations = 0u;
constant uint depth_stopped = 1u;
constant uint depth_converged = 2u;
constant uint depth_failure = 3u;

// The operator over the active sites of a depth-averaged system. Each has six
// neighbors, lower then upper along x, y and z, named by their place among
// the active sites, with the conductance to each.
kernel void depth_pair_operator(device const float2* input [[buffer(0)]],
                                device const uint* neighbors [[buffer(1)]],
                                device const float* conductance [[buffer(2)]],
                                device float2* output [[buffer(3)]],
                                constant uint& count [[buffer(4)]],
                                constant DepthFlowScales& scales [[buffer(5)]],
                                device const float* boundary_conductance [[buffer(6)]],
                                uint index [[thread_position_in_grid]]) {
  if (index >= count) {
    return;
  }
  // The site's coupling to the pressures held beyond its open exterior faces.
  float2 result = pair_scale(input[index], boundary_conductance[index]);
  for (uint slot = 0; slot < 6u; ++slot) {
    const uint neighbor = neighbors[6u * index + slot];
    if (neighbor == 0xffffffffu) {
      continue;
    }
    const float2 scaled =
        pair_scale(inverse_square(scales, slot / 2u), conductance[6u * index + slot]);
    result = pair_add(result, pair_multiply(scaled, pair_subtract(input[index], input[neighbor])));
  }
  output[index] = result;
}

kernel void depth_pair_velocity(device const float2* pressure [[buffer(0)]],
                                device const float* mobility [[buffer(1)]],
                                device float* velocity [[buffer(2)]],
                                constant FlowGridParameters& grid [[buffer(3)]],
                                device const float* face_conductance [[buffer(4)]],
                                device const float* face_pressure [[buffer(5)]],
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
  } else {
    // An exterior face: closed unless it carries a conductance to the
    // pressure held beyond it.
    const uint site = has_upper ? upper : lower;
    value = pair_scale(pair_subtract(float2(face_pressure[index], 0.0f), pressure[site]),
                       face_conductance[index]);
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
                              device const float2* scalars [[buffer(4)]],
                              constant uint& count [[buffer(5)]],
                              uint index [[thread_position_in_grid]]) {
  if (index >= count) {
    return;
  }
  const float2 alpha = scalars[depth_alpha];
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
                                 device const float2* scalars [[buffer(2)]],
                                 constant uint& count [[buffer(3)]],
                                 uint index [[thread_position_in_grid]]) {
  if (index >= count) {
    return;
  }
  const float2 beta = scalars[depth_beta];
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

// The sum of a dot product's block partials, in block order.
inline float2 pair_total(device const float2* partials, uint block_count) {
  float2 sum = float2(0.0f);
  for (uint block = 0; block < block_count; ++block) {
    sum = pair_add(sum, partials[block]);
  }
  return sum;
}

// Folds the block partials of a dot product: one thread adds one fixed-width
// group of them in order. A single thread adding every partial of a large
// system would be the slowest step of an iteration.
kernel void depth_pair_fold(device const float2* partials [[buffer(0)]],
                            device float2* folded [[buffer(1)]],
                            constant uint& block_count [[buffer(2)]],
                            constant uint& group_width [[buffer(3)]],
                            uint group [[thread_position_in_grid]]) {
  const uint first = group * group_width;
  if (first >= block_count) {
    return;
  }
  const uint last = min(block_count, first + group_width);
  float2 sum = float2(0.0f);
  for (uint block = first; block < last; ++block) {
    sum = pair_add(sum, partials[block]);
  }
  folded[group] = sum;
}

// One thread adds the block partials of a dot product and keeps the sum among
// the solve's scalars.
kernel void depth_pair_sum(device const float2* partials [[buffer(0)]],
                           device float2* scalars [[buffer(1)]],
                           constant uint& block_count [[buffer(2)]],
                           constant uint& place [[buffer(3)]],
                           uint worker [[thread_position_in_grid]]) {
  if (worker != 0u) {
    return;
  }
  scalars[place] = pair_total(partials, block_count);
}

// Begins an iteration: the step length from the curvature the operator left.
// A solve that has stopped takes a step of no length, which changes nothing.
kernel void depth_pair_step_length(device float2* scalars [[buffer(0)]],
                                   device uint* counters [[buffer(1)]],
                                   uint worker [[thread_position_in_grid]]) {
  if (worker != 0u) {
    return;
  }
  scalars[depth_alpha] = float2(0.0f);
  if (counters[depth_stopped] != 0u) {
    return;
  }
  counters[depth_iterations] += 1u;
  const float2 curvature = scalars[depth_curvature];
  if (!isfinite(curvature.x) || !(curvature.x > 0.0f)) {
    counters[depth_failure] = 1u;
    counters[depth_stopped] = 1u;
    return;
  }
  const float2 alpha = pair_divide_pair(scalars[depth_rho], curvature);
  if (!isfinite(alpha.x) || !isfinite(alpha.y)) {
    counters[depth_failure] = 2u;
    counters[depth_stopped] = 1u;
    return;
  }
  scalars[depth_alpha] = alpha;
}

// Ends an iteration: stops the solve if its residual meets the threshold,
// and otherwise leaves the weight of the next direction.
kernel void depth_pair_step_weight(device float2* scalars [[buffer(0)]],
                                   device uint* counters [[buffer(1)]],
                                   constant float2& threshold [[buffer(2)]],
                                   uint worker [[thread_position_in_grid]]) {
  if (worker != 0u) {
    return;
  }
  scalars[depth_beta] = float2(0.0f);
  if (counters[depth_stopped] != 0u) {
    return;
  }
  const float2 residual_squared = scalars[depth_residual_squared];
  if (!isfinite(residual_squared.x)) {
    counters[depth_failure] = 3u;
    counters[depth_stopped] = 1u;
    return;
  }
  if (pair_at_most(residual_squared, threshold)) {
    counters[depth_converged] = 1u;
    counters[depth_stopped] = 1u;
    return;
  }
  const float2 rho = scalars[depth_rho];
  const float2 next_rho = scalars[depth_next_rho];
  if (!isfinite(next_rho.x) || rho.x == 0.0f) {
    counters[depth_failure] = 4u;
    counters[depth_stopped] = 1u;
    return;
  }
  const float2 beta = pair_divide_pair(next_rho, rho);
  if (!isfinite(beta.x) || !isfinite(beta.y)) {
    counters[depth_failure] = 5u;
    counters[depth_stopped] = 1u;
    return;
  }
  scalars[depth_beta] = beta;
  scalars[depth_rho] = next_rho;
}
