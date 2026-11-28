float contact_growth_factor(const ContactSolveData solve, float load) {
  if (solve.law == growth_load_law_exponential) {
    return exp(-solve.sensitivity * max(load, 0.0f));
  }
  if (solve.law == growth_load_law_linear) {
    return clamp(1.0f - (load - solve.free_load) / (solve.stall_load - solve.free_load), 0.0f,
                 1.0f);
  }
  return 1.0f;
}

// The velocity, angular velocity, elongation rate, and axial load of every
// cell under a set of multipliers. An incidence entry is a row index shifted
// left by one, with the low bit set when the cell is the row's second cell.
// The response holds the velocity and elongation rate in `linear_length` and
// the angular velocity and load in `rotation`.
// The response of one cell to a set of multipliers: its velocity and
// elongation rate in `linear_length`, its angular velocity and axial load in
// `rotation`. An incidence entry is a row index shifted left by one, with the
// low bit set when the cell is the row's second cell.
ContactResponse respond_body(device const ContactRowData* rows,
                             device const ContactBodyData* bodies, device const float* multipliers,
                             device const uint* incidence_offsets,
                             device const uint* incidence_entries, const ContactSolveData solve,
                             uint body) {
  float3 force = 0.0f;
  float3 torque = 0.0f;
  float load = 0.0f;
  for (uint offset = incidence_offsets[body]; offset < incidence_offsets[body + 1]; ++offset) {
    uint entry = incidence_entries[offset];
    uint row = entry >> 1;
    float multiplier = multipliers[row];
    bool second = (entry & 1u) != 0u;
    float4 linear = second ? rows[row].second_linear : rows[row].first_linear;
    float4 angular = second ? rows[row].second_angular : rows[row].first_angular;
    force += linear.xyz * multiplier;
    torque += angular.xyz * multiplier;
    load += linear.w * multiplier;
  }
  float3 axis = bodies[body].axis_mobility.xyz;
  float transverse = bodies[body].rates.x;
  float axial = bodies[body].rates.y;
  float nominal = bodies[body].rates.z;
  float3 velocity = bodies[body].drift_linear.xyz + force * bodies[body].axis_mobility.w;
  float3 angular_velocity = bodies[body].drift_angular.xyz + torque * transverse +
                            axis * ((axial - transverse) * dot(axis, torque));
  float rate = nominal > 0.0f ? nominal * contact_growth_factor(solve, load) : nominal;
  ContactResponse result;
  result.linear_length = float4(velocity, rate);
  result.rotation = float4(angular_velocity, load);
  return result;
}

// The gap of one row under the multipliers whose response is given.
float row_gap(device const ContactRowData* rows, device const ContactResponse* response,
              float multiplier, const ContactSolveData solve, uint row) {
  uint first = rows[row].first_slot;
  uint second = rows[row].second_slot;
  float rate = dot(rows[row].first_linear.xyz, response[first].linear_length.xyz) +
               dot(rows[row].first_angular.xyz, response[first].rotation.xyz) -
               rows[row].first_linear.w * response[first].linear_length.w;
  if (second != 0xffffffffu) {
    rate += dot(rows[row].second_linear.xyz, response[second].linear_length.xyz) +
            dot(rows[row].second_angular.xyz, response[second].rotation.xyz) -
            rows[row].second_linear.w * response[second].linear_length.w;
  }
  return rows[row].first_angular.w + solve.dt * rate + rows[row].second_angular.w * multiplier;
}

// The velocity, angular velocity, elongation rate, and axial load of every
// cell under a set of multipliers.
kernel void contact_respond(device const ContactRowData* rows [[buffer(0)]],
                            device const ContactBodyData* bodies [[buffer(1)]],
                            device const float* multipliers [[buffer(2)]],
                            device const uint* incidence_offsets [[buffer(3)]],
                            device const uint* incidence_entries [[buffer(4)]],
                            device ContactResponse* response [[buffer(5)]],
                            device const ContactSolveState* state [[buffer(6)]],
                            constant ContactSolveData& solve [[buffer(7)]],
                            constant uint& initial [[buffer(8)]],
                            uint body [[thread_position_in_grid]]) {
  if (body >= solve.body_count || (initial == 0u && state[0].done != 0u)) {
    return;
  }
  response[body] =
      respond_body(rows, bodies, multipliers, incidence_offsets, incidence_entries, solve, body);
}

// The gap of every row under the multipliers whose response is given, and the
// terms of the step that led to them: step . step, step . change,
// change . change, and the row's residual. The initial evaluation has no
// previous step and writes only the residual.
kernel void contact_gaps(device const ContactRowData* rows [[buffer(0)]],
                         device const ContactResponse* response [[buffer(1)]],
                         device const float* multipliers [[buffer(2)]],
                         device float* gaps [[buffer(3)]],
                         device const float* previous_multipliers [[buffer(4)]],
                         device const float* previous_gaps [[buffer(5)]],
                         device float4* terms [[buffer(6)]],
                         device const ContactSolveState* state [[buffer(7)]],
                         constant ContactSolveData& solve [[buffer(8)]],
                         constant uint& initial [[buffer(9)]],
                         uint row [[thread_position_in_grid]]) {
  if (row >= solve.row_count || (initial == 0u && state[0].done != 0u)) {
    return;
  }
  float multiplier = multipliers[row];
  float gap = row_gap(rows, response, multiplier, solve, row);
  gaps[row] = gap;
  float residual = multiplier > 0.0f ? fabs(gap) : max(0.0f, -gap);
  if (!isfinite(gap) || !isfinite(multiplier)) {
    residual = INFINITY;
  }
  if (initial != 0u) {
    terms[row] = float4(0.0f, 0.0f, 0.0f, residual);
    return;
  }
  float step = multiplier - previous_multipliers[row];
  float change = gap - previous_gaps[row];
  terms[row] = float4(step * step, step * change, change * change, residual);
}

// Sums the first three components of neighboring elements and keeps the larger
// fourth.
kernel void contact_reduce(device const float4* input [[buffer(0)]],
                           device float4* output [[buffer(1)]],
                           constant uint& element_count [[buffer(2)]],
                           uint index [[thread_position_in_grid]]) {
  uint first = index * 2;
  if (first >= element_count) {
    return;
  }
  float4 value = input[first];
  if (first + 1 < element_count) {
    float4 other = input[first + 1];
    value = float4(value.xyz + other.xyz, max(value.w, other.w));
  }
  output[index] = value;
}

// Records the step that the reduced terms describe: counts it, stops the
// iteration when its residual is within the tolerance or not finite, and
// otherwise sets the next Barzilai-Borwein step length. The two lengths
// alternate, and a vanishing step . change keeps the current length.
kernel void contact_advance(device const float4* reduced [[buffer(0)]],
                            device ContactSolveState* state [[buffer(1)]],
                            constant ContactSolveData& solve [[buffer(2)]],
                            constant uint& initial [[buffer(3)]],
                            constant float& first_step_length [[buffer(4)]],
                            uint index [[thread_position_in_grid]]) {
  if (index != 0u) {
    return;
  }
  float4 sums = reduced[0];
  bool stop = !(sums.w > solve.tolerance) || isinf(sums.w);
  if (initial != 0u) {
    state[0].step_length = first_step_length;
    state[0].residual = sums.w;
    state[0].iterations = 0u;
    state[0].done = stop ? 1u : 0u;
    return;
  }
  if (state[0].done != 0u) {
    return;
  }
  state[0].iterations += 1u;
  state[0].residual = sums.w;
  if (stop) {
    state[0].done = 1u;
    return;
  }
  if (sums.y > 0.0f && sums.z > 0.0f) {
    float next = (state[0].iterations & 1u) != 0u ? sums.x / sums.y : sums.y / sums.z;
    if (isfinite(next) && next > 0.0f) {
      state[0].step_length = next;
    }
  }
}

// The largest threadgroup the single-group solve runs in.
constant uint contact_solve_group = 1024u;

// Combines the partial sums of every thread of the group into element zero:
// the first three components add and the fourth takes the larger.
void reduce_group(threadgroup float4* partial, uint lane, uint threads) {
  for (uint stride = contact_solve_group / 2u; stride > 0u; stride /= 2u) {
    if (lane < stride && lane + stride < threads) {
      float4 other = partial[lane + stride];
      float4 value = partial[lane];
      partial[lane] = float4(value.xyz + other.xyz, max(value.w, other.w));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// The whole projected Barzilai-Borwein solve of a contact system in one
// threadgroup, as contact_respond_step, contact_gaps_step, and
// contact_advance_step compute it a step at a time over the whole device. The
// multipliers start in pair zero and each step writes the other pair. The
// state ends with the iterations taken and the residual of the multipliers in
// pair `iterations % 2`, whose response `response` holds.
kernel void contact_solve(
    device const ContactRowData* rows [[buffer(0)]],
    device const ContactBodyData* bodies [[buffer(1)]], device float* multipliers_a [[buffer(2)]],
    device float* multipliers_b [[buffer(3)]], device float* gaps_a [[buffer(4)]],
    device float* gaps_b [[buffer(5)]], device const uint* incidence_offsets [[buffer(6)]],
    device const uint* incidence_entries [[buffer(7)]],
    device ContactResponse* response [[buffer(8)]], device ContactSolveState* state [[buffer(9)]],
    constant ContactSolveData& solve [[buffer(10)]], constant uint& max_iterations [[buffer(11)]],
    constant float& first_step_length [[buffer(12)]], uint lane [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
  threadgroup float4 partial[contact_solve_group];
  threadgroup float step_length;
  threadgroup uint done;

  device float* multipliers[2] = {multipliers_a, multipliers_b};
  device float* gaps[2] = {gaps_a, gaps_b};

  for (uint body = lane; body < solve.body_count; body += threads) {
    response[body] = respond_body(rows, bodies, multipliers_a, incidence_offsets, incidence_entries,
                                  solve, body);
  }
  threadgroup_barrier(mem_flags::mem_device);
  float4 local = float4(0.0f);
  for (uint row = lane; row < solve.row_count; row += threads) {
    float multiplier = multipliers_a[row];
    float gap = row_gap(rows, response, multiplier, solve, row);
    gaps_a[row] = gap;
    float residual = multiplier > 0.0f ? fabs(gap) : max(0.0f, -gap);
    if (!isfinite(gap) || !isfinite(multiplier)) {
      residual = INFINITY;
    }
    local.w = max(local.w, residual);
  }
  partial[lane] = local;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  reduce_group(partial, lane, threads);
  if (lane == 0u) {
    float residual = partial[0].w;
    state[0].step_length = first_step_length;
    state[0].residual = residual;
    state[0].iterations = 0u;
    done = (!(residual > solve.tolerance) || isinf(residual)) ? 1u : 0u;
    step_length = first_step_length;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

  uint current = 0u;
  for (uint iteration = 1u; iteration <= max_iterations && done == 0u; ++iteration) {
    uint next = 1u - current;
    float length = step_length;
    for (uint row = lane; row < solve.row_count; row += threads) {
      multipliers[next][row] = max(multipliers[current][row] - length * gaps[current][row], 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint body = lane; body < solve.body_count; body += threads) {
      response[body] = respond_body(rows, bodies, multipliers[next], incidence_offsets,
                                    incidence_entries, solve, body);
    }
    threadgroup_barrier(mem_flags::mem_device);
    float4 terms = float4(0.0f);
    for (uint row = lane; row < solve.row_count; row += threads) {
      float multiplier = multipliers[next][row];
      float gap = row_gap(rows, response, multiplier, solve, row);
      gaps[next][row] = gap;
      float residual = multiplier > 0.0f ? fabs(gap) : max(0.0f, -gap);
      if (!isfinite(gap) || !isfinite(multiplier)) {
        residual = INFINITY;
      }
      float step = multiplier - multipliers[current][row];
      float change = gap - gaps[current][row];
      terms += float4(step * step, step * change, change * change, 0.0f);
      terms.w = max(terms.w, residual);
    }
    partial[lane] = terms;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    reduce_group(partial, lane, threads);
    if (lane == 0u) {
      float4 sums = partial[0];
      state[0].iterations = iteration;
      state[0].residual = sums.w;
      if (!(sums.w > solve.tolerance) || isinf(sums.w)) {
        done = 1u;
      } else if (sums.y > 0.0f && sums.z > 0.0f) {
        float candidate = (iteration & 1u) != 0u ? sums.x / sums.y : sums.y / sums.z;
        if (isfinite(candidate) && candidate > 0.0f) {
          step_length = candidate;
        }
      }
      state[0].step_length = step_length;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    current = next;
  }
}

// The multiplier of `row` after one projected step from the multipliers and
// gaps in `multipliers` and `gaps`.
inline float projected_multiplier(device const float* multipliers, device const float* gaps,
                                  float step_length, uint row) {
  return max(multipliers[row] - step_length * gaps[row], 0.0f);
}

// The response of every cell to the multipliers one projected step from
// `multipliers`, computing each projection where it is read.
kernel void contact_respond_step(device const ContactRowData* rows [[buffer(0)]],
                                 device const ContactBodyData* bodies [[buffer(1)]],
                                 device const float* multipliers [[buffer(2)]],
                                 device const float* gaps [[buffer(3)]],
                                 device const uint* incidence_offsets [[buffer(4)]],
                                 device const uint* incidence_entries [[buffer(5)]],
                                 device ContactResponse* response [[buffer(6)]],
                                 device const ContactSolveState* state [[buffer(7)]],
                                 constant ContactSolveData& solve [[buffer(8)]],
                                 uint body [[thread_position_in_grid]]) {
  if (body >= solve.body_count || state[0].done != 0u) {
    return;
  }
  const float length = state[0].step_length;
  float3 force = 0.0f;
  float3 torque = 0.0f;
  float load = 0.0f;
  for (uint offset = incidence_offsets[body]; offset < incidence_offsets[body + 1]; ++offset) {
    uint entry = incidence_entries[offset];
    uint row = entry >> 1;
    float multiplier = projected_multiplier(multipliers, gaps, length, row);
    bool second = (entry & 1u) != 0u;
    float4 linear = second ? rows[row].second_linear : rows[row].first_linear;
    float4 angular = second ? rows[row].second_angular : rows[row].first_angular;
    force += linear.xyz * multiplier;
    torque += angular.xyz * multiplier;
    load += linear.w * multiplier;
  }
  float3 axis = bodies[body].axis_mobility.xyz;
  float transverse = bodies[body].rates.x;
  float axial = bodies[body].rates.y;
  float nominal = bodies[body].rates.z;
  float3 velocity = bodies[body].drift_linear.xyz + force * bodies[body].axis_mobility.w;
  float3 angular_velocity = bodies[body].drift_angular.xyz + torque * transverse +
                            axis * ((axial - transverse) * dot(axis, torque));
  float rate = nominal > 0.0f ? nominal * contact_growth_factor(solve, load) : nominal;
  response[body].linear_length = float4(velocity, rate);
  response[body].rotation = float4(angular_velocity, load);
}

// The threadgroup width of contact_gaps_step.
constant uint contact_step_group = 256u;

// Takes the projected step into `next_multipliers` and `next_gaps`, and sums
// each threadgroup's step terms into `partials`: step . step, step . change,
// change . change, and the largest row residual.
kernel void contact_gaps_step(
    device const ContactRowData* rows [[buffer(0)]],
    device const ContactResponse* response [[buffer(1)]],
    device const float* multipliers [[buffer(2)]], device const float* gaps [[buffer(3)]],
    device float* next_multipliers [[buffer(4)]], device float* next_gaps [[buffer(5)]],
    device float4* partials [[buffer(6)]], device const ContactSolveState* state [[buffer(7)]],
    constant ContactSolveData& solve [[buffer(8)]], uint row [[thread_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
  threadgroup float4 partial[contact_step_group];
  if (state[0].done != 0u) {
    return;
  }
  float4 terms = float4(0.0f);
  if (row < solve.row_count) {
    float multiplier = projected_multiplier(multipliers, gaps, state[0].step_length, row);
    float gap = row_gap(rows, response, multiplier, solve, row);
    next_multipliers[row] = multiplier;
    next_gaps[row] = gap;
    float residual = multiplier > 0.0f ? fabs(gap) : max(0.0f, -gap);
    if (!isfinite(gap) || !isfinite(multiplier)) {
      residual = INFINITY;
    }
    float step = multiplier - multipliers[row];
    float change = gap - gaps[row];
    terms = float4(step * step, step * change, change * change, residual);
  }
  partial[lane] = terms;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = contact_step_group / 2u; stride > 0u; stride /= 2u) {
    if (lane < stride) {
      float4 other = partial[lane + stride];
      float4 value = partial[lane];
      partial[lane] = float4(value.xyz + other.xyz, max(value.w, other.w));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0u) {
    partials[group] = partial[0];
  }
}

// Sums the threadgroup partials of contact_gaps_step in a fixed order and
// records the step as contact_advance does.
kernel void contact_advance_step(device const float4* partials [[buffer(0)]],
                                 device ContactSolveState* state [[buffer(1)]],
                                 constant ContactSolveData& solve [[buffer(2)]],
                                 constant uint& group_count [[buffer(3)]],
                                 uint lane [[thread_position_in_threadgroup]],
                                 uint lanes [[threads_per_threadgroup]]) {
  threadgroup float4 partial[contact_solve_group];
  if (state[0].done != 0u) {
    return;
  }
  float4 local = float4(0.0f);
  for (uint group = lane; group < group_count; group += lanes) {
    float4 value = partials[group];
    local = float4(local.xyz + value.xyz, max(local.w, value.w));
  }
  partial[lane] = local;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  reduce_group(partial, lane, lanes);
  if (lane != 0u) {
    return;
  }
  float4 sums = partial[0];
  bool stop = !(sums.w > solve.tolerance) || isinf(sums.w);
  state[0].iterations += 1u;
  state[0].residual = sums.w;
  if (stop) {
    state[0].done = 1u;
    return;
  }
  if (sums.y > 0.0f && sums.z > 0.0f) {
    float next = (state[0].iterations & 1u) != 0u ? sums.x / sums.y : sums.y / sums.z;
    if (isfinite(next) && next > 0.0f) {
      state[0].step_length = next;
    }
  }
}
