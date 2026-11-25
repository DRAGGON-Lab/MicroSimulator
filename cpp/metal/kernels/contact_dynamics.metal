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

// next = max(multipliers - step_length * gaps, 0).
kernel void contact_project(device const float* multipliers [[buffer(0)]],
                            device const float* gaps [[buffer(1)]],
                            device float* next [[buffer(2)]],
                            device const ContactSolveState* state [[buffer(3)]],
                            constant ContactSolveData& solve [[buffer(4)]],
                            uint row [[thread_position_in_grid]]) {
  if (row >= solve.row_count || state[0].done != 0u) {
    return;
  }
  next[row] = max(multipliers[row] - state[0].step_length * gaps[row], 0.0f);
}

// The velocity, angular velocity, elongation rate, and axial load of every
// cell under a set of multipliers. An incidence entry is a row index shifted
// left by one, with the low bit set when the cell is the row's second cell.
// The response holds the velocity and elongation rate in `linear_length` and
// the angular velocity and load in `rotation`.
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
  response[body].linear_length = float4(velocity, rate);
  response[body].rotation = float4(angular_velocity, load);
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
  float multiplier = multipliers[row];
  float gap = rows[row].first_angular.w + solve.dt * rate + rows[row].second_angular.w * multiplier;
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
