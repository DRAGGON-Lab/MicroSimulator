// The projected Barzilai-Borwein solve of a contact system, one cell or one
// row at a time.
//
// A step from multipliers m with gaps g and step length s takes the projected
// multipliers m' = max(m - s g, 0). Each cell responds to m', each row takes
// its gap g' under that response, and each row contributes the terms
// (|m' - m|^2, (m' - m).(g' - g), |g' - g|^2, residual). The terms are summed
// in a fixed order, so a solve repeats exactly, and the sums set the next
// step length.
//
// An incidence entry is a row index shifted left by one, with the low bit set
// when the cell is the row's second cell. A cell's entries are in row order.

// The factor by which an axial load slows elongation.
inline float contact_growth_factor(uint law, float sensitivity, float free_load, float stall_load,
                                   float load) {
  if (law == growth_load_law_exponential) {
    return exp(-sensitivity * (load > 0.0f ? load : 0.0f));
  }
  if (law == growth_load_law_linear) {
    return clamp(1.0f - (load - free_load) / (stall_load - free_load), 0.0f, 1.0f);
  }
  return 1.0f;
}

// The multiplier of a row one projected step of length `length` on.
inline float contact_projected(float multiplier, float gap, float length) {
  float value = multiplier - length * gap;
  return value > 0.0f ? value : 0.0f;
}

// The multiplier of `row` the response is taken at: the multiplier itself, or
// with `project` the multiplier one projected step on.
inline float contact_multiplier_at(device const float* multipliers, device const float* gaps,
                                   float length, uint project, uint row) {
  return project != 0u ? contact_projected(multipliers[row], gaps[row], length) : multipliers[row];
}

// The response of `body` to the multipliers: its velocity and elongation rate
// in `linear_length`, its angular velocity and axial load in `rotation`.
inline ContactResponse contact_respond_body(uint body, device const ContactRowData* rows,
                                            device const ContactBodyData* bodies,
                                            device const float* multipliers,
                                            device const float* gaps, float length, uint project,
                                            device const uint* incidence_offsets,
                                            device const uint* incidence_entries,
                                            ContactSolveData solve) {
  float force_x = 0.0f;
  float force_y = 0.0f;
  float force_z = 0.0f;
  float torque_x = 0.0f;
  float torque_y = 0.0f;
  float torque_z = 0.0f;
  float load = 0.0f;
  for (uint offset = incidence_offsets[body]; offset < incidence_offsets[body + 1u]; ++offset) {
    uint entry = incidence_entries[offset];
    uint row = entry >> 1;
    float multiplier = contact_multiplier_at(multipliers, gaps, length, project, row);
    if (multiplier == 0.0f) {
      continue;
    }
    bool second = (entry & 1u) != 0u;
    float4 linear = second ? rows[row].second_linear : rows[row].first_linear;
    float4 angular = second ? rows[row].second_angular : rows[row].first_angular;
    force_x += multiplier * linear.x;
    force_y += multiplier * linear.y;
    force_z += multiplier * linear.z;
    torque_x += multiplier * angular.x;
    torque_y += multiplier * angular.y;
    torque_z += multiplier * angular.z;
    load += linear.w * multiplier;
  }
  ContactBodyData data = bodies[body];
  float translational = data.axis_mobility.w;
  float transverse = data.rates.x;
  float axial = data.rates.y;
  float nominal = data.rates.z;
  float3 axis = float3(data.axis_mobility.x, data.axis_mobility.y, data.axis_mobility.z);
  float3 torque = float3(torque_x, torque_y, torque_z);
  float3 angular = ((torque * transverse) + (axis * (axial - transverse) * dot(axis, torque)));
  float rate = nominal > 0.0f
                   ? nominal * contact_growth_factor(solve.law, solve.sensitivity, solve.free_load,
                                                     solve.stall_load, load)
                   : nominal;
  ContactResponse result;
  result.linear_length = float4(force_x * translational + data.drift_linear.x,
                                force_y * translational + data.drift_linear.y,
                                force_z * translational + data.drift_linear.z, rate);
  result.rotation = float4(angular.x + data.drift_angular.x, angular.y + data.drift_angular.y,
                           angular.z + data.drift_angular.z, load);
  return result;
}

// The rate at which one side of a row closes under a cell's response.
inline float contact_side_rate(float4 linear, float4 angular, ContactResponse response) {
  float rate = linear.x * response.linear_length.x;
  rate += linear.y * response.linear_length.y;
  rate += linear.z * response.linear_length.z;
  rate += angular.x * response.rotation.x;
  rate += angular.y * response.rotation.y;
  rate += angular.z * response.rotation.z;
  return rate - linear.w * response.linear_length.w;
}

// The gap of `row` at `multiplier` under the response the cells hold.
inline float contact_row_gap(uint row, device const ContactRowData* rows,
                             device const ContactResponse* response, float multiplier,
                             ContactSolveData solve) {
  ContactRowData data = rows[row];
  float rate = contact_side_rate(data.first_linear, data.first_angular, response[data.first_slot]);
  if (data.second_slot != 0xffffffffu) {
    rate += contact_side_rate(data.second_linear, data.second_angular, response[data.second_slot]);
  }
  return data.first_angular.w + solve.dt * rate + data.second_angular.w * multiplier;
}

// How far a row is from its complementarity condition: a positive multiplier
// needs a zero gap and a zero multiplier a non-negative one. Infinite when
// either is not finite.
inline float contact_row_residual(float multiplier, float gap) {
  if (!isfinite(gap) || !isfinite(multiplier)) {
    return INFINITY;
  }
  if (multiplier > 0.0f) {
    return fabs(gap);
  }
  return gap < 0.0f ? -gap : 0.0f;
}

// The gap of `row` at the multipliers themselves, stored in `gaps`, and the
// row's terms: its residual alone.
inline float4 contact_row_initial(uint row, device const ContactRowData* rows,
                                  device const ContactResponse* response,
                                  device const float* multipliers, device float* gaps,
                                  ContactSolveData solve) {
  float multiplier = multipliers[row];
  float gap = contact_row_gap(row, rows, response, multiplier, solve);
  gaps[row] = gap;
  return float4(0.0f, 0.0f, 0.0f, contact_row_residual(multiplier, gap));
}

// The projected step of `row` into `next_multipliers` and `next_gaps`, and
// the row's terms.
inline float4 contact_row_step(uint row, device const ContactRowData* rows,
                               device const ContactResponse* response,
                               device const float* multipliers, device const float* gaps,
                               device float* next_multipliers, device float* next_gaps,
                               float length, ContactSolveData solve) {
  float multiplier = contact_projected(multipliers[row], gaps[row], length);
  float gap = contact_row_gap(row, rows, response, multiplier, solve);
  next_multipliers[row] = multiplier;
  next_gaps[row] = gap;
  float step = multiplier - multipliers[row];
  float change = gap - gaps[row];
  return float4(step * step, step * change, change * change, contact_row_residual(multiplier, gap));
}

// Two partial sums of terms: the first three components add and the fourth
// takes the larger.
inline float4 contact_terms_combine(float4 left, float4 right) {
  return float4(left.x + right.x, left.y + right.y, left.z + right.z,
                left.w < right.w ? right.w : left.w);
}

// The sum of `count` terms, 1 <= count <= contact_reduction_width, in a fixed
// order: for stride = width / 2, ..., 1, element i takes
// element i + stride whenever i < stride and i + stride < count. A device
// runs each stride across a threadgroup, skipping the strides no element
// takes; the host runs this.
inline float4 contact_reduction_tree(thread float4* values, uint count) {
  for (uint stride = contact_reduction_width / 2u; stride > 0u; stride /= 2u) {
    for (uint lane = 0u; lane < stride && lane + stride < count; ++lane) {
      values[lane] = contact_terms_combine(values[lane], values[lane + stride]);
    }
  }
  return values[0];
}

// What a solve step's summed terms make of its progress: the iteration it
// counts, the residual it reports, whether the solve is done, and the next
// step length. The initial evaluation only sets the residual and the first
// step length. The two Barzilai-Borwein lengths alternate, and a vanishing
// step . change keeps the current length.
inline void contact_record_step(thread ContactSolveState& state, float4 sums,
                                ContactSolveData solve, uint initial, float first_step_length,
                                uint max_iterations) {
  bool stop = !(sums.w > solve.tolerance) || !isfinite(sums.w);
  if (initial != 0u) {
    state.step_length = first_step_length;
    state.residual = sums.w;
    state.iterations = 0u;
    state.done = stop || max_iterations == 0u ? 1u : 0u;
    return;
  }
  if (state.done != 0u) {
    return;
  }
  state.iterations += 1u;
  state.residual = sums.w;
  if (stop || state.iterations >= max_iterations) {
    state.done = 1u;
    return;
  }
  if (sums.y > 0.0f && sums.z > 0.0f) {
    float next = (state.iterations & 1u) != 0u ? sums.x / sums.y : sums.y / sums.z;
    if (isfinite(next) && next > 0.0f) {
      state.step_length = next;
    }
  }
}

// A bound on the diagonal of row `row`'s system: dt times what a unit
// multiplier does to its own gap, plus its compliance. The largest over the
// rows sets the first step length.
inline float contact_side_bound(float4 linear, float4 angular, ContactBodyData body) {
  float translational = body.axis_mobility.w;
  float transverse = body.rates.x;
  float axial = body.rates.y;
  float3 axis = float3(body.axis_mobility.x, body.axis_mobility.y, body.axis_mobility.z);
  float3 torque = float3(angular.x, angular.y, angular.z);
  float3 turn = ((torque * transverse) + (axis * (axial - transverse) * dot(axis, torque)));
  float value = linear.x * (linear.x * translational);
  value += linear.y * (linear.y * translational);
  value += linear.z * (linear.z * translational);
  value += angular.x * turn.x;
  value += angular.y * turn.y;
  value += angular.z * turn.z;
  return value;
}

inline float contact_row_bound(uint row, device const ContactRowData* rows,
                               device const ContactBodyData* bodies, ContactSolveData solve) {
  ContactRowData data = rows[row];
  float entry = contact_side_bound(data.first_linear, data.first_angular, bodies[data.first_slot]);
  if (data.second_slot != 0xffffffffu) {
    entry += contact_side_bound(data.second_linear, data.second_angular, bodies[data.second_slot]);
  }
  return solve.dt * entry + data.second_angular.w;
}
