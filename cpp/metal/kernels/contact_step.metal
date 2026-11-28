// Metal entry points of the substep stages in contact_step_kernels.metal,
// which the build concatenates ahead of this file. Each stage takes the
// substep's buffer table in buffer 0 and its parameters in buffer 1.

#define MS_METAL_STEP_KERNEL(name, extent)                                        \
  kernel void step_##name##_kernel(constant ContactStepBindings& b [[buffer(0)]], \
                                   constant ContactStepParams& p [[buffer(1)]],   \
                                   uint index [[thread_position_in_grid]]) {      \
    step_##name(index, b, p);                                                     \
  }
MS_CONTACT_STEP_KERNELS(MS_METAL_STEP_KERNEL)
#undef MS_METAL_STEP_KERNEL

// The scan `p.scan` names, in one threadgroup: each thread sums a contiguous
// run, the run totals are scanned across the group, and each thread writes
// its run. Unsigned sums are exact, so this equals any other order.
kernel void step_scan_kernel(constant ContactStepBindings& b [[buffer(0)]],
                             constant ContactStepParams& p [[buffer(1)]],
                             uint lane [[thread_position_in_threadgroup]],
                             uint threads [[threads_per_threadgroup]]) {
  threadgroup uint totals[1024];
  if (step_gated(b)) {
    return;
  }
  StepScan scan = step_scan(b, p);
  uint per = (scan.count + threads - 1u) / threads;
  uint begin = min(lane * per, scan.count);
  uint end = min(begin + per, scan.count);
  uint sum = 0u;
  for (uint index = begin; index < end; ++index) {
    sum += scan.input[index];
  }
  totals[lane] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint offset = 1u; offset < threads; offset *= 2u) {
    uint value = lane >= offset ? totals[lane - offset] : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    totals[lane] += value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  uint running = lane == 0u ? 0u : totals[lane - 1u];
  for (uint index = begin; index < end; ++index) {
    running += scan.input[index];
    scan.output[index] = running;
  }
  threadgroup_barrier(mem_flags::mem_device);
  if (lane == 0u) {
    step_scan_total(b, p, scan);
  }
}

// Reduces each contact_reduction_width-wide block of `lanes` into its first
// element, as contact_reduction_tree does for a block of `count` terms. Every
// thread of the group passes the same `limit`, the largest count among the
// blocks, so the strides no block takes are skipped by all of them.
void step_tree_blocks(threadgroup float4* lanes, uint lane, uint count, uint limit) {
  uint position = lane % contact_reduction_width;
  for (uint stride = contact_reduction_width / 2u; stride > 0u; stride /= 2u) {
    if (stride >= limit) {
      continue;
    }
    if (position < stride && position + stride < count) {
      lanes[lane] = contact_terms_combine(lanes[lane], lanes[lane + stride]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

uint step_block_count(uint block, uint total) {
  uint first = block * contact_reduction_width;
  return first >= total ? 0u : min(contact_reduction_width, total - first);
}

// Sums `count` values of `partials` from `base` level by level in blocks of
// contact_reduction_width, each level after the last, and returns the sum.
float4 step_sum_levels(threadgroup float4* lanes, device float4* partials, uint base, uint count,
                       uint lane, uint threads) {
  uint per_pass = threads / contact_reduction_width;
  while (count > 1u) {
    uint next = base + count;
    uint chunks = (count + contact_reduction_width - 1u) / contact_reduction_width;
    for (uint first = 0u; first < chunks; first += per_pass) {
      uint chunk = first + lane / contact_reduction_width;
      uint index = chunk * contact_reduction_width + lane % contact_reduction_width;
      lanes[lane] = index < count ? partials[base + index] : float4(0.0f);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      step_tree_blocks(lanes, lane, step_block_count(chunk, count), step_block_count(first, count));
      if (lane % contact_reduction_width == 0u && chunk < chunks) {
        partials[next + chunk] = lanes[lane];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
    base = next;
    count = chunks;
  }
  return partials[base];
}

// The terms of every row of the substep, summed by blocks and then by levels.
float4 step_group_terms(threadgroup float4* lanes, constant ContactStepBindings& b,
                        ContactSolveData solve, uint current, float length, bool initial, uint lane,
                        uint threads) {
  device float* multipliers[2] = {b.multipliers, b.multipliers_next};
  device float* gaps[2] = {b.gaps, b.gaps_next};
  uint next = 1u - current;
  uint per_pass = threads / contact_reduction_width;
  uint block_count = (solve.row_count + contact_reduction_width - 1u) / contact_reduction_width;
  for (uint first = 0u; first < block_count; first += per_pass) {
    uint block = first + lane / contact_reduction_width;
    uint row = block * contact_reduction_width + lane % contact_reduction_width;
    float4 terms = float4(0.0f);
    if (row < solve.row_count) {
      terms = initial
                  ? contact_row_initial(row, b.rows, b.response, multipliers[current],
                                        gaps[current], solve)
                  : contact_row_step(row, b.rows, b.response, multipliers[current], gaps[current],
                                     multipliers[next], gaps[next], length, solve);
    }
    lanes[lane] = terms;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    step_tree_blocks(lanes, lane, step_block_count(block, solve.row_count),
                     step_block_count(first, solve.row_count));
    if (lane % contact_reduction_width == 0u && block < block_count) {
      b.terms[block] = lanes[lane];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  }
  return step_sum_levels(lanes, b.terms, 0u, block_count, lane, threads);
}

// The substep's whole solve in one threadgroup, whose width is a multiple of
// contact_reduction_width, from the multipliers in `multipliers`.
kernel void step_solve_group_kernel(constant ContactStepBindings& b [[buffer(0)]],
                                    constant ContactStepParams& p [[buffer(1)]],
                                    uint lane [[thread_position_in_threadgroup]],
                                    uint threads [[threads_per_threadgroup]]) {
  threadgroup float4 lanes[1024];
  threadgroup ContactSolveState progress;
  if (step_gated(b)) {
    return;
  }
  ContactSolveData solve = step_solve_data(b, p);
  uint max_iterations = step_solve_iterations(b, p);
  float first_step_length = step_word_float(b, MS_STEP_FIRST_STEP);
  device float* multipliers[2] = {b.multipliers, b.multipliers_next};
  device float* gaps[2] = {b.gaps, b.gaps_next};

  for (uint body = lane; body < solve.body_count; body += threads) {
    b.response[body] = contact_respond_body(body, b.rows, b.bodies, b.multipliers, b.gaps, 0.0f, 0u,
                                            b.incidence_offsets, b.incidence_entries, solve);
  }
  threadgroup_barrier(mem_flags::mem_device);
  if (solve.row_count == 0u) {
    if (lane == 0u) {
      ContactSolveState value;
      value.step_length = first_step_length;
      value.residual = 0.0f;
      value.iterations = 0u;
      value.done = 1u;
      b.solve_state[0] = value;
    }
    return;
  }
  float4 sums = step_group_terms(lanes, b, solve, 0u, 0.0f, true, lane, threads);
  if (lane == 0u) {
    ContactSolveState value;
    value.done = 0u;
    contact_record_step(value, sums, solve, 1u, first_step_length, max_iterations);
    progress = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  uint current = 0u;
  while (progress.done == 0u) {
    float length = progress.step_length;
    for (uint body = lane; body < solve.body_count; body += threads) {
      b.response[body] =
          contact_respond_body(body, b.rows, b.bodies, multipliers[current], gaps[current], length,
                               1u, b.incidence_offsets, b.incidence_entries, solve);
    }
    threadgroup_barrier(mem_flags::mem_device);
    sums = step_group_terms(lanes, b, solve, current, length, false, lane, threads);
    if (lane == 0u) {
      ContactSolveState value = progress;
      contact_record_step(value, sums, solve, 0u, 0.0f, max_iterations);
      progress = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    current = 1u - current;
  }
  if (lane == 0u) {
    b.solve_state[0] = progress;
  }
}

// A step of the solve over the whole device: the response of every cell to
// the multipliers in pair `p.output`, or with `p.project` to those one
// projected step from them.
kernel void step_solve_respond_kernel(constant ContactStepBindings& b [[buffer(0)]],
                                      constant ContactStepParams& p [[buffer(1)]],
                                      uint body [[thread_position_in_grid]]) {
  ContactSolveData solve = step_solve_data(b, p);
  if (body >= solve.body_count || step_gated(b) ||
      (p.project != 0u && b.solve_state[0].done != 0u)) {
    return;
  }
  device float* multipliers = p.output == 0u ? b.multipliers : b.multipliers_next;
  device float* gaps = p.output == 0u ? b.gaps : b.gaps_next;
  b.response[body] =
      contact_respond_body(body, b.rows, b.bodies, multipliers, gaps, b.solve_state[0].step_length,
                           p.project, b.incidence_offsets, b.incidence_entries, solve);
}

// The terms of one block of rows of that step, summed into `terms[group]`.
kernel void step_solve_rows_kernel(constant ContactStepBindings& b [[buffer(0)]],
                                   constant ContactStepParams& p [[buffer(1)]],
                                   uint lane [[thread_position_in_threadgroup]],
                                   uint group [[threadgroup_position_in_grid]]) {
  threadgroup float4 lanes[256];
  if (step_gated(b) || (p.project != 0u && b.solve_state[0].done != 0u)) {
    return;
  }
  ContactSolveData solve = step_solve_data(b, p);
  device float* multipliers[2] = {b.multipliers, b.multipliers_next};
  device float* gaps[2] = {b.gaps, b.gaps_next};
  uint current = p.output;
  uint next = 1u - current;
  uint row = group * contact_reduction_width + lane;
  float4 terms = float4(0.0f);
  if (row < solve.row_count) {
    terms = p.project == 0u ? contact_row_initial(row, b.rows, b.response, multipliers[current],
                                                  gaps[current], solve)
                            : contact_row_step(row, b.rows, b.response, multipliers[current],
                                               gaps[current], multipliers[next], gaps[next],
                                               b.solve_state[0].step_length, solve);
  }
  lanes[lane] = terms;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint count = step_block_count(group, solve.row_count);
  step_tree_blocks(lanes, lane, count, count);
  if (lane == 0u) {
    b.terms[group] = lanes[0];
  }
}

// Sums the block sums in one threadgroup and records the step.
kernel void step_solve_record_kernel(constant ContactStepBindings& b [[buffer(0)]],
                                     constant ContactStepParams& p [[buffer(1)]],
                                     uint lane [[thread_position_in_threadgroup]],
                                     uint threads [[threads_per_threadgroup]]) {
  threadgroup float4 lanes[1024];
  if (step_gated(b) || (p.project != 0u && b.solve_state[0].done != 0u)) {
    return;
  }
  ContactSolveData solve = step_solve_data(b, p);
  uint blocks = (solve.row_count + contact_reduction_width - 1u) / contact_reduction_width;
  float4 sums = step_sum_levels(lanes, b.terms, 0u, blocks, lane, threads);
  if (lane == 0u) {
    ContactSolveState value = b.solve_state[0];
    contact_record_step(value, sums, solve, p.project == 0u ? 1u : 0u,
                        step_word_float(b, MS_STEP_FIRST_STEP), step_solve_iterations(b, p));
    b.solve_state[0] = value;
  }
}
