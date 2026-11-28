// Metal entry points of the contact solve in contact_solve.metal, which the
// build concatenates ahead of this file.
//
// A system small enough is solved whole in one threadgroup by contact_solve.
// A larger one runs a step at a time over the device: contact_respond, then
// contact_step_tree, then contact_tree until one partial sum remains, then
// contact_record. Both paths sum the step terms by the tree of
// contact_reduction_tree, so a solve repeats exactly.

// The largest threadgroup the single-group solve runs in.
constant uint contact_solve_group = 1024u;
// The partial sums the single-group solve holds, one per block of rows.
constant uint contact_solve_blocks = 256u;

// Reduces each contact_reduction_width-wide block of `lanes` into its first
// element, as contact_reduction_tree does for a block of `count` terms. Every
// thread of the group passes the same `limit`, the largest count among the
// blocks, so the strides no block takes are skipped by all of them.
void tree_blocks(threadgroup float4* lanes, uint lane, uint count, uint limit) {
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

// The terms in block `block` of `total`.
uint block_count_of(uint block, uint total) {
  uint first = block * contact_reduction_width;
  return first >= total ? 0u : min(contact_reduction_width, total - first);
}

// The response of every cell to the multipliers, or with `project` to those
// one projected step from them; a projected response does nothing once the
// solve is done.
kernel void contact_respond(device const ContactRowData* rows [[buffer(0)]],
                            device const ContactBodyData* bodies [[buffer(1)]],
                            device const float* multipliers [[buffer(2)]],
                            device const float* gaps [[buffer(3)]],
                            device const uint* incidence_offsets [[buffer(4)]],
                            device const uint* incidence_entries [[buffer(5)]],
                            device ContactResponse* response [[buffer(6)]],
                            device const ContactSolveState* state [[buffer(7)]],
                            constant ContactSolveData& solve [[buffer(8)]],
                            constant uint& project [[buffer(9)]],
                            uint body [[thread_position_in_grid]]) {
  if (body >= solve.body_count || (project != 0u && state[0].done != 0u)) {
    return;
  }
  response[body] = contact_respond_body(body, rows, bodies, multipliers, gaps, state[0].step_length,
                                        project, incidence_offsets, incidence_entries, solve);
}

// The terms of one block of contact_reduction_width rows, summed into
// `partials`. With `initial` the rows evaluate the gaps of the multipliers
// themselves; otherwise they take the projected step into the other pair,
// unless the solve is done.
kernel void contact_step_tree(
    device const ContactRowData* rows [[buffer(0)]],
    device const ContactResponse* response [[buffer(1)]],
    device const float* multipliers [[buffer(2)]], device float* gaps [[buffer(3)]],
    device float* next_multipliers [[buffer(4)]], device float* next_gaps [[buffer(5)]],
    device float4* partials [[buffer(6)]], device const ContactSolveState* state [[buffer(7)]],
    constant ContactSolveData& solve [[buffer(8)]], constant uint& initial [[buffer(9)]],
    uint lane [[thread_position_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
  threadgroup float4 lanes[contact_reduction_width];
  if (initial == 0u && state[0].done != 0u) {
    return;
  }
  uint row = group * contact_reduction_width + lane;
  float4 terms = float4(0.0f);
  if (row < solve.row_count) {
    terms = initial != 0u
                ? contact_row_initial(row, rows, response, multipliers, gaps, solve)
                : contact_row_step(row, rows, response, multipliers, gaps, next_multipliers,
                                   next_gaps, state[0].step_length, solve);
  }
  lanes[lane] = terms;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint count = block_count_of(group, solve.row_count);
  tree_blocks(lanes, lane, count, count);
  if (lane == 0u) {
    partials[group] = lanes[0];
  }
}

// One further level of the sum: each block of contact_reduction_width partial
// sums into one.
kernel void contact_tree(device const float4* input [[buffer(0)]],
                         device float4* output [[buffer(1)]], constant uint& count [[buffer(2)]],
                         device const ContactSolveState* state [[buffer(3)]],
                         constant uint& initial [[buffer(4)]],
                         uint lane [[thread_position_in_threadgroup]],
                         uint group [[threadgroup_position_in_grid]]) {
  threadgroup float4 lanes[contact_reduction_width];
  if (initial == 0u && state[0].done != 0u) {
    return;
  }
  uint index = group * contact_reduction_width + lane;
  lanes[lane] = index < count ? input[index] : float4(0.0f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint members = block_count_of(group, count);
  tree_blocks(lanes, lane, members, members);
  if (lane == 0u) {
    output[group] = lanes[0];
  }
}

// Records the step whose summed terms `sums` holds.
kernel void contact_record(device const float4* sums [[buffer(0)]],
                           device ContactSolveState* state [[buffer(1)]],
                           constant ContactSolveData& solve [[buffer(2)]],
                           constant uint& initial [[buffer(3)]],
                           constant float& first_step_length [[buffer(4)]],
                           constant uint& max_iterations [[buffer(5)]],
                           uint index [[thread_position_in_grid]]) {
  if (index != 0u) {
    return;
  }
  ContactSolveState value = state[0];
  contact_record_step(value, sums[0], solve, initial, first_step_length, max_iterations);
  state[0] = value;
}

// The terms of every row, summed by blocks to one value. With `initial` the
// rows evaluate the gaps of the current multipliers; otherwise they step from
// the current pair into the next.
float4 solve_group_terms(threadgroup float4* lanes, threadgroup float4* blocks,
                         device const ContactRowData* rows, device const ContactResponse* response,
                         device float* current_multipliers, device float* current_gaps,
                         device float* next_multipliers, device float* next_gaps, float length,
                         bool initial, constant ContactSolveData& solve, uint lane, uint threads) {
  uint per_pass = threads / contact_reduction_width;
  uint block_count = (solve.row_count + contact_reduction_width - 1u) / contact_reduction_width;
  for (uint first = 0u; first < block_count; first += per_pass) {
    uint block = first + lane / contact_reduction_width;
    uint row = block * contact_reduction_width + lane % contact_reduction_width;
    float4 terms = float4(0.0f);
    if (row < solve.row_count) {
      terms = initial ? contact_row_initial(row, rows, response, current_multipliers, current_gaps,
                                            solve)
                      : contact_row_step(row, rows, response, current_multipliers, current_gaps,
                                         next_multipliers, next_gaps, length, solve);
    }
    lanes[lane] = terms;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tree_blocks(lanes, lane, block_count_of(block, solve.row_count),
                block_count_of(first, solve.row_count));
    if (lane % contact_reduction_width == 0u && block < block_count) {
      blocks[block] = lanes[lane];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  }
  if (block_count == 1u) {
    return blocks[0];
  }
  lanes[lane] = lane < block_count ? blocks[lane] : float4(0.0f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  tree_blocks(lanes, lane, lane < contact_reduction_width ? block_count : 0u, block_count);
  return lanes[0];
}

// The whole solve in one threadgroup, whose width is a multiple of
// contact_reduction_width, for up to contact_solve_blocks blocks of rows. The
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
  threadgroup float4 lanes[contact_solve_group];
  threadgroup float4 blocks[contact_solve_blocks];
  threadgroup ContactSolveState progress;

  device float* multipliers[2] = {multipliers_a, multipliers_b};
  device float* gaps[2] = {gaps_a, gaps_b};

  for (uint body = lane; body < solve.body_count; body += threads) {
    response[body] = contact_respond_body(body, rows, bodies, multipliers_a, gaps_a, 0.0f, 0u,
                                          incidence_offsets, incidence_entries, solve);
  }
  threadgroup_barrier(mem_flags::mem_device);
  float4 sums = solve_group_terms(lanes, blocks, rows, response, multipliers_a, gaps_a,
                                  multipliers_b, gaps_b, 0.0f, true, solve, lane, threads);
  if (lane == 0u) {
    ContactSolveState value;
    value.done = 0u;
    contact_record_step(value, sums, solve, 1u, first_step_length, max_iterations);
    progress = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

  uint current = 0u;
  while (progress.done == 0u) {
    uint next = 1u - current;
    float length = progress.step_length;
    for (uint body = lane; body < solve.body_count; body += threads) {
      response[body] =
          contact_respond_body(body, rows, bodies, multipliers[current], gaps[current], length, 1u,
                               incidence_offsets, incidence_entries, solve);
    }
    threadgroup_barrier(mem_flags::mem_device);
    sums = solve_group_terms(lanes, blocks, rows, response, multipliers[current], gaps[current],
                             multipliers[next], gaps[next], length, false, solve, lane, threads);
    if (lane == 0u) {
      ContactSolveState value = progress;
      contact_record_step(value, sums, solve, 0u, 0.0f, max_iterations);
      progress = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    current = next;
  }
  if (lane == 0u) {
    state[0] = progress;
  }
}
