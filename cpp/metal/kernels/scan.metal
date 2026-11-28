// Inclusive prefix sums of unsigned counts, in two kernels that the host
// applies level by level.

// The inclusive prefix sum of one threadgroup's block of `values` in place,
// with the block's total in `totals`. SIMD groups scan their lanes, and the
// first SIMD group scans the SIMD-group totals.
kernel void scan_blocks(device uint* values [[buffer(0)]], device uint* totals [[buffer(1)]],
                        constant uint& count [[buffer(2)]], uint index [[thread_position_in_grid]],
                        uint group [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint simd [[simdgroup_index_in_threadgroup]],
                        uint simds [[simdgroups_per_threadgroup]],
                        uint simd_width [[threads_per_simdgroup]],
                        uint width [[threads_per_threadgroup]]) {
  threadgroup uint simd_totals[32];
  uint value = index < count ? values[index] : 0u;
  uint running = simd_prefix_inclusive_sum(value);
  if (lane == simd_width - 1u) {
    simd_totals[simd] = running;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd == 0u) {
    uint total = lane < simds ? simd_totals[lane] : 0u;
    simd_totals[lane] = simd_prefix_exclusive_sum(total);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  running += simd_totals[simd];
  if (index < count) {
    values[index] = running;
  }
  if (index == min(count, (group + 1u) * width) - 1u) {
    totals[group] = running;
  }
}

// Adds the scanned totals of the blocks before each block to its values.
kernel void scan_add(device uint* values [[buffer(0)]], device const uint* totals [[buffer(1)]],
                     constant uint& count [[buffer(2)]], uint index [[thread_position_in_grid]],
                     uint group [[threadgroup_position_in_grid]]) {
  if (index < count && group > 0u) {
    values[index] += totals[group - 1u];
  }
}
