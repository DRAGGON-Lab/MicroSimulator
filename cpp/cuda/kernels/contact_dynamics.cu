#include <cstdint>

#include "contact_dynamics.cuh"

namespace microsimulator::cuda {
namespace {

namespace kernels = microsimulator::cuda::device;

constexpr std::uint32_t width = kernels::contact_reduction_width;
constexpr std::uint32_t threads_per_block = 256;
// The partial sums the single-block solve holds, one per block of rows.
constexpr std::uint32_t solve_blocks = 256;

[[nodiscard]] std::uint32_t blocks_for(std::uint32_t count, std::uint32_t threads) {
  return (count + threads - 1) / threads;
}

// Reduces each `width`-wide block of `lanes` into its first element, as
// contact_reduction_tree does for a block of `count` terms. Every thread of
// the block passes the same `limit`, the largest count among the blocks, so
// the strides no block takes are skipped by all of them.
__device__ void tree_blocks(float4* lanes, std::uint32_t lane, std::uint32_t count,
                            std::uint32_t limit) {
  const std::uint32_t position = lane % width;
  for (std::uint32_t stride = width / 2; stride > 0; stride /= 2) {
    if (stride >= limit) {
      continue;
    }
    if (position < stride && position + stride < count) {
      lanes[lane] = kernels::contact_terms_combine(lanes[lane], lanes[lane + stride]);
    }
    __syncthreads();
  }
}

// The terms in block `block` of `total`.
__device__ std::uint32_t block_count_of(std::uint32_t block, std::uint32_t total) {
  const std::uint32_t first = block * width;
  return first >= total ? 0 : (total - first < width ? total - first : width);
}

__global__ void contact_respond(const kernels::ContactRowData* rows,
                                const kernels::ContactBodyData* bodies, const float* multipliers,
                                const float* gaps, const std::uint32_t* incidence_offsets,
                                const std::uint32_t* incidence_entries,
                                kernels::ContactResponse* response,
                                const kernels::ContactSolveState* state,
                                kernels::ContactSolveData solve, std::uint32_t project) {
  const std::uint32_t body = blockIdx.x * blockDim.x + threadIdx.x;
  if (body >= solve.body_count || (project != 0 && state->done != 0)) {
    return;
  }
  response[body] =
      kernels::contact_respond_body(body, rows, bodies, multipliers, gaps, state->step_length,
                                    project, incidence_offsets, incidence_entries, solve);
}

__global__ void contact_step_tree(const kernels::ContactRowData* rows,
                                  const kernels::ContactResponse* response,
                                  const float* multipliers, float* gaps, float* next_multipliers,
                                  float* next_gaps, float4* partials,
                                  const kernels::ContactSolveState* state,
                                  kernels::ContactSolveData solve, std::uint32_t initial) {
  __shared__ float4 lanes[width];
  if (initial == 0 && state->done != 0) {
    return;
  }
  const std::uint32_t lane = threadIdx.x;
  const std::uint32_t row = blockIdx.x * width + lane;
  float4 terms = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
  if (row < solve.row_count) {
    terms = initial != 0
                ? kernels::contact_row_initial(row, rows, response, multipliers, gaps, solve)
                : kernels::contact_row_step(row, rows, response, multipliers, gaps,
                                            next_multipliers, next_gaps, state->step_length, solve);
  }
  lanes[lane] = terms;
  __syncthreads();
  const std::uint32_t members = block_count_of(blockIdx.x, solve.row_count);
  tree_blocks(lanes, lane, members, members);
  if (lane == 0) {
    partials[blockIdx.x] = lanes[0];
  }
}

__global__ void contact_tree(const float4* input, float4* output, std::uint32_t count,
                             const kernels::ContactSolveState* state, std::uint32_t initial) {
  __shared__ float4 lanes[width];
  if (initial == 0 && state->done != 0) {
    return;
  }
  const std::uint32_t lane = threadIdx.x;
  const std::uint32_t index = blockIdx.x * width + lane;
  lanes[lane] = index < count ? input[index] : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
  __syncthreads();
  const std::uint32_t members = block_count_of(blockIdx.x, count);
  tree_blocks(lanes, lane, members, members);
  if (lane == 0) {
    output[blockIdx.x] = lanes[0];
  }
}

__global__ void contact_record(const float4* sums, kernels::ContactSolveState* state,
                               kernels::ContactSolveData solve, std::uint32_t initial,
                               float first_step_length, std::uint32_t max_iterations) {
  kernels::ContactSolveState value = *state;
  kernels::contact_record_step(value, sums[0], solve, initial, first_step_length, max_iterations);
  *state = value;
}

// The terms of every row, summed by blocks to one value.
__device__ float4 solve_group_terms(float4* lanes, float4* blocks,
                                    const kernels::ContactRowData* rows,
                                    const kernels::ContactResponse* response,
                                    float* current_multipliers, float* current_gaps,
                                    float* next_multipliers, float* next_gaps, float length,
                                    bool initial, const kernels::ContactSolveData& solve) {
  const std::uint32_t lane = threadIdx.x;
  const std::uint32_t per_pass = blockDim.x / width;
  const std::uint32_t block_count = (solve.row_count + width - 1) / width;
  for (std::uint32_t first = 0; first < block_count; first += per_pass) {
    const std::uint32_t block = first + lane / width;
    const std::uint32_t row = block * width + lane % width;
    float4 terms = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    if (row < solve.row_count) {
      terms = initial ? kernels::contact_row_initial(row, rows, response, current_multipliers,
                                                     current_gaps, solve)
                      : kernels::contact_row_step(row, rows, response, current_multipliers,
                                                  current_gaps, next_multipliers, next_gaps, length,
                                                  solve);
    }
    lanes[lane] = terms;
    __syncthreads();
    tree_blocks(lanes, lane, block_count_of(block, solve.row_count),
                block_count_of(first, solve.row_count));
    if (lane % width == 0 && block < block_count) {
      blocks[block] = lanes[lane];
    }
    __syncthreads();
  }
  if (block_count == 1) {
    return blocks[0];
  }
  lanes[lane] = lane < block_count ? blocks[lane] : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
  __syncthreads();
  tree_blocks(lanes, lane, lane < width ? block_count : 0, block_count);
  return lanes[0];
}

__global__ void __launch_bounds__(contact_solve_threads)
    contact_solve(const kernels::ContactRowData* rows, const kernels::ContactBodyData* bodies,
                  float* multipliers_a, float* multipliers_b, float* gaps_a, float* gaps_b,
                  const std::uint32_t* incidence_offsets, const std::uint32_t* incidence_entries,
                  kernels::ContactResponse* response, kernels::ContactSolveState* state,
                  kernels::ContactSolveData solve, std::uint32_t max_iterations,
                  float first_step_length) {
  __shared__ float4 lanes[contact_solve_threads];
  __shared__ float4 blocks[solve_blocks];
  __shared__ kernels::ContactSolveState progress;
  float* multipliers[2] = {multipliers_a, multipliers_b};
  float* gaps[2] = {gaps_a, gaps_b};
  const std::uint32_t lane = threadIdx.x;
  const std::uint32_t threads = blockDim.x;

  for (std::uint32_t body = lane; body < solve.body_count; body += threads) {
    response[body] = kernels::contact_respond_body(body, rows, bodies, multipliers_a, gaps_a, 0.0F,
                                                   0, incidence_offsets, incidence_entries, solve);
  }
  __syncthreads();
  auto sums = solve_group_terms(lanes, blocks, rows, response, multipliers_a, gaps_a, multipliers_b,
                                gaps_b, 0.0F, true, solve);
  if (lane == 0) {
    kernels::ContactSolveState value{};
    kernels::contact_record_step(value, sums, solve, 1, first_step_length, max_iterations);
    progress = value;
  }
  __syncthreads();

  std::uint32_t current = 0;
  while (progress.done == 0) {
    const std::uint32_t next = 1 - current;
    const float length = progress.step_length;
    for (std::uint32_t body = lane; body < solve.body_count; body += threads) {
      response[body] =
          kernels::contact_respond_body(body, rows, bodies, multipliers[current], gaps[current],
                                        length, 1, incidence_offsets, incidence_entries, solve);
    }
    __syncthreads();
    sums = solve_group_terms(lanes, blocks, rows, response, multipliers[current], gaps[current],
                             multipliers[next], gaps[next], length, false, solve);
    if (lane == 0) {
      kernels::ContactSolveState value = progress;
      kernels::contact_record_step(value, sums, solve, 0, 0.0F, max_iterations);
      progress = value;
    }
    __syncthreads();
    current = next;
  }
  if (lane == 0) {
    *state = progress;
  }
}

}  // namespace

void launch_contact_respond(const device::ContactRowData* rows,
                            const device::ContactBodyData* bodies, const float* multipliers,
                            const float* gaps, const std::uint32_t* incidence_offsets,
                            const std::uint32_t* incidence_entries,
                            device::ContactResponse* response,
                            const device::ContactSolveState* state, device::ContactSolveData solve,
                            std::uint32_t project, cudaStream_t stream) {
  contact_respond<<<blocks_for(solve.body_count, threads_per_block), threads_per_block, 0,
                    stream>>>(rows, bodies, multipliers, gaps, incidence_offsets, incidence_entries,
                              response, state, solve, project);
}

void launch_contact_terms(const device::ContactRowData* rows,
                          const device::ContactResponse* response, float* current_multipliers,
                          float* current_gaps, float* next_multipliers, float* next_gaps,
                          float4* const partials[2], device::ContactSolveState* state,
                          device::ContactSolveData solve, std::uint32_t initial,
                          float first_step_length, std::uint32_t max_iterations,
                          cudaStream_t stream) {
  auto count = blocks_for(solve.row_count, width);
  contact_step_tree<<<count, width, 0, stream>>>(rows, response, current_multipliers, current_gaps,
                                                 next_multipliers, next_gaps, partials[0], state,
                                                 solve, initial);
  std::uint32_t input = 0;
  while (count > 1) {
    const auto output_count = blocks_for(count, width);
    contact_tree<<<output_count, width, 0, stream>>>(partials[input], partials[1 - input], count,
                                                     state, initial);
    input = 1 - input;
    count = output_count;
  }
  contact_record<<<1, 1, 0, stream>>>(partials[input], state, solve, initial, first_step_length,
                                      max_iterations);
}

void launch_contact_solve(const device::ContactRowData* rows, const device::ContactBodyData* bodies,
                          float* multipliers_a, float* multipliers_b, float* gaps_a, float* gaps_b,
                          const std::uint32_t* incidence_offsets,
                          const std::uint32_t* incidence_entries, device::ContactResponse* response,
                          device::ContactSolveState* state, device::ContactSolveData solve,
                          std::uint32_t max_iterations, float first_step_length,
                          cudaStream_t stream) {
  contact_solve<<<1, contact_solve_threads, 0, stream>>>(
      rows, bodies, multipliers_a, multipliers_b, gaps_a, gaps_b, incidence_offsets,
      incidence_entries, response, state, solve, max_iterations, first_step_length);
}

}  // namespace microsimulator::cuda
