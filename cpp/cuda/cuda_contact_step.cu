#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include "core/contact_step_device.hpp"
#include "cuda_contact_step.cuh"
#include "device_types.cuh"
#include "kernels/contact_step.cuh"

namespace microsimulator::cuda {
namespace {

namespace kernels = microsimulator::cuda::device;

using detail::StepBuffer;
using detail::StepKernel;

constexpr unsigned int block_size = 256;
constexpr unsigned int group_threads = 1024;
constexpr unsigned int width = kernels::contact_reduction_width;
// The most rows the host lets the single-block solve take; above that, the
// solve runs a step at a time over the device.
constexpr std::uint32_t group_rows = 4096;
// Steps of the step-at-a-time solve launched before the host reads whether
// it is done.
constexpr std::uint32_t solve_batch = 32;

static_assert(sizeof(kernels::ContactStepParams) ==
              sizeof(microsimulator::device::ContactStepParams));
static_assert(sizeof(kernels::ContactStepBindings) ==
              sizeof(microsimulator::device::ContactStepBindings));

void check_cuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

[[nodiscard]] unsigned int blocks(std::uint32_t count) {
  return (count + block_size - 1) / block_size;
}

#define MS_CUDA_STEP_KERNEL(name, extent)                                                   \
  __global__ void step_##name##_kernel(kernels::ContactStepBindings b,                      \
                                       kernels::ContactStepParams p, std::uint32_t count) { \
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;                       \
    if (index < count) {                                                                    \
      kernels::step_##name(index, b, p);                                                    \
    }                                                                                       \
  }
MS_CONTACT_STEP_KERNELS(MS_CUDA_STEP_KERNEL)
#undef MS_CUDA_STEP_KERNEL

// The scan `p.scan` names in one block; see contact_step.metal.
__global__ void step_scan_kernel(kernels::ContactStepBindings b, kernels::ContactStepParams p) {
  __shared__ unsigned int totals[group_threads];
  if (kernels::step_gated(b)) {
    return;
  }
  const auto scan = kernels::step_scan(b, p);
  const unsigned int lane = threadIdx.x;
  const unsigned int threads = blockDim.x;
  const unsigned int per = (scan.count + threads - 1) / threads;
  const unsigned int begin = min(lane * per, scan.count);
  const unsigned int end = min(begin + per, scan.count);
  unsigned int sum = 0;
  for (unsigned int index = begin; index < end; ++index) {
    sum += scan.input[index];
  }
  totals[lane] = sum;
  __syncthreads();
  for (unsigned int offset = 1; offset < threads; offset *= 2) {
    const unsigned int value = lane >= offset ? totals[lane - offset] : 0;
    __syncthreads();
    totals[lane] += value;
    __syncthreads();
  }
  unsigned int running = lane == 0 ? 0 : totals[lane - 1];
  for (unsigned int index = begin; index < end; ++index) {
    running += scan.input[index];
    scan.output[index] = running;
  }
  __syncthreads();
  if (lane == 0) {
    kernels::step_scan_total(b, p, scan);
  }
}

__device__ void tree_blocks(float4* lanes, unsigned int lane, unsigned int count,
                            unsigned int limit) {
  const unsigned int position = lane % width;
  for (unsigned int stride = width / 2; stride > 0; stride /= 2) {
    if (stride >= limit) {
      continue;
    }
    if (position < stride && position + stride < count) {
      lanes[lane] = kernels::contact_terms_combine(lanes[lane], lanes[lane + stride]);
    }
    __syncthreads();
  }
}

__device__ unsigned int block_count_of(unsigned int block, unsigned int total) {
  const unsigned int first = block * width;
  return first >= total ? 0 : min(width, total - first);
}

// Sums `count` values of `partials` from `base` level by level, each level
// after the last, and returns the sum.
__device__ float4 sum_levels(float4* lanes, float4* partials, unsigned int base,
                             unsigned int count) {
  const unsigned int lane = threadIdx.x;
  const unsigned int per_pass = blockDim.x / width;
  while (count > 1) {
    const unsigned int next = base + count;
    const unsigned int chunks = (count + width - 1) / width;
    for (unsigned int first = 0; first < chunks; first += per_pass) {
      const unsigned int chunk = first + lane / width;
      const unsigned int index = chunk * width + lane % width;
      lanes[lane] = index < count ? partials[base + index] : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
      __syncthreads();
      tree_blocks(lanes, lane, block_count_of(chunk, count), block_count_of(first, count));
      if (lane % width == 0 && chunk < chunks) {
        partials[next + chunk] = lanes[lane];
      }
      __syncthreads();
    }
    base = next;
    count = chunks;
  }
  return partials[base];
}

__device__ float4 group_terms(float4* lanes, const kernels::ContactStepBindings& b,
                              const kernels::ContactSolveData& solve, unsigned int current,
                              float length, bool initial) {
  float* multipliers[2] = {b.multipliers, b.multipliers_next};
  float* gaps[2] = {b.gaps, b.gaps_next};
  const unsigned int next = 1 - current;
  const unsigned int lane = threadIdx.x;
  const unsigned int per_pass = blockDim.x / width;
  const unsigned int block_count = (solve.row_count + width - 1) / width;
  for (unsigned int first = 0; first < block_count; first += per_pass) {
    const unsigned int block = first + lane / width;
    const unsigned int row = block * width + lane % width;
    float4 terms = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    if (row < solve.row_count) {
      terms = initial ? kernels::contact_row_initial(row, b.rows, b.response, multipliers[current],
                                                     gaps[current], solve)
                      : kernels::contact_row_step(row, b.rows, b.response, multipliers[current],
                                                  gaps[current], multipliers[next], gaps[next],
                                                  length, solve);
    }
    lanes[lane] = terms;
    __syncthreads();
    tree_blocks(lanes, lane, block_count_of(block, solve.row_count),
                block_count_of(first, solve.row_count));
    if (lane % width == 0 && block < block_count) {
      b.terms[block] = lanes[lane];
    }
    __syncthreads();
  }
  return sum_levels(lanes, b.terms, 0, block_count);
}

__global__ void __launch_bounds__(group_threads)
    step_solve_group_kernel(kernels::ContactStepBindings b, kernels::ContactStepParams p) {
  __shared__ float4 lanes[group_threads];
  __shared__ kernels::ContactSolveState progress;
  if (kernels::step_gated(b)) {
    return;
  }
  const auto solve = kernels::step_solve_data(b, p);
  const auto max_iterations = kernels::step_solve_iterations(b, p);
  const float first_step_length = kernels::step_word_float(b, MS_STEP_FIRST_STEP);
  float* multipliers[2] = {b.multipliers, b.multipliers_next};
  float* gaps[2] = {b.gaps, b.gaps_next};
  const unsigned int lane = threadIdx.x;
  const unsigned int threads = blockDim.x;
  for (unsigned int body = lane; body < solve.body_count; body += threads) {
    b.response[body] =
        kernels::contact_respond_body(body, b.rows, b.bodies, b.multipliers, b.gaps, 0.0F, 0,
                                      b.incidence_offsets, b.incidence_entries, solve);
  }
  __syncthreads();
  if (solve.row_count == 0) {
    if (lane == 0) {
      b.solve_state[0] = {first_step_length, 0.0F, 0, 1};
    }
    return;
  }
  auto sums = group_terms(lanes, b, solve, 0, 0.0F, true);
  if (lane == 0) {
    kernels::ContactSolveState value{};
    kernels::contact_record_step(value, sums, solve, 1, first_step_length, max_iterations);
    progress = value;
  }
  __syncthreads();
  unsigned int current = 0;
  while (progress.done == 0) {
    const float length = progress.step_length;
    for (unsigned int body = lane; body < solve.body_count; body += threads) {
      b.response[body] =
          kernels::contact_respond_body(body, b.rows, b.bodies, multipliers[current], gaps[current],
                                        length, 1, b.incidence_offsets, b.incidence_entries, solve);
    }
    __syncthreads();
    sums = group_terms(lanes, b, solve, current, length, false);
    if (lane == 0) {
      kernels::ContactSolveState value = progress;
      kernels::contact_record_step(value, sums, solve, 0, 0.0F, max_iterations);
      progress = value;
    }
    __syncthreads();
    current = 1 - current;
  }
  if (lane == 0) {
    b.solve_state[0] = progress;
  }
}

__global__ void step_solve_respond_kernel(kernels::ContactStepBindings b,
                                          kernels::ContactStepParams p) {
  const auto solve = kernels::step_solve_data(b, p);
  const unsigned int body = blockIdx.x * blockDim.x + threadIdx.x;
  if (body >= solve.body_count || kernels::step_gated(b) ||
      (p.project != 0 && b.solve_state[0].done != 0)) {
    return;
  }
  float* multipliers = p.output == 0 ? b.multipliers : b.multipliers_next;
  float* gaps = p.output == 0 ? b.gaps : b.gaps_next;
  b.response[body] = kernels::contact_respond_body(body, b.rows, b.bodies, multipliers, gaps,
                                                   b.solve_state[0].step_length, p.project,
                                                   b.incidence_offsets, b.incidence_entries, solve);
}

__global__ void step_solve_rows_kernel(kernels::ContactStepBindings b,
                                       kernels::ContactStepParams p) {
  __shared__ float4 lanes[width];
  if (kernels::step_gated(b) || (p.project != 0 && b.solve_state[0].done != 0)) {
    return;
  }
  const auto solve = kernels::step_solve_data(b, p);
  float* multipliers[2] = {b.multipliers, b.multipliers_next};
  float* gaps[2] = {b.gaps, b.gaps_next};
  const unsigned int current = p.output;
  const unsigned int next = 1 - current;
  const unsigned int lane = threadIdx.x;
  const unsigned int row = blockIdx.x * width + lane;
  float4 terms = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
  if (row < solve.row_count) {
    terms = p.project == 0
                ? kernels::contact_row_initial(row, b.rows, b.response, multipliers[current],
                                               gaps[current], solve)
                : kernels::contact_row_step(row, b.rows, b.response, multipliers[current],
                                            gaps[current], multipliers[next], gaps[next],
                                            b.solve_state[0].step_length, solve);
  }
  lanes[lane] = terms;
  __syncthreads();
  const unsigned int count = block_count_of(blockIdx.x, solve.row_count);
  tree_blocks(lanes, lane, count, count);
  if (lane == 0) {
    b.terms[blockIdx.x] = lanes[0];
  }
}

__global__ void step_solve_record_kernel(kernels::ContactStepBindings b,
                                         kernels::ContactStepParams p) {
  __shared__ float4 lanes[group_threads];
  if (kernels::step_gated(b) || (p.project != 0 && b.solve_state[0].done != 0)) {
    return;
  }
  const auto solve = kernels::step_solve_data(b, p);
  const unsigned int count = (solve.row_count + width - 1) / width;
  const auto sums = sum_levels(lanes, b.terms, 0, count);
  if (threadIdx.x == 0) {
    kernels::ContactSolveState value = b.solve_state[0];
    kernels::contact_record_step(value, sums, solve, p.project == 0 ? 1 : 0,
                                 kernels::step_word_float(b, MS_STEP_FIRST_STEP),
                                 kernels::step_solve_iterations(b, p));
    b.solve_state[0] = value;
  }
}

class CudaContactStepDevice final : public detail::ContactStepDevice {
 public:
  explicit CudaContactStepDevice(cudaStream_t stream) : stream_(stream) {
  }

  CudaContactStepDevice(const CudaContactStepDevice&) = delete;
  CudaContactStepDevice& operator=(const CudaContactStepDevice&) = delete;
  ~CudaContactStepDevice() override {
    for (auto* buffer : buffers_) {
      if (buffer != nullptr) {
        static_cast<void>(cudaFree(buffer));
      }
    }
  }

  void reserve(StepBuffer buffer, std::size_t bytes, bool preserve) override {
    const auto index = static_cast<std::size_t>(buffer);
    bytes = std::max<std::size_t>(bytes, 16);
    if (sizes_[index] >= bytes) {
      return;
    }
    bytes = std::bit_ceil(bytes);
    void* replacement = nullptr;
    check_cuda(cudaMalloc(&replacement, bytes), "failed to allocate a CUDA contact step buffer");
    check_cuda(cudaMemsetAsync(replacement, 0, bytes, stream_),
               "failed to clear a CUDA contact step buffer");
    if (preserve && buffers_[index] != nullptr) {
      check_cuda(cudaMemcpyAsync(replacement, buffers_[index], sizes_[index],
                                 cudaMemcpyDeviceToDevice, stream_),
                 "failed to copy a CUDA contact step buffer");
    }
    check_cuda(cudaStreamSynchronize(stream_), "CUDA contact step buffer copy failed");
    if (buffers_[index] != nullptr) {
      check_cuda(cudaFree(buffers_[index]), "failed to free a CUDA contact step buffer");
    }
    buffers_[index] = replacement;
    sizes_[index] = bytes;
    bind();
  }

  void upload(StepBuffer buffer, const void* data, std::size_t bytes, std::size_t offset) override {
    const auto index = static_cast<std::size_t>(buffer);
    if (offset + bytes > sizes_[index]) {
      throw std::out_of_range("contact step upload exceeds its buffer");
    }
    check_cuda(cudaMemcpyAsync(static_cast<std::byte*>(buffers_[index]) + offset, data, bytes,
                               cudaMemcpyHostToDevice, stream_),
               "failed to upload a CUDA contact step buffer");
    check_cuda(cudaStreamSynchronize(stream_), "CUDA contact step upload failed");
  }

  void download(StepBuffer buffer, void* data, std::size_t bytes) override {
    const auto index = static_cast<std::size_t>(buffer);
    if (bytes > sizes_[index]) {
      throw std::out_of_range("contact step download exceeds its buffer");
    }
    check_cuda(cudaMemcpyAsync(data, buffers_[index], bytes, cudaMemcpyDeviceToHost, stream_),
               "failed to download a CUDA contact step buffer");
    check_cuda(cudaStreamSynchronize(stream_), "CUDA contact step failed");
  }

  void run(StepKernel kernel, std::uint32_t count,
           const microsimulator::device::ContactStepParams& params) override {
    if (count == 0) {
      return;
    }
    const auto p = as_cuda(params);
    switch (kernel) {
#define MS_CUDA_STEP_CASE(name, extent)                                                   \
  case StepKernel::name:                                                                  \
    step_##name##_kernel<<<blocks(count), block_size, 0, stream_>>>(bindings_, p, count); \
    break;
      MS_CONTACT_STEP_KERNELS(MS_CUDA_STEP_CASE)
#undef MS_CUDA_STEP_CASE
      case StepKernel::count:
        throw std::invalid_argument("contact step kernel is invalid");
    }
    check_cuda(cudaGetLastError(), "failed to launch a CUDA contact step kernel");
  }

  void scan(const microsimulator::device::ContactStepParams& params) override {
    step_scan_kernel<<<1, group_threads, 0, stream_>>>(bindings_, as_cuda(params));
    check_cuda(cudaGetLastError(), "failed to launch the CUDA contact step scan");
  }

  void solve(const microsimulator::device::ContactStepParams& params,
             std::uint32_t rows_hint) override {
    const auto p = as_cuda(params);
    if (rows_hint <= group_rows) {
      step_solve_group_kernel<<<1, group_threads, 0, stream_>>>(bindings_, p);
      check_cuda(cudaGetLastError(), "failed to launch the CUDA contact step solve");
      return;
    }
    std::array<std::uint32_t, MS_STEP_WORDS> words{};
    download(StepBuffer::status, words.data(), sizeof(words));
    if (words[MS_STEP_FAILED] != 0 || words[MS_STEP_OVERFLOW] != 0) {
      return;
    }
    const auto rows = words[MS_STEP_ROWS];
    const auto row_blocks = std::max<std::uint32_t>((rows + width - 1) / width, 1);
    auto step = p;
    const auto launch = [&] {
      step_solve_respond_kernel<<<blocks(std::max<std::uint32_t>(p.cell_count, 1)), block_size, 0,
                                  stream_>>>(bindings_, step);
      step_solve_rows_kernel<<<row_blocks, width, 0, stream_>>>(bindings_, step);
      step_solve_record_kernel<<<1, group_threads, 0, stream_>>>(bindings_, step);
      check_cuda(cudaGetLastError(), "failed to launch the CUDA contact step solve");
    };
    step.output = 0;
    step.project = 0;
    launch();
    std::uint32_t current = 0;
    for (;;) {
      for (std::uint32_t index = 0; index < solve_batch; ++index) {
        step.output = current;
        step.project = 1;
        launch();
        current = 1 - current;
      }
      microsimulator::device::ContactSolveState state{};
      download(StepBuffer::solve_state, &state, sizeof(state));
      if (state.done != 0) {
        return;
      }
    }
  }

 private:
  [[nodiscard]] static kernels::ContactStepParams as_cuda(
      const microsimulator::device::ContactStepParams& params) {
    kernels::ContactStepParams result;
    std::memcpy(&result, &params, sizeof(result));
    return result;
  }

  void bind(){
#define MS_CUDA_STEP_BIND(name, type) \
  bindings_.name =                    \
      static_cast<decltype(bindings_.name)>(buffers_[static_cast<std::size_t>(StepBuffer::name)]);
      MS_CONTACT_STEP_BUFFERS(MS_CUDA_STEP_BIND)
#undef MS_CUDA_STEP_BIND
  }

  cudaStream_t stream_;
  std::array<void*, detail::step_buffer_count> buffers_{};
  std::array<std::size_t, detail::step_buffer_count> sizes_{};
  kernels::ContactStepBindings bindings_{};
};

}  // namespace

std::unique_ptr<detail::ContactStepDevice> make_contact_step_device(cudaStream_t stream) {
  return std::make_unique<CudaContactStepDevice>(stream);
}

}  // namespace microsimulator::cuda
