#include "growth.cuh"

namespace cm::cuda {
namespace {

__global__ void advance_growth(float* lengths, const float* growth_rates, float dt,
                               std::uint32_t count) {
  const auto index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index >= count) {
    return;
  }

  lengths[index] += growth_rates[index] * lengths[index] * dt;
}

}  // namespace

void launch_growth(float* lengths, const float* growth_rates, float dt, std::uint32_t count,
                   cudaStream_t stream) {
  constexpr std::uint32_t threads_per_block = 256;
  const auto block_count = ((count - 1) / threads_per_block) + 1;
  advance_growth<<<block_count, threads_per_block, 0, stream>>>(lengths, growth_rates, dt, count);
}

}  // namespace cm::cuda
