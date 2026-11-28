#include <cstdint>

#include "scan.cuh"

namespace microsimulator::cuda {
namespace {

__global__ void inclusive_scan_step(const std::uint32_t* input, std::uint32_t* output,
                                    std::uint32_t offset, std::uint32_t element_count) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }
  auto value = input[index];
  if (index >= offset) {
    value += input[index - offset];
  }
  output[index] = value;
}

}  // namespace

void launch_inclusive_scan_step(const std::uint32_t* input, std::uint32_t* output,
                                std::uint32_t offset, std::uint32_t element_count,
                                cudaStream_t stream) {
  constexpr std::uint32_t threads = 256;
  const auto blocks = ((element_count - 1) / threads) + 1;
  inclusive_scan_step<<<blocks, threads, 0, stream>>>(input, output, offset, element_count);
}

}  // namespace microsimulator::cuda
