#include <bit>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cuda/kernels/mechanics.cuh"

namespace {
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Buffer {
  float* data = nullptr;
  explicit Buffer(std::size_t size) {
    check(cudaMalloc(reinterpret_cast<void**>(&data), size * sizeof(float)));
  }
  ~Buffer() { cudaFree(data); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};
}

int main() {
  try {
    for (const std::uint32_t count : {1U, 2U, 3U, 31U, 255U, 256U, 257U, 511U, 512U,
                                      513U, 1023U, 1025U, 27648U, 262145U}) {
      std::vector<float> values(count);
      std::uint32_t random = 42;
      for (auto& value : values) {
        random = 1664525U * random + 1013904223U;
        value = static_cast<float>(static_cast<int>(random % 20001U) - 10000) * 0.001F;
      }
      if (count >= 3) { values[0] = 1e20F; values[1] = -1e20F; values[2] = 1.0F; }
      Buffer a(count), b(count);
      check(cudaMemcpy(a.data, values.data(), count * sizeof(float), cudaMemcpyHostToDevice));
      auto* input = a.data;
      auto* output = b.data;
      auto remaining = count;
      while (remaining > 1) {
        microsimulator::cuda::launch_reduce_sum_blocks(input, output, remaining, nullptr);
        check(cudaGetLastError());
        remaining = microsimulator::cuda::reduction_output_count(remaining);
        std::swap(input, output);
      }
      float actual;
      check(cudaMemcpy(&actual, input, sizeof(float), cudaMemcpyDeviceToHost));
      while (values.size() > 1) {
        std::vector<float> next;
        for (std::size_t i = 0; i < values.size(); i += 2) {
          auto sum = values[i];
          if (i + 1 < values.size()) sum += values[i + 1];
          next.push_back(sum);
        }
        values = std::move(next);
      }
      if (std::bit_cast<std::uint32_t>(actual) != std::bit_cast<std::uint32_t>(values[0])) {
        throw std::runtime_error("reduction changed the reference addition tree at count " +
                                 std::to_string(count));
      }
    }
    std::cout << "CUDA reduction matches reference bits at all tested boundaries\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
