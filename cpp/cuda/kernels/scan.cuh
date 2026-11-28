#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace microsimulator::cuda {

// One step of an inclusive prefix sum over `element_count` values.
void launch_inclusive_scan_step(const std::uint32_t* input, std::uint32_t* output,
                                std::uint32_t offset, std::uint32_t element_count,
                                cudaStream_t stream);

}  // namespace microsimulator::cuda
