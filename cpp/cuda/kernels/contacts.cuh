#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace microsimulator::cuda {

struct ContactParametersGpu {
  float activation_margin;
  float parallel_sine_threshold;
  float degeneracy_epsilon;
};

void launch_contact_count(const std::uint64_t* ids, const float4* centers, const float4* axes,
                          const float4* geometry, const uint2* candidates, std::uint32_t* counts,
                          ContactParametersGpu parameters, std::uint32_t candidate_count,
                          cudaStream_t stream);

void launch_inclusive_scan_step(const std::uint32_t* input, std::uint32_t* output,
                                std::uint32_t offset, std::uint32_t element_count,
                                cudaStream_t stream);

void launch_contact_fill(const std::uint64_t* ids, const float4* centers, const float4* axes,
                         const float4* geometry, const uint2* candidates,
                         const std::uint32_t* counts, const std::uint32_t* inclusive_counts,
                         std::uint64_t* first_ids, std::uint64_t* second_ids,
                         std::uint32_t* first_slots, std::uint32_t* second_slots,
                         std::uint32_t* ordinals, float4* points_on_first, float4* normals,
                         float* separations, float* weights, ContactParametersGpu parameters,
                         std::uint32_t candidate_count, cudaStream_t stream);

}  // namespace microsimulator::cuda
