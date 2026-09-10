#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace cm::cuda {

struct ContactParametersGpu {
  float activation_margin;
  float parallel_sine_threshold;
  float degeneracy_epsilon;
};

struct alignas(16) ExternalConstraintGpu {
  std::uint64_t id;
  std::uint32_t kind;
  std::uint32_t allowed_region;
  float4 geometry;
  float4 parameters;
};

static_assert(sizeof(ExternalConstraintGpu) == 48);

struct ConstraintContactParametersGpu {
  float activation_margin;
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

void launch_external_contact_count(const std::uint64_t* ids, const float4* centers,
                                   const float4* axes, const float4* geometry,
                                   const ExternalConstraintGpu* constraints, std::uint32_t* counts,
                                   ConstraintContactParametersGpu parameters,
                                   std::uint32_t cell_count, std::uint32_t constraint_count,
                                   cudaStream_t stream);

void launch_external_contact_fill(
    const std::uint64_t* ids, const float4* centers, const float4* axes, const float4* geometry,
    const ExternalConstraintGpu* constraints, const std::uint32_t* counts,
    const std::uint32_t* inclusive_counts, std::uint64_t* cell_ids, std::uint64_t* constraint_ids,
    std::uint32_t* cell_slots, std::uint32_t* constraint_kinds, std::uint32_t* locations,
    float4* points_on_cell, float4* normals, float* separations, float* weights,
    ConstraintContactParametersGpu parameters, std::uint32_t cell_count,
    std::uint32_t constraint_count, cudaStream_t stream);

}  // namespace cm::cuda
