#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace microsimulator::cuda {

struct RateInstructionGpu {
  std::uint32_t operation;
  std::uint32_t first;
  std::uint32_t second;
  std::uint32_t third;
  float value;
};

static_assert(sizeof(RateInstructionGpu) == 20);

void launch_advance_species(float* levels, const float* previous_lengths, const float4* centers,
                            const float4* geometry, const float* growth_rates,
                            const std::int32_t* cell_types, const RateInstructionGpu* instructions,
                            const std::uint32_t* outputs, float* workspace, std::uint32_t* error,
                            float dt, std::uint32_t species_count, std::uint32_t instruction_count,
                            std::uint32_t cell_count, cudaStream_t stream);

}  // namespace microsimulator::cuda
