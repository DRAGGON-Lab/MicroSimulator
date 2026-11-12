#include "core/shared_constants.hpp"
#include "microsimulator/species.hpp"
#include "species.cuh"

namespace microsimulator::cuda {
namespace {

using shared::pi;
constexpr std::uint32_t threads_per_block = 256;

__device__ float effective_volume(float length, float radius) {
  return pi * radius * radius * (length + 2.0F * radius);
}

__device__ float effective_surface_area(float length, float radius) {
  return 2.0F * pi * radius * (length + 2.0F * radius);
}

__device__ float evaluate_instruction(const RateInstructionGpu& instruction, const float* workspace,
                                      const float* species, float4 center, float4 geometry,
                                      float growth_rate, std::int32_t cell_type,
                                      float volume_change_rate) {
  switch (static_cast<RateOp>(instruction.operation)) {
    case RateOp::constant:
      return instruction.value;
    case RateOp::species:
      return species[instruction.first];
    case RateOp::position_x:
      return center.x;
    case RateOp::position_y:
      return center.y;
    case RateOp::position_z:
      return center.z;
    case RateOp::cell_length:
      return geometry.x;
    case RateOp::cell_radius:
      return geometry.y;
    case RateOp::growth_rate:
      return growth_rate;
    case RateOp::cell_type:
      return static_cast<float>(cell_type);
    case RateOp::cell_volume_change_rate:
      return volume_change_rate;
    case RateOp::cell_volume:
      return effective_volume(geometry.x, geometry.y);
    case RateOp::cell_surface_area:
      return effective_surface_area(geometry.x, geometry.y);
    case RateOp::add:
      return workspace[instruction.first] + workspace[instruction.second];
    case RateOp::subtract:
      return workspace[instruction.first] - workspace[instruction.second];
    case RateOp::multiply:
      return workspace[instruction.first] * workspace[instruction.second];
    case RateOp::divide:
      return workspace[instruction.first] / workspace[instruction.second];
    case RateOp::power:
      return powf(workspace[instruction.first], workspace[instruction.second]);
    case RateOp::minimum:
      return fminf(workspace[instruction.first], workspace[instruction.second]);
    case RateOp::maximum:
      return fmaxf(workspace[instruction.first], workspace[instruction.second]);
    case RateOp::negate:
      return -workspace[instruction.first];
    case RateOp::exponential:
      return expf(workspace[instruction.first]);
    case RateOp::logarithm:
      return logf(workspace[instruction.first]);
    case RateOp::less:
      return workspace[instruction.first] < workspace[instruction.second] ? 1.0F : 0.0F;
    case RateOp::less_equal:
      return workspace[instruction.first] <= workspace[instruction.second] ? 1.0F : 0.0F;
    case RateOp::greater:
      return workspace[instruction.first] > workspace[instruction.second] ? 1.0F : 0.0F;
    case RateOp::greater_equal:
      return workspace[instruction.first] >= workspace[instruction.second] ? 1.0F : 0.0F;
    case RateOp::equal:
      return workspace[instruction.first] == workspace[instruction.second] ? 1.0F : 0.0F;
    case RateOp::select:
      return workspace[instruction.first] != 0.0F ? workspace[instruction.second]
                                                  : workspace[instruction.third];
    default:
      return nanf("");
  }
}

__global__ void advance_species(float* levels, const float* previous_lengths, const float4* centers,
                                const float4* geometry, const float* growth_rates,
                                const std::int32_t* cell_types,
                                const RateInstructionGpu* instructions,
                                const std::uint32_t* outputs, float* workspace,
                                std::uint32_t* error, float dt, std::uint32_t species_count,
                                std::uint32_t instruction_count, std::uint32_t cell_count) {
  const auto cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= cell_count) {
    return;
  }

  const auto species_offset = cell * species_count;
  const auto radius = geometry[cell].y;
  const auto dilution =
      effective_volume(previous_lengths[cell], radius) / effective_volume(geometry[cell].x, radius);
  for (std::uint32_t species = 0; species < species_count; ++species) {
    levels[species_offset + species] *= dilution;
  }

  const auto workspace_offset = cell * instruction_count;
  auto* cell_workspace = workspace + workspace_offset;
  const auto* cell_species = levels + species_offset;
  for (std::uint32_t index = 0; index < instruction_count; ++index) {
    const auto value =
        evaluate_instruction(instructions[index], cell_workspace, cell_species, centers[cell],
                             geometry[cell], growth_rates[cell], cell_types[cell],
                             dt == 0.0f ? 0.0f
                                        : (effective_volume(geometry[cell].x, radius) -
                                           effective_volume(previous_lengths[cell], radius)) /
                                              dt);
    cell_workspace[index] = value;
    if (!isfinite(value)) {
      atomicOr(error, 1U);
    }
  }

  for (std::uint32_t species = 0; species < species_count; ++species) {
    const auto value = levels[species_offset + species] + dt * cell_workspace[outputs[species]];
    levels[species_offset + species] = value;
    if (!isfinite(value)) {
      atomicOr(error, 1U);
    }
  }
}

}  // namespace

void launch_advance_species(float* levels, const float* previous_lengths, const float4* centers,
                            const float4* geometry, const float* growth_rates,
                            const std::int32_t* cell_types, const RateInstructionGpu* instructions,
                            const std::uint32_t* outputs, float* workspace, std::uint32_t* error,
                            float dt, std::uint32_t species_count, std::uint32_t instruction_count,
                            std::uint32_t cell_count, cudaStream_t stream) {
  const auto blocks = ((cell_count - 1) / threads_per_block) + 1;
  advance_species<<<blocks, threads_per_block, 0, stream>>>(
      levels, previous_lengths, centers, geometry, growth_rates, cell_types, instructions, outputs,
      workspace, error, dt, species_count, instruction_count, cell_count);
}

}  // namespace microsimulator::cuda
