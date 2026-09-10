#include "species.cuh"

namespace cm::cuda {
namespace {

constexpr float pi = 3.14159265358979323846F;
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
  switch (instruction.operation) {
    case 0:
      return instruction.value;
    case 1:
      return species[instruction.first];
    case 2:
      return center.x;
    case 3:
      return center.y;
    case 4:
      return center.z;
    case 5:
      return geometry.x;
    case 6:
      return geometry.y;
    case 7:
      return growth_rate;
    case 8:
      return static_cast<float>(cell_type);
    case 28:
      return volume_change_rate;
    case 9:
      return effective_volume(geometry.x, geometry.y);
    case 10:
      return effective_surface_area(geometry.x, geometry.y);
    case 11:
      return workspace[instruction.first] + workspace[instruction.second];
    case 12:
      return workspace[instruction.first] - workspace[instruction.second];
    case 13:
      return workspace[instruction.first] * workspace[instruction.second];
    case 14:
      return workspace[instruction.first] / workspace[instruction.second];
    case 15:
      return powf(workspace[instruction.first], workspace[instruction.second]);
    case 16:
      return fminf(workspace[instruction.first], workspace[instruction.second]);
    case 17:
      return fmaxf(workspace[instruction.first], workspace[instruction.second]);
    case 18:
      return -workspace[instruction.first];
    case 19:
      return expf(workspace[instruction.first]);
    case 20:
      return logf(workspace[instruction.first]);
    case 21:
      return workspace[instruction.first] < workspace[instruction.second] ? 1.0F : 0.0F;
    case 22:
      return workspace[instruction.first] <= workspace[instruction.second] ? 1.0F : 0.0F;
    case 23:
      return workspace[instruction.first] > workspace[instruction.second] ? 1.0F : 0.0F;
    case 24:
      return workspace[instruction.first] >= workspace[instruction.second] ? 1.0F : 0.0F;
    case 25:
      return workspace[instruction.first] == workspace[instruction.second] ? 1.0F : 0.0F;
    case 26:
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

}  // namespace cm::cuda
