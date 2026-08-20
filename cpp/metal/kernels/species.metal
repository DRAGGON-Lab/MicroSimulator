#include <metal_stdlib>

using namespace metal;

constant float pi = 3.14159265358979323846f;

struct RateInstruction {
  uint operation;
  uint first;
  uint second;
  uint third;
  float value;
};

float effective_volume(float length, float radius) {
  return pi * radius * radius * (length + 2.0f * radius);
}

float effective_surface_area(float length, float radius) {
  return 2.0f * pi * radius * (length + 2.0f * radius);
}

float evaluate_instruction(const RateInstruction instruction,
                           device const float* workspace,
                           device const float* species,
                           float4 center,
                           float4 geometry,
                           float growth_rate,
                           int cell_type, float volume_change_rate) {
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
      return float(cell_type);
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
      return pow(workspace[instruction.first], workspace[instruction.second]);
    case 16:
      return min(workspace[instruction.first], workspace[instruction.second]);
    case 17:
      return max(workspace[instruction.first], workspace[instruction.second]);
    case 18:
      return -workspace[instruction.first];
    case 19:
      return exp(workspace[instruction.first]);
    case 20:
      return log(workspace[instruction.first]);
    case 21:
      return workspace[instruction.first] < workspace[instruction.second] ? 1.0f : 0.0f;
    case 22:
      return workspace[instruction.first] <= workspace[instruction.second] ? 1.0f : 0.0f;
    case 23:
      return workspace[instruction.first] > workspace[instruction.second] ? 1.0f : 0.0f;
    case 24:
      return workspace[instruction.first] >= workspace[instruction.second] ? 1.0f : 0.0f;
    case 25:
      return workspace[instruction.first] == workspace[instruction.second] ? 1.0f : 0.0f;
    case 26:
      return workspace[instruction.first] != 0.0f ? workspace[instruction.second]
                                                  : workspace[instruction.third];
    default:
      return NAN;
  }
}

kernel void advance_species(
    device float* levels [[buffer(0)]], device const float* previous_lengths [[buffer(1)]],
    device const float4* centers [[buffer(2)]], device const float4* geometry [[buffer(3)]],
    device const float* growth_rates [[buffer(4)]], device const int* cell_types [[buffer(5)]],
    device const RateInstruction* instructions [[buffer(6)]],
    device const uint* outputs [[buffer(7)]], device float* workspace [[buffer(8)]],
    device atomic_uint* error [[buffer(9)]], constant float& dt [[buffer(10)]],
    constant uint& species_count [[buffer(11)]], constant uint& instruction_count [[buffer(12)]],
    constant uint& cell_count [[buffer(13)]], uint cell [[thread_position_in_grid]]) {
  if (cell >= cell_count) {
    return;
  }

  uint species_offset = cell * species_count;
  float radius = geometry[cell].y;
  float dilution =
      effective_volume(previous_lengths[cell], radius) / effective_volume(geometry[cell].x, radius);
  for (uint species = 0; species < species_count; ++species) {
    levels[species_offset + species] *= dilution;
  }

  uint workspace_offset = cell * instruction_count;
  device float* cell_workspace = workspace + workspace_offset;
  device const float* cell_species = levels + species_offset;
  for (uint index = 0; index < instruction_count; ++index) {
    float value = evaluate_instruction(instructions[index], cell_workspace, cell_species,
                                       centers[cell], geometry[cell], growth_rates[cell],
                                       cell_types[cell],
            dt == 0.0f ? 0.0f : (effective_volume(geometry[cell].x, radius) -
                                 effective_volume(previous_lengths[cell], radius)) / dt);
    cell_workspace[index] = value;
    if (!isfinite(value)) {
      atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
    }
  }

  for (uint species = 0; species < species_count; ++species) {
    float value = levels[species_offset + species] + dt * cell_workspace[outputs[species]];
    levels[species_offset + species] = value;
    if (!isfinite(value)) {
      atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
    }
  }
}
