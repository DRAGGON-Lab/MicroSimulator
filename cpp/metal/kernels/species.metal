float effective_volume(float length, float radius) {
  return pi * radius * radius * (length + 2.0f * radius);
}

float effective_surface_area(float length, float radius) {
  return 2.0f * pi * radius * (length + 2.0f * radius);
}

float evaluate_instruction(const RateInstruction instruction, device const float* workspace,
                           device const float* species, float4 center, float4 geometry,
                           float growth_rate, int cell_type, float volume_change_rate) {
  switch (instruction.operation) {
    case rate_op_constant:
      return instruction.value;
    case rate_op_species:
      return species[instruction.first];
    case rate_op_position_x:
      return center.x;
    case rate_op_position_y:
      return center.y;
    case rate_op_position_z:
      return center.z;
    case rate_op_cell_length:
      return geometry.x;
    case rate_op_cell_radius:
      return geometry.y;
    case rate_op_growth_rate:
      return growth_rate;
    case rate_op_cell_type:
      return float(cell_type);
    case rate_op_cell_volume_change_rate:
      return volume_change_rate;
    case rate_op_cell_volume:
      return effective_volume(geometry.x, geometry.y);
    case rate_op_cell_surface_area:
      return effective_surface_area(geometry.x, geometry.y);
    case rate_op_add:
      return workspace[instruction.first] + workspace[instruction.second];
    case rate_op_subtract:
      return workspace[instruction.first] - workspace[instruction.second];
    case rate_op_multiply:
      return workspace[instruction.first] * workspace[instruction.second];
    case rate_op_divide:
      return workspace[instruction.first] / workspace[instruction.second];
    case rate_op_power:
      return pow(workspace[instruction.first], workspace[instruction.second]);
    case rate_op_minimum:
      return min(workspace[instruction.first], workspace[instruction.second]);
    case rate_op_maximum:
      return max(workspace[instruction.first], workspace[instruction.second]);
    case rate_op_negate:
      return -workspace[instruction.first];
    case rate_op_exponential:
      return exp(workspace[instruction.first]);
    case rate_op_logarithm:
      return log(workspace[instruction.first]);
    case rate_op_less:
      return workspace[instruction.first] < workspace[instruction.second] ? 1.0f : 0.0f;
    case rate_op_less_equal:
      return workspace[instruction.first] <= workspace[instruction.second] ? 1.0f : 0.0f;
    case rate_op_greater:
      return workspace[instruction.first] > workspace[instruction.second] ? 1.0f : 0.0f;
    case rate_op_greater_equal:
      return workspace[instruction.first] >= workspace[instruction.second] ? 1.0f : 0.0f;
    case rate_op_equal:
      return workspace[instruction.first] == workspace[instruction.second] ? 1.0f : 0.0f;
    case rate_op_select:
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
    float value =
        evaluate_instruction(instructions[index], cell_workspace, cell_species, centers[cell],
                             geometry[cell], growth_rates[cell], cell_types[cell],
                             dt == 0.0f ? 0.0f
                                        : (effective_volume(geometry[cell].x, radius) -
                                           effective_volume(previous_lengths[cell], radius)) /
                                              dt);
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
