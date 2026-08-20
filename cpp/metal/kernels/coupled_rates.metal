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

;

float effective_volume(float length, float radius) {
  return pi * radius * radius * (length + 2.0f * radius);
}

float effective_surface_area(float length, float radius) {
  return 2.0f * pi * radius * (length + 2.0f * radius);
}

float axis_coordinate(float position, float origin, float spacing, uint dimension) {
  if (dimension == 1u) {
    return 0.0f;
  }
  return clamp((position - origin) / spacing, 0.0f, float(dimension - 1u));
}

float axis_site_weight(float coordinate, uint dimension, uint site) {
  if (dimension == 1u) {
    return site == 0u ? 1.0f : 0.0f;
  }
  uint lower = uint(floor(coordinate));
  if (lower == dimension - 1u) {
    return site == lower ? 1.0f : 0.0f;
  }
  float fraction = coordinate - float(lower);
  if (site == lower) {
    return 1.0f - fraction;
  }
  if (site == lower + 1u) {
    return fraction;
  }
  return 0.0f;
}

float cell_site_weight(float4 center, GridShape shape, float4 origin, float4 spacing, uint x,
                       uint y, uint z) {
  float coordinate_x = axis_coordinate(center.x, origin.x, spacing.x, shape.x);
  float coordinate_y = axis_coordinate(center.y, origin.y, spacing.y, shape.y);
  float coordinate_z = axis_coordinate(center.z, origin.z, spacing.z, shape.z);
  return axis_site_weight(coordinate_x, shape.x, x) * axis_site_weight(coordinate_y, shape.y, y) *
         axis_site_weight(coordinate_z, shape.z, z);
}

uint stencil_component(float4 center, GridShape shape, float4 origin, float4 spacing, device const uchar* obstacles) {
  float cx = axis_coordinate(center.x, origin.x, spacing.x, shape.x);
  float cy = axis_coordinate(center.y, origin.y, spacing.y, shape.y);
  float cz = axis_coordinate(center.z, origin.z, spacing.z, shape.z);
  unsigned lx = (unsigned)floor(cx), ly = (unsigned)floor(cy), lz = (unsigned)floor(cz);
  unsigned fluid = 0, seed = 0;
  float best = 0;
  for (unsigned bit = 0; bit < 8; ++bit) {
    unsigned x = lx + (bit >> 2), y = ly + ((bit >> 1) & 1u), z = lz + (bit & 1u);
    if (x >= shape.x || y >= shape.y || z >= shape.z) continue;
    float w = axis_site_weight(cx, shape.x, x) * axis_site_weight(cy, shape.y, y) * axis_site_weight(cz, shape.z, z);
    if (w <= 0 || obstacles[site_index(shape, x, y, z)] != 0) continue;
    fluid |= 1u << bit;
    if (w > best) { best = w; seed = 1u << bit; }
  }
  unsigned connected = seed;
  for (unsigned pass = 0; pass < 8; ++pass) {
    for (unsigned bit = 0; bit < 8; ++bit) {
      if ((connected & (1u << bit)) == 0) continue;
      connected |= fluid & ((1u << (bit ^ 1u)) | (1u << (bit ^ 2u)) | (1u << (bit ^ 4u)));
    }
  }
  return connected;
}

float sample_signal(device const float* levels, GridShape shape, float4 origin, float4 spacing,
                    device const uchar* obstacles, float4 center, uint signal) {
  float coordinate_x = axis_coordinate(center.x, origin.x, spacing.x, shape.x);
  float coordinate_y = axis_coordinate(center.y, origin.y, spacing.y, shape.y);
  float coordinate_z = axis_coordinate(center.z, origin.z, spacing.z, shape.z);
  uint lower_x = uint(floor(coordinate_x));
  uint lower_y = uint(floor(coordinate_y));
  uint lower_z = uint(floor(coordinate_z));
  uint count_x = shape.x == 1u || lower_x == shape.x - 1u ? 1u : 2u;
  uint count_y = shape.y == 1u || lower_y == shape.y - 1u ? 1u : 2u;
  uint count_z = shape.z == 1u || lower_z == shape.z - 1u ? 1u : 2u;
  float result = 0.0f;
  const auto component = stencil_component(center, shape, origin, spacing, obstacles);
  float fluid_weight = 0.0f;
  bool dropped = false;
  for (uint dx = 0; dx < count_x; ++dx) {
    uint x = lower_x + dx;
    float wx = axis_site_weight(coordinate_x, shape.x, x);
    for (uint dy = 0; dy < count_y; ++dy) {
      uint y = lower_y + dy;
      float wy = axis_site_weight(coordinate_y, shape.y, y);
      for (uint dz = 0; dz < count_z; ++dz) {
        uint z = lower_z + dz;
        float wz = axis_site_weight(coordinate_z, shape.z, z);
        float weight = wx * wy * wz;
        if ((component & (1u << ((dx << 2) | (dy << 1) | dz))) == 0u) {
          if (weight != 0.0f) {
            dropped = true;
          }
          continue;
        }
        fluid_weight += weight;
        result += weight * grid_level(levels, shape, signal, x, y, z);
      }
    }
  }
  // A stencil with no fluid corner is rejected by the host's coupled-step
  // validation before any kernel runs, so the fluid weight is positive here.
  if (dropped) {
    result /= fluid_weight;
  }
  return result;
}

float cell_scatter_weight(float4 center, GridShape shape, float4 origin, float4 spacing,
                          device const uchar* obstacles, uint x, uint y, uint z) {
  // A cell only scatters into the eight sites of its own stencil, and the
  // weight is pure arithmetic, so testing it first keeps the obstacle mask out
  // of the sites a cell cannot reach - which is nearly all of them.
  float raw = cell_site_weight(center, shape, origin, spacing, x, y, z);
  if (raw == 0.0f) {
    return 0.0f;
  }
  if (obstacles[site_index(shape, x, y, z)] != 0u) {
    return 0.0f;
  }
  float coordinate_x = axis_coordinate(center.x, origin.x, spacing.x, shape.x);
  float coordinate_y = axis_coordinate(center.y, origin.y, spacing.y, shape.y);
  float coordinate_z = axis_coordinate(center.z, origin.z, spacing.z, shape.z);
  uint lower_x = uint(floor(coordinate_x));
  uint lower_y = uint(floor(coordinate_y));
  uint lower_z = uint(floor(coordinate_z));
  uint count_x = shape.x == 1u || lower_x == shape.x - 1u ? 1u : 2u;
  uint count_y = shape.y == 1u || lower_y == shape.y - 1u ? 1u : 2u;
  uint count_z = shape.z == 1u || lower_z == shape.z - 1u ? 1u : 2u;
  const auto component = stencil_component(center, shape, origin, spacing, obstacles);
  float fluid_weight = 0.0f;
  if (raw == 0.0f) return 0.0f;
  unsigned target_bit = ((x - lower_x) << 2) | ((y - lower_y) << 1) | (z - lower_z);
  if ((component & (1u << target_bit)) == 0u) return 0.0f;
  bool dropped = false;
  for (uint dx = 0; dx < count_x; ++dx) {
    uint sx = lower_x + dx;
    float wx = axis_site_weight(coordinate_x, shape.x, sx);
    for (uint dy = 0; dy < count_y; ++dy) {
      uint sy = lower_y + dy;
      float wy = axis_site_weight(coordinate_y, shape.y, sy);
      for (uint dz = 0; dz < count_z; ++dz) {
        uint sz = lower_z + dz;
        float wz = axis_site_weight(coordinate_z, shape.z, sz);
        float weight = wx * wy * wz;
        if ((component & (1u << ((dx << 2) | (dy << 1) | dz))) == 0u) {
          if (weight != 0.0f) {
            dropped = true;
          }
          continue;
        }
        fluid_weight += weight;
      }
    }
  }
  // A stencil with no fluid corner is rejected by the host's coupled-step
  // validation before any kernel runs, so the fluid weight is positive here.
  return dropped ? raw / fluid_weight : raw;
}

float evaluate_instruction(const RateInstruction instruction, device const float* workspace,
                           device const float* species, device const float* signals, float4 center,
                           float4 geometry, float growth_rate, int cell_type, float volume_change_rate) {
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
    case 27:
      return signals[instruction.first];
    default:
      return NAN;
  }
}

kernel void advance_coupled_cells(
    device float* species_levels [[buffer(0)]], device const float* previous_lengths [[buffer(1)]],
    device const float4* centers [[buffer(2)]], device const float4* geometry [[buffer(3)]],
    device const float* growth_rates [[buffer(4)]], device const int* cell_types [[buffer(5)]],
    device const RateInstruction* instructions [[buffer(6)]],
    device const uint* species_outputs [[buffer(7)]],
    device const uint* signal_outputs [[buffer(8)]], device float* workspace [[buffer(9)]],
    device const float* grid_levels [[buffer(10)]], device float* cell_signal_rates [[buffer(11)]],
    device atomic_uint* error [[buffer(12)]], constant GridShape& shape [[buffer(13)]],
    constant float4& origin [[buffer(14)]], constant float4& spacing [[buffer(15)]],
    constant float& dt [[buffer(16)]], constant uint& species_count [[buffer(17)]],
    constant uint& signal_count [[buffer(18)]], constant uint& instruction_count [[buffer(19)]],
    constant uint& cell_count [[buffer(20)]], device const uchar* obstacles [[buffer(21)]],
    uint cell [[thread_position_in_grid]]) {
  if (cell >= cell_count) {
    return;
  }

  uint species_offset = cell * species_count;
  float radius = geometry[cell].y;
  float dilution =
      effective_volume(previous_lengths[cell], radius) / effective_volume(geometry[cell].x, radius);
  for (uint species = 0; species < species_count; ++species) {
    species_levels[species_offset + species] *= dilution;
  }

  uint signal_offset = cell * signal_count;
  device float* cell_signals = cell_signal_rates + signal_offset;
  for (uint signal = 0; signal < signal_count; ++signal) {
    cell_signals[signal] =
        sample_signal(grid_levels, shape, origin, spacing, obstacles, centers[cell], signal);
  }

  uint workspace_offset = cell * instruction_count;
  device float* cell_workspace = workspace + workspace_offset;
  device const float* cell_species = species_levels + species_offset;
  for (uint index = 0; index < instruction_count; ++index) {
    float value =
        evaluate_instruction(instructions[index], cell_workspace, cell_species, cell_signals,
                             centers[cell], geometry[cell], growth_rates[cell], cell_types[cell],
            dt == 0.0f ? 0.0f : (effective_volume(geometry[cell].x, radius) -
                                 effective_volume(previous_lengths[cell], radius)) / dt);
    cell_workspace[index] = value;
    if (!isfinite(value)) {
      atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
    }
  }

  for (uint species = 0; species < species_count; ++species) {
    float value =
        species_levels[species_offset + species] + dt * cell_workspace[species_outputs[species]];
    species_levels[species_offset + species] = value;
    if (!isfinite(value)) {
      atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
    }
  }
  for (uint signal = 0; signal < signal_count; ++signal) {
    cell_signals[signal] = cell_workspace[signal_outputs[signal]];
  }
}

kernel void advance_coupled_grid(
    device const float* levels [[buffer(0)]], device float* output [[buffer(1)]],
    device const float* diffusion [[buffer(2)]], device const float4* advection [[buffer(3)]],
    device const float* fixed_values [[buffer(4)]], device const float4* centers [[buffer(5)]],
    device const float* cell_signal_rates [[buffer(6)]], device atomic_uint* error [[buffer(7)]],
    constant uint* boundary_kinds [[buffer(8)]], constant GridShape& shape [[buffer(9)]],
    constant float4& origin [[buffer(10)]], constant float4& spacing [[buffer(11)]],
    constant float& dt [[buffer(12)]], constant uint& signal_count [[buffer(13)]],
    constant uint& cell_count [[buffer(14)]], constant uint& level_count [[buffer(15)]],
    constant uint& crank_nicolson [[buffer(16)]],
    device const float* reaction_source [[buffer(17)]],
    device const float* reaction_loss [[buffer(18)]], device const uchar* obstacles [[buffer(19)]],
    device const float* x_faces [[buffer(20)]], device const float* y_faces [[buffer(21)]],
    device const float* z_faces [[buffer(22)]], constant uint& has_velocity_field [[buffer(23)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= level_count) {
    return;
  }

  uint signal = index / shape.sites;
  uint site = index - signal * shape.sites;
  uint x = site / (shape.y * shape.z);
  uint yz = site - x * shape.y * shape.z;
  uint y = yz / shape.z;
  uint z = yz - y * shape.z;
  float current = levels[index];
  if (obstacles[site] != 0u) {
    output[index] = current;
    return;
  }

  float3 lower;
  float3 upper;
  lower.x = x == 0u ? exterior_value(boundary_kinds[0], fixed_values, 0u, signal, signal_count,
                                     current, grid_level(levels, shape, signal, shape.x - 1u, y, z))
                    : grid_level(levels, shape, signal, x - 1u, y, z);
  upper.x = x + 1u == shape.x
                ? exterior_value(boundary_kinds[1], fixed_values, 1u, signal, signal_count, current,
                                 grid_level(levels, shape, signal, 0u, y, z))
                : grid_level(levels, shape, signal, x + 1u, y, z);
  lower.y = y == 0u ? exterior_value(boundary_kinds[2], fixed_values, 2u, signal, signal_count,
                                     current, grid_level(levels, shape, signal, x, shape.y - 1u, z))
                    : grid_level(levels, shape, signal, x, y - 1u, z);
  upper.y = y + 1u == shape.y
                ? exterior_value(boundary_kinds[3], fixed_values, 3u, signal, signal_count, current,
                                 grid_level(levels, shape, signal, x, 0u, z))
                : grid_level(levels, shape, signal, x, y + 1u, z);
  lower.z = z == 0u ? exterior_value(boundary_kinds[4], fixed_values, 4u, signal, signal_count,
                                     current, grid_level(levels, shape, signal, x, y, shape.z - 1u))
                    : grid_level(levels, shape, signal, x, y, z - 1u);
  upper.z = z + 1u == shape.z
                ? exterior_value(boundary_kinds[5], fixed_values, 5u, signal, signal_count, current,
                                 grid_level(levels, shape, signal, x, y, 0u))
                : grid_level(levels, shape, signal, x, y, z + 1u);

  uint3 dimensions = uint3(shape.x, shape.y, shape.z);
  GridFaceState faces = grid_face_state(shape, boundary_kinds, obstacles, x_faces, y_faces, z_faces,
                                        has_velocity_field, advection[signal], x, y, z);
  bool3 closed_lower = faces.closed_lower;
  bool3 closed_upper = faces.closed_upper;
  float3 grid_spacing = spacing.xyz;
  float rate = 0.0f;
  for (uint axis = 0; axis < 3u; ++axis) {
    if (dimensions[axis] == 1u) {
      continue;
    }
    if (closed_lower[axis]) {
      lower[axis] = current;
    }
    if (closed_upper[axis]) {
      upper[axis] = current;
    }
    float inverse_spacing = 1.0f / grid_spacing[axis];
    rate += diffusion[signal] * (lower[axis] - 2.0f * current + upper[axis]) * inverse_spacing *
            inverse_spacing;
    float lower_flux =
        faces.lower[axis] >= 0.0f ? faces.lower[axis] * lower[axis] : faces.lower[axis] * current;
    float upper_flux =
        faces.upper[axis] >= 0.0f ? faces.upper[axis] * current : faces.upper[axis] * upper[axis];
    if (closed_lower[axis]) {
      lower_flux = 0.0f;
    }
    if (closed_upper[axis]) {
      upper_flux = 0.0f;
    }
    rate -= (upper_flux - lower_flux) * inverse_spacing;
  }
  rate += reaction_source[index] - reaction_loss[index] * current;

  float source = 0.0f;
  float inverse_voxel_volume = 1.0f / (spacing.x * spacing.y * spacing.z);
  for (uint cell = 0; cell < cell_count; ++cell) {
    float weight = cell_scatter_weight(centers[cell], shape, origin, spacing, obstacles, x, y, z);
    source += weight * cell_signal_rates[cell * signal_count + signal] * inverse_voxel_volume;
  }
  float transport_scale = crank_nicolson == 0u ? dt : 0.5f * dt;
  float candidate = current + transport_scale * rate + dt * source;
  output[index] = candidate;
  if (!isfinite(candidate) || (crank_nicolson == 0u && candidate < 0.0f)) {
    atomic_fetch_or_explicit(error, 2u, memory_order_relaxed);
  }
}
