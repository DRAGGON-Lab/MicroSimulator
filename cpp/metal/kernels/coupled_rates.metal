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

uint stencil_component(float4 center, GridShape shape, float4 origin, float4 spacing,
                       device const uchar* obstacles) {
  float cx = axis_coordinate(center.x, origin.x, spacing.x, shape.x);
  float cy = axis_coordinate(center.y, origin.y, spacing.y, shape.y);
  float cz = axis_coordinate(center.z, origin.z, spacing.z, shape.z);
  unsigned lx = (unsigned)floor(cx), ly = (unsigned)floor(cy), lz = (unsigned)floor(cz);
  unsigned fluid = 0, seed = 0;
  float best = 0;
  for (unsigned bit = 0; bit < 8; ++bit) {
    unsigned x = lx + (bit >> 2), y = ly + ((bit >> 1) & 1u), z = lz + (bit & 1u);
    if (x >= shape.x || y >= shape.y || z >= shape.z) continue;
    float w = axis_site_weight(cx, shape.x, x) * axis_site_weight(cy, shape.y, y) *
              axis_site_weight(cz, shape.z, z);
    if (w <= 0 || obstacles[site_index(shape, x, y, z)] != 0) continue;
    fluid |= 1u << bit;
    if (w > best) {
      best = w;
      seed = 1u << bit;
    }
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
  // The raw weight is zero when the cell lies on a lattice plane away from
  // this site, and it is pure arithmetic, so testing it first skips the
  // obstacle and connectivity work for a site the cell does not feed.
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

// The lattice sites a cell's trilinear stencil covers: the flat index of its
// lower corner, and one bit per axis (x: 4, y: 2, z: 1) set when the stencil
// spans a second site along that axis.
struct ScatterStencil {
  uint base;
  uint spans;
};

ScatterStencil scatter_stencil(float4 center, GridShape shape, float4 origin, float4 spacing) {
  uint lower_x = uint(floor(axis_coordinate(center.x, origin.x, spacing.x, shape.x)));
  uint lower_y = uint(floor(axis_coordinate(center.y, origin.y, spacing.y, shape.y)));
  uint lower_z = uint(floor(axis_coordinate(center.z, origin.z, spacing.z, shape.z)));
  uint span_x = shape.x == 1u || lower_x == shape.x - 1u ? 0u : 4u;
  uint span_y = shape.y == 1u || lower_y == shape.y - 1u ? 0u : 2u;
  uint span_z = shape.z == 1u || lower_z == shape.z - 1u ? 0u : 1u;
  return {site_index(shape, lower_x, lower_y, lower_z), span_x | span_y | span_z};
}

bool stencil_covers(uint2 stencil, uint corner) { return (corner & ~stencil.y) == 0u; }

uint stencil_site(uint2 stencil, GridShape shape, uint corner) {
  return stencil.x + (corner >> 2) * shape.y * shape.z + ((corner >> 1) & 1u) * shape.z +
         (corner & 1u);
}

// Cell-to-grid scatter runs as a gather: every site sums the cells whose
// stencil covers it, in ascending cell order. The kernels below build that
// per-site cell list. Stencils are computed once and stored, so counting and
// filling walk identical integer site lists.
kernel void coupled_scatter_stencils(device const float4* centers [[buffer(0)]],
                                     device uint2* stencils [[buffer(1)]],
                                     constant GridShape& shape [[buffer(2)]],
                                     constant float4& origin [[buffer(3)]],
                                     constant float4& spacing [[buffer(4)]],
                                     constant uint& cell_count [[buffer(5)]],
                                     uint cell [[thread_position_in_grid]]) {
  if (cell >= cell_count) {
    return;
  }
  ScatterStencil stencil = scatter_stencil(centers[cell], shape, origin, spacing);
  stencils[cell] = uint2(stencil.base, stencil.spans);
}

kernel void coupled_scatter_count(device const uint2* stencils [[buffer(0)]],
                                  device atomic_uint* counts [[buffer(1)]],
                                  constant GridShape& shape [[buffer(2)]],
                                  constant uint& cell_count [[buffer(3)]],
                                  uint cell [[thread_position_in_grid]]) {
  if (cell >= cell_count) {
    return;
  }
  uint2 stencil = stencils[cell];
  for (uint corner = 0; corner < 8u; ++corner) {
    if (stencil_covers(stencil, corner)) {
      atomic_fetch_add_explicit(&counts[stencil_site(stencil, shape, corner)], 1u,
                                memory_order_relaxed);
    }
  }
}

// `ends` is the inclusive prefix sum of the per-site counts; site s owns
// entries [ends[s - 1], ends[s]). Cursors start at zero.
kernel void coupled_scatter_fill(device const uint2* stencils [[buffer(0)]],
                                 device atomic_uint* cursors [[buffer(1)]],
                                 device const uint* ends [[buffer(2)]],
                                 device uint* cells [[buffer(3)]],
                                 constant GridShape& shape [[buffer(4)]],
                                 constant uint& cell_count [[buffer(5)]],
                                 uint cell [[thread_position_in_grid]]) {
  if (cell >= cell_count) {
    return;
  }
  uint2 stencil = stencils[cell];
  for (uint corner = 0; corner < 8u; ++corner) {
    if (!stencil_covers(stencil, corner)) {
      continue;
    }
    uint site = stencil_site(stencil, shape, corner);
    uint first = site == 0u ? 0u : ends[site - 1u];
    uint offset = atomic_fetch_add_explicit(&cursors[site], 1u, memory_order_relaxed);
    cells[first + offset] = cell;
  }
}

// Restores the max-heap property below `root` in a site's list of `count`
// cell indices.
void scatter_sift_down(device uint* cells, uint root, uint count) {
  uint value = cells[root];
  for (;;) {
    uint child = 2u * root + 1u;
    if (child >= count) {
      break;
    }
    if (child + 1u < count && cells[child + 1u] > cells[child]) {
      ++child;
    }
    if (cells[child] <= value) {
      break;
    }
    cells[root] = cells[child];
    root = child;
  }
  cells[root] = value;
}

// Filling claims slots in arrival order; sorting each site's list restores the
// ascending cell order that makes the per-site sum reproducible. Heapsort
// keeps a site shared by many cells at O(n log n) without scratch storage.
kernel void coupled_scatter_sort(device const uint* ends [[buffer(0)]],
                                 device uint* cells [[buffer(1)]],
                                 constant uint& site_count [[buffer(2)]],
                                 uint site [[thread_position_in_grid]]) {
  if (site >= site_count) {
    return;
  }
  uint first = site == 0u ? 0u : ends[site - 1u];
  uint count = ends[site] - first;
  if (count < 2u) {
    return;
  }
  device uint* list = cells + first;
  for (uint root = count / 2u; root > 0u; --root) {
    scatter_sift_down(list, root - 1u, count);
  }
  for (uint last = count - 1u; last > 0u; --last) {
    uint largest = list[0];
    list[0] = list[last];
    list[last] = largest;
    scatter_sift_down(list, 0u, last);
  }
}

float evaluate_instruction(const RateInstruction instruction, device const float* workspace,
                           device const float* species, device const float* signals, float4 center,
                           float4 geometry, float growth_rate, int cell_type,
                           float volume_change_rate) {
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
    case rate_op_signal:
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
    device const uint* scatter_ends [[buffer(24)]], device const uint* scatter_cells [[buffer(25)]],
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
  if (cell_count != 0u) {
    uint first = site == 0u ? 0u : scatter_ends[site - 1u];
    uint last = scatter_ends[site];
    for (uint entry = first; entry < last; ++entry) {
      uint cell = scatter_cells[entry];
      float weight = cell_scatter_weight(centers[cell], shape, origin, spacing, obstacles, x, y, z);
      source += weight * cell_signal_rates[cell * signal_count + signal] * inverse_voxel_volume;
    }
  }
  float transport_scale = crank_nicolson == 0u ? dt : (crank_nicolson == 1u ? 0.5f * dt : 0.0f);
  float candidate = current + transport_scale * rate + dt * source;
  output[index] = candidate;
  if (!isfinite(candidate) || (crank_nicolson == 0u && candidate < 0.0f)) {
    atomic_fetch_or_explicit(error, 2u, memory_order_relaxed);
  }
}
