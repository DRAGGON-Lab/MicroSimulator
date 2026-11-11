#include <cmath>

#include "contacts.cuh"
#include "coupled_rates.cuh"

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

__device__ float axis_coordinate(float position, float origin, float spacing,
                                 std::uint32_t dimension) {
  if (dimension == 1) {
    return 0.0F;
  }
  return fminf(fmaxf((position - origin) / spacing, 0.0F), static_cast<float>(dimension - 1));
}

__device__ float axis_site_weight(float coordinate, std::uint32_t dimension, std::uint32_t site) {
  if (dimension == 1) {
    return site == 0 ? 1.0F : 0.0F;
  }
  const auto lower = static_cast<std::uint32_t>(floorf(coordinate));
  if (lower == dimension - 1) {
    return site == lower ? 1.0F : 0.0F;
  }
  const auto fraction = coordinate - static_cast<float>(lower);
  if (site == lower) {
    return 1.0F - fraction;
  }
  if (site == lower + 1) {
    return fraction;
  }
  return 0.0F;
}

__device__ float cell_site_weight(float4 center, SignalGridShapeGpu shape, float4 origin,
                                  float4 spacing, std::uint32_t x, std::uint32_t y,
                                  std::uint32_t z) {
  const auto coordinate_x = axis_coordinate(center.x, origin.x, spacing.x, shape.x);
  const auto coordinate_y = axis_coordinate(center.y, origin.y, spacing.y, shape.y);
  const auto coordinate_z = axis_coordinate(center.z, origin.z, spacing.z, shape.z);
  return axis_site_weight(coordinate_x, shape.x, x) * axis_site_weight(coordinate_y, shape.y, y) *
         axis_site_weight(coordinate_z, shape.z, z);
}

__device__ unsigned stencil_component(float4 center, SignalGridShapeGpu shape, float4 origin,
                                      float4 spacing, const std::uint8_t* obstacles) {
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

__device__ float sample_signal(const float* levels, SignalGridShapeGpu shape, float4 origin,
                               float4 spacing, const std::uint8_t* obstacles, float4 center,
                               std::uint32_t signal) {
  const auto coordinate_x = axis_coordinate(center.x, origin.x, spacing.x, shape.x);
  const auto coordinate_y = axis_coordinate(center.y, origin.y, spacing.y, shape.y);
  const auto coordinate_z = axis_coordinate(center.z, origin.z, spacing.z, shape.z);
  const auto lower_x = static_cast<std::uint32_t>(floorf(coordinate_x));
  const auto lower_y = static_cast<std::uint32_t>(floorf(coordinate_y));
  const auto lower_z = static_cast<std::uint32_t>(floorf(coordinate_z));
  const auto count_x = shape.x == 1 || lower_x == shape.x - 1 ? 1U : 2U;
  const auto count_y = shape.y == 1 || lower_y == shape.y - 1 ? 1U : 2U;
  const auto count_z = shape.z == 1 || lower_z == shape.z - 1 ? 1U : 2U;
  float result = 0.0F;
  const auto component = stencil_component(center, shape, origin, spacing, obstacles);
  float fluid_weight = 0.0F;
  bool dropped = false;
  for (std::uint32_t dx = 0; dx < count_x; ++dx) {
    const auto x = lower_x + dx;
    const auto weight_x = axis_site_weight(coordinate_x, shape.x, x);
    for (std::uint32_t dy = 0; dy < count_y; ++dy) {
      const auto y = lower_y + dy;
      const auto weight_y = axis_site_weight(coordinate_y, shape.y, y);
      for (std::uint32_t dz = 0; dz < count_z; ++dz) {
        const auto z = lower_z + dz;
        const auto weight_z = axis_site_weight(coordinate_z, shape.z, z);
        const auto weight = weight_x * weight_y * weight_z;
        if ((component & (1u << ((dx << 2) | (dy << 1) | dz))) == 0u) {
          if (weight != 0.0F) {
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

__device__ float cell_scatter_weight(float4 center, SignalGridShapeGpu shape, float4 origin,
                                     float4 spacing, const std::uint8_t* obstacles, std::uint32_t x,
                                     std::uint32_t y, std::uint32_t z) {
  // The raw weight is zero when the cell lies on a lattice plane away from
  // this site, and it is pure arithmetic, so testing it first skips the
  // obstacle and connectivity work for a site the cell does not feed.
  const auto raw = cell_site_weight(center, shape, origin, spacing, x, y, z);
  if (raw == 0.0F) {
    return 0.0F;
  }
  if (obstacles[site_index(shape, x, y, z)] != 0) {
    return 0.0F;
  }
  const auto coordinate_x = axis_coordinate(center.x, origin.x, spacing.x, shape.x);
  const auto coordinate_y = axis_coordinate(center.y, origin.y, spacing.y, shape.y);
  const auto coordinate_z = axis_coordinate(center.z, origin.z, spacing.z, shape.z);
  const auto lower_x = static_cast<std::uint32_t>(floorf(coordinate_x));
  const auto lower_y = static_cast<std::uint32_t>(floorf(coordinate_y));
  const auto lower_z = static_cast<std::uint32_t>(floorf(coordinate_z));
  const auto count_x = shape.x == 1 || lower_x == shape.x - 1 ? 1U : 2U;
  const auto count_y = shape.y == 1 || lower_y == shape.y - 1 ? 1U : 2U;
  const auto count_z = shape.z == 1 || lower_z == shape.z - 1 ? 1U : 2U;
  const auto component = stencil_component(center, shape, origin, spacing, obstacles);
  float fluid_weight = 0.0F;
  if (raw == 0.0f) return 0.0f;
  unsigned target_bit = ((x - lower_x) << 2) | ((y - lower_y) << 1) | (z - lower_z);
  if ((component & (1u << target_bit)) == 0u) return 0.0f;
  bool dropped = false;
  for (std::uint32_t dx = 0; dx < count_x; ++dx) {
    const auto sx = lower_x + dx;
    const auto weight_x = axis_site_weight(coordinate_x, shape.x, sx);
    for (std::uint32_t dy = 0; dy < count_y; ++dy) {
      const auto sy = lower_y + dy;
      const auto weight_y = axis_site_weight(coordinate_y, shape.y, sy);
      for (std::uint32_t dz = 0; dz < count_z; ++dz) {
        const auto sz = lower_z + dz;
        const auto weight_z = axis_site_weight(coordinate_z, shape.z, sz);
        const auto weight = weight_x * weight_y * weight_z;
        if ((component & (1u << ((dx << 2) | (dy << 1) | dz))) == 0u) {
          if (weight != 0.0F) {
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

// The lattice sites a cell's trilinear stencil covers: `x` is the flat index of
// its lower corner, and `y` holds one bit per axis (x: 4, y: 2, z: 1) set when
// the stencil spans a second site along that axis.
__device__ uint2 scatter_stencil(float4 center, SignalGridShapeGpu shape, float4 origin,
                                 float4 spacing) {
  const auto lower_x =
      static_cast<std::uint32_t>(floorf(axis_coordinate(center.x, origin.x, spacing.x, shape.x)));
  const auto lower_y =
      static_cast<std::uint32_t>(floorf(axis_coordinate(center.y, origin.y, spacing.y, shape.y)));
  const auto lower_z =
      static_cast<std::uint32_t>(floorf(axis_coordinate(center.z, origin.z, spacing.z, shape.z)));
  const auto span_x = shape.x == 1 || lower_x == shape.x - 1 ? 0U : 4U;
  const auto span_y = shape.y == 1 || lower_y == shape.y - 1 ? 0U : 2U;
  const auto span_z = shape.z == 1 || lower_z == shape.z - 1 ? 0U : 1U;
  return make_uint2(site_index(shape, lower_x, lower_y, lower_z), span_x | span_y | span_z);
}

__device__ bool stencil_covers(uint2 stencil, std::uint32_t corner) {
  return (corner & ~stencil.y) == 0U;
}

__device__ std::uint32_t stencil_site(uint2 stencil, SignalGridShapeGpu shape,
                                      std::uint32_t corner) {
  return stencil.x + (corner >> 2) * shape.y * shape.z + ((corner >> 1) & 1U) * shape.z +
         (corner & 1U);
}

// Cell-to-grid scatter runs as a gather: every site sums the cells whose
// stencil covers it, in ascending cell order. The kernels below build that
// per-site cell list. Stencils are computed once and stored, so counting and
// filling walk identical integer site lists.
__global__ void coupled_scatter_stencils(const float4* centers, uint2* stencils,
                                         SignalGridShapeGpu shape, float4 origin, float4 spacing,
                                         std::uint32_t cell_count) {
  const auto cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= cell_count) {
    return;
  }
  stencils[cell] = scatter_stencil(centers[cell], shape, origin, spacing);
}

__global__ void coupled_scatter_count(const uint2* stencils, std::uint32_t* counts,
                                      SignalGridShapeGpu shape, std::uint32_t cell_count) {
  const auto cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= cell_count) {
    return;
  }
  const auto stencil = stencils[cell];
  for (std::uint32_t corner = 0; corner < 8; ++corner) {
    if (stencil_covers(stencil, corner)) {
      atomicAdd(&counts[stencil_site(stencil, shape, corner)], 1U);
    }
  }
}

// `ends` is the inclusive prefix sum of the per-site counts; site s owns
// entries [ends[s - 1], ends[s]). Cursors start at zero.
__global__ void coupled_scatter_fill(const uint2* stencils, std::uint32_t* cursors,
                                     const std::uint32_t* ends, std::uint32_t* cells,
                                     SignalGridShapeGpu shape, std::uint32_t cell_count) {
  const auto cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= cell_count) {
    return;
  }
  const auto stencil = stencils[cell];
  for (std::uint32_t corner = 0; corner < 8; ++corner) {
    if (!stencil_covers(stencil, corner)) {
      continue;
    }
    const auto site = stencil_site(stencil, shape, corner);
    const auto first = site == 0 ? 0U : ends[site - 1];
    const auto offset = atomicAdd(&cursors[site], 1U);
    cells[first + offset] = cell;
  }
}

// Restores the max-heap property below `root` in a site's list of `count`
// cell indices.
__device__ void scatter_sift_down(std::uint32_t* cells, std::uint32_t root, std::uint32_t count) {
  const auto value = cells[root];
  for (;;) {
    auto child = 2U * root + 1U;
    if (child >= count) {
      break;
    }
    if (child + 1U < count && cells[child + 1U] > cells[child]) {
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
__global__ void coupled_scatter_sort(const std::uint32_t* ends, std::uint32_t* cells,
                                     std::uint32_t site_count) {
  const auto site = blockIdx.x * blockDim.x + threadIdx.x;
  if (site >= site_count) {
    return;
  }
  const auto first = site == 0 ? 0U : ends[site - 1];
  const auto count = ends[site] - first;
  if (count < 2U) {
    return;
  }
  auto* list = cells + first;
  for (auto root = count / 2U; root > 0U; --root) {
    scatter_sift_down(list, root - 1U, count);
  }
  for (auto last = count - 1U; last > 0U; --last) {
    const auto largest = list[0];
    list[0] = list[last];
    list[last] = largest;
    scatter_sift_down(list, 0U, last);
  }
}

__device__ float evaluate_instruction(const RateInstructionGpu& instruction, const float* workspace,
                                      const float* species, const float* signals, float4 center,
                                      float4 geometry, float growth_rate, std::int32_t cell_type,
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
    case 27:
      return signals[instruction.first];
    default:
      return nanf("");
  }
}

__global__ void advance_coupled_cells(
    float* species_levels, const float* previous_lengths, const float4* centers,
    const float4* geometry, const float* growth_rates, const std::int32_t* cell_types,
    const RateInstructionGpu* instructions, const std::uint32_t* species_outputs,
    const std::uint32_t* signal_outputs, float* workspace, const float* grid_levels,
    float* cell_signal_rates, const std::uint8_t* obstacles, std::uint32_t* error,
    SignalGridShapeGpu shape, float4 origin, float4 spacing, float dt, std::uint32_t species_count,
    std::uint32_t signal_count, std::uint32_t instruction_count, std::uint32_t cell_count) {
  const auto cell = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (cell >= cell_count) {
    return;
  }

  const auto species_offset = cell * species_count;
  const auto radius = geometry[cell].y;
  const auto dilution =
      effective_volume(previous_lengths[cell], radius) / effective_volume(geometry[cell].x, radius);
  for (std::uint32_t species = 0; species < species_count; ++species) {
    species_levels[species_offset + species] *= dilution;
  }

  const auto signal_offset = cell * signal_count;
  auto* cell_signals = cell_signal_rates + signal_offset;
  for (std::uint32_t signal = 0; signal < signal_count; ++signal) {
    cell_signals[signal] =
        sample_signal(grid_levels, shape, origin, spacing, obstacles, centers[cell], signal);
  }

  const auto workspace_offset = cell * instruction_count;
  auto* cell_workspace = workspace + workspace_offset;
  const auto* cell_species = species_levels + species_offset;
  for (std::uint32_t index = 0; index < instruction_count; ++index) {
    const auto value =
        evaluate_instruction(instructions[index], cell_workspace, cell_species, cell_signals,
                             centers[cell], geometry[cell], growth_rates[cell], cell_types[cell],
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
    const auto value =
        species_levels[species_offset + species] + dt * cell_workspace[species_outputs[species]];
    species_levels[species_offset + species] = value;
    if (!isfinite(value)) {
      atomicOr(error, 1U);
    }
  }
  for (std::uint32_t signal = 0; signal < signal_count; ++signal) {
    cell_signals[signal] = cell_workspace[signal_outputs[signal]];
  }
}

__global__ void advance_coupled_grid(
    const float* levels, float* output, const float* diffusion, const float4* advection,
    const float* fixed_values, const float* reaction_source, const float* reaction_loss,
    const float4* centers, const float* cell_signal_rates, const std::uint8_t* obstacles,
    const float* x_faces, const float* y_faces, const float* z_faces,
    std::uint32_t has_velocity_field, std::uint32_t* error, SignalGridBoundariesGpu boundaries,
    SignalGridShapeGpu shape, float4 origin, float4 spacing, float dt, std::uint32_t signal_count,
    std::uint32_t cell_count, std::uint32_t level_count, std::uint32_t crank_nicolson,
    const std::uint32_t* scatter_ends, const std::uint32_t* scatter_cells) {
  const auto index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= level_count) {
    return;
  }

  const auto signal = index / shape.sites;
  const auto site = index - signal * shape.sites;
  const auto x = site / (shape.y * shape.z);
  const auto yz = site - x * shape.y * shape.z;
  const auto y = yz / shape.z;
  const auto z = yz - y * shape.z;
  const auto current = levels[index];
  if (obstacles[site] != 0) {
    output[index] = current;
    return;
  }

  float lower[3];
  float upper[3];
  lower[0] = x == 0 ? exterior_value(boundaries.x_lower, fixed_values, 0, signal, signal_count,
                                     current, grid_level(levels, shape, signal, shape.x - 1, y, z))
                    : grid_level(levels, shape, signal, x - 1, y, z);
  upper[0] = x + 1 == shape.x
                 ? exterior_value(boundaries.x_upper, fixed_values, 1, signal, signal_count,
                                  current, grid_level(levels, shape, signal, 0, y, z))
                 : grid_level(levels, shape, signal, x + 1, y, z);
  lower[1] = y == 0 ? exterior_value(boundaries.y_lower, fixed_values, 2, signal, signal_count,
                                     current, grid_level(levels, shape, signal, x, shape.y - 1, z))
                    : grid_level(levels, shape, signal, x, y - 1, z);
  upper[1] = y + 1 == shape.y
                 ? exterior_value(boundaries.y_upper, fixed_values, 3, signal, signal_count,
                                  current, grid_level(levels, shape, signal, x, 0, z))
                 : grid_level(levels, shape, signal, x, y + 1, z);
  lower[2] = z == 0 ? exterior_value(boundaries.z_lower, fixed_values, 4, signal, signal_count,
                                     current, grid_level(levels, shape, signal, x, y, shape.z - 1))
                    : grid_level(levels, shape, signal, x, y, z - 1);
  upper[2] = z + 1 == shape.z
                 ? exterior_value(boundaries.z_upper, fixed_values, 5, signal, signal_count,
                                  current, grid_level(levels, shape, signal, x, y, 0))
                 : grid_level(levels, shape, signal, x, y, z + 1);

  const std::uint32_t dimensions[3]{shape.x, shape.y, shape.z};
  const auto faces = grid_face_state(shape, boundaries, obstacles, x_faces, y_faces, z_faces,
                                     has_velocity_field, advection[signal], x, y, z);
  const bool* closed_lower = faces.closed_lower;
  const bool* closed_upper = faces.closed_upper;
  const float* face_lower = faces.lower;
  const float* face_upper = faces.upper;
  const float grid_spacing[3]{spacing.x, spacing.y, spacing.z};
  float rate = 0.0F;
  for (std::uint32_t axis = 0; axis < 3; ++axis) {
    if (dimensions[axis] == 1) {
      continue;
    }
    if (closed_lower[axis]) {
      lower[axis] = current;
    }
    if (closed_upper[axis]) {
      upper[axis] = current;
    }
    const auto inverse_spacing = 1.0F / grid_spacing[axis];
    rate += diffusion[signal] * (lower[axis] - 2.0F * current + upper[axis]) * inverse_spacing *
            inverse_spacing;
    auto lower_flux =
        face_lower[axis] >= 0.0F ? face_lower[axis] * lower[axis] : face_lower[axis] * current;
    auto upper_flux =
        face_upper[axis] >= 0.0F ? face_upper[axis] * current : face_upper[axis] * upper[axis];
    if (closed_lower[axis]) {
      lower_flux = 0.0F;
    }
    if (closed_upper[axis]) {
      upper_flux = 0.0F;
    }
    rate -= (upper_flux - lower_flux) * inverse_spacing;
  }
  rate += reaction_source[index] - reaction_loss[index] * current;

  float source = 0.0F;
  const auto inverse_voxel_volume = 1.0F / (spacing.x * spacing.y * spacing.z);
  if (cell_count != 0) {
    const auto first = site == 0 ? 0U : scatter_ends[site - 1];
    const auto last = scatter_ends[site];
    for (auto entry = first; entry < last; ++entry) {
      const auto cell = scatter_cells[entry];
      const auto weight =
          cell_scatter_weight(centers[cell], shape, origin, spacing, obstacles, x, y, z);
      source += weight * cell_signal_rates[cell * signal_count + signal] * inverse_voxel_volume;
    }
  }
  const auto transport_scale = crank_nicolson == 0 ? dt : (crank_nicolson == 1 ? 0.5F * dt : 0.0F);
  const auto candidate = current + transport_scale * rate + dt * source;
  output[index] = candidate;
  if (!isfinite(candidate) || (!crank_nicolson && candidate < 0.0F)) {
    atomicOr(error, 2U);
  }
}

}  // namespace

cudaError_t launch_coupled_scatter_lists(const float4* centers, uint2* stencils,
                                         std::uint32_t* counts, std::uint32_t* cursors,
                                         std::uint32_t* scan_a, std::uint32_t* scan_b,
                                         std::uint32_t* cells, SignalGridShapeGpu shape,
                                         float4 origin, float4 spacing, std::uint32_t cell_count,
                                         cudaStream_t stream, const std::uint32_t** ends) {
  const auto site_bytes = static_cast<std::size_t>(shape.sites) * sizeof(std::uint32_t);
  auto status = cudaMemsetAsync(counts, 0, site_bytes, stream);
  if (status != cudaSuccess) {
    return status;
  }
  status = cudaMemsetAsync(cursors, 0, site_bytes, stream);
  if (status != cudaSuccess) {
    return status;
  }

  const auto cell_blocks = ((cell_count - 1) / threads_per_block) + 1;
  const auto site_blocks = ((shape.sites - 1) / threads_per_block) + 1;
  coupled_scatter_stencils<<<cell_blocks, threads_per_block, 0, stream>>>(
      centers, stencils, shape, origin, spacing, cell_count);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  coupled_scatter_count<<<cell_blocks, threads_per_block, 0, stream>>>(stencils, counts, shape,
                                                                       cell_count);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }

  const std::uint32_t* scan_input = counts;
  auto* scan_output = scan_a;
  std::uint32_t offset = 1;
  while (offset < shape.sites) {
    launch_inclusive_scan_step(scan_input, scan_output, offset, shape.sites, stream);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
      return status;
    }
    scan_input = scan_output;
    scan_output = scan_output == scan_a ? scan_b : scan_a;
    if (offset > shape.sites / 2) {
      break;
    }
    offset *= 2;
  }
  *ends = scan_input;

  coupled_scatter_fill<<<cell_blocks, threads_per_block, 0, stream>>>(stencils, cursors, *ends,
                                                                      cells, shape, cell_count);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  coupled_scatter_sort<<<site_blocks, threads_per_block, 0, stream>>>(*ends, cells, shape.sites);
  return cudaGetLastError();
}

cudaError_t launch_advance_coupled(
    float* species_levels, const float* previous_lengths, const float4* centers,
    const float4* geometry, const float* growth_rates, const std::int32_t* cell_types,
    const RateInstructionGpu* instructions, const std::uint32_t* species_outputs,
    const std::uint32_t* signal_outputs, float* workspace, const float* grid_levels,
    float* grid_output, const float* diffusion, const float4* advection, const float* fixed_values,
    const float* reaction_source, const float* reaction_loss, const std::uint8_t* obstacles,
    const float* x_faces, const float* y_faces, const float* z_faces,
    std::uint32_t has_velocity_field, float* cell_signal_rates, std::uint32_t* error,
    SignalGridBoundariesGpu boundaries, SignalGridShapeGpu shape, float4 origin, float4 spacing,
    float dt, std::uint32_t species_count, std::uint32_t signal_count,
    std::uint32_t instruction_count, std::uint32_t cell_count, std::uint32_t level_count,
    std::uint32_t crank_nicolson, const std::uint32_t* scatter_ends,
    const std::uint32_t* scatter_cells, cudaStream_t stream) {
  if (cell_count != 0) {
    const auto cell_blocks = ((cell_count - 1) / threads_per_block) + 1;
    advance_coupled_cells<<<cell_blocks, threads_per_block, 0, stream>>>(
        species_levels, previous_lengths, centers, geometry, growth_rates, cell_types, instructions,
        species_outputs, signal_outputs, workspace, grid_levels, cell_signal_rates, obstacles,
        error, shape, origin, spacing, dt, species_count, signal_count, instruction_count,
        cell_count);
    const auto cell_error = cudaGetLastError();
    if (cell_error != cudaSuccess) {
      return cell_error;
    }
  }
  const auto grid_blocks = ((level_count - 1) / threads_per_block) + 1;
  advance_coupled_grid<<<grid_blocks, threads_per_block, 0, stream>>>(
      grid_levels, grid_output, diffusion, advection, fixed_values, reaction_source, reaction_loss,
      centers, cell_signal_rates, obstacles, x_faces, y_faces, z_faces, has_velocity_field, error,
      boundaries, shape, origin, spacing, dt, signal_count, cell_count, level_count, crank_nicolson,
      scatter_ends, scatter_cells);
  return cudaGetLastError();
}

}  // namespace cm::cuda
