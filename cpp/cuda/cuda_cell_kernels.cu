#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cub/device/device_scan.cuh>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/cell_contact_plan.hpp"
#include "core/deposition_plan.hpp"
#include "core/external_contact_plan.hpp"
#include "core/flow_drift_plan.hpp"
#include "cuda_cell_kernels.cuh"
#include "device_types.cuh"
#include "kernels/cell_contacts.cuh"
#include "kernels/external_contacts.cuh"
#include "kernels/flow_drift.cuh"

namespace microsimulator::cuda {
namespace {

namespace kernels = microsimulator::cuda::device;

void check_cuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

// The same bytes as a host-side shared structure, in its CUDA spelling.
template <typename Device, typename Host>
[[nodiscard]] Device as_device(const Host& value) {
  static_assert(sizeof(Device) == sizeof(Host));
  Device result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

// A device array of at least one element.
template <typename T>
class DeviceArray {
 public:
  DeviceArray(std::size_t count, const char* description)
      : count_(std::max<std::size_t>(count, 1)) {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)), description);
  }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  ~DeviceArray() {
    static_cast<void>(cudaFree(data_));
  }

  [[nodiscard]] T* data() noexcept {
    return data_;
  }

  template <typename Host>
  void upload(const std::vector<Host>& values, cudaStream_t stream, const char* operation) {
    static_assert(sizeof(Host) == sizeof(T));
    check_cuda(cudaMemcpyAsync(data_, values.data(), values.size() * sizeof(T),
                               cudaMemcpyHostToDevice, stream),
               operation);
  }

  void zero(cudaStream_t stream, const char* operation) {
    check_cuda(cudaMemsetAsync(data_, 0, count_ * sizeof(T), stream), operation);
  }

  template <typename Host>
  [[nodiscard]] std::vector<Host> download(std::size_t count, cudaStream_t stream,
                                           const char* operation) {
    static_assert(sizeof(Host) == sizeof(T));
    std::vector<Host> result(count);
    check_cuda(
        cudaMemcpyAsync(result.data(), data_, count * sizeof(T), cudaMemcpyDeviceToHost, stream),
        operation);
    check_cuda(cudaStreamSynchronize(stream), operation);
    return result;
  }

 private:
  T* data_{nullptr};
  std::size_t count_{0};
};

constexpr unsigned int block_size = 256;

[[nodiscard]] unsigned int blocks(std::uint32_t count) {
  return (count + block_size - 1) / block_size;
}

__device__ unsigned int element_index() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}

// Integral of the unit tent, 1 - |u| on [-1, 1], from -1 up to `edge`.
__device__ float tent_cdf(float edge) {
  if (edge <= -1.0f) {
    return 0.0f;
  }
  if (edge < 0.0f) {
    return 0.5f * (edge + 1.0f) * (edge + 1.0f);
  }
  if (edge < 1.0f) {
    return 1.0f - 0.5f * (1.0f - edge) * (1.0f - edge);
  }
  return 1.0f;
}

// The integral over site `index` of a tent of `radius` about `center`, along
// an axis whose sites are centered at origin + index * spacing.
__device__ float axis_weight(unsigned int index, float origin, float spacing, float center,
                             float radius) {
  const float lower = (origin + (float(index) - 0.5f) * spacing - center) / radius;
  const float upper = (origin + (float(index) + 0.5f) * spacing - center) / radius;
  return tent_cdf(upper) - tent_cdf(lower);
}

// The conserved biomass of a cell whose center holds its length in w.
__device__ float cell_biomass(float4 center, float radius) {
  return shared::pi * radius * radius * (center.w + 2.0f * radius);
}

// The box of sites that the tent of a cell centered at `center` reaches; a
// cell outside the sites' extent has an empty box.
__device__ kernels::DepositBox cell_box(const kernels::DepositGrid& grid, float4 center) {
  const float centers[3] = {center.x, center.y, center.z};
  const float origins[3] = {grid.origin.x, grid.origin.y, grid.origin.z};
  const float spacings[3] = {grid.spacing.x, grid.spacing.y, grid.spacing.z};
  const unsigned int counts[3] = {grid.shape.x, grid.shape.y, grid.shape.z};
  const unsigned int largest[3] = {grid.box.x, grid.box.y, grid.box.z};
  const float radius = grid.origin.w;
  unsigned int lower[3] = {0, 0, 0};
  unsigned int extent[3] = {0, 0, 0};
  bool inside = true;
  for (int axis = 0; axis < 3; ++axis) {
    const float first = origins[axis] - spacings[axis] * 0.5f;
    const float end = origins[axis] + (float(counts[axis]) - 0.5f) * spacings[axis];
    inside = inside && centers[axis] >= first && centers[axis] < end;
  }
  if (inside) {
    for (int axis = 0; axis < 3; ++axis) {
      const int low =
          max(0, int(floorf((centers[axis] - radius - origins[axis]) / spacings[axis] + 0.5f)));
      int high = int(ceilf((centers[axis] + radius - origins[axis]) / spacings[axis] + 0.5f));
      high = min(high, max(low, int(counts[axis])));
      high = max(high, low);
      lower[axis] = unsigned(low);
      extent[axis] = min(unsigned(high - low), largest[axis]);
    }
  }
  kernels::DepositBox box;
  box.lower = make_uint4(lower[0], lower[1], lower[2], inside ? 1u : 0u);
  box.extent = make_uint4(extent[0], extent[1], extent[2], extent[0] * extent[1] * extent[2]);
  return box;
}

// The flat index of the box site at `local`, which counts x-major and
// z-minor through the box as the grid counts its sites.
__device__ unsigned int box_site(const kernels::DepositGrid& grid, const kernels::DepositBox& box,
                                 unsigned int local) {
  const unsigned int iz = local % box.extent.z;
  const unsigned int iy = (local / box.extent.z) % box.extent.y;
  const unsigned int ix = local / (box.extent.y * box.extent.z);
  return ((box.lower.x + ix) * grid.shape.y + box.lower.y + iy) * grid.shape.z + box.lower.z + iz;
}

// The box and weights of one cell, which sum to one. Where the grid has
// obstacles, only the face-connected fluid region holding the strongest
// weight carries weight: corner contact is not fluid, and biomass does not
// reach through a wall. Status is one when no fluid site carries weight.
__global__ void deposit_weights(kernels::DepositGrid grid, const float4* centers,
                                const unsigned char* solid, kernels::DepositBox* boxes,
                                float* weights, unsigned int* status) {
  const auto cell = element_index();
  if (cell >= grid.cell_count) {
    return;
  }
  const float4 center = centers[cell];
  const auto box = cell_box(grid, center);
  boxes[cell] = box;
  status[cell] = 0;
  if (box.lower.w == 0) {
    return;
  }
  float* block = weights + cell * grid.box.w;
  const unsigned int ex = box.extent.x;
  const unsigned int ey = box.extent.y;
  const unsigned int ez = box.extent.z;
  const float radius = grid.origin.w;
  float strongest = 0.0f;
  unsigned int seed = 0;
  for (unsigned int ix = 0; ix < ex; ++ix) {
    const float wx = axis_weight(box.lower.x + ix, grid.origin.x, grid.spacing.x, center.x, radius);
    for (unsigned int iy = 0; iy < ey; ++iy) {
      const float wxy =
          wx * axis_weight(box.lower.y + iy, grid.origin.y, grid.spacing.y, center.y, radius);
      for (unsigned int iz = 0; iz < ez; ++iz) {
        const unsigned int local = (ix * ey + iy) * ez + iz;
        float weight =
            wxy * axis_weight(box.lower.z + iz, grid.origin.z, grid.spacing.z, center.z, radius);
        if (grid.has_obstacles != 0 && solid[box_site(grid, box, local)] != 0) {
          weight = 0.0f;
        }
        block[local] = weight;
        if (weight > strongest) {
          strongest = weight;
          seed = local;
        }
      }
    }
  }
  if (!(strongest > 0.0f)) {
    status[cell] = 1;
    return;
  }
  if (grid.has_obstacles != 0) {
    // A negative weight marks a site reached from the seed. Sweeps repeat
    // until one reaches no new site.
    const unsigned int plane = ey * ez;
    block[seed] = -block[seed];
    bool changed = true;
    while (changed) {
      changed = false;
      for (unsigned int local = 0; local < box.extent.w; ++local) {
        if (!(block[local] > 0.0f)) {
          continue;
        }
        const unsigned int ix = local / plane;
        const unsigned int iy = (local / ez) % ey;
        const unsigned int iz = local % ez;
        const bool reached =
            (ix > 0 && block[local - plane] < 0.0f) ||
            (ix + 1 < ex && block[local + plane] < 0.0f) || (iy > 0 && block[local - ez] < 0.0f) ||
            (iy + 1 < ey && block[local + ez] < 0.0f) || (iz > 0 && block[local - 1] < 0.0f) ||
            (iz + 1 < ez && block[local + 1] < 0.0f);
        if (reached) {
          block[local] = -block[local];
          changed = true;
        }
      }
    }
    for (unsigned int local = 0; local < box.extent.w; ++local) {
      block[local] = block[local] < 0.0f ? -block[local] : 0.0f;
    }
  }
  float total = 0.0f;
  for (unsigned int local = 0; local < box.extent.w; ++local) {
    total += block[local];
  }
  for (unsigned int local = 0; local < box.extent.w; ++local) {
    block[local] /= total;
  }
}

// Counts each cell once at every site where it carries weight.
__global__ void deposit_count(kernels::DepositGrid grid, const kernels::DepositBox* boxes,
                              const float* weights, unsigned int* counts) {
  const auto cell = element_index();
  if (cell >= grid.cell_count) {
    return;
  }
  const auto box = boxes[cell];
  const float* block = weights + cell * grid.box.w;
  for (unsigned int local = 0; local < box.extent.w; ++local) {
    if (block[local] > 0.0f) {
      atomicAdd(&counts[box_site(grid, box, local)], 1u);
    }
  }
}

// Enters each cell in the range of every site where it carries weight.
// `ends` is the inclusive prefix sum of the counts; site s owns entries
// [ends[s - 1], ends[s]). Cursors start at zero.
__global__ void deposit_fill(kernels::DepositGrid grid, const kernels::DepositBox* boxes,
                             const float* weights, const unsigned int* ends, unsigned int* cursors,
                             unsigned int* entries) {
  const auto cell = element_index();
  if (cell >= grid.cell_count) {
    return;
  }
  const auto box = boxes[cell];
  const float* block = weights + cell * grid.box.w;
  for (unsigned int local = 0; local < box.extent.w; ++local) {
    if (!(block[local] > 0.0f)) {
      continue;
    }
    const unsigned int site = box_site(grid, box, local);
    const unsigned int first = site == 0 ? 0 : ends[site - 1];
    entries[first + atomicAdd(&cursors[site], 1u)] = cell;
  }
}

// Restores the max-heap property below `root` in a list of `count` cells.
__device__ void sift_down(unsigned int* list, unsigned int root, unsigned int count) {
  const unsigned int value = list[root];
  for (;;) {
    unsigned int child = 2 * root + 1;
    if (child >= count) {
      break;
    }
    if (child + 1 < count && list[child + 1] > list[child]) {
      ++child;
    }
    if (list[child] <= value) {
      break;
    }
    list[root] = list[child];
    root = child;
  }
  list[root] = value;
}

// The weight that `cell` carries at the site at grid coordinates (x, y, z),
// which its box contains.
__device__ float site_weight(const kernels::DepositGrid& grid, const kernels::DepositBox* boxes,
                             const float* weights, unsigned int cell, unsigned int x,
                             unsigned int y, unsigned int z) {
  const auto box = boxes[cell];
  const unsigned int local =
      ((x - box.lower.x) * box.extent.y + (y - box.lower.y)) * box.extent.z + (z - box.lower.z);
  return weights[cell * grid.box.w + local];
}

// Each site's biomass per unit volume, and for each channel the channel's
// per-cell value times biomass per unit volume, over the site's cells in
// ascending order. Heapsort keeps a site that many cells reach at
// O(n log n) without scratch storage.
__global__ void deposit_sites(kernels::DepositGrid grid, const unsigned int* ends,
                              unsigned int* entries, const kernels::DepositBox* boxes,
                              const float* weights, const float4* centers, const float* radii,
                              const float* channels, float* density, float* channel_density) {
  const auto site = element_index();
  if (site >= grid.shape.w) {
    return;
  }
  const unsigned int first = site == 0 ? 0 : ends[site - 1];
  const unsigned int count = ends[site] - first;
  unsigned int* list = entries + first;
  if (count > 1) {
    for (unsigned int root = count / 2; root > 0; --root) {
      sift_down(list, root - 1, count);
    }
    for (unsigned int last = count - 1; last > 0; --last) {
      const unsigned int largest = list[0];
      list[0] = list[last];
      list[last] = largest;
      sift_down(list, 0, last);
    }
  }
  const unsigned int z = site % grid.shape.z;
  const unsigned int y = (site / grid.shape.z) % grid.shape.y;
  const unsigned int x = site / (grid.shape.y * grid.shape.z);
  float mass = 0.0f;
  for (unsigned int entry = 0; entry < count; ++entry) {
    const unsigned int cell = list[entry];
    mass +=
        cell_biomass(centers[cell], radii[cell]) * site_weight(grid, boxes, weights, cell, x, y, z);
  }
  density[site] = mass / grid.spacing.w;
  for (unsigned int channel = 0; channel < grid.channel_count; ++channel) {
    float amount = 0.0f;
    for (unsigned int entry = 0; entry < count; ++entry) {
      const unsigned int cell = list[entry];
      const float deposited = cell_biomass(centers[cell], radii[cell]) *
                              site_weight(grid, boxes, weights, cell, x, y, z);
      amount += channels[channel * grid.cell_count + cell] * deposited;
    }
    channel_density[channel * grid.shape.w + site] = amount / grid.spacing.w;
  }
}

// The Kozeny-Carman drag of biomass packed to a volume fraction.
__device__ float kozeny_carman(float fraction, float coefficient) {
  const float open = 1.0f - fraction;
  return coefficient * fraction * fraction / (open * open * open);
}

// The shallow-flow coefficients of one x/y column. The column's biomass
// fraction is its density averaged over its fluid sites and capped, which
// sets the drag d. Each fluid site of base mobility b takes mobility
// b / (1 + b d), and the column takes the body force d V, where V is the
// biomass-weighted velocity that channels 0 and 1 carry.
__global__ void shallow_coupling(kernels::ShallowCouplingData data, const unsigned char* solid,
                                 const float* base, const float* density, const float* channels,
                                 float* mobility, float* force) {
  const auto column = element_index();
  if (column >= data.shape.x * data.shape.y) {
    return;
  }
  const unsigned int depth = data.shape.z;
  const unsigned int first = column * depth;
  unsigned int fluid = 0;
  float mass = 0.0f;
  for (unsigned int z = 0; z < depth; ++z) {
    fluid += solid[first + z] == 0 ? 1u : 0u;
    mass += density[first + z];
  }
  const float fraction = fminf(mass / float(max(fluid, 1u)), data.max_volume_fraction);
  const float drag = kozeny_carman(fraction, data.drag_coefficient);
  for (unsigned int z = 0; z < depth; ++z) {
    const float b = data.uniform_base != 0 ? base[0] : base[first + z];
    mobility[first + z] = solid[first + z] != 0 ? 0.0f : b / (1.0f + b * drag);
  }
  float fx = 0.0f;
  float fy = 0.0f;
  if (data.has_velocity != 0 && mass > 0.0f) {
    float mx = 0.0f;
    float my = 0.0f;
    for (unsigned int z = 0; z < depth; ++z) {
      mx += channels[first + z];
      my += channels[data.shape.w + first + z];
    }
    fx = mx * (drag / mass);
    fy = my * (drag / mass);
  }
  force[2 * column] = fx;
  force[2 * column + 1] = fy;
}

// The Brinkman drag at one site: the Kozeny-Carman drag of its biomass
// fraction, capped, and zero on a solid site.
__global__ void brinkman_drag_kernel(kernels::BrinkmanDragData data, const unsigned char* solid,
                                     const float* density, float* drag) {
  const auto site = element_index();
  if (site >= data.shape.w) {
    return;
  }
  const float fraction = fminf(density[site], data.max_volume_fraction);
  drag[site] = solid[site] != 0 ? 0.0f : kozeny_carman(fraction, data.drag_coefficient);
}

__global__ void cell_bounds_kernel(kernels::CellContactData data, const float4* centers,
                                   const float4* axes, float4* lower, float4* upper) {
  const auto cell = element_index();
  if (cell < data.cell_count) {
    kernels::cell_bounds(cell, data, centers, axes, lower, upper);
  }
}

__global__ void cell_width_leaf_kernel(unsigned int count, const float4* lower, const float4* upper,
                                       float4* output) {
  const auto index = element_index();
  if (index < (count + 1) / 2) {
    kernels::cell_width_leaf(index, count, lower, upper, output);
  }
}

__global__ void cell_width_pair_kernel(unsigned int count, const float4* input, float4* output) {
  const auto index = element_index();
  if (index < (count + 1) / 2) {
    kernels::cell_width_pair(index, count, input, output);
  }
}

__global__ void cell_grid_place_kernel(kernels::CellContactData data, const float4* lower,
                                       const float4* upper, const float4* widest,
                                       uint4* coordinates, unsigned int* bucket_counts) {
  const auto cell = element_index();
  if (cell < data.cell_count) {
    kernels::cell_grid_place(cell, data, lower, upper, widest, coordinates, bucket_counts);
  }
}

__global__ void cell_grid_fill_kernel(kernels::CellContactData data, const uint4* coordinates,
                                      const unsigned int* bucket_ends, unsigned int* bucket_cursors,
                                      unsigned int* members) {
  const auto cell = element_index();
  if (cell < data.cell_count) {
    kernels::cell_grid_fill(cell, coordinates, bucket_ends, bucket_cursors, members);
  }
}

__global__ void cell_candidate_count_kernel(kernels::CellContactData data,
                                            const unsigned long long* ids, const float4* lower,
                                            const float4* upper, const uint4* coordinates,
                                            const unsigned int* bucket_ends,
                                            const unsigned int* members, uint2* candidates,
                                            unsigned int* counts) {
  const auto cell = element_index();
  if (cell < data.cell_count) {
    counts[cell] = kernels::cell_candidate_count(cell, data, ids, lower, upper, coordinates,
                                                 bucket_ends, members, candidates);
  }
}

__global__ void cell_candidate_fill_kernel(kernels::CellContactData data,
                                           const unsigned long long* ids, const float4* lower,
                                           const float4* upper, const uint4* coordinates,
                                           const unsigned int* bucket_ends,
                                           const unsigned int* members, const unsigned int* ends,
                                           uint2* candidates) {
  const auto cell = element_index();
  if (cell < data.cell_count) {
    kernels::cell_candidate_fill(cell, data, ids, lower, upper, coordinates, bucket_ends, members,
                                 ends, candidates);
  }
}

__global__ void cell_pair_count_kernel(kernels::CellContactData data, unsigned int total,
                                       const float4* centers, const float4* axes,
                                       const uint2* candidates, unsigned int* counts) {
  const auto candidate = element_index();
  if (candidate < total) {
    counts[candidate] = kernels::cell_pair_count(candidate, total, data, centers, axes, candidates);
  }
}

__global__ void cell_pair_fill_kernel(kernels::CellContactData data, unsigned int total,
                                      const float4* centers, const float4* axes,
                                      const uint2* candidates, const unsigned int* ends,
                                      kernels::CellContactRecord* records) {
  const auto candidate = element_index();
  if (candidate < total) {
    kernels::cell_pair_fill(candidate, total, data, centers, axes, candidates, ends, records);
  }
}

__global__ void external_count(kernels::ExternalContactData data, const float4* centers,
                               const float4* axes, const kernels::ExternalConstraint* constraints,
                               unsigned int* counts) {
  const auto pair = element_index();
  if (pair < data.cell_count * data.constraint_count) {
    counts[pair] = kernels::external_pair_count(pair, data, centers, axes, constraints);
  }
}

__global__ void external_fill(kernels::ExternalContactData data, const float4* centers,
                              const float4* axes, const kernels::ExternalConstraint* constraints,
                              const unsigned int* ends, kernels::ExternalContactRecord* records) {
  const auto pair = element_index();
  if (pair < data.cell_count * data.constraint_count) {
    kernels::external_pair_fill(pair, data, centers, axes, constraints, ends, records);
  }
}

__global__ void drift_rates(kernels::FlowDriftData data, const float4* centers, const float4* axes,
                            const unsigned char* fixed_cells, const unsigned char* solid,
                            const float* x_faces, const float* y_faces, const float* z_faces,
                            float* rates, unsigned int* status) {
  const auto cell = element_index();
  if (cell < data.cell_count) {
    status[cell] = kernels::flow_drift_rates_cell(cell, data, centers, axes, fixed_cells, solid,
                                                  x_faces, y_faces, z_faces, rates);
  }
}

__global__ void drift_advance(kernels::FlowDriftData data, const float4* centers,
                              const float4* axes, const unsigned char* fixed_cells,
                              const unsigned char* solid, const float* x_faces,
                              const float* y_faces, const float* z_faces, float4* moved_centers,
                              float4* moved_axes, unsigned int* status) {
  const auto cell = element_index();
  if (cell < data.cell_count) {
    status[cell] =
        kernels::flow_drift_advance_cell(cell, data, centers, axes, fixed_cells, solid, x_faces,
                                         y_faces, z_faces, moved_centers, moved_axes);
  }
}

void check_launch(const char* operation) {
  check_cuda(cudaGetLastError(), operation);
}

// Replaces the `count` values at `values` with their inclusive prefix sum
// and returns `values`.
unsigned int* encode_scan(unsigned int* values, std::uint32_t count, cudaStream_t stream) {
  if (count == 0) {
    return values;
  }
  std::size_t bytes = 0;
  check_cuda(cub::DeviceScan::InclusiveSum(nullptr, bytes, values, values, static_cast<int>(count),
                                           stream),
             "failed to size a CUDA scan");
  DeviceArray<unsigned char> storage(bytes, "failed to allocate CUDA scan storage");
  check_cuda(cub::DeviceScan::InclusiveSum(storage.data(), bytes, values, values,
                                           static_cast<int>(count), stream),
             "failed to run a CUDA scan");
  check_cuda(cudaStreamSynchronize(stream), "failed to finish a CUDA scan");
  return values;
}

[[nodiscard]] std::uint32_t last_value(const unsigned int* ends, std::uint32_t count,
                                       cudaStream_t stream) {
  unsigned int total = 0;
  check_cuda(
      cudaMemcpyAsync(&total, ends + (count - 1), sizeof(total), cudaMemcpyDeviceToHost, stream),
      "failed to download a CUDA scan total");
  check_cuda(cudaStreamSynchronize(stream), "failed to download a CUDA scan total");
  return total;
}

}  // namespace

ContactGraph cell_contacts(const WorldState& state, const ContactParameters& parameters,
                           cudaStream_t stream) {
  auto plan = detail::plan_cell_contacts(state, parameters);
  const auto cells = plan.data.cell_count;
  if (cells < 2) {
    return detail::finish_cell_contacts(plan, {});
  }
  const auto buckets = plan.data.bucket_count;
  DeviceArray<float4> centers(cells, "failed to allocate CUDA cell centers");
  DeviceArray<float4> axes(cells, "failed to allocate CUDA cell axes");
  DeviceArray<unsigned long long> ids(cells, "failed to allocate CUDA cell ids");
  DeviceArray<float4> lower(cells, "failed to allocate CUDA cell boxes");
  DeviceArray<float4> upper(cells, "failed to allocate CUDA cell boxes");
  DeviceArray<float4> width_a(cells, "failed to allocate CUDA box widths");
  DeviceArray<float4> width_b(cells, "failed to allocate CUDA box widths");
  DeviceArray<uint4> coordinates(cells, "failed to allocate CUDA grid boxes");
  DeviceArray<unsigned int> bucket_counts(buckets, "failed to allocate CUDA bucket counts");
  DeviceArray<unsigned int> bucket_cursors(buckets, "failed to allocate CUDA bucket cursors");
  DeviceArray<unsigned int> members(cells, "failed to allocate CUDA grid members");
  DeviceArray<unsigned int> candidate_counts(cells, "failed to allocate CUDA candidate counts");
  centers.upload(plan.centers, stream, "failed to upload CUDA cell centers");
  axes.upload(plan.axes, stream, "failed to upload CUDA cell axes");
  ids.upload(plan.ids, stream, "failed to upload CUDA cell ids");
  bucket_counts.zero(stream, "failed to clear CUDA bucket counts");
  bucket_cursors.zero(stream, "failed to clear CUDA bucket cursors");

  auto data = as_device<kernels::CellContactData>(plan.data);
  cell_bounds_kernel<<<blocks(cells), block_size, 0, stream>>>(data, centers.data(), axes.data(),
                                                               lower.data(), upper.data());
  check_launch("failed to launch the CUDA cell-bounds kernel");
  auto width_count = cells;
  cell_width_leaf_kernel<<<blocks((width_count + 1) / 2), block_size, 0, stream>>>(
      width_count, lower.data(), upper.data(), width_a.data());
  check_launch("failed to launch the CUDA box-width kernel");
  width_count = (width_count + 1) / 2;
  float4* widest = width_a.data();
  float4* spare = width_b.data();
  while (width_count > 1) {
    cell_width_pair_kernel<<<blocks((width_count + 1) / 2), block_size, 0, stream>>>(width_count,
                                                                                     widest, spare);
    check_launch("failed to launch the CUDA box-width kernel");
    width_count = (width_count + 1) / 2;
    std::swap(widest, spare);
  }
  cell_grid_place_kernel<<<blocks(cells), block_size, 0, stream>>>(
      data, lower.data(), upper.data(), widest, coordinates.data(), bucket_counts.data());
  check_launch("failed to launch the CUDA grid-place kernel");
  const auto* bucket_ends = encode_scan(bucket_counts.data(), buckets, stream);
  cell_grid_fill_kernel<<<blocks(cells), block_size, 0, stream>>>(
      data, coordinates.data(), bucket_ends, bucket_cursors.data(), members.data());
  check_launch("failed to launch the CUDA grid-fill kernel");

  DeviceArray<uint2> unused(1, "failed to allocate CUDA candidates");
  cell_candidate_count_kernel<<<blocks(cells), block_size, 0, stream>>>(
      data, ids.data(), lower.data(), upper.data(), coordinates.data(), bucket_ends, members.data(),
      unused.data(), candidate_counts.data());
  check_launch("failed to launch the CUDA candidate-count kernel");
  const auto* candidate_ends = encode_scan(candidate_counts.data(), cells, stream);
  const auto candidate_total = last_value(candidate_ends, cells, stream);
  if (candidate_total == 0) {
    return detail::finish_cell_contacts(plan, {});
  }
  plan.data.candidate_capacity = candidate_total;
  data = as_device<kernels::CellContactData>(plan.data);
  DeviceArray<uint2> candidates(candidate_total, "failed to allocate CUDA candidates");
  cell_candidate_fill_kernel<<<blocks(cells), block_size, 0, stream>>>(
      data, ids.data(), lower.data(), upper.data(), coordinates.data(), bucket_ends, members.data(),
      candidate_ends, candidates.data());
  check_launch("failed to launch the CUDA candidate-fill kernel");

  DeviceArray<unsigned int> pair_counts(candidate_total, "failed to allocate CUDA contact counts");
  cell_pair_count_kernel<<<blocks(candidate_total), block_size, 0, stream>>>(
      data, candidate_total, centers.data(), axes.data(), candidates.data(), pair_counts.data());
  check_launch("failed to launch the CUDA cell-pair-count kernel");
  const auto* contact_ends = encode_scan(pair_counts.data(), candidate_total, stream);
  const auto contact_total = last_value(contact_ends, candidate_total, stream);
  plan.data.contact_capacity = contact_total;
  data = as_device<kernels::CellContactData>(plan.data);
  DeviceArray<kernels::CellContactRecord> records(contact_total,
                                                  "failed to allocate CUDA cell contacts");
  cell_pair_fill_kernel<<<blocks(candidate_total), block_size, 0, stream>>>(
      data, candidate_total, centers.data(), axes.data(), candidates.data(), contact_ends,
      records.data());
  check_launch("failed to launch the CUDA cell-pair-fill kernel");
  const auto host_records = records.download<microsimulator::device::CellContactRecord>(
      contact_total, stream, "failed to download CUDA cell contacts");
  return detail::finish_cell_contacts(plan, host_records);
}

BiomassDeposit deposit_biomass(const SignalGridSpec& spec, const DepositCells& cells,
                               const BiomassDepositParameters& parameters, cudaStream_t stream) {
  const auto plan = detail::plan_deposit(spec, cells, parameters);
  const auto grid = as_device<kernels::DepositGrid>(plan.grid);
  const auto cell_count = plan.cell_count();
  const auto site_count = plan.site_count();
  const auto channel_values = static_cast<std::size_t>(grid.channel_count) * site_count;

  DeviceArray<float4> centers(plan.centers.size(), "failed to allocate CUDA deposit centers");
  DeviceArray<float> radii(plan.radii.size(), "failed to allocate CUDA deposit radii");
  DeviceArray<float> channels(plan.channels.size(), "failed to allocate CUDA deposit channels");
  DeviceArray<unsigned char> solid(plan.solid.size(), "failed to allocate CUDA deposit obstacles");
  DeviceArray<kernels::DepositBox> boxes(plan.centers.size(),
                                         "failed to allocate CUDA deposit boxes");
  DeviceArray<float> weights(plan.weight_count(), "failed to allocate CUDA deposit weights");
  DeviceArray<unsigned int> status(cell_count, "failed to allocate CUDA deposit status");
  DeviceArray<unsigned int> counts(site_count, "failed to allocate CUDA deposit counts");
  DeviceArray<unsigned int> cursors(site_count, "failed to allocate CUDA deposit cursors");
  // Each cell enters at most every site of its box.
  DeviceArray<unsigned int> entries(plan.weight_count(), "failed to allocate CUDA deposit entries");
  DeviceArray<float> density(site_count, "failed to allocate CUDA deposit density");
  DeviceArray<float> channel_density(channel_values,
                                     "failed to allocate CUDA deposit channel density");

  centers.upload(plan.centers, stream, "failed to upload CUDA deposit centers");
  radii.upload(plan.radii, stream, "failed to upload CUDA deposit radii");
  channels.upload(plan.channels, stream, "failed to upload CUDA deposit channels");
  solid.upload(plan.solid, stream, "failed to upload CUDA deposit obstacles");
  status.zero(stream, "failed to clear CUDA deposit status");
  counts.zero(stream, "failed to clear CUDA deposit counts");
  cursors.zero(stream, "failed to clear CUDA deposit cursors");

  if (cell_count > 0) {
    deposit_weights<<<blocks(cell_count), block_size, 0, stream>>>(
        grid, centers.data(), solid.data(), boxes.data(), weights.data(), status.data());
    check_launch("failed to launch the CUDA deposit-weights kernel");
    deposit_count<<<blocks(cell_count), block_size, 0, stream>>>(grid, boxes.data(), weights.data(),
                                                                 counts.data());
    check_launch("failed to launch the CUDA deposit-count kernel");
  }
  unsigned int* ends = encode_scan(counts.data(), site_count, stream);
  if (cell_count > 0) {
    deposit_fill<<<blocks(cell_count), block_size, 0, stream>>>(
        grid, boxes.data(), weights.data(), ends, cursors.data(), entries.data());
    check_launch("failed to launch the CUDA deposit-fill kernel");
  }
  deposit_sites<<<blocks(site_count), block_size, 0, stream>>>(
      grid, ends, entries.data(), boxes.data(), weights.data(), centers.data(), radii.data(),
      channels.data(), density.data(), channel_density.data());
  check_launch("failed to launch the CUDA deposit-sites kernel");

  detail::check_deposit_status(
      status.download<std::uint32_t>(cell_count, stream, "failed to download CUDA deposit status"));
  BiomassDeposit result;
  result.density =
      density.download<float>(site_count, stream, "failed to download CUDA deposit density");
  result.channel_count = grid.channel_count;
  result.channels = channel_density.download<float>(
      channel_values, stream, "failed to download CUDA deposit channel density");
  return result;
}

ShallowFlowCoupling shallow_flow_coupling(const SignalGridSpec& spec, std::span<const float> base,
                                          const BiomassDeposit& deposit,
                                          const BiomassDragParameters& parameters,
                                          cudaStream_t stream) {
  const auto plan = detail::plan_shallow_flow_coupling(spec, base, deposit, parameters);
  const auto data = as_device<kernels::ShallowCouplingData>(plan.data);
  const auto columns = plan.column_count();
  const auto site_count = data.shape.w;

  DeviceArray<unsigned char> solid(plan.solid.size(), "failed to allocate CUDA coupling obstacles");
  DeviceArray<float> base_mobility(plan.base.size(), "failed to allocate CUDA base mobility");
  DeviceArray<float> density(deposit.density.size(), "failed to allocate CUDA coupling density");
  DeviceArray<float> velocity(plan.velocity.size(), "failed to allocate CUDA coupling velocity");
  DeviceArray<float> mobility(site_count, "failed to allocate CUDA coupling mobility");
  DeviceArray<float> force(2 * static_cast<std::size_t>(columns),
                           "failed to allocate CUDA coupling force");

  solid.upload(plan.solid, stream, "failed to upload CUDA coupling obstacles");
  base_mobility.upload(plan.base, stream, "failed to upload CUDA base mobility");
  density.upload(deposit.density, stream, "failed to upload CUDA coupling density");
  velocity.upload(plan.velocity, stream, "failed to upload CUDA coupling velocity");
  shallow_coupling<<<blocks(columns), block_size, 0, stream>>>(
      data, solid.data(), base_mobility.data(), density.data(), velocity.data(), mobility.data(),
      force.data());
  check_launch("failed to launch the CUDA shallow-coupling kernel");
  const auto forces = force.download<float>(2 * static_cast<std::size_t>(columns), stream,
                                            "failed to download CUDA coupling force");
  return detail::finish_shallow_flow_coupling(
      mobility.download<float>(site_count, stream, "failed to download CUDA coupling mobility"),
      forces);
}

std::vector<float> brinkman_drag(const SignalGridSpec& spec, const BiomassDeposit& deposit,
                                 const BiomassDragParameters& parameters, cudaStream_t stream) {
  const auto plan = detail::plan_brinkman_drag(spec, deposit, parameters);
  const auto data = as_device<kernels::BrinkmanDragData>(plan.data);
  const auto site_count = data.shape.w;
  DeviceArray<unsigned char> solid(plan.solid.size(), "failed to allocate CUDA drag obstacles");
  DeviceArray<float> density(deposit.density.size(), "failed to allocate CUDA drag density");
  DeviceArray<float> drag(site_count, "failed to allocate CUDA drag");
  solid.upload(plan.solid, stream, "failed to upload CUDA drag obstacles");
  density.upload(deposit.density, stream, "failed to upload CUDA drag density");
  brinkman_drag_kernel<<<blocks(site_count), block_size, 0, stream>>>(data, solid.data(),
                                                                      density.data(), drag.data());
  check_launch("failed to launch the CUDA Brinkman-drag kernel");
  return drag.download<float>(site_count, stream, "failed to download CUDA drag");
}

ExternalContactGraph external_contacts(const WorldState& state, const ConstraintSet& constraints,
                                       const ConstraintContactParameters& parameters,
                                       bool every_face, cudaStream_t stream) {
  const auto plan = detail::plan_external_contacts(state, constraints, parameters, every_face);
  const auto pairs = plan.pair_count();
  if (pairs == 0) {
    return ExternalContactGraph(plan.data.cell_count, {});
  }
  const auto data = as_device<kernels::ExternalContactData>(plan.data);
  const auto pair_count = static_cast<std::uint32_t>(pairs);
  DeviceArray<float4> centers(plan.centers.size(), "failed to allocate CUDA contact centers");
  DeviceArray<float4> axes(plan.axes.size(), "failed to allocate CUDA contact axes");
  DeviceArray<kernels::ExternalConstraint> packed(plan.constraints.size(),
                                                  "failed to allocate CUDA constraints");
  DeviceArray<unsigned int> counts(pairs, "failed to allocate CUDA contact counts");
  // A pair makes at most shared::external_pair_capacity contacts.
  DeviceArray<kernels::ExternalContactRecord> records(pairs * 12,
                                                      "failed to allocate CUDA contact records");
  centers.upload(plan.centers, stream, "failed to upload CUDA contact centers");
  axes.upload(plan.axes, stream, "failed to upload CUDA contact axes");
  packed.upload(plan.constraints, stream, "failed to upload CUDA constraints");
  external_count<<<blocks(pair_count), block_size, 0, stream>>>(data, centers.data(), axes.data(),
                                                                packed.data(), counts.data());
  check_launch("failed to launch the CUDA external-count kernel");
  unsigned int* ends = encode_scan(counts.data(), pair_count, stream);
  external_fill<<<blocks(pair_count), block_size, 0, stream>>>(data, centers.data(), axes.data(),
                                                               packed.data(), ends, records.data());
  check_launch("failed to launch the CUDA external-fill kernel");
  unsigned int total = 0;
  check_cuda(cudaMemcpyAsync(&total, ends + (pair_count - 1), sizeof(total), cudaMemcpyDeviceToHost,
                             stream),
             "failed to download the CUDA contact count");
  const auto host_records = records.download<microsimulator::device::ExternalContactRecord>(
      total, stream, "failed to download CUDA contact records");
  return detail::finish_external_contacts(plan, host_records);
}

namespace {

// A drift's cells, and its grid's obstacles and velocity field, on the device.
struct DriftInputs {
  explicit DriftInputs(const detail::FlowDriftPlan& plan, cudaStream_t stream)
      : centers(plan.centers.size(), "failed to allocate CUDA drift centers"),
        axes(plan.axes.size(), "failed to allocate CUDA drift axes"),
        fixed(plan.fixed.size(), "failed to allocate CUDA drift fixed flags"),
        solid(plan.solid.size(), "failed to allocate CUDA drift obstacles"),
        x_faces(plan.field->x_faces.size(), "failed to allocate CUDA drift x faces"),
        y_faces(plan.field->y_faces.size(), "failed to allocate CUDA drift y faces"),
        z_faces(plan.field->z_faces.size(), "failed to allocate CUDA drift z faces"),
        status(plan.centers.size(), "failed to allocate CUDA drift status") {
    centers.upload(plan.centers, stream, "failed to upload CUDA drift centers");
    axes.upload(plan.axes, stream, "failed to upload CUDA drift axes");
    fixed.upload(plan.fixed, stream, "failed to upload CUDA drift fixed flags");
    solid.upload(plan.solid, stream, "failed to upload CUDA drift obstacles");
    x_faces.upload(plan.field->x_faces, stream, "failed to upload CUDA drift x faces");
    y_faces.upload(plan.field->y_faces, stream, "failed to upload CUDA drift y faces");
    z_faces.upload(plan.field->z_faces, stream, "failed to upload CUDA drift z faces");
    status.zero(stream, "failed to clear CUDA drift status");
  }

  DeviceArray<float4> centers;
  DeviceArray<float4> axes;
  DeviceArray<unsigned char> fixed;
  DeviceArray<unsigned char> solid;
  DeviceArray<float> x_faces;
  DeviceArray<float> y_faces;
  DeviceArray<float> z_faces;
  DeviceArray<unsigned int> status;
};

}  // namespace

std::vector<std::array<float, 6>> flow_drift_rates(const WorldState& state, const SignalGrid& grid,
                                                   float dt, cudaStream_t stream) {
  const auto plan = detail::plan_flow_drift(state, grid, dt, 0.0F);
  const auto cells = plan.data.cell_count;
  const auto data = as_device<kernels::FlowDriftData>(plan.data);
  DriftInputs inputs(plan, stream);
  DeviceArray<float> rates(6 * plan.centers.size(), "failed to allocate CUDA drift rates");
  if (cells > 0) {
    drift_rates<<<blocks(cells), block_size, 0, stream>>>(
        data, inputs.centers.data(), inputs.axes.data(), inputs.fixed.data(), inputs.solid.data(),
        inputs.x_faces.data(), inputs.y_faces.data(), inputs.z_faces.data(), rates.data(),
        inputs.status.data());
    check_launch("failed to launch the CUDA drift-rates kernel");
  }
  detail::check_flow_drift_status(
      inputs.status.download<std::uint32_t>(cells, stream, "failed to download CUDA drift status"));
  const auto host_rates = rates.download<float>(6 * static_cast<std::size_t>(cells), stream,
                                                "failed to download CUDA drift rates");
  return detail::unpack_flow_drift_rates(host_rates, cells);
}

void advance_flow_drift(WorldState& state, const SignalGrid& grid, float dt,
                        const FlowDriftParameters& parameters, cudaStream_t stream) {
  const auto plan = detail::plan_flow_drift(state, grid, dt, parameters.max_rotation_radians);
  const auto cells = plan.data.cell_count;
  const auto data = as_device<kernels::FlowDriftData>(plan.data);
  DriftInputs inputs(plan, stream);
  DeviceArray<float4> moved_centers(plan.centers.size(), "failed to allocate CUDA drift centers");
  DeviceArray<float4> moved_axes(plan.axes.size(), "failed to allocate CUDA drift axes");
  if (cells > 0) {
    drift_advance<<<blocks(cells), block_size, 0, stream>>>(
        data, inputs.centers.data(), inputs.axes.data(), inputs.fixed.data(), inputs.solid.data(),
        inputs.x_faces.data(), inputs.y_faces.data(), inputs.z_faces.data(), moved_centers.data(),
        moved_axes.data(), inputs.status.data());
    check_launch("failed to launch the CUDA drift-advance kernel");
  }
  detail::check_flow_drift_status(
      inputs.status.download<std::uint32_t>(cells, stream, "failed to download CUDA drift status"));
  const auto centers = moved_centers.download<microsimulator::device::float4>(
      cells, stream, "failed to download CUDA drift centers");
  const auto axes = moved_axes.download<microsimulator::device::float4>(
      cells, stream, "failed to download CUDA drift axes");
  detail::apply_flow_drift_moves(state, centers, axes);
}

}  // namespace microsimulator::cuda
