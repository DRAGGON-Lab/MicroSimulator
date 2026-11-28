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

#include "core/deposition_plan.hpp"
#include "cuda_deposition.cuh"
#include "device_types.cuh"

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

void check_launch(const char* operation) {
  check_cuda(cudaGetLastError(), operation);
}

}  // namespace

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
  // The inclusive prefix sum of the counts gives each site its range of
  // entries.
  DeviceArray<unsigned int> ends_array(site_count, "failed to allocate CUDA deposit ends");
  unsigned int* ends = ends_array.data();
  std::size_t scan_bytes = 0;
  check_cuda(cub::DeviceScan::InclusiveSum(nullptr, scan_bytes, counts.data(), ends,
                                           static_cast<int>(site_count), stream),
             "failed to size the CUDA deposit scan");
  DeviceArray<unsigned char> scan_storage(scan_bytes, "failed to allocate CUDA deposit scan");
  check_cuda(cub::DeviceScan::InclusiveSum(scan_storage.data(), scan_bytes, counts.data(), ends,
                                           static_cast<int>(site_count), stream),
             "failed to run the CUDA deposit scan");
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

}  // namespace microsimulator::cuda
