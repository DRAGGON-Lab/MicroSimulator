#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <vector>

namespace cm::detail {

// Host orchestration and graph packing are shared; each backend implements all
// numerical kernels with device-resident buffers. No CPU numerical fallback.
enum class OccupancyKernel {
  geometry,
  volumes,
  concentration,
  face,
  product,
  sum,
  normalize,
  labels_init,
  labels_step,
  component_sums,
  remap,
  assemble,
  jacobi,
  residual,
  finish
};
inline constexpr const char* occupancy_kernel_names[] = {
    "occupancy_geometry",       "occupancy_volumes",     "occupancy_concentration",
    "occupancy_face",           "occupancy_product",     "occupancy_sum",
    "occupancy_normalize",      "occupancy_labels_init", "occupancy_labels_step",
    "occupancy_component_sums", "occupancy_remap",       "occupancy_assemble",
    "occupancy_jacobi",         "occupancy_residual",    "occupancy_finish"};

struct OccupancyParameters {
  std::uint32_t count{0}, auxiliary{0}, subdivisions{0}, absolute{0};
  float scalar{0}, cutoff{0}, hx{0}, hy{0}, hz{0};
};
static_assert(sizeof(OccupancyParameters) == 36);

struct OccupancyBuffer {
  virtual ~OccupancyBuffer() = default;
};
using OccupancyBufferPtr = std::shared_ptr<OccupancyBuffer>;

class OccupancyDevice {
 public:
  virtual ~OccupancyDevice() = default;
  virtual OccupancyBufferPtr allocate(std::size_t bytes, const void* data = nullptr) = 0;
  virtual void read(const OccupancyBufferPtr& buffer, void* data, std::size_t bytes) = 0;
  // Parameters always occupy binding 15; buffers occupy consecutive bindings.
  virtual void dispatch(OccupancyKernel kernel, const OccupancyParameters& parameters,
                        std::initializer_list<OccupancyBufferPtr> buffers) = 0;

  template <class T>
  OccupancyBufferPtr upload(const std::vector<T>& values) {
    return allocate(values.size() * sizeof(T), values.empty() ? nullptr : values.data());
  }
  template <class T>
  std::vector<T> download(const OccupancyBufferPtr& buffer, std::size_t count) {
    std::vector<T> result(count);
    if (count) read(buffer, result.data(), count * sizeof(T));
    return result;
  }
};

std::unique_ptr<OccupancyDevice> make_metal_occupancy_device(std::uint32_t device_index);
std::unique_ptr<OccupancyDevice> make_cuda_occupancy_device(std::uint32_t device_index);

}  // namespace cm::detail
