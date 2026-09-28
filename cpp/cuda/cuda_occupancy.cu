#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>

#include "core/occupancy_device.hpp"
#include "kernels/occupancy.cuh"

namespace cm::detail {
namespace {
void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
class DeviceScope {
 public:
  explicit DeviceScope(int index) {
    check(cudaGetDevice(&previous_), "get CUDA occupancy device");
    check(cudaSetDevice(index), "select CUDA occupancy device");
  }
  ~DeviceScope() { cudaSetDevice(previous_); }

 private:
  int previous_;
};
struct CudaOccupancyBuffer final : OccupancyBuffer {
  void* value{nullptr};
  int index;
  explicit CudaOccupancyBuffer(int device) : index(device) {}
  ~CudaOccupancyBuffer() override {
    int previous = 0;
    if (cudaGetDevice(&previous) == cudaSuccess) {
      if (cudaSetDevice(index) == cudaSuccess) cudaFree(value);
      cudaSetDevice(previous);
    }
  }
};
class CudaOccupancyDevice final : public OccupancyDevice {
 public:
  explicit CudaOccupancyDevice(std::uint32_t index) {
    int count = 0;
    check(cudaGetDeviceCount(&count), "enumerate CUDA occupancy devices");
    if (index >= static_cast<std::uint32_t>(count) || index > std::numeric_limits<int>::max())
      throw std::out_of_range("CUDA occupancy device index is unavailable");
    index_ = static_cast<int>(index);
    DeviceScope selected(index_);
    check(cudaStreamCreate(&stream_), "create CUDA occupancy stream");
  }
  ~CudaOccupancyDevice() override {
    int previous = 0;
    if (cudaGetDevice(&previous) == cudaSuccess) {
      if (cudaSetDevice(index_) == cudaSuccess) cudaStreamDestroy(stream_);
      cudaSetDevice(previous);
    }
  }
  OccupancyBufferPtr allocate(std::size_t bytes, const void* data) override {
    DeviceScope selected(index_);
    auto result = std::make_shared<CudaOccupancyBuffer>(index_);
    check(cudaMalloc(&result->value, std::max(bytes, std::size_t{4})),
          "allocate CUDA occupancy buffer");
    if (data && bytes) {
      check(cudaMemcpyAsync(result->value, data, bytes, cudaMemcpyHostToDevice, stream_),
            "upload CUDA occupancy buffer");
      check(cudaStreamSynchronize(stream_), "finish CUDA occupancy upload");
    }
    return result;
  }
  void read(const OccupancyBufferPtr& buffer, void* data, std::size_t bytes) override {
    DeviceScope selected(index_);
    check(cudaMemcpyAsync(data, static_cast<CudaOccupancyBuffer&>(*buffer).value, bytes,
                          cudaMemcpyDeviceToHost, stream_),
          "read CUDA occupancy buffer");
    check(cudaStreamSynchronize(stream_), "finish CUDA occupancy read");
  }
  void dispatch(OccupancyKernel kernel, const OccupancyParameters& p,
                std::initializer_list<OccupancyBufferPtr> buffers) override {
    if (!p.count) return;
    DeviceScope selected(index_);
    std::array<void*, 15> pointers{};
    std::size_t i = 0;
    for (const auto& buffer : buffers)
      pointers[i++] = static_cast<CudaOccupancyBuffer&>(*buffer).value;
    cuda::launch_occupancy(kernel, p, pointers.data(), stream_);
    check(cudaGetLastError(), "launch CUDA occupancy kernel");
    check(cudaStreamSynchronize(stream_), "execute CUDA occupancy kernel");
  }

 private:
  int index_;
  cudaStream_t stream_{};
};
}  // namespace
std::unique_ptr<OccupancyDevice> make_cuda_occupancy_device(std::uint32_t index) {
  return std::make_unique<CudaOccupancyDevice>(index);
}
}  // namespace cm::detail
