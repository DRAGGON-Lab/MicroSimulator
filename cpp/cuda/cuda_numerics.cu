#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>

#include "core/numerics_device.hpp"

namespace cm::detail {
namespace {

__global__ void conservative_apply(const unsigned* offsets, const unsigned* columns,
                                   const float* values, const float* x, float* y,
                                   const float* row_sums, NumericsParameters p) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= p.count) {
    return;
  }

  float total = 0, correction = 0;

  for (unsigned j = offsets[i]; j < offsets[i + 1]; ++j) {
    const float term = values[j] * (x[columns[j]] - x[i]) - correction;
    const float next = total + term;
    correction = (next - total) - term;
    total = next;
  }

  y[i] = total + row_sums[i] * x[i];
}

__global__ void nutrient_growth(const unsigned* offsets, const float* cells,
                                const float* requirements, const float* concentration,
                                const float* uptake, float* alpha, float* output,
                                NumericsParameters p) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= p.count) {
    return;
  }

  const unsigned begin = offsets[i], end = offsets[i + 1];
  const float mass = cells[6 * i], mu = cells[6 * i + 1], area = cells[6 * i + 2];
  float limitation = 1.0f, extent = INFINITY, largest = 0.0f;

  for (unsigned j = begin; j < end; ++j) {
    const float c = concentration[j], k = requirements[2 * j], y = requirements[2 * j + 1];
    limitation = fminf(limitation, c / (k + c));
    extent = fminf(extent, uptake[j] * y);
    largest = fmaxf(largest, uptake[j] * y);
  }

  for (unsigned j = begin; j < end; ++j) {
    const float c = concentration[j], k = requirements[2 * j], y = requirements[2 * j + 1];
    alpha[j] = end - begin == 1 ? mu * mass / (y * area * (k + c))
                                : (c > 0 ? mu * mass * limitation / (y * area * c) : 0.0f);
  }

  output[5 * i] = extent;
  output[5 * i + 1] = extent / cells[6 * i + 3];
  output[5 * i + 2] = output[5 * i + 1] / cells[6 * i + 4];
  output[5 * i + 3] = extent / (mass * cells[6 * i + 5]);
  output[5 * i + 4] = largest > 0 ? (largest - extent) / largest : 0.0f;
}

__global__ void numerics_ilu(const unsigned* offsets, const unsigned* columns, const float* values,
                             const unsigned* diagonal, const float* rhs, float* x,
                             NumericsParameters p) {
  if (blockIdx.x || threadIdx.x) {
    return;
  }

  for (unsigned i = 0; i < p.count; ++i) {
    float value = rhs[i];

    for (unsigned j = offsets[i]; j < diagonal[i]; ++j) {
      value -= values[j] * x[columns[j]];
    }

    x[i] = value;
  }

  for (unsigned i = p.count; i-- > 0;) {
    float value = x[i];

    for (unsigned j = diagonal[i] + 1; j < offsets[i + 1]; ++j) {
      value -= values[j] * x[columns[j]];
    }

    x[i] = value / values[diagonal[i]];
  }
}

__global__ void numerics_fill(float* y, NumericsParameters p) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < p.count) {
    y[i] = p.scalar;
  }
}

__global__ void numerics_copy(const float* x, float* y, NumericsParameters p) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < p.count) {
    y[i] = x[i];
  }
}

__global__ void numerics_axpy(const float* x, float* y, NumericsParameters p) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < p.count) {
    y[i] += p.scalar * x[i];
  }
}

__global__ void numerics_jacobi(const float* rhs, const float* ax, const float* inverse_diagonal,
                                float* x, NumericsParameters p) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < p.count) {
    x[i] += p.scalar * inverse_diagonal[i] * (rhs[i] - ax[i]);
  }
}

__global__ void numerics_dot(const float* x, const float* y, float* out, NumericsParameters p) {
  __shared__ float values[128];
  const auto local = threadIdx.x, i = blockIdx.x * blockDim.x + local;
  values[local] = i < p.count ? x[i] * y[i] : 0;
  __syncthreads();

  for (unsigned s = 64; s; s /= 2) {
    if (local < s) {
      values[local] += values[local + s];
    }

    __syncthreads();
  }

  if (!local) {
    out[blockIdx.x] = values[0];
  }
}

__global__ void numerics_apply(const unsigned* offsets, const unsigned* columns,
                               const float* values, const float* x, float* y,
                               NumericsParameters p) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= p.count) {
    return;
  }

  float total = 0, correction = 0;

  for (auto j = offsets[i]; j < offsets[i + 1]; ++j) {
    const float term = values[j] * x[columns[j]] - correction;
    const float next = total + term;
    correction = (next - total) - term;
    total = next;
  }

  y[i] = total;
}

void launch_numerics(NumericsKernel kernel, NumericsParameters p, void** b, cudaStream_t stream) {
  const auto groups = (p.count + 127U) / 128U;

  switch (kernel) {
    case NumericsKernel::conservative_apply:
      conservative_apply<<<groups, 128, 0, stream>>>((unsigned*)b[0], (unsigned*)b[1], (float*)b[2],
                                                     (float*)b[3], (float*)b[4], (float*)b[5], p);
      break;
    case NumericsKernel::growth:
      nutrient_growth<<<groups, 128, 0, stream>>>((unsigned*)b[0], (float*)b[1], (float*)b[2],
                                                  (float*)b[3], (float*)b[4], (float*)b[5],
                                                  (float*)b[6], p);
      break;
    case NumericsKernel::ilu:
      numerics_ilu<<<1, 1, 0, stream>>>((unsigned*)b[0], (unsigned*)b[1], (float*)b[2],
                                        (unsigned*)b[3], (float*)b[4], (float*)b[5], p);
      break;
    case NumericsKernel::fill:
      numerics_fill<<<groups, 128, 0, stream>>>((float*)b[0], p);
      break;
    case NumericsKernel::copy:
      numerics_copy<<<groups, 128, 0, stream>>>((float*)b[0], (float*)b[1], p);
      break;
    case NumericsKernel::axpy:
      numerics_axpy<<<groups, 128, 0, stream>>>((float*)b[0], (float*)b[1], p);
      break;
    case NumericsKernel::dot:
      numerics_dot<<<groups, 128, 0, stream>>>((float*)b[0], (float*)b[1], (float*)b[2], p);
      break;
    case NumericsKernel::apply:
      numerics_apply<<<groups, 128, 0, stream>>>((unsigned*)b[0], (unsigned*)b[1], (float*)b[2],
                                                 (float*)b[3], (float*)b[4], p);
      break;
    case NumericsKernel::jacobi:
      numerics_jacobi<<<groups, 128, 0, stream>>>((float*)b[0], (float*)b[1], (float*)b[2],
                                                  (float*)b[3], p);
      break;
  }
}

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

class DeviceScope {
 public:
  explicit DeviceScope(int index) {
    check(cudaGetDevice(&previous_), "get CUDA numerics device");
    check(cudaSetDevice(index), "select CUDA numerics device");
  }

  ~DeviceScope() {
    cudaSetDevice(previous_);
  }

 private:
  int previous_;
};

struct CudaNumericsBuffer final : NumericsBuffer {
  void* value{nullptr};
  int index;

  explicit CudaNumericsBuffer(int device) : index(device) {}

  ~CudaNumericsBuffer() override {
    int previous = 0;

    if (cudaGetDevice(&previous) == cudaSuccess) {
      if (cudaSetDevice(index) == cudaSuccess) {
        cudaFree(value);
      }

      cudaSetDevice(previous);
    }
  }
};

class CudaNumericsDevice final : public NumericsDevice {
 public:
  explicit CudaNumericsDevice(std::uint32_t index) {
    int count = 0;
    check(cudaGetDeviceCount(&count), "enumerate CUDA numerics devices");

    if (index >= static_cast<std::uint32_t>(count) || index > std::numeric_limits<int>::max()) {
      throw std::out_of_range("CUDA numerics device index is unavailable");
    }

    index_ = static_cast<int>(index);
    DeviceScope selected(index_);
    check(cudaStreamCreate(&stream_), "create CUDA numerics stream");
  }

  ~CudaNumericsDevice() override {
    int previous = 0;

    if (cudaGetDevice(&previous) == cudaSuccess) {
      if (cudaSetDevice(index_) == cudaSuccess) {
        cudaStreamDestroy(stream_);
      }

      cudaSetDevice(previous);
    }
  }

  NumericsBufferPtr allocate(std::size_t bytes, const void* data) override {
    DeviceScope selected(index_);
    auto result = std::make_shared<CudaNumericsBuffer>(index_);
    check(cudaMalloc(&result->value, std::max(bytes, std::size_t{4})),
          "allocate CUDA numerics buffer");

    if (data && bytes) {
      check(cudaMemcpyAsync(result->value, data, bytes, cudaMemcpyHostToDevice, stream_),
            "upload CUDA numerics buffer");
      check(cudaStreamSynchronize(stream_), "finish CUDA numerics upload");
    }

    return result;
  }

  void read(const NumericsBufferPtr& buffer, void* data, std::size_t bytes) override {
    DeviceScope selected(index_);
    check(cudaMemcpyAsync(data, static_cast<CudaNumericsBuffer&>(*buffer).value, bytes,
                          cudaMemcpyDeviceToHost, stream_),
          "read CUDA numerics buffer");
    check(cudaStreamSynchronize(stream_), "finish CUDA numerics read");
  }

  void dispatch(NumericsKernel kernel, const NumericsParameters& p,
                std::initializer_list<NumericsBufferPtr> buffers) override {
    if (!p.count) {
      return;
    }

    DeviceScope selected(index_);
    std::array<void*, 15> pointers{};
    std::size_t i = 0;

    for (const auto& buffer : buffers) {
      pointers[i++] = static_cast<CudaNumericsBuffer&>(*buffer).value;
    }

    launch_numerics(kernel, p, pointers.data(), stream_);
    check(cudaGetLastError(), "launch CUDA numerics kernel");
  }

 private:
  int index_;
  cudaStream_t stream_{};
};
}  // namespace

std::unique_ptr<NumericsDevice> make_cuda_numerics_device(std::uint32_t index) {
  return std::make_unique<CudaNumericsDevice>(index);
}
}  // namespace cm::detail
