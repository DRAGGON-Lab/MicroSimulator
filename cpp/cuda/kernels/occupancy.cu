#include <cmath>

#include "occupancy.cuh"

namespace cm::cuda {
namespace {
using uint = unsigned int;
using detail::OccupancyParameters;
struct Vector3 {
  float x, y, z;
  __device__ Vector3(float a, float b, float c) : x(a), y(b), z(c) {}
  __device__ Vector3 operator+(Vector3 v) const { return {x + v.x, y + v.y, z + v.z}; }
  __device__ Vector3 operator+(float v) const { return {x + v, y + v, z + v}; }
  __device__ Vector3 operator-(Vector3 v) const { return {x - v.x, y - v.y, z - v.z}; }
  __device__ Vector3 operator*(Vector3 v) const { return {x * v.x, y * v.y, z * v.z}; }
  __device__ Vector3 operator*(float v) const { return {x * v, y * v, z * v}; }
  __device__ Vector3 operator/(float v) const { return {x / v, y / v, z / v}; }
  __device__ void operator/=(float v) {
    x /= v;
    y /= v;
    z /= v;
  }
};
__device__ Vector3 operator*(float a, Vector3 b) { return b * a; }
__device__ float dot(Vector3 a, Vector3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ float clamp(float x, float lo, float hi) { return fminf(hi, fmaxf(lo, x)); }
// Compensated, ordered sums avoid nondeterministic floating-point atomics.
__device__ void occupancy_add(float value, float& total, float& correction) {
  float adjusted = value - correction;
  float next = total + adjusted;
  correction = (next - total) - adjusted;
  total = next;
}

__global__ void occupancy_geometry(const float* centers, const float* cells, const uint* walls,
                                   float* out, OccupancyParameters p) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.count) return;
  if (walls[i]) {
    out[i] = 0;
    return;
  }
  uint occupied = 0;
  Vector3 center(centers[3 * i], centers[3 * i + 1], centers[3 * i + 2]);
  for (uint x = 0; x < p.subdivisions; ++x)
    for (uint y = 0; y < p.subdivisions; ++y)
      for (uint z = 0; z < p.subdivisions; ++z) {
        Vector3 sample =
            center + (Vector3(x, y, z) + 0.5f) / float(p.subdivisions) * Vector3(p.hx, p.hy, p.hz) -
            0.5f * Vector3(p.hx, p.hy, p.hz);
        for (uint cell = 0; cell < p.auxiliary; ++cell) {
          uint base = 8 * cell;
          Vector3 delta = sample - Vector3(cells[base], cells[base + 1], cells[base + 2]);
          Vector3 direction(cells[base + 4], cells[base + 5], cells[base + 6]);
          // Scaling first avoids overflow/underflow in direction normalization.
          direction /= max(abs(direction.x), max(abs(direction.y), abs(direction.z)));
          direction /= sqrt(dot(direction, direction));
          float axial =
              clamp(dot(delta, direction), -0.5f * cells[base + 3], 0.5f * cells[base + 3]);
          Vector3 distance = delta - axial * direction;
          float radius = cells[base + 7];
          distance /= radius;
          if (dot(distance, distance) <= 1) {
            ++occupied;
            break;
          }
        }
      }
  uint samples = p.subdivisions * p.subdivisions * p.subdivisions;
  float epsilon = float(samples - occupied) / float(samples);
  out[i] = epsilon < p.cutoff ? 0 : epsilon;
}

__global__ void occupancy_volumes(const float* epsilon, float* out, OccupancyParameters p) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < p.count) out[i] = epsilon[i] < p.cutoff ? 0 : epsilon[i] * p.scalar;
}
__global__ void occupancy_concentration(const float* amount, const float* volume, float* out,
                                        OccupancyParameters p) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < p.count) out[i] = volume[i] > 0 ? amount[i] / volume[i] : 0;
}
__global__ void occupancy_face(const float* data, float* out, OccupancyParameters p) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i != 0) return;
  float aperture =
      min(data[0], data[1]) < p.cutoff ? 0 : 2 * data[0] * data[1] / (data[0] + data[1]);
  out[0] = data[2] * aperture * data[3] / data[4];
  out[1] = aperture * data[3] * data[5];
}
__global__ void occupancy_product(const float* a, const float* b, float* out,
                                  OccupancyParameters p) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < p.count) out[i] = a[i] * b[i];
}
__global__ void occupancy_sum(const float* input, float* out, OccupancyParameters p) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i != 0) return;
  float total = 0, correction = 0;
  for (uint j = 0; j < p.count; ++j)
    occupancy_add(p.absolute ? abs(input[j]) : input[j], total, correction);
  out[0] = total;
}
__global__ void occupancy_normalize(const float* values, const float* total, float* out,
                                    OccupancyParameters p) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < p.count) out[i] = values[i] / total[0];
}
__global__ void occupancy_labels_init(const float* old, const float* next, uint* out,
                                      OccupancyParameters p) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < p.count) out[i] = old[i] > 0 || next[i] > 0 ? i : 0xffffffffu;
}
__global__ void occupancy_labels_step(const uint* offsets, const uint* indices, const uint* labels,
                                      uint* out, OccupancyParameters p) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.count) return;
  uint label = labels[i];
  if (label != 0xffffffffu)
    for (uint j = offsets[i]; j < offsets[i + 1]; ++j) label = min(label, labels[indices[j]]);
  out[i] = label;
}
__global__ void occupancy_component_sums(const uint* offsets, const uint* indices,
                                         const float* amount, const float* volume, float* out,
                                         OccupancyParameters p) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.count) return;
  float expelled = 0, capacity = 0, ec = 0, vc = 0;
  for (uint entry = offsets[i]; entry < offsets[i + 1]; ++entry) {
    uint j = indices[entry];
    if (volume[j] == 0)
      occupancy_add(amount[j], expelled, ec);
    else
      occupancy_add(volume[j], capacity, vc);
  }
  out[2 * i] = expelled;
  out[2 * i + 1] = capacity;
}
__global__ void occupancy_remap(const float* amount, const float* volume, const uint* labels,
                                const float* totals, float* out, OccupancyParameters p) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.count) return;
  if (volume[i] == 0) {
    out[i] = 0;
    return;
  }
  uint component = labels[i];
  out[i] = amount[i] + totals[2 * component] * (volume[i] / totals[2 * component + 1]);
}

__global__ void occupancy_assemble(const uint* offsets, const uint* indices, const float* edges,
                                   const float* volume, const float* amount, const float* source,
                                   const float* loss, float* diagonal, float* rhs,
                                   OccupancyParameters p) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.count) return;
  float d = volume[i] * (1 + p.scalar * loss[i]);
  float b = amount[i] + p.scalar * source[i];
  for (uint j = offsets[i]; j < offsets[i + 1]; ++j) {
    float g = edges[3 * j], q = edges[3 * j + 1];
    d += p.scalar * (g + max(q, 0.0f));
    if (indices[j] == 0xffffffffu) b += p.scalar * (g + max(-q, 0.0f)) * edges[3 * j + 2];
  }
  diagonal[i] = volume[i] > 0 ? d : 1;
  rhs[i] = b;
}
__global__ void occupancy_jacobi(const float* diagonal, const float* current, const float* residual,
                                 float* out, OccupancyParameters p) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.count) return;
  out[i] = current[i] - residual[i] / diagonal[i];
}
__global__ void occupancy_residual(const uint* offsets, const uint* indices, const float* edges,
                                   const float* volume, const float* amount, const float* source,
                                   const float* loss, const float* current, float* out,
                                   OccupancyParameters p) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.count) return;
  float c = current[i];
  float value = volume[i] * c - amount[i], correction = 0;
  occupancy_add(p.scalar * (loss[i] * volume[i] * c - source[i]), value, correction);
  for (uint j = offsets[i]; j < offsets[i + 1]; ++j) {
    float neighbor = indices[j] == 0xffffffffu ? edges[3 * j + 2] : current[indices[j]];
    float g = edges[3 * j], q = edges[3 * j + 1];
    occupancy_add(p.scalar * (g * (c - neighbor) + q * (q >= 0 ? c : neighbor)), value, correction);
  }
  out[i] = value;
}
__global__ void occupancy_finish(const uint* offsets, const uint* indices, const float* edges,
                                 const float* volume, const float* loss, const float* current,
                                 float* amount, float* reaction, float* boundary,
                                 OccupancyParameters p) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.count) return;
  amount[i] = volume[i] * current[i];
  reaction[i] = -p.scalar * loss[i] * amount[i];
  float exchange = 0, correction = 0;
  for (uint j = offsets[i]; j < offsets[i + 1]; ++j)
    if (indices[j] == 0xffffffffu) {
      float g = edges[3 * j], q = edges[3 * j + 1], reservoir = edges[3 * j + 2];
      occupancy_add(
          p.scalar * (g * (reservoir - current[i]) - q * (q >= 0 ? current[i] : reservoir)),
          exchange, correction);
    }
  boundary[i] = exchange;
}
}  // namespace

void launch_occupancy(detail::OccupancyKernel kernel, const detail::OccupancyParameters& p,
                      void* const* buffers, cudaStream_t stream) {
  const unsigned count = kernel == detail::OccupancyKernel::sum ? 1 : p.count;
  const unsigned blocks = (count - 1) / 128 + 1;
  switch (kernel) {
    case detail::OccupancyKernel::geometry:
      occupancy_geometry<<<blocks, 128, 0, stream>>>(
          static_cast<const float*>(buffers[0]), static_cast<const float*>(buffers[1]),
          static_cast<const uint*>(buffers[2]), static_cast<float*>(buffers[3]), p);
      break;
    case detail::OccupancyKernel::volumes:
      occupancy_volumes<<<blocks, 128, 0, stream>>>(static_cast<const float*>(buffers[0]),
                                                    static_cast<float*>(buffers[1]), p);
      break;
    case detail::OccupancyKernel::concentration:
      occupancy_concentration<<<blocks, 128, 0, stream>>>(static_cast<const float*>(buffers[0]),
                                                          static_cast<const float*>(buffers[1]),
                                                          static_cast<float*>(buffers[2]), p);
      break;
    case detail::OccupancyKernel::face:
      occupancy_face<<<blocks, 128, 0, stream>>>(static_cast<const float*>(buffers[0]),
                                                 static_cast<float*>(buffers[1]), p);
      break;
    case detail::OccupancyKernel::product:
      occupancy_product<<<blocks, 128, 0, stream>>>(static_cast<const float*>(buffers[0]),
                                                    static_cast<const float*>(buffers[1]),
                                                    static_cast<float*>(buffers[2]), p);
      break;
    case detail::OccupancyKernel::sum:
      occupancy_sum<<<blocks, 128, 0, stream>>>(static_cast<const float*>(buffers[0]),
                                                static_cast<float*>(buffers[1]), p);
      break;
    case detail::OccupancyKernel::normalize:
      occupancy_normalize<<<blocks, 128, 0, stream>>>(static_cast<const float*>(buffers[0]),
                                                      static_cast<const float*>(buffers[1]),
                                                      static_cast<float*>(buffers[2]), p);
      break;
    case detail::OccupancyKernel::labels_init:
      occupancy_labels_init<<<blocks, 128, 0, stream>>>(static_cast<const float*>(buffers[0]),
                                                        static_cast<const float*>(buffers[1]),
                                                        static_cast<uint*>(buffers[2]), p);
      break;
    case detail::OccupancyKernel::labels_step:
      occupancy_labels_step<<<blocks, 128, 0, stream>>>(
          static_cast<const uint*>(buffers[0]), static_cast<const uint*>(buffers[1]),
          static_cast<const uint*>(buffers[2]), static_cast<uint*>(buffers[3]), p);
      break;
    case detail::OccupancyKernel::component_sums:
      occupancy_component_sums<<<blocks, 128, 0, stream>>>(
          static_cast<const uint*>(buffers[0]), static_cast<const uint*>(buffers[1]),
          static_cast<const float*>(buffers[2]), static_cast<const float*>(buffers[3]),
          static_cast<float*>(buffers[4]), p);
      break;
    case detail::OccupancyKernel::remap:
      occupancy_remap<<<blocks, 128, 0, stream>>>(
          static_cast<const float*>(buffers[0]), static_cast<const float*>(buffers[1]),
          static_cast<const uint*>(buffers[2]), static_cast<const float*>(buffers[3]),
          static_cast<float*>(buffers[4]), p);
      break;
    case detail::OccupancyKernel::assemble:
      occupancy_assemble<<<blocks, 128, 0, stream>>>(
          static_cast<const uint*>(buffers[0]), static_cast<const uint*>(buffers[1]),
          static_cast<const float*>(buffers[2]), static_cast<const float*>(buffers[3]),
          static_cast<const float*>(buffers[4]), static_cast<const float*>(buffers[5]),
          static_cast<const float*>(buffers[6]), static_cast<float*>(buffers[7]),
          static_cast<float*>(buffers[8]), p);
      break;
    case detail::OccupancyKernel::jacobi:
      occupancy_jacobi<<<blocks, 128, 0, stream>>>(
          static_cast<const float*>(buffers[0]), static_cast<const float*>(buffers[1]),
          static_cast<const float*>(buffers[2]), static_cast<float*>(buffers[3]), p);
      break;
    case detail::OccupancyKernel::residual:
      occupancy_residual<<<blocks, 128, 0, stream>>>(
          static_cast<const uint*>(buffers[0]), static_cast<const uint*>(buffers[1]),
          static_cast<const float*>(buffers[2]), static_cast<const float*>(buffers[3]),
          static_cast<const float*>(buffers[4]), static_cast<const float*>(buffers[5]),
          static_cast<const float*>(buffers[6]), static_cast<const float*>(buffers[7]),
          static_cast<float*>(buffers[8]), p);
      break;
    case detail::OccupancyKernel::finish:
      occupancy_finish<<<blocks, 128, 0, stream>>>(
          static_cast<const uint*>(buffers[0]), static_cast<const uint*>(buffers[1]),
          static_cast<const float*>(buffers[2]), static_cast<const float*>(buffers[3]),
          static_cast<const float*>(buffers[4]), static_cast<const float*>(buffers[5]),
          static_cast<float*>(buffers[6]), static_cast<float*>(buffers[7]),
          static_cast<float*>(buffers[8]), p);
      break;
  }
}
}  // namespace cm::cuda
