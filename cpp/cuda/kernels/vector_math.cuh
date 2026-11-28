#pragma once

#include <cuda_runtime.h>

// Arithmetic on CUDA's float3, which the runtime declares without operators.
namespace microsimulator::cuda::device {

__device__ inline float3 operator+(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 operator-(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 operator*(float3 a, float scale) {
  return make_float3(a.x * scale, a.y * scale, a.z * scale);
}

__device__ inline float3 operator/(float3 a, float scale) {
  return make_float3(a.x / scale, a.y / scale, a.z / scale);
}

__device__ inline float dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float length(float3 a) {
  return sqrtf(dot(a, a));
}

__device__ inline float3 cross(float3 a, float3 b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

}  // namespace microsimulator::cuda::device
