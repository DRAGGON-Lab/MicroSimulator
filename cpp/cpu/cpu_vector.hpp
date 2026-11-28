#pragma once

#include <cmath>

// Three-component single-precision vectors for the CPU kernels, with the
// operators and functions of the Metal and CUDA vector types they mirror.
namespace microsimulator::device {

struct Vec3f {
  float x;
  float y;
  float z;
};

inline Vec3f operator+(Vec3f a, Vec3f b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline Vec3f operator-(Vec3f a, Vec3f b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline Vec3f operator*(Vec3f a, float scale) {
  return {a.x * scale, a.y * scale, a.z * scale};
}

inline Vec3f operator/(Vec3f a, float scale) {
  return {a.x / scale, a.y / scale, a.z / scale};
}

inline float dot(Vec3f a, Vec3f b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline float length(Vec3f a) {
  return std::sqrt(dot(a, a));
}

inline Vec3f cross(Vec3f a, Vec3f b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

}  // namespace microsimulator::device
