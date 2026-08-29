#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace cm {

using CellId = std::uint64_t;
using Slot = std::uint32_t;

inline constexpr CellId invalid_cell_id = 0;
inline constexpr Slot invalid_slot = std::numeric_limits<Slot>::max();

struct Vec3 {
  float x{0.0F};
  float y{0.0F};
  float z{0.0F};

  [[nodiscard]] constexpr Vec3 operator+(const Vec3& other) const noexcept {
    return {x + other.x, y + other.y, z + other.z};
  }

  [[nodiscard]] constexpr Vec3 operator-(const Vec3& other) const noexcept {
    return {x - other.x, y - other.y, z - other.z};
  }

  [[nodiscard]] constexpr Vec3 operator*(float scale) const noexcept {
    return {x * scale, y * scale, z * scale};
  }
};

[[nodiscard]] inline constexpr float dot(const Vec3& left, const Vec3& right) noexcept {
  return (left.x * right.x) + (left.y * right.y) + (left.z * right.z);
}

[[nodiscard]] inline constexpr Vec3 cross(const Vec3& left, const Vec3& right) noexcept {
  return {
      (left.y * right.z) - (left.z * right.y),
      (left.z * right.x) - (left.x * right.z),
      (left.x * right.y) - (left.y * right.x),
  };
}

[[nodiscard]] inline float norm(const Vec3& value) noexcept { return std::sqrt(dot(value, value)); }

[[nodiscard]] inline Vec3 normalized(const Vec3& value) {
  const auto magnitude = norm(value);
  if (!std::isfinite(magnitude) || magnitude <= 0.0F) {
    throw std::invalid_argument("cell direction must be finite and non-zero");
  }
  return value * (1.0F / magnitude);
}

enum class BackendKind : std::uint8_t {
  cpu,
  metal,
  cuda,
};

enum class BackendFeature : std::uint8_t {
  growth,
  species,
  cell_contacts,
  cell_mechanics,
  external_constraints,
  signals,
  coupled_rates,
  depth_averaged_flow,
  resolved_flow,
};

struct BackendInfo {
  BackendKind kind{BackendKind::cpu};
  std::string name;
  std::string device;
  std::uint32_t device_index{0};
  bool native{false};
};

}  // namespace cm
