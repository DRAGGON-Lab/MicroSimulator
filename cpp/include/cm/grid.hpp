#pragma once
#include <cstdint>
#include <vector>

namespace cm {
enum class FlowAxis : std::uint8_t {
  x,
  y,
  z,
};

struct GridShape {
  std::uint32_t x{1};
  std::uint32_t y{1};
  std::uint32_t z{1};
};

struct MacVelocityField {
  std::vector<float> x_faces;
  std::vector<float> y_faces;
  std::vector<float> z_faces;
};
}  // namespace cm
