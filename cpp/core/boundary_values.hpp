#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "microsimulator/signals.hpp"

namespace microsimulator::detail {

// The reservoir concentrations of a grid's fixed boundaries as a device reads
// them. `values` begins with one concentration per signal for each of the six
// faces, in GridFace order. A face with patches then has a block of its own,
// one concentration per signal for every boundary site, with the site's first
// in-plane index varying slowest. `offsets[face]` is where that block begins,
// and zero for a face that carries the same concentrations everywhere.
struct FixedBoundaryTable {
  std::array<std::uint32_t, 6> offsets{};
  std::vector<float> values;
};

[[nodiscard]] inline FixedBoundaryTable fixed_boundary_table(const SignalGridSpec& spec) {
  constexpr std::array<GridFace, 6> faces{GridFace::x_lower, GridFace::x_upper, GridFace::y_lower,
                                          GridFace::y_upper, GridFace::z_lower, GridFace::z_upper};
  const std::size_t signal_count = spec.signal_count;
  FixedBoundaryTable table;
  table.values.assign(faces.size() * signal_count, 0.0F);
  for (std::size_t face = 0; face < faces.size(); ++face) {
    const auto& boundary = spec.boundary(faces[face]);
    if (boundary.kind != GridBoundaryKind::fixed) {
      continue;
    }
    std::copy(boundary.values.begin(), boundary.values.end(),
              table.values.begin() + static_cast<std::ptrdiff_t>(face * signal_count));
    if (boundary.patches.empty()) {
      continue;
    }
    const auto extent = spec.face_extent(faces[face]);
    const auto offset = table.values.size();
    const auto sites = static_cast<std::size_t>(extent[0]) * extent[1];
    if (offset + sites * signal_count > std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("signal grid boundary values exceed the uint32 index space");
    }
    table.offsets[face] = static_cast<std::uint32_t>(offset);
    table.values.resize(offset + sites * signal_count);
    for (std::uint32_t first = 0; first < extent[0]; ++first) {
      for (std::uint32_t second = 0; second < extent[1]; ++second) {
        const auto held = boundary.values_at(first, second);
        const auto site = static_cast<std::size_t>(first) * extent[1] + second;
        std::copy(held.begin(), held.end(),
                  table.values.begin() + static_cast<std::ptrdiff_t>(offset + site * signal_count));
      }
    }
  }
  return table;
}

}  // namespace microsimulator::detail
