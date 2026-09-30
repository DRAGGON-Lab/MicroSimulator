#pragma once

#include <limits>
#include <memory>

#include "cm/stokes_flow.hpp"

namespace cm {
inline constexpr std::uint32_t fluid_boundary = std::numeric_limits<std::uint32_t>::max();

struct FluidGeometryParameters {
  std::uint32_t surface_resolution{32};
  double maximum_surface_error_fraction{0.05};
  std::uint64_t memory_limit_bytes{1024ULL * 1024 * 1024};
  void validate() const;
};

struct FluidFragment {
  std::uint32_t site{0}, component{0};
  double volume{0};
  std::array<double, 3> centroid{};
};

struct FluidFace {
  std::uint32_t first{0}, second{fluid_boundary};
  double area{0};
  std::array<double, 3> centroid{}, normal{};  // normal out of first
  // Nonzero for a cell membrane, zero for a grid wall or open device boundary.
  CellId body_id{0};
  // Grid faces use the MAC axis/index. Internal partition and membrane faces
  // have grid_face == fluid_boundary.
  FlowAxis axis{FlowAxis::x};
  std::uint32_t grid_face{fluid_boundary};
};

struct FluidGeometryReport {
  double fluid_volume{0}, expected_fluid_volume{0}, volume_error{0};
  double maximum_surface_error{0};
  std::uint32_t component_count{0};
  std::uint64_t estimated_memory_bytes{0};
};

struct FluidOverlap {
  std::uint32_t first{0}, second{0};
  double volume{0};
};

// Immutable extracellular polyhedral mesh. Volumes, areas and coordinates use
// model units. Convex pieces retain their internal faces, so disconnected fluid
// fragments within a voxel can never be mixed merely by sharing a voxel index.
class FluidGeometry {
 public:
  FluidGeometry(const FluidGridSpec& grid, const std::vector<CapsuleBody>& bodies,
                const FluidGeometryParameters& parameters = {});
  [[nodiscard]] const FluidGridSpec& grid() const noexcept;
  [[nodiscard]] const std::vector<CapsuleBody>& bodies() const noexcept;
  [[nodiscard]] const std::vector<FluidFragment>& fragments() const noexcept;
  [[nodiscard]] const std::vector<FluidFace>& faces() const noexcept;
  [[nodiscard]] const FluidGeometryReport& report() const noexcept;
  [[nodiscard]] std::vector<FluidOverlap> overlaps(const FluidGeometry& other) const;

 private:
  struct Impl;
  std::shared_ptr<const Impl> impl_;
};
}  // namespace cm
