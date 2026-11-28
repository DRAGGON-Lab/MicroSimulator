#include "core/flow_drift_plan.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "core/device_types.hpp"
#include "core/shared_constants.hpp"

namespace microsimulator::detail {

FlowDriftPlan plan_flow_drift(const WorldState& state, const SignalGrid& grid, float dt,
                              float max_rotation_radians) {
  const auto& spec = grid.spec();
  if (!spec.velocity_field.has_value()) {
    throw std::logic_error("flow drift requires a signal grid with a velocity field");
  }
  const auto geometry = state.geometry_state();
  const auto attributes = state.cell_attributes();
  if (geometry.size() > std::numeric_limits<std::uint32_t>::max() ||
      spec.site_count() > std::numeric_limits<std::uint32_t>::max() / 4) {
    throw std::overflow_error("flow drift exceeds the uint32 index space");
  }
  FlowDriftPlan plan;
  const auto spatial_step = 0.25F * std::min({spec.spacing.x, spec.spacing.y, spec.spacing.z});
  plan.data.shape = {spec.shape.x, spec.shape.y, spec.shape.z, spec.has_obstacles() ? 1U : 0U};
  plan.data.origin = {spec.origin.x, spec.origin.y, spec.origin.z, max_rotation_radians};
  plan.data.spacing = {spec.spacing.x, spec.spacing.y, spec.spacing.z, spatial_step};
  plan.data.cell_count = static_cast<std::uint32_t>(geometry.size());
  plan.data.dt = dt;
  const auto cells = std::max<std::size_t>(geometry.size(), 1);
  plan.centers.resize(cells);
  plan.axes.resize(cells);
  plan.fixed.assign(cells, 1);
  for (std::size_t slot = 0; slot < geometry.size(); ++slot) {
    plan.centers[slot] = {geometry.position_x[slot], geometry.position_y[slot],
                          geometry.position_z[slot], geometry.lengths[slot]};
    plan.axes[slot] = {geometry.direction_x[slot], geometry.direction_y[slot],
                       geometry.direction_z[slot], geometry.radii[slot]};
    plan.fixed[slot] = attributes.fixed[slot] != 0 ? 1 : 0;
  }
  if (spec.obstacles.empty()) {
    plan.solid.assign(1, 0);
  } else {
    plan.solid.resize(spec.obstacles.size());
    std::ranges::transform(spec.obstacles, plan.solid.begin(),
                           [](std::uint8_t value) { return value != 0 ? 1 : 0; });
  }
  plan.field = &*spec.velocity_field;
  plan.revision = grid.revision();
  return plan;
}

void check_flow_drift_status(std::span<const std::uint32_t> status) {
  for (const auto value : status) {
    switch (value) {
      case shared::drift_ok:
        continue;
      case shared::drift_too_many_substeps:
        throw std::runtime_error("flow drift needs too many substeps; reduce dt");
      case shared::drift_substep_underflow:
        throw std::runtime_error("flow drift substep underflow");
      case shared::drift_invalid_substep:
        throw std::runtime_error("invalid flow drift substep");
      case shared::drift_non_finite:
        throw std::runtime_error("flow drift produced non-finite geometry");
      default:
        throw std::invalid_argument("cell direction must be finite and non-zero");
    }
  }
}

std::vector<std::array<float, 6>> unpack_flow_drift_rates(std::span<const float> rates,
                                                          std::size_t cell_count) {
  std::vector<std::array<float, 6>> result(cell_count);
  for (std::size_t cell = 0; cell < cell_count; ++cell) {
    std::ranges::copy(rates.subspan(6 * cell, 6), result[cell].begin());
  }
  return result;
}

void apply_flow_drift_moves(WorldState& state, std::span<const device::float4> centers,
                            std::span<const device::float4> axes) {
  const auto geometry = state.geometry_state();
  const auto attributes = state.cell_attributes();
  for (std::size_t slot = 0; slot < geometry.size(); ++slot) {
    if (attributes.fixed[slot] != 0) {
      continue;
    }
    state.set_cell_geometry(static_cast<Slot>(slot),
                            Vec3{centers[slot].x, centers[slot].y, centers[slot].z},
                            Vec3{axes[slot].x, axes[slot].y, axes[slot].z}, geometry.lengths[slot]);
  }
}

}  // namespace microsimulator::detail
