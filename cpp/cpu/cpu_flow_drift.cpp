#include "cpu/cpu_flow_drift.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/flow_drift_plan.hpp"
#include "core/parallel.hpp"
#include "cpu/kernels/flow_drift.hpp"

namespace microsimulator {
namespace {

// Cells handed to one thread at a time.
constexpr std::size_t cell_grain = 64;

}  // namespace

std::vector<std::array<float, 6>> flow_drift_rates_cpu(const WorldState& state,
                                                       const SignalGrid& grid, float dt) {
  const auto plan = detail::plan_flow_drift(state, grid, dt, 0.0F);
  const auto cells = static_cast<std::size_t>(plan.data.cell_count);
  const auto& field = *plan.field;
  std::vector<float> rates(6 * std::max<std::size_t>(cells, 1));
  std::vector<std::uint32_t> status(cells);
  detail::parallel_for(cells, cell_grain, [&](std::size_t begin, std::size_t end) {
    for (std::size_t cell = begin; cell < end; ++cell) {
      status[cell] = device::flow_drift_rates_cell(
          static_cast<std::uint32_t>(cell), plan.data, plan.centers.data(), plan.axes.data(),
          plan.fixed.data(), plan.solid.data(), field.x_faces.data(), field.y_faces.data(),
          field.z_faces.data(), rates.data());
    }
  });
  detail::check_flow_drift_status(status);
  return detail::unpack_flow_drift_rates(rates, cells);
}

void advance_flow_drift_cpu(WorldState& state, const SignalGrid& grid, float dt,
                            const FlowDriftParameters& parameters) {
  const auto plan = detail::plan_flow_drift(state, grid, dt, parameters.max_rotation_radians);
  const auto cells = static_cast<std::size_t>(plan.data.cell_count);
  const auto& field = *plan.field;
  std::vector<device::float4> centers(plan.centers.size());
  std::vector<device::float4> axes(plan.axes.size());
  std::vector<std::uint32_t> status(cells);
  detail::parallel_for(cells, cell_grain, [&](std::size_t begin, std::size_t end) {
    for (std::size_t cell = begin; cell < end; ++cell) {
      status[cell] = device::flow_drift_advance_cell(
          static_cast<std::uint32_t>(cell), plan.data, plan.centers.data(), plan.axes.data(),
          plan.fixed.data(), plan.solid.data(), field.x_faces.data(), field.y_faces.data(),
          field.z_faces.data(), centers.data(), axes.data());
    }
  });
  detail::check_flow_drift_status(status);
  detail::apply_flow_drift_moves(state, centers, axes);
}

}  // namespace microsimulator
