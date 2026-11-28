#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "core/device_types.hpp"
#include "microsimulator/deposition.hpp"

namespace microsimulator::detail {

// What every backend's deposit kernels read, in the layout of
// shared/device_types.h. Each array holds at least one element, so a backend binds
// a buffer even where a deposit has no cells or no channels.
struct DepositPlan {
  device::DepositGrid grid{};
  // Each cell's center with its length in w.
  std::vector<device::float4> centers;
  std::vector<float> radii;
  // Each channel's values over the cells in turn.
  std::vector<float> channels;
  // One flag per site, nonzero where the site is solid.
  std::vector<std::uint8_t> solid;

  [[nodiscard]] std::uint32_t cell_count() const noexcept {
    return grid.cell_count;
  }
  [[nodiscard]] std::uint32_t site_count() const noexcept {
    return grid.shape.w;
  }
  // Weights over all cells' boxes, and the most site entries they can make.
  [[nodiscard]] std::size_t weight_count() const noexcept {
    return static_cast<std::size_t>(grid.cell_count) * grid.box.w;
  }
};

// Validates a deposit and lays it out for the kernels.
[[nodiscard]] DepositPlan plan_deposit(const SignalGridSpec& spec, const DepositCells& cells,
                                       const BiomassDepositParameters& parameters);

// Throws when a cell's tent reached no fluid site. `status` holds the value
// deposit_cell_weights returned for each cell.
void check_deposit_status(std::span<const std::uint32_t> status);

// What every backend's closure kernel reads.
struct ShallowCouplingPlan {
  device::ShallowCouplingData data{};
  std::vector<std::uint8_t> solid;
  // One value per site, or a single value for every site.
  std::vector<float> base;
  // The deposit's two velocity channels, or one zero when it carries none.
  std::vector<float> velocity;

  [[nodiscard]] std::uint32_t column_count() const noexcept {
    return data.shape.x * data.shape.y;
  }
};

// Validates a closure and lays it out for the kernel.
[[nodiscard]] ShallowCouplingPlan plan_shallow_flow_coupling(
    const SignalGridSpec& spec, std::span<const float> base, const BiomassDeposit& deposit,
    const BiomassDragParameters& parameters);

// What every backend's Brinkman drag kernel reads.
struct BrinkmanDragPlan {
  device::BrinkmanDragData data{};
  std::vector<std::uint8_t> solid;
};

// Validates a Brinkman drag and lays it out for the kernel.
[[nodiscard]] BrinkmanDragPlan plan_brinkman_drag(const SignalGridSpec& spec,
                                                  const BiomassDeposit& deposit,
                                                  const BiomassDragParameters& parameters);

// The closure's result from the kernel's per-site mobility and the two
// in-plane force components of each column in turn.
[[nodiscard]] ShallowFlowCoupling finish_shallow_flow_coupling(std::vector<float> mobility,
                                                               std::span<const float> force);

}  // namespace microsimulator::detail
