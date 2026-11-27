#include "microsimulator/deposition.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/deposition_plan.hpp"

namespace microsimulator {
namespace {

constexpr auto uint_limit = static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max());

[[nodiscard]] bool finite(Vec3 value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

[[nodiscard]] std::uint32_t checked_uint(std::size_t value, const char* description) {
  if (value > uint_limit) {
    throw std::overflow_error(std::string(description) + " exceeds the uint32 index space");
  }
  return static_cast<std::uint32_t>(value);
}

// The most sites a tent of `radius` can reach along an axis of `count` sites
// at `spacing`: its 2 radius / spacing sites, one more for a center between
// sites, and one for rounding.
[[nodiscard]] std::uint32_t largest_extent(float radius, float spacing, std::uint32_t count) {
  const auto sites = std::ceil(2.0 * static_cast<double>(radius) / spacing) + 3.0;
  return sites >= count ? count : static_cast<std::uint32_t>(sites);
}

[[nodiscard]] std::vector<std::uint8_t> solid_flags(const SignalGridSpec& spec) {
  if (spec.obstacles.empty()) {
    return std::vector<std::uint8_t>(spec.site_count(), 0);
  }
  std::vector<std::uint8_t> solid(spec.obstacles.size());
  std::ranges::transform(spec.obstacles, solid.begin(),
                         [](std::uint8_t value) { return value != 0 ? 1 : 0; });
  return solid;
}

}  // namespace

std::size_t DepositCells::size() const noexcept {
  return positions.size();
}

void DepositCells::validate() const {
  const auto count = positions.size();
  if (lengths.size() != count || radii.size() != count) {
    throw std::invalid_argument("deposited cells need one length and one radius per position");
  }
  if (channels.size() != static_cast<std::size_t>(channel_count) * count) {
    throw std::invalid_argument("deposited cells need one value per channel and cell");
  }
  for (std::size_t cell = 0; cell < count; ++cell) {
    if (!finite(positions[cell])) {
      throw std::invalid_argument("deposited positions must be finite");
    }
    if (!std::isfinite(lengths[cell]) || lengths[cell] < 0.0F || !std::isfinite(radii[cell]) ||
        radii[cell] <= 0.0F) {
      throw std::invalid_argument(
          "deposited lengths must be finite and nonnegative, and radii finite and positive");
    }
  }
  for (const auto value : channels) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("deposited channel values must be finite");
    }
  }
}

void BiomassDepositParameters::validate() const {
  if (!std::isfinite(averaging_radius) || averaging_radius <= 0.0F) {
    throw std::invalid_argument("averaging radius must be finite and positive");
  }
}

void BiomassDragParameters::validate() const {
  if (!std::isfinite(drag_coefficient) || drag_coefficient < 0.0F) {
    throw std::invalid_argument("drag coefficient must be finite and non-negative");
  }
  if (!(max_volume_fraction > 0.0F && max_volume_fraction < 1.0F)) {
    throw std::invalid_argument("maximum volume fraction must lie strictly between zero and one");
  }
}

namespace detail {

void validate_deposit(const SignalGridSpec& spec, const DepositCells& cells,
                      const BiomassDepositParameters& parameters) {
  spec.validate_lattice();
  cells.validate();
  parameters.validate();
}

void validate_shallow_flow_coupling(const SignalGridSpec& spec, std::span<const float> base,
                                    const BiomassDeposit& deposit,
                                    const BiomassDragParameters& parameters) {
  spec.validate_lattice();
  parameters.validate();
  const auto sites = spec.site_count();
  if (deposit.density.size() != sites) {
    throw std::invalid_argument("deposit density must hold one value per grid site");
  }
  if (deposit.channel_count != 0 && deposit.channel_count != 2) {
    throw std::invalid_argument(
        "shallow flow coupling takes a deposit of no channels or of two velocity channels");
  }
  if (deposit.channels.size() != static_cast<std::size_t>(deposit.channel_count) * sites) {
    throw std::invalid_argument("deposit channels must hold one value per channel and site");
  }
  if (base.size() == 1) {
    if (!std::isfinite(base[0]) || base[0] <= 0.0F) {
      throw std::invalid_argument("base mobility must be finite and positive");
    }
  } else if (base.size() == sites) {
    for (const auto value : base) {
      if (!std::isfinite(value) || value < 0.0F) {
        throw std::invalid_argument("base mobility values must be finite and non-negative");
      }
    }
  } else {
    throw std::invalid_argument("base mobility must hold one value per grid site");
  }
}

void validate_brinkman_drag(const SignalGridSpec& spec, const BiomassDeposit& deposit,
                            const BiomassDragParameters& parameters) {
  spec.validate_lattice();
  parameters.validate();
  if (deposit.density.size() != spec.site_count()) {
    throw std::invalid_argument("deposit density must hold one value per grid site");
  }
}

DepositPlan plan_deposit(const SignalGridSpec& spec, const DepositCells& cells,
                         const BiomassDepositParameters& parameters) {
  validate_deposit(spec, cells, parameters);
  DepositPlan plan;
  const auto radius = parameters.averaging_radius;
  auto& grid = plan.grid;
  grid.shape = {spec.shape.x, spec.shape.y, spec.shape.z,
                checked_uint(spec.site_count(), "deposit site count")};
  grid.origin = {spec.origin.x, spec.origin.y, spec.origin.z, radius};
  grid.spacing = {spec.spacing.x, spec.spacing.y, spec.spacing.z, spec.voxel_volume()};
  grid.box = {largest_extent(radius, spec.spacing.x, spec.shape.x),
              largest_extent(radius, spec.spacing.y, spec.shape.y),
              largest_extent(radius, spec.spacing.z, spec.shape.z), 0};
  grid.box.w = checked_uint(static_cast<std::size_t>(grid.box.x) * grid.box.y * grid.box.z,
                            "deposit box size");
  grid.cell_count = checked_uint(cells.size(), "deposit cell count");
  grid.channel_count = cells.channel_count;
  grid.has_obstacles = spec.has_obstacles() ? 1U : 0U;
  static_cast<void>(checked_uint(plan.weight_count(), "deposit weight count"));

  plan.centers.reserve(std::max<std::size_t>(cells.size(), 1));
  for (std::size_t cell = 0; cell < cells.size(); ++cell) {
    const auto& position = cells.positions[cell];
    plan.centers.push_back({position.x, position.y, position.z, cells.lengths[cell]});
  }
  plan.radii = cells.radii;
  plan.channels = cells.channels;
  if (plan.centers.empty()) {
    plan.centers.push_back({0.0F, 0.0F, 0.0F, 0.0F});
    plan.radii.push_back(1.0F);
  }
  if (plan.channels.empty()) {
    plan.channels.push_back(0.0F);
  }
  plan.solid = solid_flags(spec);
  return plan;
}

void check_deposit_status(std::span<const std::uint32_t> status) {
  if (std::ranges::any_of(status, [](std::uint32_t value) { return value != 0; })) {
    throw std::invalid_argument("biomass deposition has no connected fluid support");
  }
}

ShallowCouplingPlan plan_shallow_flow_coupling(const SignalGridSpec& spec,
                                               std::span<const float> base,
                                               const BiomassDeposit& deposit,
                                               const BiomassDragParameters& parameters) {
  validate_shallow_flow_coupling(spec, base, deposit, parameters);
  ShallowCouplingPlan plan;
  auto& data = plan.data;
  data.shape = {spec.shape.x, spec.shape.y, spec.shape.z,
                checked_uint(spec.site_count(), "coupling site count")};
  data.drag_coefficient = parameters.drag_coefficient;
  data.max_volume_fraction = parameters.max_volume_fraction;
  data.uniform_base = base.size() == 1 ? 1U : 0U;
  data.has_velocity = deposit.channel_count == 2 ? 1U : 0U;
  plan.solid = solid_flags(spec);
  plan.base.assign(base.begin(), base.end());
  plan.velocity = deposit.channels;
  if (plan.velocity.empty()) {
    plan.velocity.push_back(0.0F);
  }
  return plan;
}

BrinkmanDragPlan plan_brinkman_drag(const SignalGridSpec& spec, const BiomassDeposit& deposit,
                                    const BiomassDragParameters& parameters) {
  validate_brinkman_drag(spec, deposit, parameters);
  BrinkmanDragPlan plan;
  plan.data.shape = {spec.shape.x, spec.shape.y, spec.shape.z,
                     checked_uint(spec.site_count(), "drag site count")};
  plan.data.drag_coefficient = parameters.drag_coefficient;
  plan.data.max_volume_fraction = parameters.max_volume_fraction;
  plan.solid = solid_flags(spec);
  return plan;
}

ShallowFlowCoupling finish_shallow_flow_coupling(std::vector<float> mobility,
                                                 std::span<const float> force) {
  ShallowFlowCoupling result;
  result.mobility = std::move(mobility);
  result.body_force.resize(force.size() / 2);
  for (std::size_t column = 0; column < result.body_force.size(); ++column) {
    result.body_force[column] = {force[2 * column], force[(2 * column) + 1]};
  }
  return result;
}

}  // namespace detail
}  // namespace microsimulator
