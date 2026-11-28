#include "cpu/cpu_deposition.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "core/deposition_plan.hpp"
#include "core/device_types.hpp"
#include "core/parallel.hpp"
#include "core/shared_constants.hpp"

namespace microsimulator {
namespace {

using device::DepositBox;
using device::DepositGrid;

// Cells and sites handed to one thread at a time. A cell's weights cost a
// box of sites, so cells go in smaller pieces.
constexpr std::size_t cell_grain = 32;
constexpr std::size_t site_grain = 4096;

// Integral of the unit tent, 1 - |u| on [-1, 1], from -1 up to `edge`.
float tent_cdf(float edge) {
  if (edge <= -1.0F) {
    return 0.0F;
  }
  if (edge < 0.0F) {
    return 0.5F * (edge + 1.0F) * (edge + 1.0F);
  }
  if (edge < 1.0F) {
    return 1.0F - 0.5F * (1.0F - edge) * (1.0F - edge);
  }
  return 1.0F;
}

// The integral over site `index` of a tent of `radius` about `center`, along
// an axis whose sites are centered at origin + index * spacing.
float axis_weight(std::uint32_t index, float origin, float spacing, float center, float radius) {
  const auto position = static_cast<float>(index);
  const float lower = (origin + (position - 0.5F) * spacing - center) / radius;
  const float upper = (origin + (position + 0.5F) * spacing - center) / radius;
  return tent_cdf(upper) - tent_cdf(lower);
}

// The conserved biomass of a cell whose center holds its length in w.
float biomass(const device::float4& center, float radius) {
  return shared::pi * radius * radius * (center.w + 2.0F * radius);
}

// The box of sites that the tent of a cell centered at `center` reaches; a
// cell outside the sites' extent has an empty box.
DepositBox cell_box(const DepositGrid& grid, const device::float4& center) {
  const std::array centers{center.x, center.y, center.z};
  const std::array origins{grid.origin.x, grid.origin.y, grid.origin.z};
  const std::array spacings{grid.spacing.x, grid.spacing.y, grid.spacing.z};
  const std::array counts{grid.shape.x, grid.shape.y, grid.shape.z};
  const std::array largest{grid.box.x, grid.box.y, grid.box.z};
  const float radius = grid.origin.w;
  std::array<std::uint32_t, 3> lower{};
  std::array<std::uint32_t, 3> extent{};
  bool inside = true;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    const float first = origins[axis] - spacings[axis] * 0.5F;
    const float end = origins[axis] + (static_cast<float>(counts[axis]) - 0.5F) * spacings[axis];
    inside = inside && centers[axis] >= first && centers[axis] < end;
  }
  if (inside) {
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const auto count = static_cast<int>(counts[axis]);
      const int low = std::max(
          0, static_cast<int>(
                 std::floor((centers[axis] - radius - origins[axis]) / spacings[axis] + 0.5F)));
      const int high = std::clamp(
          static_cast<int>(
              std::ceil((centers[axis] + radius - origins[axis]) / spacings[axis] + 0.5F)),
          low, std::max(low, count));
      lower[axis] = static_cast<std::uint32_t>(low);
      extent[axis] = std::min(static_cast<std::uint32_t>(high - low), largest[axis]);
    }
  }
  DepositBox box{};
  box.lower = {lower[0], lower[1], lower[2], inside ? 1U : 0U};
  box.extent = {extent[0], extent[1], extent[2], extent[0] * extent[1] * extent[2]};
  return box;
}

// The flat index of the box site at `local`, which counts x-major and
// z-minor through the box as the grid counts its sites.
std::uint32_t box_site(const DepositGrid& grid, const DepositBox& box, std::uint32_t local) {
  const auto iz = local % box.extent.z;
  const auto iy = (local / box.extent.z) % box.extent.y;
  const auto ix = local / (box.extent.y * box.extent.z);
  return ((box.lower.x + ix) * grid.shape.y + box.lower.y + iy) * grid.shape.z + box.lower.z + iz;
}

// Fills `block` with the weights of a cell over its box, summing to one.
// Where the grid has obstacles, only the face-connected fluid region holding
// the strongest weight carries weight: corner contact is not fluid, and
// biomass does not reach through a wall. Returns false when no fluid site
// carries weight.
bool cell_weights(const DepositGrid& grid, const device::float4& center, const DepositBox& box,
                  std::span<const std::uint8_t> solid, std::span<float> block) {
  const auto ex = box.extent.x;
  const auto ey = box.extent.y;
  const auto ez = box.extent.z;
  const float radius = grid.origin.w;
  float strongest = 0.0F;
  std::uint32_t seed = 0;
  for (std::uint32_t ix = 0; ix < ex; ++ix) {
    const float wx = axis_weight(box.lower.x + ix, grid.origin.x, grid.spacing.x, center.x, radius);
    for (std::uint32_t iy = 0; iy < ey; ++iy) {
      const float wxy =
          wx * axis_weight(box.lower.y + iy, grid.origin.y, grid.spacing.y, center.y, radius);
      for (std::uint32_t iz = 0; iz < ez; ++iz) {
        const auto local = (ix * ey + iy) * ez + iz;
        float weight =
            wxy * axis_weight(box.lower.z + iz, grid.origin.z, grid.spacing.z, center.z, radius);
        if (grid.has_obstacles != 0 && solid[box_site(grid, box, local)] != 0) {
          weight = 0.0F;
        }
        block[local] = weight;
        if (weight > strongest) {
          strongest = weight;
          seed = local;
        }
      }
    }
  }
  if (!(strongest > 0.0F)) {
    return false;
  }
  const auto size = box.extent.w;
  if (grid.has_obstacles != 0) {
    // A fill from the seed through face neighbors that carry weight; the
    // sites it does not reach lose their weight.
    std::vector<std::uint8_t> reached(size, 0);
    std::vector<std::uint32_t> frontier{seed};
    reached[seed] = 1;
    const auto plane = ey * ez;
    while (!frontier.empty()) {
      const auto local = frontier.back();
      frontier.pop_back();
      const auto ix = local / plane;
      const auto iy = (local / ez) % ey;
      const auto iz = local % ez;
      const auto visit = [&](bool exists, std::uint32_t neighbor) {
        if (exists && reached[neighbor] == 0 && block[neighbor] > 0.0F) {
          reached[neighbor] = 1;
          frontier.push_back(neighbor);
        }
      };
      visit(ix > 0, local - plane);
      visit(ix + 1 < ex, local + plane);
      visit(iy > 0, local - ez);
      visit(iy + 1 < ey, local + ez);
      visit(iz > 0, local - 1);
      visit(iz + 1 < ez, local + 1);
    }
    for (std::uint32_t local = 0; local < size; ++local) {
      if (reached[local] == 0) {
        block[local] = 0.0F;
      }
    }
  }
  float total = 0.0F;
  for (std::uint32_t local = 0; local < size; ++local) {
    total += block[local];
  }
  for (std::uint32_t local = 0; local < size; ++local) {
    block[local] /= total;
  }
  return true;
}

// The Kozeny-Carman drag of biomass packed to a volume fraction.
float kozeny_carman(float fraction, float coefficient) {
  const float open = 1.0F - fraction;
  return coefficient * fraction * fraction / (open * open * open);
}

}  // namespace

BiomassDeposit deposit_biomass_cpu(const SignalGridSpec& spec, const DepositCells& cells,
                                   const BiomassDepositParameters& parameters) {
  const auto plan = detail::plan_deposit(spec, cells, parameters);
  const auto& grid = plan.grid;
  const auto cell_count = static_cast<std::size_t>(plan.cell_count());
  const auto site_count = static_cast<std::size_t>(plan.site_count());
  const auto stride = static_cast<std::size_t>(grid.box.w);

  // Each cell's box and weights are its own, so cells are weighed in
  // parallel.
  std::vector<DepositBox> boxes(cell_count);
  std::vector<float> weights(plan.weight_count());
  std::vector<std::uint32_t> status(cell_count);
  detail::parallel_for(cell_count, cell_grain, [&](std::size_t begin, std::size_t end) {
    for (std::size_t cell = begin; cell < end; ++cell) {
      const auto& center = plan.centers[cell];
      boxes[cell] = cell_box(grid, center);
      if (boxes[cell].lower.w == 0) {
        continue;
      }
      const std::span block(weights.data() + cell * stride, boxes[cell].extent.w);
      status[cell] = cell_weights(grid, center, boxes[cell], plan.solid, block) ? 0U : 1U;
    }
  });
  detail::check_deposit_status(status);

  // Cells add into the sites they reach one at a time in ascending order, so
  // each site sums its cells in that order whatever the thread count.
  std::vector<float> mass(site_count, 0.0F);
  std::vector<float> amounts(static_cast<std::size_t>(grid.channel_count) * site_count, 0.0F);
  for (std::size_t cell = 0; cell < cell_count; ++cell) {
    const auto& box = boxes[cell];
    const float cell_mass = biomass(plan.centers[cell], plan.radii[cell]);
    const float* block = weights.data() + cell * stride;
    for (std::uint32_t local = 0; local < box.extent.w; ++local) {
      if (!(block[local] > 0.0F)) {
        continue;
      }
      const auto site = box_site(grid, box, local);
      const float deposited = cell_mass * block[local];
      mass[site] += deposited;
      for (std::uint32_t channel = 0; channel < grid.channel_count; ++channel) {
        amounts[channel * site_count + site] +=
            plan.channels[channel * cell_count + cell] * deposited;
      }
    }
  }
  const float volume = grid.spacing.w;
  detail::parallel_for(site_count, site_grain, [&](std::size_t begin, std::size_t end) {
    for (std::size_t site = begin; site < end; ++site) {
      mass[site] /= volume;
      for (std::uint32_t channel = 0; channel < grid.channel_count; ++channel) {
        amounts[channel * site_count + site] /= volume;
      }
    }
  });
  BiomassDeposit result;
  result.channel_count = grid.channel_count;
  result.density = std::move(mass);
  result.channels = std::move(amounts);
  return result;
}

ShallowFlowCoupling shallow_flow_coupling_cpu(const SignalGridSpec& spec,
                                              std::span<const float> base,
                                              const BiomassDeposit& deposit,
                                              const BiomassDragParameters& parameters) {
  const auto plan = detail::plan_shallow_flow_coupling(spec, base, deposit, parameters);
  const auto& data = plan.data;
  const auto columns = static_cast<std::size_t>(plan.column_count());
  const auto depth = data.shape.z;
  std::vector<float> mobility(data.shape.w);
  std::vector<float> force(2 * columns);
  // A column's biomass fraction is its density averaged over its fluid sites
  // and capped, which sets the drag d. Each fluid site of base mobility b
  // takes mobility b / (1 + b d), and the column takes the body force d V,
  // where V is the biomass-weighted velocity that channels 0 and 1 carry.
  detail::parallel_for(columns, site_grain, [&](std::size_t begin, std::size_t end) {
    for (std::size_t column = begin; column < end; ++column) {
      const auto first = column * depth;
      std::uint32_t fluid = 0;
      float mass = 0.0F;
      for (std::uint32_t z = 0; z < depth; ++z) {
        fluid += plan.solid[first + z] == 0 ? 1U : 0U;
        mass += deposit.density[first + z];
      }
      const float fraction =
          std::min(mass / static_cast<float>(std::max(fluid, 1U)), data.max_volume_fraction);
      const float drag = kozeny_carman(fraction, data.drag_coefficient);
      for (std::uint32_t z = 0; z < depth; ++z) {
        const float b = data.uniform_base != 0 ? plan.base[0] : plan.base[first + z];
        mobility[first + z] = plan.solid[first + z] != 0 ? 0.0F : b / (1.0F + b * drag);
      }
      float fx = 0.0F;
      float fy = 0.0F;
      if (data.has_velocity != 0 && mass > 0.0F) {
        float mx = 0.0F;
        float my = 0.0F;
        for (std::uint32_t z = 0; z < depth; ++z) {
          mx += plan.velocity[first + z];
          my += plan.velocity[data.shape.w + first + z];
        }
        fx = mx * (drag / mass);
        fy = my * (drag / mass);
      }
      force[2 * column] = fx;
      force[2 * column + 1] = fy;
    }
  });
  return detail::finish_shallow_flow_coupling(std::move(mobility), force);
}

std::vector<float> brinkman_drag_cpu(const SignalGridSpec& spec, const BiomassDeposit& deposit,
                                     const BiomassDragParameters& parameters) {
  const auto plan = detail::plan_brinkman_drag(spec, deposit, parameters);
  const auto sites = static_cast<std::size_t>(plan.data.shape.w);
  std::vector<float> drag(sites);
  detail::parallel_for(sites, site_grain, [&](std::size_t begin, std::size_t end) {
    for (std::size_t site = begin; site < end; ++site) {
      const float fraction = std::min(deposit.density[site], plan.data.max_volume_fraction);
      drag[site] =
          plan.solid[site] != 0 ? 0.0F : kozeny_carman(fraction, plan.data.drag_coefficient);
    }
  });
  return drag;
}

}  // namespace microsimulator
