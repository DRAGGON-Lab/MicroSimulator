#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "microsimulator/signals.hpp"
#include "microsimulator/types.hpp"

namespace microsimulator {

// Cells to deposit on a grid, as parallel arrays by cell. Each cell may carry
// `channel_count` values, such as an intracellular concentration or a
// velocity component, which the deposit spreads with its biomass. `channels`
// holds each channel's values over the cells in turn.
struct DepositCells {
  std::vector<Vec3> positions;
  std::vector<float> lengths;
  std::vector<float> radii;
  std::uint32_t channel_count{0};
  std::vector<float> channels;

  [[nodiscard]] std::size_t size() const noexcept;
  void validate() const;
};

// The radius of the separable tent each cell spreads its biomass with, in
// physical length units, so that the averaging does not change under grid
// refinement.
struct BiomassDepositParameters {
  float averaging_radius{4.0F};

  void validate() const;
};

// Conserved biomass B = pi r^2 (L + 2r) per unit volume at each grid site,
// and for each channel its per-cell value times B per unit volume. Each cell
// spreads over the sites of its tent within one face-connected fluid region,
// renormalized there, so the sum over sites of density times site volume is
// the total biomass of the cells inside the grid. Cells outside the grid's
// extent deposit nothing.
struct BiomassDeposit {
  std::vector<float> density;
  std::uint32_t channel_count{0};
  std::vector<float> channels;
};

// The empirical drag of packed biomass. Biomass that fills a volume fraction
// f, capped at `max_volume_fraction`, has Kozeny-Carman drag
// d = c f^2 / (1 - f)^3 with c the `drag_coefficient`. Shallow flow takes f
// from each column's mean density, resolved flow from each site's density.
struct BiomassDragParameters {
  float drag_coefficient{100.0F};
  float max_volume_fraction{0.9F};

  void validate() const;
};

// The shallow-flow coefficients of a deposit: per site, mobility
// b / (1 + b d) on fluid and zero on solid, from base mobility b; per x/y
// column, x-major and y-minor, the body force d V, where V is the
// biomass-weighted in-plane velocity of the cells. The force is zero unless
// the deposit's two channels carry that velocity.
struct ShallowFlowCoupling {
  std::vector<float> mobility;
  std::vector<std::array<float, 2>> body_force;
};

namespace detail {

// Checks a deposit's grid, cells, and parameters together.
void validate_deposit(const SignalGridSpec& spec, const DepositCells& cells,
                      const BiomassDepositParameters& parameters);

// Checks a closure's grid, base mobility, deposit, and parameters together.
// The base mobility holds one value for every site or a single value for all
// of them.
void validate_shallow_flow_coupling(const SignalGridSpec& spec, std::span<const float> base,
                                    const BiomassDeposit& deposit,
                                    const BiomassDragParameters& parameters);

// Checks a Brinkman drag's grid, deposit, and parameters together.
void validate_brinkman_drag(const SignalGridSpec& spec, const BiomassDeposit& deposit,
                            const BiomassDragParameters& parameters);

}  // namespace detail

}  // namespace microsimulator
