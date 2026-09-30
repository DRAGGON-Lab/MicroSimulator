#pragma once

#include "cm/backend.hpp"
#include "cm/culture_simulation.hpp"

namespace cm::detail {
class CultureSimulation {
 public:
  CultureSimulation(const WorldState&, const CultureConfiguration&,
                    const std::vector<double>& concentrations,
                    const std::vector<double>& biochemical_volumes, const BackendInfo&);
  CultureSimulation(const WorldState&, const CultureCheckpoint&, const BackendInfo&);
  void step(WorldState&, ComputeBackend&, const SpeciesRatePlan&, double dt);
  std::pair<CellId, CellId> divide(WorldState&, CellId, double fraction);
  void set_species(CellId, std::span<const float>);
  void set_force(CellId, std::array<double, 3>, std::array<double, 3>);

  [[nodiscard]] const CultureCheckpoint& checkpoint() const {
    return data_;
  }

  [[nodiscard]] const FluidGeometry& geometry() const {
    return *geometry_;
  }

 private:
  double reserve_step(CultureCheckpoint&, std::shared_ptr<FluidGeometry>&, WorldState&,
                      ComputeBackend&, const SpeciesRatePlan&, double maximum_dt, double remaining,
                      bool zero_plan);
  double kinetic_step(CultureCheckpoint&, std::shared_ptr<FluidGeometry>&, WorldState&,
                      ComputeBackend&, const SpeciesRatePlan&, double);
  CultureCheckpoint data_;
  std::shared_ptr<FluidGeometry> geometry_;
  StokesFlowSolver flow_;
  SoluteTransportSolver transport_;
  GrowthExecutor growth_;
};
}  // namespace cm::detail
