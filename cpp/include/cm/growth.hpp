#pragma once

#include <memory>
#include <vector>

#include "cm/types.hpp"

namespace cm {
enum class GrowthKind : std::uint8_t { monod, essential };

struct GrowthRequirement {
  std::uint32_t solute{0};
  double half_saturation{1};  // amount / model volume
  double biomass_yield{1};    // grams / amount
};

struct CellGrowthModel {
  CellId cell_id{0};
  GrowthKind kind{GrowthKind::monod};
  double mu_max{0};           // 1 / model time
  double biomass_density{1};  // grams / biochemical model volume
  double volume_ratio{1};     // biochemical / geometric volume
  std::vector<GrowthRequirement> requirements;
  void validate(std::size_t solute_count) const;
};

struct GrowthInput {
  double biochemical_volume{0}, surface_area{0};
  std::vector<double> concentrations, uptake;  // requirement order, canonical units
};

struct GrowthEvaluation {
  std::vector<double> uptake_velocities;
  double biomass_gain{0}, biochemical_volume_gain{0}, geometric_volume_gain{0};
  double specific_rate{0}, stoichiometric_residual{0};
};

class GrowthExecutor {
 public:
  explicit GrowthExecutor(BackendKind = BackendKind::cpu, std::uint32_t device_index = 0);
  ~GrowthExecutor();
  std::vector<GrowthEvaluation> evaluate(const std::vector<CellGrowthModel>&,
                                         const std::vector<GrowthInput>&, double dt);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace cm
