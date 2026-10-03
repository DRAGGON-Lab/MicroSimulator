#pragma once

#include <cstdint>
#include <optional>

#include "cm/constraints.hpp"
#include "cm/coupled_rates.hpp"
#include "cm/culture_simulation.hpp"
#include "cm/signals.hpp"
#include "cm/species.hpp"
#include "cm/world_state.hpp"

namespace cm {

inline constexpr std::uint32_t checkpoint_schema_version = 5;

struct SimulationCheckpoint {
  std::uint32_t schema_version{checkpoint_schema_version};
  double time{0.0};
  WorldStateCheckpoint world;
  ConstraintSetCheckpoint constraints;
  SpeciesRatePlan species_rate_plan;
  std::optional<SignalGridCheckpoint> signal_grid;
  std::optional<CoupledRatePlan> coupled_rate_plan;
  std::optional<CultureCheckpoint> culture;

  void validate() const;
};

}  // namespace cm
