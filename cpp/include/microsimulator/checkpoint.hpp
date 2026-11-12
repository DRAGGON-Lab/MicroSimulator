#pragma once

#include <cstdint>
#include <optional>

#include "microsimulator/constraints.hpp"
#include "microsimulator/coupled_rates.hpp"
#include "microsimulator/signals.hpp"
#include "microsimulator/species.hpp"
#include "microsimulator/world_state.hpp"

namespace microsimulator {

inline constexpr std::uint32_t checkpoint_schema_version = 4;

struct SimulationCheckpoint {
  std::uint32_t schema_version{checkpoint_schema_version};
  double time{0.0};
  WorldStateCheckpoint world;
  ConstraintSetCheckpoint constraints;
  SpeciesRatePlan species_rate_plan;
  std::optional<SignalGridCheckpoint> signal_grid;
  std::optional<CoupledRatePlan> coupled_rate_plan;

  void validate() const;
};

}  // namespace microsimulator
