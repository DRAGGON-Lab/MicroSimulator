#include "cm/checkpoint.hpp"

#include <cmath>
#include <stdexcept>

namespace cm {

namespace {
void validate_culture_state(const SimulationCheckpoint& checkpoint) {
  if (checkpoint.culture) {
    if (checkpoint.signal_grid || checkpoint.coupled_rate_plan ||
        !checkpoint.constraints.planes.empty() || !checkpoint.constraints.spheres.empty() ||
        !checkpoint.constraints.boxes.empty() || !checkpoint.constraints.cylinders.empty()) {
      throw std::invalid_argument(
          "fluid flow cannot share legacy signals or mechanical checkpoint.constraints");
    }

    checkpoint.culture->validate(checkpoint.world);
  }
}
}  // namespace

void SimulationCheckpoint::validate() const {
  if (schema_version != checkpoint_schema_version) {
    throw std::invalid_argument("unsupported MicroSimulator checkpoint schema version");
  }

  if (!std::isfinite(time) || time < 0.0) {
    throw std::invalid_argument("checkpoint time must be finite and non-negative");
  }

  world.validate();
  constraints.validate();
  species_rate_plan.validate();

  validate_culture_state(*this);

  if (species_rate_plan.species_count() != world.species_count) {
    throw std::invalid_argument("checkpoint rate plan and world species counts disagree");
  }

  if (signal_grid.has_value()) {
    signal_grid->validate();
  }

  if (coupled_rate_plan.has_value()) {
    coupled_rate_plan->validate();

    if (!signal_grid.has_value()) {
      throw std::invalid_argument("checkpoint coupled rate plan requires a signal grid");
    }

    if (coupled_rate_plan->species_count() != world.species_count ||
        coupled_rate_plan->signal_count() != signal_grid->spec.signal_count) {
      throw std::invalid_argument("checkpoint coupled rate plan counts disagree with state");
    }
  }
}

}  // namespace cm
