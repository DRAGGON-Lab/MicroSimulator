#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include "backend_devices.hpp"
#include "cm/simulation.hpp"

namespace {
void check(bool value, const char* message) {
  if (!value) {
    throw std::runtime_error(message);
  }
}

void run(cm::BackendKind backend, std::uint32_t device) {
  cm::Simulation simulation(backend, 1, 1, device);
  cm::CellInit cell;
  cell.position = {5.5, 5.5, 5.5};
  cell.radius = 2;
  cell.length = 1;
  cell.growth_rate = 0.01F;
  cell.species = {3};
  cell.fixed = true;
  const auto id = simulation.add_cell(cell);
  cm::CultureConfiguration config;
  config.grid.shape = {12, 12, 12};
  config.maximum_retries = 1;
  config.ports = {{.name = "in"}, {.name = "out", .upper = true}};
  config.solutes = {{"nutrient", 0.1}};
  config.reservoirs = {{"in", {2}}, {"out", {2}}};
  config.exchange = {{id, 0, 0, 0.01, 0}};
  config.biomass_requirements = {{0, 2}};
  simulation.configure_culture(config, {2}, {80});
  const auto original = simulation.checkpoint();
  simulation.step(0.1F);
  const auto state = *simulation.culture_checkpoint();
  check(simulation.time() > 0 && state.cells[0].biochemical_volume > 80, "media simulation growth");
  const double geometric_growth =
      state.cells[0].body.geometric_volume() - original.culture->cells[0].body.geometric_volume();
  check(std::abs(state.cells[0].biochemical_volume - 80 - geometric_growth) < 1e-11,
        "independent biochemical biomass increment");
  const double extracellular_initial =
      std::accumulate(original.culture->extracellular_amounts.begin(),
                      original.culture->extracellular_amounts.end(), 0.0);
  const double extracellular_final =
      std::accumulate(state.extracellular_amounts.begin(), state.extracellular_amounts.end(), 0.0);
  double reservoir = 0;

  for (const auto& t : state.reservoir_totals) {
    reservoir += t.amounts[0];
  }

  const double initial_total = extracellular_initial + 240;
  const double final_total =
      extracellular_final + state.cells[0].species_amounts[0] + 2 * geometric_growth;
  check(std::abs(final_total - initial_total - reservoir) < 0.02,
        "fluid cell and biomass amount budget");
  cm::Simulation restart(backend, simulation.checkpoint(), device);
  simulation.step(0.1F);
  restart.step(0.1F);
  const auto continued = *simulation.culture_checkpoint(), resumed = *restart.culture_checkpoint();
  check(continued.cells[0].body.position == resumed.cells[0].body.position &&
            continued.cells[0].species_amounts == resumed.cells[0].species_amounts &&
            continued.extracellular_amounts == resumed.extracellular_amounts,
        "media restart continuation");
  auto kinetics_checkpoint = simulation.checkpoint();
  kinetics_checkpoint.culture->configuration.exchange.clear();
  kinetics_checkpoint.world.cells[0].growth_rate = 0;
  kinetics_checkpoint.species_rate_plan =
      cm::SpeciesRatePlan(1, {{.operation = cm::RateOp::cell_volume}}, {0});
  cm::Simulation kinetics(backend, kinetics_checkpoint, device);
  const auto& old_cell = kinetics_checkpoint.culture->cells[0];
  kinetics.step(0.125F);
  const auto kinetic_amount = kinetics.culture_checkpoint()->cells[0].species_amounts[0];
  const double expected_amount = old_cell.species_amounts[0] +
                                 0.125 * old_cell.biochemical_volume * old_cell.biochemical_volume;
  check(std::abs(kinetic_amount / expected_amount - 1) < 2e-6,
        "native kinetics uses explicit biochemical volume");
  const auto before = simulation.checkpoint();
  using enum cm::RateOp;
  simulation.set_species_rate_plan(cm::SpeciesRatePlan(
      1, {{.operation = constant, .value = 0}, {.operation = divide, .first = 0, .second = 0}},
      {1}));
  bool failed = false;

  try {
    simulation.step(0.1F);
  } catch (const std::exception&) {
    failed = true;
  }

  check(failed && simulation.time() == before.time &&
            simulation.culture_checkpoint()->extracellular_amounts ==
                before.culture->extracellular_amounts &&
            simulation.culture_checkpoint()->cells[0].body.length ==
                before.culture->cells[0].body.length,
        "media failed interval rollback");
  simulation.restore_checkpoint(before);
  check(simulation.time() == before.time, "media explicit restore");
  std::cout << "media simulation backend " << static_cast<int>(backend) << " passed\n";
}
}  // namespace

int main() {
  cm::test::for_each_backend_device(run);
}
