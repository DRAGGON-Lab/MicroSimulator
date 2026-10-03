#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include "backend_devices.hpp"
#include "cm/growth.hpp"
#include "cm/simulation.hpp"

void check(bool value, const char* message) {
  if (!value) {
    throw std::runtime_error(message);
  }
}

void run(cm::BackendKind backend, std::uint32_t device) {
  cm::CellGrowthModel model;
  model.cell_id = 1;
  model.mu_max = 0.01;
  model.biomass_density = 1;
  model.requirements = {{0, 1, 2}};
  cm::GrowthExecutor executor(backend, device);
  const auto e = executor.evaluate({model}, {{10, 5, {3}, {0.2}}}, 0.1)[0];
  check(std::abs(e.uptake_velocities[0] - 0.0025) < 1e-8, "native Monod coefficient");
  check(std::abs(e.biomass_gain - 0.4) < 1e-7, "native accepted uptake yield");
  cm::Simulation sim(backend, 1, 0, device);
  cm::CellInit c;
  c.position = {5.5, 5.5, 5.5};
  c.radius = 2;
  c.length = 1;
  c.fixed = true;
  c.growth_rate = 0;
  const auto id = sim.add_cell(c);
  cm::CultureConfiguration config;
  config.grid.shape = {12, 12, 12};
  config.ports = {{.name = "in"}, {.name = "out", .upper = true}};
  config.solutes = {{"substrate", 0.1}};
  config.reservoirs = {{"in", {0}, cm::ChemicalBoundaryKind::advective},
                       {"out", {0}, cm::ChemicalBoundaryKind::advective}};
  model.cell_id = id;
  config.growth = {model};
  config.maximum_retries = 0;
  config.coupling_tolerance = 2e-6;
  sim.configure_culture(config, {1});
  const auto before = *sim.culture_checkpoint();
  sim.step(0.01F);
  const auto after = *sim.culture_checkpoint();
  const double gained = after.cells[0].biochemical_volume - before.cells[0].biochemical_volume;
  check(gained > 0, "nutrient driven growth");
  check(std::abs(gained - 2 * after.cells[0].uptake_totals[0]) < 1e-10, "biomass uptake budget");
  double supplied = 0;

  for (const auto& r : after.reservoir_totals) {
    supplied += r.amounts[0];
  }

  const double lost =
      std::accumulate(before.extracellular_amounts.begin(), before.extracellular_amounts.end(),
                      0.0) -
      std::accumulate(after.extracellular_amounts.begin(), after.extracellular_amounts.end(), 0.0);
  check(std::abs(lost + supplied - gained / 2) < 0.02, "whole culture chemical budget");
  cm::Simulation resumed(backend, sim.checkpoint(), device);
  check(resumed.culture_checkpoint()->cells[0].biochemical_volume ==
            after.cells[0].biochemical_volume,
        "growth checkpoint");
  std::cout << "culture growth backend " << static_cast<int>(backend) << " gain=" << gained << '\n';
}

int main() {
  cm::test::for_each_backend_device(run);
}
