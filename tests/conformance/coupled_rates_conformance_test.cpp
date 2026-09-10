#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include "backend_devices.hpp"
#include "cm/simulation.hpp"

namespace {

cm::RateInstruction operation(cm::RateOp op, std::uint32_t first = 0, std::uint32_t second = 0,
                               float value = 0.0F) {
  return {.operation = op, .first = first, .second = second, .value = value};
}

cm::Simulation make_reference(cm::SignalIntegrationKind integration, bool masked = false) {
  cm::Simulation simulation(cm::BackendKind::cpu, 513, 3);
  cm::SignalGridSpec grid;
  grid.signal_count = 2;
  grid.shape = {.x = 9, .y = 7, .z = 5};
  grid.origin = {-3.0F, -2.0F, -1.0F};
  grid.spacing = {0.75F, 1.25F, 2.0F};
  grid.diffusion = {0.08F, 0.03F};
  grid.advection = {{0.05F, -0.02F, 0.01F}, {-0.03F, 0.04F, -0.01F}};
  grid.integration = integration;
  grid.x_lower.kind = cm::GridBoundaryKind::periodic;
  grid.x_upper.kind = cm::GridBoundaryKind::periodic;
  grid.z_lower.kind = cm::GridBoundaryKind::fixed;
  grid.z_lower.values = {0.9F, 1.1F};
  grid.z_upper.kind = cm::GridBoundaryKind::fixed;
  grid.z_upper.values = {1.2F, 1.0F};
  cm::SignalGridAffineReaction reaction;
  reaction.source_rates.resize(grid.level_count());
  reaction.loss_rates.resize(grid.level_count());
  for (std::size_t index = 0; index < grid.level_count(); ++index) {
    reaction.source_rates[index] = 0.004F * static_cast<float>(index % 7);
    reaction.loss_rates[index] = 0.003F * static_cast<float>(index % 11);
  }
  grid.reaction = std::move(reaction);
  if (masked) {
    std::vector<std::uint8_t> obstacles(grid.site_count(), 0);
    const auto solid_site = [&](std::uint32_t x, std::uint32_t y, std::uint32_t z) {
      return (static_cast<std::size_t>(x) * grid.shape.y * grid.shape.z) +
             (static_cast<std::size_t>(y) * grid.shape.z) + z;
    };
    obstacles[solid_site(4, 3, 2)] = 1;
    obstacles[solid_site(2, 1, 1)] = 1;
    for (std::size_t signal = 0; signal < grid.signal_count; ++signal) {
      for (std::size_t site = 0; site < grid.site_count(); ++site) {
        if (obstacles[site] != 0) {
          grid.reaction->source_rates[(signal * grid.site_count()) + site] = 0.0F;
          grid.reaction->loss_rates[(signal * grid.site_count()) + site] = 0.0F;
        }
      }
    }
    grid.obstacles = std::move(obstacles);
  }
  std::vector<float> levels(grid.level_count());
  for (std::size_t index = 0; index < levels.size(); ++index) {
    levels[index] = grid.solid_site(index % grid.site_count())
                        ? 0.0F
                        : 1.0F + (0.002F * static_cast<float>(index % 113));
  }
  simulation.configure_signal_grid(grid, std::move(levels));

  for (std::size_t index = 0; index < 513; ++index) {
    const auto x = static_cast<float>(index % 17) / 16.0F;
    const auto y = static_cast<float>((index * 7) % 19) / 18.0F;
    const auto z = static_cast<float>((index * 11) % 23) / 22.0F;
    cm::CellInit cell;
    cell.position = {
        grid.origin.x + (x * grid.spacing.x * static_cast<float>(grid.shape.x - 1)),
        grid.origin.y + (y * grid.spacing.y * static_cast<float>(grid.shape.y - 1)),
        grid.origin.z + (z * grid.spacing.z * static_cast<float>(grid.shape.z - 1)),
    };
    cell.length = 1.5F + (0.01F * static_cast<float>(index % 31));
    cell.radius = 0.3F + (0.005F * static_cast<float>(index % 7));
    cell.growth_rate = 0.01F * static_cast<float>(index % 5);
    cell.cell_type = static_cast<std::int32_t>(index % 4);
    cell.species = {
        0.5F + (0.01F * static_cast<float>(index % 29)),
        0.75F + (0.02F * static_cast<float>(index % 17)),
        1.0F + (0.015F * static_cast<float>(index % 13)),
    };
    simulation.add_cell(cell);
  }

  using enum cm::RateOp;
  std::vector<cm::RateInstruction> instructions{
      operation(species, 0),
      operation(species, 1),
      operation(species, 2),
      operation(signal, 0),
      operation(signal, 1),
      operation(constant, 0, 0, 0.02F),
      operation(constant, 0, 0, -0.01F),
      operation(multiply, 3, 5),
      operation(add, 0, 7),
      operation(multiply, 1, 6),
      operation(add, 2, 9),
      operation(multiply, 0, 5),
      operation(multiply, 4, 6),
  };
  simulation.set_coupled_rate_plan(
      cm::CoupledRatePlan(3, 2, std::move(instructions), {8, 9, 10}, {11, 12}));
  return simulation;
}

void assert_close(const cm::Simulation& actual, const cm::Simulation& expected) {
  assert(actual.time() == expected.time());
  const auto actual_cells = actual.cells();
  const auto expected_cells = expected.cells();
  assert(actual_cells.size() == expected_cells.size());
  for (std::size_t cell = 0; cell < actual_cells.size(); ++cell) {
    assert(std::abs(actual_cells[cell].length - expected_cells[cell].length) <= 2.0e-6F);
    for (std::size_t species = 0; species < actual_cells[cell].species.size(); ++species) {
      assert(std::abs(actual_cells[cell].species[species] -
                      expected_cells[cell].species[species]) <= 2.0e-5F);
    }
  }
  const auto actual_grid = actual.signal_levels();
  const auto expected_grid = expected.signal_levels();
  assert(actual_grid.size() == expected_grid.size());
  for (std::size_t index = 0; index < actual_grid.size(); ++index) {
    assert(std::abs(actual_grid[index] - expected_grid[index]) <= 1.0e-4F);
  }
}

void run_case(cm::SignalIntegrationKind integration, float dt, bool masked = false) {
  auto source = make_reference(integration, masked);
  const auto checkpoint = source.checkpoint();
  cm::Simulation expected(cm::BackendKind::cpu, checkpoint);
  expected.step(dt);

  cm::test::for_each_backend_device([&](cm::BackendKind backend, std::uint32_t device_index) {
    cm::Simulation candidate(backend, checkpoint, device_index);
    if (!candidate.supports(cm::BackendFeature::coupled_rates)) {
      std::cout << "backend " << static_cast<int>(backend)
                << " does not advertise coupled rates; skipping\n";
      return;
    }
    candidate.step(dt);
    assert_close(candidate, expected);
    assert(candidate.last_signal_solve_report().has_value());
    assert(candidate.last_signal_solve_report()->converged);
  });
}

}  // namespace

int main() {
  run_case(cm::SignalIntegrationKind::forward_euler, 0.01F);
  run_case(cm::SignalIntegrationKind::crank_nicolson, 0.5F);
  run_case(cm::SignalIntegrationKind::forward_euler, 0.01F, true);
  run_case(cm::SignalIntegrationKind::crank_nicolson, 0.5F, true);
  return 0;
}
