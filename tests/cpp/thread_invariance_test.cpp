#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "microsimulator/simulation.hpp"
#include "microsimulator/threads.hpp"

// Every scenario here is large enough that each threaded loop it reaches is
// split into several pieces, and must produce the same bits on any number of
// threads.

namespace {

using microsimulator::BackendKind;
using microsimulator::CellInit;
using microsimulator::RateInstruction;
using microsimulator::RateOp;
using microsimulator::SignalGridSpec;
using microsimulator::Simulation;

void append(std::vector<float>& values, const microsimulator::SignalGridVelocityField& field) {
  values.insert(values.end(), field.x_faces.begin(), field.x_faces.end());
  values.insert(values.end(), field.y_faces.begin(), field.y_faces.end());
  values.insert(values.end(), field.z_faces.begin(), field.z_faces.end());
}

void append(std::vector<float>& values, const Simulation& simulation) {
  for (const auto& cell : simulation.cells()) {
    values.insert(values.end(), {cell.position.x, cell.position.y, cell.position.z,
                                 cell.direction.x, cell.direction.y, cell.direction.z,
                                 cell.length});
    values.insert(values.end(), cell.species.begin(), cell.species.end());
  }
}

// Rows of nearly touching cells, each turned by a small deterministic angle.
void add_packed_cells(Simulation& simulation, std::uint32_t columns, std::uint32_t rows,
                      microsimulator::Vec3 origin, std::vector<float> species = {}) {
  std::uint32_t state = 2463534242U;
  for (std::uint32_t row = 0; row < rows; ++row) {
    for (std::uint32_t column = 0; column < columns; ++column) {
      state = (state * 1664525U) + 1013904223U;
      const auto angle = 0.1F * (static_cast<float>(state >> 8U) / 16777216.0F - 0.5F);
      CellInit cell;
      cell.position = {origin.x + 2.98F * static_cast<float>(column) +
                           (row % 2 == 0 ? 0.0F : 1.4F),
                       origin.y + 0.98F * static_cast<float>(row), origin.z};
      cell.direction = {std::cos(angle), std::sin(angle), 0.0F};
      cell.length = 2.0F;
      cell.radius = 0.5F;
      cell.growth_rate = 1.0F + 0.2F * static_cast<float>(column % 3);
      cell.species = species;
      simulation.add_cell(cell);
    }
  }
}

std::vector<float> contact_dynamics() {
  Simulation simulation;
  add_packed_cells(simulation, 30, 30, {0.0F, 0.0F, 0.0F});
  microsimulator::PlaneConstraintInit floor;
  floor.point = {0.0F, -0.6F, 0.0F};
  floor.inward_normal = {0.0F, 1.0F, 0.0F};
  simulation.add_plane_constraint(floor);
  std::vector<float> values;
  for (int step = 0; step < 3; ++step) {
    const auto result = simulation.step(0.02F, {}, {}, {});
    assert(result.report.max_rows > 2000);
    values.insert(values.end(), result.axial_loads.begin(), result.axial_loads.end());
    values.push_back(static_cast<float>(result.report.iterations));
  }
  append(values, simulation);
  const auto graph = simulation.find_cell_contacts({});
  for (const auto& contact : graph.contacts()) {
    values.insert(values.end(),
                  {static_cast<float>(contact.first_id), static_cast<float>(contact.second_id),
                   contact.signed_separation, contact.normal.x, contact.normal.y});
  }
  return values;
}

SignalGridSpec grid(std::uint32_t nx, std::uint32_t ny, std::uint32_t nz) {
  SignalGridSpec spec;
  spec.signal_count = 2;
  spec.shape = {.x = nx, .y = ny, .z = nz};
  spec.spacing = {1.0F, 1.0F, 1.0F};
  spec.diffusion = {0.6F, 0.2F};
  spec.advection = {{0.1F, 0.0F, 0.0F}, {0.0F, -0.05F, 0.0F}};
  spec.x_lower = {.kind = microsimulator::GridBoundaryKind::fixed, .values = {2.0F, 0.5F}};
  spec.obstacles.assign(spec.site_count(), 0);
  for (std::size_t site = 0; site < spec.site_count(); site += 37) {
    spec.obstacles[site] = 1;
  }
  return spec;
}

std::vector<float> initial_levels(const SignalGridSpec& spec) {
  std::vector<float> levels(spec.level_count());
  for (std::size_t index = 0; index < levels.size(); ++index) {
    levels[index] = spec.solid_site(index % spec.site_count()) ? 0.0F
                                                              : static_cast<float>(index % 13);
  }
  return levels;
}

std::vector<float> signal_grid(microsimulator::SignalIntegrationKind integration, float dt) {
  auto spec = grid(28, 26, 24);
  spec.integration = integration;
  Simulation simulation;
  simulation.configure_signal_grid(spec, initial_levels(spec));
  for (int step = 0; step < 3; ++step) {
    simulation.step(dt);
  }
  auto values = simulation.signal_levels();
  if (const auto report = simulation.last_signal_solve_report()) {
    values.push_back(static_cast<float>(report->iterations));
    values.push_back(report->residual_rms);
  }
  return values;
}

std::vector<float> coupled_rates(microsimulator::SignalIntegrationKind integration) {
  auto spec = grid(40, 40, 4);
  spec.obstacles.clear();
  spec.integration = integration;
  Simulation simulation(BackendKind::cpu, 0, 2);
  simulation.configure_signal_grid(spec, initial_levels(spec));
  add_packed_cells(simulation, 12, 36, {1.5F, 1.5F, 1.5F}, {1.0F, 3.0F});
  simulation.set_coupled_rate_plan(microsimulator::CoupledRatePlan(
      2, 2,
      {
          {.operation = RateOp::signal, .first = 0},
          {.operation = RateOp::species, .first = 1},
          {.operation = RateOp::constant, .value = -0.02F},
          {.operation = RateOp::multiply, .first = 0, .second = 2},
          {.operation = RateOp::multiply, .first = 0, .second = 1},
          {.operation = RateOp::cell_volume},
          {.operation = RateOp::multiply, .first = 3, .second = 5},
          {.operation = RateOp::negate, .first = 3},
      },
      {4, 7}, {6, 3}));
  for (int step = 0; step < 3; ++step) {
    simulation.step(0.01F);
  }
  auto values = simulation.signal_levels();
  append(values, simulation);
  return values;
}

std::vector<float> species_rates() {
  Simulation simulation(BackendKind::cpu, 0, 2);
  add_packed_cells(simulation, 12, 36, {0.0F, 0.0F, 0.0F}, {1.0F, 3.0F});
  simulation.set_species_rate_plan(microsimulator::SpeciesRatePlan(
      2,
      {
          {.operation = RateOp::species, .first = 0},
          {.operation = RateOp::species, .first = 1},
          {.operation = RateOp::multiply, .first = 0, .second = 1},
          {.operation = RateOp::cell_length},
          {.operation = RateOp::subtract, .first = 3, .second = 2},
      },
      {4, 2}));
  for (int step = 0; step < 3; ++step) {
    simulation.step(0.01F);
  }
  std::vector<float> values;
  append(values, simulation);
  return values;
}

SignalGridSpec duct(std::uint32_t nx, std::uint32_t ny, std::uint32_t nz) {
  SignalGridSpec spec;
  spec.signal_count = 1;
  spec.shape = {.x = nx, .y = ny, .z = nz};
  spec.spacing = {0.7F, 1.1F, 0.6F};
  spec.diffusion = {0.0F};
  spec.advection = {{0.0F, 0.0F, 0.0F}};
  spec.y_lower = {.kind = microsimulator::GridBoundaryKind::fixed, .values = {0.0F}};
  spec.y_upper = {.kind = microsimulator::GridBoundaryKind::fixed, .values = {0.0F}};
  spec.obstacles.assign(spec.site_count(), 0);
  for (std::uint32_t x = 4; x < 8; ++x) {
    for (std::uint32_t z = 0; z < nz; ++z) {
      spec.obstacles[(static_cast<std::size_t>(x) * ny + ny / 2) * nz + z] = 1;
    }
  }
  return spec;
}

std::vector<float> depth_averaged_flow() {
  const auto spec = duct(80, 72, 2);
  std::vector<float> mobility(spec.site_count());
  for (std::size_t site = 0; site < mobility.size(); ++site) {
    // Mobility is uniform through the depth of each column.
    const auto column = site / spec.shape.z;
    mobility[site] = spec.solid_site(site) ? 0.0F : 0.4F + 0.01F * static_cast<float>(column % 17);
  }
  microsimulator::DepthAveragedFlowParameters parameters;
  parameters.mean_inlet_speed = 3.5F;
  parameters.relative_tolerance = 1.0e-6F;
  Simulation simulation;
  const auto result = simulation.solve_depth_averaged_flow(spec, mobility, parameters);
  std::vector<float> values{static_cast<float>(result.report.iterations),
                            result.report.relative_residual};
  append(values, result.field);
  return values;
}

std::vector<float> resolved_flow() {
  const auto spec = duct(14, 16, 10);
  std::vector<float> drag(spec.site_count(), 0.0F);
  for (std::size_t site = 0; site < drag.size(); site += 5) {
    drag[site] = 12.0F;
  }
  microsimulator::ResolvedFlowParameters parameters;
  parameters.mean_inlet_speed = 2.0F;
  parameters.relative_tolerance = 1.0e-5F;
  parameters.inner_relative_tolerance = 1.0e-5F;
  Simulation simulation;
  const auto result = simulation.solve_resolved_flow(spec, drag, parameters);
  std::vector<float> values{static_cast<float>(result.report.outer_iterations),
                            static_cast<float>(result.report.inner_iterations),
                            result.report.relative_residual};
  append(values, result.field);
  return values;
}

template <typename Scenario>
void require_thread_invariance(const Scenario& scenario) {
  microsimulator::set_cpu_thread_count(1);
  const auto reference = scenario();
  assert(!reference.empty());
  assert(std::ranges::all_of(reference, [](float value) { return std::isfinite(value); }));
  for (const std::size_t threads : {2U, 5U, 8U}) {
    microsimulator::set_cpu_thread_count(threads);
    assert(scenario() == reference);
  }
}

}  // namespace

int main() {
  using microsimulator::SignalIntegrationKind;
  require_thread_invariance(contact_dynamics);
  require_thread_invariance([] { return signal_grid(SignalIntegrationKind::forward_euler, 0.05F); });
  require_thread_invariance([] { return signal_grid(SignalIntegrationKind::crank_nicolson, 0.5F); });
  require_thread_invariance([] { return signal_grid(SignalIntegrationKind::backward_euler, 0.5F); });
  require_thread_invariance([] { return coupled_rates(SignalIntegrationKind::forward_euler); });
  require_thread_invariance([] { return coupled_rates(SignalIntegrationKind::crank_nicolson); });
  require_thread_invariance(species_rates);
  require_thread_invariance(depth_averaged_flow);
  require_thread_invariance(resolved_flow);
  return 0;
}
