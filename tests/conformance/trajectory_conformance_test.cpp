#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "backend_devices.hpp"
#include "cm/simulation.hpp"

namespace {

constexpr float geometry_tolerance = 2.0e-3F;
constexpr float species_tolerance = 2.0e-4F;
constexpr float signal_tolerance = 5.0e-4F;
constexpr std::array time_steps{0.01F, 0.015F, 0.02F};

cm::RateInstruction operation(cm::RateOp op, std::uint32_t first = 0, std::uint32_t second = 0,
                              float value = 0.0F) {
  return {.operation = op, .first = first, .second = second, .value = value};
}

cm::Simulation make_simulation(cm::BackendKind backend, std::uint32_t device_index) {
  cm::Simulation simulation(backend, 8, 2, device_index);

  cm::SignalGridSpec grid;
  grid.signal_count = 1;
  grid.shape = {.x = 9, .y = 9, .z = 5};
  grid.origin = {-4.0F, -4.0F, -2.0F};
  grid.spacing = {1.0F, 1.0F, 1.0F};
  grid.diffusion = {0.02F};
  grid.advection = {{0.01F, -0.005F, 0.0F}};
  std::vector<float> levels(grid.level_count());

  for (std::size_t index = 0; index < levels.size(); ++index) {
    levels[index] = 0.5F + 0.001F * static_cast<float>(index % 37);
  }

  simulation.configure_signal_grid(grid, std::move(levels));

  const std::array positions{
      cm::Vec3{0.0F, 0.0F, 0.0F},
      cm::Vec3{0.0F, 0.65F, 0.0F},
      cm::Vec3{0.2F, 1.3F, 0.1F},
  };
  const std::array directions{
      cm::Vec3{1.0F, 0.0F, 0.0F},
      cm::Vec3{1.0F, 0.1F, 0.0F},
      cm::Vec3{0.95F, -0.1F, 0.08F},
  };

  for (std::size_t index = 0; index < positions.size(); ++index) {
    cm::CellInit cell;
    cell.position = positions[index];
    cell.direction = directions[index];
    cell.length = 3.0F + 0.2F * static_cast<float>(index);
    cell.radius = 0.45F;
    cell.growth_rate = 0.02F + 0.01F * static_cast<float>(index);
    cell.cell_type = static_cast<std::int32_t>(index);
    cell.fixed = index == 0;
    cell.species = {
        0.5F + 0.1F * static_cast<float>(index),
        0.8F - 0.05F * static_cast<float>(index),
    };
    simulation.add_cell(cell);
  }

  cm::PlaneConstraintInit plane;
  plane.point = {0.0F, 0.4F, 0.0F};
  plane.inward_normal = {0.0F, 1.0F, 0.0F};
  plane.coefficient = 1.25F;
  simulation.add_plane_constraint(plane);

  using enum cm::RateOp;
  std::vector<cm::RateInstruction> instructions{
      operation(species, 0),
      operation(species, 1),
      operation(signal, 0),
      operation(constant, 0, 0, 0.015F),
      operation(constant, 0, 0, -0.01F),
      operation(multiply, 2, 3),
      operation(add, 0, 5),
      operation(multiply, 1, 4),
      operation(constant, 0, 0, 0.02F),
      operation(multiply, 0, 8),
  };
  simulation.set_coupled_rate_plan(cm::CoupledRatePlan(2, 1, std::move(instructions), {6, 7}, {9}));

  return simulation;
}

bool close(float actual, float expected, float tolerance) {
  return std::abs(actual - expected) <= tolerance + tolerance * std::abs(expected);
}

void compare_cells(const cm::Simulation& actual, const cm::Simulation& expected) {
  const auto actual_cells = actual.cells();
  const auto expected_cells = expected.cells();
  assert(actual_cells.size() == expected_cells.size());

  for (std::size_t index = 0; index < expected_cells.size(); ++index) {
    const auto& left = actual_cells[index];
    const auto& right = expected_cells[index];
    assert(left.id == right.id);
    assert(left.slot == right.slot);
    assert(left.cell_type == right.cell_type);
    assert(left.fixed == right.fixed);
    assert(left.growth_rate == right.growth_rate);
    assert(close(left.position.x, right.position.x, geometry_tolerance));
    assert(close(left.position.y, right.position.y, geometry_tolerance));
    assert(close(left.position.z, right.position.z, geometry_tolerance));
    assert(close(left.direction.x, right.direction.x, geometry_tolerance));
    assert(close(left.direction.y, right.direction.y, geometry_tolerance));
    assert(close(left.direction.z, right.direction.z, geometry_tolerance));
    assert(close(left.length, right.length, geometry_tolerance));
    assert(left.radius == right.radius);
    assert(left.species.size() == right.species.size());

    for (std::size_t species = 0; species < right.species.size(); ++species) {
      assert(close(left.species[species], right.species[species], species_tolerance));
    }
  }
}

void compare_signals(const cm::Simulation& actual, const cm::Simulation& expected) {
  const auto actual_levels = actual.signal_levels();
  const auto expected_levels = expected.signal_levels();
  assert(actual_levels.size() == expected_levels.size());

  for (std::size_t index = 0; index < expected_levels.size(); ++index) {
    assert(close(actual_levels[index], expected_levels[index], signal_tolerance));
  }
}

void compare_state(const cm::Simulation& actual, const cm::Simulation& expected) {
  assert(actual.time() == expected.time());
  compare_cells(actual, expected);
  compare_signals(actual, expected);
  actual.validate();
  expected.validate();
}

void run_trajectory(cm::BackendKind backend, std::uint32_t device_index) {
  auto reference = make_simulation(cm::BackendKind::cpu, 0);
  auto candidate = make_simulation(backend, device_index);
  cm::MechanicsParameters mechanics;
  mechanics.residual_rms_tolerance = 2.0e-5F;
  mechanics.max_iterations = 256;

  for (std::size_t step = 0; step < time_steps.size(); ++step) {
    reference.step(time_steps[step]);
    candidate.step(time_steps[step]);
    compare_state(candidate, reference);
    assert(reference.last_signal_solve_report()->converged);
    assert(candidate.last_signal_solve_report()->converged);

    const auto expected_solve = reference.relax_cell_mechanics(mechanics);
    const auto actual_solve = candidate.relax_cell_mechanics(mechanics);
    assert(expected_solve.report.status == cm::SolverStatus::converged);
    assert(actual_solve.report.status == cm::SolverStatus::converged);
    assert(actual_solve.report.final_residual_rms <= mechanics.residual_rms_tolerance);
    compare_state(candidate, reference);

    if (step == 0) {
      const auto expected_daughters = reference.divide_equal(2);
      const auto actual_daughters = candidate.divide_equal(2);
      assert(actual_daughters == expected_daughters);
      assert(candidate.lineage_parent(actual_daughters.first) == 2);
      assert(candidate.lineage_parent(actual_daughters.second) == 2);
      compare_state(candidate, reference);
    }
  }

  const auto expected_contacts = reference.find_cell_contacts();
  const auto actual_contacts = candidate.find_cell_contacts();
  assert(actual_contacts.size() == expected_contacts.size());
}

}  // namespace

int main() {
  cm::test::for_each_backend_device(run_trajectory);

  return 0;
}
