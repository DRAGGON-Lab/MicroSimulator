#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "microsimulator/mechanics.hpp"
#include "microsimulator/simulation.hpp"

namespace {

bool close(float actual, float expected, float tolerance = 1.0e-5F) {
  return std::abs(actual - expected) <= tolerance;
}

void test_integration_applies_declared_geometry_semantics() {
  microsimulator::WorldState state;
  microsimulator::CellInit first;
  first.length = 4.0F;
  const auto first_id = state.add_cell(first);
  microsimulator::CellInit second = first;
  second.position = {2.0F, 0.0F, 0.0F};
  const auto second_id = state.add_cell(second);

  microsimulator::MechanicsSolveResult result;
  result.corrections = {
      microsimulator::CellCorrection{{1.0F, 2.0F, 3.0F}, {0.0F, 0.0F, 0.2F}, -1.0F},
      microsimulator::CellCorrection{{-1.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.0F}, 0.25F},
  };
  microsimulator::MechanicsIntegrationParameters parameters;
  parameters.max_rotation_radians = 0.1F;
  const float desired_increments[] = {0.5F, 0.5F};
  microsimulator::integrate_mechanics_result(state, result, parameters, desired_increments);

  const auto integrated_first = state.cell(first_id);
  assert(close(integrated_first.position.x, 1.0F));
  assert(close(integrated_first.position.y, 2.0F));
  assert(close(integrated_first.position.z, 3.0F));
  assert(close(integrated_first.direction.x, std::cos(0.1F)));
  assert(close(integrated_first.direction.y, std::sin(0.1F)));
  assert(close(integrated_first.length, 4.0F));

  const auto integrated_second = state.cell(second_id);
  assert(close(integrated_second.position.x, 1.0F));
  assert(close(integrated_second.length, 4.75F));
  state.validate();
}

void test_validation_is_atomic_and_requires_convergence() {
  microsimulator::WorldState state;
  microsimulator::CellInit cell;
  const auto id = state.add_cell(cell);

  microsimulator::MechanicsSolveResult invalid;
  invalid.corrections = {microsimulator::CellCorrection{}};
  invalid.corrections[0].translation.x = std::numeric_limits<float>::quiet_NaN();
  bool rejected = false;
  try {
    microsimulator::integrate_mechanics_result(state, invalid);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
  assert(close(state.cell(id).position.x, 0.0F));

  microsimulator::MechanicsSolveResult unconverged;
  unconverged.corrections = {microsimulator::CellCorrection{}};
  unconverged.report.status = microsimulator::SolverStatus::iteration_limit;
  rejected = false;
  try {
    microsimulator::integrate_mechanics_result(state, unconverged);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);
  assert(close(state.cell(id).position.x, 0.0F));
}

void test_relaxation_cannot_create_biomass() {
  microsimulator::WorldState state;
  const auto id = state.add_cell(microsimulator::CellInit{});
  const auto before = state.cell(id).length;
  microsimulator::MechanicsSolveResult result;
  result.corrections = {microsimulator::CellCorrection{{}, {}, 0.5F}};
  microsimulator::integrate_mechanics_result(state, result);
  assert(state.cell(id).length == before);
}

void test_fixed_cell_integration_only_applies_declared_growth() {
  microsimulator::WorldState state;
  microsimulator::CellInit cell;
  cell.position = {1.0F, 2.0F, 3.0F};
  cell.direction = {1.0F, 0.0F, 0.0F};
  cell.length = 4.0F;
  cell.fixed = true;
  const auto id = state.add_cell(cell);

  microsimulator::MechanicsSolveResult result;
  result.corrections = {microsimulator::CellCorrection{
      .translation = {8.0F, 7.0F, 6.0F},
      .rotation = {0.0F, 0.0F, 1.0F},
      .length = -2.0F,
  }};
  const float desired_increments[] = {0.5F};
  microsimulator::integrate_mechanics_result(state, result, {}, desired_increments);

  const auto integrated = state.cell(id);
  assert(close(integrated.position.x, 1.0F));
  assert(close(integrated.position.y, 2.0F));
  assert(close(integrated.position.z, 3.0F));
  assert(close(integrated.direction.x, 1.0F));
  assert(close(integrated.direction.y, 0.0F));
  assert(close(integrated.direction.z, 0.0F));
  assert(close(integrated.length, 4.5F));
}

void test_simulation_relaxation_reduces_penetration() {
  microsimulator::Simulation simulation;
  microsimulator::CellInit first;
  first.length = 4.0F;
  microsimulator::CellInit second = first;
  second.position.y = 0.8F;
  const auto first_id = simulation.add_cell(first);
  const auto second_id = simulation.add_cell(second);
  const auto before = simulation.find_cell_contacts();
  assert(before.size() == 2);

  const auto result = simulation.relax_cell_mechanics();
  assert(result.report.status == microsimulator::SolverStatus::converged);
  const auto after = simulation.find_cell_contacts();
  assert(after.size() == 2);
  assert(after.contacts()[0].signed_separation > before.contacts()[0].signed_separation);
  assert(simulation.cell(first_id).position.y < 0.0F);
  assert(simulation.cell(second_id).position.y > 0.8F);
  assert(close(simulation.cell(first_id).length, 4.0F));
  assert(close(simulation.cell(second_id).length, 4.0F));
  simulation.validate();
}

float deepest_wall_penetration(microsimulator::Simulation& simulation) {
  auto deepest = 0.0F;
  const auto graph = simulation.find_external_contacts();
  for (const auto& contact : graph.contacts()) {
    deepest = std::min(deepest, contact.signed_separation);
  }
  return deepest;
}

void test_simulation_relaxation_leaves_no_cell_inside_a_wall() {
  microsimulator::Simulation simulation;
  microsimulator::CellInit cell;
  cell.position = {1.3F, 0.0F, 0.0F};
  cell.direction = {0.1F, 1.0F, 0.0F};
  cell.length = 1.0F;
  cell.radius = 0.5F;
  simulation.add_cell(cell);
  microsimulator::BoxConstraintInit wall;
  wall.half_extents = {1.0F, 2.0F, 3.0F};
  simulation.add_box_constraint(wall);
  assert(deepest_wall_penetration(simulation) < -0.2F);

  const auto result = simulation.relax_cell_mechanics();
  assert(result.report.status == microsimulator::SolverStatus::converged);
  assert(deepest_wall_penetration(simulation) >= -1.0e-4F);
  simulation.validate();
}

void test_simulation_relaxation_cannot_carry_a_cell_through_a_thin_wall() {
  microsimulator::Simulation simulation;
  // A wall barely wider than the cell's radius, with a fixed neighbor that
  // overlaps the cell almost completely from the side away from the wall.
  microsimulator::BoxConstraintInit wall;
  wall.center = {0.3F, 0.0F, 0.0F};
  wall.half_extents = {0.3F, 5.0F, 5.0F};
  simulation.add_box_constraint(wall);
  microsimulator::CellInit cell;
  cell.position = {-0.55F, 0.0F, 0.0F};
  cell.direction = {0.0F, 1.0F, 0.0F};
  cell.length = 1.0F;
  cell.radius = 0.5F;
  const auto cell_id = simulation.add_cell(cell);
  microsimulator::CellInit neighbor = cell;
  neighbor.position = {-0.6F, 0.0F, 0.0F};
  neighbor.fixed = true;
  simulation.add_cell(neighbor);

  microsimulator::MechanicsParameters parameters;
  parameters.gamma = 1.0e4F;
  const auto result = simulation.relax_cell_mechanics(parameters);
  assert(result.report.status == microsimulator::SolverStatus::converged);
  // The correction alone would put the centerline past the wall's mid-plane,
  // where the nearest face is the far one.
  assert(result.corrections[0].translation.x > 0.85F);
  assert(simulation.cell(cell_id).position.x < -0.49F);
  assert(deepest_wall_penetration(simulation) >= -1.0e-4F);
}

// Two parallel rods side by side, overlapping by `overlap`.
microsimulator::Simulation overlapping_pair(float overlap) {
  microsimulator::Simulation simulation(microsimulator::BackendKind::cpu, 0, 0, 0);
  microsimulator::CellInit cell;
  cell.length = 2.0F;
  cell.radius = 0.5F;
  cell.growth_rate = 1.0F;
  simulation.add_cell(cell);
  cell.position = {0.0F, 1.0F - overlap, 0.0F};
  simulation.add_cell(cell);
  return simulation;
}

void test_growth_limit_scales_elongation_by_compression() {
  microsimulator::GrowthLimitParameters limit;
  limit.free_overlap = 0.1F;
  limit.stall_overlap = 0.5F;
  const float dt = 0.1F;
  const float full = 2.0F * dt;
  // Overlap in radii, and the fraction of the full elongation it leaves.
  const float cases[][2] = {{0.0F, 1.0F}, {0.1F, 1.0F}, {0.3F, 0.5F}, {0.5F, 0.0F}, {0.9F, 0.0F}};
  for (const auto& [overlap_radii, expected] : cases) {
    auto simulation = overlapping_pair(overlap_radii * 0.5F);
    simulation.step(dt, limit);
    for (const auto& cell : simulation.cells()) {
      assert(close(cell.length, 2.0F + (expected * full), 1.0e-4F));
    }
    assert(close(static_cast<float>(simulation.time()), dt));
  }

  // Cells out of contact grow as an unlimited step grows them.
  auto apart = overlapping_pair(-1.0F);
  auto reference = overlapping_pair(-1.0F);
  apart.step(dt, limit);
  reference.step(dt);
  assert(apart.cells()[0].length == reference.cells()[0].length);

  // The limit never turns shrinkage into growth or holds it back.
  auto shrinking = overlapping_pair(0.45F);
  for (const auto& cell : shrinking.cells()) {
    shrinking.set_cell_attributes(cell.id, -1.0F, cell.cell_type);
  }
  shrinking.step(dt, limit);
  assert(close(shrinking.cells()[0].length, 2.0F - full, 1.0e-4F));
}

void test_growth_limit_rejects_invalid_parameters() {
  auto simulation = overlapping_pair(0.1F);
  const auto rejected = [&](float free_overlap, float stall_overlap) {
    microsimulator::GrowthLimitParameters limit;
    limit.free_overlap = free_overlap;
    limit.stall_overlap = stall_overlap;
    try {
      simulation.step(0.1F, limit);
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  assert(rejected(-0.1F, 0.4F));
  assert(rejected(0.4F, 0.4F));
  assert(rejected(0.1F, std::numeric_limits<float>::infinity()));
  assert(rejected(std::numeric_limits<float>::quiet_NaN(), 0.4F));
  // A rejected step leaves the cells and the clock as they were.
  assert(simulation.cells()[0].length == 2.0F);
  assert(simulation.time() == 0.0);
}

}  // namespace

int main() {
  test_relaxation_cannot_create_biomass();
  test_integration_applies_declared_geometry_semantics();
  test_validation_is_atomic_and_requires_convergence();
  test_fixed_cell_integration_only_applies_declared_growth();
  test_simulation_relaxation_reduces_penetration();
  test_simulation_relaxation_leaves_no_cell_inside_a_wall();
  test_simulation_relaxation_cannot_carry_a_cell_through_a_thin_wall();
  test_growth_limit_scales_elongation_by_compression();
  test_growth_limit_rejects_invalid_parameters();
  return 0;
}
