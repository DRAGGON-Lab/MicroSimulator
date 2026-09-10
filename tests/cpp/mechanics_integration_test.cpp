#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "cm/mechanics.hpp"
#include "cm/simulation.hpp"

namespace {

bool close(float actual, float expected, float tolerance = 1.0e-5F) {
  return std::abs(actual - expected) <= tolerance;
}

void test_integration_applies_declared_geometry_semantics() {
  cm::WorldState state;
  cm::CellInit first;
  first.length = 4.0F;
  const auto first_id = state.add_cell(first);
  cm::CellInit second = first;
  second.position = {2.0F, 0.0F, 0.0F};
  const auto second_id = state.add_cell(second);

  cm::MechanicsSolveResult result;
  result.corrections = {
      cm::CellCorrection{{1.0F, 2.0F, 3.0F}, {0.0F, 0.0F, 0.2F}, -1.0F},
      cm::CellCorrection{{-1.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.0F}, 0.25F},
  };
  cm::MechanicsIntegrationParameters parameters;
  parameters.max_rotation_radians = 0.1F;
  const float desired_increments[] = {0.5F, 0.5F};
  cm::integrate_mechanics_result(state, result, parameters, desired_increments);

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
  cm::WorldState state;
  cm::CellInit cell;
  const auto id = state.add_cell(cell);

  cm::MechanicsSolveResult invalid;
  invalid.corrections = {cm::CellCorrection{}};
  invalid.corrections[0].translation.x = std::numeric_limits<float>::quiet_NaN();
  bool rejected = false;
  try {
    cm::integrate_mechanics_result(state, invalid);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
  assert(close(state.cell(id).position.x, 0.0F));

  cm::MechanicsSolveResult unconverged;
  unconverged.corrections = {cm::CellCorrection{}};
  unconverged.report.status = cm::SolverStatus::iteration_limit;
  rejected = false;
  try {
    cm::integrate_mechanics_result(state, unconverged);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);
  assert(close(state.cell(id).position.x, 0.0F));
}

void test_relaxation_cannot_create_biomass() {
  cm::WorldState state;
  const auto id = state.add_cell(cm::CellInit{});
  const auto before = state.cell(id).length;
  cm::MechanicsSolveResult result;
  result.corrections = {cm::CellCorrection{{}, {}, 0.5F}};
  cm::integrate_mechanics_result(state, result);
  assert(state.cell(id).length == before);
}

void test_fixed_cell_integration_only_applies_declared_growth() {
  cm::WorldState state;
  cm::CellInit cell;
  cell.position = {1.0F, 2.0F, 3.0F};
  cell.direction = {1.0F, 0.0F, 0.0F};
  cell.length = 4.0F;
  cell.fixed = true;
  const auto id = state.add_cell(cell);

  cm::MechanicsSolveResult result;
  result.corrections = {cm::CellCorrection{
      .translation = {8.0F, 7.0F, 6.0F},
      .rotation = {0.0F, 0.0F, 1.0F},
      .length = -2.0F,
  }};
  const float desired_increments[] = {0.5F};
  cm::integrate_mechanics_result(state, result, {}, desired_increments);

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
  cm::Simulation simulation;
  cm::CellInit first;
  first.length = 4.0F;
  cm::CellInit second = first;
  second.position.y = 0.8F;
  const auto first_id = simulation.add_cell(first);
  const auto second_id = simulation.add_cell(second);
  const auto before = simulation.find_cell_contacts();
  assert(before.size() == 2);

  const auto result = simulation.relax_cell_mechanics();
  assert(result.report.status == cm::SolverStatus::converged);
  const auto after = simulation.find_cell_contacts();
  assert(after.size() == 2);
  assert(after.contacts()[0].signed_separation > before.contacts()[0].signed_separation);
  assert(simulation.cell(first_id).position.y < 0.0F);
  assert(simulation.cell(second_id).position.y > 0.8F);
  assert(close(simulation.cell(first_id).length, 4.0F));
  assert(close(simulation.cell(second_id).length, 4.0F));
  simulation.validate();
}

}  // namespace

int main() {
  test_relaxation_cannot_create_biomass();
  test_integration_applies_declared_geometry_semantics();
  test_validation_is_atomic_and_requires_convergence();
  test_fixed_cell_integration_only_applies_declared_growth();
  test_simulation_relaxation_reduces_penetration();
  return 0;
}
