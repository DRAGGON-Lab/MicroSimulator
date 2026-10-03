#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include "backend_devices.hpp"
#include "cm/simulation.hpp"

namespace {

constexpr float absolute_tolerance = 3.0e-4F;
constexpr float relative_tolerance = 3.0e-4F;

bool close(float actual, float expected) {
  const auto tolerance = absolute_tolerance + relative_tolerance * std::abs(expected);

  return std::abs(actual - expected) <= tolerance;
}

cm::CellId add_capsule(cm::Simulation& simulation, cm::Vec3 center, cm::Vec3 axis,
                       float length = 4.0F, float radius = 0.5F) {
  cm::CellInit cell;
  cell.position = center;
  cell.direction = axis;
  cell.length = length;
  cell.radius = radius;

  return simulation.add_cell(cell);
}

void populate_mixed_colony(cm::Simulation& simulation) {
  add_capsule(simulation, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F});
  add_capsule(simulation, {0.2F, 0.8F, 0.0F}, {1.0F, 0.2F, 0.0F});
  add_capsule(simulation, {-0.3F, 0.1F, 0.7F}, {0.0F, 1.0F, 0.3F});
  add_capsule(simulation, {4.8F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F});
}

void require_close(float actual, float expected, std::string_view scenario, std::size_t index,
                   std::string_view field) {
  if (close(actual, expected)) {
    return;
  }

  std::cerr << scenario << " correction " << index << ' ' << field << ": actual=" << actual
            << " expected=" << expected << '\n';
  std::abort();
}

void compare_corrections(const cm::MechanicsSolveResult& actual,
                         const cm::MechanicsSolveResult& expected, std::string_view scenario) {
  assert(actual.report.status == expected.report.status);
  assert(actual.report.breakdown == expected.report.breakdown);
  require_close(actual.report.initial_residual_rms, expected.report.initial_residual_rms, scenario,
                0, "initial_residual_rms");
  assert(actual.corrections.size() == expected.corrections.size());

  for (std::size_t index = 0; index < expected.corrections.size(); ++index) {
    const auto& left = actual.corrections[index];
    const auto& right = expected.corrections[index];
    require_close(left.translation.x, right.translation.x, scenario, index, "translation.x");
    require_close(left.translation.y, right.translation.y, scenario, index, "translation.y");
    require_close(left.translation.z, right.translation.z, scenario, index, "translation.z");
    require_close(left.rotation.x, right.rotation.x, scenario, index, "rotation.x");
    require_close(left.rotation.y, right.rotation.y, scenario, index, "rotation.y");
    require_close(left.rotation.z, right.rotation.z, scenario, index, "rotation.z");
    require_close(left.length, right.length, scenario, index, "length");
  }
}

void run_converged_colony(cm::BackendKind backend, std::uint32_t device_index) {
  cm::Simulation reference(cm::BackendKind::cpu);
  cm::Simulation candidate(backend, 0, 0, device_index);
  populate_mixed_colony(reference);
  populate_mixed_colony(candidate);

  cm::MechanicsParameters parameters;
  parameters.residual_rms_tolerance = 2.0e-5F;
  const auto expected = reference.solve_cell_mechanics(parameters);
  const auto actual = candidate.solve_cell_mechanics(parameters);
  assert(expected.report.status == cm::SolverStatus::converged);
  assert(actual.report.final_residual_rms <= parameters.residual_rms_tolerance);
  compare_corrections(actual, expected, "converged colony");
}

void run_iteration_limit(cm::BackendKind backend, std::uint32_t device_index) {
  cm::Simulation reference(cm::BackendKind::cpu);
  cm::Simulation candidate(backend, 0, 0, device_index);
  populate_mixed_colony(reference);
  populate_mixed_colony(candidate);

  cm::MechanicsParameters parameters;
  parameters.residual_rms_tolerance = 0.0F;
  parameters.max_iterations = 1;
  const auto expected = reference.solve_cell_mechanics(parameters);
  const auto actual = candidate.solve_cell_mechanics(parameters);
  assert(expected.report.status == cm::SolverStatus::iteration_limit);
  assert(actual.report.iterations == 1);
  compare_corrections(actual, expected, "iteration limit");
}

void run_empty_systems(cm::BackendKind backend, std::uint32_t device_index) {
  cm::Simulation empty(backend, 0, 0, device_index);
  const auto empty_result = empty.solve_cell_mechanics();
  assert(empty_result.report.status == cm::SolverStatus::converged);
  assert(empty_result.corrections.empty());

  cm::Simulation separated(backend, 0, 0, device_index);
  add_capsule(separated, {}, {1.0F, 0.0F, 0.0F});
  add_capsule(separated, {10.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F});
  const auto separated_result = separated.solve_cell_mechanics();
  assert(separated_result.report.status == cm::SolverStatus::converged);
  assert(separated_result.report.iterations == 0);

  for (const auto& correction : separated_result.corrections) {
    assert(correction.translation.x == 0.0F);
    assert(correction.translation.y == 0.0F);
    assert(correction.translation.z == 0.0F);
  }
}

void run_buffer_growth(cm::BackendKind backend, std::uint32_t device_index) {
  cm::Simulation simulation(backend, 0, 0, device_index);
  add_capsule(simulation, {}, {1.0F, 0.0F, 0.0F});
  add_capsule(simulation, {0.0F, 0.8F, 0.0F}, {1.0F, 0.0F, 0.0F});
  const auto first = simulation.solve_cell_mechanics();
  assert(first.report.status == cm::SolverStatus::converged);

  for (std::size_t index = 0; index < 7; ++index) {
    add_capsule(simulation, {0.1F * static_cast<float>(index), 0.2F, 0.1F}, {1.0F, 0.1F, 0.0F});
  }

  const auto grown = simulation.solve_cell_mechanics();
  assert(grown.report.status == cm::SolverStatus::converged);
  assert(grown.corrections.size() == 9);
}

void run_integrated_relaxation(cm::BackendKind backend, std::uint32_t device_index) {
  cm::Simulation reference(cm::BackendKind::cpu);
  cm::Simulation candidate(backend, 0, 0, device_index);
  populate_mixed_colony(reference);
  populate_mixed_colony(candidate);

  cm::MechanicsParameters parameters;
  parameters.residual_rms_tolerance = 2.0e-5F;
  const auto expected = reference.relax_cell_mechanics(parameters);
  const auto actual = candidate.relax_cell_mechanics(parameters);
  compare_corrections(actual, expected, "integrated relaxation");
  const auto expected_cells = reference.cells();
  const auto actual_cells = candidate.cells();
  assert(actual_cells.size() == expected_cells.size());

  for (std::size_t index = 0; index < expected_cells.size(); ++index) {
    assert(actual_cells[index].id == expected_cells[index].id);
    assert(close(actual_cells[index].position.x, expected_cells[index].position.x));
    assert(close(actual_cells[index].position.y, expected_cells[index].position.y));
    assert(close(actual_cells[index].position.z, expected_cells[index].position.z));
    assert(close(actual_cells[index].direction.x, expected_cells[index].direction.x));
    assert(close(actual_cells[index].direction.y, expected_cells[index].direction.y));
    assert(close(actual_cells[index].direction.z, expected_cells[index].direction.z));
    assert(close(actual_cells[index].length, expected_cells[index].length));
  }
}

void run_fixed_cell_relaxation(cm::BackendKind backend, std::uint32_t device_index) {
  cm::Simulation reference(cm::BackendKind::cpu);
  cm::Simulation candidate(backend, 0, 0, device_index);
  const auto reference_fixed = add_capsule(reference, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F});
  const auto candidate_fixed = add_capsule(candidate, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F});
  const auto reference_free = add_capsule(reference, {0.0F, 0.8F, 0.0F}, {1.0F, 0.0F, 0.0F});
  const auto candidate_free = add_capsule(candidate, {0.0F, 0.8F, 0.0F}, {1.0F, 0.0F, 0.0F});
  reference.set_cell_fixed(reference_fixed, true);
  candidate.set_cell_fixed(candidate_fixed, true);

  cm::MechanicsParameters parameters;
  parameters.residual_rms_tolerance = 2.0e-5F;
  const auto expected = reference.solve_cell_mechanics(parameters);
  const auto actual = candidate.solve_cell_mechanics(parameters);
  compare_corrections(actual, expected, "fixed-cell solve");
  assert(actual.report.status == cm::SolverStatus::converged);
  assert(actual.corrections[0].translation.x == 0.0F);
  assert(actual.corrections[0].translation.y == 0.0F);
  assert(actual.corrections[0].translation.z == 0.0F);
  assert(actual.corrections[0].rotation.x == 0.0F);
  assert(actual.corrections[0].rotation.y == 0.0F);
  assert(actual.corrections[0].rotation.z == 0.0F);
  assert(actual.corrections[0].length == 0.0F);
  assert(actual.corrections[1].translation.y > 0.0F);

  const auto expected_relaxation = reference.relax_cell_mechanics(parameters);
  const auto actual_relaxation = candidate.relax_cell_mechanics(parameters);
  compare_corrections(actual_relaxation, expected_relaxation, "fixed-cell relaxation");
  assert(candidate.cell(candidate_fixed).position.x == 0.0F);
  assert(candidate.cell(candidate_fixed).position.y == 0.0F);
  assert(candidate.cell(candidate_fixed).position.z == 0.0F);
  assert(candidate.cell(candidate_free).position.y > 0.8F);
  assert(
      close(candidate.cell(candidate_free).position.y, reference.cell(reference_free).position.y));
}

}  // namespace

int main() {
  cm::test::for_each_backend_device([](cm::BackendKind backend, std::uint32_t device_index) {
    cm::Simulation capability_probe(backend, 0, 0, device_index);

    if (!capability_probe.supports(cm::BackendFeature::cell_mechanics)) {
      return;
    }

    run_empty_systems(backend, device_index);
    run_converged_colony(backend, device_index);
    run_iteration_limit(backend, device_index);
    run_buffer_growth(backend, device_index);
    run_integrated_relaxation(backend, device_index);
    run_fixed_cell_relaxation(backend, device_index);
  });

  return 0;
}
