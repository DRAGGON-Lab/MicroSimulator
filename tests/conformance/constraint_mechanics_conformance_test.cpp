#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "backend_devices.hpp"
#include "cm/simulation.hpp"

namespace {

constexpr float absolute_tolerance = 3.0e-4F;
constexpr float relative_tolerance = 3.0e-4F;

bool close(float actual, float expected) {
  const auto tolerance = absolute_tolerance + relative_tolerance * std::abs(expected);
  return std::abs(actual - expected) <= tolerance;
}

using Fixture = void (*)(cm::Simulation&);

void populate_plane(cm::Simulation& simulation) {
  cm::CellInit cell;
  cell.position = {0.0F, 0.4F, 0.0F};
  cell.direction = {1.0F, 0.0F, 0.0F};
  cell.length = 2.0F;
  cell.radius = 0.5F;
  simulation.add_cell(cell);

  cm::PlaneConstraintInit plane;
  plane.inward_normal = {0.0F, 1.0F, 0.0F};
  plane.coefficient = 1.25F;
  simulation.add_plane_constraint(plane);
}

void populate_fixed_plane(cm::Simulation& simulation) {
  cm::CellInit cell;
  cell.position = {0.0F, 0.4F, 0.0F};
  cell.direction = {1.0F, 0.0F, 0.0F};
  cell.length = 2.0F;
  cell.radius = 0.5F;
  cell.fixed = true;
  simulation.add_cell(cell);

  cm::PlaneConstraintInit plane;
  plane.inward_normal = {0.0F, 1.0F, 0.0F};
  plane.coefficient = 1.25F;
  simulation.add_plane_constraint(plane);
}

void populate_outside_sphere(cm::Simulation& simulation) {
  cm::CellInit cell;
  cell.position = {1.2F, 0.0F, 0.0F};
  cell.length = 0.0F;
  cell.radius = 0.5F;
  simulation.add_cell(cell);

  cm::SphereConstraintInit sphere;
  sphere.radius = 1.0F;
  sphere.coefficient = 0.75F;
  simulation.add_sphere_constraint(sphere);
}

void populate_inside_sphere(cm::Simulation& simulation) {
  cm::CellInit cell;
  cell.position = {4.8F, 0.0F, 0.0F};
  cell.length = 0.0F;
  cell.radius = 0.5F;
  simulation.add_cell(cell);

  cm::SphereConstraintInit sphere;
  sphere.radius = 5.0F;
  sphere.allowed_region = cm::SphereRegion::inside;
  simulation.add_sphere_constraint(sphere);
}

void populate_outside_box(cm::Simulation& simulation) {
  cm::CellInit cell;
  cell.position = {1.4F, 0.0F, 0.0F};
  cell.direction = {0.0F, 1.0F, 0.0F};
  cell.length = 2.0F;
  cell.radius = 0.5F;
  simulation.add_cell(cell);

  cm::BoxConstraintInit box;
  box.coefficient = 1.25F;
  simulation.add_box_constraint(box);
}

void populate_inside_box(cm::Simulation& simulation) {
  cm::CellInit cell;
  cell.position = {4.8F, 0.0F, 0.0F};
  cell.length = 0.0F;
  cell.radius = 0.5F;
  simulation.add_cell(cell);

  cm::BoxConstraintInit box;
  box.half_extents = {5.0F, 5.0F, 5.0F};
  box.allowed_region = cm::ConstraintRegion::inside;
  simulation.add_box_constraint(box);
}

void compare_results(const cm::MechanicsSolveResult& actual,
                     const cm::MechanicsSolveResult& expected) {
  assert(actual.report.status == expected.report.status);
  assert(actual.report.breakdown == expected.report.breakdown);
  assert(close(actual.report.initial_residual_rms, expected.report.initial_residual_rms));
  assert(actual.corrections.size() == expected.corrections.size());
  for (std::size_t index = 0; index < expected.corrections.size(); ++index) {
    const auto& left = actual.corrections[index];
    const auto& right = expected.corrections[index];
    assert(close(left.translation.x, right.translation.x));
    assert(close(left.translation.y, right.translation.y));
    assert(close(left.translation.z, right.translation.z));
    assert(close(left.rotation.x, right.rotation.x));
    assert(close(left.rotation.y, right.rotation.y));
    assert(close(left.rotation.z, right.rotation.z));
    assert(close(left.length, right.length));
  }
}

void compare_cells(const cm::Simulation& actual, const cm::Simulation& expected) {
  const auto actual_cells = actual.cells();
  const auto expected_cells = expected.cells();
  assert(actual_cells.size() == expected_cells.size());
  for (std::size_t index = 0; index < expected_cells.size(); ++index) {
    const auto& left = actual_cells[index];
    const auto& right = expected_cells[index];
    assert(left.id == right.id);
    assert(close(left.position.x, right.position.x));
    assert(close(left.position.y, right.position.y));
    assert(close(left.position.z, right.position.z));
    assert(close(left.direction.x, right.direction.x));
    assert(close(left.direction.y, right.direction.y));
    assert(close(left.direction.z, right.direction.z));
    assert(close(left.length, right.length));
  }
}

void run_fixture(cm::BackendKind backend, std::uint32_t device_index, Fixture fixture) {
  cm::Simulation reference(cm::BackendKind::cpu);
  cm::Simulation candidate(backend, 0, 0, device_index);
  fixture(reference);
  fixture(candidate);

  cm::MechanicsParameters parameters;
  parameters.residual_rms_tolerance = 2.0e-5F;
  const auto expected = reference.relax_cell_mechanics(parameters);
  const auto actual = candidate.relax_cell_mechanics(parameters);
  compare_results(actual, expected);
  compare_cells(candidate, reference);
}

void reject_unsupported_backend(cm::BackendKind backend, std::uint32_t device_index) {
  cm::Simulation simulation(backend, 0, 0, device_index);
  populate_plane(simulation);
  bool rejected = false;
  try {
    static_cast<void>(simulation.solve_cell_mechanics());
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);
}

}  // namespace

int main() {
  cm::test::for_each_backend_device([](cm::BackendKind backend, std::uint32_t device_index) {
    cm::Simulation capability_probe(backend, 0, 0, device_index);
    if (!capability_probe.supports(cm::BackendFeature::external_constraints)) {
      reject_unsupported_backend(backend, device_index);
      return;
    }
    run_fixture(backend, device_index, populate_plane);
    run_fixture(backend, device_index, populate_fixed_plane);
    run_fixture(backend, device_index, populate_outside_sphere);
    run_fixture(backend, device_index, populate_inside_sphere);
    run_fixture(backend, device_index, populate_outside_box);
    run_fixture(backend, device_index, populate_inside_box);
  });
  return 0;
}
