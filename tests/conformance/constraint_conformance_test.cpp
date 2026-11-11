#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "backend_devices.hpp"
#include "microsimulator/simulation.hpp"

namespace {

constexpr float absolute_tolerance = 2.0e-5F;
constexpr float relative_tolerance = 2.0e-5F;

bool close(float actual, float expected) {
  const auto tolerance = absolute_tolerance + relative_tolerance * std::abs(expected);
  return std::abs(actual - expected) <= tolerance;
}

void add_cell(microsimulator::Simulation& simulation, microsimulator::Vec3 position,
              microsimulator::Vec3 direction, float length, float radius = 0.5F) {
  microsimulator::CellInit cell;
  cell.position = position;
  cell.direction = direction;
  cell.length = length;
  cell.radius = radius;
  simulation.add_cell(cell);
}

void populate_mixed_constraints(microsimulator::Simulation& simulation) {
  add_cell(simulation, {0.0F, 0.4F, 0.0F}, {1.0F, 0.0F, 0.0F}, 2.0F);
  add_cell(simulation, {1.2F, 0.6F, 0.0F}, {0.0F, 1.0F, 0.0F}, 0.0F);
  add_cell(simulation, {4.8F, 2.0F, 0.0F}, {0.0F, 0.0F, 1.0F}, 0.0F);

  microsimulator::PlaneConstraintInit plane;
  plane.inward_normal = {0.0F, 2.0F, 0.0F};
  plane.coefficient = 1.25F;
  assert(simulation.add_plane_constraint(plane) == 1);

  microsimulator::SphereConstraintInit outside;
  outside.radius = 1.0F;
  outside.coefficient = 0.75F;
  assert(simulation.add_sphere_constraint(outside) == 2);

  microsimulator::SphereConstraintInit inside;
  inside.radius = 5.0F;
  inside.coefficient = 1.5F;
  inside.allowed_region = microsimulator::SphereRegion::inside;
  assert(simulation.add_sphere_constraint(inside) == 3);

  microsimulator::PlaneConstraintInit second_plane;
  second_plane.point = {6.0F, 0.0F, 0.0F};
  second_plane.inward_normal = {-1.0F, 0.0F, 0.0F};
  assert(simulation.add_plane_constraint(second_plane) == 4);

  microsimulator::BoxConstraintInit wall;
  wall.center = {0.0F, 1.3F, 0.0F};
  wall.half_extents = {3.0F, 0.5F, 1.0F};
  wall.coefficient = 0.9F;
  assert(simulation.add_box_constraint(wall) == 5);
}

void populate_box_regions(microsimulator::Simulation& simulation) {
  add_cell(simulation, {1.4F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}, 2.0F);
  add_cell(simulation, {1.3F, 1.3F, 0.0F}, {0.0F, 0.0F, 1.0F}, 0.0F);
  add_cell(simulation, {0.5F, 0.6F, 0.0F}, {1.0F, 0.0F, 0.0F}, 0.0F);
  add_cell(simulation, {4.8F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}, 0.0F);

  microsimulator::BoxConstraintInit outside;
  outside.half_extents = {2.0F, 1.0F, 3.0F};
  outside.coefficient = 1.25F;
  assert(simulation.add_box_constraint(outside) == 1);

  microsimulator::BoxConstraintInit chamber;
  chamber.half_extents = {5.0F, 5.0F, 5.0F};
  chamber.coefficient = 0.5F;
  chamber.allowed_region = microsimulator::ConstraintRegion::inside;
  assert(simulation.add_box_constraint(chamber) == 2);
}

void populate_cylinder_regions(microsimulator::Simulation& simulation) {
  add_cell(simulation, {1.4F, 0.0F, 0.0F}, {0.0F, 0.0F, 1.0F}, 2.0F);
  add_cell(simulation, {1.3F, 0.0F, 1.3F}, {0.0F, 1.0F, 0.0F}, 0.0F);
  add_cell(simulation, {0.5F, 0.0F, 1.4F}, {1.0F, 0.0F, 0.0F}, 0.0F);
  add_cell(simulation, {4.8F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}, 0.0F);

  microsimulator::CylinderConstraintInit pillar;
  pillar.coefficient = 1.25F;
  assert(simulation.add_cylinder_constraint(pillar) == 1);

  microsimulator::CylinderConstraintInit dish;
  dish.radius = 5.0F;
  dish.half_height = 5.0F;
  dish.coefficient = 0.5F;
  dish.allowed_region = microsimulator::ConstraintRegion::inside;
  assert(simulation.add_cylinder_constraint(dish) == 2);
}

void populate_degenerate_sphere(microsimulator::Simulation& simulation) {
  add_cell(simulation, {}, {0.0F, 1.0F, 0.0F}, 0.0F);
  microsimulator::SphereConstraintInit sphere;
  sphere.radius = 2.0F;
  simulation.add_sphere_constraint(sphere);
}

void populate_degenerate_box(microsimulator::Simulation& simulation) {
  add_cell(simulation, {}, {0.0F, 1.0F, 0.0F}, 0.0F);
  microsimulator::BoxConstraintInit box;
  simulation.add_box_constraint(box);
}

void populate_midspan_sphere(microsimulator::Simulation& simulation) {
  add_cell(simulation, {0.0F, 0.75F, 0.0F}, {1.0F, 0.0F, 0.0F}, 3.5F);
  microsimulator::SphereConstraintInit sphere;
  simulation.add_sphere_constraint(sphere);
}

void populate_midspan_box(microsimulator::Simulation& simulation) {
  add_cell(simulation, {0.0F, 0.75F, 0.0F}, {1.0F, 0.0F, 0.0F}, 3.5F);
  microsimulator::BoxConstraintInit box;
  box.half_extents = {1.0F, 1.0F, 5.0F};
  simulation.add_box_constraint(box);
}

void populate_centered_wall_crossing(microsimulator::Simulation& simulation) {
  add_cell(simulation, {}, {1.0F, 0.0F, 0.0F}, 3.5F);
  microsimulator::BoxConstraintInit wall;
  wall.half_extents = {1.0F, 5.0F, 5.0F};
  simulation.add_box_constraint(wall);
}

void populate_midspan_cylinder(microsimulator::Simulation& simulation) {
  add_cell(simulation, {0.0F, 0.75F, 0.0F}, {1.0F, 0.0F, 0.0F}, 3.5F);
  microsimulator::CylinderConstraintInit cylinder;
  cylinder.center = {0.4F, 0.0F, 0.0F};
  cylinder.half_height = 5.0F;
  simulation.add_cylinder_constraint(cylinder);
}

void compare_graphs(const microsimulator::ExternalContactGraph& actual,
                    const microsimulator::ExternalContactGraph& expected) {
  assert(actual.cell_count() == expected.cell_count());
  assert(actual.size() == expected.size());
  const auto actual_contacts = actual.contacts();
  const auto expected_contacts = expected.contacts();
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const auto& left = actual_contacts[index];
    const auto& right = expected_contacts[index];
    assert(left.cell_id == right.cell_id);
    assert(left.cell_slot == right.cell_slot);
    assert(left.constraint_id == right.constraint_id);
    assert(left.constraint_kind == right.constraint_kind);
    assert(left.location == right.location);
    assert(close(left.point_on_cell.x, right.point_on_cell.x));
    assert(close(left.point_on_cell.y, right.point_on_cell.y));
    assert(close(left.point_on_cell.z, right.point_on_cell.z));
    assert(close(left.normal.x, right.normal.x));
    assert(close(left.normal.y, right.normal.y));
    assert(close(left.normal.z, right.normal.z));
    assert(close(left.signed_separation, right.signed_separation));
    assert(close(left.weight, right.weight));
  }
  for (std::size_t slot = 0; slot < expected.cell_count(); ++slot) {
    const auto actual_incidence =
        actual.incident_contact_indices(static_cast<microsimulator::Slot>(slot));
    const auto expected_incidence =
        expected.incident_contact_indices(static_cast<microsimulator::Slot>(slot));
    assert(actual_incidence.size() == expected_incidence.size());
    for (std::size_t index = 0; index < expected_incidence.size(); ++index) {
      assert(actual_incidence[index] == expected_incidence[index]);
    }
  }
}

void run_fixture(microsimulator::BackendKind backend, std::uint32_t device_index,
                 void (*populate)(microsimulator::Simulation&)) {
  microsimulator::Simulation reference(microsimulator::BackendKind::cpu);
  microsimulator::Simulation candidate(backend, 0, 0, device_index);
  populate(reference);
  populate(candidate);
  compare_graphs(candidate.find_external_contacts(), reference.find_external_contacts());
}

void run_empty_inputs(microsimulator::BackendKind backend, std::uint32_t device_index) {
  microsimulator::Simulation empty(backend, 0, 0, device_index);
  const auto no_cells = empty.find_external_contacts();
  assert(no_cells.empty());
  assert(no_cells.cell_count() == 0);

  microsimulator::Simulation no_constraints(backend, 0, 0, device_index);
  add_cell(no_constraints, {}, {1.0F, 0.0F, 0.0F}, 1.0F);
  const auto graph = no_constraints.find_external_contacts();
  assert(graph.empty());
  assert(graph.cell_count() == 1);
}

}  // namespace

int main() {
  microsimulator::test::for_each_backend_device(
      [](microsimulator::BackendKind backend, std::uint32_t device_index) {
        microsimulator::Simulation capability_probe(backend, 0, 0, device_index);
        if (!capability_probe.supports(microsimulator::BackendFeature::external_constraints)) {
          return;
        }
        run_empty_inputs(backend, device_index);
        run_fixture(backend, device_index, populate_mixed_constraints);
        run_fixture(backend, device_index, populate_box_regions);
        run_fixture(backend, device_index, populate_cylinder_regions);
        run_fixture(backend, device_index, populate_degenerate_sphere);
        run_fixture(backend, device_index, populate_degenerate_box);
        run_fixture(backend, device_index, populate_midspan_sphere);
        run_fixture(backend, device_index, populate_midspan_box);
        run_fixture(backend, device_index, populate_centered_wall_crossing);
        run_fixture(backend, device_index, populate_midspan_cylinder);
      });
  return 0;
}
