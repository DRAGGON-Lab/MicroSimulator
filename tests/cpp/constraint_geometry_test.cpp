#include <cassert>
#include <cmath>
#include <stdexcept>

#include "cm/constraints.hpp"
#include "cm/simulation.hpp"

namespace {

bool close(float actual, float expected, float tolerance = 1.0e-5F) {
  return std::abs(actual - expected) <= tolerance;
}

cm::CellId add_capsule(cm::WorldState& state, cm::Vec3 center, cm::Vec3 axis, float length = 2.0F,
                       float radius = 0.5F) {
  cm::CellInit cell;
  cell.position = center;
  cell.direction = axis;
  cell.length = length;
  cell.radius = radius;
  return state.add_cell(cell);
}

void test_constraint_ids_and_validation() {
  cm::ConstraintSet constraints;
  cm::PlaneConstraintInit plane;
  plane.inward_normal = {0.0F, 2.0F, 0.0F};
  const auto plane_id = constraints.add_plane(plane);
  cm::SphereConstraintInit sphere;
  const auto sphere_id = constraints.add_sphere(sphere);
  assert(plane_id == 1);
  assert(sphere_id == 2);
  assert(constraints.size() == 2);
  assert(close(constraints.planes()[0].inward_normal.y, 1.0F));

  plane.coefficient = 0.0F;
  bool rejected = false;
  try {
    static_cast<void>(constraints.add_plane(plane));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);

  plane.coefficient = 1.0F;
  plane.inward_normal = {};
  rejected = false;
  try {
    static_cast<void>(constraints.add_plane(plane));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
}

void test_external_contact_graph_rejects_invalid_location_tag() {
  cm::ExternalContact contact;
  contact.cell_id = 1;
  contact.cell_slot = 0;
  contact.constraint_id = 1;
  contact.location = static_cast<cm::RodContactLocation>(255);
  contact.normal = {1.0F, 0.0F, 0.0F};

  bool rejected = false;
  try {
    static_cast<void>(cm::ExternalContactGraph(1, {contact}));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
}

void test_parallel_plane_contact_uses_two_weighted_endpoints() {
  cm::WorldState state;
  add_capsule(state, {0.0F, 0.4F, 0.0F}, {1.0F, 0.0F, 0.0F});
  cm::ConstraintSet constraints;
  cm::PlaneConstraintInit plane;
  plane.inward_normal = {0.0F, 1.0F, 0.0F};
  plane.coefficient = 2.0F;
  const auto plane_id = constraints.add_plane(plane);

  const auto graph = cm::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 2);
  assert(graph.incident_contact_indices(0).size() == 2);
  for (const auto& contact : graph.contacts()) {
    assert(contact.constraint_id == plane_id);
    assert(contact.constraint_kind == cm::ExternalConstraintKind::plane);
    assert(close(contact.signed_separation, -0.1F));
    assert(close(contact.normal.y, -1.0F));
    assert(close(contact.point_on_cell.y, -0.1F));
    assert(close(contact.weight, std::sqrt(2.0F)));
  }
}

void test_perpendicular_plane_emits_one_endpoint() {
  cm::WorldState state;
  add_capsule(state, {0.0F, 1.0F, 0.0F}, {0.0F, 1.0F, 0.0F});
  cm::ConstraintSet constraints;
  cm::PlaneConstraintInit plane;
  plane.inward_normal = {0.0F, 1.0F, 0.0F};
  constraints.add_plane(plane);

  const auto graph = cm::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 1);
  assert(graph.contacts()[0].location == cm::RodContactLocation::negative);
  assert(graph.contacts()[0].location == cm::RodEndpoint::negative);
  assert(close(graph.contacts()[0].signed_separation, -0.5F));
  assert(close(graph.contacts()[0].weight, 1.0F));
}

void test_outside_and_inside_spheres_have_typed_orientation() {
  cm::WorldState outside_state;
  add_capsule(outside_state, {1.2F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}, 0.0F);
  cm::ConstraintSet outside_constraints;
  cm::SphereConstraintInit outside;
  outside.radius = 1.0F;
  const auto outside_id = outside_constraints.add_sphere(outside);
  const auto outside_graph = cm::find_external_contacts_cpu(outside_state, outside_constraints);
  assert(outside_graph.size() == 2);
  for (const auto& contact : outside_graph.contacts()) {
    assert(contact.constraint_id == outside_id);
    assert(contact.constraint_kind == cm::ExternalConstraintKind::sphere);
    assert(close(contact.signed_separation, -0.3F));
    assert(close(contact.normal.x, -1.0F));
    assert(close(contact.point_on_cell.x, 0.7F));
    assert(close(contact.weight, std::sqrt(0.5F)));
  }

  cm::WorldState inside_state;
  add_capsule(inside_state, {4.8F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}, 0.0F);
  cm::ConstraintSet inside_constraints;
  cm::SphereConstraintInit inside;
  inside.radius = 5.0F;
  inside.allowed_region = cm::SphereRegion::inside;
  inside_constraints.add_sphere(inside);
  const auto inside_graph = cm::find_external_contacts_cpu(inside_state, inside_constraints);
  assert(inside_graph.size() == 2);
  for (const auto& contact : inside_graph.contacts()) {
    assert(close(contact.signed_separation, -0.3F));
    assert(close(contact.normal.x, 1.0F));
    assert(close(contact.point_on_cell.x, 5.3F));
  }
}

void test_degenerate_sphere_normal_is_finite_and_deterministic() {
  cm::WorldState state;
  add_capsule(state, {}, {0.0F, 1.0F, 0.0F}, 0.0F);
  cm::ConstraintSet constraints;
  cm::SphereConstraintInit sphere;
  sphere.radius = 2.0F;
  constraints.add_sphere(sphere);
  const auto graph = cm::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 2);
  for (const auto& contact : graph.contacts()) {
    assert(close(cm::norm(contact.normal), 1.0F));
    assert(close(contact.normal.x, -1.0F));
  }
}

void test_outside_sphere_detects_midspan_capsule_contact() {
  cm::WorldState state;
  add_capsule(state, {0.0F, 0.75F, 0.0F}, {1.0F, 0.0F, 0.0F}, 3.5F);
  cm::ConstraintSet constraints;
  cm::SphereConstraintInit sphere;
  const auto sphere_id = constraints.add_sphere(sphere);

  const auto graph = cm::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 1);
  const auto& contact = graph.contacts()[0];
  assert(contact.constraint_id == sphere_id);
  assert(contact.constraint_kind == cm::ExternalConstraintKind::sphere);
  assert(contact.location == cm::RodContactLocation::interior);
  assert(close(contact.point_on_cell.x, 0.0F));
  assert(close(contact.point_on_cell.y, 0.25F));
  assert(close(contact.normal.x, 0.0F));
  assert(close(contact.normal.y, -1.0F));
  assert(close(contact.signed_separation, -0.75F));
  assert(close(contact.weight, 1.0F));
}

void test_box_ids_validation_and_checkpoint() {
  cm::ConstraintSet constraints;
  cm::BoxConstraintInit box;
  box.half_extents = {2.0F, 1.0F, 3.0F};
  const auto box_id = constraints.add_box(box);
  assert(box_id == 1);
  assert(constraints.size() == 1);
  assert(close(constraints.boxes()[0].half_extents.y, 1.0F));

  box.half_extents = {1.0F, 0.0F, 1.0F};
  bool rejected = false;
  try {
    static_cast<void>(constraints.add_box(box));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);

  box.half_extents = {1.0F, 1.0F, 1.0F};
  box.coefficient = -1.0F;
  rejected = false;
  try {
    static_cast<void>(constraints.add_box(box));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);

  const auto checkpoint = constraints.checkpoint();
  assert(checkpoint.boxes.size() == 1);
  const cm::ConstraintSet restored(checkpoint);
  assert(restored.boxes().size() == 1);
  assert(close(restored.boxes()[0].half_extents.z, 3.0F));
}

void test_outside_box_face_contact_uses_two_weighted_endpoints() {
  cm::WorldState state;
  add_capsule(state, {1.4F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F});
  cm::ConstraintSet constraints;
  cm::BoxConstraintInit box;
  box.coefficient = 2.0F;
  const auto box_id = constraints.add_box(box);

  const auto graph = cm::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 2);
  for (const auto& contact : graph.contacts()) {
    assert(contact.constraint_id == box_id);
    assert(contact.constraint_kind == cm::ExternalConstraintKind::box);
    assert(close(contact.signed_separation, -0.1F));
    assert(close(contact.normal.x, -1.0F));
    assert(close(contact.point_on_cell.x, 0.9F));
    assert(close(contact.weight, std::sqrt(2.0F)));
  }
}

void test_outside_box_corner_contact_has_diagonal_normal() {
  cm::WorldState state;
  add_capsule(state, {1.3F, 1.3F, 0.0F}, {0.0F, 0.0F, 1.0F}, 0.0F);
  cm::ConstraintSet constraints;
  cm::BoxConstraintInit box;
  constraints.add_box(box);

  const auto graph = cm::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 2);
  const auto diagonal = 1.0F / std::sqrt(2.0F);
  for (const auto& contact : graph.contacts()) {
    assert(close(contact.signed_separation, 0.3F * std::sqrt(2.0F) - 0.5F));
    assert(close(contact.normal.x, -diagonal));
    assert(close(contact.normal.y, -diagonal));
    assert(close(cm::norm(contact.normal), 1.0F));
  }
}

void test_box_interior_endpoint_escapes_toward_nearest_face() {
  cm::WorldState state;
  add_capsule(state, {0.5F, 0.6F, 0.0F}, {1.0F, 0.0F, 0.0F}, 0.0F);
  cm::ConstraintSet constraints;
  cm::BoxConstraintInit box;
  box.half_extents = {2.0F, 1.0F, 3.0F};
  constraints.add_box(box);

  const auto graph = cm::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 2);
  for (const auto& contact : graph.contacts()) {
    assert(close(contact.signed_separation, -0.9F));
    assert(close(contact.normal.y, -1.0F));
    assert(close(contact.point_on_cell.y, 0.1F));
  }
}

void test_inside_box_confines_like_a_chamber() {
  cm::WorldState state;
  add_capsule(state, {4.8F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}, 0.0F);
  cm::ConstraintSet constraints;
  cm::BoxConstraintInit box;
  box.half_extents = {5.0F, 5.0F, 5.0F};
  box.allowed_region = cm::ConstraintRegion::inside;
  constraints.add_box(box);

  const auto graph = cm::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 2);
  for (const auto& contact : graph.contacts()) {
    assert(close(contact.signed_separation, -0.3F));
    assert(close(contact.normal.x, 1.0F));
    assert(close(contact.point_on_cell.x, 5.3F));
  }
}

void test_box_center_degeneracy_is_finite_and_deterministic() {
  cm::WorldState state;
  add_capsule(state, {}, {0.0F, 1.0F, 0.0F}, 0.0F);
  cm::ConstraintSet constraints;
  cm::BoxConstraintInit box;
  constraints.add_box(box);

  const auto graph = cm::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 2);
  for (const auto& contact : graph.contacts()) {
    assert(close(cm::norm(contact.normal), 1.0F));
    assert(close(contact.normal.x, -1.0F));
    assert(close(contact.signed_separation, -1.5F));
  }
}

void test_outside_box_detects_midspan_capsule_contact() {
  cm::WorldState state;
  add_capsule(state, {0.0F, 0.75F, 0.0F}, {1.0F, 0.0F, 0.0F}, 3.5F);
  cm::ConstraintSet constraints;
  cm::BoxConstraintInit box;
  box.half_extents = {1.0F, 1.0F, 5.0F};
  const auto box_id = constraints.add_box(box);

  const auto graph = cm::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 1);
  const auto& contact = graph.contacts()[0];
  assert(contact.constraint_id == box_id);
  assert(contact.constraint_kind == cm::ExternalConstraintKind::box);
  assert(contact.location == cm::RodContactLocation::interior);
  assert(close(contact.point_on_cell.x, 0.0F));
  assert(close(contact.point_on_cell.y, 0.25F));
  assert(close(contact.normal.x, 0.0F));
  assert(close(contact.normal.y, -1.0F));
  assert(close(contact.signed_separation, -0.75F));
  assert(close(contact.weight, 1.0F));
}

void test_outside_box_detects_centered_wall_crossing() {
  cm::WorldState state;
  add_capsule(state, {}, {1.0F, 0.0F, 0.0F}, 3.5F);
  cm::ConstraintSet constraints;
  cm::BoxConstraintInit wall;
  wall.half_extents = {1.0F, 5.0F, 5.0F};
  constraints.add_box(wall);

  const auto graph = cm::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 1);
  const auto& contact = graph.contacts()[0];
  assert(contact.location == cm::RodContactLocation::interior);
  assert(close(contact.point_on_cell.x, -0.5F));
  assert(close(contact.normal.x, -1.0F));
  assert(close(contact.signed_separation, -1.5F));
}

void test_cylinder_ids_validation_and_checkpoint() {
  cm::ConstraintSet constraints;
  cm::CylinderConstraintInit cylinder;
  cylinder.radius = 2.0F;
  cylinder.half_height = 3.0F;
  const auto cylinder_id = constraints.add_cylinder(cylinder);
  assert(cylinder_id == 1);
  assert(constraints.size() == 1);
  assert(close(constraints.cylinders()[0].half_height, 3.0F));

  cylinder.radius = 0.0F;
  bool rejected = false;
  try {
    static_cast<void>(constraints.add_cylinder(cylinder));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);

  cylinder.radius = 2.0F;
  cylinder.half_height = -1.0F;
  rejected = false;
  try {
    static_cast<void>(constraints.add_cylinder(cylinder));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);

  const auto checkpoint = constraints.checkpoint();
  assert(checkpoint.cylinders.size() == 1);
  const cm::ConstraintSet restored(checkpoint);
  assert(restored.cylinders().size() == 1);
  assert(close(restored.cylinders()[0].radius, 2.0F));
}

void test_outside_cylinder_barrel_contact_uses_two_weighted_endpoints() {
  cm::WorldState state;
  add_capsule(state, {1.4F, 0.0F, 0.0F}, {0.0F, 0.0F, 1.0F});
  cm::ConstraintSet constraints;
  cm::CylinderConstraintInit cylinder;
  cylinder.coefficient = 2.0F;
  const auto cylinder_id = constraints.add_cylinder(cylinder);

  const auto graph = cm::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 2);
  for (const auto& contact : graph.contacts()) {
    assert(contact.constraint_id == cylinder_id);
    assert(contact.constraint_kind == cm::ExternalConstraintKind::cylinder);
    assert(close(contact.signed_separation, -0.1F));
    assert(close(contact.normal.x, -1.0F));
    assert(close(contact.point_on_cell.x, 0.9F));
    assert(close(contact.weight, std::sqrt(2.0F)));
  }
}

void test_outside_cylinder_rim_contact_has_blended_normal() {
  cm::WorldState state;
  add_capsule(state, {1.3F, 0.0F, 1.3F}, {0.0F, 1.0F, 0.0F}, 0.0F);
  cm::ConstraintSet constraints;
  cm::CylinderConstraintInit cylinder;
  constraints.add_cylinder(cylinder);

  const auto graph = cm::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 2);
  const auto diagonal = 1.0F / std::sqrt(2.0F);
  for (const auto& contact : graph.contacts()) {
    assert(close(contact.signed_separation, 0.3F * std::sqrt(2.0F) - 0.5F));
    assert(close(contact.normal.x, -diagonal));
    assert(close(contact.normal.z, -diagonal));
    assert(close(cm::norm(contact.normal), 1.0F));
  }
}

void test_outside_cylinder_cap_contact_points_axially() {
  cm::WorldState state;
  add_capsule(state, {0.5F, 0.0F, 1.4F}, {1.0F, 0.0F, 0.0F}, 0.0F);
  cm::ConstraintSet constraints;
  cm::CylinderConstraintInit cylinder;
  constraints.add_cylinder(cylinder);

  const auto graph = cm::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 2);
  for (const auto& contact : graph.contacts()) {
    assert(close(contact.signed_separation, -0.1F));
    assert(close(contact.normal.z, -1.0F));
    assert(close(contact.point_on_cell.z, 0.9F));
  }
}

void test_inside_cylinder_confines_like_a_dish() {
  cm::WorldState state;
  add_capsule(state, {4.8F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}, 0.0F);
  cm::ConstraintSet constraints;
  cm::CylinderConstraintInit cylinder;
  cylinder.radius = 5.0F;
  cylinder.half_height = 5.0F;
  cylinder.allowed_region = cm::ConstraintRegion::inside;
  constraints.add_cylinder(cylinder);

  const auto graph = cm::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 2);
  for (const auto& contact : graph.contacts()) {
    assert(close(contact.signed_separation, -0.3F));
    assert(close(contact.normal.x, 1.0F));
    assert(close(contact.point_on_cell.x, 5.3F));
  }
}

void test_cylinder_axis_degeneracy_is_finite_and_radial() {
  cm::WorldState state;
  add_capsule(state, {}, {0.0F, 1.0F, 0.0F}, 0.0F);
  cm::ConstraintSet constraints;
  cm::CylinderConstraintInit cylinder;
  constraints.add_cylinder(cylinder);

  const auto graph = cm::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 2);
  for (const auto& contact : graph.contacts()) {
    assert(close(cm::norm(contact.normal), 1.0F));
    assert(close(contact.normal.x, -1.0F));
    assert(close(contact.signed_separation, -1.5F));
  }
}

void test_outside_cylinder_detects_midspan_capsule_contact() {
  cm::WorldState state;
  add_capsule(state, {0.0F, 0.75F, 0.0F}, {1.0F, 0.0F, 0.0F}, 3.5F);
  cm::ConstraintSet constraints;
  cm::CylinderConstraintInit cylinder;
  cylinder.center = {0.4F, 0.0F, 0.0F};
  cylinder.half_height = 5.0F;
  const auto cylinder_id = constraints.add_cylinder(cylinder);

  const auto graph = cm::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 1);
  const auto& contact = graph.contacts()[0];
  assert(contact.constraint_id == cylinder_id);
  assert(contact.constraint_kind == cm::ExternalConstraintKind::cylinder);
  assert(contact.location == cm::RodContactLocation::interior);
  assert(close(contact.point_on_cell.x, 0.4F));
  assert(close(contact.point_on_cell.y, 0.25F));
  assert(close(contact.normal.x, 0.0F));
  assert(close(contact.normal.y, -1.0F));
  assert(close(contact.signed_separation, -0.75F));
  assert(close(contact.weight, 1.0F));
}

void test_simulation_exposes_cpu_constraint_graph() {
  cm::Simulation simulation;
  cm::CellInit cell;
  cell.position.y = 0.4F;
  cell.length = 2.0F;
  simulation.add_cell(cell);
  cm::PlaneConstraintInit plane;
  plane.inward_normal = {0.0F, 1.0F, 0.0F};
  simulation.add_plane_constraint(plane);
  assert(simulation.supports(cm::BackendFeature::external_constraints));
  const auto graph = simulation.find_external_contacts();
  assert(graph.size() == 2);
}

}  // namespace

int main() {
  test_constraint_ids_and_validation();
  test_external_contact_graph_rejects_invalid_location_tag();
  test_parallel_plane_contact_uses_two_weighted_endpoints();
  test_perpendicular_plane_emits_one_endpoint();
  test_outside_and_inside_spheres_have_typed_orientation();
  test_degenerate_sphere_normal_is_finite_and_deterministic();
  test_outside_sphere_detects_midspan_capsule_contact();
  test_box_ids_validation_and_checkpoint();
  test_outside_box_face_contact_uses_two_weighted_endpoints();
  test_outside_box_corner_contact_has_diagonal_normal();
  test_box_interior_endpoint_escapes_toward_nearest_face();
  test_inside_box_confines_like_a_chamber();
  test_box_center_degeneracy_is_finite_and_deterministic();
  test_outside_box_detects_midspan_capsule_contact();
  test_outside_box_detects_centered_wall_crossing();
  test_cylinder_ids_validation_and_checkpoint();
  test_outside_cylinder_barrel_contact_uses_two_weighted_endpoints();
  test_outside_cylinder_rim_contact_has_blended_normal();
  test_outside_cylinder_cap_contact_points_axially();
  test_inside_cylinder_confines_like_a_dish();
  test_cylinder_axis_degeneracy_is_finite_and_radial();
  test_outside_cylinder_detects_midspan_capsule_contact();
  test_simulation_exposes_cpu_constraint_graph();
  return 0;
}
