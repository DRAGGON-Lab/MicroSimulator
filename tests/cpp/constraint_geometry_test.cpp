#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

#include "microsimulator/constraints.hpp"
#include "microsimulator/simulation.hpp"

namespace {

bool close(float actual, float expected, float tolerance = 1.0e-5F) {
  return std::abs(actual - expected) <= tolerance;
}

microsimulator::CellId add_capsule(microsimulator::WorldState& state, microsimulator::Vec3 center,
                                   microsimulator::Vec3 axis, float length = 2.0F,
                                   float radius = 0.5F) {
  microsimulator::CellInit cell;
  cell.position = center;
  cell.direction = axis;
  cell.length = length;
  cell.radius = radius;
  return state.add_cell(cell);
}

void test_constraint_ids_and_validation() {
  microsimulator::ConstraintSet constraints;
  microsimulator::PlaneConstraintInit plane;
  plane.inward_normal = {0.0F, 2.0F, 0.0F};
  const auto plane_id = constraints.add_plane(plane);
  microsimulator::SphereConstraintInit sphere;
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
  microsimulator::ExternalContact contact;
  contact.cell_id = 1;
  contact.cell_slot = 0;
  contact.constraint_id = 1;
  contact.location = static_cast<microsimulator::RodContactLocation>(255);
  contact.normal = {1.0F, 0.0F, 0.0F};

  bool rejected = false;
  try {
    static_cast<void>(microsimulator::ExternalContactGraph(1, {contact}));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
}

void test_parallel_plane_contact_uses_two_weighted_endpoints() {
  microsimulator::WorldState state;
  add_capsule(state, {0.0F, 0.4F, 0.0F}, {1.0F, 0.0F, 0.0F});
  microsimulator::ConstraintSet constraints;
  microsimulator::PlaneConstraintInit plane;
  plane.inward_normal = {0.0F, 1.0F, 0.0F};
  plane.coefficient = 2.0F;
  const auto plane_id = constraints.add_plane(plane);

  const auto graph = microsimulator::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 2);
  assert(graph.incident_contact_indices(0).size() == 2);
  for (const auto& contact : graph.contacts()) {
    assert(contact.constraint_id == plane_id);
    assert(contact.constraint_kind == microsimulator::ExternalConstraintKind::plane);
    assert(close(contact.signed_separation, -0.1F));
    assert(close(contact.normal.y, -1.0F));
    assert(close(contact.point_on_cell.y, -0.1F));
    assert(close(contact.weight, std::sqrt(2.0F)));
  }
}

void test_perpendicular_plane_emits_one_endpoint() {
  microsimulator::WorldState state;
  add_capsule(state, {0.0F, 1.0F, 0.0F}, {0.0F, 1.0F, 0.0F});
  microsimulator::ConstraintSet constraints;
  microsimulator::PlaneConstraintInit plane;
  plane.inward_normal = {0.0F, 1.0F, 0.0F};
  constraints.add_plane(plane);

  const auto graph = microsimulator::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 1);
  assert(graph.contacts()[0].location == microsimulator::RodContactLocation::negative);
  assert(graph.contacts()[0].location == microsimulator::RodEndpoint::negative);
  assert(close(graph.contacts()[0].signed_separation, -0.5F));
  assert(close(graph.contacts()[0].weight, 1.0F));
}

void test_outside_and_inside_spheres_have_typed_orientation() {
  microsimulator::WorldState outside_state;
  add_capsule(outside_state, {1.2F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}, 0.0F);
  microsimulator::ConstraintSet outside_constraints;
  microsimulator::SphereConstraintInit outside;
  outside.radius = 1.0F;
  const auto outside_id = outside_constraints.add_sphere(outside);
  const auto outside_graph =
      microsimulator::find_external_contacts_cpu(outside_state, outside_constraints);
  assert(outside_graph.size() == 2);
  for (const auto& contact : outside_graph.contacts()) {
    assert(contact.constraint_id == outside_id);
    assert(contact.constraint_kind == microsimulator::ExternalConstraintKind::sphere);
    assert(close(contact.signed_separation, -0.3F));
    assert(close(contact.normal.x, -1.0F));
    assert(close(contact.point_on_cell.x, 0.7F));
    assert(close(contact.weight, std::sqrt(0.5F)));
  }

  microsimulator::WorldState inside_state;
  add_capsule(inside_state, {4.8F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}, 0.0F);
  microsimulator::ConstraintSet inside_constraints;
  microsimulator::SphereConstraintInit inside;
  inside.radius = 5.0F;
  inside.allowed_region = microsimulator::SphereRegion::inside;
  inside_constraints.add_sphere(inside);
  const auto inside_graph =
      microsimulator::find_external_contacts_cpu(inside_state, inside_constraints);
  assert(inside_graph.size() == 2);
  for (const auto& contact : inside_graph.contacts()) {
    assert(close(contact.signed_separation, -0.3F));
    assert(close(contact.normal.x, 1.0F));
    assert(close(contact.point_on_cell.x, 5.3F));
  }
}

void test_degenerate_sphere_normal_is_finite_and_deterministic() {
  microsimulator::WorldState state;
  add_capsule(state, {}, {0.0F, 1.0F, 0.0F}, 0.0F);
  microsimulator::ConstraintSet constraints;
  microsimulator::SphereConstraintInit sphere;
  sphere.radius = 2.0F;
  constraints.add_sphere(sphere);
  const auto graph = microsimulator::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 2);
  for (const auto& contact : graph.contacts()) {
    assert(close(microsimulator::norm(contact.normal), 1.0F));
    assert(close(contact.normal.x, -1.0F));
  }
}

void test_outside_sphere_detects_midspan_capsule_contact() {
  microsimulator::WorldState state;
  add_capsule(state, {0.0F, 0.75F, 0.0F}, {1.0F, 0.0F, 0.0F}, 3.5F);
  microsimulator::ConstraintSet constraints;
  microsimulator::SphereConstraintInit sphere;
  const auto sphere_id = constraints.add_sphere(sphere);

  const auto graph = microsimulator::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 1);
  const auto& contact = graph.contacts()[0];
  assert(contact.constraint_id == sphere_id);
  assert(contact.constraint_kind == microsimulator::ExternalConstraintKind::sphere);
  assert(contact.location == microsimulator::RodContactLocation::interior);
  assert(close(contact.point_on_cell.x, 0.0F));
  assert(close(contact.point_on_cell.y, 0.25F));
  assert(close(contact.normal.x, 0.0F));
  assert(close(contact.normal.y, -1.0F));
  assert(close(contact.signed_separation, -0.75F));
  assert(close(contact.weight, 1.0F));
}

void test_box_ids_validation_and_checkpoint() {
  microsimulator::ConstraintSet constraints;
  microsimulator::BoxConstraintInit box;
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
  const microsimulator::ConstraintSet restored(checkpoint);
  assert(restored.boxes().size() == 1);
  assert(close(restored.boxes()[0].half_extents.z, 3.0F));
}

void test_outside_box_face_contact_uses_two_weighted_endpoints() {
  microsimulator::WorldState state;
  add_capsule(state, {1.4F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F});
  microsimulator::ConstraintSet constraints;
  microsimulator::BoxConstraintInit box;
  box.coefficient = 2.0F;
  const auto box_id = constraints.add_box(box);

  const auto graph = microsimulator::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 2);
  for (const auto& contact : graph.contacts()) {
    assert(contact.constraint_id == box_id);
    assert(contact.constraint_kind == microsimulator::ExternalConstraintKind::box);
    assert(close(contact.signed_separation, -0.1F));
    assert(close(contact.normal.x, -1.0F));
    assert(close(contact.point_on_cell.x, 0.9F));
    assert(close(contact.weight, std::sqrt(2.0F)));
  }
}

void test_outside_box_corner_contact_has_diagonal_normal() {
  microsimulator::WorldState state;
  add_capsule(state, {1.3F, 1.3F, 0.0F}, {0.0F, 0.0F, 1.0F}, 0.0F);
  microsimulator::ConstraintSet constraints;
  microsimulator::BoxConstraintInit box;
  constraints.add_box(box);

  const auto graph = microsimulator::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 2);
  const auto diagonal = 1.0F / std::sqrt(2.0F);
  for (const auto& contact : graph.contacts()) {
    assert(close(contact.signed_separation, 0.3F * std::sqrt(2.0F) - 0.5F));
    assert(close(contact.normal.x, -diagonal));
    assert(close(contact.normal.y, -diagonal));
    assert(close(microsimulator::norm(contact.normal), 1.0F));
  }
}

void test_box_interior_endpoint_escapes_toward_nearest_face() {
  microsimulator::WorldState state;
  add_capsule(state, {0.5F, 0.6F, 0.0F}, {1.0F, 0.0F, 0.0F}, 0.0F);
  microsimulator::ConstraintSet constraints;
  microsimulator::BoxConstraintInit box;
  box.half_extents = {2.0F, 1.0F, 3.0F};
  constraints.add_box(box);

  const auto graph = microsimulator::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 2);
  for (const auto& contact : graph.contacts()) {
    assert(close(contact.signed_separation, -0.9F));
    assert(close(contact.normal.y, -1.0F));
    assert(close(contact.point_on_cell.y, 0.1F));
  }
}

void test_inside_box_confines_like_a_chamber() {
  microsimulator::WorldState state;
  add_capsule(state, {4.8F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}, 0.0F);
  microsimulator::ConstraintSet constraints;
  microsimulator::BoxConstraintInit box;
  box.half_extents = {5.0F, 5.0F, 5.0F};
  box.allowed_region = microsimulator::ConstraintRegion::inside;
  constraints.add_box(box);

  const auto graph = microsimulator::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 2);
  for (const auto& contact : graph.contacts()) {
    assert(close(contact.signed_separation, -0.3F));
    assert(close(contact.normal.x, 1.0F));
    assert(close(contact.point_on_cell.x, 5.3F));
  }
}

void test_box_center_degeneracy_is_finite_and_deterministic() {
  microsimulator::WorldState state;
  add_capsule(state, {}, {0.0F, 1.0F, 0.0F}, 0.0F);
  microsimulator::ConstraintSet constraints;
  microsimulator::BoxConstraintInit box;
  constraints.add_box(box);

  const auto graph = microsimulator::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 2);
  for (const auto& contact : graph.contacts()) {
    assert(close(microsimulator::norm(contact.normal), 1.0F));
    assert(close(contact.normal.x, -1.0F));
    assert(close(contact.signed_separation, -1.5F));
  }
}

void test_outside_box_detects_midspan_capsule_contact() {
  microsimulator::WorldState state;
  add_capsule(state, {0.0F, 0.75F, 0.0F}, {1.0F, 0.0F, 0.0F}, 3.5F);
  microsimulator::ConstraintSet constraints;
  microsimulator::BoxConstraintInit box;
  box.half_extents = {1.0F, 1.0F, 5.0F};
  const auto box_id = constraints.add_box(box);

  const auto graph = microsimulator::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 1);
  const auto& contact = graph.contacts()[0];
  assert(contact.constraint_id == box_id);
  assert(contact.constraint_kind == microsimulator::ExternalConstraintKind::box);
  assert(contact.location == microsimulator::RodContactLocation::interior);
  assert(close(contact.point_on_cell.x, 0.0F));
  assert(close(contact.point_on_cell.y, 0.25F));
  assert(close(contact.normal.x, 0.0F));
  assert(close(contact.normal.y, -1.0F));
  assert(close(contact.signed_separation, -0.75F));
  assert(close(contact.weight, 1.0F));
}

void test_outside_box_detects_centered_wall_crossing() {
  microsimulator::WorldState state;
  add_capsule(state, {}, {1.0F, 0.0F, 0.0F}, 3.5F);
  microsimulator::ConstraintSet constraints;
  microsimulator::BoxConstraintInit wall;
  wall.half_extents = {1.0F, 5.0F, 5.0F};
  constraints.add_box(wall);

  const auto graph = microsimulator::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 1);
  const auto& contact = graph.contacts()[0];
  assert(contact.location == microsimulator::RodContactLocation::interior);
  assert(close(contact.point_on_cell.x, -0.5F));
  assert(close(contact.normal.x, -1.0F));
  assert(close(contact.signed_separation, -1.5F));
}

void test_outside_box_tilted_face_contact_keeps_both_endpoints() {
  microsimulator::WorldState state;
  add_capsule(state, {1.3F, 0.0F, 0.0F}, {0.1F, 1.0F, 0.0F}, 1.0F);
  microsimulator::ConstraintSet constraints;
  microsimulator::BoxConstraintInit box;
  box.half_extents = {1.0F, 2.0F, 3.0F};
  constraints.add_box(box);

  // Both ends are inside the face at different depths. One row per end keeps
  // the solve from trading the deeper end's overlap for the shallower end's.
  const auto half_reach = 0.5F * 0.1F / std::sqrt(1.01F);
  const auto graph = microsimulator::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 2);
  const auto contacts = graph.contacts();
  assert(contacts[0].location == microsimulator::RodContactLocation::negative);
  assert(contacts[1].location == microsimulator::RodContactLocation::positive);
  assert(close(contacts[0].signed_separation, -0.2F - half_reach));
  assert(close(contacts[1].signed_separation, -0.2F + half_reach));
  for (const auto& contact : contacts) {
    assert(close(contact.normal.x, -1.0F));
    assert(close(contact.weight, 1.0F / std::sqrt(2.0F)));
  }
}

void test_outside_box_clear_endpoint_emits_no_row() {
  microsimulator::WorldState state;
  add_capsule(state, {1.4F, 0.0F, 0.0F}, {0.6F, 0.8F, 0.0F}, 1.0F);
  microsimulator::ConstraintSet constraints;
  microsimulator::BoxConstraintInit box;
  box.half_extents = {1.0F, 2.0F, 3.0F};
  constraints.add_box(box);

  const auto graph = microsimulator::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 1);
  const auto& contact = graph.contacts()[0];
  assert(contact.location == microsimulator::RodContactLocation::negative);
  assert(close(contact.signed_separation, -0.4F));
  assert(close(contact.weight, 1.0F));
}

float deepest_penetration(const microsimulator::WorldState& state,
                          const microsimulator::ConstraintSet& constraints) {
  auto deepest = 0.0F;
  const auto graph = microsimulator::find_external_contacts_cpu(state, constraints);
  for (const auto& contact : graph.contacts()) {
    deepest = std::min(deepest, contact.signed_separation);
  }
  return deepest;
}

void test_projection_lays_a_tilted_cell_along_a_wall() {
  microsimulator::WorldState state;
  const auto id = add_capsule(state, {1.3F, 0.25F, 0.0F}, {0.1F, 1.0F, 0.0F}, 1.0F);
  microsimulator::ConstraintSet constraints;
  microsimulator::BoxConstraintInit box;
  box.half_extents = {1.0F, 2.0F, 3.0F};
  constraints.add_box(box);

  microsimulator::project_cells_out_of_constraints(state, constraints);

  // Each end moves out by its own depth, so the rod ends up parallel to the
  // face, in contact with the wall without being inside it.
  const auto after = state.cell(id);
  assert(close(after.position.x, 1.5F, 1.0e-4F));
  assert(close(after.position.y, 0.25F, 1.0e-4F));
  assert(close(after.direction.x, 0.0F, 1.0e-4F));
  assert(close(after.direction.y, 1.0F, 1.0e-4F));
  assert(close(after.length, 1.0F));
  const auto resting = deepest_penetration(state, constraints);
  assert(resting < 0.0F);
  assert(resting >= -1.0e-4F);
}

void test_projection_turns_a_wedged_cell_flat() {
  microsimulator::WorldState state;
  // A rod standing almost upright in a slot too low for it.
  const auto id = add_capsule(state, {0.0F, 0.0F, 0.8F}, {0.3F, 0.0F, 1.0F}, 1.5F);
  microsimulator::ConstraintSet constraints;
  microsimulator::PlaneConstraintInit floor;
  floor.inward_normal = {0.0F, 0.0F, 1.0F};
  constraints.add_plane(floor);
  microsimulator::PlaneConstraintInit ceiling;
  ceiling.point = {0.0F, 0.0F, 1.6F};
  ceiling.inward_normal = {0.0F, 0.0F, -1.0F};
  constraints.add_plane(ceiling);
  const auto before = deepest_penetration(state, constraints);
  assert(before < -0.4F);

  for (int pass = 0; pass < 4; ++pass) {
    microsimulator::project_cells_out_of_constraints(state, constraints);
  }

  const auto after = state.cell(id);
  assert(deepest_penetration(state, constraints) >= -1.0e-3F);
  assert(std::abs(after.direction.z) * after.length <= 0.6F + 2.0e-3F);
  assert(close(after.length, 1.5F));
}

void test_projection_resolves_a_corner_and_leaves_fixed_cells() {
  microsimulator::WorldState state;
  const auto mobile = add_capsule(state, {0.2F, 0.3F, 0.0F}, {0.0F, 0.0F, 1.0F}, 0.0F);
  microsimulator::CellInit pinned;
  pinned.position = {4.0F, 0.1F, 0.0F};
  pinned.direction = {0.0F, 0.0F, 1.0F};
  pinned.length = 0.0F;
  pinned.radius = 0.5F;
  pinned.fixed = true;
  const auto fixed = state.add_cell(pinned);
  microsimulator::ConstraintSet constraints;
  microsimulator::PlaneConstraintInit floor;
  floor.inward_normal = {0.0F, 1.0F, 0.0F};
  constraints.add_plane(floor);
  microsimulator::PlaneConstraintInit side;
  side.inward_normal = {1.0F, 0.0F, 0.0F};
  constraints.add_plane(side);

  microsimulator::project_cells_out_of_constraints(state, constraints);

  assert(close(state.cell(mobile).position.x, 0.5F, 1.0e-4F));
  assert(close(state.cell(mobile).position.y, 0.5F, 1.0e-4F));
  assert(state.cell(fixed).position.y == 0.1F);
}

void test_cylinder_ids_validation_and_checkpoint() {
  microsimulator::ConstraintSet constraints;
  microsimulator::CylinderConstraintInit cylinder;
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
  const microsimulator::ConstraintSet restored(checkpoint);
  assert(restored.cylinders().size() == 1);
  assert(close(restored.cylinders()[0].radius, 2.0F));
}

void test_outside_cylinder_barrel_contact_uses_two_weighted_endpoints() {
  microsimulator::WorldState state;
  add_capsule(state, {1.4F, 0.0F, 0.0F}, {0.0F, 0.0F, 1.0F});
  microsimulator::ConstraintSet constraints;
  microsimulator::CylinderConstraintInit cylinder;
  cylinder.coefficient = 2.0F;
  const auto cylinder_id = constraints.add_cylinder(cylinder);

  const auto graph = microsimulator::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 2);
  for (const auto& contact : graph.contacts()) {
    assert(contact.constraint_id == cylinder_id);
    assert(contact.constraint_kind == microsimulator::ExternalConstraintKind::cylinder);
    assert(close(contact.signed_separation, -0.1F));
    assert(close(contact.normal.x, -1.0F));
    assert(close(contact.point_on_cell.x, 0.9F));
    assert(close(contact.weight, std::sqrt(2.0F)));
  }
}

void test_outside_cylinder_rim_contact_has_blended_normal() {
  microsimulator::WorldState state;
  add_capsule(state, {1.3F, 0.0F, 1.3F}, {0.0F, 1.0F, 0.0F}, 0.0F);
  microsimulator::ConstraintSet constraints;
  microsimulator::CylinderConstraintInit cylinder;
  constraints.add_cylinder(cylinder);

  const auto graph = microsimulator::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 2);
  const auto diagonal = 1.0F / std::sqrt(2.0F);
  for (const auto& contact : graph.contacts()) {
    assert(close(contact.signed_separation, 0.3F * std::sqrt(2.0F) - 0.5F));
    assert(close(contact.normal.x, -diagonal));
    assert(close(contact.normal.z, -diagonal));
    assert(close(microsimulator::norm(contact.normal), 1.0F));
  }
}

void test_outside_cylinder_cap_contact_points_axially() {
  microsimulator::WorldState state;
  add_capsule(state, {0.5F, 0.0F, 1.4F}, {1.0F, 0.0F, 0.0F}, 0.0F);
  microsimulator::ConstraintSet constraints;
  microsimulator::CylinderConstraintInit cylinder;
  constraints.add_cylinder(cylinder);

  const auto graph = microsimulator::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 2);
  for (const auto& contact : graph.contacts()) {
    assert(close(contact.signed_separation, -0.1F));
    assert(close(contact.normal.z, -1.0F));
    assert(close(contact.point_on_cell.z, 0.9F));
  }
}

void test_inside_cylinder_confines_like_a_dish() {
  microsimulator::WorldState state;
  add_capsule(state, {4.8F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}, 0.0F);
  microsimulator::ConstraintSet constraints;
  microsimulator::CylinderConstraintInit cylinder;
  cylinder.radius = 5.0F;
  cylinder.half_height = 5.0F;
  cylinder.allowed_region = microsimulator::ConstraintRegion::inside;
  constraints.add_cylinder(cylinder);

  const auto graph = microsimulator::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 2);
  for (const auto& contact : graph.contacts()) {
    assert(close(contact.signed_separation, -0.3F));
    assert(close(contact.normal.x, 1.0F));
    assert(close(contact.point_on_cell.x, 5.3F));
  }
}

void test_cylinder_axis_degeneracy_is_finite_and_radial() {
  microsimulator::WorldState state;
  add_capsule(state, {}, {0.0F, 1.0F, 0.0F}, 0.0F);
  microsimulator::ConstraintSet constraints;
  microsimulator::CylinderConstraintInit cylinder;
  constraints.add_cylinder(cylinder);

  const auto graph = microsimulator::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 2);
  for (const auto& contact : graph.contacts()) {
    assert(close(microsimulator::norm(contact.normal), 1.0F));
    assert(close(contact.normal.x, -1.0F));
    assert(close(contact.signed_separation, -1.5F));
  }
}

void test_outside_cylinder_detects_midspan_capsule_contact() {
  microsimulator::WorldState state;
  add_capsule(state, {0.0F, 0.75F, 0.0F}, {1.0F, 0.0F, 0.0F}, 3.5F);
  microsimulator::ConstraintSet constraints;
  microsimulator::CylinderConstraintInit cylinder;
  cylinder.center = {0.4F, 0.0F, 0.0F};
  cylinder.half_height = 5.0F;
  const auto cylinder_id = constraints.add_cylinder(cylinder);

  const auto graph = microsimulator::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 1);
  const auto& contact = graph.contacts()[0];
  assert(contact.constraint_id == cylinder_id);
  assert(contact.constraint_kind == microsimulator::ExternalConstraintKind::cylinder);
  assert(contact.location == microsimulator::RodContactLocation::interior);
  assert(close(contact.point_on_cell.x, 0.4F));
  assert(close(contact.point_on_cell.y, 0.25F));
  assert(close(contact.normal.x, 0.0F));
  assert(close(contact.normal.y, -1.0F));
  assert(close(contact.signed_separation, -0.75F));
  assert(close(contact.weight, 1.0F));
}

void test_simulation_exposes_cpu_constraint_graph() {
  microsimulator::Simulation simulation;
  microsimulator::CellInit cell;
  cell.position.y = 0.4F;
  cell.length = 2.0F;
  simulation.add_cell(cell);
  microsimulator::PlaneConstraintInit plane;
  plane.inward_normal = {0.0F, 1.0F, 0.0F};
  simulation.add_plane_constraint(plane);
  assert(simulation.supports(microsimulator::BackendFeature::external_constraints));
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
  test_outside_box_tilted_face_contact_keeps_both_endpoints();
  test_outside_box_clear_endpoint_emits_no_row();
  test_projection_lays_a_tilted_cell_along_a_wall();
  test_projection_turns_a_wedged_cell_flat();
  test_projection_resolves_a_corner_and_leaves_fixed_cells();
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
