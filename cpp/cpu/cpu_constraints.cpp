#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <ranges>
#include <tuple>
#include <vector>

#include "cm/constraints.hpp"

namespace cm {
namespace {

constexpr float inverse_sqrt_two = 0.7071067811865475F;
constexpr std::size_t segment_minimization_iterations = 40;

struct EndpointGeometry {
  RodContactLocation location;
  Vec3 centerline_point;
};

std::array<EndpointGeometry, 2> endpoints(const CellGeometryView& geometry, std::size_t slot) {
  const Vec3 center{geometry.position_x[slot], geometry.position_y[slot],
                    geometry.position_z[slot]};
  const Vec3 axis{geometry.direction_x[slot], geometry.direction_y[slot],
                  geometry.direction_z[slot]};
  const auto half_length = geometry.lengths[slot] * 0.5F;
  return {
      EndpointGeometry{RodContactLocation::negative, center - axis * half_length},
      EndpointGeometry{RodContactLocation::positive, center + axis * half_length},
  };
}

struct SurfacePoint {
  float signed_distance;
  Vec3 outward;
};

struct CenterlineMinimum {
  Vec3 point;
  SurfacePoint surface;
};

bool segment_intersects_bounds(Vec3 start, Vec3 end, Vec3 lower, Vec3 upper) {
  const auto delta = end - start;
  const std::array<float, 3> starts{start.x, start.y, start.z};
  const std::array<float, 3> deltas{delta.x, delta.y, delta.z};
  const std::array<float, 3> lowers{lower.x, lower.y, lower.z};
  const std::array<float, 3> uppers{upper.x, upper.y, upper.z};
  auto entry = 0.0F;
  auto exit = 1.0F;
  for (std::size_t axis = 0; axis < starts.size(); ++axis) {
    if (deltas[axis] == 0.0F) {
      if (starts[axis] < lowers[axis] || starts[axis] > uppers[axis]) {
        return false;
      }
      continue;
    }
    auto first = (lowers[axis] - starts[axis]) / deltas[axis];
    auto second = (uppers[axis] - starts[axis]) / deltas[axis];
    if (first > second) {
      std::swap(first, second);
    }
    entry = std::max(entry, first);
    exit = std::min(exit, second);
    if (entry > exit) {
      return false;
    }
  }
  return true;
}

template <typename Surface>
CenterlineMinimum minimize_surface_on_segment(Vec3 start, Vec3 end, const Surface& surface_at) {
  const auto delta = end - start;
  auto lower = 0.0F;
  auto upper = 1.0F;
  for (std::size_t iteration = 0; iteration < segment_minimization_iterations; ++iteration) {
    const auto first_parameter = lower + (upper - lower) / 3.0F;
    const auto second_parameter = upper - (upper - lower) / 3.0F;
    const auto first = surface_at(start + delta * first_parameter);
    const auto second = surface_at(start + delta * second_parameter);
    if (first.signed_distance < second.signed_distance) {
      upper = second_parameter;
    } else if (second.signed_distance < first.signed_distance) {
      lower = first_parameter;
    } else {
      lower = first_parameter;
      upper = second_parameter;
    }
  }

  CenterlineMinimum result{start, surface_at(start)};
  const std::array<float, 5> candidates{1.0F, 0.5F, lower, (lower + upper) * 0.5F, upper};
  for (const auto parameter : candidates) {
    const auto point = start + delta * parameter;
    const auto surface = surface_at(point);
    if (surface.signed_distance < result.surface.signed_distance) {
      result = {point, surface};
    }
  }
  return result;
}

void append_outside_minimum_contacts(std::vector<ExternalContact>& contacts,
                                     const CellGeometryView& geometry, std::size_t slot,
                                     ConstraintId constraint_id,
                                     ExternalConstraintKind constraint_kind, float coefficient,
                                     const std::array<EndpointGeometry, 2>& cell_endpoints,
                                     const std::array<SurfacePoint, 2>& endpoint_surfaces,
                                     const CenterlineMinimum& minimum,
                                     const ConstraintContactParameters& parameters) {
  const auto radius = geometry.radii[slot];
  const auto minimum_separation = minimum.surface.signed_distance - radius;
  if (minimum_separation >= parameters.activation_margin) {
    return;
  }

  struct Candidate {
    RodContactLocation location;
    Vec3 centerline_point;
    SurfacePoint surface;
  };
  std::array<Candidate, 2> candidates{};
  std::size_t count = 0;
  for (std::size_t index = 0; index < cell_endpoints.size(); ++index) {
    const auto endpoint_separation = endpoint_surfaces[index].signed_distance - radius;
    if (endpoint_separation < parameters.activation_margin &&
        std::abs(endpoint_surfaces[index].signed_distance - minimum.surface.signed_distance) <=
            parameters.degeneracy_epsilon) {
      candidates[count++] = {
          cell_endpoints[index].location,
          cell_endpoints[index].centerline_point,
          endpoint_surfaces[index],
      };
    }
  }
  if (count == 0) {
    candidates[count++] = {RodContactLocation::interior, minimum.point, minimum.surface};
  }

  const auto weight = coefficient * (count == 2 ? inverse_sqrt_two : 1.0F);
  for (std::size_t index = 0; index < count; ++index) {
    const auto normal = candidates[index].surface.outward * -1.0F;
    contacts.push_back({
        .cell_id = geometry.ids[slot],
        .cell_slot = static_cast<Slot>(slot),
        .constraint_id = constraint_id,
        .constraint_kind = constraint_kind,
        .location = candidates[index].location,
        .point_on_cell = candidates[index].centerline_point + normal * radius,
        .normal = normal,
        .signed_separation = candidates[index].surface.signed_distance - radius,
        .weight = weight,
    });
  }
}

void append_plane_contacts(std::vector<ExternalContact>& contacts, const CellGeometryView& geometry,
                           std::size_t slot, const PlaneConstraint& plane,
                           const ConstraintContactParameters& parameters) {
  const auto cell_endpoints = endpoints(geometry, slot);
  std::array<float, 2> separations{};
  std::array<bool, 2> active{};
  for (std::size_t index = 0; index < cell_endpoints.size(); ++index) {
    separations[index] =
        dot(cell_endpoints[index].centerline_point - plane.point, plane.inward_normal) -
        geometry.radii[slot];
    active[index] = separations[index] < parameters.activation_margin;
  }
  const auto active_count = static_cast<unsigned>(active[0]) + static_cast<unsigned>(active[1]);
  const auto weight = plane.coefficient * (active_count == 2 ? inverse_sqrt_two : 1.0F);
  const auto normal = plane.inward_normal * -1.0F;
  for (std::size_t index = 0; index < cell_endpoints.size(); ++index) {
    if (!active[index]) {
      continue;
    }
    contacts.push_back({
        .cell_id = geometry.ids[slot],
        .cell_slot = static_cast<Slot>(slot),
        .constraint_id = plane.id,
        .constraint_kind = ExternalConstraintKind::plane,
        .location = cell_endpoints[index].location,
        .point_on_cell = cell_endpoints[index].centerline_point + normal * geometry.radii[slot],
        .normal = normal,
        .signed_separation = separations[index],
        .weight = weight,
    });
  }
}

void append_sphere_contacts(std::vector<ExternalContact>& contacts,
                            const CellGeometryView& geometry, std::size_t slot,
                            const SphereConstraint& sphere,
                            const ConstraintContactParameters& parameters) {
  const auto cell_endpoints = endpoints(geometry, slot);
  std::array<SurfacePoint, 2> surfaces{};
  for (std::size_t index = 0; index < cell_endpoints.size(); ++index) {
    const auto center_delta = cell_endpoints[index].centerline_point - sphere.center;
    const auto distance = norm(center_delta);
    const auto radial = distance > parameters.degeneracy_epsilon ? center_delta * (1.0F / distance)
                                                                 : Vec3{1.0F, 0.0F, 0.0F};
    surfaces[index] = {distance - sphere.radius, radial};
  }
  if (sphere.allowed_region == SphereRegion::outside) {
    const auto start = cell_endpoints[0].centerline_point;
    const auto end = cell_endpoints[1].centerline_point;
    const auto delta = end - start;
    const auto length_squared = dot(delta, delta);
    const auto parameter =
        length_squared > parameters.degeneracy_epsilon * parameters.degeneracy_epsilon
            ? std::clamp(-dot(start - sphere.center, delta) / length_squared, 0.0F, 1.0F)
            : 0.0F;
    const auto point = start + delta * parameter;
    const auto center_delta = point - sphere.center;
    const auto distance = norm(center_delta);
    const auto radial = distance > parameters.degeneracy_epsilon ? center_delta * (1.0F / distance)
                                                                 : Vec3{1.0F, 0.0F, 0.0F};
    append_outside_minimum_contacts(
        contacts, geometry, slot, sphere.id, ExternalConstraintKind::sphere, sphere.coefficient,
        cell_endpoints, surfaces, {point, {distance - sphere.radius, radial}}, parameters);
    return;
  }

  std::array<float, 2> separations{};
  std::array<Vec3, 2> normals{};
  std::array<bool, 2> active{};
  for (std::size_t index = 0; index < cell_endpoints.size(); ++index) {
    separations[index] = -surfaces[index].signed_distance - geometry.radii[slot];
    normals[index] = surfaces[index].outward;
    active[index] = separations[index] < parameters.activation_margin;
  }
  const auto active_count = static_cast<unsigned>(active[0]) + static_cast<unsigned>(active[1]);
  const auto weight = sphere.coefficient * (active_count == 2 ? inverse_sqrt_two : 1.0F);
  for (std::size_t index = 0; index < cell_endpoints.size(); ++index) {
    if (!active[index]) {
      continue;
    }
    contacts.push_back({
        .cell_id = geometry.ids[slot],
        .cell_slot = static_cast<Slot>(slot),
        .constraint_id = sphere.id,
        .constraint_kind = ExternalConstraintKind::sphere,
        .location = cell_endpoints[index].location,
        .point_on_cell =
            cell_endpoints[index].centerline_point + normals[index] * geometry.radii[slot],
        .normal = normals[index],
        .signed_separation = separations[index],
        .weight = weight,
    });
  }
}

SurfacePoint box_surface(const Vec3& point, const BoxConstraint& box, float degeneracy_epsilon) {
  const auto delta = point - box.center;
  const Vec3 clamped{
      std::clamp(delta.x, -box.half_extents.x, box.half_extents.x),
      std::clamp(delta.y, -box.half_extents.y, box.half_extents.y),
      std::clamp(delta.z, -box.half_extents.z, box.half_extents.z),
  };
  const auto outside_vector = delta - clamped;
  const auto outside_distance = norm(outside_vector);
  if (outside_distance > degeneracy_epsilon) {
    return {outside_distance, outside_vector * (1.0F / outside_distance)};
  }
  const std::array<float, 3> clearances{
      box.half_extents.x - std::abs(delta.x),
      box.half_extents.y - std::abs(delta.y),
      box.half_extents.z - std::abs(delta.z),
  };
  std::size_t nearest_axis = 0;
  for (std::size_t axis = 1; axis < clearances.size(); ++axis) {
    if (clearances[axis] < clearances[nearest_axis]) {
      nearest_axis = axis;
    }
  }
  const std::array<float, 3> offsets{delta.x, delta.y, delta.z};
  const auto sign =
      std::abs(offsets[nearest_axis]) <= degeneracy_epsilon || offsets[nearest_axis] >= 0.0F
          ? 1.0F
          : -1.0F;
  Vec3 outward{};
  if (nearest_axis == 0) {
    outward = {sign, 0.0F, 0.0F};
  } else if (nearest_axis == 1) {
    outward = {0.0F, sign, 0.0F};
  } else {
    outward = {0.0F, 0.0F, sign};
  }
  return {-clearances[nearest_axis], outward};
}

void append_box_contacts(std::vector<ExternalContact>& contacts, const CellGeometryView& geometry,
                         std::size_t slot, const BoxConstraint& box,
                         const ConstraintContactParameters& parameters) {
  const auto cell_endpoints = endpoints(geometry, slot);
  std::array<SurfacePoint, 2> surfaces{};
  for (std::size_t index = 0; index < cell_endpoints.size(); ++index) {
    surfaces[index] =
        box_surface(cell_endpoints[index].centerline_point, box, parameters.degeneracy_epsilon);
  }
  if (box.allowed_region == ConstraintRegion::outside) {
    const auto reach = geometry.radii[slot] + parameters.activation_margin;
    const Vec3 lower{
        box.center.x - box.half_extents.x - reach,
        box.center.y - box.half_extents.y - reach,
        box.center.z - box.half_extents.z - reach,
    };
    const Vec3 upper{
        box.center.x + box.half_extents.x + reach,
        box.center.y + box.half_extents.y + reach,
        box.center.z + box.half_extents.z + reach,
    };
    if (!segment_intersects_bounds(cell_endpoints[0].centerline_point,
                                   cell_endpoints[1].centerline_point, lower, upper)) {
      return;
    }
    const auto minimum = minimize_surface_on_segment(
        cell_endpoints[0].centerline_point, cell_endpoints[1].centerline_point,
        [&box, &parameters](const Vec3& point) {
          return box_surface(point, box, parameters.degeneracy_epsilon);
        });
    append_outside_minimum_contacts(contacts, geometry, slot, box.id, ExternalConstraintKind::box,
                                    box.coefficient, cell_endpoints, surfaces, minimum, parameters);
    return;
  }

  std::array<float, 2> separations{};
  std::array<Vec3, 2> normals{};
  std::array<bool, 2> active{};
  for (std::size_t index = 0; index < cell_endpoints.size(); ++index) {
    separations[index] = -surfaces[index].signed_distance - geometry.radii[slot];
    normals[index] = surfaces[index].outward;
    active[index] = separations[index] < parameters.activation_margin;
  }
  const auto active_count = static_cast<unsigned>(active[0]) + static_cast<unsigned>(active[1]);
  const auto weight = box.coefficient * (active_count == 2 ? inverse_sqrt_two : 1.0F);
  for (std::size_t index = 0; index < cell_endpoints.size(); ++index) {
    if (!active[index]) {
      continue;
    }
    contacts.push_back({
        .cell_id = geometry.ids[slot],
        .cell_slot = static_cast<Slot>(slot),
        .constraint_id = box.id,
        .constraint_kind = ExternalConstraintKind::box,
        .location = cell_endpoints[index].location,
        .point_on_cell =
            cell_endpoints[index].centerline_point + normals[index] * geometry.radii[slot],
        .normal = normals[index],
        .signed_separation = separations[index],
        .weight = weight,
    });
  }
}

SurfacePoint cylinder_surface(const Vec3& point, const CylinderConstraint& cylinder,
                              float degeneracy_epsilon) {
  const Vec3 delta{point.x - cylinder.center.x, point.y - cylinder.center.y, 0.0F};
  const auto radial_distance = norm(delta);
  const auto radial = radial_distance > degeneracy_epsilon ? delta * (1.0F / radial_distance)
                                                           : Vec3{1.0F, 0.0F, 0.0F};
  const auto z_offset = point.z - cylinder.center.z;
  const auto z_sign = z_offset >= 0.0F ? 1.0F : -1.0F;
  const Vec3 axial{0.0F, 0.0F, z_sign};
  const auto radial_excess = radial_distance - cylinder.radius;
  const auto axial_excess = std::abs(z_offset) - cylinder.half_height;
  if (radial_excess > 0.0F && axial_excess > 0.0F) {
    const auto distance =
        std::sqrt((radial_excess * radial_excess) + (axial_excess * axial_excess));
    return {distance, (radial * radial_excess + axial * axial_excess) * (1.0F / distance)};
  }
  if (radial_excess > 0.0F) {
    return {radial_excess, radial};
  }
  if (axial_excess > 0.0F) {
    return {axial_excess, axial};
  }
  if (-radial_excess <= -axial_excess) {
    return {radial_excess, radial};
  }
  return {axial_excess, axial};
}

CenterlineMinimum minimize_cylinder_surface_on_segment(
    Vec3 start, Vec3 end, const CylinderConstraint& cylinder,
    const ConstraintContactParameters& parameters) {
  const auto surface_at = [&cylinder, &parameters](const Vec3& point) {
    return cylinder_surface(point, cylinder, parameters.degeneracy_epsilon);
  };
  auto result = minimize_surface_on_segment(start, end, surface_at);
  const auto delta = end - start;
  const auto consider = [&result, &start, &delta, &surface_at](float parameter) {
    const auto point = start + delta * std::clamp(parameter, 0.0F, 1.0F);
    const auto surface = surface_at(point);
    if (surface.signed_distance <= result.surface.signed_distance) {
      result = {point, surface};
    }
  };

  if (std::abs(delta.z) > parameters.degeneracy_epsilon) {
    consider((cylinder.center.z - start.z) / delta.z);
  }
  const auto radial_length_squared = delta.x * delta.x + delta.y * delta.y;
  if (radial_length_squared > parameters.degeneracy_epsilon * parameters.degeneracy_epsilon) {
    consider(-((start.x - cylinder.center.x) * delta.x + (start.y - cylinder.center.y) * delta.y) /
             radial_length_squared);
  }
  return result;
}

void append_cylinder_contacts(std::vector<ExternalContact>& contacts,
                              const CellGeometryView& geometry, std::size_t slot,
                              const CylinderConstraint& cylinder,
                              const ConstraintContactParameters& parameters) {
  const auto cell_endpoints = endpoints(geometry, slot);
  std::array<SurfacePoint, 2> surfaces{};
  for (std::size_t index = 0; index < cell_endpoints.size(); ++index) {
    surfaces[index] = cylinder_surface(cell_endpoints[index].centerline_point, cylinder,
                                       parameters.degeneracy_epsilon);
  }
  if (cylinder.allowed_region == ConstraintRegion::outside) {
    const auto reach = geometry.radii[slot] + parameters.activation_margin;
    const Vec3 lower{
        cylinder.center.x - cylinder.radius - reach,
        cylinder.center.y - cylinder.radius - reach,
        cylinder.center.z - cylinder.half_height - reach,
    };
    const Vec3 upper{
        cylinder.center.x + cylinder.radius + reach,
        cylinder.center.y + cylinder.radius + reach,
        cylinder.center.z + cylinder.half_height + reach,
    };
    if (!segment_intersects_bounds(cell_endpoints[0].centerline_point,
                                   cell_endpoints[1].centerline_point, lower, upper)) {
      return;
    }
    const auto minimum = minimize_cylinder_surface_on_segment(cell_endpoints[0].centerline_point,
                                                              cell_endpoints[1].centerline_point,
                                                              cylinder, parameters);
    append_outside_minimum_contacts(contacts, geometry, slot, cylinder.id,
                                    ExternalConstraintKind::cylinder, cylinder.coefficient,
                                    cell_endpoints, surfaces, minimum, parameters);
    return;
  }

  std::array<float, 2> separations{};
  std::array<Vec3, 2> normals{};
  std::array<bool, 2> active{};
  for (std::size_t index = 0; index < cell_endpoints.size(); ++index) {
    separations[index] = -surfaces[index].signed_distance - geometry.radii[slot];
    normals[index] = surfaces[index].outward;
    active[index] = separations[index] < parameters.activation_margin;
  }
  const auto active_count = static_cast<unsigned>(active[0]) + static_cast<unsigned>(active[1]);
  const auto weight = cylinder.coefficient * (active_count == 2 ? inverse_sqrt_two : 1.0F);
  for (std::size_t index = 0; index < cell_endpoints.size(); ++index) {
    if (!active[index]) {
      continue;
    }
    contacts.push_back({
        .cell_id = geometry.ids[slot],
        .cell_slot = static_cast<Slot>(slot),
        .constraint_id = cylinder.id,
        .constraint_kind = ExternalConstraintKind::cylinder,
        .location = cell_endpoints[index].location,
        .point_on_cell =
            cell_endpoints[index].centerline_point + normals[index] * geometry.radii[slot],
        .normal = normals[index],
        .signed_separation = separations[index],
        .weight = weight,
    });
  }
}

}  // namespace

ExternalContactGraph find_external_contacts_cpu(const WorldState& state,
                                                const ConstraintSet& constraints,
                                                const ConstraintContactParameters& parameters) {
  validate_constraint_contact_parameters(parameters);
  state.validate();
  const auto geometry = state.geometry_state();
  std::vector<ExternalContact> contacts;
  for (std::size_t slot = 0; slot < geometry.size(); ++slot) {
    for (const auto& plane : constraints.planes()) {
      append_plane_contacts(contacts, geometry, slot, plane, parameters);
    }
    for (const auto& sphere : constraints.spheres()) {
      append_sphere_contacts(contacts, geometry, slot, sphere, parameters);
    }
    for (const auto& box : constraints.boxes()) {
      append_box_contacts(contacts, geometry, slot, box, parameters);
    }
    for (const auto& cylinder : constraints.cylinders()) {
      append_cylinder_contacts(contacts, geometry, slot, cylinder, parameters);
    }
  }
  std::ranges::sort(contacts, {}, [](const ExternalContact& contact) {
    return std::tuple{contact.cell_id, contact.constraint_id, contact.location};
  });
  return ExternalContactGraph(geometry.size(), std::move(contacts));
}

}  // namespace cm
