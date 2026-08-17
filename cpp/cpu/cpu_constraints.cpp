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

struct EndpointGeometry {
  RodEndpoint endpoint;
  Vec3 centerline_point;
};

std::array<EndpointGeometry, 2> endpoints(const CellGeometryView& geometry, std::size_t slot) {
  const Vec3 center{geometry.position_x[slot], geometry.position_y[slot],
                    geometry.position_z[slot]};
  const Vec3 axis{geometry.direction_x[slot], geometry.direction_y[slot],
                  geometry.direction_z[slot]};
  const auto half_length = geometry.lengths[slot] * 0.5F;
  return {
      EndpointGeometry{RodEndpoint::negative, center - axis * half_length},
      EndpointGeometry{RodEndpoint::positive, center + axis * half_length},
  };
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
        .endpoint = cell_endpoints[index].endpoint,
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
  std::array<float, 2> separations{};
  std::array<Vec3, 2> normals{};
  std::array<bool, 2> active{};
  for (std::size_t index = 0; index < cell_endpoints.size(); ++index) {
    const auto center_delta = cell_endpoints[index].centerline_point - sphere.center;
    const auto distance = norm(center_delta);
    const auto radial = distance > parameters.degeneracy_epsilon ? center_delta * (1.0F / distance)
                                                                 : Vec3{1.0F, 0.0F, 0.0F};
    if (sphere.allowed_region == SphereRegion::outside) {
      separations[index] = distance - sphere.radius - geometry.radii[slot];
      normals[index] = radial * -1.0F;
    } else {
      separations[index] = sphere.radius - distance - geometry.radii[slot];
      normals[index] = radial;
    }
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
        .endpoint = cell_endpoints[index].endpoint,
        .point_on_cell =
            cell_endpoints[index].centerline_point + normals[index] * geometry.radii[slot],
        .normal = normals[index],
        .signed_separation = separations[index],
        .weight = weight,
    });
  }
}

struct BoxSurface {
  float signed_distance;
  Vec3 outward;
};

BoxSurface box_surface(const Vec3& point, const BoxConstraint& box, float degeneracy_epsilon) {
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
  const auto sign = offsets[nearest_axis] >= 0.0F ? 1.0F : -1.0F;
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
  std::array<float, 2> separations{};
  std::array<Vec3, 2> normals{};
  std::array<bool, 2> active{};
  for (std::size_t index = 0; index < cell_endpoints.size(); ++index) {
    const auto surface = box_surface(cell_endpoints[index].centerline_point, box,
                                     parameters.degeneracy_epsilon);
    if (box.allowed_region == ConstraintRegion::outside) {
      separations[index] = surface.signed_distance - geometry.radii[slot];
      normals[index] = surface.outward * -1.0F;
    } else {
      separations[index] = -surface.signed_distance - geometry.radii[slot];
      normals[index] = surface.outward;
    }
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
        .endpoint = cell_endpoints[index].endpoint,
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
  }
  std::ranges::sort(contacts, {}, [](const ExternalContact& contact) {
    return std::tuple{contact.cell_id, contact.constraint_id, contact.endpoint};
  });
  return ExternalContactGraph(geometry.size(), std::move(contacts));
}

}  // namespace cm
