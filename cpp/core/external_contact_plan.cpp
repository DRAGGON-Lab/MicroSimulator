#include "core/external_contact_plan.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace microsimulator::detail {

ExternalContactPlan plan_external_contacts(const WorldState& state,
                                           const ConstraintSet& constraints,
                                           const ConstraintContactParameters& parameters,
                                           bool every_face) {
  validate_constraint_contact_parameters(parameters);
  state.validate();
  const auto geometry = state.geometry_state();
  if (geometry.size() > std::numeric_limits<std::uint32_t>::max() ||
      constraints.size() > std::numeric_limits<std::uint32_t>::max() ||
      (constraints.size() != 0 &&
       geometry.size() > std::numeric_limits<std::uint32_t>::max() / constraints.size())) {
    throw std::overflow_error("external contact pairs exceed the uint32 index space");
  }
  ExternalContactPlan plan;
  plan.data.cell_count = static_cast<std::uint32_t>(geometry.size());
  plan.data.constraint_count = static_cast<std::uint32_t>(constraints.size());
  plan.data.every_face = every_face ? 1U : 0U;
  plan.data.activation_margin = parameters.activation_margin;
  plan.data.degeneracy_epsilon = parameters.degeneracy_epsilon;

  plan.centers.resize(std::max<std::size_t>(geometry.size(), 1));
  plan.axes.resize(plan.centers.size());
  plan.cell_ids.assign(geometry.ids.begin(), geometry.ids.end());
  for (std::size_t slot = 0; slot < geometry.size(); ++slot) {
    plan.centers[slot] = {geometry.position_x[slot], geometry.position_y[slot],
                          geometry.position_z[slot], geometry.lengths[slot]};
    plan.axes[slot] = {geometry.direction_x[slot], geometry.direction_y[slot],
                       geometry.direction_z[slot], geometry.radii[slot]};
  }

  auto& packed = plan.constraints;
  packed.reserve(std::max<std::size_t>(constraints.size(), 1));
  for (const auto& plane : constraints.planes()) {
    packed.push_back({
        .id = plane.id,
        .kind = static_cast<std::uint32_t>(ExternalConstraintKind::plane),
        .allowed_region = 0,
        .geometry = {plane.point.x, plane.point.y, plane.point.z, 0.0F},
        .parameters = {plane.inward_normal.x, plane.inward_normal.y, plane.inward_normal.z,
                       plane.coefficient},
    });
  }
  for (const auto& sphere : constraints.spheres()) {
    packed.push_back({
        .id = sphere.id,
        .kind = static_cast<std::uint32_t>(ExternalConstraintKind::sphere),
        .allowed_region = static_cast<std::uint32_t>(sphere.allowed_region),
        .geometry = {sphere.center.x, sphere.center.y, sphere.center.z, sphere.radius},
        .parameters = {0.0F, 0.0F, 0.0F, sphere.coefficient},
    });
  }
  for (const auto& box : constraints.boxes()) {
    packed.push_back({
        .id = box.id,
        .kind = static_cast<std::uint32_t>(ExternalConstraintKind::box),
        .allowed_region = static_cast<std::uint32_t>(box.allowed_region),
        .geometry = {box.center.x, box.center.y, box.center.z, 0.0F},
        .parameters = {box.half_extents.x, box.half_extents.y, box.half_extents.z, box.coefficient},
    });
  }
  for (const auto& cylinder : constraints.cylinders()) {
    packed.push_back({
        .id = cylinder.id,
        .kind = static_cast<std::uint32_t>(ExternalConstraintKind::cylinder),
        .allowed_region = static_cast<std::uint32_t>(cylinder.allowed_region),
        .geometry = {cylinder.center.x, cylinder.center.y, cylinder.center.z, cylinder.radius},
        .parameters = {cylinder.half_height, 0.0F, 0.0F, cylinder.coefficient},
    });
  }
  std::ranges::sort(packed, {}, &device::ExternalConstraint::id);
  if (packed.empty()) {
    packed.push_back({});
  }
  return plan;
}

ExternalContactGraph finish_external_contacts(
    const ExternalContactPlan& plan, std::span<const device::ExternalContactRecord> records) {
  std::vector<ExternalContact> contacts;
  contacts.reserve(records.size());
  for (const auto& record : records) {
    const auto& constraint = plan.constraints[record.constraint];
    contacts.push_back({
        .cell_id = plan.cell_ids[record.cell],
        .cell_slot = static_cast<Slot>(record.cell),
        .constraint_id = constraint.id,
        .constraint_kind = static_cast<ExternalConstraintKind>(constraint.kind),
        .location = static_cast<RodContactLocation>(record.location),
        .point_on_cell = {record.point.x, record.point.y, record.point.z},
        .normal = {record.normal.x, record.normal.y, record.normal.z},
        .signed_separation = record.point.w,
        .weight = record.normal.w,
    });
  }
  std::ranges::stable_sort(contacts, {}, [](const ExternalContact& contact) {
    return std::tuple{contact.cell_id, contact.constraint_id, contact.location};
  });
  return ExternalContactGraph(plan.data.cell_count, std::move(contacts));
}

}  // namespace microsimulator::detail
