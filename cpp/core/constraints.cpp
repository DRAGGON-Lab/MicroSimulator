#include "cm/constraints.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace cm {
namespace {

bool finite(const Vec3& value) {
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

void validate_coefficient(float coefficient) {
  if (!std::isfinite(coefficient) || coefficient <= 0.0F) {
    throw std::invalid_argument("constraint coefficient must be finite and positive");
  }
}

void validate_plane(const PlaneConstraint& plane) {
  if (plane.id == invalid_constraint_id || !finite(plane.point) || !finite(plane.inward_normal)) {
    throw std::invalid_argument("checkpoint plane contains an invalid field");
  }
  validate_coefficient(plane.coefficient);
  if (std::abs(norm(plane.inward_normal) - 1.0F) > 1.0e-5F) {
    throw std::invalid_argument("checkpoint plane inward normal is not normalized");
  }
}

void validate_sphere(const SphereConstraint& sphere) {
  if (sphere.id == invalid_constraint_id || !finite(sphere.center) ||
      !std::isfinite(sphere.radius) || sphere.radius <= 0.0F) {
    throw std::invalid_argument("checkpoint sphere contains invalid geometry");
  }
  validate_coefficient(sphere.coefficient);
  switch (sphere.allowed_region) {
    case ConstraintRegion::outside:
    case ConstraintRegion::inside:
      return;
  }
  throw std::invalid_argument("checkpoint sphere uses an unknown allowed region");
}

bool positive_finite_extents(const Vec3& half_extents) {
  return std::isfinite(half_extents.x) && half_extents.x > 0.0F &&
         std::isfinite(half_extents.y) && half_extents.y > 0.0F &&
         std::isfinite(half_extents.z) && half_extents.z > 0.0F;
}

void validate_box(const BoxConstraint& box) {
  if (box.id == invalid_constraint_id || !finite(box.center) ||
      !positive_finite_extents(box.half_extents)) {
    throw std::invalid_argument("checkpoint box contains invalid geometry");
  }
  validate_coefficient(box.coefficient);
  switch (box.allowed_region) {
    case ConstraintRegion::outside:
    case ConstraintRegion::inside:
      return;
  }
  throw std::invalid_argument("checkpoint box uses an unknown allowed region");
}

void validate_cylinder(const CylinderConstraint& cylinder) {
  if (cylinder.id == invalid_constraint_id || !finite(cylinder.center) ||
      !std::isfinite(cylinder.radius) || cylinder.radius <= 0.0F ||
      !std::isfinite(cylinder.half_height) || cylinder.half_height <= 0.0F) {
    throw std::invalid_argument("checkpoint cylinder contains invalid geometry");
  }
  validate_coefficient(cylinder.coefficient);
  switch (cylinder.allowed_region) {
    case ConstraintRegion::outside:
    case ConstraintRegion::inside:
      return;
  }
  throw std::invalid_argument("checkpoint cylinder uses an unknown allowed region");
}

void validate_constraint_state(ConstraintId next_id, std::span<const PlaneConstraint> planes,
                               std::span<const SphereConstraint> spheres,
                               std::span<const BoxConstraint> boxes,
                               std::span<const CylinderConstraint> cylinders) {
  if (next_id == invalid_constraint_id) {
    throw std::invalid_argument("checkpoint next constraint identifier is invalid");
  }
  std::size_t total = planes.size();
  for (const auto count : {spheres.size(), boxes.size(), cylinders.size()}) {
    if (count > std::numeric_limits<std::size_t>::max() - total) {
      throw std::overflow_error("checkpoint constraint count overflow");
    }
    total += count;
  }
  std::unordered_set<ConstraintId> ids;
  ids.reserve(total);
  ConstraintId previous_plane = invalid_constraint_id;
  for (const auto& plane : planes) {
    validate_plane(plane);
    if (plane.id <= previous_plane || plane.id >= next_id) {
      throw std::invalid_argument("checkpoint plane identifiers are not ordered and allocated");
    }
    if (!ids.insert(plane.id).second) {
      throw std::invalid_argument("checkpoint contains a duplicate constraint identifier");
    }
    previous_plane = plane.id;
  }
  ConstraintId previous_sphere = invalid_constraint_id;
  for (const auto& sphere : spheres) {
    validate_sphere(sphere);
    if (sphere.id <= previous_sphere || sphere.id >= next_id) {
      throw std::invalid_argument("checkpoint sphere identifiers are not ordered and allocated");
    }
    if (!ids.insert(sphere.id).second) {
      throw std::invalid_argument("checkpoint contains a duplicate constraint identifier");
    }
    previous_sphere = sphere.id;
  }
  ConstraintId previous_box = invalid_constraint_id;
  for (const auto& box : boxes) {
    validate_box(box);
    if (box.id <= previous_box || box.id >= next_id) {
      throw std::invalid_argument("checkpoint box identifiers are not ordered and allocated");
    }
    if (!ids.insert(box.id).second) {
      throw std::invalid_argument("checkpoint contains a duplicate constraint identifier");
    }
    previous_box = box.id;
  }
  ConstraintId previous_cylinder = invalid_constraint_id;
  for (const auto& cylinder : cylinders) {
    validate_cylinder(cylinder);
    if (cylinder.id <= previous_cylinder || cylinder.id >= next_id) {
      throw std::invalid_argument("checkpoint cylinder identifiers are not ordered and allocated");
    }
    if (!ids.insert(cylinder.id).second) {
      throw std::invalid_argument("checkpoint contains a duplicate constraint identifier");
    }
    previous_cylinder = cylinder.id;
  }
}

std::size_t checked_offset_count(std::size_t cell_count) {
  if (cell_count > static_cast<std::size_t>(invalid_slot) ||
      cell_count == std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("external contact graph exceeds the slot index space");
  }
  return cell_count + 1;
}

void validate_contact(const ExternalContact& contact, std::size_t cell_count) {
  if (contact.cell_id == invalid_cell_id || contact.constraint_id == invalid_constraint_id) {
    throw std::invalid_argument("external contact has an invalid identity");
  }
  if (contact.cell_slot >= cell_count) {
    throw std::invalid_argument("external contact cell slot is invalid");
  }
  if (!finite(contact.point_on_cell) || !finite(contact.normal) ||
      !std::isfinite(contact.signed_separation) || !std::isfinite(contact.weight) ||
      contact.weight <= 0.0F) {
    throw std::invalid_argument("external contact contains a non-finite or invalid field");
  }
  if (std::abs(norm(contact.normal) - 1.0F) > 1.0e-4F) {
    throw std::invalid_argument("external contact normal is not unit length");
  }
}

}  // namespace

void ConstraintSetCheckpoint::validate() const {
  validate_constraint_state(next_id, planes, spheres, boxes, cylinders);
}

ConstraintSet::ConstraintSet(const ConstraintSetCheckpoint& checkpoint)
    : next_id_(checkpoint.next_id),
      planes_(checkpoint.planes),
      spheres_(checkpoint.spheres),
      boxes_(checkpoint.boxes),
      cylinders_(checkpoint.cylinders) {
  checkpoint.validate();
}

ConstraintId ConstraintSet::allocate_id() {
  if (next_id_ == invalid_constraint_id || next_id_ == std::numeric_limits<ConstraintId>::max()) {
    throw std::overflow_error("constraint identifier space exhausted");
  }
  return next_id_++;
}

ConstraintId ConstraintSet::add_plane(const PlaneConstraintInit& plane) {
  if (!finite(plane.point) || !finite(plane.inward_normal)) {
    throw std::invalid_argument("plane fields must be finite");
  }
  const auto normal_magnitude = norm(plane.inward_normal);
  if (!std::isfinite(normal_magnitude) || normal_magnitude <= 0.0F) {
    throw std::invalid_argument("plane inward normal must be non-zero");
  }
  validate_coefficient(plane.coefficient);
  const auto id = allocate_id();
  planes_.push_back({
      .id = id,
      .point = plane.point,
      .inward_normal = plane.inward_normal * (1.0F / normal_magnitude),
      .coefficient = plane.coefficient,
  });
  return id;
}

ConstraintId ConstraintSet::add_sphere(const SphereConstraintInit& sphere) {
  if (!finite(sphere.center) || !std::isfinite(sphere.radius) || sphere.radius <= 0.0F) {
    throw std::invalid_argument("sphere geometry must be finite with a positive radius");
  }
  validate_coefficient(sphere.coefficient);
  const auto id = allocate_id();
  spheres_.push_back({
      .id = id,
      .center = sphere.center,
      .radius = sphere.radius,
      .coefficient = sphere.coefficient,
      .allowed_region = sphere.allowed_region,
  });
  return id;
}

ConstraintId ConstraintSet::add_box(const BoxConstraintInit& box) {
  if (!finite(box.center) || !positive_finite_extents(box.half_extents)) {
    throw std::invalid_argument("box geometry must be finite with positive half extents");
  }
  validate_coefficient(box.coefficient);
  const auto id = allocate_id();
  boxes_.push_back({
      .id = id,
      .center = box.center,
      .half_extents = box.half_extents,
      .coefficient = box.coefficient,
      .allowed_region = box.allowed_region,
  });
  return id;
}

ConstraintId ConstraintSet::add_cylinder(const CylinderConstraintInit& cylinder) {
  if (!finite(cylinder.center) || !std::isfinite(cylinder.radius) || cylinder.radius <= 0.0F ||
      !std::isfinite(cylinder.half_height) || cylinder.half_height <= 0.0F) {
    throw std::invalid_argument(
        "cylinder geometry must be finite with a positive radius and half height");
  }
  validate_coefficient(cylinder.coefficient);
  const auto id = allocate_id();
  cylinders_.push_back({
      .id = id,
      .center = cylinder.center,
      .radius = cylinder.radius,
      .half_height = cylinder.half_height,
      .coefficient = cylinder.coefficient,
      .allowed_region = cylinder.allowed_region,
  });
  return id;
}

std::size_t ConstraintSet::size() const noexcept {
  return planes_.size() + spheres_.size() + boxes_.size() + cylinders_.size();
}

bool ConstraintSet::empty() const noexcept {
  return planes_.empty() && spheres_.empty() && boxes_.empty() && cylinders_.empty();
}

std::span<const PlaneConstraint> ConstraintSet::planes() const& noexcept { return planes_; }

std::span<const SphereConstraint> ConstraintSet::spheres() const& noexcept { return spheres_; }

std::span<const BoxConstraint> ConstraintSet::boxes() const& noexcept { return boxes_; }

std::span<const CylinderConstraint> ConstraintSet::cylinders() const& noexcept {
  return cylinders_;
}

ConstraintSetCheckpoint ConstraintSet::checkpoint() const {
  ConstraintSetCheckpoint result{
      .next_id = next_id_,
      .planes = planes_,
      .spheres = spheres_,
      .boxes = boxes_,
      .cylinders = cylinders_,
  };
  result.validate();
  return result;
}

void ConstraintSet::validate() const {
  validate_constraint_state(next_id_, planes_, spheres_, boxes_, cylinders_);
}

void validate_constraint_contact_parameters(const ConstraintContactParameters& parameters) {
  if (!std::isfinite(parameters.activation_margin) || parameters.activation_margin < 0.0F) {
    throw std::invalid_argument("constraint activation margin must be finite and non-negative");
  }
  if (!std::isfinite(parameters.degeneracy_epsilon) || parameters.degeneracy_epsilon <= 0.0F) {
    throw std::invalid_argument("constraint degeneracy epsilon must be finite and positive");
  }
}

ExternalContactGraph::ExternalContactGraph(std::size_t cell_count,
                                           std::vector<ExternalContact> contacts)
    : cell_count_(cell_count),
      contacts_(std::move(contacts)),
      incidence_offsets_(checked_offset_count(cell_count)) {
  for (const auto& contact : contacts_) {
    validate_contact(contact, cell_count_);
    ++incidence_offsets_[static_cast<std::size_t>(contact.cell_slot) + 1];
  }
  for (std::size_t index = 1; index < incidence_offsets_.size(); ++index) {
    incidence_offsets_[index] += incidence_offsets_[index - 1];
  }

  incidence_contact_indices_.resize(contacts_.size());
  auto cursors = incidence_offsets_;
  for (std::size_t index = 0; index < contacts_.size(); ++index) {
    const auto slot = static_cast<std::size_t>(contacts_[index].cell_slot);
    incidence_contact_indices_[cursors[slot]++] = index;
  }
}

std::size_t ExternalContactGraph::cell_count() const noexcept { return cell_count_; }

std::size_t ExternalContactGraph::size() const noexcept { return contacts_.size(); }

bool ExternalContactGraph::empty() const noexcept { return contacts_.empty(); }

std::span<const ExternalContact> ExternalContactGraph::contacts() const& noexcept {
  return contacts_;
}

std::span<const std::size_t> ExternalContactGraph::incident_contact_indices(Slot slot) const& {
  const auto index = static_cast<std::size_t>(slot);
  if (index >= cell_count_) {
    throw std::out_of_range("external contact incidence slot is out of range");
  }
  const auto begin = incidence_offsets_[index];
  const auto end = incidence_offsets_[index + 1];
  return std::span<const std::size_t>(incidence_contact_indices_).subspan(begin, end - begin);
}

}  // namespace cm
