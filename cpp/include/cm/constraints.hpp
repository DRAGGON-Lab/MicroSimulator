#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "cm/types.hpp"
#include "cm/world_state.hpp"

namespace cm {

using ConstraintId = std::uint64_t;
inline constexpr ConstraintId invalid_constraint_id = 0;

enum class ConstraintRegion : std::uint8_t {
  outside,
  inside,
};

using SphereRegion = ConstraintRegion;

enum class ExternalConstraintKind : std::uint8_t {
  plane,
  sphere,
  box,
  cylinder,
};

enum class RodEndpoint : std::uint8_t {
  negative,
  positive,
};

struct PlaneConstraintInit {
  Vec3 point{};
  Vec3 inward_normal{0.0F, 1.0F, 0.0F};
  float coefficient{1.0F};
};

struct SphereConstraintInit {
  Vec3 center{};
  float radius{1.0F};
  float coefficient{1.0F};
  SphereRegion allowed_region{SphereRegion::outside};
};

struct BoxConstraintInit {
  Vec3 center{};
  Vec3 half_extents{1.0F, 1.0F, 1.0F};
  float coefficient{1.0F};
  ConstraintRegion allowed_region{ConstraintRegion::outside};
};

struct CylinderConstraintInit {
  Vec3 center{};
  float radius{1.0F};
  float half_height{1.0F};
  float coefficient{1.0F};
  ConstraintRegion allowed_region{ConstraintRegion::outside};
};

struct PlaneConstraint {
  ConstraintId id{invalid_constraint_id};
  Vec3 point{};
  Vec3 inward_normal{0.0F, 1.0F, 0.0F};
  float coefficient{1.0F};
};

struct SphereConstraint {
  ConstraintId id{invalid_constraint_id};
  Vec3 center{};
  float radius{1.0F};
  float coefficient{1.0F};
  SphereRegion allowed_region{SphereRegion::outside};
};

struct BoxConstraint {
  ConstraintId id{invalid_constraint_id};
  Vec3 center{};
  Vec3 half_extents{1.0F, 1.0F, 1.0F};
  float coefficient{1.0F};
  ConstraintRegion allowed_region{ConstraintRegion::outside};
};

struct CylinderConstraint {
  ConstraintId id{invalid_constraint_id};
  Vec3 center{};
  float radius{1.0F};
  float half_height{1.0F};
  float coefficient{1.0F};
  ConstraintRegion allowed_region{ConstraintRegion::outside};
};

struct ConstraintSetCheckpoint {
  ConstraintId next_id{1};
  std::vector<PlaneConstraint> planes;
  std::vector<SphereConstraint> spheres;
  std::vector<BoxConstraint> boxes;
  std::vector<CylinderConstraint> cylinders;

  void validate() const;
};

class ConstraintSet {
 public:
  ConstraintSet() = default;
  explicit ConstraintSet(const ConstraintSetCheckpoint& checkpoint);

  ConstraintId add_plane(const PlaneConstraintInit& plane);
  ConstraintId add_sphere(const SphereConstraintInit& sphere);
  ConstraintId add_box(const BoxConstraintInit& box);
  ConstraintId add_cylinder(const CylinderConstraintInit& cylinder);

  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] bool empty() const noexcept;
  [[nodiscard]] std::span<const PlaneConstraint> planes() const& noexcept;
  [[nodiscard]] std::span<const PlaneConstraint> planes() && = delete;
  [[nodiscard]] std::span<const SphereConstraint> spheres() const& noexcept;
  [[nodiscard]] std::span<const SphereConstraint> spheres() && = delete;
  [[nodiscard]] std::span<const BoxConstraint> boxes() const& noexcept;
  [[nodiscard]] std::span<const BoxConstraint> boxes() && = delete;
  [[nodiscard]] std::span<const CylinderConstraint> cylinders() const& noexcept;
  [[nodiscard]] std::span<const CylinderConstraint> cylinders() && = delete;
  [[nodiscard]] ConstraintSetCheckpoint checkpoint() const;
  void validate() const;

 private:
  [[nodiscard]] ConstraintId allocate_id();

  ConstraintId next_id_{1};
  std::vector<PlaneConstraint> planes_;
  std::vector<SphereConstraint> spheres_;
  std::vector<BoxConstraint> boxes_;
  std::vector<CylinderConstraint> cylinders_;
};

struct ConstraintContactParameters {
  float activation_margin{0.0F};
  float degeneracy_epsilon{1.0e-6F};
};

struct ExternalContact {
  CellId cell_id{invalid_cell_id};
  Slot cell_slot{invalid_slot};
  ConstraintId constraint_id{invalid_constraint_id};
  ExternalConstraintKind constraint_kind{ExternalConstraintKind::plane};
  RodEndpoint endpoint{RodEndpoint::negative};
  Vec3 point_on_cell{};
  Vec3 normal{};
  float signed_separation{0.0F};
  float weight{1.0F};
};

void validate_constraint_contact_parameters(const ConstraintContactParameters& parameters);

class ExternalContactGraph {
 public:
  ExternalContactGraph() = default;
  ExternalContactGraph(std::size_t cell_count, std::vector<ExternalContact> contacts);

  [[nodiscard]] std::size_t cell_count() const noexcept;
  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] bool empty() const noexcept;
  [[nodiscard]] std::span<const ExternalContact> contacts() const& noexcept;
  [[nodiscard]] std::span<const ExternalContact> contacts() && = delete;
  [[nodiscard]] std::span<const std::size_t> incident_contact_indices(Slot slot) const&;
  [[nodiscard]] std::span<const std::size_t> incident_contact_indices(Slot slot) && = delete;

 private:
  std::size_t cell_count_{0};
  std::vector<ExternalContact> contacts_;
  std::vector<std::size_t> incidence_offsets_{0};
  std::vector<std::size_t> incidence_contact_indices_;
};

[[nodiscard]] ExternalContactGraph find_external_contacts_cpu(
    const WorldState& state, const ConstraintSet& constraints,
    const ConstraintContactParameters& parameters = ConstraintContactParameters{});

}  // namespace cm
