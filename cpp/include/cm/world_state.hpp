#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cm/types.hpp"

namespace cm {

struct CellInit {
  Vec3 position{};
  Vec3 direction{1.0F, 0.0F, 0.0F};
  float length{3.5F};
  float radius{0.5F};
  float growth_rate{1.0F};
  std::int32_t cell_type{0};
  bool fixed{false};
  std::vector<float> species;
};

struct CellSnapshot {
  CellId id{invalid_cell_id};
  Slot slot{invalid_slot};
  Vec3 position{};
  Vec3 direction{1.0F, 0.0F, 0.0F};
  float length{0.0F};
  float radius{0.0F};
  float growth_rate{0.0F};
  std::int32_t cell_type{0};
  bool fixed{false};
  std::vector<float> species;
};

struct LineageEntry {
  CellId child{invalid_cell_id};
  CellId parent{invalid_cell_id};
};

struct WorldStateCheckpoint {
  std::size_t species_count{0};
  CellId next_id{1};
  std::vector<CellSnapshot> cells;
  std::vector<LineageEntry> lineage;

  void validate() const;
};

struct GrowthStateView {
  std::span<float> lengths;
  std::span<const float> growth_rates;
};

struct CellGeometryView {
  std::span<const CellId> ids;
  std::span<const float> position_x;
  std::span<const float> position_y;
  std::span<const float> position_z;
  std::span<const float> direction_x;
  std::span<const float> direction_y;
  std::span<const float> direction_z;
  std::span<const float> lengths;
  std::span<const float> radii;

  [[nodiscard]] std::size_t size() const noexcept { return ids.size(); }
};

struct CellAttributeView {
  std::span<const float> growth_rates;
  std::span<const std::int32_t> cell_types;
  std::span<const std::uint8_t> fixed;
};

struct SpeciesStateView {
  std::span<float> levels;
  std::size_t cell_count{0};
  std::size_t species_count{0};
};

struct ConstSpeciesStateView {
  std::span<const float> levels;
  std::size_t cell_count{0};
  std::size_t species_count{0};
};

class WorldState {
 public:
  explicit WorldState(std::size_t reserved_capacity = 0, std::size_t species_count = 0);
  explicit WorldState(const WorldStateCheckpoint& checkpoint);

  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] bool empty() const noexcept;
  [[nodiscard]] bool contains(CellId id) const noexcept;
  [[nodiscard]] std::size_t species_count() const noexcept;

  CellId add_cell(const CellInit& cell);
  void remove_cell(CellId id);
  std::pair<CellId, CellId> divide(CellId parent_id, float first_fraction);
  std::pair<CellId, CellId> divide_equal(CellId parent_id);
  void advance_growth(float dt);
  void set_cell_geometry(Slot slot, Vec3 position, Vec3 direction, float length);
  void set_cell_geometry(CellId id, Vec3 position, Vec3 direction, float length);
  void set_cell_attributes(CellId id, float growth_rate, std::int32_t cell_type);
  void set_cell_fixed(CellId id, bool fixed);
  void set_species(CellId id, std::span<const float> levels);
  [[nodiscard]] GrowthStateView growth_state() noexcept;
  [[nodiscard]] CellGeometryView geometry_state() const noexcept;
  [[nodiscard]] CellAttributeView cell_attributes() const noexcept;
  [[nodiscard]] SpeciesStateView species_state() noexcept;
  [[nodiscard]] ConstSpeciesStateView species_state() const noexcept;

  [[nodiscard]] CellSnapshot cell(CellId id) const;
  [[nodiscard]] std::vector<CellSnapshot> cells() const;
  [[nodiscard]] std::optional<CellId> lineage_parent(CellId id) const noexcept;
  [[nodiscard]] WorldStateCheckpoint checkpoint() const;

  void validate() const;

 private:
  [[nodiscard]] CellId allocate_id();
  [[nodiscard]] Slot slot_for(CellId id) const;
  void append(CellId id, const CellInit& cell);
  void replace(Slot slot, CellId id, const CellInit& cell);

  CellId next_id_{1};
  std::size_t species_count_{0};
  std::vector<CellId> ids_;
  std::vector<float> position_x_;
  std::vector<float> position_y_;
  std::vector<float> position_z_;
  std::vector<float> direction_x_;
  std::vector<float> direction_y_;
  std::vector<float> direction_z_;
  std::vector<float> length_;
  std::vector<float> radius_;
  std::vector<float> growth_rate_;
  std::vector<std::int32_t> cell_type_;
  std::vector<std::uint8_t> fixed_;
  std::vector<float> species_;
  std::unordered_map<CellId, Slot> id_to_slot_;
  std::unordered_map<CellId, CellId> lineage_;
};

}  // namespace cm
