#include "cm/world_state.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace cm {
namespace {

void validate_cell(const CellInit& cell, std::size_t species_count) {
  const std::array values{
      cell.position.x,  cell.position.y, cell.position.z, cell.direction.x, cell.direction.y,
      cell.direction.z, cell.length,     cell.radius,     cell.growth_rate,
  };
  if (!std::ranges::all_of(values, [](float value) { return std::isfinite(value); })) {
    throw std::invalid_argument("cell fields must be finite");
  }
  if (cell.length < 0.0F) {
    throw std::invalid_argument("cell length must be non-negative");
  }
  if (cell.radius <= 0.0F) {
    throw std::invalid_argument("cell radius must be positive");
  }
  static_cast<void>(normalized(cell.direction));
  if (!cell.species.empty() && cell.species.size() != species_count) {
    throw std::invalid_argument("cell species count does not match the world state");
  }
  if (!std::ranges::all_of(cell.species, [](float value) { return std::isfinite(value); })) {
    throw std::invalid_argument("cell species levels must be finite");
  }
}

}  // namespace

WorldState::WorldState(std::size_t reserved_capacity, std::size_t species_count)
    : species_count_(species_count) {
  if (species_count != 0 &&
      reserved_capacity > std::numeric_limits<std::size_t>::max() / species_count) {
    throw std::overflow_error("reserved species storage size overflow");
  }
  ids_.reserve(reserved_capacity);
  position_x_.reserve(reserved_capacity);
  position_y_.reserve(reserved_capacity);
  position_z_.reserve(reserved_capacity);
  direction_x_.reserve(reserved_capacity);
  direction_y_.reserve(reserved_capacity);
  direction_z_.reserve(reserved_capacity);
  length_.reserve(reserved_capacity);
  radius_.reserve(reserved_capacity);
  growth_rate_.reserve(reserved_capacity);
  cell_type_.reserve(reserved_capacity);
  fixed_.reserve(reserved_capacity);
  species_.reserve(reserved_capacity * species_count);
  id_to_slot_.reserve(reserved_capacity);
  lineage_.reserve(reserved_capacity);
}

void WorldStateCheckpoint::validate() const {
  if (next_id == invalid_cell_id) {
    throw std::invalid_argument("checkpoint next cell identifier is invalid");
  }
  if (cells.size() > static_cast<std::size_t>(invalid_slot)) {
    throw std::overflow_error("checkpoint exceeds the cell slot space");
  }
  if (species_count != 0 &&
      cells.size() > std::numeric_limits<std::size_t>::max() / species_count) {
    throw std::overflow_error("checkpoint species storage size overflow");
  }

  std::unordered_set<CellId> active_ids;
  active_ids.reserve(cells.size());
  for (std::size_t index = 0; index < cells.size(); ++index) {
    const auto& cell = cells[index];
    if (cell.slot != static_cast<Slot>(index)) {
      throw std::invalid_argument("checkpoint cell slots are not compact and ordered");
    }
    if (cell.id == invalid_cell_id || cell.id >= next_id) {
      throw std::invalid_argument("checkpoint cell identifier is outside the allocated range");
    }
    if (!active_ids.insert(cell.id).second) {
      throw std::invalid_argument("checkpoint contains a duplicate active cell identifier");
    }
    if (cell.species.size() != species_count) {
      throw std::invalid_argument("checkpoint cell species count does not match the world");
    }
    validate_cell(
        {
            .position = cell.position,
            .direction = cell.direction,
            .length = cell.length,
            .radius = cell.radius,
            .growth_rate = cell.growth_rate,
            .cell_type = cell.cell_type,
            .fixed = cell.fixed,
            .species = cell.species,
        },
        species_count);
    if (std::abs(norm(cell.direction) - 1.0F) > 1.0e-5F) {
      throw std::invalid_argument("checkpoint cell direction is not normalized");
    }
  }

  std::unordered_set<CellId> lineage_children;
  lineage_children.reserve(lineage.size());
  for (const auto& entry : lineage) {
    if (entry.child == invalid_cell_id || entry.parent == invalid_cell_id ||
        entry.parent >= entry.child || entry.child >= next_id) {
      throw std::invalid_argument("checkpoint lineage violates monotonic cell identity");
    }
    if (!lineage_children.insert(entry.child).second) {
      throw std::invalid_argument("checkpoint contains a duplicate lineage child");
    }
  }
}

WorldState::WorldState(const WorldStateCheckpoint& checkpoint)
    : WorldState(checkpoint.cells.size(), checkpoint.species_count) {
  checkpoint.validate();
  next_id_ = checkpoint.next_id;
  for (const auto& cell : checkpoint.cells) {
    ids_.push_back(cell.id);
    position_x_.push_back(cell.position.x);
    position_y_.push_back(cell.position.y);
    position_z_.push_back(cell.position.z);
    direction_x_.push_back(cell.direction.x);
    direction_y_.push_back(cell.direction.y);
    direction_z_.push_back(cell.direction.z);
    length_.push_back(cell.length);
    radius_.push_back(cell.radius);
    growth_rate_.push_back(cell.growth_rate);
    cell_type_.push_back(cell.cell_type);
    fixed_.push_back(static_cast<std::uint8_t>(cell.fixed));
    species_.insert(species_.end(), cell.species.begin(), cell.species.end());
    id_to_slot_.emplace(cell.id, cell.slot);
  }
  for (const auto& entry : checkpoint.lineage) {
    lineage_.emplace(entry.child, entry.parent);
  }
  validate();
}

std::size_t WorldState::size() const noexcept { return ids_.size(); }

bool WorldState::empty() const noexcept { return ids_.empty(); }

bool WorldState::contains(CellId id) const noexcept { return id_to_slot_.contains(id); }

std::size_t WorldState::species_count() const noexcept { return species_count_; }

CellId WorldState::allocate_id() {
  if (next_id_ == invalid_cell_id || next_id_ == std::numeric_limits<CellId>::max()) {
    throw std::overflow_error("cell identifier space exhausted");
  }
  return next_id_++;
}

Slot WorldState::slot_for(CellId id) const {
  const auto found = id_to_slot_.find(id);
  if (found == id_to_slot_.end()) {
    throw std::out_of_range("unknown cell id " + std::to_string(id));
  }
  return found->second;
}

void WorldState::append(CellId id, const CellInit& cell) {
  validate_cell(cell, species_count_);
  if (ids_.size() >= static_cast<std::size_t>(invalid_slot)) {
    throw std::overflow_error("cell slot space exhausted");
  }
  const auto direction = normalized(cell.direction);
  const auto slot = static_cast<Slot>(ids_.size());
  ids_.push_back(id);
  position_x_.push_back(cell.position.x);
  position_y_.push_back(cell.position.y);
  position_z_.push_back(cell.position.z);
  direction_x_.push_back(direction.x);
  direction_y_.push_back(direction.y);
  direction_z_.push_back(direction.z);
  length_.push_back(cell.length);
  radius_.push_back(cell.radius);
  growth_rate_.push_back(cell.growth_rate);
  cell_type_.push_back(cell.cell_type);
  fixed_.push_back(static_cast<std::uint8_t>(cell.fixed));
  if (cell.species.empty()) {
    species_.insert(species_.end(), species_count_, 0.0F);
  } else {
    species_.insert(species_.end(), cell.species.begin(), cell.species.end());
  }
  id_to_slot_.emplace(id, slot);
}

void WorldState::replace(Slot slot, CellId id, const CellInit& cell) {
  validate_cell(cell, species_count_);
  const auto index = static_cast<std::size_t>(slot);
  if (index >= size()) {
    throw std::out_of_range("cell slot is out of range");
  }
  const auto direction = normalized(cell.direction);
  ids_[index] = id;
  position_x_[index] = cell.position.x;
  position_y_[index] = cell.position.y;
  position_z_[index] = cell.position.z;
  direction_x_[index] = direction.x;
  direction_y_[index] = direction.y;
  direction_z_[index] = direction.z;
  length_[index] = cell.length;
  radius_[index] = cell.radius;
  growth_rate_[index] = cell.growth_rate;
  cell_type_[index] = cell.cell_type;
  fixed_[index] = static_cast<std::uint8_t>(cell.fixed);
  const auto species_begin = species_.begin() + static_cast<std::ptrdiff_t>(index * species_count_);
  if (cell.species.empty()) {
    std::fill_n(species_begin, species_count_, 0.0F);
  } else {
    std::copy(cell.species.begin(), cell.species.end(), species_begin);
  }
  id_to_slot_[id] = slot;
}

CellId WorldState::add_cell(const CellInit& cell) {
  const auto id = allocate_id();
  append(id, cell);
  return id;
}

std::pair<CellId, CellId> WorldState::divide(CellId parent_id, float first_fraction) {
  if (!std::isfinite(first_fraction) || first_fraction <= 0.0F || first_fraction >= 1.0F) {
    throw std::invalid_argument("first daughter fraction must be finite and between zero and one");
  }
  const auto parent = cell(parent_id);
  const auto available_length = parent.length - (2.0F * parent.radius);
  if (!(available_length >= 0.0F)) {
    throw std::domain_error("parent is too short to divide into valid daughters");
  }

  const auto first_length = available_length * first_fraction;
  const auto second_length = available_length - first_length;
  const auto first_offset = parent.direction * ((parent.length - first_length) * 0.5F);
  const auto second_offset = parent.direction * ((parent.length - second_length) * 0.5F);
  CellInit first_daughter{
      .position = parent.position - first_offset,
      .direction = parent.direction,
      .length = first_length,
      .radius = parent.radius,
      .growth_rate = parent.growth_rate,
      .cell_type = parent.cell_type,
      .fixed = parent.fixed,
      .species = parent.species,
  };

  const auto first_id = allocate_id();
  const auto second_id = allocate_id();
  const auto parent_slot = parent.slot;

  id_to_slot_.erase(parent_id);
  replace(parent_slot, first_id, first_daughter);
  auto second_daughter = first_daughter;
  second_daughter.position = parent.position + second_offset;
  second_daughter.length = second_length;
  append(second_id, second_daughter);
  lineage_[first_id] = parent_id;
  lineage_[second_id] = parent_id;
  return {first_id, second_id};
}

std::pair<CellId, CellId> WorldState::divide_equal(CellId parent_id) {
  return divide(parent_id, 0.5F);
}

void WorldState::remove_cell(CellId id) {
  const auto slot = slot_for(id);
  const auto last = ids_.size() - 1;
  if (slot != last) {
    ids_[slot] = ids_[last];
    position_x_[slot] = position_x_[last];
    position_y_[slot] = position_y_[last];
    position_z_[slot] = position_z_[last];
    direction_x_[slot] = direction_x_[last];
    direction_y_[slot] = direction_y_[last];
    direction_z_[slot] = direction_z_[last];
    length_[slot] = length_[last];
    radius_[slot] = radius_[last];
    growth_rate_[slot] = growth_rate_[last];
    cell_type_[slot] = cell_type_[last];
    fixed_[slot] = fixed_[last];
    for (std::size_t index = 0; index < species_count_; ++index) {
      species_[(slot * species_count_) + index] = species_[(last * species_count_) + index];
    }
    id_to_slot_[ids_[slot]] = static_cast<Slot>(slot);
  }
  ids_.pop_back();
  position_x_.pop_back();
  position_y_.pop_back();
  position_z_.pop_back();
  direction_x_.pop_back();
  direction_y_.pop_back();
  direction_z_.pop_back();
  length_.pop_back();
  radius_.pop_back();
  growth_rate_.pop_back();
  cell_type_.pop_back();
  fixed_.pop_back();
  species_.resize(ids_.size() * species_count_);
  id_to_slot_.erase(id);
}

void WorldState::advance_growth(float dt) {
  if (!std::isfinite(dt) || dt < 0.0F) {
    throw std::invalid_argument("time step must be finite and non-negative");
  }
  for (std::size_t index = 0; index < size(); ++index) {
    length_[index] += growth_rate_[index] * length_[index] * dt;
  }
}

void WorldState::set_cell_geometry(Slot slot, Vec3 position, Vec3 direction, float length) {
  const auto index = static_cast<std::size_t>(slot);
  if (index >= size()) {
    throw std::out_of_range("cell geometry slot is out of range");
  }
  const CellInit candidate{
      .position = position,
      .direction = direction,
      .length = length,
      .radius = radius_[index],
      .growth_rate = growth_rate_[index],
      .cell_type = cell_type_[index],
      .fixed = fixed_[index] != 0,
      .species = {},
  };
  validate_cell(candidate, species_count_);
  const auto unit_direction = normalized(direction);
  position_x_[index] = position.x;
  position_y_[index] = position.y;
  position_z_[index] = position.z;
  direction_x_[index] = unit_direction.x;
  direction_y_[index] = unit_direction.y;
  direction_z_[index] = unit_direction.z;
  length_[index] = length;
}

void WorldState::set_cell_geometry(CellId id, Vec3 position, Vec3 direction, float length) {
  set_cell_geometry(slot_for(id), position, direction, length);
}

void WorldState::set_cell_attributes(CellId id, float growth_rate, std::int32_t cell_type) {
  if (!std::isfinite(growth_rate)) {
    throw std::invalid_argument("cell growth rate must be finite");
  }
  const auto index = static_cast<std::size_t>(slot_for(id));
  growth_rate_[index] = growth_rate;
  cell_type_[index] = cell_type;
}

void WorldState::set_cell_fixed(CellId id, bool fixed) {
  fixed_[static_cast<std::size_t>(slot_for(id))] = static_cast<std::uint8_t>(fixed);
}

void WorldState::set_species(CellId id, std::span<const float> levels) {
  if (levels.size() != species_count_) {
    throw std::invalid_argument("cell species count does not match the world state");
  }
  if (!std::ranges::all_of(levels, [](float value) { return std::isfinite(value); })) {
    throw std::invalid_argument("cell species levels must be finite");
  }
  const auto offset = static_cast<std::size_t>(slot_for(id)) * species_count_;
  std::copy(levels.begin(), levels.end(), species_.begin() + static_cast<std::ptrdiff_t>(offset));
}

GrowthStateView WorldState::growth_state() noexcept {
  return {
      .lengths = length_,
      .growth_rates = growth_rate_,
  };
}

CellGeometryView WorldState::geometry_state() const noexcept {
  return {
      .ids = ids_,
      .position_x = position_x_,
      .position_y = position_y_,
      .position_z = position_z_,
      .direction_x = direction_x_,
      .direction_y = direction_y_,
      .direction_z = direction_z_,
      .lengths = length_,
      .radii = radius_,
  };
}

CellAttributeView WorldState::cell_attributes() const noexcept {
  return {
      .growth_rates = growth_rate_,
      .cell_types = cell_type_,
      .fixed = fixed_,
  };
}

SpeciesStateView WorldState::species_state() noexcept {
  return {
      .levels = species_,
      .cell_count = size(),
      .species_count = species_count_,
  };
}

ConstSpeciesStateView WorldState::species_state() const noexcept {
  return {
      .levels = species_,
      .cell_count = size(),
      .species_count = species_count_,
  };
}

CellSnapshot WorldState::cell(CellId id) const {
  const auto slot = slot_for(id);
  const auto index = static_cast<std::size_t>(slot);
  const auto species_offset = index * species_count_;
  return {
      .id = ids_[index],
      .slot = slot,
      .position = {position_x_[index], position_y_[index], position_z_[index]},
      .direction = {direction_x_[index], direction_y_[index], direction_z_[index]},
      .length = length_[index],
      .radius = radius_[index],
      .growth_rate = growth_rate_[index],
      .cell_type = cell_type_[index],
      .fixed = fixed_[index] != 0,
      .species = std::vector<float>(
          species_.begin() + static_cast<std::ptrdiff_t>(species_offset),
          species_.begin() + static_cast<std::ptrdiff_t>(species_offset + species_count_)),
  };
}

std::vector<CellSnapshot> WorldState::cells() const {
  std::vector<CellSnapshot> result;
  result.reserve(size());
  for (const auto id : ids_) {
    result.push_back(cell(id));
  }
  return result;
}

std::optional<CellId> WorldState::lineage_parent(CellId id) const noexcept {
  const auto found = lineage_.find(id);
  if (found == lineage_.end()) {
    return std::nullopt;
  }
  return found->second;
}

WorldStateCheckpoint WorldState::checkpoint() const {
  validate();
  WorldStateCheckpoint result{
      .species_count = species_count_,
      .next_id = next_id_,
      .cells = cells(),
      .lineage = {},
  };
  result.lineage.reserve(lineage_.size());
  for (const auto& [child, parent] : lineage_) {
    result.lineage.push_back({.child = child, .parent = parent});
  }
  std::ranges::sort(result.lineage, {}, &LineageEntry::child);
  result.validate();
  return result;
}

void WorldState::validate() const {
  const auto expected = ids_.size();
  const std::array sizes{
      position_x_.size(),  position_y_.size(),  position_z_.size(), direction_x_.size(),
      direction_y_.size(), direction_z_.size(), length_.size(),     radius_.size(),
      growth_rate_.size(), cell_type_.size(),   fixed_.size(),
  };
  if (!std::ranges::all_of(sizes, [expected](std::size_t size) { return size == expected; })) {
    throw std::logic_error("world state arrays have inconsistent lengths");
  }
  if (id_to_slot_.size() != expected) {
    throw std::logic_error("cell id index has the wrong size");
  }
  if (species_count_ != 0 && expected > std::numeric_limits<std::size_t>::max() / species_count_) {
    throw std::logic_error("world species storage size overflow");
  }
  if (species_.size() != expected * species_count_) {
    throw std::logic_error("world species storage has the wrong size");
  }
  if (!std::ranges::all_of(species_, [](float value) { return std::isfinite(value); })) {
    throw std::logic_error("world species levels must be finite");
  }
  if (next_id_ == invalid_cell_id) {
    throw std::logic_error("world next cell identifier is invalid");
  }
  for (std::size_t index = 0; index < expected; ++index) {
    const auto id = ids_[index];
    if (id == invalid_cell_id) {
      throw std::logic_error("active cell has an invalid identifier");
    }
    if (id >= next_id_) {
      throw std::logic_error("active cell identifier is outside the allocated range");
    }
    const auto found = id_to_slot_.find(id);
    if (found == id_to_slot_.end() || found->second != static_cast<Slot>(index)) {
      throw std::logic_error("cell id and slot index disagree");
    }
    const CellInit value{
        .position = {position_x_[index], position_y_[index], position_z_[index]},
        .direction = {direction_x_[index], direction_y_[index], direction_z_[index]},
        .length = length_[index],
        .radius = radius_[index],
        .growth_rate = growth_rate_[index],
        .cell_type = cell_type_[index],
        .fixed = fixed_[index] != 0,
        .species = {},
    };
    validate_cell(value, species_count_);
    if (std::abs(norm(value.direction) - 1.0F) > 1.0e-5F) {
      throw std::logic_error("cell direction is not normalized");
    }
    if (fixed_[index] > 1) {
      throw std::logic_error("cell fixed flag is invalid");
    }
  }
  for (const auto& [child, parent] : lineage_) {
    if (child == invalid_cell_id || parent == invalid_cell_id || parent >= child ||
        child >= next_id_) {
      throw std::logic_error("world lineage violates monotonic cell identity");
    }
  }
}

}  // namespace cm
