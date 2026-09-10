#include "cm/contact_graph.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace cm {
namespace {

struct CapsuleBounds {
  CellId id;
  Slot slot;
  double minimum_x;
  double maximum_x;
  double minimum_y;
  double maximum_y;
  double minimum_z;
  double maximum_z;
};

bool finite(const Vec3& value) {
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

std::size_t checked_offset_count(std::size_t cell_count) {
  if (cell_count > static_cast<std::size_t>(invalid_slot) ||
      cell_count == std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("contact graph exceeds the slot index space");
  }
  return cell_count + 1;
}

void validate_contact(const CellContact& contact, std::size_t cell_count) {
  if (contact.first_id == invalid_cell_id || contact.second_id == invalid_cell_id ||
      contact.first_id >= contact.second_id) {
    throw std::invalid_argument("contact cell identifiers are not canonical");
  }
  if (contact.first_slot >= cell_count || contact.second_slot >= cell_count ||
      contact.first_slot == contact.second_slot) {
    throw std::invalid_argument("contact slots are invalid");
  }
  if (contact.ordinal > 1) {
    throw std::invalid_argument("cell contact ordinal exceeds the capsule-pair contract");
  }
  if (!finite(contact.point_on_first) || !finite(contact.normal) ||
      !std::isfinite(contact.signed_separation) || !std::isfinite(contact.weight) ||
      contact.weight <= 0.0F) {
    throw std::invalid_argument("contact contains a non-finite or invalid field");
  }
  if (std::abs(norm(contact.normal) - 1.0F) > 1.0e-4F) {
    throw std::invalid_argument("contact normal is not unit length");
  }
}

}  // namespace

void validate_contact_parameters(const ContactParameters& parameters) {
  if (!std::isfinite(parameters.activation_margin) || parameters.activation_margin < 0.0F) {
    throw std::invalid_argument("contact activation margin must be finite and non-negative");
  }
  if (!std::isfinite(parameters.parallel_sine_threshold) ||
      parameters.parallel_sine_threshold < 0.0F || parameters.parallel_sine_threshold > 1.0F) {
    throw std::invalid_argument("contact parallel threshold must be between zero and one");
  }
  if (!std::isfinite(parameters.degeneracy_epsilon) || parameters.degeneracy_epsilon <= 0.0F) {
    throw std::invalid_argument("contact degeneracy epsilon must be finite and positive");
  }
}

std::vector<ContactCandidate> find_cell_contact_candidates(const WorldState& state,
                                                           const ContactParameters& parameters) {
  validate_contact_parameters(parameters);
  const auto geometry = state.geometry_state();
  std::vector<CapsuleBounds> bounds;
  bounds.reserve(geometry.size());
  const auto margin_per_cell = static_cast<double>(parameters.activation_margin) * 0.5;
  for (std::size_t index = 0; index < geometry.size(); ++index) {
    const auto half_length = static_cast<double>(geometry.lengths[index]) * 0.5;
    const auto padding = static_cast<double>(geometry.radii[index]) + margin_per_cell;
    const auto extent_x =
        std::abs(static_cast<double>(geometry.direction_x[index])) * half_length + padding;
    const auto extent_y =
        std::abs(static_cast<double>(geometry.direction_y[index])) * half_length + padding;
    const auto extent_z =
        std::abs(static_cast<double>(geometry.direction_z[index])) * half_length + padding;
    const auto center_x = static_cast<double>(geometry.position_x[index]);
    const auto center_y = static_cast<double>(geometry.position_y[index]);
    const auto center_z = static_cast<double>(geometry.position_z[index]);
    bounds.push_back({
        .id = geometry.ids[index],
        .slot = static_cast<Slot>(index),
        .minimum_x = center_x - extent_x,
        .maximum_x = center_x + extent_x,
        .minimum_y = center_y - extent_y,
        .maximum_y = center_y + extent_y,
        .minimum_z = center_z - extent_z,
        .maximum_z = center_z + extent_z,
    });
  }
  std::ranges::sort(bounds, [](const CapsuleBounds& left, const CapsuleBounds& right) {
    return std::tuple{left.minimum_x, left.id} < std::tuple{right.minimum_x, right.id};
  });

  std::vector<const CapsuleBounds*> active;
  std::vector<ContactCandidate> candidates;
  for (const auto& current : bounds) {
    const auto expired = std::ranges::remove_if(active, [&current](const CapsuleBounds* candidate) {
      return candidate->maximum_x < current.minimum_x;
    });
    active.erase(expired.begin(), expired.end());
    for (const auto* candidate : active) {
      const auto overlaps_y =
          candidate->maximum_y >= current.minimum_y && current.maximum_y >= candidate->minimum_y;
      const auto overlaps_z =
          candidate->maximum_z >= current.minimum_z && current.maximum_z >= candidate->minimum_z;
      if (!overlaps_y || !overlaps_z) {
        continue;
      }
      candidates.push_back(candidate->id < current.id
                               ? ContactCandidate{candidate->slot, current.slot}
                               : ContactCandidate{current.slot, candidate->slot});
    }
    active.push_back(&current);
  }
  std::ranges::sort(
      candidates, [&geometry](const ContactCandidate& left, const ContactCandidate& right) {
        return std::tuple{geometry.ids[left.first_slot], geometry.ids[left.second_slot]} <
               std::tuple{geometry.ids[right.first_slot], geometry.ids[right.second_slot]};
      });
  return candidates;
}

ContactGraph::ContactGraph(std::size_t cell_count, std::vector<CellContact> contacts)
    : cell_count_(cell_count),
      contacts_(std::move(contacts)),
      incidence_offsets_(checked_offset_count(cell_count)),
      neighbor_offsets_(checked_offset_count(cell_count)) {
  if (contacts_.size() > std::numeric_limits<std::size_t>::max() / 2) {
    throw std::overflow_error("contact incidence index size overflow");
  }

  for (const auto& contact : contacts_) {
    validate_contact(contact, cell_count_);
    ++incidence_offsets_[static_cast<std::size_t>(contact.first_slot) + 1];
    ++incidence_offsets_[static_cast<std::size_t>(contact.second_slot) + 1];
  }
  for (std::size_t index = 1; index < incidence_offsets_.size(); ++index) {
    incidence_offsets_[index] += incidence_offsets_[index - 1];
  }

  incidence_contact_indices_.resize(contacts_.size() * 2);
  auto cursors = incidence_offsets_;
  for (std::size_t index = 0; index < contacts_.size(); ++index) {
    const auto& contact = contacts_[index];
    incidence_contact_indices_[cursors[contact.first_slot]++] = index;
    incidence_contact_indices_[cursors[contact.second_slot]++] = index;
  }

  std::vector<std::vector<CellId>> neighbors(cell_count_);
  for (const auto& contact : contacts_) {
    neighbors[contact.first_slot].push_back(contact.second_id);
    neighbors[contact.second_slot].push_back(contact.first_id);
  }
  for (std::size_t slot = 0; slot < cell_count_; ++slot) {
    auto& ids = neighbors[slot];
    std::ranges::sort(ids);
    const auto unique_end = std::ranges::unique(ids).begin();
    ids.erase(unique_end, ids.end());
    neighbor_offsets_[slot + 1] = neighbor_offsets_[slot] + ids.size();
    neighbor_ids_.insert(neighbor_ids_.end(), ids.begin(), ids.end());
  }
}

std::size_t ContactGraph::cell_count() const noexcept { return cell_count_; }

std::size_t ContactGraph::size() const noexcept { return contacts_.size(); }

bool ContactGraph::empty() const noexcept { return contacts_.empty(); }

std::span<const CellContact> ContactGraph::contacts() const& noexcept { return contacts_; }

std::span<const std::size_t> ContactGraph::incident_contact_indices(Slot slot) const& {
  const auto index = static_cast<std::size_t>(slot);
  if (index >= cell_count_) {
    throw std::out_of_range("contact incidence slot is out of range");
  }
  const auto begin = incidence_offsets_[index];
  const auto end = incidence_offsets_[index + 1];
  return std::span<const std::size_t>(incidence_contact_indices_).subspan(begin, end - begin);
}

std::span<const CellId> ContactGraph::neighbor_ids(Slot slot) const& {
  const auto index = static_cast<std::size_t>(slot);
  if (index >= cell_count_) {
    throw std::out_of_range("contact neighbor slot is out of range");
  }
  const auto begin = neighbor_offsets_[index];
  const auto end = neighbor_offsets_[index + 1];
  return std::span<const CellId>(neighbor_ids_).subspan(begin, end - begin);
}

}  // namespace cm
