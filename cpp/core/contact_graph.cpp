#include "microsimulator/contact_graph.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "core/parallel.hpp"

namespace microsimulator {
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

  [[nodiscard]] std::array<double, 3> minimum() const {
    return {minimum_x, minimum_y, minimum_z};
  }

  [[nodiscard]] std::array<double, 3> maximum() const {
    return {maximum_x, maximum_y, maximum_z};
  }
};

// Boxes that touch count as overlapping.
[[nodiscard]] bool overlap(const CapsuleBounds& left, const CapsuleBounds& right) {
  return left.maximum_x >= right.minimum_x && right.maximum_x >= left.minimum_x &&
         left.maximum_y >= right.minimum_y && right.maximum_y >= left.minimum_y &&
         left.maximum_z >= right.minimum_z && right.maximum_z >= left.minimum_z;
}

// Cells handed to one thread at a time.
constexpr std::size_t bounds_grain = 2048;
constexpr std::size_t search_grain = 256;
constexpr std::size_t adjacency_grain = 1024;
// A grid with more boxes than this multiple of the cell count is too sparse
// to index densely.
constexpr std::size_t grid_boxes_per_cell = 8;
constexpr std::size_t grid_spare_boxes = 4096;

// The overlapping pairs of boxes, each ordered by identifier and the whole
// sorted, found through a uniform grid. Every grid box is at least as wide as
// the widest capsule box, so two overlapping capsule boxes have centers in
// the same or adjacent grid boxes. Returns nothing when the cells are spread
// too thinly for a dense grid.
[[nodiscard]] std::optional<std::vector<ContactCandidate>> grid_candidates(
    std::span<const CapsuleBounds> bounds) {
  const auto count = bounds.size();
  std::array<double, 3> low = bounds.front().minimum();
  std::array<double, 3> high = bounds.front().maximum();
  std::array<double, 3> width{};
  for (const auto& box : bounds) {
    const auto minimum = box.minimum();
    const auto maximum = box.maximum();
    for (std::size_t axis = 0; axis < 3; ++axis) {
      low[axis] = std::min(low[axis], minimum[axis]);
      high[axis] = std::max(high[axis], maximum[axis]);
      width[axis] = std::max(width[axis], maximum[axis] - minimum[axis]);
    }
  }
  std::array<std::size_t, 3> dimensions{};
  std::array<double, 3> inverse_width{};
  auto box_count = std::size_t{1};
  const auto box_limit = grid_boxes_per_cell * count + grid_spare_boxes;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    // The margin keeps rounding from placing overlapping boxes two apart.
    const auto box_width = std::max(width[axis] * 1.001, std::numeric_limits<double>::min());
    const auto extent = std::floor((high[axis] - low[axis]) / box_width) + 1.0;
    if (!(extent <= static_cast<double>(box_limit))) {
      return std::nullopt;
    }
    dimensions[axis] = static_cast<std::size_t>(extent);
    inverse_width[axis] = 1.0 / box_width;
    box_count *= dimensions[axis];
    if (box_count > box_limit) {
      return std::nullopt;
    }
  }
  const auto coordinates = [&](const CapsuleBounds& box) {
    const auto minimum = box.minimum();
    const auto maximum = box.maximum();
    std::array<std::size_t, 3> result{};
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const auto center = 0.5 * (minimum[axis] + maximum[axis]);
      const auto position = std::max(0.0, (center - low[axis]) * inverse_width[axis]);
      result[axis] = std::min(dimensions[axis] - 1, static_cast<std::size_t>(position));
    }
    return result;
  };
  const auto flat = [&](std::size_t x, std::size_t y, std::size_t z) {
    return (x * dimensions[1] + y) * dimensions[2] + z;
  };

  std::vector<std::uint32_t> box_of(count);
  detail::parallel_for(count, bounds_grain, [&](std::size_t begin, std::size_t end) {
    for (std::size_t index = begin; index < end; ++index) {
      const auto at = coordinates(bounds[index]);
      box_of[index] = static_cast<std::uint32_t>(flat(at[0], at[1], at[2]));
    }
  });
  std::vector<std::uint32_t> starts(box_count + 1, 0);
  for (const auto box : box_of) {
    ++starts[static_cast<std::size_t>(box) + 1];
  }
  for (std::size_t box = 0; box < box_count; ++box) {
    starts[box + 1] += starts[box];
  }
  std::vector<std::uint32_t> members(count);
  {
    std::vector<std::uint32_t> cursors(starts.begin(), starts.end() - 1);
    for (std::size_t index = 0; index < count; ++index) {
      members[cursors[box_of[index]]++] = static_cast<std::uint32_t>(index);
    }
  }

  // Each cell finds its partners of higher identifier, so that the cells
  // taken in identifier order yield the pairs already sorted.
  std::vector<std::uint32_t> order(count);
  for (std::size_t index = 0; index < count; ++index) {
    order[index] = static_cast<std::uint32_t>(index);
  }
  if (!std::ranges::is_sorted(bounds, {}, &CapsuleBounds::id)) {
    std::ranges::sort(order, {}, [&](std::uint32_t index) { return bounds[index].id; });
  }
  std::vector<std::vector<ContactCandidate>> pieces((count + search_grain - 1) / search_grain);
  detail::parallel_for(count, search_grain, [&](std::size_t begin, std::size_t end) {
    auto& piece = pieces[begin / search_grain];
    std::vector<std::pair<CellId, Slot>> partners;
    for (std::size_t position = begin; position < end; ++position) {
      const auto& current = bounds[order[position]];
      const auto at = coordinates(current);
      std::array<std::size_t, 3> first{};
      std::array<std::size_t, 3> last{};
      for (std::size_t axis = 0; axis < 3; ++axis) {
        first[axis] = at[axis] == 0 ? 0 : at[axis] - 1;
        last[axis] = std::min(dimensions[axis] - 1, at[axis] + 1);
      }
      partners.clear();
      for (auto x = first[0]; x <= last[0]; ++x) {
        for (auto y = first[1]; y <= last[1]; ++y) {
          for (auto z = first[2]; z <= last[2]; ++z) {
            const auto box = flat(x, y, z);
            for (auto member = starts[box]; member < starts[box + 1]; ++member) {
              const auto& other = bounds[members[member]];
              if (other.id > current.id && overlap(current, other)) {
                partners.emplace_back(other.id, other.slot);
              }
            }
          }
        }
      }
      std::ranges::sort(partners);
      for (const auto& [id, slot] : partners) {
        piece.push_back({current.slot, slot});
      }
    }
  });
  return detail::concatenate(pieces);
}

// The same pairs found by sweeping the boxes along x, for cells spread too
// thinly for a grid.
[[nodiscard]] std::vector<ContactCandidate> sweep_candidates(std::vector<CapsuleBounds> bounds,
                                                             const CellGeometryView& geometry) {
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
      if (!overlap(*candidate, current)) {
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
  std::vector<CapsuleBounds> bounds(geometry.size());
  const auto margin_per_cell = static_cast<double>(parameters.activation_margin) * 0.5;
  detail::parallel_for(bounds.size(), bounds_grain, [&](std::size_t begin, std::size_t end) {
    for (std::size_t index = begin; index < end; ++index) {
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
      bounds[index] = {
          .id = geometry.ids[index],
          .slot = static_cast<Slot>(index),
          .minimum_x = center_x - extent_x,
          .maximum_x = center_x + extent_x,
          .minimum_y = center_y - extent_y,
          .maximum_y = center_y + extent_y,
          .minimum_z = center_z - extent_z,
          .maximum_z = center_z + extent_z,
      };
    }
  });
  if (bounds.size() < 2) {
    return {};
  }
  // A box that is not finite has no place in a grid; the sweep orders it.
  const auto finite_bounds = std::ranges::all_of(bounds, [](const CapsuleBounds& box) {
    return std::isfinite(box.minimum_x) && std::isfinite(box.maximum_x) &&
           std::isfinite(box.minimum_y) && std::isfinite(box.maximum_y) &&
           std::isfinite(box.minimum_z) && std::isfinite(box.maximum_z);
  });
  if (finite_bounds) {
    if (auto candidates = grid_candidates(bounds)) {
      return std::move(*candidates);
    }
  }
  return sweep_candidates(std::move(bounds), geometry);
}

ContactGraph::ContactGraph(std::size_t cell_count, std::vector<CellContact> contacts)
    : cell_count_(cell_count),
      contacts_(std::move(contacts)),
      incidence_offsets_(checked_offset_count(cell_count)),
      neighbor_offsets_(checked_offset_count(cell_count)) {
  if (contacts_.size() > std::numeric_limits<std::size_t>::max() / 2) {
    throw std::overflow_error("contact incidence index size overflow");
  }

  detail::parallel_for(contacts_.size(), adjacency_grain, [&](std::size_t begin, std::size_t end) {
    for (std::size_t index = begin; index < end; ++index) {
      validate_contact(contacts_[index], cell_count_);
    }
  });
  for (const auto& contact : contacts_) {
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

  // Each cell's neighbors are the distinct partners of its incident contacts.
  // They are sorted in the cell's own stretch of the incidence layout and
  // then packed together.
  std::vector<CellId> partners(incidence_contact_indices_.size());
  std::vector<std::size_t> distinct(cell_count_);
  detail::parallel_for(cell_count_, adjacency_grain, [&](std::size_t begin, std::size_t end) {
    for (std::size_t slot = begin; slot < end; ++slot) {
      const auto first = incidence_offsets_[slot];
      const auto last = incidence_offsets_[slot + 1];
      for (auto entry = first; entry < last; ++entry) {
        const auto& contact = contacts_[incidence_contact_indices_[entry]];
        partners[entry] = contact.first_slot == slot ? contact.second_id : contact.first_id;
      }
      const auto ids = std::span(partners).subspan(first, last - first);
      std::ranges::sort(ids);
      distinct[slot] = ids.size() - std::ranges::unique(ids).size();
    }
  });
  for (std::size_t slot = 0; slot < cell_count_; ++slot) {
    neighbor_offsets_[slot + 1] = neighbor_offsets_[slot] + distinct[slot];
  }
  neighbor_ids_.resize(neighbor_offsets_.back());
  detail::parallel_for(cell_count_, adjacency_grain, [&](std::size_t begin, std::size_t end) {
    for (std::size_t slot = begin; slot < end; ++slot) {
      std::copy_n(partners.begin() + static_cast<std::ptrdiff_t>(incidence_offsets_[slot]),
                  distinct[slot],
                  neighbor_ids_.begin() + static_cast<std::ptrdiff_t>(neighbor_offsets_[slot]));
    }
  });
}

std::size_t ContactGraph::cell_count() const noexcept {
  return cell_count_;
}

std::size_t ContactGraph::size() const noexcept {
  return contacts_.size();
}

bool ContactGraph::empty() const noexcept {
  return contacts_.empty();
}

std::span<const CellContact> ContactGraph::contacts() const& noexcept {
  return contacts_;
}

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

}  // namespace microsimulator
