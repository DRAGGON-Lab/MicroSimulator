#include "core/cell_contact_plan.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace microsimulator::detail {

CellContactPlan plan_cell_contacts(const WorldState& state, const ContactParameters& parameters) {
  validate_contact_parameters(parameters);
  const auto geometry = state.geometry_state();
  // A bucket count of twice the cells, rounded up to a power of two, fits the
  // uint32 index space.
  if (geometry.size() > std::numeric_limits<std::uint32_t>::max() / 4) {
    throw std::overflow_error("cell contact detection exceeds the uint32 index space");
  }
  CellContactPlan plan;
  plan.data.cell_count = static_cast<std::uint32_t>(geometry.size());
  plan.data.bucket_count = std::bit_ceil(std::max<std::uint32_t>(2 * plan.data.cell_count, 2));
  plan.data.activation_margin = parameters.activation_margin;
  plan.data.parallel_sine_threshold = parameters.parallel_sine_threshold;
  plan.data.degeneracy_epsilon = parameters.degeneracy_epsilon;
  plan.centers.resize(std::max<std::size_t>(geometry.size(), 1));
  plan.axes.resize(plan.centers.size());
  plan.ids.assign(geometry.ids.begin(), geometry.ids.end());
  for (std::size_t slot = 0; slot < geometry.size(); ++slot) {
    plan.centers[slot] = {geometry.position_x[slot], geometry.position_y[slot],
                          geometry.position_z[slot], geometry.lengths[slot]};
    plan.axes[slot] = {geometry.direction_x[slot], geometry.direction_y[slot],
                       geometry.direction_z[slot], geometry.radii[slot]};
  }
  return plan;
}

std::vector<ContactCandidate> finish_cell_candidates(const CellContactPlan& plan,
                                                     std::span<const device::uint2> candidates) {
  std::vector<ContactCandidate> result;
  result.reserve(candidates.size());
  for (const auto& candidate : candidates) {
    result.push_back({static_cast<Slot>(candidate.x), static_cast<Slot>(candidate.y)});
  }
  const auto key = [&plan](const ContactCandidate& candidate) {
    return std::tuple{plan.ids[candidate.first_slot], plan.ids[candidate.second_slot]};
  };
  if (!std::ranges::is_sorted(result, {}, key)) {
    std::ranges::sort(result, {}, key);
  }
  return result;
}

ContactGraph finish_cell_contacts(const CellContactPlan& plan,
                                  std::span<const device::CellContactRecord> records) {
  std::vector<CellContact> contacts;
  contacts.reserve(records.size());
  for (const auto& record : records) {
    contacts.push_back({
        .first_id = plan.ids[record.first],
        .second_id = plan.ids[record.second],
        .first_slot = static_cast<Slot>(record.first),
        .second_slot = static_cast<Slot>(record.second),
        .ordinal = static_cast<std::uint8_t>(record.ordinal),
        .point_on_first = {record.point.x, record.point.y, record.point.z},
        .normal = {record.normal.x, record.normal.y, record.normal.z},
        .signed_separation = record.point.w,
        .weight = record.normal.w,
    });
  }
  const auto key = [](const CellContact& contact) {
    return std::tuple{contact.first_id, contact.second_id, contact.ordinal};
  };
  if (!std::ranges::is_sorted(contacts, {}, key)) {
    std::ranges::sort(contacts, {}, key);
  }
  return ContactGraph(plan.data.cell_count, std::move(contacts));
}

}  // namespace microsimulator::detail
