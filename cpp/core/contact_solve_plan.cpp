#include "core/contact_solve_plan.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace microsimulator::detail {

ContactSolvePlan plan_contact_solve(const ContactSystem& system,
                                    std::span<const float> initial_multipliers, float tolerance) {
  validate_contact_system(system);
  if (initial_multipliers.size() != system.rows.size()) {
    throw std::invalid_argument("contact multiplier count does not match the contact rows");
  }
  for (const auto multiplier : initial_multipliers) {
    if (!std::isfinite(multiplier) || multiplier < 0.0F) {
      throw std::invalid_argument("contact multipliers must be finite and non-negative");
    }
  }
  if (!std::isfinite(tolerance) || tolerance <= 0.0F) {
    throw std::invalid_argument("contact solve tolerance must be finite and positive");
  }
  // An incidence entry packs a row index with one bit for the row's side.
  if (system.bodies.size() >= std::numeric_limits<std::uint32_t>::max() ||
      system.rows.size() > std::numeric_limits<std::uint32_t>::max() / 2) {
    throw std::overflow_error("contact system exceeds the uint32 index space");
  }

  ContactSolvePlan plan;
  plan.data = {
      .dt = system.dt,
      .tolerance = tolerance,
      .law = static_cast<std::uint32_t>(system.growth.law),
      .sensitivity = system.growth.sensitivity,
      .free_load = system.growth.free_load,
      .stall_load = system.growth.stall_load,
      .row_count = static_cast<std::uint32_t>(system.rows.size()),
      .body_count = static_cast<std::uint32_t>(system.bodies.size()),
  };
  const auto body_count = system.bodies.size();
  plan.rows.resize(std::max<std::size_t>(system.rows.size(), 1));
  plan.incidence_offsets.assign(body_count + 1, 0);
  for (std::size_t index = 0; index < system.rows.size(); ++index) {
    const auto& row = system.rows[index];
    const auto& first = row.first_jacobian;
    const auto& second = row.second_jacobian;
    plan.rows[index] = {
        .first_linear = {first[0], first[1], first[2], row.first_length_lever},
        .first_angular = {first[3], first[4], first[5], row.offset},
        .second_linear = {second[0], second[1], second[2], row.second_length_lever},
        .second_angular = {second[3], second[4], second[5], row.compliance},
        .first_slot = row.first_slot,
        .second_slot = row.second_slot,
        .padding0 = 0,
        .padding1 = 0,
    };
    ++plan.incidence_offsets[row.first_slot + 1];
    if (row.second_slot != invalid_slot) {
      ++plan.incidence_offsets[row.second_slot + 1];
    }
  }
  for (std::size_t body = 0; body < body_count; ++body) {
    plan.incidence_offsets[body + 1] += plan.incidence_offsets[body];
  }
  plan.incidence_entries.resize(std::max<std::size_t>(plan.incidence_offsets.back(), 1));
  std::vector<std::uint32_t> cursors(plan.incidence_offsets.begin(),
                                     plan.incidence_offsets.end() - 1);
  for (std::size_t index = 0; index < system.rows.size(); ++index) {
    const auto& row = system.rows[index];
    const auto entry = static_cast<std::uint32_t>(index) << 1U;
    plan.incidence_entries[cursors[row.first_slot]++] = entry;
    if (row.second_slot != invalid_slot) {
      plan.incidence_entries[cursors[row.second_slot]++] = entry | 1U;
    }
  }
  plan.bodies.resize(std::max<std::size_t>(body_count, 1));
  for (std::size_t body = 0; body < body_count; ++body) {
    const auto& source = system.bodies[body];
    plan.bodies[body] = {
        .axis_mobility = {source.axis.x, source.axis.y, source.axis.z,
                          source.translational_mobility},
        .rates = {source.transverse_mobility, source.axial_mobility, source.length_rate, 0.0F},
        .drift_linear = {source.drift[0], source.drift[1], source.drift[2], 0.0F},
        .drift_angular = {source.drift[3], source.drift[4], source.drift[5], 0.0F},
    };
  }
  plan.multipliers.assign(initial_multipliers.begin(), initial_multipliers.end());
  plan.multipliers.resize(std::max<std::size_t>(plan.multipliers.size(), 1), 0.0F);
  return plan;
}

}  // namespace microsimulator::detail
