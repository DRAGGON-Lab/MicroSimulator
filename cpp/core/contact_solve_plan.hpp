#pragma once

#include <span>
#include <vector>

#include "core/device_types.hpp"
#include "microsimulator/contact_dynamics.hpp"

namespace microsimulator::detail {

// A contact system in the layout of shared/device_types.h that every
// backend's solve kernels read.
struct ContactSolvePlan {
  device::ContactSolveData data{};
  std::vector<device::ContactRowData> rows;
  std::vector<device::ContactBodyData> bodies;
  // Each cell's incidence entries run from offsets[cell] to offsets[cell + 1]
  // in row order. An entry is a row index shifted left by one, with the low
  // bit set for the row's second cell.
  std::vector<device::uint> incidence_offsets;
  std::vector<device::uint> incidence_entries;
  std::vector<float> multipliers;
};

// Validates a system and its initial multipliers and lays them out. Each
// array holds at least one element.
[[nodiscard]] ContactSolvePlan plan_contact_solve(const ContactSystem& system,
                                                  std::span<const float> initial_multipliers,
                                                  float tolerance);

}  // namespace microsimulator::detail
