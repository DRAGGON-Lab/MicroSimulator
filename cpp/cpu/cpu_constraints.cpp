#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "core/external_contact_plan.hpp"
#include "core/parallel.hpp"
#include "cpu/cpu_scan.hpp"
#include "cpu/kernels/external_contacts.hpp"
#include "microsimulator/constraints.hpp"

namespace microsimulator {
namespace {

// Pairs handed to one thread at a time.
constexpr std::size_t pair_grain = 256;

// The contacts of every cell with every constraint, by the kernels of
// cpu/kernels/external_contacts.hpp.
ExternalContactGraph external_contacts(const WorldState& state, const ConstraintSet& constraints,
                                       const ConstraintContactParameters& parameters,
                                       bool every_face) {
  const auto plan = detail::plan_external_contacts(state, constraints, parameters, every_face);
  const auto pairs = plan.pair_count();
  if (pairs == 0) {
    return ExternalContactGraph(plan.data.cell_count, {});
  }
  std::vector<std::uint32_t> counts(pairs);
  detail::parallel_for(pairs, pair_grain, [&](std::size_t begin, std::size_t end) {
    for (std::size_t pair = begin; pair < end; ++pair) {
      counts[pair] = device::external_pair_count(static_cast<std::uint32_t>(pair), plan.data,
                                                 plan.centers.data(), plan.axes.data(),
                                                 plan.constraints.data());
    }
  });
  const auto ends = detail::inclusive_scan(std::move(counts));
  std::vector<device::ExternalContactRecord> records(std::max<std::size_t>(ends.back(), 1));
  detail::parallel_for(pairs, pair_grain, [&](std::size_t begin, std::size_t end) {
    for (std::size_t pair = begin; pair < end; ++pair) {
      device::external_pair_fill(static_cast<std::uint32_t>(pair), plan.data, plan.centers.data(),
                                 plan.axes.data(), plan.constraints.data(), ends.data(),
                                 records.data());
    }
  });
  return detail::finish_external_contacts(plan, std::span(records.data(), ends.back()));
}

}  // namespace

ExternalContactGraph find_external_contacts_cpu(const WorldState& state,
                                                const ConstraintSet& constraints,
                                                const ConstraintContactParameters& parameters) {
  return external_contacts(state, constraints, parameters, false);
}

ExternalContactGraph find_external_contact_faces_cpu(
    const WorldState& state, const ConstraintSet& constraints,
    const ConstraintContactParameters& parameters) {
  return external_contacts(state, constraints, parameters, true);
}

}  // namespace microsimulator
