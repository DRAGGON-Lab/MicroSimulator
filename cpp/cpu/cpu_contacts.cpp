#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/cell_contact_plan.hpp"
#include "core/parallel.hpp"
#include "cpu/cpu_scan.hpp"
#include "cpu/kernels/cell_contacts.hpp"
#include "microsimulator/contact_graph.hpp"

namespace microsimulator {
namespace {

// Cells and candidate pairs handed to one thread at a time.
constexpr std::size_t cell_grain = 512;
constexpr std::size_t pair_grain = 512;

struct Detection {
  std::vector<device::uint2> candidates;
  std::vector<device::CellContactRecord> records;
};

// The broad phase of cpu/kernels/cell_contacts.hpp.
std::vector<device::uint2> broad_phase(detail::CellContactPlan& plan) {
  const auto cells = static_cast<std::size_t>(plan.data.cell_count);
  if (cells < 2) {
    return {};
  }
  const auto* centers = plan.centers.data();
  const auto* axes = plan.axes.data();
  std::vector<device::float4> lower(cells);
  std::vector<device::float4> upper(cells);
  detail::parallel_for(cells, cell_grain, [&](std::size_t begin, std::size_t end) {
    for (std::size_t cell = begin; cell < end; ++cell) {
      device::cell_bounds(static_cast<std::uint32_t>(cell), plan.data, centers, axes, lower.data(),
                          upper.data());
    }
  });
  // Pairwise levels of the largest-width reduction.
  auto count = cells;
  std::vector<device::float4> widths((count + 1) / 2);
  detail::parallel_for(widths.size(), cell_grain, [&](std::size_t begin, std::size_t end) {
    for (std::size_t index = begin; index < end; ++index) {
      device::cell_width_leaf(static_cast<std::uint32_t>(index), static_cast<std::uint32_t>(count),
                              lower.data(), upper.data(), widths.data());
    }
  });
  count = widths.size();
  std::vector<device::float4> next(widths.size());
  while (count > 1) {
    const auto half = (count + 1) / 2;
    for (std::size_t index = 0; index < half; ++index) {
      device::cell_width_pair(static_cast<std::uint32_t>(index), static_cast<std::uint32_t>(count),
                              widths.data(), next.data());
    }
    std::swap(widths, next);
    count = half;
  }

  const auto buckets = static_cast<std::size_t>(plan.data.bucket_count);
  std::vector<device::uint4> coordinates(cells);
  std::vector<std::uint32_t> bucket_counts(buckets, 0);
  detail::parallel_for(cells, cell_grain, [&](std::size_t begin, std::size_t end) {
    for (std::size_t cell = begin; cell < end; ++cell) {
      device::cell_grid_place(static_cast<std::uint32_t>(cell), plan.data, lower.data(),
                              upper.data(), widths.data(), coordinates.data(),
                              bucket_counts.data());
    }
  });
  const auto bucket_ends = detail::inclusive_scan(std::move(bucket_counts));
  std::vector<std::uint32_t> cursors(buckets, 0);
  std::vector<std::uint32_t> members(cells);
  detail::parallel_for(cells, cell_grain, [&](std::size_t begin, std::size_t end) {
    for (std::size_t cell = begin; cell < end; ++cell) {
      device::cell_grid_fill(static_cast<std::uint32_t>(cell), coordinates.data(),
                             bucket_ends.data(), cursors.data(), members.data());
    }
  });

  const auto* ids = plan.ids.data();
  std::vector<std::uint32_t> counts(cells);
  std::vector<device::uint2> candidates(1);
  detail::parallel_for(cells, cell_grain, [&](std::size_t begin, std::size_t end) {
    for (std::size_t cell = begin; cell < end; ++cell) {
      counts[cell] = device::cell_candidate_count(
          static_cast<std::uint32_t>(cell), plan.data, ids, lower.data(), upper.data(),
          coordinates.data(), bucket_ends.data(), members.data(), candidates.data());
    }
  });
  const auto ends = detail::inclusive_scan(std::move(counts));
  plan.data.candidate_capacity = ends.back();
  candidates.resize(std::max<std::size_t>(ends.back(), 1));
  detail::parallel_for(cells, cell_grain, [&](std::size_t begin, std::size_t end) {
    for (std::size_t cell = begin; cell < end; ++cell) {
      device::cell_candidate_fill(static_cast<std::uint32_t>(cell), plan.data, ids, lower.data(),
                                  upper.data(), coordinates.data(), bucket_ends.data(),
                                  members.data(), ends.data(), candidates.data());
    }
  });
  candidates.resize(ends.back());
  return candidates;
}

// The narrow phase of cpu/kernels/cell_contacts.hpp over the candidates.
std::vector<device::CellContactRecord> narrow_phase(detail::CellContactPlan& plan,
                                                    const std::vector<device::uint2>& candidates) {
  if (candidates.empty()) {
    return {};
  }
  const auto total = static_cast<std::uint32_t>(candidates.size());
  std::vector<std::uint32_t> counts(candidates.size());
  detail::parallel_for(candidates.size(), pair_grain, [&](std::size_t begin, std::size_t end) {
    for (std::size_t index = begin; index < end; ++index) {
      counts[index] =
          device::cell_pair_count(static_cast<std::uint32_t>(index), total, plan.data,
                                  plan.centers.data(), plan.axes.data(), candidates.data());
    }
  });
  const auto ends = detail::inclusive_scan(std::move(counts));
  plan.data.contact_capacity = ends.back();
  std::vector<device::CellContactRecord> records(std::max<std::size_t>(ends.back(), 1));
  detail::parallel_for(candidates.size(), pair_grain, [&](std::size_t begin, std::size_t end) {
    for (std::size_t index = begin; index < end; ++index) {
      device::cell_pair_fill(static_cast<std::uint32_t>(index), total, plan.data,
                             plan.centers.data(), plan.axes.data(), candidates.data(), ends.data(),
                             records.data());
    }
  });
  records.resize(ends.back());
  return records;
}

}  // namespace

std::vector<ContactCandidate> find_cell_contact_candidates(const WorldState& state,
                                                           const ContactParameters& parameters) {
  auto plan = detail::plan_cell_contacts(state, parameters);
  const auto candidates = broad_phase(plan);
  return detail::finish_cell_candidates(plan, candidates);
}

ContactGraph find_cell_contacts_cpu(const WorldState& state, const ContactParameters& parameters) {
  auto plan = detail::plan_cell_contacts(state, parameters);
  const auto candidates = broad_phase(plan);
  const auto records = narrow_phase(plan, candidates);
  return detail::finish_cell_contacts(plan, records);
}

ContactGraph find_cell_contacts_cpu_exhaustive(const WorldState& state,
                                               const ContactParameters& parameters) {
  auto plan = detail::plan_cell_contacts(state, parameters);
  const auto cells = static_cast<std::size_t>(plan.data.cell_count);
  if (cells > 1 && cells - 1 > std::numeric_limits<std::uint32_t>::max() / cells) {
    throw std::overflow_error("exhaustive contact candidate count overflow");
  }
  std::vector<device::uint2> candidates;
  candidates.reserve(cells < 2 ? 0 : cells * (cells - 1) / 2);
  for (std::size_t first = 0; first < cells; ++first) {
    for (std::size_t second = first + 1; second < cells; ++second) {
      const auto low = plan.ids[first] < plan.ids[second] ? first : second;
      const auto high = low == first ? second : first;
      candidates.push_back({static_cast<std::uint32_t>(low), static_cast<std::uint32_t>(high)});
    }
  }
  const auto records = narrow_phase(plan, candidates);
  return detail::finish_cell_contacts(plan, records);
}

}  // namespace microsimulator
