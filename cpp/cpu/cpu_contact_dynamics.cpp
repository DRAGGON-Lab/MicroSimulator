#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/contact_response.hpp"
#include "core/contact_solve_plan.hpp"
#include "core/contact_solver.hpp"
#include "core/parallel.hpp"
#include "cpu/kernels/contact_solve.hpp"
#include "microsimulator/contact_dynamics.hpp"

namespace microsimulator {
namespace {

// Bodies and reduction chunks handed to one thread at a time.
constexpr std::size_t body_grain = 128;
constexpr std::size_t chunk_grain = 1;

constexpr std::size_t width = device::contact_reduction_width;

// A contact system on the host, in the layout the solve kernels read.
struct HostSolve {
  detail::ContactSolvePlan plan;
  std::vector<device::ContactResponse> response;
  std::array<std::vector<float>, 2> multipliers;
  std::array<std::vector<float>, 2> gaps;
  std::vector<device::float4> terms;
  device::ContactSolveState state{};

  [[nodiscard]] std::size_t rows() const noexcept {
    return plan.data.row_count;
  }

  [[nodiscard]] std::size_t bodies() const noexcept {
    return plan.data.body_count;
  }
};

HostSolve make_solve(const ContactSystem& system, std::span<const float> multipliers,
                     float tolerance) {
  HostSolve solve;
  solve.plan = detail::plan_contact_solve(system, multipliers, tolerance);
  const auto rows = solve.plan.multipliers.size();
  solve.response.resize(std::max<std::size_t>(solve.bodies(), 1));
  solve.multipliers[0] = solve.plan.multipliers;
  solve.multipliers[1].resize(rows);
  solve.gaps[0].resize(rows);
  solve.gaps[1].resize(rows);
  return solve;
}

// The response of every body to the multipliers in pair `current`, or with
// `project` to those one projected step from them.
void respond(HostSolve& solve, std::size_t current, std::uint32_t project) {
  const auto& plan = solve.plan;
  detail::parallel_for(solve.bodies(), body_grain, [&](std::size_t begin, std::size_t end) {
    for (std::size_t body = begin; body < end; ++body) {
      solve.response[body] = device::contact_respond_body(
          static_cast<std::uint32_t>(body), plan.rows.data(), plan.bodies.data(),
          solve.multipliers[current].data(), solve.gaps[current].data(), solve.state.step_length,
          project, plan.incidence_offsets.data(), plan.incidence_entries.data(), plan.data);
    }
  });
}

// Sums the terms of every chunk of `width` rows by the reduction tree of
// cpu/kernels/contact_solve.hpp, each row's terms from `row_terms(row)`,
// then the chunk sums the same way until one remains.
template <typename RowTerms>
device::float4 sum_terms(HostSolve& solve, const RowTerms& row_terms) {
  const auto rows = solve.rows();
  auto count = (rows + width - 1) / width;
  auto& sums = solve.terms;
  sums.resize(count);
  detail::parallel_for(count, chunk_grain, [&](std::size_t begin, std::size_t end) {
    std::array<device::float4, width> lanes{};
    for (std::size_t chunk = begin; chunk < end; ++chunk) {
      const auto first = chunk * width;
      const auto members = std::min(width, rows - first);
      for (std::size_t lane = 0; lane < members; ++lane) {
        lanes[lane] = row_terms(static_cast<std::uint32_t>(first + lane));
      }
      sums[chunk] =
          device::contact_reduction_tree(lanes.data(), static_cast<std::uint32_t>(members));
    }
  });
  std::array<device::float4, width> lanes{};
  while (count > 1) {
    const auto chunks = (count + width - 1) / width;
    for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
      const auto first = chunk * width;
      const auto members = std::min(width, count - first);
      for (std::size_t lane = 0; lane < members; ++lane) {
        lanes[lane] = sums[first + lane];
      }
      sums[chunk] =
          device::contact_reduction_tree(lanes.data(), static_cast<std::uint32_t>(members));
    }
    count = chunks;
  }
  return count == 0 ? device::float4{} : sums.front();
}

// Responds to the multipliers in pair zero and evaluates their gaps.
void initialize(HostSolve& solve, float first_step_length, std::uint32_t max_iterations) {
  const auto& plan = solve.plan;
  respond(solve, 0, 0);
  const auto sums = sum_terms(solve, [&](std::uint32_t row) {
    return device::contact_row_initial(row, plan.rows.data(), solve.response.data(),
                                       solve.multipliers[0].data(), solve.gaps[0].data(),
                                       plan.data);
  });
  device::contact_record_step(solve.state, sums, plan.data, 1U, first_step_length, max_iterations);
}

// One projected step from pair `current` into the other pair.
void step(HostSolve& solve, std::size_t current, std::uint32_t max_iterations) {
  const auto& plan = solve.plan;
  const auto next = 1 - current;
  respond(solve, current, 1);
  const auto length = solve.state.step_length;
  const auto sums = sum_terms(solve, [&](std::uint32_t row) {
    return device::contact_row_step(row, plan.rows.data(), solve.response.data(),
                                    solve.multipliers[current].data(), solve.gaps[current].data(),
                                    solve.multipliers[next].data(), solve.gaps[next].data(), length,
                                    plan.data);
  });
  device::contact_record_step(solve.state, sums, plan.data, 0U, 0.0F, max_iterations);
}

[[nodiscard]] float diagonal_bound(const detail::ContactSolvePlan& plan) {
  auto bound = 0.0F;
  for (std::uint32_t row = 0; row < plan.data.row_count; ++row) {
    const auto entry =
        device::contact_row_bound(row, plan.rows.data(), plan.bodies.data(), plan.data);
    bound = bound < entry ? entry : bound;
  }
  return bound;
}

}  // namespace

ContactResponse evaluate_contact_response(const ContactSystem& system,
                                          std::span<const float> multipliers) {
  auto solve = make_solve(system, multipliers, 1.0F);
  respond(solve, 0, 0);
  return detail::unpack_contact_response(std::span(solve.response).first(solve.bodies()));
}

std::vector<float> evaluate_contact_gaps(const ContactSystem& system,
                                         std::span<const float> multipliers) {
  auto solve = make_solve(system, multipliers, 1.0F);
  initialize(solve, 1.0F, 0);
  return {solve.gaps[0].begin(), solve.gaps[0].begin() + static_cast<std::ptrdiff_t>(solve.rows())};
}

float contact_system_diagonal_bound(const ContactSystem& system) {
  const std::vector<float> zero(system.rows.size(), 0.0F);
  return diagonal_bound(detail::plan_contact_solve(system, zero, 1.0F));
}

ContactSolveResult solve_contact_system_cpu(const ContactSystem& system,
                                            std::span<const float> initial_multipliers,
                                            const ContactSolveParameters& parameters) {
  auto solve = make_solve(system, initial_multipliers, parameters.tolerance);
  ContactSolveResult result;
  if (solve.bodies() == 0) {
    return result;
  }
  if (solve.rows() == 0) {
    respond(solve, 0, 0);
    result.response =
        detail::unpack_contact_response(std::span(solve.response).first(solve.bodies()));
    return result;
  }
  const auto bound = diagonal_bound(solve.plan);
  if (!std::isfinite(bound) || bound <= 0.0F) {
    throw std::invalid_argument("contact system has no positive diagonal");
  }
  initialize(solve, 1.0F / bound, parameters.max_iterations);
  std::size_t current = 0;
  while (solve.state.done == 0U) {
    step(solve, current, parameters.max_iterations);
    current = 1 - current;
  }
  result.report = detail::device_contact_solve_report(solve.state.iterations, solve.state.residual,
                                                      parameters.tolerance);
  result.multipliers.assign(
      solve.multipliers[current].begin(),
      solve.multipliers[current].begin() + static_cast<std::ptrdiff_t>(solve.rows()));
  result.response =
      detail::unpack_contact_response(std::span(solve.response).first(solve.bodies()));
  return result;
}

}  // namespace microsimulator
