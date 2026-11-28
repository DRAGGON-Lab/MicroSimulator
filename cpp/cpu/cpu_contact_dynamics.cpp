#include <algorithm>
#include <array>
#include <bit>
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
#include "cpu/cpu_contact_solve.hpp"
#include "cpu/kernels/contact_step.hpp"
#include "microsimulator/contact_dynamics.hpp"

namespace microsimulator {
namespace {

// Bodies and reduction chunks handed to one thread at a time.
constexpr std::size_t body_grain = 128;
constexpr std::size_t chunk_grain = 1;
// Rows each member of a solve's team takes at least. A team pays a few
// barriers per iteration, so a member needs enough rows to cover them.
constexpr std::size_t team_rows = 512;

constexpr std::size_t width = device::contact_reduction_width;

// A contact system in the layout the solve kernels read, with its
// multipliers and gaps in two pairs that steps alternate between.
struct HostSolve {
  device::ContactSolveData data{};
  const device::ContactRowData* rows{nullptr};
  const device::ContactBodyData* bodies{nullptr};
  const std::uint32_t* incidence_offsets{nullptr};
  const std::uint32_t* incidence_entries{nullptr};
  device::ContactResponse* response{nullptr};
  std::array<float*, 2> multipliers{};
  std::array<float*, 2> gaps{};
  std::vector<device::float4> terms;
  device::ContactSolveState state{};

  [[nodiscard]] std::size_t row_count() const noexcept {
    return data.row_count;
  }

  [[nodiscard]] std::size_t body_count() const noexcept {
    return data.body_count;
  }
};

// A system laid out by plan_contact_solve, with the storage its solve needs.
struct OwnedSolve {
  detail::ContactSolvePlan plan;
  std::vector<device::ContactResponse> response;
  std::array<std::vector<float>, 2> multipliers;
  std::array<std::vector<float>, 2> gaps;
  HostSolve view;
};

void make_solve(OwnedSolve& owned, const ContactSystem& system, std::span<const float> multipliers,
                float tolerance) {
  owned.plan = detail::plan_contact_solve(system, multipliers, tolerance);
  const auto rows = owned.plan.multipliers.size();
  owned.response.resize(std::max<std::size_t>(owned.plan.data.body_count, 1));
  owned.multipliers[0] = owned.plan.multipliers;
  owned.multipliers[1].resize(rows);
  owned.gaps[0].resize(rows);
  owned.gaps[1].resize(rows);
  auto& view = owned.view;
  view.data = owned.plan.data;
  view.rows = owned.plan.rows.data();
  view.bodies = owned.plan.bodies.data();
  view.incidence_offsets = owned.plan.incidence_offsets.data();
  view.incidence_entries = owned.plan.incidence_entries.data();
  view.response = owned.response.data();
  view.multipliers = {owned.multipliers[0].data(), owned.multipliers[1].data()};
  view.gaps = {owned.gaps[0].data(), owned.gaps[1].data()};
}

// The response of bodies [begin, end) to the multipliers in pair `current`,
// or with `project` to those one projected step from them.
void respond_range(HostSolve& solve, std::size_t begin, std::size_t end, std::size_t current,
                   std::uint32_t project) {
  for (std::size_t body = begin; body < end; ++body) {
    solve.response[body] = device::contact_respond_body(
        static_cast<std::uint32_t>(body), solve.rows, solve.bodies, solve.multipliers[current],
        solve.gaps[current], solve.state.step_length, project, solve.incidence_offsets,
        solve.incidence_entries, solve.data);
  }
}

// The response of every body, as respond_range gives it.
void respond(HostSolve& solve, std::size_t current, std::uint32_t project) {
  detail::parallel_for(solve.body_count(), body_grain, [&](std::size_t begin, std::size_t end) {
    respond_range(solve, begin, end, current, project);
  });
}

// The sum of the terms of each chunk of `width` rows in [begin, end) by the
// reduction tree, each row's terms from `row_terms(row)`.
template <typename RowTerms>
void sum_chunks(HostSolve& solve, std::size_t begin, std::size_t end, const RowTerms& row_terms) {
  const auto rows = solve.row_count();
  std::array<device::float4, width> lanes{};
  for (std::size_t chunk = begin; chunk < end; ++chunk) {
    const auto first = chunk * width;
    const auto members = std::min(width, rows - first);
    for (std::size_t lane = 0; lane < members; ++lane) {
      lanes[lane] = row_terms(static_cast<std::uint32_t>(first + lane));
    }
    solve.terms[chunk] =
        device::contact_reduction_tree(lanes.data(), static_cast<std::uint32_t>(members));
  }
}

device::float4 sum_levels(std::vector<device::float4>& sums, std::size_t count);

// Sums the terms of every chunk of `width` rows by the reduction tree of
// cpu/kernels/contact_solve.hpp, each row's terms from `row_terms(row)`,
// then the chunk sums the same way until one remains.
template <typename RowTerms>
device::float4 sum_terms(HostSolve& solve, const RowTerms& row_terms) {
  const auto rows = solve.row_count();
  auto count = (rows + width - 1) / width;
  auto& sums = solve.terms;
  sums.resize(count);
  detail::parallel_for(count, chunk_grain, [&](std::size_t begin, std::size_t end) {
    sum_chunks(solve, begin, end, row_terms);
  });
  return sum_levels(sums, count);
}

// Sums the first `count` chunk sums of `sums` by the reduction tree, level by
// level, until one remains.
device::float4 sum_levels(std::vector<device::float4>& sums, std::size_t count) {
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
  respond(solve, 0, 0);
  const auto sums = sum_terms(solve, [&](std::uint32_t row) {
    return device::contact_row_initial(row, solve.rows, solve.response, solve.multipliers[0],
                                       solve.gaps[0], solve.data);
  });
  device::contact_record_step(solve.state, sums, solve.data, 1U, first_step_length, max_iterations);
}

// The whole solve from the multipliers in pair zero, by a team of threads
// that stays together for every iteration. Each member responds for its own
// bodies and sums the terms of its own chunks of rows, the first member sums
// the chunk sums and records the step, and the members meet at a barrier
// between these phases. The sums are those of sum_terms, so the solve gives
// the same result on any number of threads. The state ends with the
// iterations taken and the residual of the multipliers in pair
// `iterations % 2`, whose response `response` holds.
void run_solve(HostSolve& solve, float first_step_length, std::uint32_t max_iterations) {
  const auto rows = solve.row_count();
  if (rows == 0) {
    respond(solve, 0, 0);
    solve.state = {.step_length = first_step_length, .residual = 0.0F, .iterations = 0, .done = 1};
    return;
  }
  const auto bodies = solve.body_count();
  const auto chunks = (rows + width - 1) / width;
  solve.terms.resize(chunks);
  const auto wanted = std::min(chunks, (rows + team_rows - 1) / team_rows);
  detail::parallel_team(
      wanted, [&](std::size_t member, std::size_t members, detail::TeamBarrier& barrier) {
        const auto body_begin = bodies * member / members;
        const auto body_end = bodies * (member + 1) / members;
        const auto chunk_begin = chunks * member / members;
        const auto chunk_end = chunks * (member + 1) / members;
        respond_range(solve, body_begin, body_end, 0, 0);
        barrier.arrive_and_wait();
        sum_chunks(solve, chunk_begin, chunk_end, [&](std::uint32_t row) {
          return device::contact_row_initial(row, solve.rows, solve.response, solve.multipliers[0],
                                             solve.gaps[0], solve.data);
        });
        barrier.arrive_and_wait();
        if (member == 0) {
          device::contact_record_step(solve.state, sum_levels(solve.terms, chunks), solve.data, 1U,
                                      first_step_length, max_iterations);
        }
        barrier.arrive_and_wait();
        std::size_t current = 0;
        while (solve.state.done == 0U) {
          const auto next = 1 - current;
          respond_range(solve, body_begin, body_end, current, 1);
          barrier.arrive_and_wait();
          const auto length = solve.state.step_length;
          sum_chunks(solve, chunk_begin, chunk_end, [&](std::uint32_t row) {
            return device::contact_row_step(
                row, solve.rows, solve.response, solve.multipliers[current], solve.gaps[current],
                solve.multipliers[next], solve.gaps[next], length, solve.data);
          });
          barrier.arrive_and_wait();
          if (member == 0) {
            device::contact_record_step(solve.state, sum_levels(solve.terms, chunks), solve.data,
                                        0U, 0.0F, max_iterations);
          }
          barrier.arrive_and_wait();
          current = next;
        }
      });
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
  OwnedSolve owned;
  make_solve(owned, system, multipliers, 1.0F);
  respond(owned.view, 0, 0);
  return detail::unpack_contact_response(std::span(owned.response).first(owned.view.body_count()));
}

std::vector<float> evaluate_contact_gaps(const ContactSystem& system,
                                         std::span<const float> multipliers) {
  OwnedSolve owned;
  make_solve(owned, system, multipliers, 1.0F);
  if (owned.view.row_count() == 0) {
    return {};
  }
  initialize(owned.view, 1.0F, 0);
  return {owned.gaps[0].begin(),
          owned.gaps[0].begin() + static_cast<std::ptrdiff_t>(owned.view.row_count())};
}

float contact_system_diagonal_bound(const ContactSystem& system) {
  const std::vector<float> zero(system.rows.size(), 0.0F);
  return diagonal_bound(detail::plan_contact_solve(system, zero, 1.0F));
}

ContactSolveResult solve_contact_system_cpu(const ContactSystem& system,
                                            std::span<const float> initial_multipliers,
                                            const ContactSolveParameters& parameters) {
  OwnedSolve owned;
  make_solve(owned, system, initial_multipliers, parameters.tolerance);
  auto& solve = owned.view;
  ContactSolveResult result;
  if (solve.body_count() == 0) {
    return result;
  }
  if (solve.row_count() == 0) {
    respond(solve, 0, 0);
    result.response =
        detail::unpack_contact_response(std::span(owned.response).first(solve.body_count()));
    return result;
  }
  const auto bound = diagonal_bound(owned.plan);
  if (!std::isfinite(bound) || bound <= 0.0F) {
    throw std::invalid_argument("contact system has no positive diagonal");
  }
  run_solve(solve, 1.0F / bound, parameters.max_iterations);
  const auto current = solve.state.iterations % 2;
  result.report = detail::device_contact_solve_report(solve.state.iterations, solve.state.residual,
                                                      parameters.tolerance);
  result.multipliers.assign(
      owned.multipliers[current].begin(),
      owned.multipliers[current].begin() + static_cast<std::ptrdiff_t>(solve.row_count()));
  result.response =
      detail::unpack_contact_response(std::span(owned.response).first(solve.body_count()));
  return result;
}

namespace detail {

void solve_resident_contacts_cpu(const device::ContactStepBindings& bindings,
                                 const device::ContactStepParams& params) {
  if (bindings.status[MS_STEP_FAILED] != 0 || bindings.status[MS_STEP_OVERFLOW] != 0) {
    return;
  }
  HostSolve solve;
  solve.data = device::step_solve_data(bindings, params);
  solve.rows = bindings.rows;
  solve.bodies = bindings.bodies;
  solve.incidence_offsets = bindings.incidence_offsets;
  solve.incidence_entries = bindings.incidence_entries;
  solve.response = bindings.response;
  solve.multipliers = {bindings.multipliers, bindings.multipliers_next};
  solve.gaps = {bindings.gaps, bindings.gaps_next};
  run_solve(solve, std::bit_cast<float>(bindings.status[MS_STEP_FIRST_STEP]),
            device::step_solve_iterations(bindings, params));
  bindings.solve_state[0] = solve.state;
}

}  // namespace detail
}  // namespace microsimulator
