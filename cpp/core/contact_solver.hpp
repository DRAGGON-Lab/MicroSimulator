#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>

#include "microsimulator/contact_dynamics.hpp"

namespace microsimulator::detail {

// What one projected-gradient step reports. `step` is the change in the
// multipliers and `change` the change in their gaps.
struct ContactSolverSums {
  double step_dot_step{0.0};
  double step_dot_change{0.0};
  double change_dot_change{0.0};
  float residual{0.0F};
};

// Where a run of iterations ended.
struct ContactSolverProgress {
  // Iterations taken in this run.
  std::uint32_t iterations{0};
  float residual{0.0F};
};

// The projected-gradient solve of a contact system over a backend's own
// storage for the multipliers and their gaps.
struct ContactSolverOperations {
  // Loads the initial multipliers, evaluates their gaps, and sets the first
  // step length. Returns the residual.
  std::function<float(float)> initialize;
  // Takes up to the given number of steps, stopping after the first whose
  // residual is within the tolerance or not finite. Each step is
  // multipliers = max(multipliers - step_length * gaps, 0) followed by
  // `next_contact_step_length`.
  std::function<ContactSolverProgress(std::uint32_t)> iterate;
  // Steps requested per call of `iterate`. A device backend runs a batch
  // without returning to the host.
  std::uint32_t batch{1};
};

// The residual contribution of one row: a positive multiplier needs a zero
// gap, and a zero multiplier needs a non-negative one.
[[nodiscard]] inline float contact_row_residual(float multiplier, float gap) {
  return multiplier > 0.0F ? std::abs(gap) : std::max(0.0F, -gap);
}

// The Barzilai-Borwein step length after the step numbered `iteration`,
// counting from one. The two lengths alternate. The contact energy is convex,
// so step . change is non-negative; a vanishing product keeps the current
// length.
[[nodiscard]] inline float next_contact_step_length(float current, const ContactSolverSums& sums,
                                                    std::uint32_t iteration) {
  if (!(sums.step_dot_change > 0.0) || !(sums.change_dot_change > 0.0)) {
    return current;
  }
  const auto next = (iteration % 2U) == 1U ? sums.step_dot_step / sums.step_dot_change
                                           : sums.step_dot_change / sums.change_dot_change;
  return std::isfinite(next) && next > 0.0 ? static_cast<float>(next) : current;
}

// Runs `iterate`'s contract over a host-driven step: `step(step_length)`
// takes one projected step and returns its sums.
template <typename Step>
[[nodiscard]] ContactSolverProgress iterate_contact_steps(std::uint32_t count, float tolerance,
                                                          float& step_length,
                                                          std::uint32_t& iteration, Step&& step) {
  ContactSolverProgress progress;
  for (std::uint32_t index = 0; index < count; ++index) {
    const ContactSolverSums sums = step(step_length);
    ++iteration;
    ++progress.iterations;
    progress.residual = sums.residual;
    if (!(sums.residual > tolerance) || !std::isfinite(sums.residual)) {
      break;
    }
    step_length = next_contact_step_length(step_length, sums, iteration);
  }
  return progress;
}

// Barzilai-Borwein projected gradient descent on the convex contact energy,
// whose gradient is the gap. `first_step_length` must be positive.
[[nodiscard]] inline ContactSolveReport solve_contact_system(
    const ContactSolverOperations& operations, float first_step_length,
    const ContactSolveParameters& parameters) {
  ContactSolveReport report;
  report.residual = operations.initialize(first_step_length);
  if (!std::isfinite(report.residual)) {
    report.status = ContactSolveStatus::breakdown;
    return report;
  }
  if (report.residual <= parameters.tolerance) {
    return report;
  }

  report.status = ContactSolveStatus::iteration_limit;
  const auto batch = std::max<std::uint32_t>(1, operations.batch);
  while (report.iterations < parameters.max_iterations) {
    const auto requested = std::min(batch, parameters.max_iterations - report.iterations);
    const auto progress = operations.iterate(requested);
    report.iterations += progress.iterations;
    report.residual = progress.residual;
    if (!std::isfinite(progress.residual)) {
      report.status = ContactSolveStatus::breakdown;
      return report;
    }
    if (progress.residual <= parameters.tolerance) {
      report.status = ContactSolveStatus::converged;
      return report;
    }
    if (progress.iterations == 0) {
      report.status = ContactSolveStatus::breakdown;
      return report;
    }
  }
  return report;
}

}  // namespace microsimulator::detail
