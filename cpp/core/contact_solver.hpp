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

// The vector operations of the projected-gradient solve of a contact system.
// A backend supplies them over its own storage for the multipliers and their
// gaps; the iteration, its step lengths, and its convergence test are the
// same for every backend.
struct ContactSolverOperations {
  // Loads the initial multipliers and evaluates their gaps. Returns the
  // residual.
  std::function<float()> initialize;
  // multipliers = max(multipliers - step_length * gaps, 0), then evaluates
  // the gaps of the new multipliers.
  std::function<ContactSolverSums(float)> step;
};

// The residual contribution of one row: a positive multiplier needs a zero
// gap, and a zero multiplier needs a non-negative one.
[[nodiscard]] inline float contact_row_residual(float multiplier, float gap) {
  return multiplier > 0.0F ? std::abs(gap) : std::max(0.0F, -gap);
}

// Barzilai-Borwein projected gradient descent on the convex contact energy,
// whose gradient is the gap. The two Barzilai-Borwein step lengths alternate.
// `first_step_length` must be positive.
[[nodiscard]] inline ContactSolveReport solve_contact_system(
    const ContactSolverOperations& operations, float first_step_length,
    const ContactSolveParameters& parameters) {
  ContactSolveReport report;
  report.residual = operations.initialize();
  if (!std::isfinite(report.residual)) {
    report.status = ContactSolveStatus::breakdown;
    return report;
  }
  if (report.residual <= parameters.tolerance) {
    return report;
  }

  report.status = ContactSolveStatus::iteration_limit;
  auto step_length = first_step_length;
  for (std::uint32_t iteration = 1; iteration <= parameters.max_iterations; ++iteration) {
    const auto sums = operations.step(step_length);
    report.iterations = iteration;
    report.residual = sums.residual;
    if (!std::isfinite(sums.residual) || !std::isfinite(sums.step_dot_change)) {
      report.status = ContactSolveStatus::breakdown;
      return report;
    }
    if (sums.residual <= parameters.tolerance) {
      report.status = ContactSolveStatus::converged;
      return report;
    }
    // The energy is convex, so step . change is non-negative; a vanishing
    // product keeps the previous step length.
    if (sums.step_dot_change > 0.0 && sums.change_dot_change > 0.0) {
      const auto next = (iteration % 2U) == 1U ? sums.step_dot_step / sums.step_dot_change
                                               : sums.step_dot_change / sums.change_dot_change;
      if (std::isfinite(next) && next > 0.0) {
        step_length = static_cast<float>(next);
      }
    }
  }
  return report;
}

}  // namespace microsimulator::detail
