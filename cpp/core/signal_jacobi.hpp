#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>

#include "microsimulator/signals.hpp"

namespace microsimulator::detail {

struct SignalSweepResult {
  // Residual RMS of the iterate the sweep produced.
  float residual_rms{0.0F};
  // False when the sweep computed an invalid value.
  bool valid{true};
};

// The operations of the Jacobi iteration behind an implicit signal step. A
// backend supplies them over its own storage for the iterate; the stopping
// rule is the same for every backend.
struct SignalJacobiOperations {
  // Residual RMS of the starting iterate.
  std::function<float()> initial_residual_rms;
  std::function<float()> right_hand_side_rms;
  // Largest diagonal entry of the implicit operator, from
  // signal_implicit_operator_scale.
  float operator_scale{1.0F};
  // Called once, before the first sweep, when the starting iterate is not
  // already converged.
  std::function<void()> begin_sweeps;
  // Replaces the iterate with its Jacobi update.
  std::function<SignalSweepResult()> sweep;
};

// Sweeps until the residual RMS falls to the threshold or the iteration limit
// is reached.
//
// The relative term scales the residual the step starts with, not the field it
// starts from. A field's own magnitude says nothing about how much of it this
// step has to change, so scaling by the field lets a small source fall under
// the threshold and be discarded; scaling by the initial residual asks for a
// fixed reduction of whatever this step actually has to resolve.
//
// A residual cannot fall below what float32 can represent. It is the
// difference between the right-hand side and the implicit operator applied to
// the iterate, so its rounding error scales with the field's magnitude, which
// the right-hand side carries, times the operator's largest diagonal entry: a
// fine lattice or a strong loss makes that entry large and the reachable
// residual correspondingly coarser. An absolute tolerance asking for less
// than this floor is raised to it rather than making the solve unreachable.
[[nodiscard]] inline SignalSolveReport solve_signal_jacobi(
    const SignalJacobiOperations& operations, const SignalSolveParameters& parameters) {
  SignalSolveReport report;
  report.residual_rms = operations.initial_residual_rms();
  const auto floor = std::numeric_limits<float>::epsilon() * operations.right_hand_side_rms() *
                     operations.operator_scale;
  const auto threshold = std::max(parameters.absolute_tolerance, floor) +
                         (parameters.relative_tolerance * report.residual_rms);
  if (std::isfinite(report.residual_rms) && report.residual_rms <= threshold) {
    report.converged = true;
    return report;
  }
  report.converged = false;
  if (!std::isfinite(report.residual_rms) || !std::isfinite(threshold)) {
    return report;
  }

  operations.begin_sweeps();
  for (std::uint32_t iteration = 1; iteration <= parameters.max_iterations; ++iteration) {
    const auto result = operations.sweep();
    report.residual_rms = result.residual_rms;
    report.iterations = iteration;
    if (!result.valid || !std::isfinite(report.residual_rms)) {
      return report;
    }
    if (report.residual_rms <= threshold) {
      report.converged = true;
      return report;
    }
  }
  return report;
}

}  // namespace microsimulator::detail
