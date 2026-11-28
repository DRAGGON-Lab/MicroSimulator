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

// Where a run of sweeps ended.
struct SignalSweepBatch {
  // Sweeps taken in this run.
  std::uint32_t iterations{0};
  // Residual RMS of the last iterate the run produced.
  float residual_rms{0.0F};
  // False when a sweep computed an invalid value.
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
  // Replaces the iterate with its Jacobi update. Used when `sweep_batch` is
  // empty.
  std::function<SignalSweepResult()> sweep;
  // Takes up to the given number of sweeps without returning to the host,
  // stopping after the first whose residual RMS is within the given
  // threshold, is not finite, or whose sweep is invalid. A backend that
  // leaves it empty sweeps one at a time through `sweep`.
  std::function<SignalSweepBatch(std::uint32_t, float)> sweep_batch;
  // Sweeps requested per call of `sweep_batch`.
  std::uint32_t batch{1};
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
  if (operations.sweep_batch) {
    const auto batch = std::max<std::uint32_t>(1, operations.batch);
    while (report.iterations < parameters.max_iterations) {
      const auto requested = std::min(batch, parameters.max_iterations - report.iterations);
      const auto result = operations.sweep_batch(requested, threshold);
      report.iterations += result.iterations;
      report.residual_rms = result.residual_rms;
      if (!result.valid || !std::isfinite(report.residual_rms)) {
        return report;
      }
      if (report.residual_rms <= threshold) {
        report.converged = true;
        return report;
      }
      if (result.iterations == 0) {
        return report;
      }
    }
    return report;
  }
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
