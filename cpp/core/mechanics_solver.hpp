#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>

#include "core/shared_constants.hpp"
#include "microsimulator/mechanics.hpp"

namespace microsimulator::detail {

// The vector operations of the conjugate-gradient solve of the regularized
// contact system. A backend supplies them over its own storage for the
// solution, residual, and search direction; the iteration, its convergence
// test, and its breakdown reporting are the same for every backend.
struct MechanicsSolverOperations {
  // solution = 0, residual = right-hand side, direction = residual. Returns
  // residual . residual.
  std::function<float()> initialize;
  // Applies the system operator to the direction and returns the curvature
  // direction . (A direction).
  std::function<float()> apply_direction;
  // solution += alpha direction and residual -= alpha (A direction). Returns
  // residual . residual.
  std::function<float(float)> update;
  // Recomputes residual = right-hand side - A solution from the solution
  // itself. Returns residual . residual.
  std::function<float()> recompute_residual;
  // direction = residual + beta direction.
  std::function<void(float)> update_direction;
};

// Iteration limit of the mechanics solve: the caller's limit, or the number of
// unknowns when the caller leaves it at zero.
[[nodiscard]] inline std::uint32_t mechanics_iteration_limit(const MechanicsParameters& parameters,
                                                             std::size_t cell_count) {
  if (parameters.max_iterations != 0) {
    return parameters.max_iterations;
  }
  constexpr std::size_t degrees_of_freedom = shared::mechanics_degrees_of_freedom;
  if (cell_count > std::numeric_limits<std::uint32_t>::max() / degrees_of_freedom) {
    throw std::overflow_error("default mechanics iteration limit exceeds uint32");
  }
  return static_cast<std::uint32_t>(cell_count * degrees_of_freedom);
}

// Solves the system and reports how the iteration ended. A residual that
// reaches the tolerance by recurrence is recomputed from the solution before
// it is accepted; if the recomputed residual misses the tolerance the search
// restarts from it. `cell_count` must be positive.
[[nodiscard]] inline SolverReport solve_mechanics_system(
    const MechanicsSolverOperations& operations, std::size_t cell_count,
    const MechanicsParameters& parameters) {
  const auto count = static_cast<float>(cell_count);
  const auto breakdown = [](SolverReport& report, SolverBreakdown reason) {
    report.status = SolverStatus::breakdown;
    report.breakdown = reason;
  };

  SolverReport report;
  auto residual_squared = operations.initialize();
  report.initial_residual_rms = std::sqrt(residual_squared / count);
  report.final_residual_rms = report.initial_residual_rms;
  if (!std::isfinite(report.initial_residual_rms)) {
    breakdown(report, SolverBreakdown::non_finite_residual);
    return report;
  }
  if (report.initial_residual_rms <= parameters.residual_rms_tolerance) {
    return report;
  }

  report.status = SolverStatus::iteration_limit;
  const auto maximum_iterations = mechanics_iteration_limit(parameters, cell_count);
  for (std::uint32_t iteration = 0; iteration < maximum_iterations; ++iteration) {
    const auto curvature = operations.apply_direction();
    if (!std::isfinite(curvature)) {
      breakdown(report, SolverBreakdown::non_finite_curvature);
      break;
    }
    if (curvature <= 0.0F) {
      breakdown(report, SolverBreakdown::non_positive_curvature);
      break;
    }

    const auto alpha = residual_squared / curvature;
    const auto next_residual_squared = operations.update(alpha);
    report.iterations = iteration + 1;
    const auto recurrence_rms = std::sqrt(next_residual_squared / count);
    if (!std::isfinite(recurrence_rms)) {
      breakdown(report, SolverBreakdown::non_finite_residual);
      break;
    }

    if (recurrence_rms <= parameters.residual_rms_tolerance) {
      residual_squared = operations.recompute_residual();
      const auto recomputed_rms = std::sqrt(residual_squared / count);
      if (!std::isfinite(recomputed_rms)) {
        breakdown(report, SolverBreakdown::non_finite_residual);
        break;
      }
      if (recomputed_rms <= parameters.residual_rms_tolerance) {
        report.status = SolverStatus::converged;
        break;
      }
      operations.update_direction(0.0F);
      continue;
    }

    operations.update_direction(next_residual_squared / residual_squared);
    residual_squared = next_residual_squared;
  }

  report.final_residual_rms = std::sqrt(operations.recompute_residual() / count);
  if (!std::isfinite(report.final_residual_rms) && report.status != SolverStatus::breakdown) {
    breakdown(report, SolverBreakdown::non_finite_residual);
  }
  return report;
}

}  // namespace microsimulator::detail
