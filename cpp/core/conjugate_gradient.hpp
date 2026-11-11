#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace microsimulator::detail {

// The vector operations of a Jacobi-preconditioned conjugate-gradient solve of
// A x = b. A backend supplies them over its own storage for the solution x,
// residual r, preconditioned residual z, search direction p, and transformed
// direction A p; the iteration itself is the same for every backend.
struct ConjugateGradientOperations {
  // x = 0, r = b, z = M^-1 r, p = z.
  std::function<void()> initialize;
  std::function<double()> residual_dot_residual;
  std::function<double()> residual_dot_preconditioned;
  // Computes A p and returns p . A p.
  std::function<double()> apply_direction;
  // x += alpha p and r -= alpha A p.
  std::function<void(double)> update;
  // z = M^-1 r.
  std::function<void()> precondition;
  // p = z + beta p.
  std::function<void(double)> update_direction;
  // Largest step magnitude the backend's vector arithmetic can represent.
  double step_limit{std::numeric_limits<double>::infinity()};
};

struct ConjugateGradientReport {
  std::uint32_t iterations{0};
  double relative_residual{0.0};
};

// Iterates until the residual norm falls to `tolerance` times the norm of the
// right-hand side. Throws std::runtime_error, prefixed with `label`, on
// breakdown or when `max_iterations` is reached first.
[[nodiscard]] inline ConjugateGradientReport preconditioned_conjugate_gradient(
    const ConjugateGradientOperations& operations, float tolerance, std::uint32_t max_iterations,
    std::string_view label) {
  const auto fail = [&](const char* detail) {
    return std::runtime_error(std::string(label) + " conjugate gradient " + detail);
  };
  operations.initialize();
  const auto rhs_norm_squared = operations.residual_dot_residual();
  if (rhs_norm_squared == 0.0) {
    return {};
  }
  const auto rhs_norm = std::sqrt(rhs_norm_squared);
  auto rho = operations.residual_dot_preconditioned();
  auto relative = 1.0;
  for (std::uint32_t iteration = 1; iteration <= max_iterations; ++iteration) {
    const auto curvature = operations.apply_direction();
    if (!std::isfinite(curvature) || curvature <= 0.0) {
      throw fail("encountered non-positive curvature");
    }
    const auto alpha = rho / curvature;
    if (!std::isfinite(alpha) || std::abs(alpha) > operations.step_limit) {
      throw fail("produced a non-finite step");
    }
    operations.update(alpha);
    relative = std::sqrt(std::max(0.0, operations.residual_dot_residual())) / rhs_norm;
    if (!std::isfinite(relative)) {
      throw fail("produced a non-finite residual");
    }
    if (relative <= tolerance) {
      return {.iterations = iteration, .relative_residual = relative};
    }
    operations.precondition();
    const auto next_rho = operations.residual_dot_preconditioned();
    if (!std::isfinite(next_rho) || rho == 0.0) {
      throw fail("encountered a preconditioner breakdown");
    }
    const auto beta = next_rho / rho;
    if (!std::isfinite(beta) || std::abs(beta) > operations.step_limit) {
      throw fail("produced a non-finite direction");
    }
    operations.update_direction(beta);
    rho = next_rho;
  }
  throw std::runtime_error(std::string(label) + " conjugate gradient did not converge: relative " +
                           std::to_string(relative));
}

}  // namespace microsimulator::detail
