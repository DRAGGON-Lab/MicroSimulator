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

// Why a solve whose scalars are kept on the device stopped short.
enum class ConjugateGradientFailure : std::uint32_t {
  none = 0,
  curvature = 1,
  step = 2,
  residual = 3,
  breakdown = 4,
  direction = 5,
};

// Where a solve that iterates on the device stands after a batch.
struct ConjugateGradientProgress {
  // Iterations taken; one that meets the threshold is the last.
  std::uint32_t iterations{0};
  bool converged{false};
  ConjugateGradientFailure failure{ConjugateGradientFailure::none};
  // r . r after the last iteration taken.
  double residual_squared{0.0};
};

// A solve whose step length and direction weight are computed on the device,
// so that the host waits once for a batch of iterations and not several times
// for each. The iteration is that of preconditioned_conjugate_gradient.
struct BatchedConjugateGradientOperations {
  // x = 0, r = b, z = M^-1 r, p = z, and the device's count of iterations
  // cleared. Returns r . r.
  std::function<double()> initialize;
  // Takes up to `count` further iterations, stopping at the first after
  // which r . r is at most `threshold`, or which fails. An iteration past
  // that point changes nothing.
  std::function<ConjugateGradientProgress(std::uint32_t count, double threshold)> iterate;
};

// Iterates in batches of `batch` until the residual norm falls to `tolerance`
// times the norm of the right-hand side. Throws as
// preconditioned_conjugate_gradient does.
[[nodiscard]] inline ConjugateGradientReport batched_conjugate_gradient(
    const BatchedConjugateGradientOperations& operations, float tolerance,
    std::uint32_t max_iterations, std::uint32_t batch, std::string_view label) {
  const auto fail = [&](const char* detail) {
    return std::runtime_error(std::string(label) + " conjugate gradient " + detail);
  };
  const auto rhs_norm_squared = operations.initialize();
  if (rhs_norm_squared == 0.0) {
    return {};
  }
  const auto rhs_norm = std::sqrt(rhs_norm_squared);
  const auto bound = static_cast<double>(tolerance) * rhs_norm;
  const auto threshold = bound * bound;
  auto relative = 1.0;
  std::uint32_t encoded = 0;
  while (encoded < max_iterations) {
    const auto count = std::min(batch, max_iterations - encoded);
    const auto progress = operations.iterate(count, threshold);
    encoded += count;
    switch (progress.failure) {
      case ConjugateGradientFailure::none:
        break;
      case ConjugateGradientFailure::curvature:
        throw fail("encountered non-positive curvature");
      case ConjugateGradientFailure::step:
        throw fail("produced a non-finite step");
      case ConjugateGradientFailure::residual:
        throw fail("produced a non-finite residual");
      case ConjugateGradientFailure::breakdown:
        throw fail("encountered a preconditioner breakdown");
      case ConjugateGradientFailure::direction:
        throw fail("produced a non-finite direction");
    }
    relative = std::sqrt(std::max(0.0, progress.residual_squared)) / rhs_norm;
    if (progress.converged) {
      return {.iterations = progress.iterations, .relative_residual = relative};
    }
  }
  throw std::runtime_error(std::string(label) + " conjugate gradient did not converge: relative " +
                           std::to_string(relative));
}

}  // namespace microsimulator::detail
