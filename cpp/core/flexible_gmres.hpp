#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cm::detail {

// The same scalar Arnoldi control is used with CPU or device-resident vectors.
// Only the preconditioner may vary; apply must represent a fixed linear map.
template <class Vector>
struct FlexibleKrylovOperations {
  std::function<Vector()> make_zero;
  std::function<void(const Vector&, Vector&)> copy;
  std::function<void(Vector&, double, const Vector&)> axpy;
  std::function<double(const Vector&, const Vector&)> dot;
  std::function<void(const Vector&, Vector&)> apply;
  std::function<void(const Vector&, Vector&)> precondition;
};

template <class Vector>
struct FlexibleKrylovResult {
  Vector solution;
  std::uint32_t iterations;
  double relative_residual;
};

template <class Vector>
FlexibleKrylovResult<Vector> flexible_gmres(const FlexibleKrylovOperations<Vector>& op,
                                            const Vector& rhs, double tolerance,
                                            std::uint32_t max_iterations) {
  auto solution = op.make_zero();
  auto residual = op.make_zero();
  auto work = op.make_zero();
  const double rhs_norm = std::sqrt(op.dot(rhs, rhs));
  if (!std::isfinite(rhs_norm)) throw std::runtime_error("FGMRES non-finite right-hand side");
  if (rhs_norm == 0) return {std::move(solution), 0, 0};
  std::uint32_t iterations = 0;
  constexpr std::uint32_t restart = 40;
  while (true) {
    op.apply(solution, work);
    op.copy(rhs, residual);
    op.axpy(residual, -1, work);
    const double beta = std::sqrt(op.dot(residual, residual));
    if (!std::isfinite(beta)) throw std::runtime_error("FGMRES non-finite true residual");
    if (beta <= tolerance * rhs_norm) return {std::move(solution), iterations, beta / rhs_norm};
    if (iterations >= max_iterations)
      throw std::runtime_error("resolved-flow FGMRES did not converge: relative residual " +
                               std::to_string(beta / rhs_norm));
    const auto count = std::min(restart, max_iterations - iterations);
    std::vector<Vector> basis, directions;
    auto first = op.make_zero();
    op.axpy(first, 1 / beta, residual);
    basis.push_back(std::move(first));
    std::vector<std::vector<double>> h(count + 1, std::vector<double>(count));
    std::vector<double> cosine(count), sine(count), g(count + 1);
    g[0] = beta;
    std::uint32_t used = 0;
    for (std::uint32_t j = 0; j < count; ++j) {
      auto direction = op.make_zero();
      op.precondition(basis[j], direction);
      op.apply(direction, work);
      // Twice-modified Gram-Schmidt limits loss of orthogonality in binary32.
      for (unsigned pass = 0; pass < 2; ++pass) {
        for (std::uint32_t i = 0; i <= j; ++i) {
          const double projection = op.dot(basis[i], work);
          h[i][j] += projection;
          op.axpy(work, -projection, basis[i]);
        }
      }
      h[j + 1][j] = std::sqrt(op.dot(work, work));
      if (!std::isfinite(h[j + 1][j])) throw std::runtime_error("FGMRES non-finite Arnoldi vector");
      const bool happy = h[j + 1][j] <= 1e-14;
      auto next = op.make_zero();
      if (!happy) op.axpy(next, 1 / h[j + 1][j], work);
      basis.push_back(std::move(next));
      directions.push_back(std::move(direction));
      for (std::uint32_t i = 0; i < j; ++i) {
        const double upper = cosine[i] * h[i][j] + sine[i] * h[i + 1][j];
        h[i + 1][j] = -sine[i] * h[i][j] + cosine[i] * h[i + 1][j];
        h[i][j] = upper;
      }
      const double diagonal = std::hypot(h[j][j], h[j + 1][j]);
      if (diagonal == 0) throw std::runtime_error("FGMRES Arnoldi breakdown");
      cosine[j] = h[j][j] / diagonal;
      sine[j] = h[j + 1][j] / diagonal;
      h[j][j] = diagonal;
      h[j + 1][j] = 0;
      g[j + 1] = -sine[j] * g[j];
      g[j] *= cosine[j];
      ++iterations;
      used = j + 1;
      if (happy || std::abs(g[j + 1]) <= tolerance * rhs_norm) break;
    }
    std::vector<double> weights(used);
    for (std::uint32_t i = used; i-- > 0;) {
      double value = g[i];
      for (std::uint32_t j = i + 1; j < used; ++j) value -= h[i][j] * weights[j];
      weights[i] = value / h[i][i];
    }
    for (std::uint32_t i = 0; i < used; ++i) op.axpy(solution, weights[i], directions[i]);
  }
}

}  // namespace cm::detail
