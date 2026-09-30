#pragma once
#include <array>

#include "numerics_multigrid.hpp"

namespace cm::detail {
// Galerkin aggregation of the SPD velocity block by MAC component.
// Pressure and port constraints remain in the outer FGMRES operator.
using VelocityCoordinate = std::array<std::uint32_t, 4>;  // component, x, y, z

inline CsrMatrix velocity_coarse_inverse(const CsrMatrix& a,
                                         const std::vector<double>& inverse_diagonal) {
  const auto n = a.size();
  // Dense Cholesky at the coarsest level. Only the inverse coefficients are
  // prepared on the host; every application is native on the selected device.
  std::vector<double> l(std::size_t(n) * n);

  for (std::uint32_t i = 0; i < n; ++i) {
    for (auto j = a.offsets[i]; j < a.offsets[i + 1]; ++j) {
      l[std::size_t(i) * n + a.columns[j]] = a.values[j];
    }
  }

  for (std::uint32_t i = 0; i < n; ++i) {
    for (std::uint32_t j = 0; j <= i; ++j) {
      auto& x = l[std::size_t(i) * n + j];

      for (std::uint32_t k = 0; k < j; ++k) {
        x -= l[std::size_t(i) * n + k] * l[std::size_t(j) * n + k];
      }

      if (i == j) {
        if (!(x > 1e-12 / inverse_diagonal[i])) {
          throw std::invalid_argument("fluid velocity has unconstrained rigid motion");
        }

        x = std::sqrt(x);
      } else {
        x /= l[std::size_t(j) * n + j];
      }
    }
  }

  CsrMatrix::Rows inverse(n);

  for (std::uint32_t column = 0; column < n; ++column) {
    std::vector<double> x(n);

    for (std::uint32_t i = 0; i < n; ++i) {
      double v = i == column ? 1 : 0;

      for (std::uint32_t j = 0; j < i; ++j) {
        v -= l[std::size_t(i) * n + j] * x[j];
      }

      x[i] = v / l[std::size_t(i) * n + i];
    }

    for (std::uint32_t i = n; i-- > 0;) {
      for (std::uint32_t j = i + 1; j < n; ++j) {
        x[i] -= l[std::size_t(j) * n + i] * x[j];
      }

      x[i] /= l[std::size_t(i) * n + i];
      inverse[i][column] = x[i];
    }
  }

  return CsrMatrix(inverse);
}

inline std::vector<MultigridLevel> build_velocity_hierarchy(
    CsrMatrix a, std::vector<VelocityCoordinate> coordinates) {
  std::vector<MultigridLevel> levels;

  if (!a.size()) {
    return levels;
  }

  while (true) {
    MultigridLevel level;
    const auto n = a.size();
    level.inverse_diagonal.resize(n);

    for (std::uint32_t i = 0; i < n; ++i) {
      for (auto j = a.offsets[i]; j < a.offsets[i + 1]; ++j) {
        if (a.columns[j] == i) {
          level.inverse_diagonal[i] = 1 / a.values[j];
        }
      }
    }

    if (n <= 96) {
      level.inverse = velocity_coarse_inverse(a, level.inverse_diagonal);
      level.a = std::move(a);
      levels.push_back(std::move(level));

      return levels;
    }

    std::map<VelocityCoordinate, std::uint32_t> aggregates;
    std::vector<VelocityCoordinate> coarse_coordinates;
    std::vector<std::uint32_t> parent(n);

    for (std::uint32_t i = 0; i < n; ++i) {
      auto c = coordinates[i];

      for (std::size_t k = 1; k < 4; ++k) {
        c[k] /= 2;
      }

      auto [it, inserted] = aggregates.emplace(c, static_cast<std::uint32_t>(aggregates.size()));
      parent[i] = it->second;

      if (inserted) {
        coarse_coordinates.push_back(c);
      }
    }

    const auto nc = aggregates.size();

    if (nc >= n) {
      throw std::runtime_error("fluid multigrid failed to coarsen");
    }

    CsrMatrix::Rows coarse(nc), restriction(nc), prolongation(n);

    for (std::uint32_t i = 0; i < n; ++i) {
      restriction[parent[i]][i] = 1;
      prolongation[i][parent[i]] = 1;

      for (auto k = a.offsets[i]; k < a.offsets[i + 1]; ++k) {
        coarse[parent[i]][parent[a.columns[k]]] += a.values[k];
      }
    }

    level.a = std::move(a);
    level.restriction = CsrMatrix(restriction);
    level.prolongation = CsrMatrix(prolongation);
    levels.push_back(std::move(level));
    a = CsrMatrix(coarse);
    coordinates = std::move(coarse_coordinates);
  }
}

}  // namespace cm::detail
