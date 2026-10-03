#pragma once
#include <limits>

#include "flexible_gmres.hpp"
#include "numerics_multigrid.hpp"

namespace cm::detail {
inline std::uint32_t checked_index(std::size_t n) {
  if (n >= std::numeric_limits<std::uint32_t>::max()) {
    throw std::length_error("sparse matrix exceeds uint32 indexing");
  }

  return static_cast<std::uint32_t>(n);
}

using SparseRow = std::vector<std::pair<std::uint32_t, double>>;

struct SparseMatrix {
  std::vector<std::uint32_t> offsets{0}, columns;
  std::vector<double> values, scale, row_sums;
  bool symmetric;

  explicit SparseMatrix(std::vector<SparseRow>& rows, bool symmetric_scaling = true)
      : symmetric(symmetric_scaling) {
    compute_scaling(rows);

    for (std::size_t i = 0; i < rows.size(); ++i) {
      auto& row = rows[i];
      std::sort(row.begin(), row.end());

      for (std::size_t k = 0; k < row.size();) {
        const auto j = row[k].first;
        double a = 0;

        do {
          a += row[k++].second;
        } while (k < row.size() && row[k].first == j);

        if (a != 0) {
          columns.push_back(j);
          values.push_back(a * scale[i] * (symmetric ? scale[j] : 1));
        }
      }

      double row_sum = 0;

      for (auto k = offsets.back(); k < columns.size(); ++k) {
        row_sum += values[k];
      }

      row_sums.push_back(row_sum);
      offsets.push_back(checked_index(columns.size()));
    }
  }

  void compute_scaling(const std::vector<SparseRow>& rows) {
    scale.resize(rows.size(), 1);
    std::vector<double> diagonal(rows.size());

    for (std::size_t i = 0; i < rows.size(); ++i) {
      for (auto [j, a] : rows[i]) {
        if (i == j) {
          diagonal[i] += a;
        }
      }
    }

    // Symmetric equilibration; pressure and rate constraints use the diagonal
    // approximation to their velocity Schur complement. The operator itself
    // remains unchanged, including all inter-component couplings.
    for (auto& d : diagonal) {
      d = std::abs(d);
    }

    // The second Schur level covers rigid-body velocities coupled through
    // marker forces. Use simultaneous updates so ordering cannot change scaling.
    for (unsigned pass = 0; pass < 3; ++pass) {
      auto next = diagonal;

      for (std::size_t i = 0; i < rows.size(); ++i) {
        if (diagonal[i] == 0) {
          for (auto [j, a] : rows[i]) {
            if (diagonal[j] > 0) {
              next[i] += a * a / diagonal[j];
            }
          }
        }
      }

      diagonal = std::move(next);
    }

    for (std::size_t i = 0; i < rows.size(); ++i) {
      if (!(std::isfinite(diagonal[i]) && diagonal[i] > 0)) {
        throw std::invalid_argument("unconstrained linear degree of freedom");
      }

      scale[i] = symmetric ? 1 / std::sqrt(diagonal[i]) : 1 / diagonal[i];
    }
  }

  void apply(const std::vector<double>& x, std::vector<double>& y) const {
    for (std::size_t i = 0; i < y.size(); ++i) {
      double sum = symmetric ? 0 : row_sums[i] * x[i];

      for (auto k = offsets[i]; k < offsets[i + 1]; ++k) {
        sum += values[k] * (symmetric ? x[columns[k]] : x[columns[k]] - x[i]);
      }

      y[i] = sum;
    }
  }
};

double norm2(const std::vector<double>&);
FlexibleKrylovResult<std::vector<double>> solve_cpu(const SparseMatrix&, const std::vector<double>&,
                                                    const LinearSolveParameters&,
                                                    const std::vector<MultigridLevel>&, bool);
FlexibleKrylovResult<std::vector<double>> solve_gpu(NumericsDevice&, const SparseMatrix&,
                                                    const std::vector<double>&,
                                                    const LinearSolveParameters&,
                                                    const std::vector<MultigridLevel>&, bool);
}  // namespace cm::detail
