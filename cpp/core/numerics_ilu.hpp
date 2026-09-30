#pragma once

#include <limits>
#include <queue>

#include "numerics_sparse.hpp"

namespace cm::detail {
// Threshold incomplete factorization of the entire coupled operator. Dropping
// and pivot safeguards affect only preconditioning, never the physical matrix.
struct IncompleteLu {
  using Row = std::vector<std::pair<std::uint32_t, double>>;
  CsrMatrix factors;
  std::vector<std::uint32_t> diagonals;

  IncompleteLu(const std::vector<std::uint32_t>& offsets, const std::vector<std::uint32_t>& columns,
               const std::vector<double>& values) {
    const auto n = offsets.size() - 1;
    std::vector<Row> upper(n);
    std::vector<double> diagonal(n);
    std::vector<double> row(n);
    std::vector<std::uint32_t> stamp(n, std::numeric_limits<std::uint32_t>::max());
    std::vector<std::uint32_t> touched;
    diagonals.resize(n);
    constexpr std::size_t fill = 80;
    constexpr double drop = 1e-4;

    for (std::uint32_t i = 0; i < n; ++i) {
      touched.clear();
      std::priority_queue<std::uint32_t, std::vector<std::uint32_t>, std::greater<>> pending;
      auto add = [&](std::uint32_t j, double value) {
        if (stamp[j] != i) {
          stamp[j] = i;
          row[j] = value;
          touched.push_back(j);

          if (j < i) {
            pending.push(j);
          }
        } else {
          row[j] += value;
        }
      };
      double scale = 0;

      for (auto j = offsets[i]; j < offsets[i + 1]; ++j) {
        add(columns[j], values[j]);
        scale = std::max(scale, std::abs(values[j]));
      }

      while (!pending.empty()) {
        const auto j = pending.top();
        pending.pop();

        if (std::abs(row[j]) < drop * scale) {
          row[j] = 0;
          continue;
        }

        const double multiplier = row[j] / diagonal[j];
        row[j] = multiplier;

        for (auto [k, value] : upper[j]) {
          add(k, -multiplier * value);
        }
      }

      double pivot = stamp[i] == i ? row[i] : 0;

      if (std::abs(pivot) < 1e-6 * scale) {
        pivot = std::copysign(1e-6 * scale, pivot == 0 ? -1 : pivot);
      }

      diagonal[i] = pivot;

      for (bool lower : {true, false}) {
        auto candidates = select_fill(row, touched, i, lower, drop * scale, fill);

        for (auto [j, value] : candidates) {
          factors.columns.push_back(j);
          factors.values.push_back(value);
        }

        if (lower) {
          diagonals[i] = static_cast<std::uint32_t>(factors.values.size());
          factors.columns.push_back(i);
          factors.values.push_back(pivot);
        } else {
          upper[i] = std::move(candidates);
        }
      }

      factors.offsets.push_back(static_cast<std::uint32_t>(factors.values.size()));
    }
  }

  static Row select_fill(const std::vector<double>& row, const std::vector<std::uint32_t>& touched,
                         std::uint32_t i, bool lower, double threshold, std::size_t fill) {
    Row candidates;

    for (auto j : touched) {
      if (lower ? j >= i : j <= i) {
        continue;
      }

      if (!std::isfinite(row[j])) {
        throw std::runtime_error("nonfinite sparse preconditioner");
      }

      if (std::abs(row[j]) >= threshold) {
        candidates.emplace_back(j, row[j]);
      }
    }

    auto magnitude = [](auto a, auto b) {
      if (std::abs(a.second) != std::abs(b.second)) {
        return std::abs(a.second) > std::abs(b.second);
      }

      return a.first < b.first;
    };

    if (candidates.size() > fill) {
      std::nth_element(candidates.begin(), candidates.begin() + fill, candidates.end(), magnitude);
      candidates.resize(fill);
    }

    std::sort(candidates.begin(), candidates.end());

    return candidates;
  }

  void apply(const std::vector<double>& rhs, std::vector<double>& x) const {
    const auto n = factors.size();

    for (std::uint32_t i = 0; i < n; ++i) {
      double value = rhs[i];

      for (auto j = factors.offsets[i]; j < diagonals[i]; ++j) {
        value -= factors.values[j] * x[factors.columns[j]];
      }

      x[i] = value;
    }

    for (std::uint32_t i = n; i-- > 0;) {
      double value = x[i];

      for (auto j = diagonals[i] + 1; j < factors.offsets[i + 1]; ++j) {
        value -= factors.values[j] * x[factors.columns[j]];
      }

      x[i] = value / factors.values[diagonals[i]];
    }
  }
};
}  // namespace cm::detail
