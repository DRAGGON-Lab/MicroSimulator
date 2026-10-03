#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <vector>

#include "numerics_device.hpp"

namespace cm::detail {
struct CsrMatrix {
  std::vector<std::uint32_t> offsets{0}, columns;
  std::vector<double> values;
  using Rows = std::vector<std::map<std::uint32_t, double>>;

  explicit CsrMatrix(const Rows& rows = {}) {
    for (const auto& row : rows) {
      for (auto [j, a] : row) {
        if (a != 0) {
          columns.push_back(j);
          values.push_back(a);
        }
      }

      offsets.push_back(static_cast<std::uint32_t>(values.size()));
    }
  }

  std::uint32_t size() const {
    return static_cast<std::uint32_t>(offsets.size() - 1);
  }
};

struct CpuSparseAlgebra {
  using Vector = std::vector<double>;
  using Matrix = CsrMatrix;

  Vector make(std::uint32_t n) const {
    return Vector(n);
  }

  Vector upload(const std::vector<double>& x) const {
    return x;
  }

  Matrix upload(const CsrMatrix& a) const {
    return a;
  }

  void fill(Vector& x, std::uint32_t n) const {
    std::fill_n(x.begin(), n, 0);
  }

  void copy(const Vector& x, Vector& y, std::uint32_t n) const {
    std::copy_n(x.begin(), n, y.begin());
  }

  void axpy(const Vector& x, Vector& y, double v, std::uint32_t n) const {
    for (std::uint32_t i = 0; i < n; ++i) {
      y[i] += v * x[i];
    }
  }

  void apply(const Matrix& a, const Vector& x, Vector& y) const {
    for (std::uint32_t i = 0; i < a.size(); ++i) {
      double v = 0;

      for (auto j = a.offsets[i]; j < a.offsets[i + 1]; ++j) {
        v += a.values[j] * x[a.columns[j]];
      }

      y[i] = v;
    }
  }

  void jacobi(const Vector& rhs, const Vector& ax, const Vector& d, Vector& x,
              std::uint32_t n) const {
    for (std::uint32_t i = 0; i < n; ++i) {
      x[i] += 0.5 * d[i] * (rhs[i] - ax[i]);
    }
  }
};

struct GpuSparseAlgebra {
  using Vector = NumericsBufferPtr;

  struct Matrix {
    Vector offsets, columns, values;
    std::uint32_t n;
  };

  NumericsDevice& device;

  Vector make(std::uint32_t n) const {
    return device.allocate(std::size_t(n) * sizeof(float));
  }

  Vector upload(const std::vector<double>& x) const {
    std::vector<float> f(x.begin(), x.end());

    for (float v : f) {
      if (!std::isfinite(v)) {
        throw std::overflow_error("sparse preconditioner overflow");
      }
    }

    return device.upload(f);
  }

  Matrix upload(const CsrMatrix& a) const {
    return {device.upload(a.offsets), device.upload(a.columns), upload(a.values), a.size()};
  }

  void fill(Vector& x, std::uint32_t n) const {
    device.dispatch(NumericsKernel::fill, {n, 0}, {x});
  }

  void copy(const Vector& x, Vector& y, std::uint32_t n) const {
    device.dispatch(NumericsKernel::copy, {n, 0}, {x, y});
  }

  void axpy(const Vector& x, Vector& y, double v, std::uint32_t n) const {
    device.dispatch(NumericsKernel::axpy, {n, float(v)}, {x, y});
  }

  void apply(const Matrix& a, const Vector& x, Vector& y) const {
    device.dispatch(NumericsKernel::apply, {a.n, 0}, {a.offsets, a.columns, a.values, x, y});
  }

  void jacobi(const Vector& rhs, const Vector& ax, const Vector& d, Vector& x,
              std::uint32_t n) const {
    device.dispatch(NumericsKernel::jacobi, {n, 0.5F}, {rhs, ax, d, x});
  }
};

}  // namespace cm::detail
