#pragma once
#include "numerics_sparse.hpp"

namespace cm::detail {
struct MultigridLevel {
  CsrMatrix a, restriction, prolongation, inverse;
  std::vector<double> inverse_diagonal;
};

template <class Algebra>
class SparseMultigrid {
  using Vector = typename Algebra::Vector;
  using Matrix = typename Algebra::Matrix;

  struct Level {
    std::uint32_t n;
    Matrix a, restriction, prolongation, inverse;
    Vector diagonal, rhs, x, work, residual;
  };

  Algebra algebra_;
  std::vector<Level> levels_;

  void cycle(std::size_t index) {
    auto& l = levels_[index];
    algebra_.fill(l.x, l.n);

    if (index + 1 == levels_.size()) {
      algebra_.apply(l.inverse, l.rhs, l.x);

      return;
    }

    auto smooth = [&] {
      for (unsigned sweep = 0; sweep < 4; ++sweep) {
        algebra_.apply(l.a, l.x, l.work);
        algebra_.jacobi(l.rhs, l.work, l.diagonal, l.x, l.n);
      }
    };
    smooth();
    algebra_.apply(l.a, l.x, l.work);
    algebra_.copy(l.rhs, l.residual, l.n);
    algebra_.axpy(l.work, l.residual, -1, l.n);
    algebra_.apply(l.restriction, l.residual, levels_[index + 1].rhs);
    cycle(index + 1);
    algebra_.apply(l.prolongation, levels_[index + 1].x, l.work);
    algebra_.axpy(l.work, l.x, 1, l.n);
    smooth();
  }

 public:
  SparseMultigrid(Algebra algebra, const std::vector<MultigridLevel>& data) : algebra_(algebra) {
    for (const auto& l : data) {
      const auto n = l.a.size();
      levels_.push_back({n, algebra_.upload(l.a), algebra_.upload(l.restriction),
                         algebra_.upload(l.prolongation), algebra_.upload(l.inverse),
                         algebra_.upload(l.inverse_diagonal), algebra_.make(n), algebra_.make(n),
                         algebra_.make(n), algebra_.make(n)});
    }
  }

  // Apply to the leading block; the caller preserves any remaining components.
  void apply(const Vector& rhs, Vector& result) {
    if (levels_.empty()) {
      return;
    }

    auto& l = levels_[0];
    algebra_.copy(rhs, l.rhs, l.n);
    cycle(0);
    algebra_.copy(l.x, result, l.n);
  }
};
}  // namespace cm::detail
