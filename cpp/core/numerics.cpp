#include <numeric>

#include "numerics_ilu.hpp"
#include "numerics_linear.hpp"

namespace cm {
namespace {
void require(bool condition, const char* message) {
  if (!condition) {
    throw std::invalid_argument(message);
  }
}

bool positive(double x) {
  return std::isfinite(x) && x > 0;
}
}  // namespace

namespace detail {
double norm2(const std::vector<double>& x) {
  return std::sqrt(std::inner_product(x.begin(), x.end(), x.begin(), 0.0));
}

detail::FlexibleKrylovResult<std::vector<double>> solve_cpu(
    const SparseMatrix& matrix, const std::vector<double>& rhs,
    const LinearSolveParameters& parameters, const std::vector<detail::MultigridLevel>& hierarchy,
    bool constrained) {
  using Vector = std::vector<double>;
  detail::FlexibleKrylovOperations<Vector> op;
  op.make_zero = [&] {
    return Vector(rhs.size());
  };
  op.copy = [](const Vector& x, Vector& y) {
    y = x;
  };
  op.axpy = [](Vector& y, double a, const Vector& x) {
    for (std::size_t i = 0; i < x.size(); ++i) {
      y[i] += a * x[i];
    }
  };
  op.dot = [](const Vector& x, const Vector& y) {
    return std::inner_product(x.begin(), x.end(), y.begin(), 0.0);
  };
  op.apply = [&](const Vector& x, Vector& y) {
    matrix.apply(x, y);
  };
  detail::SparseMultigrid multigrid(detail::CpuSparseAlgebra{}, hierarchy);
  std::optional<detail::IncompleteLu> ilu;

  if (constrained) {
    ilu.emplace(matrix.offsets, matrix.columns, matrix.values);
  }

  op.precondition = [&](const Vector& x, Vector& y) {
    if (ilu) {
      ilu->apply(x, y);
    } else {
      y = x;
      multigrid.apply(x, y);
    }
  };
  const double n = norm2(rhs);
  const double tolerance =
      n == 0 ? parameters.relative_tolerance
             : std::max(parameters.relative_tolerance, parameters.absolute_tolerance / n);

  return detail::flexible_gmres(op, rhs, tolerance, parameters.max_iterations, 120);
}

detail::FlexibleKrylovResult<std::vector<double>> solve_gpu(
    detail::NumericsDevice& device, const SparseMatrix& matrix, const std::vector<double>& rhs,
    const LinearSolveParameters& parameters, const std::vector<detail::MultigridLevel>& hierarchy,
    bool constrained) {
  using namespace detail;
  using Vector = NumericsBufferPtr;
  const auto count = checked_index(rhs.size());
  const auto offsets = device.upload(matrix.offsets), columns = device.upload(matrix.columns);
  auto floats = [](const std::vector<double>& data) {
    std::vector<float> result(data.begin(), data.end());

    for (float x : result) {
      if (!std::isfinite(x)) {
        throw std::overflow_error("linear solver float32 representation overflow");
      }
    }

    return result;
  };
  const auto values = device.upload(floats(matrix.values)), input = device.upload(floats(rhs));
  constexpr std::uint32_t group_size = 128;
  const auto groups = (count + group_size - 1) / group_size;
  const auto reduction = device.allocate(std::size_t(groups) * sizeof(float));
  const auto row_sums = device.upload(floats(matrix.row_sums));
  FlexibleKrylovOperations<Vector> op;
  op.make_zero = [&] {
    auto v = device.allocate(std::size_t(count) * sizeof(float));
    device.dispatch(NumericsKernel::fill, {.count = count}, {v});

    return v;
  };
  op.copy = [&](const Vector& x, Vector& y) {
    device.dispatch(NumericsKernel::copy, {.count = count}, {x, y});
  };
  op.axpy = [&](Vector& y, double a, const Vector& x) {
    device.dispatch(NumericsKernel::axpy, {.count = count, .scalar = float(a)}, {x, y});
  };
  op.dot = [&](const Vector& x, const Vector& y) {
    device.dispatch(NumericsKernel::dot, {.count = count}, {x, y, reduction});
    auto partial = device.download<float>(reduction, groups);

    return std::accumulate(partial.begin(), partial.end(), 0.0);
  };
  op.apply = [&](const Vector& x, Vector& y) {
    if (matrix.symmetric) {
      device.dispatch(NumericsKernel::apply, {.count = count}, {offsets, columns, values, x, y});
    } else {
      device.dispatch(NumericsKernel::conservative_apply, {.count = count},
                      {offsets, columns, values, x, y, row_sums});
    }
  };
  SparseMultigrid multigrid(GpuSparseAlgebra{device}, hierarchy);
  GpuSparseAlgebra::Matrix factors{};
  NumericsBufferPtr diagonals;

  if (constrained) {
    const IncompleteLu ilu(matrix.offsets, matrix.columns, matrix.values);
    factors = GpuSparseAlgebra{device}.upload(ilu.factors);
    diagonals = device.upload(ilu.diagonals);
  }

  op.precondition = [&](const Vector& x, Vector& y) {
    if (constrained) {
      device.dispatch(NumericsKernel::ilu, {count, 0},
                      {factors.offsets, factors.columns, factors.values, diagonals, x, y});
    } else {
      op.copy(x, y);
      multigrid.apply(x, y);
    }
  };
  const double n = norm2(rhs);
  const double tolerance =
      n == 0 ? parameters.relative_tolerance
             : std::max(parameters.relative_tolerance, parameters.absolute_tolerance / n);
  auto result = flexible_gmres(op, input, tolerance, parameters.max_iterations, 120);
  const auto output = device.download<float>(result.solution, count);

  return {std::vector<double>(output.begin(), output.end()), result.iterations,
          result.relative_residual};
}
}  // namespace detail

void LinearSolveParameters::validate() const {
  require(positive(relative_tolerance) && relative_tolerance < 1 &&
              std::isfinite(absolute_tolerance) && absolute_tolerance >= 0,
          "invalid linear solver tolerances");
  require(max_iterations > 0 && memory_limit_bytes > 0, "invalid linear solver resource limits");
}

std::unique_ptr<detail::NumericsDevice> detail::make_numerics_device(BackendKind kind,
                                                                     std::uint32_t index) {
  switch (kind) {
    case BackendKind::cpu:
      require(index == 0, "CPU numerics exposes only device zero");
      return nullptr;
    case BackendKind::metal:
#ifdef CM_HAS_METAL
      return detail::make_metal_numerics_device(index);
#else
      throw std::runtime_error("Metal numerics was not built");
#endif
    case BackendKind::cuda:
#ifdef CM_HAS_CUDA
      return detail::make_cuda_numerics_device(index);
#else
      throw std::runtime_error("CUDA numerics was not built");
#endif
  }

  throw std::invalid_argument("unknown numerics backend");
}

namespace {
void validate_linear_solution(const std::vector<double>& solution) {
  for (std::size_t i = 0; i < solution.size(); ++i) {
    if (!std::isfinite(solution[i])) {
      throw std::runtime_error("nonfinite sparse linear solution");
    }
  }
}
}  // namespace

detail::NumericsLinearResult detail::solve_numerics_linear(
    NumericsDevice* device, NumericsLinearRows rows, std::vector<double> rhs,
    const LinearSolveParameters& parameters) {
  parameters.validate();
  require(rows.size() == rhs.size(), "sparse linear system dimensions mismatch");

  if (rhs.empty()) {
    return {};
  }

  require(rhs.size() * std::uint64_t{12288} <= parameters.memory_limit_bytes,
          "sparse linear system exceeds memory_limit_bytes");
  SparseMatrix matrix(rows, false);

  for (std::size_t i = 0; i < rhs.size(); ++i) {
    rhs[i] *= matrix.scale[i];
  }

  // GPU Krylov vectors use float32. Defect correction retains a double
  // authoritative solution and checks the original operator, while every
  // correction solve still executes on the selected native device.
  auto working = parameters;

  if (device) {
    working.relative_tolerance = std::max(working.relative_tolerance, 1e-4);
    working.absolute_tolerance = 0;
  }

  auto solved = device ? solve_gpu(*device, matrix, rhs, working, {}, true)
                       : solve_cpu(matrix, rhs, parameters, {}, true);
  const double norm = norm2(rhs);
  const double target =
      std::max(parameters.absolute_tolerance, parameters.relative_tolerance * norm);
  double absolute = INFINITY;

  for (unsigned refinement = 0; refinement < 8; ++refinement) {
    std::vector<double> residual(rhs.size());
    matrix.apply(solved.solution, residual);
    double scale = 0;

    for (std::size_t i = 0; i < rhs.size(); ++i) {
      residual[i] = rhs[i] - residual[i];
      scale = std::max(scale, std::abs(residual[i]));
    }

    absolute = norm2(residual);

    if (std::isfinite(absolute) && absolute <= 1.05 * target) {
      break;
    }

    if (!device || !positive(scale) || refinement == 7) {
      throw std::runtime_error("scalar linear true residual failed; candidate rejected");
    }

    for (double& r : residual) {
      r /= scale;
    }

    auto correction = solve_gpu(*device, matrix, residual, working, {}, true);
    solved.iterations += correction.iterations;

    for (std::size_t i = 0; i < rhs.size(); ++i) {
      solved.solution[i] += scale * correction.solution[i];
    }
  }

  NumericsLinearResult result{solved.solution, solved.iterations, norm == 0 ? 0 : absolute / norm,
                              absolute};

  validate_linear_solution(result.solution);

  return result;
}
}  // namespace cm
