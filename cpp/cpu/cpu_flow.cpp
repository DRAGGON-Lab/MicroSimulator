#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cm/flow.hpp"
#include "core/flow_system.hpp"

namespace cm {
namespace {

struct ConjugateGradientResult {
  std::vector<double> solution;
  std::uint32_t iterations{0};
  double relative_residual{0.0};
};

[[nodiscard]] double dot_product(std::span<const double> left, std::span<const double> right) {
  if (left.size() != right.size()) {
    throw std::logic_error("conjugate-gradient vectors have inconsistent sizes");
  }
  double result = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    result += left[index] * right[index];
  }
  return result;
}

template <typename Apply>
[[nodiscard]] ConjugateGradientResult conjugate_gradient(
    Apply&& apply, std::span<const float> right_hand_side, std::span<const float> diagonal,
    float tolerance, std::uint32_t max_iterations, const char* label) {
  if (right_hand_side.size() != diagonal.size()) {
    throw std::logic_error(std::string(label) + " arrays have inconsistent sizes");
  }
  std::vector<double> solution(right_hand_side.size(), 0.0);
  std::vector<double> residual(right_hand_side.begin(), right_hand_side.end());
  const auto rhs_norm_squared = dot_product(residual, residual);
  if (rhs_norm_squared == 0.0) {
    return {.solution = std::move(solution)};
  }
  const auto rhs_norm = std::sqrt(rhs_norm_squared);
  std::vector<double> preconditioned(residual.size(), 0.0);
  for (std::size_t index = 0; index < residual.size(); ++index) {
    if (diagonal[index] > 0.0F) {
      preconditioned[index] = residual[index] / diagonal[index];
    }
  }
  auto direction = preconditioned;
  auto rho = dot_product(residual, preconditioned);
  auto relative = 1.0;
  for (std::uint32_t iteration = 1; iteration <= max_iterations; ++iteration) {
    std::vector<double> transformed;
    apply(direction, transformed);
    const auto curvature = dot_product(direction, transformed);
    if (!std::isfinite(curvature) || curvature <= 0.0) {
      throw std::runtime_error(std::string(label) +
                               " conjugate gradient encountered non-positive curvature");
    }
    const auto alpha = rho / curvature;
    for (std::size_t index = 0; index < solution.size(); ++index) {
      solution[index] += alpha * direction[index];
      residual[index] -= alpha * transformed[index];
    }
    relative = std::sqrt(dot_product(residual, residual)) / rhs_norm;
    if (!std::isfinite(relative)) {
      throw std::runtime_error(std::string(label) +
                               " conjugate gradient produced a non-finite residual");
    }
    if (relative <= tolerance) {
      return {
          .solution = std::move(solution), .iterations = iteration, .relative_residual = relative};
    }
    for (std::size_t index = 0; index < residual.size(); ++index) {
      preconditioned[index] = diagonal[index] > 0.0F ? residual[index] / diagonal[index] : 0.0;
    }
    const auto next_rho = dot_product(residual, preconditioned);
    if (!std::isfinite(next_rho) || rho == 0.0) {
      throw std::runtime_error(std::string(label) +
                               " conjugate gradient encountered a preconditioner breakdown");
    }
    const auto beta = next_rho / rho;
    for (std::size_t index = 0; index < direction.size(); ++index) {
      direction[index] = preconditioned[index] + beta * direction[index];
    }
    rho = next_rho;
  }
  throw std::runtime_error(std::string(label) + " conjugate gradient did not converge: relative " +
                           std::to_string(relative));
}

[[nodiscard]] std::array<const GridBoundary*, 2> axis_boundaries(const SignalGridSpec& spec,
                                                                 std::size_t axis) {
  const std::array<std::array<const GridBoundary*, 2>, 3> boundaries{{
      {&spec.x_lower, &spec.x_upper},
      {&spec.y_lower, &spec.y_upper},
      {&spec.z_lower, &spec.z_upper},
  }};
  return boundaries[axis];
}

void validate_relative_tolerance(float tolerance, const char* name) {
  if (!std::isfinite(tolerance) || tolerance < std::numeric_limits<float>::epsilon()) {
    throw std::invalid_argument(std::string(name) +
                                " must be finite and at least float machine epsilon");
  }
}

}  // namespace

void DepthAveragedFlowParameters::validate() const {
  if (!std::isfinite(mean_inlet_speed) || mean_inlet_speed == 0.0F) {
    throw std::invalid_argument("depth-averaged mean inlet speed must be finite and nonzero");
  }
  validate_relative_tolerance(relative_tolerance, "depth-averaged relative tolerance");
  if (max_iterations == 0) {
    throw std::invalid_argument("depth-averaged iteration limit must be positive");
  }
  switch (axis) {
    case FlowAxis::x:
    case FlowAxis::y:
    case FlowAxis::z:
      return;
  }
  throw std::invalid_argument("unknown depth-averaged flow axis");
}

void ResolvedFlowParameters::validate() const {
  if (!std::isfinite(mean_inlet_speed) || mean_inlet_speed == 0.0F) {
    throw std::invalid_argument("resolved-flow mean inlet speed must be finite and nonzero");
  }
  validate_relative_tolerance(relative_tolerance, "resolved-flow outer relative tolerance");
  validate_relative_tolerance(inner_relative_tolerance, "resolved-flow inner relative tolerance");
  if (max_outer_iterations == 0 || max_inner_iterations == 0) {
    throw std::invalid_argument("resolved-flow iteration limits must be positive");
  }
  switch (axis) {
    case FlowAxis::x:
    case FlowAxis::y:
    case FlowAxis::z:
      return;
  }
  throw std::invalid_argument("unknown resolved-flow axis");
}

void validate_flow_grid(const SignalGridSpec& spec, FlowAxis axis) {
  spec.validate();
  const auto axis_index = static_cast<std::size_t>(axis);
  if (axis_index >= 3) {
    throw std::invalid_argument("unknown flow axis");
  }
  for (std::size_t candidate = 0; candidate < 3; ++candidate) {
    const auto boundaries = axis_boundaries(spec, candidate);
    if (boundaries[0]->kind == GridBoundaryKind::periodic ||
        boundaries[1]->kind == GridBoundaryKind::periodic) {
      throw std::invalid_argument("native flow solvers do not support periodic boundaries");
    }
  }
  for (const auto* boundary : axis_boundaries(spec, axis_index)) {
    if (boundary->kind != GridBoundaryKind::fixed) {
      throw std::invalid_argument(
          "native flow-axis boundaries must be fixed to identify inlet and outlet");
    }
  }
}

DepthAveragedFlowResult solve_depth_averaged_flow_cpu(
    const SignalGridSpec& spec, std::span<const float> mobility,
    const DepthAveragedFlowParameters& parameters) {
  parameters.validate();
  const detail::DepthAveragedFlowSystem system(spec, mobility, parameters.axis);
  const auto solve =
      conjugate_gradient([&](std::span<const double> input,
                             std::vector<double>& output) { system.apply(input, output); },
                         system.right_hand_side(), system.diagonal(), parameters.relative_tolerance,
                         parameters.max_iterations, "depth-averaged flow");
  const auto unscaled = system.velocity(solve.solution);
  const auto scaled = detail::scale_velocity(
      spec, system.layout(), unscaled, system.open_inlet_faces(), parameters.mean_inlet_speed);
  return {
      .field = scaled.field,
      .report = {.iterations = solve.iterations,
                 .relative_residual = static_cast<float>(solve.relative_residual),
                 .mean_inlet_speed = parameters.mean_inlet_speed,
                 .max_speed = scaled.max_speed},
  };
}

ResolvedFlowResult solve_resolved_flow_cpu(const SignalGridSpec& spec, std::span<const float> drag,
                                           const ResolvedFlowParameters& parameters) {
  parameters.validate();
  const detail::ResolvedFlowSystem system(spec, drag, parameters.axis);
  std::uint64_t inner_iterations = 0;
  const auto solve_momentum = [&](std::span<const double> right_hand_side) {
    std::vector<float> rhs(right_hand_side.size());
    std::transform(right_hand_side.begin(), right_hand_side.end(), rhs.begin(),
                   [](double value) { return static_cast<float>(value); });
    const auto result = conjugate_gradient(
        [&](std::span<const double> input, std::vector<double>& output) {
          system.apply_momentum(input, output);
        },
        rhs, system.diagonal(), parameters.inner_relative_tolerance,
        parameters.max_inner_iterations, "resolved-flow momentum");
    if (inner_iterations > std::numeric_limits<std::uint64_t>::max() - result.iterations) {
      throw std::overflow_error("resolved-flow inner iteration count overflow");
    }
    inner_iterations += result.iterations;
    return result.solution;
  };

  std::vector<double> force(system.force().begin(), system.force().end());
  const auto particular = solve_momentum(force);
  auto schur_rhs_double = system.divergence(particular);
  std::vector<float> schur_rhs(schur_rhs_double.size());
  for (std::size_t index = 0; index < schur_rhs.size(); ++index) {
    schur_rhs[index] = static_cast<float>(-schur_rhs_double[index]);
  }
  const auto pressure_diagonal = system.pressure_diagonal();
  const auto pressure = conjugate_gradient(
      [&](std::span<const double> input, std::vector<double>& output) {
        const auto gradient = system.gradient(input);
        const auto response = solve_momentum(gradient);
        output = system.divergence(response);
        for (auto& value : output) {
          value = -value;
        }
      },
      schur_rhs, pressure_diagonal, parameters.relative_tolerance, parameters.max_outer_iterations,
      "resolved-flow pressure");

  const auto correction = solve_momentum(system.gradient(pressure.solution));
  std::vector<double> velocity(particular.size());
  for (std::size_t index = 0; index < velocity.size(); ++index) {
    velocity[index] = particular[index] - correction[index];
  }
  const auto divergence = system.divergence(velocity);
  double divergence_square_sum = 0.0;
  std::size_t fluid_count = 0;
  for (std::size_t site = 0; site < divergence.size(); ++site) {
    if (system.fluid()[site] != 0) {
      divergence_square_sum += divergence[site] * divergence[site];
      ++fluid_count;
    }
  }
  const auto divergence_rms =
      fluid_count == 0 ? 0.0 : std::sqrt(divergence_square_sum / static_cast<double>(fluid_count));
  std::vector<float> unscaled(velocity.size());
  std::transform(velocity.begin(), velocity.end(), unscaled.begin(),
                 [](double value) { return static_cast<float>(value); });
  const auto scaled = detail::scale_velocity(
      spec, system.layout(), unscaled, system.open_inlet_faces(), parameters.mean_inlet_speed);
  return {
      .field = scaled.field,
      .report = {.outer_iterations = pressure.iterations,
                 .inner_iterations = inner_iterations,
                 .divergence_rms = static_cast<float>(divergence_rms * std::abs(scaled.factor)),
                 .mean_inlet_speed = parameters.mean_inlet_speed,
                 .max_speed = scaled.max_speed,
                 .min_gap_voxels = system.minimum_gap_voxels()},
  };
}

}  // namespace cm
