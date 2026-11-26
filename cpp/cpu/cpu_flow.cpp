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

#include "core/conjugate_gradient.hpp"
#include "core/flexible_gmres.hpp"
#include "core/flow_system.hpp"
#include "microsimulator/flow.hpp"

namespace microsimulator {
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
  return detail::parallel_reduce(
      left.size(), detail::flow_grain, 0.0,
      [&](std::size_t begin, std::size_t end) {
        double piece = 0.0;
        for (std::size_t index = begin; index < end; ++index) {
          piece += left[index] * right[index];
        }
        return piece;
      },
      [](double first, double second) { return first + second; });
}

// Calls body(index) for every index below `count`.
template <typename Body>
void for_each_entry(std::size_t count, const Body& body) {
  detail::parallel_for(count, detail::flow_grain, [&](std::size_t begin, std::size_t end) {
    for (std::size_t index = begin; index < end; ++index) {
      body(index);
    }
  });
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
  std::vector<double> preconditioned(residual.size(), 0.0);
  std::vector<double> direction;
  std::vector<double> transformed;
  const auto precondition = [&] {
    for_each_entry(residual.size(), [&](std::size_t index) {
      preconditioned[index] = diagonal[index] > 0.0F ? residual[index] / diagonal[index] : 0.0;
    });
  };
  const detail::ConjugateGradientOperations operations{
      .initialize =
          [&] {
            precondition();
            direction = preconditioned;
          },
      .residual_dot_residual = [&] { return dot_product(residual, residual); },
      .residual_dot_preconditioned = [&] { return dot_product(residual, preconditioned); },
      .apply_direction =
          [&] {
            apply(direction, transformed);
            return dot_product(direction, transformed);
          },
      .update =
          [&](double alpha) {
            for_each_entry(solution.size(), [&](std::size_t index) {
              solution[index] += alpha * direction[index];
              residual[index] -= alpha * transformed[index];
            });
          },
      .precondition = precondition,
      .update_direction =
          [&](double beta) {
            for_each_entry(direction.size(), [&](std::size_t index) {
              direction[index] = preconditioned[index] + beta * direction[index];
            });
          },
  };
  const auto report =
      detail::preconditioned_conjugate_gradient(operations, tolerance, max_iterations, label);
  return {.solution = std::move(solution),
          .iterations = report.iterations,
          .relative_residual = report.relative_residual};
}

// Solves a depth-averaged pressure system for the pressures of its active
// sites.
[[nodiscard]] ConjugateGradientResult solve_active_depth_system(
    const detail::DepthAveragedFlowSystem& system, float tolerance, std::uint32_t max_iterations) {
  return conjugate_gradient([&](std::span<const double> input,
                                std::vector<double>& output) { system.apply(input, output); },
                            system.active().right_hand_side, system.active().diagonal, tolerance,
                            max_iterations, "depth-averaged flow");
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
  if (!std::isfinite(mean_inlet_speed) || (mean_inlet_speed == 0.0F && body_force.empty())) {
    throw std::invalid_argument(
        "depth-averaged mean inlet speed must be finite and nonzero without a body force");
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

namespace detail {

std::vector<double> solve_depth_averaged_pressure(const DepthAveragedFlowSystem& system,
                                                  float tolerance, std::uint32_t max_iterations) {
  return system.scatter(solve_active_depth_system(system, tolerance, max_iterations).solution);
}

}  // namespace detail

void ResolvedPortFlowParameters::validate() const {
  validate_relative_tolerance(relative_tolerance, "resolved-flow outer relative tolerance");
  validate_relative_tolerance(inner_relative_tolerance, "resolved-flow inner relative tolerance");
  if (max_outer_iterations == 0 || max_inner_iterations == 0) {
    throw std::invalid_argument("resolved-flow iteration limits must be positive");
  }
}

void DepthAveragedPortFlowParameters::validate() const {
  validate_relative_tolerance(relative_tolerance, "depth-averaged relative tolerance");
  if (max_iterations == 0) {
    throw std::invalid_argument("depth-averaged iteration limit must be positive");
  }
}

void validate_flow_ports(const SignalGridSpec& spec, std::span<const FlowPort> ports) {
  spec.validate();
  for (std::size_t axis = 0; axis < 3; ++axis) {
    const auto boundaries = axis_boundaries(spec, axis);
    if (boundaries[0]->kind == GridBoundaryKind::periodic ||
        boundaries[1]->kind == GridBoundaryKind::periodic) {
      throw std::invalid_argument("native flow solvers do not support periodic boundaries");
    }
  }
  if (ports.empty()) {
    throw std::invalid_argument("flow needs at least one port");
  }
  const detail::FlowGridLayout layout(spec);
  std::vector<std::uint8_t> claimed(layout.total_face_count(), 0);
  std::vector<std::uint8_t> visited(layout.site_count(), 0);
  std::vector<std::size_t> pending;
  bool holds_pressure = false;
  for (const auto& port : ports) {
    if (port.kind != FlowPortKind::speed && port.kind != FlowPortKind::pressure) {
      throw std::invalid_argument("unknown flow port kind");
    }
    if (!std::isfinite(port.value)) {
      throw std::invalid_argument("flow port values must be finite");
    }
    static_cast<void>(detail::port_face_geometry(port.face));
    if (spec.boundary(port.face).kind != GridBoundaryKind::fixed) {
      throw std::invalid_argument("a flow port must open a fixed boundary");
    }
    const auto extent = spec.face_extent(port.face);
    if (port.patch.end[0] <= port.patch.begin[0] || port.patch.end[1] <= port.patch.begin[1] ||
        port.patch.end[0] > extent[0] || port.patch.end[1] > extent[1]) {
      throw std::invalid_argument("a flow port must span at least one site of its face");
    }
    bool open = false;
    detail::for_each_port_face(layout, port, [&](std::size_t face, std::size_t site) {
      if (claimed[face] != 0) {
        throw std::invalid_argument("flow ports must not overlap");
      }
      claimed[face] = 1;
      if (spec.solid_site(site)) {
        return;
      }
      open = true;
      if (port.kind == FlowPortKind::pressure && visited[site] == 0) {
        visited[site] = 1;
        pending.push_back(site);
      }
    });
    if (!open) {
      throw std::invalid_argument("a flow port is entirely blocked");
    }
    holds_pressure = holds_pressure || port.kind == FlowPortKind::pressure;
  }
  if (!holds_pressure) {
    throw std::invalid_argument("flow needs at least one pressure port");
  }
  const auto drives = [&](const FlowPort& port) {
    return port.kind == FlowPortKind::speed ||
           port.value != std::find_if(ports.begin(), ports.end(), [](const FlowPort& other) {
                           return other.kind == FlowPortKind::pressure;
                         })->value;
  };
  if (std::none_of(ports.begin(), ports.end(), drives)) {
    throw std::invalid_argument(
        "flow ports state no speed and hold one pressure, so nothing flows");
  }
  // A speed port's flow has to leave, or arrive, through a pressure port.
  while (!pending.empty()) {
    const auto site = pending.back();
    pending.pop_back();
    for (std::size_t component = 0; component < 3; ++component) {
      for (const auto offset : {-1, 1}) {
        const auto neighbor = layout.neighbor_site(site, component, offset);
        if (neighbor && visited[*neighbor] == 0 && !spec.solid_site(*neighbor)) {
          visited[*neighbor] = 1;
          pending.push_back(*neighbor);
        }
      }
    }
  }
  for (const auto& port : ports) {
    if (port.kind != FlowPortKind::speed) {
      continue;
    }
    bool reached = false;
    detail::for_each_port_face(layout, port, [&](std::size_t, std::size_t site) {
      reached = reached || visited[site] != 0;
    });
    if (!reached) {
      throw std::runtime_error("a speed port carries no flow: no pressure port is reachable");
    }
  }
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
  const detail::FlowGridLayout layout(spec);
  std::vector<std::uint8_t> visited(layout.site_count(), 0);
  std::vector<std::size_t> pending;
  for (std::size_t i = 0; i < layout.site_count(); ++i) {
    if (!spec.solid_site(i) && layout.site_coordinates(i)[axis_index] == 0) {
      visited[i] = 1;
      pending.push_back(i);
    }
  }
  if (pending.empty()) throw std::invalid_argument("the flow inlet boundary is entirely blocked");
  bool reachable = false;
  while (!pending.empty()) {
    const auto site = pending.back();
    pending.pop_back();
    if (layout.site_coordinates(site)[axis_index] + 1 == layout.dimensions()[axis_index]) {
      reachable = true;
      break;
    }
    for (std::size_t component = 0; component < 3; ++component)
      for (const auto offset : {-1, 1}) {
        const auto neighbor = layout.neighbor_site(site, component, offset);
        if (neighbor && !visited[*neighbor] && !spec.solid_site(*neighbor)) {
          visited[*neighbor] = 1;
          pending.push_back(*neighbor);
        }
      }
  }
  if (!reachable)
    throw std::runtime_error("the device carries no through-flow: the outlet is unreachable");
}

DepthAveragedFlowResult solve_depth_averaged_flow_cpu(
    const SignalGridSpec& spec, std::span<const float> mobility,
    const DepthAveragedFlowParameters& parameters) {
  return detail::solve_shallow_axis_flow(
      spec, mobility, parameters,
      [&](const detail::DepthAveragedFlowSystem& system, DepthAveragedFlowReport& report) {
        const auto solve = solve_active_depth_system(system, parameters.relative_tolerance,
                                                     parameters.max_iterations);
        report.iterations += solve.iterations;
        report.relative_residual =
            std::max(report.relative_residual, static_cast<float>(solve.relative_residual));
        return system.velocity(system.scatter(solve.solution));
      });
}

DepthAveragedPortFlowResult solve_depth_averaged_port_flow_cpu(
    const SignalGridSpec& spec, std::span<const FlowPort> ports, std::span<const float> mobility,
    const DepthAveragedPortFlowParameters& parameters) {
  parameters.validate();
  const detail::ShallowPortFlow flow(spec, ports, mobility);
  DepthAveragedPortFlowReport report;
  const auto solution = flow.solve([&](const detail::DepthAveragedFlowSystem& system) {
    const auto solve =
        solve_active_depth_system(system, parameters.relative_tolerance, parameters.max_iterations);
    report.iterations += solve.iterations;
    report.relative_residual =
        std::max(report.relative_residual, static_cast<float>(solve.relative_residual));
    return system.velocity(system.scatter(solve.solution));
  });
  report.solves = solution.solves;
  report.max_speed = solution.max_speed;
  return {
      .field =
          detail::make_velocity_field(spec, flow.reduction().original_layout(), solution.velocity),
      .report = report,
      .ports = solution.ports,
  };
}

namespace {

[[nodiscard]] detail::ResolvedSystemSolve solve_resolved_system(
    const detail::ResolvedFlowSystem& system, const detail::ResolvedSolveControls& parameters) {
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

  const double continuity_scale =
      1.0 / *std::min_element(system.layout().spacing().begin(), system.layout().spacing().end());
  const auto nu = system.layout().total_face_count();
  const auto np = system.layout().site_count();
  const auto pressure_diagonal = system.pressure_diagonal();
  using Vector = std::vector<double>;
  detail::FlexibleKrylovOperations<Vector> ops;
  ops.make_zero = [&] { return Vector(nu + np, 0.0); };
  ops.copy = [](const Vector& source, Vector& target) { target = source; };
  ops.axpy = [](Vector& target, double alpha, const Vector& source) {
    for_each_entry(target.size(), [&](std::size_t i) { target[i] += alpha * source[i]; });
  };
  ops.dot = [](const Vector& a, const Vector& b) { return dot_product(a, b); };
  ops.apply = [&](const Vector& input, Vector& output) {
    const auto u = std::span<const double>(input).first(nu);
    const auto p = std::span<const double>(input).subspan(nu);
    Vector momentum;
    system.apply_momentum(u, momentum);
    const auto gradient = system.gradient(p);
    const auto divergence = system.divergence(u);
    for_each_entry(nu, [&](std::size_t i) { output[i] = momentum[i] + gradient[i]; });
    for_each_entry(np, [&](std::size_t i) { output[nu + i] = continuity_scale * divergence[i]; });
  };
  ops.precondition = [&](const Vector& input, Vector& output) {
    const auto u = solve_momentum(std::span<const double>(input).first(nu));
    std::copy(u.begin(), u.end(), output.begin());
    for_each_entry(np, [&](std::size_t i) {
      output[nu + i] =
          pressure_diagonal[i] > 0 ? -input[nu + i] / (continuity_scale * pressure_diagonal[i]) : 0;
    });
  };
  auto rhs = ops.make_zero();
  std::copy(system.force().begin(), system.force().end(), rhs.begin());
  const auto solution = detail::flexible_gmres(ops, rhs, parameters.relative_tolerance,
                                               parameters.max_outer_iterations);
  Vector velocity(solution.solution.begin(),
                  solution.solution.begin() + static_cast<std::ptrdiff_t>(nu));
  Vector residual = ops.make_zero();
  ops.apply(solution.solution, residual);
  double momentum_square = 0.0, force_square = 0.0;
  for (std::size_t i = 0; i < nu; ++i) {
    momentum_square += (residual[i] - rhs[i]) * (residual[i] - rhs[i]);
    force_square += rhs[i] * rhs[i];
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
  return {
      .velocity = std::move(unscaled),
      .outer_iterations = solution.iterations,
      .inner_iterations = inner_iterations,
      .relative_residual = static_cast<float>(solution.relative_residual),
      .momentum_relative_residual = static_cast<float>(std::sqrt(momentum_square / force_square)),
      .divergence_rms = divergence_rms,
  };
}

}  // namespace

ResolvedFlowResult solve_resolved_flow_cpu(const SignalGridSpec& spec, std::span<const float> drag,
                                           const ResolvedFlowParameters& parameters) {
  return detail::solve_resolved_axis_flow(spec, drag, parameters, solve_resolved_system);
}

ResolvedPortFlowResult solve_resolved_port_flow_cpu(const SignalGridSpec& spec,
                                                    std::span<const FlowPort> ports,
                                                    std::span<const float> drag,
                                                    const ResolvedPortFlowParameters& parameters) {
  return detail::solve_resolved_port_flow(spec, ports, drag, parameters, solve_resolved_system);
}

}  // namespace microsimulator
