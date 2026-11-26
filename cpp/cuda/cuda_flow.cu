#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/conjugate_gradient.hpp"
#include "core/flexible_gmres.hpp"
#include "core/flow_system.hpp"
#include "cuda_flow.cuh"
#include "kernels/flow.cuh"

namespace microsimulator::cuda {
namespace {

void check_cuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

void check_launch(const char* operation) {
  check_cuda(cudaGetLastError(), operation);
}

std::uint32_t checked_count(std::size_t count, const char* description) {
  if (count == 0 || count > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error(std::string("CUDA flow ") + description +
                              " must fit the nonzero uint32 index space");
  }
  return static_cast<std::uint32_t>(count);
}

FlowGridParameters make_grid_parameters(const detail::FlowGridLayout& layout) {
  const auto site_count = checked_count(layout.site_count(), "site count");
  const auto face_count = checked_count(layout.total_face_count(), "face count");
  const auto offsets = layout.face_offsets();
  const auto counts = layout.face_counts();
  for (const auto value : offsets) {
    static_cast<void>(checked_count(value == 0 ? 1 : value, "face offset"));
  }
  for (const auto value : counts) {
    static_cast<void>(checked_count(value, "component face count"));
  }
  const auto spacing = layout.spacing();
  FlowGridParameters result{};
  for (std::size_t axis = 0; axis < 3; ++axis) {
    result.dimensions[axis] = layout.dimensions()[axis];
    result.spacing[axis] = spacing[axis];
    result.inverse_spacing_squared[axis] = 1.0F / (spacing[axis] * spacing[axis]);
    result.face_offsets[axis] = static_cast<std::uint32_t>(offsets[axis]);
    result.face_counts[axis] = static_cast<std::uint32_t>(counts[axis]);
  }
  result.face_offsets[3] = face_count;
  result.face_counts[3] = face_count;
  result.site_count = site_count;
  result.total_face_count = face_count;
  return result;
}

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer(std::size_t count, const char* description) : count_(count) {
    if (count == 0 || count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::overflow_error(std::string("invalid CUDA flow buffer size for ") + description);
    }
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)), description);
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      static_cast<void>(cudaFree(data_));
    }
  }

  [[nodiscard]] T* data() noexcept {
    return data_;
  }
  [[nodiscard]] const T* data() const noexcept {
    return data_;
  }
  [[nodiscard]] std::size_t count() const noexcept {
    return count_;
  }

 private:
  T* data_{nullptr};
  std::size_t count_{0};
};

template <typename T>
void upload(DeviceBuffer<T>& destination, std::span<const T> source, cudaStream_t stream,
            const char* operation) {
  if (destination.count() != source.size()) {
    throw std::logic_error(std::string(operation) + ": buffer size mismatch");
  }
  check_cuda(cudaMemcpyAsync(destination.data(), source.data(), source.size_bytes(),
                             cudaMemcpyHostToDevice, stream),
             operation);
}

template <typename T>
std::vector<T> download(const DeviceBuffer<T>& source, cudaStream_t stream, const char* operation) {
  std::vector<T> result(source.count());
  check_cuda(cudaMemcpyAsync(result.data(), source.data(), result.size() * sizeof(T),
                             cudaMemcpyDeviceToHost, stream),
             operation);
  check_cuda(cudaStreamSynchronize(stream), operation);
  return result;
}

struct PcgWorkspace {
  explicit PcgWorkspace(std::uint32_t count)
      : residual(count, "failed to allocate CUDA flow residual"),
        preconditioned(count, "failed to allocate CUDA flow preconditioned residual"),
        direction(count, "failed to allocate CUDA flow direction"),
        transformed(count, "failed to allocate CUDA flow transformed vector"),
        partials((count + flow_reduction_width - 1) / flow_reduction_width,
                 "failed to allocate CUDA flow reduction partials"),
        host_partials(partials.count()) {
  }

  DeviceBuffer<float> residual;
  DeviceBuffer<float> preconditioned;
  DeviceBuffer<float> direction;
  DeviceBuffer<float> transformed;
  DeviceBuffer<float> partials;
  std::vector<float> host_partials;
};

struct PcgReport {
  std::uint32_t iterations{0};
  float relative_residual{0.0F};
};

double dot(const float* left, const float* right, std::uint32_t count, PcgWorkspace& workspace,
           cudaStream_t stream) {
  launch_flow_dot_partial(left, right, workspace.partials.data(), count, stream);
  check_launch("failed to launch CUDA flow reduction");
  check_cuda(cudaMemcpyAsync(workspace.host_partials.data(), workspace.partials.data(),
                             workspace.host_partials.size() * sizeof(float), cudaMemcpyDeviceToHost,
                             stream),
             "failed to download CUDA flow reduction");
  check_cuda(cudaStreamSynchronize(stream), "CUDA flow reduction failed");
  double result = 0.0;
  for (const auto value : workspace.host_partials) {
    result += value;
  }
  return result;
}

template <typename Apply>
PcgReport solve_pcg(const float* right_hand_side, const float* diagonal, float* solution,
                    PcgWorkspace& workspace, std::uint32_t count, float tolerance,
                    std::uint32_t max_iterations, const char* label, cudaStream_t stream,
                    Apply&& apply) {
  const detail::ConjugateGradientOperations operations{
      .initialize =
          [&] {
            launch_flow_pcg_initialize(right_hand_side, diagonal, solution,
                                       workspace.residual.data(), workspace.preconditioned.data(),
                                       workspace.direction.data(), count, stream);
            check_launch("failed to launch CUDA flow PCG initialization");
          },
      .residual_dot_residual =
          [&] {
            return dot(workspace.residual.data(), workspace.residual.data(), count, workspace,
                       stream);
          },
      .residual_dot_preconditioned =
          [&] {
            return dot(workspace.residual.data(), workspace.preconditioned.data(), count, workspace,
                       stream);
          },
      .apply_direction =
          [&] {
            apply(workspace.direction.data(), workspace.transformed.data());
            return dot(workspace.direction.data(), workspace.transformed.data(), count, workspace,
                       stream);
          },
      .update =
          [&](double alpha) {
            launch_flow_pcg_update(solution, workspace.residual.data(), workspace.direction.data(),
                                   workspace.transformed.data(), static_cast<float>(alpha), count,
                                   stream);
            check_launch("failed to launch CUDA flow PCG update");
          },
      .precondition =
          [&] {
            launch_flow_pcg_precondition(workspace.residual.data(), diagonal,
                                         workspace.preconditioned.data(), count, stream);
            check_launch("failed to launch CUDA flow PCG preconditioner");
          },
      .update_direction =
          [&](double beta) {
            launch_flow_pcg_direction(workspace.preconditioned.data(), workspace.direction.data(),
                                      static_cast<float>(beta), count, stream);
            check_launch("failed to launch CUDA flow PCG direction update");
          },
      .step_limit = std::numeric_limits<float>::max(),
  };
  const auto report =
      detail::preconditioned_conjugate_gradient(operations, tolerance, max_iterations, label);
  return {.iterations = report.iterations,
          .relative_residual = static_cast<float>(report.relative_residual)};
}

struct DepthPcgWorkspace {
  explicit DepthPcgWorkspace(std::uint32_t count)
      : residual(count, "CUDA depth residual"),
        preconditioned(count, "CUDA depth preconditioned residual"),
        direction(count, "CUDA depth direction"),
        transformed(count, "CUDA depth transformed vector"),
        partials((count + flow_reduction_width - 1) / flow_reduction_width,
                 "CUDA depth reduction partials"),
        folded((partials.count() + depth_fold_width - 1) / depth_fold_width,
               "CUDA depth folded partials"),
        scalars(depth_scalar_count, "CUDA depth solve scalars"),
        counters(depth_counter_count, "CUDA depth solve counters") {
  }

  DeviceBuffer<double> residual, preconditioned, direction, transformed, partials, folded;
  // The solve's scalars and counters, kept on the device so that the host
  // waits once for each batch of iterations.
  DeviceBuffer<double> scalars;
  DeviceBuffer<std::uint32_t> counters;
};

// Launches the dot product of two vectors, left among the solve's scalars at
// `place`.
void launch_depth_dot(const double* left, const double* right, std::uint32_t count,
                      DepthPcgWorkspace& workspace, std::uint32_t place, cudaStream_t stream) {
  launch_depth_flow_dot_partial(left, right, workspace.partials.data(), count, stream);
  check_launch("failed to launch CUDA depth reduction");
  launch_depth_flow_sum(workspace.partials.data(), workspace.folded.data(),
                        workspace.scalars.data(),
                        static_cast<std::uint32_t>(workspace.partials.count()), place, stream);
  check_launch("failed to launch CUDA depth reduction sum");
}

// Iterations of the depth-averaged solve launched before the host waits.
constexpr std::uint32_t depth_batch = 64;

// The depth-averaged pressure solve over a system's active sites.
PcgReport solve_depth_pcg(const double* right_hand_side, const float* diagonal,
                          const std::uint32_t* neighbors, const float* conductance,
                          const float* boundary_conductance, double* solution,
                          DepthPcgWorkspace& workspace, DepthFlowScales scales, std::uint32_t count,
                          float tolerance, std::uint32_t max_iterations, cudaStream_t stream) {
  auto* scalars = workspace.scalars.data();
  auto* counters = workspace.counters.data();
  const auto read_state = [&](const char* operation) {
    std::array<double, depth_scalar_count> scalar_values{};
    std::array<std::uint32_t, depth_counter_count> counter_values{};
    check_cuda(cudaMemcpyAsync(scalar_values.data(), scalars, sizeof(scalar_values),
                               cudaMemcpyDeviceToHost, stream),
               "failed to download CUDA depth solve scalars");
    check_cuda(cudaMemcpyAsync(counter_values.data(), counters, sizeof(counter_values),
                               cudaMemcpyDeviceToHost, stream),
               "failed to download CUDA depth solve counters");
    check_cuda(cudaStreamSynchronize(stream), operation);
    return std::pair{scalar_values, counter_values};
  };
  const detail::BatchedConjugateGradientOperations operations{
      .initialize =
          [&] {
            check_cuda(cudaMemsetAsync(scalars, 0, depth_scalar_count * sizeof(double), stream),
                       "failed to clear CUDA depth solve scalars");
            check_cuda(
                cudaMemsetAsync(counters, 0, depth_counter_count * sizeof(std::uint32_t), stream),
                "failed to clear CUDA depth solve counters");
            launch_depth_flow_pcg_initialize(
                right_hand_side, diagonal, solution, workspace.residual.data(),
                workspace.preconditioned.data(), workspace.direction.data(), count, stream);
            check_launch("failed to initialize CUDA depth PCG");
            launch_depth_dot(workspace.residual.data(), workspace.residual.data(), count, workspace,
                             depth_residual_squared, stream);
            launch_depth_dot(workspace.residual.data(), workspace.preconditioned.data(), count,
                             workspace, depth_rho, stream);
            return read_state("CUDA depth PCG initialization failed").first[depth_residual_squared];
          },
      .iterate =
          [&](std::uint32_t iterations, double threshold) {
            for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
              launch_depth_flow_operator_double(workspace.direction.data(), neighbors, conductance,
                                                boundary_conductance, workspace.transformed.data(),
                                                scales, count, stream);
              check_launch("failed to launch CUDA depth operator");
              launch_depth_dot(workspace.direction.data(), workspace.transformed.data(), count,
                               workspace, depth_curvature, stream);
              launch_depth_flow_step_length(scalars, counters, stream);
              check_launch("failed to launch CUDA depth step length");
              launch_depth_flow_pcg_update(solution, workspace.residual.data(),
                                           workspace.direction.data(), workspace.transformed.data(),
                                           scalars, count, stream);
              check_launch("failed to update CUDA depth PCG");
              launch_depth_dot(workspace.residual.data(), workspace.residual.data(), count,
                               workspace, depth_residual_squared, stream);
              launch_depth_flow_pcg_precondition(workspace.residual.data(), diagonal,
                                                 workspace.preconditioned.data(), count, stream);
              check_launch("failed to precondition CUDA depth residual");
              launch_depth_dot(workspace.residual.data(), workspace.preconditioned.data(), count,
                               workspace, depth_next_rho, stream);
              launch_depth_flow_step_weight(scalars, counters, threshold, stream);
              check_launch("failed to launch CUDA depth step weight");
              launch_depth_flow_pcg_direction(workspace.preconditioned.data(),
                                              workspace.direction.data(), scalars, count, stream);
              check_launch("failed to update CUDA depth direction");
            }
            const auto [scalar_values, counter_values] =
                read_state("CUDA depth PCG iteration failed");
            return detail::ConjugateGradientProgress{
                .iterations = counter_values[depth_iterations],
                .converged = counter_values[depth_converged] != 0,
                .failure =
                    static_cast<detail::ConjugateGradientFailure>(counter_values[depth_failure]),
                .residual_squared = scalar_values[depth_residual_squared],
            };
          },
  };
  const auto report = detail::batched_conjugate_gradient(operations, tolerance, max_iterations,
                                                         depth_batch, "CUDA depth-averaged flow");
  return {.iterations = report.iterations,
          .relative_residual = static_cast<float>(report.relative_residual)};
}

// Solves one depth-averaged pressure system and returns the velocity on every
// face of its grid.
std::vector<float> solve_depth_system(const detail::DepthAveragedFlowSystem& system,
                                      float tolerance, std::uint32_t max_iterations,
                                      PcgReport& report, cudaStream_t stream) {
  const auto grid = make_grid_parameters(system.layout());
  const auto& active = system.active();
  const auto count = checked_count(active.size(), "depth active site count");
  DeviceBuffer<float> mobility_buffer(grid.site_count, "failed to allocate CUDA depth mobility");
  DeviceBuffer<float> diagonal_buffer(count, "failed to allocate CUDA depth diagonal");
  DeviceBuffer<float> boundary_buffer(count, "failed to allocate CUDA depth boundary conductance");
  DeviceBuffer<std::uint32_t> site_buffer(count, "failed to allocate CUDA depth active sites");
  DeviceBuffer<std::uint32_t> neighbor_buffer(active.neighbors.size(),
                                              "failed to allocate CUDA depth neighbors");
  DeviceBuffer<float> conductance_buffer(active.conductance.size(),
                                         "failed to allocate CUDA depth conductance");
  DeviceBuffer<float> face_conductance_buffer(grid.total_face_count,
                                              "failed to allocate CUDA depth face conductance");
  DeviceBuffer<float> face_pressure_buffer(grid.total_face_count,
                                           "failed to allocate CUDA depth face pressure");
  DeviceBuffer<double> rhs_buffer(count, "failed to allocate CUDA depth right-hand side");
  DeviceBuffer<double> solution_buffer(count, "failed to allocate CUDA depth active pressure");
  DeviceBuffer<double> pressure_buffer(grid.site_count, "failed to allocate CUDA depth pressure");
  DeviceBuffer<float> velocity_buffer(grid.total_face_count,
                                      "failed to allocate CUDA depth velocity");
  upload(mobility_buffer, std::span<const float>(system.mobility()), stream,
         "failed to upload CUDA depth mobility");
  upload(diagonal_buffer, std::span<const float>(active.diagonal), stream,
         "failed to upload CUDA depth diagonal");
  upload(boundary_buffer, std::span<const float>(active.boundary_conductance), stream,
         "failed to upload CUDA depth boundary conductance");
  upload(site_buffer, std::span<const std::uint32_t>(active.sites), stream,
         "failed to upload CUDA depth active sites");
  upload(neighbor_buffer, std::span<const std::uint32_t>(active.neighbors), stream,
         "failed to upload CUDA depth neighbors");
  upload(conductance_buffer, std::span<const float>(active.conductance), stream,
         "failed to upload CUDA depth conductance");
  upload(face_conductance_buffer, std::span<const float>(system.face_conductance()), stream,
         "failed to upload CUDA depth face conductance");
  upload(face_pressure_buffer, std::span<const float>(system.face_pressure()), stream,
         "failed to upload CUDA depth face pressure");
  const std::vector<double> rhs(active.right_hand_side.begin(), active.right_hand_side.end());
  upload(rhs_buffer, std::span<const double>(rhs), stream,
         "failed to upload CUDA depth right-hand side");
  const auto spacing = system.layout().spacing();
  DepthFlowScales scales{};
  for (std::size_t axis = 0; axis < 3; ++axis) {
    scales.inverse_square[axis] = 1.0 / (static_cast<double>(spacing[axis]) * spacing[axis]);
  }
  DepthPcgWorkspace workspace(count);
  report =
      solve_depth_pcg(rhs_buffer.data(), diagonal_buffer.data(), neighbor_buffer.data(),
                      conductance_buffer.data(), boundary_buffer.data(), solution_buffer.data(),
                      workspace, scales, count, tolerance, max_iterations, stream);
  // The velocity kernel reads a pressure at every site of the grid; it is
  // zero at a site that is not active.
  check_cuda(cudaMemsetAsync(pressure_buffer.data(), 0, grid.site_count * sizeof(double), stream),
             "failed to clear CUDA depth pressure");
  launch_depth_flow_scatter(solution_buffer.data(), site_buffer.data(), pressure_buffer.data(),
                            count, stream);
  check_launch("failed to launch CUDA depth pressure scatter");
  launch_depth_flow_velocity_double(pressure_buffer.data(), mobility_buffer.data(),
                                    face_conductance_buffer.data(), face_pressure_buffer.data(),
                                    velocity_buffer.data(), grid, stream);
  check_launch("failed to launch CUDA depth-averaged velocity reconstruction");
  return download(velocity_buffer, stream, "failed to download CUDA depth velocity");
}

}  // namespace

DepthAveragedPortFlowResult solve_depth_averaged_port_flow(
    const SignalGridSpec& spec, std::span<const FlowPort> ports, std::span<const float> mobility,
    const DepthAveragedPortFlowParameters& parameters, cudaStream_t stream) {
  parameters.validate();
  const detail::ShallowPortFlow flow(spec, ports, mobility);
  DepthAveragedPortFlowReport report;
  const auto solution = flow.solve([&](const detail::DepthAveragedFlowSystem& system) {
    PcgReport solve;
    auto velocity = solve_depth_system(system, parameters.relative_tolerance,
                                       parameters.max_iterations, solve, stream);
    report.iterations += solve.iterations;
    report.relative_residual = std::max(report.relative_residual, solve.relative_residual);
    return velocity;
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

DepthAveragedFlowResult solve_depth_averaged_flow(const SignalGridSpec& spec,
                                                  std::span<const float> mobility,
                                                  const DepthAveragedFlowParameters& parameters,
                                                  cudaStream_t stream) {
  return detail::solve_shallow_axis_flow(
      spec, mobility, parameters,
      [&](const detail::DepthAveragedFlowSystem& system, DepthAveragedFlowReport& report) {
        PcgReport solve;
        auto flux = solve_depth_system(system, parameters.relative_tolerance,
                                       parameters.max_iterations, solve, stream);
        report.iterations += solve.iterations;
        report.relative_residual = std::max(report.relative_residual, solve.relative_residual);
        return flux;
      });
}

namespace {

// Solves one resolved block system and returns its velocity on every face.
detail::ResolvedSystemSolve solve_resolved_system(const detail::ResolvedFlowSystem& system,
                                                  const detail::ResolvedSolveControls& parameters,
                                                  cudaStream_t stream) {
  const auto grid = make_grid_parameters(system.layout());
  DeviceBuffer<std::uint8_t> fluid_buffer(grid.site_count, "failed to allocate CUDA fluid mask");
  DeviceBuffer<std::uint8_t> active_buffer(grid.total_face_count,
                                           "failed to allocate CUDA active face mask");
  DeviceBuffer<std::uint8_t> exists_buffer(grid.total_face_count,
                                           "failed to allocate CUDA face existence mask");
  DeviceBuffer<float> face_drag_buffer(grid.total_face_count, "failed to allocate CUDA face drag");
  DeviceBuffer<float> face_diagonal_buffer(grid.total_face_count,
                                           "failed to allocate CUDA momentum diagonal");
  DeviceBuffer<float> force_buffer(grid.total_face_count, "failed to allocate CUDA momentum force");
  const auto pressure_diagonal = system.pressure_diagonal();
  DeviceBuffer<float> pressure_diagonal_buffer(grid.site_count,
                                               "failed to allocate CUDA pressure diagonal");
  upload(fluid_buffer, std::span<const std::uint8_t>(system.fluid()), stream,
         "failed to upload CUDA fluid mask");
  upload(active_buffer, std::span<const std::uint8_t>(system.active()), stream,
         "failed to upload CUDA active face mask");
  upload(exists_buffer, std::span<const std::uint8_t>(system.exists()), stream,
         "failed to upload CUDA face existence mask");
  upload(face_drag_buffer, std::span<const float>(system.face_drag()), stream,
         "failed to upload CUDA face drag");
  upload(face_diagonal_buffer, std::span<const float>(system.diagonal()), stream,
         "failed to upload CUDA momentum diagonal");
  upload(force_buffer, std::span<const float>(system.force()), stream,
         "failed to upload CUDA momentum force");
  upload(pressure_diagonal_buffer, std::span<const float>(pressure_diagonal), stream,
         "failed to upload CUDA pressure diagonal");

  DeviceBuffer<float> gradient(grid.total_face_count, "CUDA block gradient");
  PcgWorkspace inner_workspace(grid.total_face_count), outer_workspace(grid.site_count);
  struct Block {
    DeviceBuffer<float> u, p;
    Block(std::size_t nu, std::size_t np)
        : u(nu, "CUDA Krylov velocity"), p(np, "CUDA Krylov pressure") {
    }
  };
  using Vector = std::shared_ptr<Block>;
  const double continuity_scale =
      1.0 / *std::min_element(system.layout().spacing().begin(), system.layout().spacing().end());
  const auto combine = [&](const float* source, float* target, double alpha, float beta,
                           std::uint32_t count) {
    if (!std::isfinite(alpha) || std::abs(alpha) > std::numeric_limits<float>::max())
      throw std::runtime_error("non-finite CUDA Krylov coefficient");
    launch_flow_vector_combine(source, target, static_cast<float>(alpha), beta, count, stream);
    check_launch("CUDA Krylov vector update");
  };
  detail::FlexibleKrylovOperations<Vector> ops;
  ops.make_zero = [&] {
    auto value = std::make_shared<Block>(grid.total_face_count, grid.site_count);
    check_cuda(cudaMemsetAsync(value->u.data(), 0, grid.total_face_count * sizeof(float), stream),
               "zero CUDA Krylov velocity");
    check_cuda(cudaMemsetAsync(value->p.data(), 0, grid.site_count * sizeof(float), stream),
               "zero CUDA Krylov pressure");
    return value;
  };
  ops.copy = [&](const Vector& source, Vector& target) {
    combine(source->u.data(), target->u.data(), 1, 0, grid.total_face_count);
    combine(source->p.data(), target->p.data(), 1, 0, grid.site_count);
  };
  ops.axpy = [&](Vector& target, double alpha, const Vector& source) {
    combine(source->u.data(), target->u.data(), alpha, 1, grid.total_face_count);
    combine(source->p.data(), target->p.data(), alpha, 1, grid.site_count);
  };
  ops.dot = [&](const Vector& a, const Vector& b) {
    return dot(a->u.data(), b->u.data(), grid.total_face_count, inner_workspace, stream) +
           dot(a->p.data(), b->p.data(), grid.site_count, outer_workspace, stream);
  };
  ops.apply = [&](const Vector& input, Vector& output) {
    launch_resolved_flow_momentum(input->u.data(), active_buffer.data(), exists_buffer.data(),
                                  face_drag_buffer.data(), output->u.data(), grid, stream);
    launch_resolved_flow_gradient(input->p.data(), fluid_buffer.data(), active_buffer.data(),
                                  gradient.data(), grid, stream);
    combine(gradient.data(), output->u.data(), 1, 1, grid.total_face_count);
    launch_resolved_flow_divergence(input->u.data(), fluid_buffer.data(), output->p.data(), grid,
                                    stream);
    combine(output->p.data(), output->p.data(), continuity_scale, 0, grid.site_count);
  };
  std::uint64_t inner_iterations = 0;
  ops.precondition = [&](const Vector& input, Vector& output) {
    const auto report = solve_pcg(
        input->u.data(), face_diagonal_buffer.data(), output->u.data(), inner_workspace,
        grid.total_face_count, parameters.inner_relative_tolerance, parameters.max_inner_iterations,
        "CUDA momentum preconditioner", stream, [&](const float* x, float* y) {
          launch_resolved_flow_momentum(x, active_buffer.data(), exists_buffer.data(),
                                        face_drag_buffer.data(), y, grid, stream);
          check_launch("CUDA preconditioner momentum");
        });
    inner_iterations += report.iterations;
    launch_flow_pcg_precondition(input->p.data(), pressure_diagonal_buffer.data(), output->p.data(),
                                 grid.site_count, stream);
    combine(output->p.data(), output->p.data(), -1 / continuity_scale, 0, grid.site_count);
  };
  auto rhs = ops.make_zero();
  combine(force_buffer.data(), rhs->u.data(), 1, 0, grid.total_face_count);
  const auto solution = detail::flexible_gmres(ops, rhs, parameters.relative_tolerance,
                                               parameters.max_outer_iterations);
  auto residual = ops.make_zero();
  ops.apply(solution.solution, residual);
  ops.axpy(residual, -1, rhs);
  const double momentum_square =
      dot(residual->u.data(), residual->u.data(), grid.total_face_count, inner_workspace, stream);
  const double force_square =
      dot(rhs->u.data(), rhs->u.data(), grid.total_face_count, inner_workspace, stream);
  const double divergence_square =
      dot(residual->p.data(), residual->p.data(), grid.site_count, outer_workspace, stream);
  const auto fluid_count =
      std::count(system.fluid().begin(), system.fluid().end(), std::uint8_t{1});
  const double divergence_rms =
      fluid_count == 0
          ? 0.0
          : std::sqrt(divergence_square / static_cast<double>(fluid_count)) / continuity_scale;
  return {
      .velocity = download(solution.solution->u, stream, "download CUDA velocity"),
      .outer_iterations = solution.iterations,
      .inner_iterations = inner_iterations,
      .relative_residual = static_cast<float>(solution.relative_residual),
      .momentum_relative_residual = static_cast<float>(std::sqrt(momentum_square / force_square)),
      .divergence_rms = divergence_rms,
  };
}

}  // namespace

ResolvedFlowResult solve_resolved_flow(const SignalGridSpec& spec, std::span<const float> drag,
                                       const ResolvedFlowParameters& parameters,
                                       cudaStream_t stream) {
  return detail::solve_resolved_axis_flow(
      spec, drag, parameters,
      [&](const detail::ResolvedFlowSystem& system, const detail::ResolvedSolveControls& controls) {
        return solve_resolved_system(system, controls, stream);
      });
}

ResolvedPortFlowResult solve_resolved_port_flow(const SignalGridSpec& spec,
                                                std::span<const FlowPort> ports,
                                                std::span<const float> drag,
                                                const ResolvedPortFlowParameters& parameters,
                                                cudaStream_t stream) {
  return detail::solve_resolved_port_flow(
      spec, ports, drag, parameters,
      [&](const detail::ResolvedFlowSystem& system, const detail::ResolvedSolveControls& controls) {
        return solve_resolved_system(system, controls, stream);
      });
}

}  // namespace microsimulator::cuda
