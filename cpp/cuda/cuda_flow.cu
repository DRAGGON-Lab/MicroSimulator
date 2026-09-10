#include <cuda_runtime.h>

#include <algorithm>
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

#include "core/flexible_gmres.hpp"
#include "core/flow_system.hpp"
#include "cuda_flow.cuh"
#include "kernels/flow.cuh"

namespace cm::cuda {
namespace {

void check_cuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

void check_launch(const char* operation) { check_cuda(cudaGetLastError(), operation); }

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
  result.flow_axis = static_cast<std::uint32_t>(layout.flow_axis());
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

  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t count() const noexcept { return count_; }

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
        host_partials(partials.count()) {}

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
  launch_flow_pcg_initialize(right_hand_side, diagonal, solution, workspace.residual.data(),
                             workspace.preconditioned.data(), workspace.direction.data(), count,
                             stream);
  check_launch("failed to launch CUDA flow PCG initialization");
  const auto rhs_norm_squared = dot(right_hand_side, right_hand_side, count, workspace, stream);
  if (rhs_norm_squared == 0.0) {
    return {};
  }
  const auto rhs_norm = std::sqrt(rhs_norm_squared);
  auto rho =
      dot(workspace.residual.data(), workspace.preconditioned.data(), count, workspace, stream);
  auto relative = 1.0;
  for (std::uint32_t iteration = 1; iteration <= max_iterations; ++iteration) {
    apply(workspace.direction.data(), workspace.transformed.data());
    const auto curvature =
        dot(workspace.direction.data(), workspace.transformed.data(), count, workspace, stream);
    if (!std::isfinite(curvature) || curvature <= 0.0) {
      throw std::runtime_error(std::string(label) +
                               " conjugate gradient encountered non-positive curvature");
    }
    const auto alpha_double = rho / curvature;
    if (!std::isfinite(alpha_double) ||
        std::abs(alpha_double) > std::numeric_limits<float>::max()) {
      throw std::runtime_error(std::string(label) +
                               " conjugate gradient produced a non-finite step");
    }
    const auto alpha = static_cast<float>(alpha_double);
    launch_flow_pcg_update(solution, workspace.residual.data(), workspace.direction.data(),
                           workspace.transformed.data(), alpha, count, stream);
    check_launch("failed to launch CUDA flow PCG update");
    const auto residual_squared =
        dot(workspace.residual.data(), workspace.residual.data(), count, workspace, stream);
    relative = std::sqrt(std::max(0.0, residual_squared)) / rhs_norm;
    if (!std::isfinite(relative)) {
      throw std::runtime_error(std::string(label) +
                               " conjugate gradient produced a non-finite residual");
    }
    if (relative <= tolerance) {
      return {.iterations = iteration, .relative_residual = static_cast<float>(relative)};
    }
    launch_flow_pcg_precondition(workspace.residual.data(), diagonal,
                                 workspace.preconditioned.data(), count, stream);
    check_launch("failed to launch CUDA flow PCG preconditioner");
    const auto next_rho =
        dot(workspace.residual.data(), workspace.preconditioned.data(), count, workspace, stream);
    if (!std::isfinite(next_rho) || rho == 0.0) {
      throw std::runtime_error(std::string(label) +
                               " conjugate gradient encountered a preconditioner breakdown");
    }
    const auto beta_double = next_rho / rho;
    if (!std::isfinite(beta_double) || std::abs(beta_double) > std::numeric_limits<float>::max()) {
      throw std::runtime_error(std::string(label) +
                               " conjugate gradient produced a non-finite direction");
    }
    launch_flow_pcg_direction(workspace.preconditioned.data(), workspace.direction.data(),
                              static_cast<float>(beta_double), count, stream);
    check_launch("failed to launch CUDA flow PCG direction update");
    rho = next_rho;
  }
  throw std::runtime_error(std::string(label) + " conjugate gradient did not converge: relative " +
                           std::to_string(relative));
}

}  // namespace

DepthAveragedFlowResult solve_depth_averaged_flow(const SignalGridSpec& spec,
                                                  std::span<const float> mobility,
                                                  const DepthAveragedFlowParameters& parameters,
                                                  cudaStream_t stream) {
  parameters.validate();
  const detail::ShallowFlowReduction reduction(spec, mobility, parameters.axis);
  const detail::DepthAveragedFlowSystem system(reduction.grid(), reduction.conductance(),
                                               parameters.axis);
  const auto grid = make_grid_parameters(system.layout());
  DeviceBuffer<float> mobility_buffer(grid.site_count, "failed to allocate CUDA depth mobility");
  DeviceBuffer<float> diagonal_buffer(grid.site_count, "failed to allocate CUDA depth diagonal");
  DeviceBuffer<float> rhs_buffer(grid.site_count, "failed to allocate CUDA depth right-hand side");
  DeviceBuffer<float> pressure_buffer(grid.site_count, "failed to allocate CUDA depth pressure");
  DeviceBuffer<float> velocity_buffer(grid.total_face_count,
                                      "failed to allocate CUDA depth velocity");
  upload(mobility_buffer, std::span<const float>(system.mobility()), stream,
         "failed to upload CUDA depth mobility");
  upload(diagonal_buffer, std::span<const float>(system.diagonal()), stream,
         "failed to upload CUDA depth diagonal");
  upload(rhs_buffer, std::span<const float>(system.right_hand_side()), stream,
         "failed to upload CUDA depth right-hand side");
  PcgWorkspace workspace(grid.site_count);
  const auto report =
      solve_pcg(rhs_buffer.data(), diagonal_buffer.data(), pressure_buffer.data(), workspace,
                grid.site_count, parameters.relative_tolerance, parameters.max_iterations,
                "CUDA depth-averaged flow", stream, [&](const float* input, float* output) {
                  launch_depth_flow_operator(input, mobility_buffer.data(), diagonal_buffer.data(),
                                             output, grid, stream);
                  check_launch("failed to launch CUDA depth-averaged operator");
                });
  launch_depth_flow_velocity(pressure_buffer.data(), mobility_buffer.data(), velocity_buffer.data(),
                             grid, stream);
  check_launch("failed to launch CUDA depth-averaged velocity reconstruction");
  const auto velocity = download(velocity_buffer, stream, "failed to download CUDA depth velocity");
  const auto scaled =
      detail::scale_velocity(spec, reduction.original_layout(), reduction.lift(velocity),
                             reduction.open_inlet_faces(), parameters.mean_inlet_speed);
  return {
      .field = scaled.field,
      .report = {.iterations = report.iterations,
                 .relative_residual = report.relative_residual,
                 .mean_inlet_speed = parameters.mean_inlet_speed,
                 .max_speed = scaled.max_speed},
  };
}

ResolvedFlowResult solve_resolved_flow(const SignalGridSpec& spec, std::span<const float> drag,
                                       const ResolvedFlowParameters& parameters,
                                       cudaStream_t stream) {
  parameters.validate();
  const detail::ResolvedFlowSystem system(spec, drag, parameters.axis);
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
        : u(nu, "CUDA Krylov velocity"), p(np, "CUDA Krylov pressure") {}
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
  const auto velocity = download(solution.solution->u, stream, "download CUDA velocity");
  const auto scaled = detail::scale_velocity(
      spec, system.layout(), velocity, system.open_inlet_faces(), parameters.mean_inlet_speed);
  return {
      .field = scaled.field,
      .report = {.outer_iterations = solution.iterations,
                 .inner_iterations = inner_iterations,
                 .relative_residual = static_cast<float>(solution.relative_residual),
                 .momentum_relative_residual =
                     static_cast<float>(std::sqrt(momentum_square / force_square)),
                 .divergence_rms = static_cast<float>(divergence_rms * std::abs(scaled.factor)),
                 .mean_inlet_speed = parameters.mean_inlet_speed,
                 .max_speed = scaled.max_speed,
                 .min_gap_voxels = system.minimum_gap_voxels()},
  };
}

}  // namespace cm::cuda
