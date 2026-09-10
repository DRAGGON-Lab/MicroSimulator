#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cm/metal/flow_source.hpp"
#include "core/flexible_gmres.hpp"
#include "core/flow_system.hpp"
#include "metal_flow.hpp"

namespace cm::metal {
namespace {

struct alignas(16) MetalUInt4 {
  std::uint32_t x;
  std::uint32_t y;
  std::uint32_t z;
  std::uint32_t w;
};

struct alignas(16) MetalFloat4 {
  float x;
  float y;
  float z;
  float w;
};

struct alignas(16) MetalFlowGridParameters {
  MetalUInt4 dimensions;
  MetalFloat4 spacing;
  MetalFloat4 inverse_spacing_squared;
  MetalUInt4 face_offsets;
  MetalUInt4 face_counts;
  std::uint32_t flow_axis;
  std::uint32_t site_count;
  std::uint32_t total_face_count;
  std::uint32_t padding;
};

static_assert(sizeof(MetalUInt4) == 16);
static_assert(sizeof(MetalFloat4) == 16);
static_assert(sizeof(MetalFlowGridParameters) == 96);

[[noreturn]] void throw_metal_error(const char* operation, NSError* error) {
  const char* detail = error == nil ? "unknown Metal error" : error.localizedDescription.UTF8String;
  throw std::runtime_error(std::string(operation) + ": " + detail);
}

id<MTLDevice> select_device(std::uint32_t device_index) {
  NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
  if (devices.count == 0) {
    id<MTLDevice> default_device = MTLCreateSystemDefaultDevice();
    if (device_index == 0 && default_device != nil) {
      return default_device;
    }
    if (default_device == nil) {
      throw std::runtime_error("Metal is unavailable on this system");
    }
  }
  if (static_cast<NSUInteger>(device_index) >= devices.count) {
    throw std::out_of_range("Metal device index is unavailable");
  }
  return devices[device_index];
}

id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device, id<MTLLibrary> library,
                                          NSString* name) {
  id<MTLFunction> function = [library newFunctionWithName:name];
  if (function == nil) {
    throw std::runtime_error(std::string("Metal flow function is missing: ") + name.UTF8String);
  }
  NSError* error = nil;
  id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                               error:&error];
  if (pipeline == nil) {
    throw_metal_error("failed to create a Metal flow pipeline", error);
  }
  return pipeline;
}

void wait_for_command(id<MTLCommandBuffer> command, const char* operation) {
  [command commit];
  [command waitUntilCompleted];
  if (command.status == MTLCommandBufferStatusError) {
    throw_metal_error(operation, command.error);
  }
}

std::uint32_t checked_count(std::size_t count, const char* description) {
  if (count == 0 || count > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error(std::string("Metal flow ") + description +
                              " must fit the nonzero uint32 index space");
  }
  return static_cast<std::uint32_t>(count);
}

MetalFlowGridParameters make_grid_parameters(const detail::FlowGridLayout& layout) {
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
  return {
      .dimensions = {layout.dimensions()[0], layout.dimensions()[1], layout.dimensions()[2], 0},
      .spacing = {spacing[0], spacing[1], spacing[2], 0.0F},
      .inverse_spacing_squared =
          {
              1.0F / (spacing[0] * spacing[0]),
              1.0F / (spacing[1] * spacing[1]),
              1.0F / (spacing[2] * spacing[2]),
              0.0F,
          },
      .face_offsets = {static_cast<std::uint32_t>(offsets[0]),
                       static_cast<std::uint32_t>(offsets[1]),
                       static_cast<std::uint32_t>(offsets[2]), face_count},
      .face_counts = {static_cast<std::uint32_t>(counts[0]), static_cast<std::uint32_t>(counts[1]),
                      static_cast<std::uint32_t>(counts[2]), face_count},
      .flow_axis = static_cast<std::uint32_t>(layout.flow_axis()),
      .site_count = site_count,
      .total_face_count = face_count,
      .padding = 0,
  };
}

struct PcgReport {
  std::uint32_t iterations{0};
  float relative_residual{0.0F};
};

}  // namespace

struct FlowSolver::Impl {
  explicit Impl(std::uint32_t device_index) {
    @autoreleasepool {
      device = select_device(device_index);
      queue = [device newCommandQueue];
      if (queue == nil) {
        throw std::runtime_error("failed to create a Metal flow command queue");
      }
      NSString* source = [NSString stringWithUTF8String:flow_source];
      if (source == nil) {
        throw std::runtime_error("Metal flow source is not valid UTF-8");
      }
      NSError* error = nil;
      id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
      if (library == nil) {
        throw_metal_error("failed to compile Metal flow", error);
      }
      depth_operator = make_pipeline(device, library, @"depth_flow_operator");
      depth_velocity = make_pipeline(device, library, @"depth_flow_velocity");
      momentum = make_pipeline(device, library, @"resolved_flow_momentum");
      gradient = make_pipeline(device, library, @"resolved_flow_gradient");
      divergence = make_pipeline(device, library, @"resolved_flow_divergence");
      pcg_initialize = make_pipeline(device, library, @"flow_pcg_initialize");
      pcg_update = make_pipeline(device, library, @"flow_pcg_update");
      pcg_precondition = make_pipeline(device, library, @"flow_pcg_precondition");
      pcg_direction = make_pipeline(device, library, @"flow_pcg_direction");
      vector_negate = make_pipeline(device, library, @"flow_vector_negate");
      vector_combine = make_pipeline(device, library, @"flow_vector_combine");
      vector_subtract = make_pipeline(device, library, @"flow_vector_subtract");
      dot_partial = make_pipeline(device, library, @"flow_dot_partial");
      if (dot_partial.maxTotalThreadsPerThreadgroup < reduction_width) {
        throw std::runtime_error("Metal flow reduction requires 64 threads per threadgroup");
      }
    }
  }

  id<MTLBuffer> allocate(std::size_t byte_count, const char* description) const {
    id<MTLBuffer> buffer = [device newBufferWithLength:byte_count
                                               options:MTLResourceStorageModeShared];
    if (buffer == nil) {
      throw std::runtime_error(std::string("failed to allocate Metal flow ") + description);
    }
    return buffer;
  }

  template <typename T>
  id<MTLBuffer> upload(std::span<const T> values, const char* description) const {
    if (values.empty()) {
      throw std::logic_error(std::string("cannot upload an empty Metal flow ") + description);
    }
    auto buffer = allocate(values.size_bytes(), description);
    std::memcpy(buffer.contents, values.data(), values.size_bytes());
    return buffer;
  }

  id<MTLBuffer> float_buffer(std::size_t count, const char* description) const {
    return allocate(count * sizeof(float), description);
  }

  template <typename Bind>
  void dispatch(id<MTLComputePipelineState> pipeline, std::uint32_t count, const char* operation,
                Bind&& bind) const {
    @autoreleasepool {
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      if (command == nil || encoder == nil) {
        throw std::runtime_error(std::string(operation) + ": failed to create a command");
      }
      [encoder setComputePipelineState:pipeline];
      bind(encoder);
      const auto width = std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256);
      [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
      [encoder endEncoding];
      wait_for_command(command, operation);
    }
  }

  struct PcgWorkspace {
    id<MTLBuffer> residual;
    id<MTLBuffer> preconditioned;
    id<MTLBuffer> direction;
    id<MTLBuffer> transformed;
    id<MTLBuffer> partials;
  };

  PcgWorkspace make_workspace(std::uint32_t count, const char* description) const {
    const auto partial_count = (count + reduction_width - 1) / reduction_width;
    return {
        .residual = float_buffer(count, (std::string(description) + " residual").c_str()),
        .preconditioned =
            float_buffer(count, (std::string(description) + " preconditioned residual").c_str()),
        .direction = float_buffer(count, (std::string(description) + " direction").c_str()),
        .transformed = float_buffer(count, (std::string(description) + " transformed").c_str()),
        .partials =
            float_buffer(partial_count, (std::string(description) + " reduction partials").c_str()),
    };
  }

  double dot(id<MTLBuffer> left, id<MTLBuffer> right, std::uint32_t count,
             id<MTLBuffer> partials) const {
    const auto group_count = (count + reduction_width - 1) / reduction_width;
    @autoreleasepool {
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      if (command == nil || encoder == nil) {
        throw std::runtime_error("Metal flow reduction failed to create a command");
      }
      [encoder setComputePipelineState:dot_partial];
      [encoder setBuffer:left offset:0 atIndex:0];
      [encoder setBuffer:right offset:0 atIndex:1];
      [encoder setBuffer:partials offset:0 atIndex:2];
      [encoder setBytes:&count length:sizeof(count) atIndex:3];
      [encoder dispatchThreadgroups:MTLSizeMake(group_count, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(reduction_width, 1, 1)];
      [encoder endEncoding];
      wait_for_command(command, "Metal flow reduction failed");
    }
    const auto* values = static_cast<const float*>(partials.contents);
    double result = 0.0;
    for (std::uint32_t index = 0; index < group_count; ++index) {
      result += values[index];
    }
    return result;
  }

  template <typename Apply>
  PcgReport solve_pcg(id<MTLBuffer> right_hand_side, id<MTLBuffer> diagonal, id<MTLBuffer> solution,
                      PcgWorkspace& workspace, std::uint32_t count, float tolerance,
                      std::uint32_t max_iterations, const char* label, Apply&& apply) const {
    dispatch(pcg_initialize, count, "Metal flow PCG initialization failed",
             [&](id<MTLComputeCommandEncoder> encoder) {
               [encoder setBuffer:right_hand_side offset:0 atIndex:0];
               [encoder setBuffer:diagonal offset:0 atIndex:1];
               [encoder setBuffer:solution offset:0 atIndex:2];
               [encoder setBuffer:workspace.residual offset:0 atIndex:3];
               [encoder setBuffer:workspace.preconditioned offset:0 atIndex:4];
               [encoder setBuffer:workspace.direction offset:0 atIndex:5];
               [encoder setBytes:&count length:sizeof(count) atIndex:6];
             });
    const auto rhs_norm_squared = dot(right_hand_side, right_hand_side, count, workspace.partials);
    if (rhs_norm_squared == 0.0) {
      return {};
    }
    const auto rhs_norm = std::sqrt(rhs_norm_squared);
    auto rho = dot(workspace.residual, workspace.preconditioned, count, workspace.partials);
    auto relative = 1.0;
    for (std::uint32_t iteration = 1; iteration <= max_iterations; ++iteration) {
      apply(workspace.direction, workspace.transformed);
      const auto curvature =
          dot(workspace.direction, workspace.transformed, count, workspace.partials);
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
      dispatch(pcg_update, count, "Metal flow PCG update failed",
               [&](id<MTLComputeCommandEncoder> encoder) {
                 [encoder setBuffer:solution offset:0 atIndex:0];
                 [encoder setBuffer:workspace.residual offset:0 atIndex:1];
                 [encoder setBuffer:workspace.direction offset:0 atIndex:2];
                 [encoder setBuffer:workspace.transformed offset:0 atIndex:3];
                 [encoder setBytes:&alpha length:sizeof(alpha) atIndex:4];
                 [encoder setBytes:&count length:sizeof(count) atIndex:5];
               });
      const auto residual_squared =
          dot(workspace.residual, workspace.residual, count, workspace.partials);
      relative = std::sqrt(std::max(0.0, residual_squared)) / rhs_norm;
      if (!std::isfinite(relative)) {
        throw std::runtime_error(std::string(label) +
                                 " conjugate gradient produced a non-finite residual");
      }
      if (relative <= tolerance) {
        return {.iterations = iteration, .relative_residual = static_cast<float>(relative)};
      }
      dispatch(pcg_precondition, count, "Metal flow PCG preconditioner failed",
               [&](id<MTLComputeCommandEncoder> encoder) {
                 [encoder setBuffer:workspace.residual offset:0 atIndex:0];
                 [encoder setBuffer:diagonal offset:0 atIndex:1];
                 [encoder setBuffer:workspace.preconditioned offset:0 atIndex:2];
                 [encoder setBytes:&count length:sizeof(count) atIndex:3];
               });
      const auto next_rho =
          dot(workspace.residual, workspace.preconditioned, count, workspace.partials);
      if (!std::isfinite(next_rho) || rho == 0.0) {
        throw std::runtime_error(std::string(label) +
                                 " conjugate gradient encountered a preconditioner breakdown");
      }
      const auto beta_double = next_rho / rho;
      if (!std::isfinite(beta_double) ||
          std::abs(beta_double) > std::numeric_limits<float>::max()) {
        throw std::runtime_error(std::string(label) +
                                 " conjugate gradient produced a non-finite direction");
      }
      const auto beta = static_cast<float>(beta_double);
      dispatch(pcg_direction, count, "Metal flow PCG direction update failed",
               [&](id<MTLComputeCommandEncoder> encoder) {
                 [encoder setBuffer:workspace.preconditioned offset:0 atIndex:0];
                 [encoder setBuffer:workspace.direction offset:0 atIndex:1];
                 [encoder setBytes:&beta length:sizeof(beta) atIndex:2];
                 [encoder setBytes:&count length:sizeof(count) atIndex:3];
               });
      rho = next_rho;
    }
    throw std::runtime_error(std::string(label) +
                             " conjugate gradient did not converge: relative " +
                             std::to_string(relative));
  }

  void apply_depth(id<MTLBuffer> input, id<MTLBuffer> mobility_buffer,
                   id<MTLBuffer> diagonal_buffer, id<MTLBuffer> output,
                   const MetalFlowGridParameters& grid) const {
    dispatch(depth_operator, grid.site_count, "Metal depth-averaged operator failed",
             [&](id<MTLComputeCommandEncoder> encoder) {
               [encoder setBuffer:input offset:0 atIndex:0];
               [encoder setBuffer:mobility_buffer offset:0 atIndex:1];
               [encoder setBuffer:diagonal_buffer offset:0 atIndex:2];
               [encoder setBuffer:output offset:0 atIndex:3];
               [encoder setBytes:&grid length:sizeof(grid) atIndex:4];
             });
  }

  void apply_momentum(id<MTLBuffer> input, id<MTLBuffer> active_buffer, id<MTLBuffer> exists_buffer,
                      id<MTLBuffer> face_drag_buffer, id<MTLBuffer> output,
                      const MetalFlowGridParameters& grid) const {
    dispatch(momentum, grid.total_face_count, "Metal resolved-flow momentum operator failed",
             [&](id<MTLComputeCommandEncoder> encoder) {
               [encoder setBuffer:input offset:0 atIndex:0];
               [encoder setBuffer:active_buffer offset:0 atIndex:1];
               [encoder setBuffer:exists_buffer offset:0 atIndex:2];
               [encoder setBuffer:face_drag_buffer offset:0 atIndex:3];
               [encoder setBuffer:output offset:0 atIndex:4];
               [encoder setBytes:&grid length:sizeof(grid) atIndex:5];
             });
  }

  void apply_gradient(id<MTLBuffer> pressure, id<MTLBuffer> fluid_buffer,
                      id<MTLBuffer> active_buffer, id<MTLBuffer> output,
                      const MetalFlowGridParameters& grid) const {
    dispatch(gradient, grid.total_face_count, "Metal resolved-flow gradient failed",
             [&](id<MTLComputeCommandEncoder> encoder) {
               [encoder setBuffer:pressure offset:0 atIndex:0];
               [encoder setBuffer:fluid_buffer offset:0 atIndex:1];
               [encoder setBuffer:active_buffer offset:0 atIndex:2];
               [encoder setBuffer:output offset:0 atIndex:3];
               [encoder setBytes:&grid length:sizeof(grid) atIndex:4];
             });
  }

  void apply_divergence(id<MTLBuffer> velocity, id<MTLBuffer> fluid_buffer, id<MTLBuffer> output,
                        const MetalFlowGridParameters& grid) const {
    dispatch(divergence, grid.site_count, "Metal resolved-flow divergence failed",
             [&](id<MTLComputeCommandEncoder> encoder) {
               [encoder setBuffer:velocity offset:0 atIndex:0];
               [encoder setBuffer:fluid_buffer offset:0 atIndex:1];
               [encoder setBuffer:output offset:0 atIndex:2];
               [encoder setBytes:&grid length:sizeof(grid) atIndex:3];
             });
  }

  void negate(id<MTLBuffer> input, id<MTLBuffer> output, std::uint32_t count) const {
    dispatch(vector_negate, count, "Metal flow vector negation failed",
             [&](id<MTLComputeCommandEncoder> encoder) {
               [encoder setBuffer:input offset:0 atIndex:0];
               [encoder setBuffer:output offset:0 atIndex:1];
               [encoder setBytes:&count length:sizeof(count) atIndex:2];
             });
  }

  void subtract(id<MTLBuffer> left, id<MTLBuffer> right, id<MTLBuffer> output,
                std::uint32_t count) const {
    dispatch(vector_subtract, count, "Metal flow vector subtraction failed",
             [&](id<MTLComputeCommandEncoder> encoder) {
               [encoder setBuffer:left offset:0 atIndex:0];
               [encoder setBuffer:right offset:0 atIndex:1];
               [encoder setBuffer:output offset:0 atIndex:2];
               [encoder setBytes:&count length:sizeof(count) atIndex:3];
             });
  }

  void combine(id<MTLBuffer> source, id<MTLBuffer> target, float alpha, float beta,
               std::uint32_t count) const {
    if (!std::isfinite(alpha)) throw std::runtime_error("non-finite Metal Krylov coefficient");
    dispatch(vector_combine, count, "Metal Krylov vector update",
             [&](id<MTLComputeCommandEncoder> encoder) {
               [encoder setBuffer:source offset:0 atIndex:0];
               [encoder setBuffer:target offset:0 atIndex:1];
               [encoder setBytes:&alpha length:sizeof(alpha) atIndex:2];
               [encoder setBytes:&beta length:sizeof(beta) atIndex:3];
               [encoder setBytes:&count length:sizeof(count) atIndex:4];
             });
  }

  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> depth_operator;
  id<MTLComputePipelineState> depth_velocity;
  id<MTLComputePipelineState> momentum;
  id<MTLComputePipelineState> gradient;
  id<MTLComputePipelineState> divergence;
  id<MTLComputePipelineState> pcg_initialize;
  id<MTLComputePipelineState> pcg_update;
  id<MTLComputePipelineState> pcg_precondition;
  id<MTLComputePipelineState> pcg_direction;
  id<MTLComputePipelineState> vector_negate;
  id<MTLComputePipelineState> vector_subtract;
  id<MTLComputePipelineState> vector_combine;
  id<MTLComputePipelineState> dot_partial;
  static constexpr std::uint32_t reduction_width = 64;
};

FlowSolver::FlowSolver(std::uint32_t device_index) : impl_(std::make_unique<Impl>(device_index)) {}

FlowSolver::~FlowSolver() = default;

DepthAveragedFlowResult FlowSolver::solve_depth_averaged(
    const SignalGridSpec& spec, std::span<const float> mobility,
    const DepthAveragedFlowParameters& parameters) {
  parameters.validate();
  const detail::ShallowFlowReduction reduction(spec, mobility, parameters.axis);
  const detail::DepthAveragedFlowSystem system(reduction.grid(), reduction.conductance(),
                                               parameters.axis);
  const auto grid = make_grid_parameters(system.layout());
  const auto mobility_buffer = impl_->upload<float>(system.mobility(), "depth mobility");
  const auto diagonal_buffer = impl_->upload<float>(system.diagonal(), "depth diagonal");
  const auto rhs_buffer = impl_->upload<float>(system.right_hand_side(), "depth right-hand side");
  const auto pressure_buffer = impl_->float_buffer(grid.site_count, "depth pressure");
  const auto velocity_buffer = impl_->float_buffer(grid.total_face_count, "depth velocity");
  auto workspace = impl_->make_workspace(grid.site_count, "depth");
  const auto report =
      impl_->solve_pcg(rhs_buffer, diagonal_buffer, pressure_buffer, workspace, grid.site_count,
                       parameters.relative_tolerance, parameters.max_iterations,
                       "Metal depth-averaged flow", [&](id<MTLBuffer> input, id<MTLBuffer> output) {
                         impl_->apply_depth(input, mobility_buffer, diagonal_buffer, output, grid);
                       });
  impl_->dispatch(impl_->depth_velocity, grid.total_face_count,
                  "Metal depth-averaged velocity reconstruction failed",
                  [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:pressure_buffer offset:0 atIndex:0];
                    [encoder setBuffer:mobility_buffer offset:0 atIndex:1];
                    [encoder setBuffer:velocity_buffer offset:0 atIndex:2];
                    [encoder setBytes:&grid length:sizeof(grid) atIndex:3];
                  });
  const auto* velocity_values = static_cast<const float*>(velocity_buffer.contents);
  const std::span<const float> velocity(velocity_values, grid.total_face_count);
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

ResolvedFlowResult FlowSolver::solve_resolved(const SignalGridSpec& spec,
                                              std::span<const float> drag,
                                              const ResolvedFlowParameters& parameters) {
  parameters.validate();
  const detail::ResolvedFlowSystem system(spec, drag, parameters.axis);
  const auto grid = make_grid_parameters(system.layout());
  const auto fluid_buffer = impl_->upload<std::uint8_t>(system.fluid(), "fluid mask");
  const auto active_buffer = impl_->upload<std::uint8_t>(system.active(), "active face mask");
  const auto exists_buffer = impl_->upload<std::uint8_t>(system.exists(), "face existence mask");
  const auto face_drag_buffer = impl_->upload<float>(system.face_drag(), "face drag");
  const auto face_diagonal_buffer = impl_->upload<float>(system.diagonal(), "momentum diagonal");
  const auto force_buffer = impl_->upload<float>(system.force(), "momentum force");
  const auto pressure_diagonal = system.pressure_diagonal();
  const auto pressure_diagonal_buffer =
      impl_->upload<float>(pressure_diagonal, "pressure diagonal");

  const auto gradient = impl_->float_buffer(grid.total_face_count, "block gradient");
  auto inner_workspace = impl_->make_workspace(grid.total_face_count, "momentum");
  auto outer_workspace = impl_->make_workspace(grid.site_count, "pressure");
  struct Vector {
    id<MTLBuffer> u;
    id<MTLBuffer> p;
  };
  const double continuity_scale =
      1.0 / *std::min_element(system.layout().spacing().begin(), system.layout().spacing().end());
  detail::FlexibleKrylovOperations<Vector> ops;
  ops.make_zero = [&] {
    Vector v{impl_->float_buffer(grid.total_face_count, "Krylov velocity"),
             impl_->float_buffer(grid.site_count, "Krylov pressure")};
    std::memset(v.u.contents, 0, grid.total_face_count * sizeof(float));
    std::memset(v.p.contents, 0, grid.site_count * sizeof(float));
    return v;
  };
  ops.copy = [&](const Vector& source, Vector& target) {
    impl_->combine(source.u, target.u, 1, 0, grid.total_face_count);
    impl_->combine(source.p, target.p, 1, 0, grid.site_count);
  };
  ops.axpy = [&](Vector& target, double alpha, const Vector& source) {
    impl_->combine(source.u, target.u, static_cast<float>(alpha), 1, grid.total_face_count);
    impl_->combine(source.p, target.p, static_cast<float>(alpha), 1, grid.site_count);
  };
  ops.dot = [&](const Vector& a, const Vector& b) {
    return impl_->dot(a.u, b.u, grid.total_face_count, inner_workspace.partials) +
           impl_->dot(a.p, b.p, grid.site_count, outer_workspace.partials);
  };
  ops.apply = [&](const Vector& input, Vector& output) {
    impl_->apply_momentum(input.u, active_buffer, exists_buffer, face_drag_buffer, output.u, grid);
    impl_->apply_gradient(input.p, fluid_buffer, active_buffer, gradient, grid);
    impl_->combine(gradient, output.u, 1, 1, grid.total_face_count);
    impl_->apply_divergence(input.u, fluid_buffer, output.p, grid);
    impl_->combine(output.p, output.p, static_cast<float>(continuity_scale), 0, grid.site_count);
  };
  std::uint64_t inner_iterations = 0;
  ops.precondition = [&](const Vector& input, Vector& output) {
    const auto report = impl_->solve_pcg(
        input.u, face_diagonal_buffer, output.u, inner_workspace, grid.total_face_count,
        parameters.inner_relative_tolerance, parameters.max_inner_iterations,
        "Metal momentum preconditioner", [&](id<MTLBuffer> x, id<MTLBuffer> y) {
          impl_->apply_momentum(x, active_buffer, exists_buffer, face_drag_buffer, y, grid);
        });
    inner_iterations += report.iterations;
    impl_->dispatch(impl_->pcg_precondition, grid.site_count, "Metal pressure preconditioner",
                    [&](id<MTLComputeCommandEncoder> encoder) {
                      [encoder setBuffer:input.p offset:0 atIndex:0];
                      [encoder setBuffer:pressure_diagonal_buffer offset:0 atIndex:1];
                      [encoder setBuffer:output.p offset:0 atIndex:2];
                      [encoder setBytes:&grid.site_count length:sizeof(grid.site_count) atIndex:3];
                    });
    impl_->combine(output.p, output.p, static_cast<float>(-1 / continuity_scale), 0,
                   grid.site_count);
  };
  auto rhs = ops.make_zero();
  impl_->combine(force_buffer, rhs.u, 1, 0, grid.total_face_count);
  const auto solution = detail::flexible_gmres(ops, rhs, parameters.relative_tolerance,
                                               parameters.max_outer_iterations);
  auto residual = ops.make_zero();
  ops.apply(solution.solution, residual);
  ops.axpy(residual, -1, rhs);
  const auto momentum_square =
      impl_->dot(residual.u, residual.u, grid.total_face_count, inner_workspace.partials);
  const auto force_square =
      impl_->dot(rhs.u, rhs.u, grid.total_face_count, inner_workspace.partials);
  const auto divergence_square =
      impl_->dot(residual.p, residual.p, grid.site_count, outer_workspace.partials);
  const auto fluid_count =
      std::count(system.fluid().begin(), system.fluid().end(), std::uint8_t{1});
  const double divergence_rms =
      fluid_count == 0
          ? 0.0
          : std::sqrt(divergence_square / static_cast<double>(fluid_count)) / continuity_scale;
  const auto* values = static_cast<const float*>(solution.solution.u.contents);
  const auto scaled = detail::scale_velocity(
      spec, system.layout(), std::span<const float>(values, grid.total_face_count),
      system.open_inlet_faces(), parameters.mean_inlet_speed);
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

}  // namespace cm::metal
