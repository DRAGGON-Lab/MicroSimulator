#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
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

#include "core/conjugate_gradient.hpp"
#include "core/device_types.hpp"
#include "core/flexible_gmres.hpp"
#include "core/flow_system.hpp"
#include "core/shared_constants.hpp"
#include "metal_compile_options.hpp"
#include "metal_flow.hpp"
#include "microsimulator/metal/depth_flow_source.hpp"
#include "microsimulator/metal/flow_source.hpp"

namespace microsimulator::metal {
namespace {

using MetalUInt4 = device::uint4;
using MetalFloat4 = device::float4;
using MetalFlowGridParameters = device::FlowGridParameters;
using MetalDepthFlowScales = device::DepthFlowScales;

// A double-word value: an unevaluated sum of two floats, laid out as a Metal
// float2.
struct MetalFloatPair {
  float high;
  float low;
};

static_assert(sizeof(MetalFloatPair) == sizeof(device::float2));

MetalFloatPair split(double value) {
  const auto high = static_cast<float>(value);
  return {high, static_cast<float>(value - static_cast<double>(high))};
}

[[noreturn]] void throw_metal_error(const char* operation, NSError* error) {
  const char* detail = error == nil ? "unknown Metal error" : error.localizedDescription.UTF8String;
  throw std::runtime_error(std::string(operation) + ": " + detail);
}

id<MTLLibrary> compile_library(id<MTLDevice> device, const char* text, MTLCompileOptions* options,
                               const char* operation) {
  NSString* source = [NSString stringWithUTF8String:text];
  if (source == nil) {
    throw std::runtime_error(std::string(operation) + ": source is not valid UTF-8");
  }
  NSError* error = nil;
  id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
  if (library == nil) {
    throw_metal_error(operation, error);
  }
  return library;
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
      .site_count = site_count,
      .total_face_count = face_count,
      .padding0 = 0,
      .padding1 = 0,
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
      id<MTLLibrary> library =
          compile_library(device, flow_source, nil, "failed to compile Metal flow");
      id<MTLLibrary> depth_library = compile_library(device, depth_flow_source, safe_math_options(),
                                                     "failed to compile Metal depth flow");
      depth_operator = make_pipeline(device, depth_library, @"depth_pair_operator");
      depth_velocity = make_pipeline(device, depth_library, @"depth_pair_velocity");
      depth_initialize = make_pipeline(device, depth_library, @"depth_pair_initialize");
      depth_update = make_pipeline(device, depth_library, @"depth_pair_update");
      depth_precondition = make_pipeline(device, depth_library, @"depth_pair_precondition");
      depth_direction = make_pipeline(device, depth_library, @"depth_pair_direction");
      depth_dot = make_pipeline(device, depth_library, @"depth_pair_dot");
      depth_sum = make_pipeline(device, depth_library, @"depth_pair_sum");
      depth_fold = make_pipeline(device, depth_library, @"depth_pair_fold");
      depth_step_length = make_pipeline(device, depth_library, @"depth_pair_step_length");
      depth_step_weight = make_pipeline(device, depth_library, @"depth_pair_step_weight");
      depth_solve = make_pipeline(device, depth_library, @"depth_pair_solve");
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
        throw std::runtime_error("Metal flow reduction requires " +
                                 std::to_string(reduction_width) + " threads per threadgroup");
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
    const detail::ConjugateGradientOperations operations{
        .initialize =
            [&] {
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
            },
        .residual_dot_residual =
            [&] { return dot(workspace.residual, workspace.residual, count, workspace.partials); },
        .residual_dot_preconditioned =
            [&] {
              return dot(workspace.residual, workspace.preconditioned, count, workspace.partials);
            },
        .apply_direction =
            [&] {
              apply(workspace.direction, workspace.transformed);
              return dot(workspace.direction, workspace.transformed, count, workspace.partials);
            },
        .update =
            [&](double step) {
              const auto alpha = static_cast<float>(step);
              dispatch(pcg_update, count, "Metal flow PCG update failed",
                       [&](id<MTLComputeCommandEncoder> encoder) {
                         [encoder setBuffer:solution offset:0 atIndex:0];
                         [encoder setBuffer:workspace.residual offset:0 atIndex:1];
                         [encoder setBuffer:workspace.direction offset:0 atIndex:2];
                         [encoder setBuffer:workspace.transformed offset:0 atIndex:3];
                         [encoder setBytes:&alpha length:sizeof(alpha) atIndex:4];
                         [encoder setBytes:&count length:sizeof(count) atIndex:5];
                       });
            },
        .precondition =
            [&] {
              dispatch(pcg_precondition, count, "Metal flow PCG preconditioner failed",
                       [&](id<MTLComputeCommandEncoder> encoder) {
                         [encoder setBuffer:workspace.residual offset:0 atIndex:0];
                         [encoder setBuffer:diagonal offset:0 atIndex:1];
                         [encoder setBuffer:workspace.preconditioned offset:0 atIndex:2];
                         [encoder setBytes:&count length:sizeof(count) atIndex:3];
                       });
            },
        .update_direction =
            [&](double step) {
              const auto beta = static_cast<float>(step);
              dispatch(pcg_direction, count, "Metal flow PCG direction update failed",
                       [&](id<MTLComputeCommandEncoder> encoder) {
                         [encoder setBuffer:workspace.preconditioned offset:0 atIndex:0];
                         [encoder setBuffer:workspace.direction offset:0 atIndex:1];
                         [encoder setBytes:&beta length:sizeof(beta) atIndex:2];
                         [encoder setBytes:&count length:sizeof(count) atIndex:3];
                       });
            },
        .step_limit = std::numeric_limits<float>::max(),
    };
    const auto report =
        detail::preconditioned_conjugate_gradient(operations, tolerance, max_iterations, label);
    return {.iterations = report.iterations,
            .relative_residual = static_cast<float>(report.relative_residual)};
  }

  id<MTLBuffer> pair_buffer(std::size_t count, const char* description) const {
    return allocate(count * sizeof(MetalFloatPair), description);
  }

  // Where the depth-averaged kernels keep a solve's scalars and counters.
  enum DepthScalar : std::uint32_t {
    depth_rho = 0,
    depth_curvature = 1,
    depth_residual_squared = 2,
    depth_next_rho = 3,
    depth_scalar_count = 6,
  };
  enum DepthCounter : std::uint32_t {
    depth_iterations = 0,
    depth_stopped = 1,
    depth_converged = 2,
    depth_failure = 3,
    depth_counter_count = 4,
  };

  // Encodes one kernel over `count` threads into a command that other
  // dispatches share. Dispatches of one encoder run in the order encoded.
  template <typename Bind>
  static void encode(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline,
                     std::uint32_t count, Bind&& bind) {
    [encoder setComputePipelineState:pipeline];
    bind(encoder);
    const auto width = std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256);
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  }

  // Encodes the dot product of two double-word vectors, left among the
  // solve's scalars at `place`.
  void encode_depth_dot(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> left,
                        id<MTLBuffer> right, std::uint32_t count, id<MTLBuffer> partials,
                        id<MTLBuffer> folded, id<MTLBuffer> scalars, std::uint32_t place) const {
    const auto block_width = reduction_width;
    auto block_count = (count + block_width - 1) / block_width;
    encode(encoder, depth_dot, block_count, [&](id<MTLComputeCommandEncoder> bound) {
      [bound setBuffer:left offset:0 atIndex:0];
      [bound setBuffer:right offset:0 atIndex:1];
      [bound setBuffer:partials offset:0 atIndex:2];
      [bound setBytes:&count length:sizeof(count) atIndex:3];
      [bound setBytes:&block_width length:sizeof(block_width) atIndex:4];
    });
    // The partials are added in groups, and the groups by one thread, so
    // that no thread adds more than a few hundred values.
    id<MTLBuffer> terms = partials;
    if (block_count > 2 * depth_fold_width) {
      const auto group_width = depth_fold_width;
      const auto group_count = (block_count + group_width - 1) / group_width;
      encode(encoder, depth_fold, group_count, [&](id<MTLComputeCommandEncoder> bound) {
        [bound setBuffer:partials offset:0 atIndex:0];
        [bound setBuffer:folded offset:0 atIndex:1];
        [bound setBytes:&block_count length:sizeof(block_count) atIndex:2];
        [bound setBytes:&group_width length:sizeof(group_width) atIndex:3];
      });
      terms = folded;
      block_count = group_count;
    }
    encode(encoder, depth_sum, 1, [&](id<MTLComputeCommandEncoder> bound) {
      [bound setBuffer:terms offset:0 atIndex:0];
      [bound setBuffer:scalars offset:0 atIndex:1];
      [bound setBytes:&block_count length:sizeof(block_count) atIndex:2];
      [bound setBytes:&place length:sizeof(place) atIndex:3];
    });
  }

  // Runs what `body` encodes as one command and waits for it.
  template <typename Body>
  void run_command(const char* operation, Body&& body) const {
    @autoreleasepool {
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      if (command == nil || encoder == nil) {
        throw std::runtime_error(std::string(operation) + ": failed to create a command");
      }
      body(encoder);
      [encoder endEncoding];
      wait_for_command(command, operation);
    }
  }

  // Solves one depth-averaged pressure system and returns the velocity on
  // every face of its grid.
  std::vector<float> solve_depth_system(const detail::DepthAveragedFlowSystem& system,
                                        float tolerance, std::uint32_t max_iterations,
                                        PcgReport& report) const {
    const auto grid = make_grid_parameters(system.layout());
    const auto& active = system.active();
    const auto count = static_cast<std::uint32_t>(active.size());
    const auto mobility_buffer = upload<float>(system.mobility(), "depth mobility");
    const auto diagonal_buffer = upload<float>(active.diagonal, "depth diagonal");
    const auto rhs_buffer = upload<float>(active.right_hand_side, "depth right-hand side");
    const auto boundary_buffer =
        upload<float>(active.boundary_conductance, "depth boundary conductance");
    const auto neighbor_buffer = upload<std::uint32_t>(active.neighbors, "depth neighbors");
    const auto conductance_buffer = upload<float>(active.conductance, "depth conductance");
    const auto face_conductance_buffer =
        upload<float>(system.face_conductance(), "depth face conductance");
    const auto face_pressure_buffer = upload<float>(system.face_pressure(), "depth face pressure");
    const auto solution_buffer = pair_buffer(count, "depth active pressure");
    const auto velocity_buffer = float_buffer(grid.total_face_count, "depth velocity");
    report = solve_depth_pcg(rhs_buffer, diagonal_buffer, neighbor_buffer, conductance_buffer,
                             boundary_buffer, solution_buffer, count, grid.spacing, tolerance,
                             max_iterations);
    // The velocity kernel reads a pressure at every site of the grid; a new
    // buffer is zero, which is the pressure of a site that is not active.
    const auto pressure_buffer = pair_buffer(grid.site_count, "depth pressure");
    const auto* solved = static_cast<const MetalFloatPair*>(solution_buffer.contents);
    auto* pressure = static_cast<MetalFloatPair*>(pressure_buffer.contents);
    for (std::uint32_t index = 0; index < count; ++index) {
      pressure[active.sites[index]] = solved[index];
    }
    dispatch(depth_velocity, grid.total_face_count,
             "Metal depth-averaged velocity reconstruction failed",
             [&](id<MTLComputeCommandEncoder> encoder) {
               [encoder setBuffer:pressure_buffer offset:0 atIndex:0];
               [encoder setBuffer:mobility_buffer offset:0 atIndex:1];
               [encoder setBuffer:velocity_buffer offset:0 atIndex:2];
               [encoder setBytes:&grid length:sizeof(grid) atIndex:3];
               [encoder setBuffer:face_conductance_buffer offset:0 atIndex:4];
               [encoder setBuffer:face_pressure_buffer offset:0 atIndex:5];
             });
    const auto* values = static_cast<const float*>(velocity_buffer.contents);
    return {values, values + grid.total_face_count};
  }

  // The depth-averaged pressure solve over a system's active sites, on
  // double-word vectors, which follow the double-precision vectors of the CPU
  // and CUDA backends. The solve's scalars stay on the device, so the host
  // waits once for each batch of iterations.
  PcgReport solve_depth_pcg(id<MTLBuffer> right_hand_side, id<MTLBuffer> diagonal,
                            id<MTLBuffer> neighbors, id<MTLBuffer> conductance,
                            id<MTLBuffer> boundary_conductance, id<MTLBuffer> solution,
                            std::uint32_t count, const MetalFloat4& spacing, float tolerance,
                            std::uint32_t max_iterations) const {
    const std::array inverse_square{
        split(1.0 / (static_cast<double>(spacing.x) * spacing.x)),
        split(1.0 / (static_cast<double>(spacing.y) * spacing.y)),
        split(1.0 / (static_cast<double>(spacing.z) * spacing.z)),
    };
    const MetalDepthFlowScales scales{
        .inverse_square_high = {inverse_square[0].high, inverse_square[1].high,
                                inverse_square[2].high, 0.0F},
        .inverse_square_low = {inverse_square[0].low, inverse_square[1].low, inverse_square[2].low,
                               0.0F},
    };
    // A system of up to this many sites is solved in one threadgroup, every
    // iteration in one command.
    const auto in_one_group = count <= depth_group_sites;
    const auto residual = pair_buffer(count, "depth residual");
    const auto preconditioned = pair_buffer(count, "depth preconditioned residual");
    const auto direction = pair_buffer(count, "depth direction");
    const auto transformed = pair_buffer(count, "depth transformed vector");
    const auto partial_count = (count + reduction_width - 1) / reduction_width;
    const auto partials = pair_buffer(partial_count, "depth reduction partials");
    const auto folded = pair_buffer((partial_count + depth_fold_width - 1) / depth_fold_width,
                                    "depth folded partials");
    const auto scalars = pair_buffer(depth_scalar_count, "depth solve scalars");
    const auto counters =
        allocate(depth_counter_count * sizeof(std::uint32_t), "depth solve counters");
    const auto scalar = [&](std::uint32_t place) {
      const auto value = static_cast<const MetalFloatPair*>(scalars.contents)[place];
      return static_cast<double>(value.high) + static_cast<double>(value.low);
    };
    const auto counter = [&](std::uint32_t place) {
      return static_cast<const std::uint32_t*>(counters.contents)[place];
    };
    const detail::BatchedConjugateGradientOperations operations{
        .initialize =
            [&] {
              std::memset(scalars.contents, 0, scalars.length);
              std::memset(counters.contents, 0, counters.length);
              run_command("Metal depth-averaged PCG initialization failed",
                          [&](id<MTLComputeCommandEncoder> encoder) {
                            encode(encoder, depth_initialize, count,
                                   [&](id<MTLComputeCommandEncoder> bound) {
                                     [bound setBuffer:right_hand_side offset:0 atIndex:0];
                                     [bound setBuffer:diagonal offset:0 atIndex:1];
                                     [bound setBuffer:solution offset:0 atIndex:2];
                                     [bound setBuffer:residual offset:0 atIndex:3];
                                     [bound setBuffer:preconditioned offset:0 atIndex:4];
                                     [bound setBuffer:direction offset:0 atIndex:5];
                                     [bound setBytes:&count length:sizeof(count) atIndex:6];
                                   });
                            encode_depth_dot(encoder, residual, residual, count, partials, folded,
                                             scalars, depth_residual_squared);
                            encode_depth_dot(encoder, residual, preconditioned, count, partials,
                                             folded, scalars, depth_rho);
                          });
              return scalar(depth_residual_squared);
            },
        .iterate =
            [&](std::uint32_t iterations, double threshold) {
              const auto bound_squared = split(threshold);
              if (in_one_group) {
                run_command(
                    "Metal depth-averaged PCG failed", [&](id<MTLComputeCommandEncoder> encoder) {
                      [encoder setComputePipelineState:depth_solve];
                      [encoder setBuffer:solution offset:0 atIndex:0];
                      [encoder setBuffer:residual offset:0 atIndex:1];
                      [encoder setBuffer:preconditioned offset:0 atIndex:2];
                      [encoder setBuffer:direction offset:0 atIndex:3];
                      [encoder setBuffer:transformed offset:0 atIndex:4];
                      [encoder setBuffer:diagonal offset:0 atIndex:5];
                      [encoder setBuffer:neighbors offset:0 atIndex:6];
                      [encoder setBuffer:conductance offset:0 atIndex:7];
                      [encoder setBuffer:boundary_conductance offset:0 atIndex:8];
                      [encoder setBuffer:scalars offset:0 atIndex:11];
                      [encoder setBuffer:counters offset:0 atIndex:12];
                      [encoder setBytes:&count length:sizeof(count) atIndex:13];
                      [encoder setBytes:&scales length:sizeof(scales) atIndex:14];
                      [encoder setBytes:&bound_squared length:sizeof(bound_squared) atIndex:15];
                      [encoder setBytes:&iterations length:sizeof(iterations) atIndex:16];
                      const auto width =
                          std::min<NSUInteger>(depth_solve.maxTotalThreadsPerThreadgroup, 1024);
                      [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                              threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
                    });
                return detail::ConjugateGradientProgress{
                    .iterations = counter(depth_iterations),
                    .converged = counter(depth_converged) != 0,
                    .failure =
                        static_cast<detail::ConjugateGradientFailure>(counter(depth_failure)),
                    .residual_squared = scalar(depth_residual_squared),
                };
              }
              run_command(
                  "Metal depth-averaged PCG iteration failed",
                  [&](id<MTLComputeCommandEncoder> encoder) {
                    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
                      encode(encoder, depth_operator, count,
                             [&](id<MTLComputeCommandEncoder> bound) {
                               [bound setBuffer:direction offset:0 atIndex:0];
                               [bound setBuffer:neighbors offset:0 atIndex:1];
                               [bound setBuffer:conductance offset:0 atIndex:2];
                               [bound setBuffer:transformed offset:0 atIndex:3];
                               [bound setBytes:&count length:sizeof(count) atIndex:4];
                               [bound setBytes:&scales length:sizeof(scales) atIndex:5];
                               [bound setBuffer:boundary_conductance offset:0 atIndex:6];
                             });
                      encode_depth_dot(encoder, direction, transformed, count, partials, folded,
                                       scalars, depth_curvature);
                      encode(encoder, depth_step_length, 1,
                             [&](id<MTLComputeCommandEncoder> bound) {
                               [bound setBuffer:scalars offset:0 atIndex:0];
                               [bound setBuffer:counters offset:0 atIndex:1];
                             });
                      encode(encoder, depth_update, count, [&](id<MTLComputeCommandEncoder> bound) {
                        [bound setBuffer:solution offset:0 atIndex:0];
                        [bound setBuffer:residual offset:0 atIndex:1];
                        [bound setBuffer:direction offset:0 atIndex:2];
                        [bound setBuffer:transformed offset:0 atIndex:3];
                        [bound setBuffer:scalars offset:0 atIndex:4];
                        [bound setBytes:&count length:sizeof(count) atIndex:5];
                      });
                      encode_depth_dot(encoder, residual, residual, count, partials, folded,
                                       scalars, depth_residual_squared);
                      encode(encoder, depth_precondition, count,
                             [&](id<MTLComputeCommandEncoder> bound) {
                               [bound setBuffer:residual offset:0 atIndex:0];
                               [bound setBuffer:diagonal offset:0 atIndex:1];
                               [bound setBuffer:preconditioned offset:0 atIndex:2];
                               [bound setBytes:&count length:sizeof(count) atIndex:3];
                             });
                      encode_depth_dot(encoder, residual, preconditioned, count, partials, folded,
                                       scalars, depth_next_rho);
                      encode(
                          encoder, depth_step_weight, 1, [&](id<MTLComputeCommandEncoder> bound) {
                            [bound setBuffer:scalars offset:0 atIndex:0];
                            [bound setBuffer:counters offset:0 atIndex:1];
                            [bound setBytes:&bound_squared length:sizeof(bound_squared) atIndex:2];
                          });
                      encode(encoder, depth_direction, count,
                             [&](id<MTLComputeCommandEncoder> bound) {
                               [bound setBuffer:preconditioned offset:0 atIndex:0];
                               [bound setBuffer:direction offset:0 atIndex:1];
                               [bound setBuffer:scalars offset:0 atIndex:2];
                               [bound setBytes:&count length:sizeof(count) atIndex:3];
                             });
                    }
                  });
              return detail::ConjugateGradientProgress{
                  .iterations = counter(depth_iterations),
                  .converged = counter(depth_converged) != 0,
                  .failure = static_cast<detail::ConjugateGradientFailure>(counter(depth_failure)),
                  .residual_squared = scalar(depth_residual_squared),
              };
            },
    };
    const auto report = detail::batched_conjugate_gradient(
        operations, tolerance, max_iterations, in_one_group ? max_iterations : depth_batch,
        "Metal depth-averaged flow");
    return {.iterations = report.iterations,
            .relative_residual = static_cast<float>(report.relative_residual)};
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

  // Solves one resolved block system and returns its velocity on every face.
  detail::ResolvedSystemSolve solve_resolved_system(
      const detail::ResolvedFlowSystem& system,
      const detail::ResolvedSolveControls& parameters) const {
    const auto grid = make_grid_parameters(system.layout());
    const auto fluid_buffer = upload<std::uint8_t>(system.fluid(), "fluid mask");
    const auto active_buffer = upload<std::uint8_t>(system.active(), "active face mask");
    const auto exists_buffer = upload<std::uint8_t>(system.exists(), "face existence mask");
    const auto face_drag_buffer = upload<float>(system.face_drag(), "face drag");
    const auto face_diagonal_buffer = upload<float>(system.diagonal(), "momentum diagonal");
    const auto force_buffer = upload<float>(system.force(), "momentum force");
    const auto pressure_diagonal = system.pressure_diagonal();
    const auto pressure_diagonal_buffer = upload<float>(pressure_diagonal, "pressure diagonal");

    const auto gradient = float_buffer(grid.total_face_count, "block gradient");
    auto inner_workspace = make_workspace(grid.total_face_count, "momentum");
    auto outer_workspace = make_workspace(grid.site_count, "pressure");
    struct Vector {
      id<MTLBuffer> u;
      id<MTLBuffer> p;
    };
    const double continuity_scale =
        1.0 / *std::min_element(system.layout().spacing().begin(), system.layout().spacing().end());
    detail::FlexibleKrylovOperations<Vector> ops;
    ops.make_zero = [&] {
      Vector v{float_buffer(grid.total_face_count, "Krylov velocity"),
               float_buffer(grid.site_count, "Krylov pressure")};
      std::memset(v.u.contents, 0, grid.total_face_count * sizeof(float));
      std::memset(v.p.contents, 0, grid.site_count * sizeof(float));
      return v;
    };
    ops.copy = [&](const Vector& source, Vector& target) {
      combine(source.u, target.u, 1, 0, grid.total_face_count);
      combine(source.p, target.p, 1, 0, grid.site_count);
    };
    ops.axpy = [&](Vector& target, double alpha, const Vector& source) {
      combine(source.u, target.u, static_cast<float>(alpha), 1, grid.total_face_count);
      combine(source.p, target.p, static_cast<float>(alpha), 1, grid.site_count);
    };
    ops.dot = [&](const Vector& a, const Vector& b) {
      return dot(a.u, b.u, grid.total_face_count, inner_workspace.partials) +
             dot(a.p, b.p, grid.site_count, outer_workspace.partials);
    };
    ops.apply = [&](const Vector& input, Vector& output) {
      apply_momentum(input.u, active_buffer, exists_buffer, face_drag_buffer, output.u, grid);
      apply_gradient(input.p, fluid_buffer, active_buffer, gradient, grid);
      combine(gradient, output.u, 1, 1, grid.total_face_count);
      apply_divergence(input.u, fluid_buffer, output.p, grid);
      combine(output.p, output.p, static_cast<float>(continuity_scale), 0, grid.site_count);
    };
    std::uint64_t inner_iterations = 0;
    ops.precondition = [&](const Vector& input, Vector& output) {
      const auto report =
          solve_pcg(input.u, face_diagonal_buffer, output.u, inner_workspace, grid.total_face_count,
                    parameters.inner_relative_tolerance, parameters.max_inner_iterations,
                    "Metal momentum preconditioner", [&](id<MTLBuffer> x, id<MTLBuffer> y) {
                      apply_momentum(x, active_buffer, exists_buffer, face_drag_buffer, y, grid);
                    });
      inner_iterations += report.iterations;
      dispatch(pcg_precondition, grid.site_count, "Metal pressure preconditioner",
               [&](id<MTLComputeCommandEncoder> encoder) {
                 [encoder setBuffer:input.p offset:0 atIndex:0];
                 [encoder setBuffer:pressure_diagonal_buffer offset:0 atIndex:1];
                 [encoder setBuffer:output.p offset:0 atIndex:2];
                 [encoder setBytes:&grid.site_count length:sizeof(grid.site_count) atIndex:3];
               });
      combine(output.p, output.p, static_cast<float>(-1 / continuity_scale), 0, grid.site_count);
    };
    auto rhs = ops.make_zero();
    combine(force_buffer, rhs.u, 1, 0, grid.total_face_count);
    const auto solution = detail::flexible_gmres(ops, rhs, parameters.relative_tolerance,
                                                 parameters.max_outer_iterations);
    auto residual = ops.make_zero();
    ops.apply(solution.solution, residual);
    ops.axpy(residual, -1, rhs);
    const auto momentum_square =
        dot(residual.u, residual.u, grid.total_face_count, inner_workspace.partials);
    const auto force_square = dot(rhs.u, rhs.u, grid.total_face_count, inner_workspace.partials);
    const auto divergence_square =
        dot(residual.p, residual.p, grid.site_count, outer_workspace.partials);
    const auto fluid_count =
        std::count(system.fluid().begin(), system.fluid().end(), std::uint8_t{1});
    const double divergence_rms =
        fluid_count == 0
            ? 0.0
            : std::sqrt(divergence_square / static_cast<double>(fluid_count)) / continuity_scale;
    const auto* values = static_cast<const float*>(solution.solution.u.contents);
    return {
        .velocity = {values, values + grid.total_face_count},
        .outer_iterations = solution.iterations,
        .inner_iterations = inner_iterations,
        .relative_residual = static_cast<float>(solution.relative_residual),
        .momentum_relative_residual = static_cast<float>(std::sqrt(momentum_square / force_square)),
        .divergence_rms = divergence_rms,
    };
  }

  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> depth_operator;
  id<MTLComputePipelineState> depth_velocity;
  id<MTLComputePipelineState> depth_initialize;
  id<MTLComputePipelineState> depth_update;
  id<MTLComputePipelineState> depth_precondition;
  id<MTLComputePipelineState> depth_direction;
  id<MTLComputePipelineState> depth_dot;
  id<MTLComputePipelineState> depth_sum;
  id<MTLComputePipelineState> depth_fold;
  id<MTLComputePipelineState> depth_step_length;
  id<MTLComputePipelineState> depth_step_weight;
  id<MTLComputePipelineState> depth_solve;
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
  static constexpr std::uint32_t reduction_width = shared::flow_reduction_width;
  // Iterations of the depth-averaged solve encoded into one command.
  static constexpr std::uint32_t depth_batch = 64;
  // Block partials one thread adds when a dot product is folded.
  static constexpr std::uint32_t depth_fold_width = 64;
  static constexpr std::uint32_t depth_group_sites = 1U << 16;
};

FlowSolver::FlowSolver(std::uint32_t device_index) : impl_(std::make_unique<Impl>(device_index)) {
}

FlowSolver::~FlowSolver() = default;

DepthAveragedFlowResult FlowSolver::solve_depth_averaged(
    const SignalGridSpec& spec, std::span<const float> mobility,
    const DepthAveragedFlowParameters& parameters) {
  return detail::solve_shallow_axis_flow(
      spec, mobility, parameters,
      [&](const detail::DepthAveragedFlowSystem& system, DepthAveragedFlowReport& report) {
        PcgReport solve;
        auto flux = impl_->solve_depth_system(system, parameters.relative_tolerance,
                                              parameters.max_iterations, solve);
        report.iterations += solve.iterations;
        report.relative_residual = std::max(report.relative_residual, solve.relative_residual);
        return flux;
      });
}

DepthAveragedPortFlowResult FlowSolver::solve_depth_averaged_ports(
    const SignalGridSpec& spec, std::span<const FlowPort> ports, std::span<const float> mobility,
    const DepthAveragedPortFlowParameters& parameters) {
  parameters.validate();
  const detail::ShallowPortFlow flow(spec, ports, mobility);
  DepthAveragedPortFlowReport report;
  const auto solution = flow.solve([&](const detail::DepthAveragedFlowSystem& system) {
    PcgReport solve;
    auto velocity = impl_->solve_depth_system(system, parameters.relative_tolerance,
                                              parameters.max_iterations, solve);
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

ResolvedFlowResult FlowSolver::solve_resolved(const SignalGridSpec& spec,
                                              std::span<const float> drag,
                                              const ResolvedFlowParameters& parameters) {
  return detail::solve_resolved_axis_flow(
      spec, drag, parameters,
      [&](const detail::ResolvedFlowSystem& system, const detail::ResolvedSolveControls& controls) {
        return impl_->solve_resolved_system(system, controls);
      });
}

ResolvedPortFlowResult FlowSolver::solve_resolved_ports(
    const SignalGridSpec& spec, std::span<const FlowPort> ports, std::span<const float> drag,
    const ResolvedPortFlowParameters& parameters) {
  return detail::solve_resolved_port_flow(
      spec, ports, drag, parameters,
      [&](const detail::ResolvedFlowSystem& system, const detail::ResolvedSolveControls& controls) {
        return impl_->solve_resolved_system(system, controls);
      });
}

}  // namespace microsimulator::metal
