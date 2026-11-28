#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/boundary_values.hpp"
#include "core/contact_response.hpp"
#include "core/contact_solve_plan.hpp"
#include "core/contact_solver.hpp"
#include "core/device_types.hpp"
#include "core/shared_constants.hpp"
#include "core/signal_jacobi.hpp"
#include "metal_cell_kernels.hpp"
#include "metal_compile_options.hpp"
#include "metal_contact_step.hpp"
#include "metal_flow.hpp"
#include "microsimulator/backend.hpp"
#include "microsimulator/metal/contact_dynamics_source.hpp"
#include "microsimulator/metal/coupled_rates_source.hpp"
#include "microsimulator/metal/growth_source.hpp"
#include "microsimulator/metal/signals_source.hpp"
#include "microsimulator/metal/species_source.hpp"

namespace microsimulator {
namespace {

using MetalFloat2 = device::float2;
using MetalUInt2 = device::uint2;
using MetalFloat4 = device::float4;
using MetalUInt4 = device::uint4;
using MetalContactResponse = device::ContactResponse;
using MetalContactRow = device::ContactRowData;
using MetalContactBody = device::ContactBodyData;
using MetalContactSolve = device::ContactSolveData;
using MetalContactState = device::ContactSolveState;
using MetalRateInstruction = device::RateInstruction;
using MetalGridShape = device::GridShape;
using MetalSweepState = device::SignalSweepState;

[[noreturn]] void throw_metal_error(const char* operation, NSError* error) {
  const char* detail = error == nil ? "unknown Metal error" : error.localizedDescription.UTF8String;
  throw std::runtime_error(std::string(operation) + ": " + detail);
}

id<MTLLibrary> compile_library(id<MTLDevice> device, const char* source_text, const char* operation,
                               MTLCompileOptions* options = nil) {
  NSString* source = [NSString stringWithUTF8String:source_text];
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

id<MTLComputePipelineState> compile_pipeline(id<MTLDevice> device, id<MTLLibrary> library,
                                             NSString* function_name, const char* operation) {
  id<MTLFunction> function = [library newFunctionWithName:function_name];
  if (function == nil) {
    throw std::runtime_error(std::string(operation) + ": function is missing from the library");
  }
  NSError* error = nil;
  id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                               error:&error];
  if (pipeline == nil) {
    throw_metal_error(operation, error);
  }
  return pipeline;
}

id<MTLBuffer> allocate_shared_buffer(id<MTLDevice> device, std::size_t byte_count,
                                     const char* description) {
  id<MTLBuffer> buffer = [device newBufferWithLength:byte_count
                                             options:MTLResourceStorageModeShared];
  if (buffer == nil) {
    throw std::runtime_error(std::string("failed to allocate Metal ") + description);
  }
  return buffer;
}

// Writes a grid's boundary kinds and fixed reservoir concentrations as the
// transport kernels read them, growing `buffer` to hold the concentrations.
// The result is the six boundary kinds, the six block offsets of
// detail::FixedBoundaryTable, and the grid's advection scheme.
std::array<std::uint32_t, 13> write_boundaries(id<MTLDevice> device, const SignalGridSpec& spec,
                                               id<MTLBuffer> __strong& buffer,
                                               const char* description) {
  const auto table = detail::fixed_boundary_table(spec);
  const auto byte_count = table.values.size() * sizeof(float);
  if (buffer == nil || buffer.length < byte_count) {
    buffer = allocate_shared_buffer(device, std::bit_ceil(byte_count), description);
  }
  std::memcpy(buffer.contents, table.values.data(), byte_count);
  constexpr std::array<GridFace, 6> faces{GridFace::x_lower, GridFace::x_upper, GridFace::y_lower,
                                          GridFace::y_upper, GridFace::z_lower, GridFace::z_upper};
  std::array<std::uint32_t, 13> boundaries{};
  for (std::size_t face = 0; face < faces.size(); ++face) {
    boundaries[face] = static_cast<std::uint32_t>(spec.boundary(faces[face]).kind);
    boundaries[faces.size() + face] = table.offsets[face];
  }
  boundaries[12] = static_cast<std::uint32_t>(spec.advection_scheme);
  return boundaries;
}

void wait_for_command(id<MTLCommandBuffer> command_buffer, const char* operation) {
  [command_buffer commit];
  [command_buffer waitUntilCompleted];
  if (command_buffer.status == MTLCommandBufferStatusError) {
    throw_metal_error(operation, command_buffer.error);
  }
}

void dispatch_1d(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline,
                 std::uint32_t count) {
  if (count == 0) {
    return;
  }
  const auto width = std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256);
  [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
}

id<MTLDevice> select_metal_device(std::uint32_t device_index) {
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

class MetalBackend final : public ComputeBackend {
 public:
  explicit MetalBackend(std::uint32_t device_index) : device_index_(device_index) {
    @autoreleasepool {
      device_ = select_metal_device(device_index_);
      queue_ = [device_ newCommandQueue];
      if (queue_ == nil) {
        throw std::runtime_error("failed to create a Metal command queue");
      }

      const auto growth_library =
          compile_library(device_, metal::growth_source, "failed to compile Metal growth");
      growth_pipeline_ = compile_pipeline(device_, growth_library, @"advance_growth",
                                          "failed to create the Metal growth pipeline");

      const auto species_library =
          compile_library(device_, metal::species_source, "failed to compile Metal species");
      species_pipeline_ = compile_pipeline(device_, species_library, @"advance_species",
                                           "failed to create the Metal species pipeline");

      const auto signals_library =
          compile_library(device_, metal::signals_source, "failed to compile Metal signals");
      signals_pipeline_ = compile_pipeline(device_, signals_library, @"advance_signal_grid",
                                           "failed to create the Metal signals pipeline");
      signals_cn_jacobi_pipeline_ =
          compile_pipeline(device_, signals_library, @"crank_nicolson_jacobi",
                           "failed to create the Metal signal Jacobi pipeline");
      signals_cn_residual_pipeline_ =
          compile_pipeline(device_, signals_library, @"crank_nicolson_residual_terms",
                           "failed to create the Metal signal residual pipeline");
      signals_square_pipeline_ =
          compile_pipeline(device_, signals_library, @"signal_square_terms",
                           "failed to create the Metal signal square pipeline");
      signals_sweep_check_pipeline_ =
          compile_pipeline(device_, signals_library, @"signal_sweep_check",
                           "failed to create the Metal signal sweep check pipeline");
      signal_gate_open_ = allocate_shared_buffer(device_, sizeof(std::uint32_t), "signal gate");
      *static_cast<std::uint32_t*>(signal_gate_open_.contents) = 0;
      signal_sweep_state_ =
          allocate_shared_buffer(device_, sizeof(MetalSweepState), "signal sweep state");
      signals_reduce_pipeline_ =
          compile_pipeline(device_, signals_library, @"reduce_signal_sum_pairs",
                           "failed to create the Metal signal reduction pipeline");

      const auto coupled_library = compile_library(device_, metal::coupled_rates_source,
                                                   "failed to compile Metal coupled rates");
      coupled_cells_pipeline_ =
          compile_pipeline(device_, coupled_library, @"advance_coupled_cells",
                           "failed to create the Metal coupled-cell pipeline");
      coupled_grid_pipeline_ = compile_pipeline(device_, coupled_library, @"advance_coupled_grid",
                                                "failed to create the Metal coupled-grid pipeline");
      coupled_scatter_stencil_pipeline_ =
          compile_pipeline(device_, coupled_library, @"coupled_scatter_stencils",
                           "failed to create the Metal coupled scatter-stencil pipeline");
      coupled_scatter_count_pipeline_ =
          compile_pipeline(device_, coupled_library, @"coupled_scatter_count",
                           "failed to create the Metal coupled scatter-count pipeline");
      coupled_scatter_fill_pipeline_ =
          compile_pipeline(device_, coupled_library, @"coupled_scatter_fill",
                           "failed to create the Metal coupled scatter-fill pipeline");
      coupled_scatter_sort_pipeline_ =
          compile_pipeline(device_, coupled_library, @"coupled_scatter_sort",
                           "failed to create the Metal coupled scatter-sort pipeline");
      inclusive_scan_pipeline_ =
          compile_pipeline(device_, coupled_library, @"inclusive_scan_step",
                           "failed to create the Metal inclusive-scan pipeline");

      const auto contact_dynamics_library =
          compile_library(device_, metal::contact_dynamics_source,
                          "failed to compile Metal contact dynamics", metal::safe_math_options());
      contact_respond_pipeline_ =
          compile_pipeline(device_, contact_dynamics_library, @"contact_respond",
                           "failed to create the Metal contact-respond pipeline");
      contact_step_tree_pipeline_ =
          compile_pipeline(device_, contact_dynamics_library, @"contact_step_tree",
                           "failed to create the Metal contact-step pipeline");
      contact_tree_pipeline_ = compile_pipeline(device_, contact_dynamics_library, @"contact_tree",
                                                "failed to create the Metal contact-tree pipeline");
      contact_record_pipeline_ =
          compile_pipeline(device_, contact_dynamics_library, @"contact_record",
                           "failed to create the Metal contact-record pipeline");
      contact_solve_pipeline_ =
          compile_pipeline(device_, contact_dynamics_library, @"contact_solve",
                           "failed to create the Metal contact-solve pipeline");
      flow_solver_ = std::make_unique<metal::FlowSolver>(device_index_);
      cell_kernels_ = std::make_unique<metal::CellKernels>(device_index_);
    }
  }

  [[nodiscard]] BackendInfo info() const override {
    @autoreleasepool {
      const char* device_name = device_.name.UTF8String;
      return {
          .kind = BackendKind::metal,
          .name = "metal",
          .device = device_name == nullptr ? "unknown Apple GPU" : device_name,
          .device_index = device_index_,
          .native = true,
      };
    }
  }

  void advance_growth(WorldState& state, float dt) override {
    if (!std::isfinite(dt) || dt < 0.0F) {
      throw std::invalid_argument("time step must be finite and non-negative");
    }
    auto view = state.growth_state();
    if (view.lengths.empty()) {
      return;
    }
    if (view.lengths.size() > std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("Metal growth launch exceeds the uint32 index space");
    }
    ensure_growth_capacity(view.lengths.size());

    const auto byte_count = view.lengths.size_bytes();
    std::memcpy(lengths_.contents, view.lengths.data(), byte_count);
    std::memcpy(growth_rates_.contents, view.growth_rates.data(), byte_count);

    @autoreleasepool {
      id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      if (command_buffer == nil || encoder == nil) {
        throw std::runtime_error("failed to create a Metal growth command");
      }

      const auto count = static_cast<std::uint32_t>(view.lengths.size());
      [encoder setComputePipelineState:growth_pipeline_];
      [encoder setBuffer:lengths_ offset:0 atIndex:0];
      [encoder setBuffer:growth_rates_ offset:0 atIndex:1];
      [encoder setBytes:&dt length:sizeof(dt) atIndex:2];
      [encoder setBytes:&count length:sizeof(count) atIndex:3];

      const auto width = std::min<NSUInteger>(growth_pipeline_.maxTotalThreadsPerThreadgroup, 256);
      [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
      [encoder endEncoding];
      wait_for_command(command_buffer, "Metal growth command failed");
    }

    std::memcpy(view.lengths.data(), lengths_.contents, byte_count);
  }

  void advance_species(WorldState& state, const SpeciesRatePlan& plan,
                       std::span<const float> previous_lengths, float dt) override {
    if (!std::isfinite(dt) || dt < 0.0F) {
      throw std::invalid_argument("species time step must be finite and non-negative");
    }
    state.validate();
    plan.validate();
    if (plan.species_count() != state.species_count()) {
      throw std::invalid_argument("species rate plan and world state species counts disagree");
    }
    if (previous_lengths.size() != state.size()) {
      throw std::invalid_argument("previous cell lengths and world state cell counts disagree");
    }
    if (state.empty() || state.species_count() == 0) {
      return;
    }
    if (state.size() > std::numeric_limits<std::uint32_t>::max() ||
        state.species_count() > std::numeric_limits<std::uint32_t>::max() ||
        plan.instructions().size() > std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("Metal species launch exceeds the uint32 index space");
    }
    if (!std::ranges::all_of(previous_lengths,
                             [](float value) { return std::isfinite(value) && value >= 0.0F; })) {
      throw std::invalid_argument("previous cell lengths must be finite and non-negative");
    }
    if (state.size() > std::numeric_limits<std::size_t>::max() / state.species_count() ||
        state.size() > std::numeric_limits<std::size_t>::max() / plan.instructions().size()) {
      throw std::overflow_error("Metal species buffer size overflow");
    }

    const auto level_count = state.size() * state.species_count();
    const auto workspace_count = state.size() * plan.instructions().size();
    if (level_count > std::numeric_limits<std::uint32_t>::max() ||
        workspace_count > std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("Metal flattened species storage exceeds the uint32 index space");
    }
    if (level_count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        workspace_count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        plan.instructions().size() >
            std::numeric_limits<std::size_t>::max() / sizeof(MetalRateInstruction)) {
      throw std::overflow_error("Metal species allocation size overflow");
    }
    ensure_species_capacity(state.size(), level_count, plan.instructions().size(),
                            state.species_count(), workspace_count);

    const auto geometry = state.geometry_state();
    const auto attributes = state.cell_attributes();
    auto species_state = state.species_state();
    std::memcpy(species_levels_.contents, species_state.levels.data(),
                species_state.levels.size_bytes());
    std::memcpy(species_previous_lengths_.contents, previous_lengths.data(),
                previous_lengths.size_bytes());
    std::memcpy(species_growth_rates_.contents, attributes.growth_rates.data(),
                attributes.growth_rates.size_bytes());
    std::memcpy(species_cell_types_.contents, attributes.cell_types.data(),
                attributes.cell_types.size_bytes());
    auto* centers = static_cast<MetalFloat4*>(species_centers_.contents);
    auto* shapes = static_cast<MetalFloat4*>(species_geometry_.contents);
    for (std::size_t index = 0; index < state.size(); ++index) {
      centers[index] = {geometry.position_x[index], geometry.position_y[index],
                        geometry.position_z[index], 0.0F};
      shapes[index] = {geometry.lengths[index], geometry.radii[index], 0.0F, 0.0F};
    }
    auto* instructions = static_cast<MetalRateInstruction*>(species_instructions_.contents);
    for (std::size_t index = 0; index < plan.instructions().size(); ++index) {
      const auto& instruction = plan.instructions()[index];
      instructions[index] = {
          .operation = static_cast<std::uint32_t>(instruction.operation),
          .first = instruction.first,
          .second = instruction.second,
          .third = instruction.third,
          .value = instruction.value,
      };
    }
    std::memcpy(species_outputs_.contents, plan.outputs().data(), plan.outputs().size_bytes());
    *static_cast<std::uint32_t*>(species_error_.contents) = 0;

    const auto cell_count = static_cast<std::uint32_t>(state.size());
    const auto species_count = static_cast<std::uint32_t>(state.species_count());
    const auto instruction_count = static_cast<std::uint32_t>(plan.instructions().size());
    @autoreleasepool {
      id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      if (command_buffer == nil || encoder == nil) {
        throw std::runtime_error("failed to create a Metal species command");
      }
      [encoder setComputePipelineState:species_pipeline_];
      [encoder setBuffer:species_levels_ offset:0 atIndex:0];
      [encoder setBuffer:species_previous_lengths_ offset:0 atIndex:1];
      [encoder setBuffer:species_centers_ offset:0 atIndex:2];
      [encoder setBuffer:species_geometry_ offset:0 atIndex:3];
      [encoder setBuffer:species_growth_rates_ offset:0 atIndex:4];
      [encoder setBuffer:species_cell_types_ offset:0 atIndex:5];
      [encoder setBuffer:species_instructions_ offset:0 atIndex:6];
      [encoder setBuffer:species_outputs_ offset:0 atIndex:7];
      [encoder setBuffer:species_workspace_ offset:0 atIndex:8];
      [encoder setBuffer:species_error_ offset:0 atIndex:9];
      [encoder setBytes:&dt length:sizeof(dt) atIndex:10];
      [encoder setBytes:&species_count length:sizeof(species_count) atIndex:11];
      [encoder setBytes:&instruction_count length:sizeof(instruction_count) atIndex:12];
      [encoder setBytes:&cell_count length:sizeof(cell_count) atIndex:13];
      dispatch_1d(encoder, species_pipeline_, cell_count);
      [encoder endEncoding];
      wait_for_command(command_buffer, "Metal species command failed");
    }
    if (*static_cast<const std::uint32_t*>(species_error_.contents) != 0) {
      throw std::domain_error("Metal species kernel produced a non-finite value");
    }
    std::memcpy(species_state.levels.data(), species_levels_.contents,
                species_state.levels.size_bytes());
  }

  SignalSolveReport advance_signal_grid(SignalGrid& grid, float dt) override {
    grid.validate_step(dt);
    if (dt == 0.0F) {
      return {};
    }
    const auto& spec = grid.spec();
    const auto levels = grid.levels();
    const auto level_count = static_cast<std::uint32_t>(levels.size());
    const auto signal_count = spec.signal_count;
    const auto& statics = sync_grid(grid);
    ensure_signal_level_capacity(levels.size());
    const auto& boundary_kinds = statics.boundary_kinds;
    const auto has_velocity_field = statics.has_velocity_field;
    const auto fluid_count = statics.fluid_count;
    const auto fluid_level_count = signal_count * fluid_count;
    std::memcpy(signal_levels_.contents, levels.data(), levels.size_bytes());
    *static_cast<std::uint32_t*>(signal_error_.contents) = 0;

    const MetalGridShape shape{spec.shape.x, spec.shape.y, spec.shape.z,
                               static_cast<std::uint32_t>(spec.site_count())};
    const MetalFloat4 spacing{spec.spacing.x, spec.spacing.y, spec.spacing.z, 0.0F};
    const auto crank_nicolson = static_cast<std::uint32_t>(spec.integration);
    @autoreleasepool {
      id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      if (command_buffer == nil || encoder == nil) {
        throw std::runtime_error("failed to create a Metal signal-grid command");
      }
      [encoder setComputePipelineState:signals_pipeline_];
      [encoder setBuffer:signal_levels_ offset:0 atIndex:0];
      [encoder setBuffer:signal_output_ offset:0 atIndex:1];
      [encoder setBuffer:signal_diffusion_ offset:0 atIndex:2];
      [encoder setBuffer:signal_advection_ offset:0 atIndex:3];
      [encoder setBuffer:signal_fixed_values_ offset:0 atIndex:4];
      [encoder setBuffer:signal_error_ offset:0 atIndex:5];
      [encoder setBytes:boundary_kinds.data()
                 length:boundary_kinds.size() * sizeof(std::uint32_t)
                atIndex:6];
      [encoder setBytes:&shape length:sizeof(shape) atIndex:7];
      [encoder setBytes:&spacing length:sizeof(spacing) atIndex:8];
      [encoder setBytes:&dt length:sizeof(dt) atIndex:9];
      [encoder setBytes:&signal_count length:sizeof(signal_count) atIndex:10];
      [encoder setBytes:&level_count length:sizeof(level_count) atIndex:11];
      [encoder setBytes:&crank_nicolson length:sizeof(crank_nicolson) atIndex:12];
      [encoder setBuffer:signal_reaction_source_ offset:0 atIndex:13];
      [encoder setBuffer:signal_reaction_loss_ offset:0 atIndex:14];
      [encoder setBuffer:signal_obstacles_ offset:0 atIndex:15];
      [encoder setBuffer:signal_x_faces_ offset:0 atIndex:16];
      [encoder setBuffer:signal_y_faces_ offset:0 atIndex:17];
      [encoder setBuffer:signal_z_faces_ offset:0 atIndex:18];
      [encoder setBytes:&has_velocity_field length:sizeof(has_velocity_field) atIndex:19];
      [encoder setBuffer:signal_fluid_sites_ offset:0 atIndex:20];
      [encoder setBytes:&fluid_count length:sizeof(fluid_count) atIndex:21];
      dispatch_1d(encoder, signals_pipeline_, fluid_level_count);
      [encoder endEncoding];
      wait_for_command(command_buffer, "Metal signal-grid command failed");
    }
    if (*static_cast<const std::uint32_t*>(signal_error_.contents) != 0) {
      throw std::domain_error(
          "Metal signal-grid kernel produced a non-finite or negative concentration");
    }
    id<MTLBuffer> result_buffer = signal_output_;
    SignalSolveReport report;
    if (crank_nicolson != 0) {
      const auto solve = solve_signal_crank_nicolson(
          signal_levels_, signal_output_, signal_diffusion_, signal_advection_,
          signal_fixed_values_, signal_reaction_source_, signal_reaction_loss_, signal_obstacles_,
          signal_x_faces_, signal_y_faces_, signal_z_faces_, has_velocity_field, signal_error_,
          boundary_kinds, shape, spacing, (crank_nicolson == 2 ? dt : 0.5F * dt), signal_count,
          level_count, spec.solver,
          grid.layout().implicit_operator_scale(crank_nicolson == 2 ? dt : 0.5F * dt));
      result_buffer = solve.first;
      report = solve.second;
      if (!report.converged) {
        throw std::runtime_error(
            signal_solve_failure("Metal Implicit signal solve", spec, report.iterations));
      }
    }
    const auto* output = static_cast<const float*>(result_buffer.contents);
    grid.accept_step(std::vector<float>(output, output + levels.size()));
    return report;
  }

  SignalSolveReport advance_coupled(WorldState& state, SignalGrid& grid,
                                    const CoupledRatePlan& plan,
                                    std::span<const float> previous_lengths, float dt) override {
    validate_coupled_step(state, grid, plan, previous_lengths, dt);
    const auto checked_product = [](std::size_t left, std::size_t right, const char* name) {
      if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::overflow_error(std::string("Metal coupled ") + name + " size overflow");
      }
      return left * right;
    };
    const auto cell_count_size = state.size();
    const auto species_count_size = state.species_count();
    const auto signal_count_size = plan.signal_count();
    const auto instruction_count_size = plan.instructions().size();
    const auto species_level_count =
        checked_product(cell_count_size, species_count_size, "species level");
    const auto workspace_count =
        checked_product(cell_count_size, instruction_count_size, "workspace");
    const auto cell_signal_count =
        checked_product(cell_count_size, signal_count_size, "cell signal");
    const auto grid_level_count = grid.levels().size();
    const auto scatter_entry_count =
        checked_product(cell_count_size, scatter_stencil_sites, "scatter entry");
    for (const auto count : {cell_count_size, species_count_size, signal_count_size,
                             instruction_count_size, species_level_count, workspace_count,
                             cell_signal_count, grid_level_count, scatter_entry_count}) {
      if (count > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Metal coupled launch exceeds the uint32 index space");
      }
    }
    ensure_coupled_scatter_capacity(cell_count_size, grid.spec().site_count());
    ensure_coupled_capacity(cell_count_size, species_level_count, instruction_count_size,
                            species_count_size, signal_count_size, workspace_count,
                            cell_signal_count, grid_level_count);

    const auto geometry = state.geometry_state();
    const auto attributes = state.cell_attributes();
    auto species_state = state.species_state();
    const auto& spec = grid.spec();
    const auto grid_levels = grid.levels();
    if (!species_state.levels.empty()) {
      std::memcpy(coupled_species_levels_.contents, species_state.levels.data(),
                  species_state.levels.size_bytes());
    }
    if (!previous_lengths.empty()) {
      std::memcpy(coupled_previous_lengths_.contents, previous_lengths.data(),
                  previous_lengths.size_bytes());
      std::memcpy(coupled_growth_rates_.contents, attributes.growth_rates.data(),
                  attributes.growth_rates.size_bytes());
      std::memcpy(coupled_cell_types_.contents, attributes.cell_types.data(),
                  attributes.cell_types.size_bytes());
    }
    auto* centers = static_cast<MetalFloat4*>(coupled_centers_.contents);
    auto* cell_geometry = static_cast<MetalFloat4*>(coupled_geometry_.contents);
    for (std::size_t index = 0; index < cell_count_size; ++index) {
      centers[index] = {geometry.position_x[index], geometry.position_y[index],
                        geometry.position_z[index], 0.0F};
      cell_geometry[index] = {geometry.lengths[index], geometry.radii[index], 0.0F, 0.0F};
    }
    auto* instructions = static_cast<MetalRateInstruction*>(coupled_instructions_.contents);
    for (std::size_t index = 0; index < instruction_count_size; ++index) {
      const auto& instruction = plan.instructions()[index];
      instructions[index] = {
          .operation = static_cast<std::uint32_t>(instruction.operation),
          .first = instruction.first,
          .second = instruction.second,
          .third = instruction.third,
          .value = instruction.value,
      };
    }
    if (!plan.species_outputs().empty()) {
      std::memcpy(coupled_species_outputs_.contents, plan.species_outputs().data(),
                  plan.species_outputs().size_bytes());
    }
    std::memcpy(coupled_signal_outputs_.contents, plan.signal_outputs().data(),
                plan.signal_outputs().size_bytes());
    std::memcpy(coupled_grid_levels_.contents, grid_levels.data(), grid_levels.size_bytes());
    const auto& statics = sync_grid(grid);
    const auto& boundary_kinds = statics.boundary_kinds;
    const auto has_velocity_field = statics.has_velocity_field;
    *static_cast<std::uint32_t*>(coupled_error_.contents) = 0;

    const auto cell_count = static_cast<std::uint32_t>(cell_count_size);
    const auto species_count = static_cast<std::uint32_t>(species_count_size);
    const auto signal_count = static_cast<std::uint32_t>(signal_count_size);
    const auto instruction_count = static_cast<std::uint32_t>(instruction_count_size);
    const auto level_count = static_cast<std::uint32_t>(grid_level_count);
    const MetalGridShape shape{spec.shape.x, spec.shape.y, spec.shape.z,
                               static_cast<std::uint32_t>(spec.site_count())};
    const MetalFloat4 origin{spec.origin.x, spec.origin.y, spec.origin.z, 0.0F};
    const MetalFloat4 spacing{spec.spacing.x, spec.spacing.y, spec.spacing.z, 0.0F};
    const auto crank_nicolson = static_cast<std::uint32_t>(spec.integration);
    @autoreleasepool {
      id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      if (command_buffer == nil || encoder == nil) {
        throw std::runtime_error("failed to create a Metal coupled-rate command");
      }
      id<MTLBuffer> scatter_ends = coupled_scatter_counts_;
      if (cell_count != 0) {
        [encoder setComputePipelineState:coupled_cells_pipeline_];
        [encoder setBuffer:coupled_species_levels_ offset:0 atIndex:0];
        [encoder setBuffer:coupled_previous_lengths_ offset:0 atIndex:1];
        [encoder setBuffer:coupled_centers_ offset:0 atIndex:2];
        [encoder setBuffer:coupled_geometry_ offset:0 atIndex:3];
        [encoder setBuffer:coupled_growth_rates_ offset:0 atIndex:4];
        [encoder setBuffer:coupled_cell_types_ offset:0 atIndex:5];
        [encoder setBuffer:coupled_instructions_ offset:0 atIndex:6];
        [encoder setBuffer:coupled_species_outputs_ offset:0 atIndex:7];
        [encoder setBuffer:coupled_signal_outputs_ offset:0 atIndex:8];
        [encoder setBuffer:coupled_workspace_ offset:0 atIndex:9];
        [encoder setBuffer:coupled_grid_levels_ offset:0 atIndex:10];
        [encoder setBuffer:coupled_cell_signal_rates_ offset:0 atIndex:11];
        [encoder setBuffer:coupled_error_ offset:0 atIndex:12];
        [encoder setBytes:&shape length:sizeof(shape) atIndex:13];
        [encoder setBytes:&origin length:sizeof(origin) atIndex:14];
        [encoder setBytes:&spacing length:sizeof(spacing) atIndex:15];
        [encoder setBytes:&dt length:sizeof(dt) atIndex:16];
        [encoder setBytes:&species_count length:sizeof(species_count) atIndex:17];
        [encoder setBytes:&signal_count length:sizeof(signal_count) atIndex:18];
        [encoder setBytes:&instruction_count length:sizeof(instruction_count) atIndex:19];
        [encoder setBytes:&cell_count length:sizeof(cell_count) atIndex:20];
        [encoder setBuffer:signal_obstacles_ offset:0 atIndex:21];
        dispatch_1d(encoder, coupled_cells_pipeline_, cell_count);
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        scatter_ends = encode_coupled_scatter_lists(encoder, shape, origin, spacing, cell_count);
      }

      [encoder setComputePipelineState:coupled_grid_pipeline_];
      [encoder setBuffer:coupled_grid_levels_ offset:0 atIndex:0];
      [encoder setBuffer:coupled_grid_output_ offset:0 atIndex:1];
      [encoder setBuffer:signal_diffusion_ offset:0 atIndex:2];
      [encoder setBuffer:signal_advection_ offset:0 atIndex:3];
      [encoder setBuffer:signal_fixed_values_ offset:0 atIndex:4];
      [encoder setBuffer:coupled_centers_ offset:0 atIndex:5];
      [encoder setBuffer:coupled_cell_signal_rates_ offset:0 atIndex:6];
      [encoder setBuffer:coupled_error_ offset:0 atIndex:7];
      [encoder setBytes:boundary_kinds.data()
                 length:boundary_kinds.size() * sizeof(std::uint32_t)
                atIndex:8];
      [encoder setBytes:&shape length:sizeof(shape) atIndex:9];
      [encoder setBytes:&origin length:sizeof(origin) atIndex:10];
      [encoder setBytes:&spacing length:sizeof(spacing) atIndex:11];
      [encoder setBytes:&dt length:sizeof(dt) atIndex:12];
      [encoder setBytes:&signal_count length:sizeof(signal_count) atIndex:13];
      [encoder setBytes:&cell_count length:sizeof(cell_count) atIndex:14];
      [encoder setBytes:&level_count length:sizeof(level_count) atIndex:15];
      [encoder setBytes:&crank_nicolson length:sizeof(crank_nicolson) atIndex:16];
      [encoder setBuffer:signal_reaction_source_ offset:0 atIndex:17];
      [encoder setBuffer:signal_reaction_loss_ offset:0 atIndex:18];
      [encoder setBuffer:signal_obstacles_ offset:0 atIndex:19];
      [encoder setBuffer:signal_x_faces_ offset:0 atIndex:20];
      [encoder setBuffer:signal_y_faces_ offset:0 atIndex:21];
      [encoder setBuffer:signal_z_faces_ offset:0 atIndex:22];
      [encoder setBytes:&has_velocity_field length:sizeof(has_velocity_field) atIndex:23];
      [encoder setBuffer:scatter_ends offset:0 atIndex:24];
      [encoder setBuffer:coupled_scatter_cells_ offset:0 atIndex:25];
      dispatch_1d(encoder, coupled_grid_pipeline_, level_count);
      [encoder endEncoding];
      wait_for_command(command_buffer, "Metal coupled-rate command failed");
    }

    const auto error = *static_cast<const std::uint32_t*>(coupled_error_.contents);
    if (error != 0) {
      throw std::domain_error("Metal coupled-rate kernel produced an invalid value");
    }
    id<MTLBuffer> result_buffer = coupled_grid_output_;
    SignalSolveReport report;
    if (crank_nicolson != 0) {
      const auto solve = solve_signal_crank_nicolson(
          coupled_grid_levels_, coupled_grid_output_, signal_diffusion_, signal_advection_,
          signal_fixed_values_, signal_reaction_source_, signal_reaction_loss_, signal_obstacles_,
          signal_x_faces_, signal_y_faces_, signal_z_faces_, has_velocity_field, coupled_error_,
          boundary_kinds, shape, spacing, (crank_nicolson == 2 ? dt : 0.5F * dt), signal_count,
          level_count, spec.solver,
          grid.layout().implicit_operator_scale(crank_nicolson == 2 ? dt : 0.5F * dt));
      result_buffer = solve.first;
      report = solve.second;
      if (!report.converged) {
        throw std::runtime_error(
            signal_solve_failure("Metal Implicit coupled signal solve", spec, report.iterations));
      }
    }
    const auto* output = static_cast<const float*>(result_buffer.contents);
    std::vector<float> next_grid(output, output + grid_level_count);
    // The grid refuses levels it cannot hold, so it takes its own before the
    // species take theirs: a refused step changes neither.
    grid.accept_step(std::move(next_grid));
    if (!species_state.levels.empty()) {
      std::memcpy(species_state.levels.data(), coupled_species_levels_.contents,
                  species_state.levels.size_bytes());
    }
    return report;
  }

  [[nodiscard]] detail::ContactStepDevice& contact_step_device() override {
    if (!step_device_) {
      step_device_ = metal::make_contact_step_device(device_, queue_);
    }
    return *step_device_;
  }

  [[nodiscard]] ContactGraph find_cell_contacts(const WorldState& state,
                                                const ContactParameters& parameters) override {
    return cell_kernels_->cell_contacts(state, parameters);
  }

  [[nodiscard]] ExternalContactGraph find_external_contacts(
      const WorldState& state, const ConstraintSet& constraints,
      const ConstraintContactParameters& parameters) override {
    return cell_kernels_->external_contacts(state, constraints, parameters, false);
  }

  [[nodiscard]] ExternalContactGraph find_external_contact_faces(
      const WorldState& state, const ConstraintSet& constraints,
      const ConstraintContactParameters& parameters) override {
    return cell_kernels_->external_contacts(state, constraints, parameters, true);
  }

  [[nodiscard]] std::vector<std::array<float, 6>> flow_drift_rates(const WorldState& state,
                                                                   const SignalGrid& grid,
                                                                   float dt) override {
    return cell_kernels_->flow_drift_rates(state, grid, dt);
  }

  void advance_flow_drift(WorldState& state, const SignalGrid& grid, float dt,
                          const FlowDriftParameters& parameters) override {
    cell_kernels_->advance_flow_drift(state, grid, dt, parameters);
  }

  [[nodiscard]] ContactSolveResult solve_contact_forces(
      const ContactSystem& system, std::span<const float> initial_multipliers,
      const ContactSolveParameters& parameters) override {
    const auto plan = detail::plan_contact_solve(system, initial_multipliers, parameters.tolerance);
    ContactSolveResult result;
    result.multipliers.assign(initial_multipliers.begin(), initial_multipliers.end());
    const auto row_count = plan.data.row_count;
    const auto body_count = plan.data.body_count;
    if (body_count == 0) {
      return result;
    }
    ensure_contact_dynamics_capacity(body_count, std::max<std::uint32_t>(row_count, 1));
    upload_contact_plan(plan);
    if (row_count == 0) {
      // Without contacts every cell moves with its drift and grows freely.
      respond_to_contacts(plan.data);
      result.response = download_contact_response(body_count);
      return result;
    }
    const auto bound = contact_system_diagonal_bound(system);
    if (!std::isfinite(bound) || bound <= 0.0F) {
      throw std::invalid_argument("contact system has no positive diagonal");
    }
    const auto first_step_length = 1.0F / bound;
    const auto group_width = contact_solve_group_width();
    if (row_count <= contact_solve_group_rows && group_width > 0) {
      // One threadgroup runs the whole solve, so it costs one command.
      run_contact_solve(plan.data, parameters.max_iterations, first_step_length, group_width);
    } else {
      run_contact_steps(plan.data, parameters.max_iterations, first_step_length);
    }
    const auto state = *static_cast<const MetalContactState*>(contact_state_.contents);
    result.report =
        detail::device_contact_solve_report(state.iterations, state.residual, parameters.tolerance);
    const auto* multipliers =
        static_cast<const float*>(contact_multipliers_[state.iterations % 2].contents);
    result.multipliers.assign(multipliers, multipliers + row_count);
    result.response = download_contact_response(body_count);
    return result;
  }

  [[nodiscard]] DepthAveragedFlowResult solve_depth_averaged_flow(
      const SignalGridSpec& spec, std::span<const float> mobility,
      const DepthAveragedFlowParameters& parameters) override {
    return flow_solver_->solve_depth_averaged(spec, mobility, parameters);
  }

  [[nodiscard]] DepthAveragedPortFlowResult solve_depth_averaged_port_flow(
      const SignalGridSpec& spec, std::span<const FlowPort> ports, std::span<const float> mobility,
      const DepthAveragedPortFlowParameters& parameters) override {
    return flow_solver_->solve_depth_averaged_ports(spec, ports, mobility, parameters);
  }

  [[nodiscard]] ResolvedFlowResult solve_resolved_flow(
      const SignalGridSpec& spec, std::span<const float> drag,
      const ResolvedFlowParameters& parameters) override {
    return flow_solver_->solve_resolved(spec, drag, parameters);
  }

  [[nodiscard]] ResolvedPortFlowResult solve_resolved_port_flow(
      const SignalGridSpec& spec, std::span<const FlowPort> ports, std::span<const float> drag,
      const ResolvedPortFlowParameters& parameters) override {
    return flow_solver_->solve_resolved_ports(spec, ports, drag, parameters);
  }

  [[nodiscard]] BiomassDeposit deposit_biomass(
      const SignalGridSpec& spec, const DepositCells& cells,
      const BiomassDepositParameters& parameters) override {
    return cell_kernels_->deposit(spec, cells, parameters);
  }

  [[nodiscard]] ShallowFlowCoupling shallow_flow_coupling(
      const SignalGridSpec& spec, std::span<const float> base, const BiomassDeposit& deposit,
      const BiomassDragParameters& parameters) override {
    return cell_kernels_->shallow_flow_coupling(spec, base, deposit, parameters);
  }

  [[nodiscard]] std::vector<float> brinkman_drag(const SignalGridSpec& spec,
                                                 const BiomassDeposit& deposit,
                                                 const BiomassDragParameters& parameters) override {
    return cell_kernels_->brinkman_drag(spec, deposit, parameters);
  }

 private:
  void ensure_growth_capacity(std::size_t count) {
    if (count <= growth_capacity_) {
      return;
    }
    const auto growth_capacity = std::bit_ceil(count);
    const auto byte_count = growth_capacity * sizeof(float);
    lengths_ = allocate_shared_buffer(device_, byte_count, "growth lengths");
    growth_rates_ = allocate_shared_buffer(device_, byte_count, "growth rates");
    growth_capacity_ = growth_capacity;
  }

  void ensure_species_capacity(std::size_t cell_count, std::size_t level_count,
                               std::size_t instruction_count, std::size_t species_count,
                               std::size_t workspace_count) {
    if (cell_count > species_cell_capacity_) {
      const auto species_cell_capacity = std::bit_ceil(cell_count);
      species_previous_lengths_ = allocate_shared_buffer(
          device_, species_cell_capacity * sizeof(float), "species previous lengths");
      species_centers_ = allocate_shared_buffer(
          device_, species_cell_capacity * sizeof(MetalFloat4), "species cell centers");
      species_geometry_ = allocate_shared_buffer(
          device_, species_cell_capacity * sizeof(MetalFloat4), "species cell geometry");
      species_growth_rates_ = allocate_shared_buffer(device_, species_cell_capacity * sizeof(float),
                                                     "species growth rates");
      species_cell_types_ = allocate_shared_buffer(
          device_, species_cell_capacity * sizeof(std::int32_t), "species cell types");
      species_cell_capacity_ = species_cell_capacity;
    }
    if (level_count > species_level_capacity_) {
      const auto species_level_capacity = std::bit_ceil(level_count);
      species_levels_ =
          allocate_shared_buffer(device_, species_level_capacity * sizeof(float), "species levels");
      species_level_capacity_ = species_level_capacity;
    }
    if (instruction_count > species_instruction_capacity_) {
      const auto species_instruction_capacity = std::bit_ceil(instruction_count);
      species_instructions_ = allocate_shared_buffer(
          device_, species_instruction_capacity * sizeof(MetalRateInstruction),
          "species rate instructions");
      species_instruction_capacity_ = species_instruction_capacity;
    }
    if (species_count > species_output_capacity_) {
      const auto species_output_capacity = std::bit_ceil(species_count);
      species_outputs_ = allocate_shared_buffer(
          device_, species_output_capacity * sizeof(std::uint32_t), "species rate outputs");
      species_output_capacity_ = species_output_capacity;
    }
    if (workspace_count > species_workspace_capacity_) {
      const auto species_workspace_capacity = std::bit_ceil(workspace_count);
      species_workspace_ = allocate_shared_buffer(
          device_, species_workspace_capacity * sizeof(float), "species rate workspace");
      species_workspace_capacity_ = species_workspace_capacity;
    }
    if (species_error_ == nil) {
      species_error_ = allocate_shared_buffer(device_, sizeof(std::uint32_t), "species error flag");
    }
  }

  static std::size_t largest_face_count(const SignalGridSpec& spec) {
    return std::max({spec.x_face_count(), spec.y_face_count(), spec.z_face_count(),
                     static_cast<std::size_t>(1)});
  }

  void fill_velocity_faces(const SignalGridSpec& spec, id<MTLBuffer> x_buffer,
                           id<MTLBuffer> y_buffer, id<MTLBuffer> z_buffer) {
    if (!spec.velocity_field.has_value()) {
      return;
    }
    const auto& field = *spec.velocity_field;
    std::memcpy(x_buffer.contents, field.x_faces.data(), field.x_faces.size() * sizeof(float));
    std::memcpy(y_buffer.contents, field.y_faces.data(), field.y_faces.size() * sizeof(float));
    std::memcpy(z_buffer.contents, field.z_faces.data(), field.z_faces.size() * sizeof(float));
  }

  // What the transport kernels are told of the grid on the device.
  struct GridStatics {
    // The six boundary kinds, the six block offsets of
    // detail::FixedBoundaryTable, and the advection scheme.
    std::array<std::uint32_t, 13> boundary_kinds{};
    std::uint32_t has_velocity_field{0};
    std::uint32_t fluid_count{0};
  };

  // Puts on the device everything transport reads of a grid besides its
  // levels: coefficients, reaction field, obstacles, face velocities,
  // boundary values and the list of fluid sites. They change only with the
  // grid's specification, so a step whose grid the device already holds
  // uploads nothing.
  const GridStatics& sync_grid(const SignalGrid& grid) {
    if (grid.revision() == grid_revision_) {
      return grid_statics_;
    }
    grid_revision_ = 0;
    const auto& spec = grid.spec();
    const auto level_count = spec.level_count();
    const auto sites = spec.site_count();
    ensure_grid_static_capacity(level_count, spec.signal_count);
    std::memcpy(signal_diffusion_.contents, spec.diffusion.data(),
                spec.diffusion.size() * sizeof(float));
    auto* advection = static_cast<MetalFloat4*>(signal_advection_.contents);
    for (std::size_t signal = 0; signal < spec.signal_count; ++signal) {
      advection[signal] = {spec.advection[signal].x, spec.advection[signal].y,
                           spec.advection[signal].z, 0.0F};
    }
    auto* reaction_source = static_cast<float*>(signal_reaction_source_.contents);
    auto* reaction_loss = static_cast<float*>(signal_reaction_loss_.contents);
    if (spec.reaction.has_value()) {
      std::memcpy(reaction_source, spec.reaction->source_rates.data(), level_count * sizeof(float));
      std::memcpy(reaction_loss, spec.reaction->loss_rates.data(), level_count * sizeof(float));
    } else {
      std::fill_n(reaction_source, level_count, 0.0F);
      std::fill_n(reaction_loss, level_count, 0.0F);
    }
    // The kernels write fluid levels only, so their output buffers must hold
    // zero wherever this grid has an obstacle. They do unless the device last
    // held a grid with other obstacles.
    auto* obstacles = static_cast<std::uint8_t*>(signal_obstacles_.contents);
    auto obstacles_differ = grid_site_count_ != sites;
    if (spec.has_obstacles()) {
      obstacles_differ =
          obstacles_differ || std::memcmp(obstacles, spec.obstacles.data(), sites) != 0;
      std::memcpy(obstacles, spec.obstacles.data(), sites);
    } else {
      obstacles_differ =
          obstacles_differ ||
          std::any_of(obstacles, obstacles + sites, [](std::uint8_t value) { return value != 0; });
      std::fill_n(obstacles, sites, std::uint8_t{0});
    }
    if (obstacles_differ) {
      for (id<MTLBuffer> buffer : {signal_output_, signal_cn_a_, signal_cn_b_}) {
        if (buffer != nil) {
          std::memset(buffer.contents, 0, buffer.length);
        }
      }
    }
    ensure_signal_face_capacity(largest_face_count(spec));
    fill_velocity_faces(spec, signal_x_faces_, signal_y_faces_, signal_z_faces_);
    grid_statics_.boundary_kinds =
        write_boundaries(device_, spec, signal_fixed_values_, "signal-grid boundary values");
    grid_statics_.has_velocity_field = static_cast<std::uint32_t>(spec.velocity_field.has_value());
    const std::span<const std::uint32_t> fluid = grid.layout().fluid_sites;
    const auto fluid_bytes = std::max<std::size_t>(fluid.size(), 1) * sizeof(std::uint32_t);
    if (signal_fluid_sites_ == nil || signal_fluid_sites_.length < fluid_bytes) {
      signal_fluid_sites_ =
          allocate_shared_buffer(device_, std::bit_ceil(fluid_bytes), "signal-grid fluid sites");
    }
    std::memcpy(signal_fluid_sites_.contents, fluid.data(), fluid.size_bytes());
    grid_statics_.fluid_count = static_cast<std::uint32_t>(fluid.size());
    grid_site_count_ = sites;
    grid_revision_ = grid.revision();
    return grid_statics_;
  }

  void ensure_grid_static_capacity(std::size_t level_count, std::size_t signal_count) {
    if (level_count > grid_static_capacity_) {
      const auto capacity = std::bit_ceil(level_count);
      const auto byte_count = capacity * sizeof(float);
      signal_reaction_source_ =
          allocate_shared_buffer(device_, byte_count, "signal-grid affine sources");
      signal_reaction_loss_ =
          allocate_shared_buffer(device_, byte_count, "signal-grid affine losses");
      signal_obstacles_ = allocate_shared_buffer(device_, capacity, "signal-grid obstacles");
      grid_static_capacity_ = capacity;
      grid_site_count_ = 0;
    }
    if (signal_count > signal_count_capacity_) {
      const auto signal_count_capacity = std::bit_ceil(signal_count);
      signal_diffusion_ = allocate_shared_buffer(device_, signal_count_capacity * sizeof(float),
                                                 "signal-grid diffusion");
      signal_advection_ = allocate_shared_buffer(
          device_, signal_count_capacity * sizeof(MetalFloat4), "signal-grid advection");
      signal_count_capacity_ = signal_count_capacity;
    }
  }

  void ensure_signal_level_capacity(std::size_t level_count) {
    if (level_count > signal_level_capacity_) {
      const auto signal_level_capacity = std::bit_ceil(level_count);
      const auto byte_count = signal_level_capacity * sizeof(float);
      signal_levels_ = allocate_shared_buffer(device_, byte_count, "signal-grid levels");
      signal_output_ = allocate_shared_buffer(device_, byte_count, "signal-grid output");
      signal_level_capacity_ = signal_level_capacity;
    }
    if (signal_error_ == nil) {
      signal_error_ =
          allocate_shared_buffer(device_, sizeof(std::uint32_t), "signal-grid error flag");
    }
  }

  void ensure_signal_face_capacity(std::size_t face_count) {
    if (face_count > signal_face_capacity_) {
      const auto signal_face_capacity = std::bit_ceil(face_count);
      const auto byte_count = signal_face_capacity * sizeof(float);
      signal_x_faces_ = allocate_shared_buffer(device_, byte_count, "signal-grid x faces");
      signal_y_faces_ = allocate_shared_buffer(device_, byte_count, "signal-grid y faces");
      signal_z_faces_ = allocate_shared_buffer(device_, byte_count, "signal-grid z faces");
      signal_face_capacity_ = signal_face_capacity;
    }
  }

  void ensure_signal_solve_capacity(std::uint32_t level_count) {
    if (level_count <= signal_solve_capacity_) {
      return;
    }
    const auto signal_solve_capacity = std::bit_ceil(static_cast<std::size_t>(level_count));
    const auto byte_count = signal_solve_capacity * sizeof(float);
    signal_cn_a_ = allocate_shared_buffer(device_, byte_count, "signal Jacobi field A");
    signal_cn_b_ = allocate_shared_buffer(device_, byte_count, "signal Jacobi field B");
    signal_cn_terms_ = allocate_shared_buffer(device_, byte_count, "signal residual terms");
    signal_cn_reduce_a_ = allocate_shared_buffer(device_, byte_count, "signal reduction A");
    signal_cn_reduce_b_ = allocate_shared_buffer(device_, byte_count, "signal reduction B");
    signal_solve_capacity_ = signal_solve_capacity;
  }

  // Encodes the sum of the first `element_count` terms. Its dispatches do
  // nothing while `gate`, when given, holds a nonzero value.
  id<MTLBuffer> encode_signal_reduction(id<MTLComputeCommandEncoder> encoder,
                                        std::uint32_t element_count, id<MTLBuffer> gate = nil) {
    if (gate == nil) {
      gate = signal_gate_open_;
    }
    id<MTLBuffer> input = signal_cn_terms_;
    id<MTLBuffer> output = signal_cn_reduce_a_;
    auto count = element_count;
    while (count > 1) {
      const auto output_count = (count + 1) / 2;
      [encoder setComputePipelineState:signals_reduce_pipeline_];
      [encoder setBuffer:input offset:0 atIndex:0];
      [encoder setBuffer:output offset:0 atIndex:1];
      [encoder setBytes:&count length:sizeof(count) atIndex:2];
      [encoder setBuffer:gate offset:0 atIndex:3];
      dispatch_1d(encoder, signals_reduce_pipeline_, output_count);
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      input = output;
      output = output == signal_cn_reduce_a_ ? signal_cn_reduce_b_ : signal_cn_reduce_a_;
      count = output_count;
    }
    return input;
  }

  id<MTLBuffer> encode_signal_residual(
      id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> current, id<MTLBuffer> right_hand_side,
      id<MTLBuffer> diffusion, id<MTLBuffer> advection, id<MTLBuffer> fixed_values,
      id<MTLBuffer> reaction_source, id<MTLBuffer> reaction_loss, id<MTLBuffer> obstacles,
      id<MTLBuffer> x_faces, id<MTLBuffer> y_faces, id<MTLBuffer> z_faces,
      std::uint32_t has_velocity_field, const std::array<std::uint32_t, 13>& boundary_kinds,
      const MetalGridShape& shape, const MetalFloat4& spacing, float half_dt,
      std::uint32_t signal_count, std::uint32_t level_count, id<MTLBuffer> gate = nil) {
    if (gate == nil) {
      gate = signal_gate_open_;
    }
    [encoder setComputePipelineState:signals_cn_residual_pipeline_];
    [encoder setBuffer:current offset:0 atIndex:0];
    [encoder setBuffer:right_hand_side offset:0 atIndex:1];
    [encoder setBuffer:signal_cn_terms_ offset:0 atIndex:2];
    [encoder setBuffer:diffusion offset:0 atIndex:3];
    [encoder setBuffer:advection offset:0 atIndex:4];
    [encoder setBuffer:fixed_values offset:0 atIndex:5];
    [encoder setBytes:boundary_kinds.data()
               length:boundary_kinds.size() * sizeof(std::uint32_t)
              atIndex:6];
    [encoder setBytes:&shape length:sizeof(shape) atIndex:7];
    [encoder setBytes:&spacing length:sizeof(spacing) atIndex:8];
    [encoder setBytes:&half_dt length:sizeof(half_dt) atIndex:9];
    [encoder setBytes:&signal_count length:sizeof(signal_count) atIndex:10];
    [encoder setBytes:&level_count length:sizeof(level_count) atIndex:11];
    [encoder setBuffer:reaction_source offset:0 atIndex:12];
    [encoder setBuffer:reaction_loss offset:0 atIndex:13];
    [encoder setBuffer:obstacles offset:0 atIndex:14];
    [encoder setBuffer:x_faces offset:0 atIndex:15];
    [encoder setBuffer:y_faces offset:0 atIndex:16];
    [encoder setBuffer:z_faces offset:0 atIndex:17];
    [encoder setBytes:&has_velocity_field length:sizeof(has_velocity_field) atIndex:18];
    const auto fluid_count = grid_statics_.fluid_count;
    const auto fluid_level_count = signal_count * fluid_count;
    [encoder setBuffer:signal_fluid_sites_ offset:0 atIndex:19];
    [encoder setBytes:&fluid_count length:sizeof(fluid_count) atIndex:20];
    [encoder setBuffer:gate offset:0 atIndex:21];
    dispatch_1d(encoder, signals_cn_residual_pipeline_, fluid_level_count);
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    return encode_signal_reduction(encoder, fluid_level_count, gate);
  }

  [[nodiscard]] float signal_residual_rms(id<MTLBuffer> current, id<MTLBuffer> right_hand_side,
                                          id<MTLBuffer> diffusion, id<MTLBuffer> advection,
                                          id<MTLBuffer> fixed_values, id<MTLBuffer> reaction_source,
                                          id<MTLBuffer> reaction_loss, id<MTLBuffer> obstacles,
                                          id<MTLBuffer> x_faces, id<MTLBuffer> y_faces,
                                          id<MTLBuffer> z_faces, std::uint32_t has_velocity_field,
                                          const std::array<std::uint32_t, 13>& boundary_kinds,
                                          const MetalGridShape& shape, const MetalFloat4& spacing,
                                          float half_dt, std::uint32_t signal_count,
                                          std::uint32_t level_count) {
    @autoreleasepool {
      id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      if (command_buffer == nil || encoder == nil) {
        throw std::runtime_error("failed to create a Metal signal residual command");
      }
      const auto reduction = encode_signal_residual(
          encoder, current, right_hand_side, diffusion, advection, fixed_values, reaction_source,
          reaction_loss, obstacles, x_faces, y_faces, z_faces, has_velocity_field, boundary_kinds,
          shape, spacing, half_dt, signal_count, level_count);
      [encoder endEncoding];
      wait_for_command(command_buffer, "Metal signal residual failed");
      const auto sum =
          grid_statics_.fluid_count == 0 ? 0.0F : *static_cast<const float*>(reduction.contents);
      return std::sqrt(sum / static_cast<float>(level_count));
    }
  }

  [[nodiscard]] float signal_rhs_rms(id<MTLBuffer> right_hand_side, std::uint32_t signal_count,
                                     std::uint32_t level_count) {
    const auto fluid_count = grid_statics_.fluid_count;
    const auto fluid_level_count = signal_count * fluid_count;
    const auto sites = level_count / signal_count;
    if (fluid_level_count == 0) {
      return 0.0F;
    }
    @autoreleasepool {
      id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      if (command_buffer == nil || encoder == nil) {
        throw std::runtime_error("failed to create a Metal signal norm command");
      }
      [encoder setComputePipelineState:signals_square_pipeline_];
      [encoder setBuffer:right_hand_side offset:0 atIndex:0];
      [encoder setBuffer:signal_cn_terms_ offset:0 atIndex:1];
      [encoder setBytes:&fluid_level_count length:sizeof(fluid_level_count) atIndex:2];
      [encoder setBuffer:signal_fluid_sites_ offset:0 atIndex:3];
      [encoder setBytes:&fluid_count length:sizeof(fluid_count) atIndex:4];
      [encoder setBytes:&sites length:sizeof(sites) atIndex:5];
      dispatch_1d(encoder, signals_square_pipeline_, fluid_level_count);
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      const auto reduction = encode_signal_reduction(encoder, fluid_level_count);
      [encoder endEncoding];
      wait_for_command(command_buffer, "Metal signal norm failed");
      const auto sum = *static_cast<const float*>(reduction.contents);
      return std::sqrt(sum / static_cast<float>(level_count));
    }
  }

  [[nodiscard]] std::pair<id<MTLBuffer>, SignalSolveReport> solve_signal_crank_nicolson(
      id<MTLBuffer> initial, id<MTLBuffer> right_hand_side, id<MTLBuffer> diffusion,
      id<MTLBuffer> advection, id<MTLBuffer> fixed_values, id<MTLBuffer> reaction_source,
      id<MTLBuffer> reaction_loss, id<MTLBuffer> obstacles, id<MTLBuffer> x_faces,
      id<MTLBuffer> y_faces, id<MTLBuffer> z_faces, std::uint32_t has_velocity_field,
      id<MTLBuffer> error, const std::array<std::uint32_t, 13>& boundary_kinds,
      const MetalGridShape& shape, const MetalFloat4& spacing, float dt, std::uint32_t signal_count,
      std::uint32_t level_count, const SignalSolveParameters& parameters, float operator_scale) {
    ensure_signal_solve_capacity(level_count);
    const auto half_dt = dt;
    const auto fluid_count = grid_statics_.fluid_count;
    const auto fluid_level_count = signal_count * fluid_count;
    id<MTLBuffer> current = initial;
    // Encodes one Jacobi sweep from `from` into `to` and its residual terms'
    // sum, all skipped while `gate` holds a nonzero value. Returns the buffer
    // that holds the sum.
    const auto encode_sweep = [&](id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> from,
                                  id<MTLBuffer> to, id<MTLBuffer> gate) {
      [encoder setComputePipelineState:signals_cn_jacobi_pipeline_];
      [encoder setBuffer:from offset:0 atIndex:0];
      [encoder setBuffer:to offset:0 atIndex:1];
      [encoder setBuffer:right_hand_side offset:0 atIndex:2];
      [encoder setBuffer:diffusion offset:0 atIndex:3];
      [encoder setBuffer:advection offset:0 atIndex:4];
      [encoder setBuffer:fixed_values offset:0 atIndex:5];
      [encoder setBuffer:error offset:0 atIndex:6];
      [encoder setBytes:boundary_kinds.data()
                 length:boundary_kinds.size() * sizeof(std::uint32_t)
                atIndex:7];
      [encoder setBytes:&shape length:sizeof(shape) atIndex:8];
      [encoder setBytes:&spacing length:sizeof(spacing) atIndex:9];
      [encoder setBytes:&half_dt length:sizeof(half_dt) atIndex:10];
      [encoder setBytes:&signal_count length:sizeof(signal_count) atIndex:11];
      [encoder setBytes:&level_count length:sizeof(level_count) atIndex:12];
      [encoder setBuffer:reaction_source offset:0 atIndex:13];
      [encoder setBuffer:reaction_loss offset:0 atIndex:14];
      [encoder setBuffer:obstacles offset:0 atIndex:15];
      [encoder setBuffer:x_faces offset:0 atIndex:16];
      [encoder setBuffer:y_faces offset:0 atIndex:17];
      [encoder setBuffer:z_faces offset:0 atIndex:18];
      [encoder setBytes:&has_velocity_field length:sizeof(has_velocity_field) atIndex:19];
      [encoder setBuffer:signal_fluid_sites_ offset:0 atIndex:20];
      [encoder setBytes:&fluid_count length:sizeof(fluid_count) atIndex:21];
      [encoder setBuffer:gate offset:0 atIndex:22];
      dispatch_1d(encoder, signals_cn_jacobi_pipeline_, fluid_level_count);
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      return encode_signal_residual(encoder, to, right_hand_side, diffusion, advection,
                                    fixed_values, reaction_source, reaction_loss, obstacles,
                                    x_faces, y_faces, z_faces, has_velocity_field, boundary_kinds,
                                    shape, spacing, half_dt, signal_count, level_count, gate);
    };
    const detail::SignalJacobiOperations operations{
        .initial_residual_rms =
            [&] {
              return signal_residual_rms(
                  initial, right_hand_side, diffusion, advection, fixed_values, reaction_source,
                  reaction_loss, obstacles, x_faces, y_faces, z_faces, has_velocity_field,
                  boundary_kinds, shape, spacing, half_dt, signal_count, level_count);
            },
        .right_hand_side_rms =
            [&] { return signal_rhs_rms(right_hand_side, signal_count, level_count); },
        .operator_scale = operator_scale,
        .begin_sweeps = [&] { *static_cast<std::uint32_t*>(error.contents) = 0; },
        .sweep_batch =
            [&](std::uint32_t count, float threshold) {
              auto* state = static_cast<MetalSweepState*>(signal_sweep_state_.contents);
              *state = {};
              // Sweeps write the two scratch iterates in turn, never the one the
              // solve started from.
              std::vector<id<MTLBuffer>> iterates{current};
              for (std::uint32_t sweep = 0; sweep < count; ++sweep) {
                iterates.push_back(iterates.back() == signal_cn_a_ ? signal_cn_b_ : signal_cn_a_);
              }
              @autoreleasepool {
                id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
                if (command_buffer == nil || encoder == nil) {
                  throw std::runtime_error("failed to create a Metal signal Jacobi command");
                }
                for (std::uint32_t sweep = 0; sweep < count; ++sweep) {
                  const auto reduction = encode_sweep(encoder, iterates[sweep], iterates[sweep + 1],
                                                      signal_sweep_state_);
                  [encoder setComputePipelineState:signals_sweep_check_pipeline_];
                  [encoder setBuffer:reduction offset:0 atIndex:0];
                  [encoder setBuffer:error offset:0 atIndex:1];
                  [encoder setBuffer:signal_sweep_state_ offset:0 atIndex:2];
                  [encoder setBytes:&threshold length:sizeof(threshold) atIndex:3];
                  [encoder setBytes:&level_count length:sizeof(level_count) atIndex:4];
                  [encoder setBytes:&fluid_level_count length:sizeof(fluid_level_count) atIndex:5];
                  dispatch_1d(encoder, signals_sweep_check_pipeline_, 1);
                  [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
                [encoder endEncoding];
                wait_for_command(command_buffer, "Metal signal Jacobi batch failed");
              }
              current = iterates[state->iterations];
              return detail::SignalSweepBatch{
                  .iterations = state->iterations,
                  .residual_rms = state->residual_rms,
                  .valid = state->invalid == 0,
              };
            },
        .batch = signal_sweep_batch,
    };
    const auto report = detail::solve_signal_jacobi(operations, parameters);
    return {current, report};
  }

  void ensure_coupled_capacity(std::size_t cell_count, std::size_t species_level_count,
                               std::size_t instruction_count, std::size_t species_count,
                               std::size_t signal_count, std::size_t workspace_count,
                               std::size_t cell_signal_count, std::size_t grid_level_count) {
    const auto at_least_one = [](std::size_t count) { return std::max<std::size_t>(count, 1); };
    const auto requested_cells = at_least_one(cell_count);
    if (requested_cells > coupled_cell_capacity_) {
      const auto coupled_cell_capacity = std::bit_ceil(requested_cells);
      coupled_previous_lengths_ = allocate_shared_buffer(
          device_, coupled_cell_capacity * sizeof(float), "coupled previous lengths");
      coupled_centers_ = allocate_shared_buffer(
          device_, coupled_cell_capacity * sizeof(MetalFloat4), "coupled cell centers");
      coupled_geometry_ = allocate_shared_buffer(
          device_, coupled_cell_capacity * sizeof(MetalFloat4), "coupled cell geometry");
      coupled_growth_rates_ = allocate_shared_buffer(device_, coupled_cell_capacity * sizeof(float),
                                                     "coupled growth rates");
      coupled_cell_types_ = allocate_shared_buffer(
          device_, coupled_cell_capacity * sizeof(std::int32_t), "coupled cell types");
      coupled_cell_capacity_ = coupled_cell_capacity;
    }
    const auto requested_species_levels = at_least_one(species_level_count);
    if (requested_species_levels > coupled_species_level_capacity_) {
      const auto coupled_species_level_capacity = std::bit_ceil(requested_species_levels);
      coupled_species_levels_ = allocate_shared_buffer(
          device_, coupled_species_level_capacity * sizeof(float), "coupled species levels");
      coupled_species_level_capacity_ = coupled_species_level_capacity;
    }
    const auto requested_instructions = at_least_one(instruction_count);
    if (requested_instructions > coupled_instruction_capacity_) {
      const auto coupled_instruction_capacity = std::bit_ceil(requested_instructions);
      coupled_instructions_ = allocate_shared_buffer(
          device_, coupled_instruction_capacity * sizeof(MetalRateInstruction),
          "coupled rate instructions");
      coupled_instruction_capacity_ = coupled_instruction_capacity;
    }
    const auto requested_species_outputs = at_least_one(species_count);
    if (requested_species_outputs > coupled_species_output_capacity_) {
      const auto coupled_species_output_capacity = std::bit_ceil(requested_species_outputs);
      coupled_species_outputs_ =
          allocate_shared_buffer(device_, coupled_species_output_capacity * sizeof(std::uint32_t),
                                 "coupled species outputs");
      coupled_species_output_capacity_ = coupled_species_output_capacity;
    }
    const auto requested_signal_outputs = at_least_one(signal_count);
    if (requested_signal_outputs > coupled_signal_output_capacity_) {
      const auto coupled_signal_output_capacity = std::bit_ceil(requested_signal_outputs);
      coupled_signal_outputs_ =
          allocate_shared_buffer(device_, coupled_signal_output_capacity * sizeof(std::uint32_t),
                                 "coupled signal outputs");
      coupled_signal_output_capacity_ = coupled_signal_output_capacity;
    }
    const auto requested_workspace = at_least_one(workspace_count);
    if (requested_workspace > coupled_workspace_capacity_) {
      const auto coupled_workspace_capacity = std::bit_ceil(requested_workspace);
      coupled_workspace_ = allocate_shared_buffer(
          device_, coupled_workspace_capacity * sizeof(float), "coupled workspace");
      coupled_workspace_capacity_ = coupled_workspace_capacity;
    }
    const auto requested_cell_signals = at_least_one(cell_signal_count);
    if (requested_cell_signals > coupled_cell_signal_capacity_) {
      const auto coupled_cell_signal_capacity = std::bit_ceil(requested_cell_signals);
      coupled_cell_signal_rates_ = allocate_shared_buffer(
          device_, coupled_cell_signal_capacity * sizeof(float), "coupled cell signal rates");
      coupled_cell_signal_capacity_ = coupled_cell_signal_capacity;
    }
    if (grid_level_count > coupled_grid_level_capacity_) {
      const auto coupled_grid_level_capacity = std::bit_ceil(grid_level_count);
      const auto byte_count = coupled_grid_level_capacity * sizeof(float);
      coupled_grid_levels_ = allocate_shared_buffer(device_, byte_count, "coupled grid levels");
      coupled_grid_output_ = allocate_shared_buffer(device_, byte_count, "coupled grid output");
      coupled_grid_level_capacity_ = coupled_grid_level_capacity;
    }
    if (coupled_error_ == nil) {
      coupled_error_ = allocate_shared_buffer(device_, sizeof(std::uint32_t), "coupled error flag");
    }
  }

  // Encodes an inclusive prefix sum of `counts` and returns the buffer that
  // holds it: one of the two scratch buffers, or `counts` itself when there is
  // a single element.
  id<MTLBuffer> encode_inclusive_scan(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> counts,
                                      id<MTLBuffer> scratch_a, id<MTLBuffer> scratch_b,
                                      std::uint32_t element_count) {
    id<MTLBuffer> scan_input = counts;
    id<MTLBuffer> scan_output = scratch_a;
    std::uint32_t offset = 1;
    while (offset < element_count) {
      [encoder setComputePipelineState:inclusive_scan_pipeline_];
      [encoder setBuffer:scan_input offset:0 atIndex:0];
      [encoder setBuffer:scan_output offset:0 atIndex:1];
      [encoder setBytes:&offset length:sizeof(offset) atIndex:2];
      [encoder setBytes:&element_count length:sizeof(element_count) atIndex:3];
      dispatch_1d(encoder, inclusive_scan_pipeline_, element_count);
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      scan_input = scan_output;
      scan_output = scan_output == scratch_a ? scratch_b : scratch_a;
      if (offset > element_count / 2) {
        break;
      }
      offset *= 2;
    }
    return scan_input;
  }

  // Encodes the per-site cell lists the coupled grid kernel gathers from and
  // returns the buffer holding each site's inclusive end offset into
  // `coupled_scatter_cells_`.
  id<MTLBuffer> encode_coupled_scatter_lists(id<MTLComputeCommandEncoder> encoder,
                                             const MetalGridShape& shape, const MetalFloat4& origin,
                                             const MetalFloat4& spacing, std::uint32_t cell_count) {
    const auto site_count = shape.sites;
    std::memset(coupled_scatter_counts_.contents, 0, site_count * sizeof(std::uint32_t));
    std::memset(coupled_scatter_cursors_.contents, 0, site_count * sizeof(std::uint32_t));

    [encoder setComputePipelineState:coupled_scatter_stencil_pipeline_];
    [encoder setBuffer:coupled_centers_ offset:0 atIndex:0];
    [encoder setBuffer:coupled_scatter_stencils_ offset:0 atIndex:1];
    [encoder setBytes:&shape length:sizeof(shape) atIndex:2];
    [encoder setBytes:&origin length:sizeof(origin) atIndex:3];
    [encoder setBytes:&spacing length:sizeof(spacing) atIndex:4];
    [encoder setBytes:&cell_count length:sizeof(cell_count) atIndex:5];
    dispatch_1d(encoder, coupled_scatter_stencil_pipeline_, cell_count);
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [encoder setComputePipelineState:coupled_scatter_count_pipeline_];
    [encoder setBuffer:coupled_scatter_stencils_ offset:0 atIndex:0];
    [encoder setBuffer:coupled_scatter_counts_ offset:0 atIndex:1];
    [encoder setBytes:&shape length:sizeof(shape) atIndex:2];
    [encoder setBytes:&cell_count length:sizeof(cell_count) atIndex:3];
    dispatch_1d(encoder, coupled_scatter_count_pipeline_, cell_count);
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    id<MTLBuffer> ends =
        encode_inclusive_scan(encoder, coupled_scatter_counts_, coupled_scatter_scan_a_,
                              coupled_scatter_scan_b_, site_count);

    [encoder setComputePipelineState:coupled_scatter_fill_pipeline_];
    [encoder setBuffer:coupled_scatter_stencils_ offset:0 atIndex:0];
    [encoder setBuffer:coupled_scatter_cursors_ offset:0 atIndex:1];
    [encoder setBuffer:ends offset:0 atIndex:2];
    [encoder setBuffer:coupled_scatter_cells_ offset:0 atIndex:3];
    [encoder setBytes:&shape length:sizeof(shape) atIndex:4];
    [encoder setBytes:&cell_count length:sizeof(cell_count) atIndex:5];
    dispatch_1d(encoder, coupled_scatter_fill_pipeline_, cell_count);
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [encoder setComputePipelineState:coupled_scatter_sort_pipeline_];
    [encoder setBuffer:ends offset:0 atIndex:0];
    [encoder setBuffer:coupled_scatter_cells_ offset:0 atIndex:1];
    [encoder setBytes:&site_count length:sizeof(site_count) atIndex:2];
    dispatch_1d(encoder, coupled_scatter_sort_pipeline_, site_count);
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    return ends;
  }

  void ensure_coupled_scatter_capacity(std::size_t cell_count, std::size_t site_count) {
    const auto requested_cells = std::max<std::size_t>(cell_count, 1);
    if (requested_cells > coupled_scatter_cell_capacity_) {
      const auto coupled_scatter_cell_capacity = std::bit_ceil(requested_cells);
      coupled_scatter_stencils_ = allocate_shared_buffer(
          device_, coupled_scatter_cell_capacity * sizeof(MetalUInt2), "coupled scatter stencils");
      coupled_scatter_cells_ = allocate_shared_buffer(
          device_, coupled_scatter_cell_capacity * scatter_stencil_sites * sizeof(std::uint32_t),
          "coupled scatter cell lists");
      coupled_scatter_cell_capacity_ = coupled_scatter_cell_capacity;
    }
    if (site_count > coupled_scatter_site_capacity_) {
      const auto coupled_scatter_site_capacity = std::bit_ceil(site_count);
      const auto byte_count = coupled_scatter_site_capacity * sizeof(std::uint32_t);
      coupled_scatter_counts_ =
          allocate_shared_buffer(device_, byte_count, "coupled scatter counts");
      coupled_scatter_cursors_ =
          allocate_shared_buffer(device_, byte_count, "coupled scatter cursors");
      coupled_scatter_scan_a_ =
          allocate_shared_buffer(device_, byte_count, "coupled scatter scan A");
      coupled_scatter_scan_b_ =
          allocate_shared_buffer(device_, byte_count, "coupled scatter scan B");
      coupled_scatter_site_capacity_ = coupled_scatter_site_capacity;
    }
  }

  // Projected-gradient steps encoded in one command buffer. The device stops
  // itself at convergence, so the steps left in a batch do nothing.
  static constexpr std::uint32_t contact_solve_batch = 32;
  // Terms each threadgroup of the step-at-a-time solve sums.
  static constexpr std::uint32_t contact_reduction_width = shared::contact_reduction_width;
  // Implicit signal sweeps encoded per command. A step takes about a dozen.
  static constexpr std::uint32_t signal_sweep_batch = 16;
  // Systems of up to this many rows are solved in one threadgroup. Above it,
  // the solve spreads over the whole device a batch of steps at a time.
  static constexpr std::uint32_t contact_solve_group_rows = 4096;

  void ensure_contact_dynamics_capacity(std::size_t body_count, std::size_t row_count) {
    if (body_count > contact_body_capacity_) {
      const auto capacity = std::bit_ceil(body_count);
      contact_bodies_ =
          allocate_shared_buffer(device_, capacity * sizeof(MetalContactBody), "contact bodies");
      contact_response_ = allocate_shared_buffer(device_, capacity * sizeof(MetalContactResponse),
                                                 "contact response");
      contact_incidence_offsets_ = allocate_shared_buffer(
          device_, (capacity + 1) * sizeof(std::uint32_t), "contact incidence offsets");
      contact_body_capacity_ = capacity;
    }
    if (row_count > contact_row_capacity_) {
      const auto capacity = std::bit_ceil(row_count);
      contact_rows_ =
          allocate_shared_buffer(device_, capacity * sizeof(MetalContactRow), "contact rows");
      contact_incidence_entries_ = allocate_shared_buffer(
          device_, capacity * 2 * sizeof(std::uint32_t), "contact incidence entries");
      const auto partials = (capacity + contact_reduction_width - 1) / contact_reduction_width;
      for (std::size_t index = 0; index < 2; ++index) {
        contact_multipliers_[index] =
            allocate_shared_buffer(device_, capacity * sizeof(float), "contact multipliers");
        contact_gaps_[index] =
            allocate_shared_buffer(device_, capacity * sizeof(float), "contact gaps");
        contact_reduce_[index] =
            allocate_shared_buffer(device_, partials * sizeof(MetalFloat4), "contact partial sums");
      }
      contact_row_capacity_ = capacity;
    }
    if (contact_state_ == nil) {
      contact_state_ =
          allocate_shared_buffer(device_, sizeof(MetalContactState), "contact solve state");
    }
  }

  void upload_contact_plan(const detail::ContactSolvePlan& plan) {
    std::memcpy(contact_rows_.contents, plan.rows.data(),
                plan.rows.size() * sizeof(MetalContactRow));
    std::memcpy(contact_bodies_.contents, plan.bodies.data(),
                plan.bodies.size() * sizeof(MetalContactBody));
    std::memcpy(contact_incidence_offsets_.contents, plan.incidence_offsets.data(),
                plan.incidence_offsets.size() * sizeof(std::uint32_t));
    std::memcpy(contact_incidence_entries_.contents, plan.incidence_entries.data(),
                plan.incidence_entries.size() * sizeof(std::uint32_t));
    std::memcpy(contact_multipliers_[0].contents, plan.multipliers.data(),
                plan.multipliers.size() * sizeof(float));
  }

  // The width of the single-group solve's threadgroup: the largest multiple of
  // the reduction width the pipeline allows, up to 1024, or zero when it
  // allows less than one.
  [[nodiscard]] std::uint32_t contact_solve_group_width() const {
    const auto allowed =
        std::min<NSUInteger>(contact_solve_pipeline_.maxTotalThreadsPerThreadgroup, 1024);
    return static_cast<std::uint32_t>(allowed / contact_reduction_width * contact_reduction_width);
  }

  // Runs the whole solve from the multipliers in pair zero in one threadgroup.
  void run_contact_solve(const MetalContactSolve& solve, std::uint32_t max_iterations,
                         float first_step_length, std::uint32_t width) {
    @autoreleasepool {
      id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      if (command_buffer == nil || encoder == nil) {
        throw std::runtime_error("failed to create a Metal contact solve command");
      }
      [encoder setComputePipelineState:contact_solve_pipeline_];
      [encoder setBuffer:contact_rows_ offset:0 atIndex:0];
      [encoder setBuffer:contact_bodies_ offset:0 atIndex:1];
      [encoder setBuffer:contact_multipliers_[0] offset:0 atIndex:2];
      [encoder setBuffer:contact_multipliers_[1] offset:0 atIndex:3];
      [encoder setBuffer:contact_gaps_[0] offset:0 atIndex:4];
      [encoder setBuffer:contact_gaps_[1] offset:0 atIndex:5];
      [encoder setBuffer:contact_incidence_offsets_ offset:0 atIndex:6];
      [encoder setBuffer:contact_incidence_entries_ offset:0 atIndex:7];
      [encoder setBuffer:contact_response_ offset:0 atIndex:8];
      [encoder setBuffer:contact_state_ offset:0 atIndex:9];
      [encoder setBytes:&solve length:sizeof(solve) atIndex:10];
      [encoder setBytes:&max_iterations length:sizeof(max_iterations) atIndex:11];
      [encoder setBytes:&first_step_length length:sizeof(first_step_length) atIndex:12];
      [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
      [encoder endEncoding];
      wait_for_command(command_buffer, "Metal contact solve failed");
    }
  }

  // The response to the multipliers in pair zero, without a solve.
  void respond_to_contacts(const MetalContactSolve& solve) {
    @autoreleasepool {
      id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      if (command_buffer == nil || encoder == nil) {
        throw std::runtime_error("failed to create a Metal contact response command");
      }
      encode_contact_respond(encoder, solve, 0, 0);
      [encoder endEncoding];
      wait_for_command(command_buffer, "Metal contact response failed");
    }
  }

  [[nodiscard]] ContactResponse download_contact_response(std::uint32_t body_count) const {
    const auto* packed = static_cast<const MetalContactResponse*>(contact_response_.contents);
    return detail::unpack_contact_response({packed, body_count});
  }

  void encode_contact_respond(id<MTLComputeCommandEncoder> encoder, const MetalContactSolve& solve,
                              std::size_t current, std::uint32_t project) {
    [encoder setComputePipelineState:contact_respond_pipeline_];
    [encoder setBuffer:contact_rows_ offset:0 atIndex:0];
    [encoder setBuffer:contact_bodies_ offset:0 atIndex:1];
    [encoder setBuffer:contact_multipliers_[current] offset:0 atIndex:2];
    [encoder setBuffer:contact_gaps_[current] offset:0 atIndex:3];
    [encoder setBuffer:contact_incidence_offsets_ offset:0 atIndex:4];
    [encoder setBuffer:contact_incidence_entries_ offset:0 atIndex:5];
    [encoder setBuffer:contact_response_ offset:0 atIndex:6];
    [encoder setBuffer:contact_state_ offset:0 atIndex:7];
    [encoder setBytes:&solve length:sizeof(solve) atIndex:8];
    [encoder setBytes:&project length:sizeof(project) atIndex:9];
    dispatch_1d(encoder, contact_respond_pipeline_, solve.body_count);
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  }

  // Encodes one evaluation of the rows: the initial gaps of pair zero, or the
  // projected step from pair `current`. Their terms are summed level by level
  // and recorded in the solve state.
  void encode_contact_terms(id<MTLComputeCommandEncoder> encoder, const MetalContactSolve& solve,
                            std::size_t current, std::uint32_t initial, float first_step_length,
                            std::uint32_t max_iterations) {
    const auto next = 1 - current;
    auto count = (solve.row_count + contact_reduction_width - 1) / contact_reduction_width;
    [encoder setComputePipelineState:contact_step_tree_pipeline_];
    [encoder setBuffer:contact_rows_ offset:0 atIndex:0];
    [encoder setBuffer:contact_response_ offset:0 atIndex:1];
    [encoder setBuffer:contact_multipliers_[current] offset:0 atIndex:2];
    [encoder setBuffer:contact_gaps_[current] offset:0 atIndex:3];
    [encoder setBuffer:contact_multipliers_[next] offset:0 atIndex:4];
    [encoder setBuffer:contact_gaps_[next] offset:0 atIndex:5];
    [encoder setBuffer:contact_reduce_[0] offset:0 atIndex:6];
    [encoder setBuffer:contact_state_ offset:0 atIndex:7];
    [encoder setBytes:&solve length:sizeof(solve) atIndex:8];
    [encoder setBytes:&initial length:sizeof(initial) atIndex:9];
    [encoder dispatchThreadgroups:MTLSizeMake(count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(contact_reduction_width, 1, 1)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    std::size_t input = 0;
    while (count > 1) {
      const auto output_count = (count + contact_reduction_width - 1) / contact_reduction_width;
      [encoder setComputePipelineState:contact_tree_pipeline_];
      [encoder setBuffer:contact_reduce_[input] offset:0 atIndex:0];
      [encoder setBuffer:contact_reduce_[1 - input] offset:0 atIndex:1];
      [encoder setBytes:&count length:sizeof(count) atIndex:2];
      [encoder setBuffer:contact_state_ offset:0 atIndex:3];
      [encoder setBytes:&initial length:sizeof(initial) atIndex:4];
      [encoder dispatchThreadgroups:MTLSizeMake(output_count, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(contact_reduction_width, 1, 1)];
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      input = 1 - input;
      count = output_count;
    }
    [encoder setComputePipelineState:contact_record_pipeline_];
    [encoder setBuffer:contact_reduce_[input] offset:0 atIndex:0];
    [encoder setBuffer:contact_state_ offset:0 atIndex:1];
    [encoder setBytes:&solve length:sizeof(solve) atIndex:2];
    [encoder setBytes:&initial length:sizeof(initial) atIndex:3];
    [encoder setBytes:&first_step_length length:sizeof(first_step_length) atIndex:4];
    [encoder setBytes:&max_iterations length:sizeof(max_iterations) atIndex:5];
    dispatch_1d(encoder, contact_record_pipeline_, 1);
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  }

  // Runs the solve a step at a time over the whole device, in batches of
  // steps that each cost one command. The device stops itself when the solve
  // is done, so the steps left in a batch do nothing.
  void run_contact_steps(const MetalContactSolve& solve, std::uint32_t max_iterations,
                         float first_step_length) {
    std::size_t current = 0;
    auto initial = true;
    for (;;) {
      @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
          throw std::runtime_error("failed to create a Metal contact iteration command");
        }
        if (initial) {
          encode_contact_respond(encoder, solve, 0, 0);
          encode_contact_terms(encoder, solve, 0, 1, first_step_length, max_iterations);
          initial = false;
        }
        for (std::uint32_t index = 0; index < contact_solve_batch; ++index) {
          encode_contact_respond(encoder, solve, current, 1);
          encode_contact_terms(encoder, solve, current, 0, 0.0F, max_iterations);
          current = 1 - current;
        }
        [encoder endEncoding];
        wait_for_command(command_buffer, "Metal contact solve iteration failed");
      }
      if (static_cast<const MetalContactState*>(contact_state_.contents)->done != 0) {
        return;
      }
    }
  }

  std::uint32_t device_index_{0};
  id<MTLDevice> device_{nil};
  id<MTLCommandQueue> queue_{nil};
  id<MTLComputePipelineState> growth_pipeline_{nil};
  id<MTLComputePipelineState> species_pipeline_{nil};
  id<MTLComputePipelineState> signals_pipeline_{nil};
  id<MTLComputePipelineState> signals_cn_jacobi_pipeline_{nil};
  id<MTLComputePipelineState> signals_cn_residual_pipeline_{nil};
  id<MTLComputePipelineState> signals_square_pipeline_{nil};
  id<MTLComputePipelineState> signals_reduce_pipeline_{nil};
  id<MTLComputePipelineState> signals_sweep_check_pipeline_{nil};
  // A gate that is always open, and the progress of a batch of sweeps.
  id<MTLBuffer> signal_gate_open_{nil};
  id<MTLBuffer> signal_sweep_state_{nil};
  id<MTLComputePipelineState> coupled_cells_pipeline_{nil};
  id<MTLComputePipelineState> coupled_grid_pipeline_{nil};
  id<MTLComputePipelineState> coupled_scatter_stencil_pipeline_{nil};
  id<MTLComputePipelineState> coupled_scatter_count_pipeline_{nil};
  id<MTLComputePipelineState> coupled_scatter_fill_pipeline_{nil};
  id<MTLComputePipelineState> coupled_scatter_sort_pipeline_{nil};
  id<MTLComputePipelineState> inclusive_scan_pipeline_{nil};
  id<MTLComputePipelineState> contact_respond_pipeline_{nil};
  id<MTLComputePipelineState> contact_step_tree_pipeline_{nil};
  id<MTLComputePipelineState> contact_tree_pipeline_{nil};
  id<MTLComputePipelineState> contact_record_pipeline_{nil};
  id<MTLComputePipelineState> contact_solve_pipeline_{nil};
  std::unique_ptr<metal::FlowSolver> flow_solver_;
  std::unique_ptr<metal::CellKernels> cell_kernels_;
  std::unique_ptr<detail::ContactStepDevice> step_device_;

  id<MTLBuffer> lengths_{nil};
  id<MTLBuffer> growth_rates_{nil};
  std::size_t growth_capacity_{0};

  id<MTLBuffer> species_levels_{nil};
  id<MTLBuffer> species_previous_lengths_{nil};
  id<MTLBuffer> species_centers_{nil};
  id<MTLBuffer> species_geometry_{nil};
  id<MTLBuffer> species_growth_rates_{nil};
  id<MTLBuffer> species_cell_types_{nil};
  id<MTLBuffer> species_instructions_{nil};
  id<MTLBuffer> species_outputs_{nil};
  id<MTLBuffer> species_workspace_{nil};
  id<MTLBuffer> species_error_{nil};
  std::size_t species_cell_capacity_{0};
  std::size_t species_level_capacity_{0};
  std::size_t species_instruction_capacity_{0};
  std::size_t species_output_capacity_{0};
  std::size_t species_workspace_capacity_{0};

  id<MTLBuffer> signal_levels_{nil};
  id<MTLBuffer> signal_output_{nil};
  id<MTLBuffer> signal_diffusion_{nil};
  id<MTLBuffer> signal_advection_{nil};
  id<MTLBuffer> signal_fixed_values_{nil};
  id<MTLBuffer> signal_reaction_source_{nil};
  id<MTLBuffer> signal_reaction_loss_{nil};
  id<MTLBuffer> signal_obstacles_{nil};
  id<MTLBuffer> signal_x_faces_{nil};
  id<MTLBuffer> signal_y_faces_{nil};
  id<MTLBuffer> signal_z_faces_{nil};
  std::size_t signal_face_capacity_{0};
  id<MTLBuffer> signal_fluid_sites_{nil};
  id<MTLBuffer> signal_error_{nil};
  std::size_t signal_level_capacity_{0};
  std::size_t grid_static_capacity_{0};
  std::size_t signal_count_capacity_{0};
  // The grid specification the transport buffers hold, zero for none, and
  // what the kernels are told of it.
  std::uint64_t grid_revision_{0};
  GridStatics grid_statics_;
  // The site count of the obstacle mask on the device, zero for none.
  std::size_t grid_site_count_{0};
  id<MTLBuffer> signal_cn_a_{nil};
  id<MTLBuffer> signal_cn_b_{nil};
  id<MTLBuffer> signal_cn_terms_{nil};
  id<MTLBuffer> signal_cn_reduce_a_{nil};
  id<MTLBuffer> signal_cn_reduce_b_{nil};
  std::size_t signal_solve_capacity_{0};

  id<MTLBuffer> coupled_species_levels_{nil};
  id<MTLBuffer> coupled_previous_lengths_{nil};
  id<MTLBuffer> coupled_centers_{nil};
  id<MTLBuffer> coupled_geometry_{nil};
  id<MTLBuffer> coupled_growth_rates_{nil};
  id<MTLBuffer> coupled_cell_types_{nil};
  id<MTLBuffer> coupled_instructions_{nil};
  id<MTLBuffer> coupled_species_outputs_{nil};
  id<MTLBuffer> coupled_signal_outputs_{nil};
  id<MTLBuffer> coupled_workspace_{nil};
  id<MTLBuffer> coupled_cell_signal_rates_{nil};
  id<MTLBuffer> coupled_grid_levels_{nil};
  id<MTLBuffer> coupled_grid_output_{nil};
  id<MTLBuffer> coupled_error_{nil};
  std::size_t coupled_cell_capacity_{0};
  std::size_t coupled_species_level_capacity_{0};
  std::size_t coupled_instruction_capacity_{0};
  std::size_t coupled_species_output_capacity_{0};
  std::size_t coupled_signal_output_capacity_{0};
  std::size_t coupled_workspace_capacity_{0};
  std::size_t coupled_cell_signal_capacity_{0};
  std::size_t coupled_grid_level_capacity_{0};
  // A trilinear stencil covers at most eight lattice sites.
  static constexpr std::size_t scatter_stencil_sites = 8;
  id<MTLBuffer> coupled_scatter_stencils_{nil};
  id<MTLBuffer> coupled_scatter_cells_{nil};
  id<MTLBuffer> coupled_scatter_counts_{nil};
  id<MTLBuffer> coupled_scatter_cursors_{nil};
  id<MTLBuffer> coupled_scatter_scan_a_{nil};
  id<MTLBuffer> coupled_scatter_scan_b_{nil};
  std::size_t coupled_scatter_cell_capacity_{0};
  std::size_t coupled_scatter_site_capacity_{0};

  id<MTLBuffer> contact_rows_{nil};
  id<MTLBuffer> contact_bodies_{nil};
  id<MTLBuffer> contact_response_{nil};
  id<MTLBuffer> contact_incidence_offsets_{nil};
  id<MTLBuffer> contact_incidence_entries_{nil};
  std::array<id<MTLBuffer>, 2> contact_multipliers_{};
  std::array<id<MTLBuffer>, 2> contact_gaps_{};
  std::array<id<MTLBuffer>, 2> contact_reduce_{};
  id<MTLBuffer> contact_state_{nil};
  std::size_t contact_body_capacity_{0};
  std::size_t contact_row_capacity_{0};
};

// A backend that leaves out any provider method is abstract.
static_assert(!std::is_abstract_v<MetalBackend>, "the Metal backend must implement every provider");

}  // namespace

std::unique_ptr<ComputeBackend> make_metal_backend(std::uint32_t device_index) {
  return std::make_unique<MetalBackend>(device_index);
}

std::size_t metal_backend_device_count() noexcept {
  @autoreleasepool {
    const auto count = MTLCopyAllDevices().count;
    return count == 0 && MTLCreateSystemDefaultDevice() != nil ? 1 : count;
  }
}

}  // namespace microsimulator
