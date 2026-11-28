#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_cell_kernels.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/deposition_plan.hpp"
#include "core/device_types.hpp"
#include "core/external_contact_plan.hpp"
#include "core/flow_drift_plan.hpp"
#include "microsimulator/metal/cell_kernels_source.hpp"

namespace microsimulator::metal {
namespace {

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
    throw std::runtime_error("Metal is unavailable on this system");
  }
  if (static_cast<NSUInteger>(device_index) >= devices.count) {
    throw std::out_of_range("Metal device index is unavailable");
  }
  return devices[device_index];
}

// Safe math keeps isfinite meaningful, which the drift status relies on.
MTLCompileOptions* safe_math_options() {
  MTLCompileOptions* options = [MTLCompileOptions new];
  if (@available(macOS 15.0, *)) {
    options.mathMode = MTLMathModeSafe;
    options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
  } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    options.fastMathEnabled = NO;
#pragma clang diagnostic pop
  }
  return options;
}

id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device, id<MTLLibrary> library,
                                          NSString* name) {
  id<MTLFunction> function = [library newFunctionWithName:name];
  if (function == nil) {
    throw std::runtime_error(std::string("Metal cell kernel function is missing: ") +
                             name.UTF8String);
  }
  NSError* error = nil;
  id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                               error:&error];
  if (pipeline == nil) {
    throw_metal_error("failed to create a Metal cell kernel pipeline", error);
  }
  return pipeline;
}

void dispatch(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline,
              std::uint32_t count) {
  if (count == 0) {
    return;
  }
  const auto width = std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256);
  [encoder setComputePipelineState:pipeline];
  [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

// Encodes the inclusive prefix sum of the `count` values at `offset` in
// `values`, in place. Each threadgroup scans a block and records its total at
// `totals_offset` in `totals`; the totals are scanned the same way, and each
// block then adds the totals before it.
void encode_scan(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> blocks,
                 id<MTLComputePipelineState> add, id<MTLBuffer> values, NSUInteger offset,
                 std::uint32_t count, id<MTLBuffer> totals, NSUInteger totals_offset) {
  if (count == 0) {
    return;
  }
  const auto width = std::min<NSUInteger>(blocks.maxTotalThreadsPerThreadgroup, 1024);
  const auto groups = static_cast<std::uint32_t>((count + width - 1) / width);
  [encoder setComputePipelineState:blocks];
  [encoder setBuffer:values offset:offset atIndex:0];
  [encoder setBuffer:totals offset:totals_offset atIndex:1];
  [encoder setBytes:&count length:sizeof(count) atIndex:2];
  [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  if (groups == 1) {
    return;
  }
  const auto next = totals_offset + groups * sizeof(std::uint32_t);
  encode_scan(encoder, blocks, add, totals, totals_offset, groups, totals, next);
  [encoder setComputePipelineState:add];
  [encoder setBuffer:values offset:offset atIndex:0];
  [encoder setBuffer:totals offset:totals_offset atIndex:1];
  [encoder setBytes:&count length:sizeof(count) atIndex:2];
  [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

// The bytes of block totals that scanning `count` values needs at every
// level, for blocks of at least 32 values.
[[nodiscard]] std::size_t scan_totals_bytes(std::uint32_t count) {
  std::size_t total = 1;
  while (count > 1) {
    count = (count + 31) / 32;
    total += count;
  }
  return total * sizeof(std::uint32_t);
}

// A shared-storage buffer that grows to the largest size asked of it.
class Buffer {
 public:
  void reserve(id<MTLDevice> device, std::size_t bytes, const char* description) {
    bytes = std::max<std::size_t>(bytes, sizeof(float));
    if (buffer_ != nil && buffer_.length >= bytes) {
      return;
    }
    buffer_ = [device newBufferWithLength:std::bit_ceil(bytes)
                                  options:MTLResourceStorageModeShared];
    if (buffer_ == nil) {
      throw std::runtime_error(std::string("failed to allocate Metal ") + description);
    }
  }

  template <typename T>
  void write(id<MTLDevice> device, const std::vector<T>& values, const char* description) {
    reserve(device, values.size() * sizeof(T), description);
    std::memcpy(buffer_.contents, values.data(), values.size() * sizeof(T));
  }

  void zero(id<MTLDevice> device, std::size_t bytes, const char* description) {
    reserve(device, bytes, description);
    std::memset(buffer_.contents, 0, bytes);
  }

  template <typename T>
  [[nodiscard]] std::vector<T> read(std::size_t count) const {
    const auto* values = static_cast<const T*>(buffer_.contents);
    return std::vector<T>(values, values + count);
  }

  [[nodiscard]] id<MTLBuffer> get() const noexcept {
    return buffer_;
  }

 private:
  id<MTLBuffer> buffer_ = nil;
};

}  // namespace

struct CellKernels::Impl {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> weights;
  id<MTLComputePipelineState> count;
  id<MTLComputePipelineState> scan_blocks;
  id<MTLComputePipelineState> scan_add;
  id<MTLComputePipelineState> fill;
  id<MTLComputePipelineState> sites;
  id<MTLComputePipelineState> coupling;
  id<MTLComputePipelineState> drag;
  id<MTLComputePipelineState> external_count;
  id<MTLComputePipelineState> external_fill;
  id<MTLComputePipelineState> drift_rates;
  id<MTLComputePipelineState> drift_advance;

  Buffer centers, radii, channels, solid, boxes, cell_weights, status, counts, scan_totals, cursors,
      entries, density, channel_density;
  Buffer base, coupling_solid, deposit_density, velocity, mobility, force, drag_values;
  Buffer contact_centers, contact_axes, contact_constraints, pair_counts, pair_scan_totals,
      contact_records;
  Buffer drift_centers, drift_axes, drift_fixed, drift_solid, drift_x_faces, drift_y_faces,
      drift_z_faces, drift_output, drift_moved_centers, drift_moved_axes, drift_status;
  // The revision of the grid whose obstacles and velocity field the drift
  // buffers hold, and whether they hold any.
  std::uint64_t drift_revision{0};
  bool drift_loaded{false};

  // Uploads a drift's cells, and its grid's obstacles and velocity field when
  // they are not the ones the buffers hold.
  void load_drift(const detail::FlowDriftPlan& plan) {
    drift_centers.write(device, plan.centers, "drift centers");
    drift_axes.write(device, plan.axes, "drift axes");
    drift_fixed.write(device, plan.fixed, "drift fixed flags");
    drift_status.zero(device, plan.centers.size() * sizeof(std::uint32_t), "drift status");
    if (drift_loaded && drift_revision == plan.revision) {
      return;
    }
    drift_solid.write(device, plan.solid, "drift obstacles");
    drift_x_faces.write(device, plan.field->x_faces, "drift x faces");
    drift_y_faces.write(device, plan.field->y_faces, "drift y faces");
    drift_z_faces.write(device, plan.field->z_faces, "drift z faces");
    drift_revision = plan.revision;
    drift_loaded = true;
  }

  void bind_drift_inputs(id<MTLComputeCommandEncoder> encoder) {
    [encoder setBuffer:drift_centers.get() offset:0 atIndex:0];
    [encoder setBuffer:drift_axes.get() offset:0 atIndex:1];
    [encoder setBuffer:drift_fixed.get() offset:0 atIndex:2];
    [encoder setBuffer:drift_solid.get() offset:0 atIndex:3];
    [encoder setBuffer:drift_x_faces.get() offset:0 atIndex:4];
    [encoder setBuffer:drift_y_faces.get() offset:0 atIndex:5];
    [encoder setBuffer:drift_z_faces.get() offset:0 atIndex:6];
  }

  // Encodes the inclusive prefix sum of the `count` values in `values`, in
  // place, with `totals` holding the block totals, and returns `values`.
  id<MTLBuffer> encode_scan(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> values,
                            id<MTLBuffer> totals, std::uint32_t count) const {
    microsimulator::metal::encode_scan(encoder, scan_blocks, scan_add, values, 0, count, totals, 0);
    return values;
  }
};

CellKernels::CellKernels(std::uint32_t device_index) : impl_(std::make_unique<Impl>()) {
  @autoreleasepool {
    auto& impl = *impl_;
    impl.device = select_device(device_index);
    impl.queue = [impl.device newCommandQueue];
    if (impl.queue == nil) {
      throw std::runtime_error("failed to create a Metal deposition command queue");
    }
    NSString* source = [NSString stringWithUTF8String:cell_kernels_source];
    NSError* error = nil;
    id<MTLLibrary> library = [impl.device newLibraryWithSource:source
                                                       options:safe_math_options()
                                                         error:&error];
    if (library == nil) {
      throw_metal_error("failed to compile the Metal cell kernels", error);
    }
    impl.weights = make_pipeline(impl.device, library, @"deposit_weights");
    impl.count = make_pipeline(impl.device, library, @"deposit_count");
    impl.scan_blocks = make_pipeline(impl.device, library, @"scan_blocks");
    impl.scan_add = make_pipeline(impl.device, library, @"scan_add");
    impl.fill = make_pipeline(impl.device, library, @"deposit_fill");
    impl.sites = make_pipeline(impl.device, library, @"deposit_sites");
    impl.coupling = make_pipeline(impl.device, library, @"shallow_coupling");
    impl.drag = make_pipeline(impl.device, library, @"brinkman_drag");
    impl.external_count = make_pipeline(impl.device, library, @"external_count");
    impl.external_fill = make_pipeline(impl.device, library, @"external_fill");
    impl.drift_rates = make_pipeline(impl.device, library, @"drift_rates");
    impl.drift_advance = make_pipeline(impl.device, library, @"drift_advance");
  }
}

CellKernels::~CellKernels() = default;

BiomassDeposit CellKernels::deposit(const SignalGridSpec& spec, const DepositCells& cells,
                                    const BiomassDepositParameters& parameters) {
  const auto plan = detail::plan_deposit(spec, cells, parameters);
  const auto cell_count = plan.cell_count();
  const auto site_count = plan.site_count();
  const auto channel_values = static_cast<std::size_t>(plan.grid.channel_count) * site_count;
  auto& impl = *impl_;
  @autoreleasepool {
    impl.centers.write(impl.device, plan.centers, "deposit centers");
    impl.radii.write(impl.device, plan.radii, "deposit radii");
    impl.channels.write(impl.device, plan.channels, "deposit channels");
    impl.solid.write(impl.device, plan.solid, "deposit obstacles");
    impl.boxes.reserve(impl.device, plan.centers.size() * sizeof(device::DepositBox),
                       "deposit boxes");
    impl.cell_weights.reserve(impl.device, plan.weight_count() * sizeof(float), "deposit weights");
    impl.status.zero(impl.device, cell_count * sizeof(std::uint32_t), "deposit status");
    const auto site_bytes = static_cast<std::size_t>(site_count) * sizeof(std::uint32_t);
    impl.counts.zero(impl.device, site_bytes, "deposit counts");
    impl.cursors.zero(impl.device, site_bytes, "deposit cursors");
    impl.scan_totals.reserve(impl.device, scan_totals_bytes(site_count), "deposit scan");
    // Each cell enters at most every site of its box.
    impl.entries.reserve(impl.device, plan.weight_count() * sizeof(std::uint32_t),
                         "deposit entries");
    impl.density.reserve(impl.device, site_count * sizeof(float), "deposit density");
    impl.channel_density.reserve(impl.device, channel_values * sizeof(float),
                                 "deposit channel density");

    id<MTLCommandBuffer> command = [impl.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (command == nil || encoder == nil) {
      throw std::runtime_error("failed to create a Metal deposition command");
    }
    const auto& grid = plan.grid;

    [encoder setBuffer:impl.centers.get() offset:0 atIndex:0];
    [encoder setBuffer:impl.solid.get() offset:0 atIndex:1];
    [encoder setBuffer:impl.boxes.get() offset:0 atIndex:2];
    [encoder setBuffer:impl.cell_weights.get() offset:0 atIndex:3];
    [encoder setBuffer:impl.status.get() offset:0 atIndex:4];
    [encoder setBytes:&grid length:sizeof(grid) atIndex:5];
    dispatch(encoder, impl.weights, cell_count);

    [encoder setBuffer:impl.boxes.get() offset:0 atIndex:0];
    [encoder setBuffer:impl.cell_weights.get() offset:0 atIndex:1];
    [encoder setBuffer:impl.counts.get() offset:0 atIndex:2];
    [encoder setBytes:&grid length:sizeof(grid) atIndex:3];
    dispatch(encoder, impl.count, cell_count);

    id<MTLBuffer> ends =
        impl.encode_scan(encoder, impl.counts.get(), impl.scan_totals.get(), site_count);

    [encoder setBuffer:impl.boxes.get() offset:0 atIndex:0];
    [encoder setBuffer:impl.cell_weights.get() offset:0 atIndex:1];
    [encoder setBuffer:ends offset:0 atIndex:2];
    [encoder setBuffer:impl.cursors.get() offset:0 atIndex:3];
    [encoder setBuffer:impl.entries.get() offset:0 atIndex:4];
    [encoder setBytes:&grid length:sizeof(grid) atIndex:5];
    dispatch(encoder, impl.fill, cell_count);

    [encoder setBuffer:ends offset:0 atIndex:0];
    [encoder setBuffer:impl.entries.get() offset:0 atIndex:1];
    [encoder setBuffer:impl.boxes.get() offset:0 atIndex:2];
    [encoder setBuffer:impl.cell_weights.get() offset:0 atIndex:3];
    [encoder setBuffer:impl.centers.get() offset:0 atIndex:4];
    [encoder setBuffer:impl.radii.get() offset:0 atIndex:5];
    [encoder setBuffer:impl.channels.get() offset:0 atIndex:6];
    [encoder setBuffer:impl.density.get() offset:0 atIndex:7];
    [encoder setBuffer:impl.channel_density.get() offset:0 atIndex:8];
    [encoder setBytes:&grid length:sizeof(grid) atIndex:9];
    dispatch(encoder, impl.sites, site_count);

    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError) {
      throw_metal_error("Metal deposition command failed", command.error);
    }
  }
  detail::check_deposit_status(impl.status.read<std::uint32_t>(cell_count));
  BiomassDeposit result;
  result.density = impl.density.read<float>(site_count);
  result.channel_count = plan.grid.channel_count;
  result.channels = impl.channel_density.read<float>(channel_values);
  return result;
}

ShallowFlowCoupling CellKernels::shallow_flow_coupling(const SignalGridSpec& spec,
                                                       std::span<const float> base,
                                                       const BiomassDeposit& deposit,
                                                       const BiomassDragParameters& parameters) {
  const auto plan = detail::plan_shallow_flow_coupling(spec, base, deposit, parameters);
  const auto columns = plan.column_count();
  const auto site_count = plan.data.shape.w;
  auto& impl = *impl_;
  @autoreleasepool {
    impl.coupling_solid.write(impl.device, plan.solid, "coupling obstacles");
    impl.base.write(impl.device, plan.base, "coupling base mobility");
    impl.deposit_density.write(impl.device, deposit.density, "coupling density");
    impl.velocity.write(impl.device, plan.velocity, "coupling velocity");
    impl.mobility.reserve(impl.device, site_count * sizeof(float), "coupling mobility");
    impl.force.reserve(impl.device, 2 * static_cast<std::size_t>(columns) * sizeof(float),
                       "coupling force");

    id<MTLCommandBuffer> command = [impl.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (command == nil || encoder == nil) {
      throw std::runtime_error("failed to create a Metal coupling command");
    }
    [encoder setBuffer:impl.coupling_solid.get() offset:0 atIndex:0];
    [encoder setBuffer:impl.base.get() offset:0 atIndex:1];
    [encoder setBuffer:impl.deposit_density.get() offset:0 atIndex:2];
    [encoder setBuffer:impl.velocity.get() offset:0 atIndex:3];
    [encoder setBuffer:impl.mobility.get() offset:0 atIndex:4];
    [encoder setBuffer:impl.force.get() offset:0 atIndex:5];
    [encoder setBytes:&plan.data length:sizeof(plan.data) atIndex:6];
    dispatch(encoder, impl.coupling, columns);
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError) {
      throw_metal_error("Metal coupling command failed", command.error);
    }
  }
  const auto force = impl.force.read<float>(2 * static_cast<std::size_t>(columns));
  return detail::finish_shallow_flow_coupling(impl.mobility.read<float>(site_count), force);
}

std::vector<float> CellKernels::brinkman_drag(const SignalGridSpec& spec,
                                              const BiomassDeposit& deposit,
                                              const BiomassDragParameters& parameters) {
  const auto plan = detail::plan_brinkman_drag(spec, deposit, parameters);
  const auto site_count = plan.data.shape.w;
  auto& impl = *impl_;
  @autoreleasepool {
    impl.coupling_solid.write(impl.device, plan.solid, "drag obstacles");
    impl.deposit_density.write(impl.device, deposit.density, "drag density");
    impl.drag_values.reserve(impl.device, site_count * sizeof(float), "drag");
    id<MTLCommandBuffer> command = [impl.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (command == nil || encoder == nil) {
      throw std::runtime_error("failed to create a Metal drag command");
    }
    [encoder setBuffer:impl.coupling_solid.get() offset:0 atIndex:0];
    [encoder setBuffer:impl.deposit_density.get() offset:0 atIndex:1];
    [encoder setBuffer:impl.drag_values.get() offset:0 atIndex:2];
    [encoder setBytes:&plan.data length:sizeof(plan.data) atIndex:3];
    dispatch(encoder, impl.drag, site_count);
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError) {
      throw_metal_error("Metal drag command failed", command.error);
    }
  }
  return impl.drag_values.read<float>(site_count);
}

ExternalContactGraph CellKernels::external_contacts(const WorldState& state,
                                                    const ConstraintSet& constraints,
                                                    const ConstraintContactParameters& parameters,
                                                    bool every_face) {
  const auto plan = detail::plan_external_contacts(state, constraints, parameters, every_face);
  const auto pairs = plan.pair_count();
  if (pairs == 0) {
    return ExternalContactGraph(plan.data.cell_count, {});
  }
  const auto pair_count = static_cast<std::uint32_t>(pairs);
  // A pair makes at most shared::external_pair_capacity contacts.
  const auto capacity = pairs * 12;
  auto& impl = *impl_;
  std::uint32_t total = 0;
  @autoreleasepool {
    impl.contact_centers.write(impl.device, plan.centers, "external contact centers");
    impl.contact_axes.write(impl.device, plan.axes, "external contact axes");
    impl.contact_constraints.write(impl.device, plan.constraints, "external constraints");
    const auto pair_bytes = pairs * sizeof(std::uint32_t);
    impl.pair_counts.reserve(impl.device, pair_bytes, "external contact counts");
    impl.pair_scan_totals.reserve(impl.device, scan_totals_bytes(pair_count),
                                  "external contact scan");
    impl.contact_records.reserve(impl.device, capacity * sizeof(device::ExternalContactRecord),
                                 "external contact records");

    id<MTLCommandBuffer> command = [impl.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (command == nil || encoder == nil) {
      throw std::runtime_error("failed to create a Metal external contact command");
    }
    [encoder setBuffer:impl.contact_centers.get() offset:0 atIndex:0];
    [encoder setBuffer:impl.contact_axes.get() offset:0 atIndex:1];
    [encoder setBuffer:impl.contact_constraints.get() offset:0 atIndex:2];
    [encoder setBuffer:impl.pair_counts.get() offset:0 atIndex:3];
    [encoder setBytes:&plan.data length:sizeof(plan.data) atIndex:4];
    dispatch(encoder, impl.external_count, pair_count);

    id<MTLBuffer> ends =
        impl.encode_scan(encoder, impl.pair_counts.get(), impl.pair_scan_totals.get(), pair_count);

    [encoder setBuffer:impl.contact_centers.get() offset:0 atIndex:0];
    [encoder setBuffer:impl.contact_axes.get() offset:0 atIndex:1];
    [encoder setBuffer:impl.contact_constraints.get() offset:0 atIndex:2];
    [encoder setBuffer:ends offset:0 atIndex:3];
    [encoder setBuffer:impl.contact_records.get() offset:0 atIndex:4];
    [encoder setBytes:&plan.data length:sizeof(plan.data) atIndex:5];
    dispatch(encoder, impl.external_fill, pair_count);
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError) {
      throw_metal_error("Metal external contact command failed", command.error);
    }
    total = static_cast<const std::uint32_t*>(ends.contents)[pair_count - 1];
  }
  const auto* records =
      static_cast<const device::ExternalContactRecord*>(impl.contact_records.get().contents);
  return detail::finish_external_contacts(plan, {records, total});
}

std::vector<std::array<float, 6>> CellKernels::flow_drift_rates(const WorldState& state,
                                                                const SignalGrid& grid, float dt) {
  const auto plan = detail::plan_flow_drift(state, grid, dt, 0.0F);
  const auto cells = plan.data.cell_count;
  auto& impl = *impl_;
  @autoreleasepool {
    impl.load_drift(plan);
    impl.drift_output.reserve(impl.device, 6 * plan.centers.size() * sizeof(float), "drift rates");
    id<MTLCommandBuffer> command = [impl.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (command == nil || encoder == nil) {
      throw std::runtime_error("failed to create a Metal drift command");
    }
    impl.bind_drift_inputs(encoder);
    [encoder setBuffer:impl.drift_output.get() offset:0 atIndex:7];
    [encoder setBuffer:impl.drift_status.get() offset:0 atIndex:8];
    [encoder setBytes:&plan.data length:sizeof(plan.data) atIndex:9];
    dispatch(encoder, impl.drift_rates, cells);
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError) {
      throw_metal_error("Metal drift command failed", command.error);
    }
  }
  detail::check_flow_drift_status(impl.drift_status.read<std::uint32_t>(cells));
  const auto rates = impl.drift_output.read<float>(6 * static_cast<std::size_t>(cells));
  return detail::unpack_flow_drift_rates(rates, cells);
}

void CellKernels::advance_flow_drift(WorldState& state, const SignalGrid& grid, float dt,
                                     const FlowDriftParameters& parameters) {
  const auto plan = detail::plan_flow_drift(state, grid, dt, parameters.max_rotation_radians);
  const auto cells = plan.data.cell_count;
  auto& impl = *impl_;
  @autoreleasepool {
    impl.load_drift(plan);
    const auto bytes = plan.centers.size() * sizeof(device::float4);
    impl.drift_moved_centers.reserve(impl.device, bytes, "drift moved centers");
    impl.drift_moved_axes.reserve(impl.device, bytes, "drift moved axes");
    id<MTLCommandBuffer> command = [impl.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (command == nil || encoder == nil) {
      throw std::runtime_error("failed to create a Metal drift command");
    }
    impl.bind_drift_inputs(encoder);
    [encoder setBuffer:impl.drift_moved_centers.get() offset:0 atIndex:7];
    [encoder setBuffer:impl.drift_moved_axes.get() offset:0 atIndex:8];
    [encoder setBuffer:impl.drift_status.get() offset:0 atIndex:9];
    [encoder setBytes:&plan.data length:sizeof(plan.data) atIndex:10];
    dispatch(encoder, impl.drift_advance, cells);
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError) {
      throw_metal_error("Metal drift command failed", command.error);
    }
  }
  detail::check_flow_drift_status(impl.drift_status.read<std::uint32_t>(cells));
  const auto centers = impl.drift_moved_centers.read<device::float4>(cells);
  const auto axes = impl.drift_moved_axes.read<device::float4>(cells);
  detail::apply_flow_drift_moves(state, centers, axes);
}

}  // namespace microsimulator::metal
