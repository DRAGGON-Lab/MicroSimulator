#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_contact_step.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "core/contact_step_device.hpp"
#include "core/shared_constants.hpp"
#include "metal_compile_options.hpp"
#include "microsimulator/metal/contact_step_source.hpp"

namespace microsimulator::metal {
namespace {

using detail::StepBuffer;
using detail::StepKernel;

// A threadgroup of the solve that runs whole in one group, and the most rows
// the host lets it take: above that, the solve runs a step at a time over the
// device.
constexpr NSUInteger group_threads = 1024;
constexpr std::uint32_t group_rows = 4096;
// Steps of the step-at-a-time solve enqueued before the host reads whether it
// is done.
constexpr std::uint32_t solve_batch = 32;

[[noreturn]] void throw_metal_error(const char* operation, NSError* error) {
  const char* detail = error == nil ? "unknown Metal error" : error.localizedDescription.UTF8String;
  throw std::runtime_error(std::string(operation) + ": " + detail);
}

id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device, id<MTLLibrary> library,
                                          NSString* name) {
  id<MTLFunction> function = [library newFunctionWithName:name];
  if (function == nil) {
    throw std::runtime_error(std::string("Metal contact step function is missing: ") +
                             name.UTF8String);
  }
  NSError* error = nil;
  id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                               error:&error];
  if (pipeline == nil) {
    throw_metal_error("failed to create a Metal contact step pipeline", error);
  }
  return pipeline;
}

class MetalContactStepDevice final : public detail::ContactStepDevice {
 public:
  MetalContactStepDevice(id<MTLDevice> device, id<MTLCommandQueue> queue)
      : device_(device), queue_(queue) {
    @autoreleasepool {
      NSString* source = [NSString stringWithUTF8String:contact_step_source];
      NSError* error = nil;
      id<MTLLibrary> library = [device_ newLibraryWithSource:source
                                                     options:safe_math_options()
                                                       error:&error];
      if (library == nil) {
        throw_metal_error("failed to compile the Metal contact step", error);
      }
      std::size_t index = 0;
#define MS_METAL_STEP_PIPELINE(name, extent) \
  pipelines_[index++] = make_pipeline(device_, library, @"step_" #name "_kernel");
      MS_CONTACT_STEP_KERNELS(MS_METAL_STEP_PIPELINE)
#undef MS_METAL_STEP_PIPELINE
      scan_ = make_pipeline(device_, library, @"step_scan_kernel");
      solve_group_ = make_pipeline(device_, library, @"step_solve_group_kernel");
      solve_respond_ = make_pipeline(device_, library, @"step_solve_respond_kernel");
      solve_rows_ = make_pipeline(device_, library, @"step_solve_rows_kernel");
      solve_record_ = make_pipeline(device_, library, @"step_solve_record_kernel");
    }
  }

  MetalContactStepDevice(const MetalContactStepDevice&) = delete;
  MetalContactStepDevice& operator=(const MetalContactStepDevice&) = delete;
  ~MetalContactStepDevice() override {
    @autoreleasepool {
      flush();
    }
  }

  void reserve(StepBuffer buffer, std::size_t bytes, bool preserve) override {
    const auto index = static_cast<std::size_t>(buffer);
    bytes = std::max<std::size_t>(bytes, 16);
    if (buffers_[index] != nil && buffers_[index].length >= bytes) {
      return;
    }
    @autoreleasepool {
      flush();
      id<MTLBuffer> replacement = [device_ newBufferWithLength:std::bit_ceil(bytes)
                                                       options:MTLResourceStorageModeShared];
      if (replacement == nil) {
        throw std::runtime_error("failed to allocate a Metal contact step buffer");
      }
      if (preserve && buffers_[index] != nil) {
        std::memcpy(replacement.contents, buffers_[index].contents, buffers_[index].length);
      }
      buffers_[index] = replacement;
      bind();
    }
  }

  void upload(StepBuffer buffer, const void* data, std::size_t bytes, std::size_t offset) override {
    @autoreleasepool {
      flush();
    }
    id<MTLBuffer> target = buffers_[static_cast<std::size_t>(buffer)];
    if (target == nil || offset + bytes > target.length) {
      throw std::out_of_range("contact step upload exceeds its buffer");
    }
    std::memcpy(static_cast<std::byte*>(target.contents) + offset, data, bytes);
  }

  void download(StepBuffer buffer, void* data, std::size_t bytes) override {
    @autoreleasepool {
      flush();
    }
    id<MTLBuffer> source = buffers_[static_cast<std::size_t>(buffer)];
    if (source == nil || bytes > source.length) {
      throw std::out_of_range("contact step download exceeds its buffer");
    }
    std::memcpy(data, source.contents, bytes);
  }

  void run(StepKernel kernel, std::uint32_t count,
           const device::ContactStepParams& params) override {
    if (count == 0) {
      return;
    }
    auto pipeline = pipelines_.at(static_cast<std::size_t>(kernel));
    auto* encoder = begin(pipeline, params);
    const auto width = std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256);
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  }

  void scan(const device::ContactStepParams& params) override {
    dispatch_group(scan_, params);
  }

  void solve(const device::ContactStepParams& params, std::uint32_t rows_hint) override {
    if (rows_hint <= group_rows) {
      dispatch_group(solve_group_, params);
      return;
    }
    // A large system solves a step at a time, with the host reading after
    // each batch of steps whether the device has finished.
    std::array<std::uint32_t, MS_STEP_WORDS> words{};
    download(StepBuffer::status, words.data(), sizeof(words));
    if (words[MS_STEP_FAILED] != 0 || words[MS_STEP_OVERFLOW] != 0) {
      return;
    }
    const auto rows = words[MS_STEP_ROWS];
    auto step = params;
    step.output = 0;
    step.project = 0;
    respond(step);
    rows_and_record(step, rows);
    std::uint32_t current = 0;
    for (;;) {
      for (std::uint32_t index = 0; index < solve_batch; ++index) {
        step.output = current;
        step.project = 1;
        respond(step);
        rows_and_record(step, rows);
        current = 1 - current;
      }
      device::ContactSolveState state{};
      download(StepBuffer::solve_state, &state, sizeof(state));
      if (state.done != 0) {
        return;
      }
    }
  }

 private:
  void respond(const device::ContactStepParams& params) {
    auto* encoder = begin(solve_respond_, params);
    const auto width = std::min<NSUInteger>(solve_respond_.maxTotalThreadsPerThreadgroup, 256);
    [encoder dispatchThreads:MTLSizeMake(std::max<std::uint32_t>(params.cell_count, 1), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  }

  void rows_and_record(const device::ContactStepParams& params, std::uint32_t rows) {
    auto* encoder = begin(solve_rows_, params);
    const auto blocks = std::max<std::uint32_t>(
        (rows + shared::contact_reduction_width - 1) / shared::contact_reduction_width, 1);
    [encoder dispatchThreadgroups:MTLSizeMake(blocks, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(shared::contact_reduction_width, 1, 1)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    dispatch_group(solve_record_, params);
  }

  void dispatch_group(id<MTLComputePipelineState> pipeline,
                      const device::ContactStepParams& params) {
    auto* encoder = begin(pipeline, params);
    // A multiple of the reduction width, as the solve's sums need.
    const auto allowed =
        std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, group_threads);
    const auto width = std::max<NSUInteger>(
        allowed / shared::contact_reduction_width * shared::contact_reduction_width,
        shared::contact_reduction_width);
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  }

  // The open encoder, with the pipeline, the buffer table, and the
  // parameters set.
  id<MTLComputeCommandEncoder> begin(id<MTLComputePipelineState> pipeline,
                                     const device::ContactStepParams& params) {
    if (encoder_ == nil) {
      command_ = [queue_ commandBuffer];
      encoder_ = [command_ computeCommandEncoder];
      if (command_ == nil || encoder_ == nil) {
        throw std::runtime_error("failed to create a Metal contact step command");
      }
      std::array<id<MTLResource>, detail::step_buffer_count> resources{};
      std::size_t count = 0;
      for (const auto& buffer : buffers_) {
        if (buffer != nil) {
          resources[count++] = buffer;
        }
      }
      [encoder_ useResources:resources.data()
                       count:count
                       usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    }
    [encoder_ setComputePipelineState:pipeline];
    [encoder_ setBytes:&bindings_ length:sizeof(bindings_) atIndex:0];
    [encoder_ setBytes:&params length:sizeof(params) atIndex:1];
    return encoder_;
  }

  // Runs what is enqueued and waits for it.
  void flush() {
    if (encoder_ == nil) {
      return;
    }
    [encoder_ endEncoding];
    [command_ commit];
    [command_ waitUntilCompleted];
    const auto failed = command_.status == MTLCommandBufferStatusError;
    NSError* error = command_.error;
    encoder_ = nil;
    command_ = nil;
    if (failed) {
      throw_metal_error("Metal contact step command failed", error);
    }
  }

  void bind(){
#define MS_METAL_STEP_BIND(name, type)                                                       \
  bindings_.name = buffers_[static_cast<std::size_t>(StepBuffer::name)] == nil               \
                       ? nullptr                                                             \
                       : reinterpret_cast<std::remove_pointer_t<decltype(bindings_.name)>*>( \
                             buffers_[static_cast<std::size_t>(StepBuffer::name)].gpuAddress);
      MS_CONTACT_STEP_BUFFERS(MS_METAL_STEP_BIND)
#undef MS_METAL_STEP_BIND
  }

  id<MTLDevice> device_;
  id<MTLCommandQueue> queue_;
  std::array<id<MTLComputePipelineState>, detail::step_kernel_count> pipelines_{};
  id<MTLComputePipelineState> scan_{nil};
  id<MTLComputePipelineState> solve_group_{nil};
  id<MTLComputePipelineState> solve_respond_{nil};
  id<MTLComputePipelineState> solve_rows_{nil};
  id<MTLComputePipelineState> solve_record_{nil};
  std::array<id<MTLBuffer>, detail::step_buffer_count> buffers_{};
  device::ContactStepBindings bindings_{};
  id<MTLCommandBuffer> command_{nil};
  id<MTLComputeCommandEncoder> encoder_{nil};
};

}  // namespace

std::unique_ptr<detail::ContactStepDevice> make_contact_step_device(id<MTLDevice> device,
                                                                    id<MTLCommandQueue> queue) {
  return std::make_unique<MetalContactStepDevice>(device, queue);
}

}  // namespace microsimulator::metal
