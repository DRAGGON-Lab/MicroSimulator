#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

#include "cm/metal/occupancy_source.hpp"
#include "core/occupancy_device.hpp"

namespace cm::detail {
namespace {
[[noreturn]] void fail(const char* action, NSError* error) {
  throw std::runtime_error(
      std::string(action) + ": " +
      (error ? error.localizedDescription.UTF8String : "Metal resource unavailable"));
}
struct MetalOccupancyBuffer final : OccupancyBuffer {
  id<MTLBuffer> value;
  explicit MetalOccupancyBuffer(id<MTLBuffer> buffer) : value(buffer) {}
};
class MetalOccupancyDevice final : public OccupancyDevice {
 public:
  explicit MetalOccupancyDevice(std::uint32_t index) {
    @autoreleasepool {
      NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
      if (devices.count == 0 && index == 0)
        device_ = MTLCreateSystemDefaultDevice();
      else if (index < devices.count)
        device_ = devices[index];
      if (!device_) throw std::out_of_range("Metal occupancy device index is unavailable");
      queue_ = [device_ newCommandQueue];
      if (!queue_) fail("create occupancy command queue", nil);
      MTLCompileOptions* options = [MTLCompileOptions new];
      if (@available(macOS 15.0, *)) {
        options.mathMode = MTLMathModeSafe;
        options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
      } else {
        options.fastMathEnabled = NO;
      }
      NSError* error = nil;
      id<MTLLibrary> library =
          [device_ newLibraryWithSource:[NSString stringWithUTF8String:cm::metal::occupancy_source]
                                options:options
                                  error:&error];
      if (!library) fail("compile Metal occupancy kernels", error);
      for (const char* name : occupancy_kernel_names) {
        id<MTLFunction> function =
            [library newFunctionWithName:[NSString stringWithUTF8String:name]];
        if (!function) fail("find Metal occupancy kernel", nil);
        auto pipeline = [device_ newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) fail("create Metal occupancy pipeline", error);
        pipelines_.push_back(pipeline);
      }
    }
  }
  OccupancyBufferPtr allocate(std::size_t bytes, const void* data) override {
    @autoreleasepool {
      id<MTLBuffer> buffer = [device_ newBufferWithLength:std::max(bytes, std::size_t{4})
                                                  options:MTLResourceStorageModeShared];
      if (!buffer) fail("allocate Metal occupancy buffer", nil);
      if (data && bytes) std::memcpy(buffer.contents, data, bytes);
      return std::make_shared<MetalOccupancyBuffer>(buffer);
    }
  }
  void read(const OccupancyBufferPtr& buffer, void* data, std::size_t bytes) override {
    std::memcpy(data, static_cast<MetalOccupancyBuffer&>(*buffer).value.contents, bytes);
  }
  void dispatch(OccupancyKernel kernel, const OccupancyParameters& p,
                std::initializer_list<OccupancyBufferPtr> buffers) override {
    if (p.count == 0) return;
    @autoreleasepool {
      auto pipeline = pipelines_[static_cast<std::size_t>(kernel)];
      id<MTLCommandBuffer> command = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      if (!command || !encoder) fail("create Metal occupancy command", nil);
      [encoder setComputePipelineState:pipeline];
      NSUInteger index = 0;
      for (const auto& buffer : buffers)
        [encoder setBuffer:static_cast<MetalOccupancyBuffer&>(*buffer).value
                    offset:0
                   atIndex:index++];
      [encoder setBytes:&p length:sizeof(p) atIndex:15];
      const auto count = kernel == OccupancyKernel::sum ? 1 : p.count;
      const auto width = std::min(NSUInteger{64}, pipeline.maxTotalThreadsPerThreadgroup);
      [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status == MTLCommandBufferStatusError)
        fail("execute Metal occupancy kernel", command.error);
    }
  }

 private:
  id<MTLDevice> device_;
  id<MTLCommandQueue> queue_;
  std::vector<id<MTLComputePipelineState>> pipelines_;
};
}  // namespace
std::unique_ptr<OccupancyDevice> make_metal_occupancy_device(std::uint32_t index) {
  return std::make_unique<MetalOccupancyDevice>(index);
}
}  // namespace cm::detail
