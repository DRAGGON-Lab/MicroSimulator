#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

#include "cm/metal/numerics_source.hpp"
#include "core/numerics_device.hpp"

namespace cm::detail {
namespace {
[[noreturn]] void fail(const char* action, NSError* error) {
  throw std::runtime_error(
      std::string(action) + ": " +
      (error ? error.localizedDescription.UTF8String : "Metal resource unavailable"));
}

struct MetalNumericsBuffer final : NumericsBuffer {
  id<MTLBuffer> value;

  explicit MetalNumericsBuffer(id<MTLBuffer> buffer) : value(buffer) {}
};

class MetalNumericsDevice final : public NumericsDevice {
 public:
  explicit MetalNumericsDevice(std::uint32_t index) {
    @autoreleasepool {
      NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();

      if (devices.count == 0 && index == 0) {
        device_ = MTLCreateSystemDefaultDevice();
      } else if (index < devices.count) {
        device_ = devices[index];
      }

      if (!device_) {
        throw std::out_of_range("Metal numerics device index is unavailable");
      }

      queue_ = [device_ newCommandQueue];

      if (!queue_) {
        fail("create numerics command queue", nil);
      }

      MTLCompileOptions* options = [MTLCompileOptions new];

      if (@available(macOS 15.0, *)) {
        options.mathMode = MTLMathModeSafe;
        options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
      } else {
        options.fastMathEnabled = NO;
      }

      NSError* error = nil;
      id<MTLLibrary> library =
          [device_ newLibraryWithSource:[NSString stringWithUTF8String:cm::metal::numerics_source]
                                options:options
                                  error:&error];
      if (!library) {
        fail("compile Metal numerics kernels", error);
      }

      for (const char* name : numerics_kernel_names) {
        id<MTLFunction> function =
            [library newFunctionWithName:[NSString stringWithUTF8String:name]];

        if (!function) {
          fail("find Metal numerics kernel", nil);
        }

        auto pipeline = [device_ newComputePipelineStateWithFunction:function error:&error];

        if (!pipeline) {
          fail("create Metal numerics pipeline", error);
        }

        pipelines_.push_back(pipeline);
      }
    }
  }

  ~MetalNumericsDevice() override {
    try {
      finish();
    } catch (...) { /* Errors during a solve are reported by read(). */
    }
  }

  NumericsBufferPtr allocate(std::size_t bytes, const void* data) override {
    @autoreleasepool {
      id<MTLBuffer> buffer = [device_ newBufferWithLength:std::max(bytes, std::size_t{4})
                                                  options:MTLResourceStorageModeShared];
      if (!buffer) {
        fail("allocate Metal numerics buffer", nil);
      }

      if (data && bytes) {
        std::memcpy(buffer.contents, data, bytes);
      }

      return std::make_shared<MetalNumericsBuffer>(buffer);
    }
  }

  void read(const NumericsBufferPtr& buffer, void* data, std::size_t bytes) override {
    finish();
    std::memcpy(data, static_cast<MetalNumericsBuffer&>(*buffer).value.contents, bytes);
  }

  void dispatch(NumericsKernel kernel, const NumericsParameters& p,
                std::initializer_list<NumericsBufferPtr> buffers) override {
    if (p.count == 0) {
      return;
    }
    @autoreleasepool {
      auto pipeline = pipelines_[static_cast<std::size_t>(kernel)];

      if (!command_) {
        command_ = [queue_ commandBuffer];
        encoder_ = [command_ computeCommandEncoder];
      }

      auto encoder = encoder_;

      if (!command_ || !encoder) {
        fail("create Metal numerics command", nil);
      }
      [encoder setComputePipelineState:pipeline];
      NSUInteger index = 0;

      for (const auto& buffer : buffers) {
        [encoder setBuffer:static_cast<MetalNumericsBuffer&>(*buffer).value
                    offset:0
                   atIndex:index++];
      }
      [encoder setBytes:&p length:sizeof(p) atIndex:15];
      const auto count = kernel == NumericsKernel::ilu   ? 1U
                         : kernel == NumericsKernel::dot ? ((p.count + 127U) / 128U) * 128U
                                                         : p.count;
      const auto width = kernel == NumericsKernel::ilu ? NSUInteger{1} : NSUInteger{128};
      [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    }
  }

 private:
  void finish() {
    @autoreleasepool {
      if (!command_) {
        return;
      }
      auto command = command_;
      [encoder_ endEncoding];
      encoder_ = nil;
      command_ = nil;
      [command commit];
      [command waitUntilCompleted];
      if (command.status == MTLCommandBufferStatusError) {
        fail("execute Metal numerics kernel", command.error);
      }
    }
  }

  id<MTLDevice> device_;
  id<MTLCommandQueue> queue_;
  id<MTLCommandBuffer> command_;
  id<MTLComputeCommandEncoder> encoder_;
  std::vector<id<MTLComputePipelineState>> pipelines_;
};
}  // namespace

std::unique_ptr<NumericsDevice> make_metal_numerics_device(std::uint32_t index) {
  return std::make_unique<MetalNumericsDevice>(index);
}
}  // namespace cm::detail
