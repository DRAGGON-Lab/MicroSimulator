#include "cpu/cpu_contact_step.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "core/contact_step_device.hpp"
#include "core/parallel.hpp"
#include "cpu/cpu_contact_solve.hpp"
#include "cpu/kernels/contact_step.hpp"

namespace microsimulator {
namespace {

// Elements handed to one thread at a time.
constexpr std::size_t element_grain = 512;

using detail::StepBuffer;
using detail::StepKernel;

// The stages of cpu/kernels/contact_step.hpp on the host. Each runs when it
// is enqueued, so nothing waits.
class CpuContactStepDevice final : public detail::ContactStepDevice {
 public:
  void reserve(StepBuffer buffer, std::size_t bytes, bool preserve) override {
    auto& storage = storage_[static_cast<std::size_t>(buffer)];
    const auto elements = (bytes + sizeof(device::float4) - 1) / sizeof(device::float4);
    if (elements <= storage.size()) {
      return;
    }
    if (preserve) {
      storage.resize(elements);
    } else {
      storage.assign(elements, device::float4{});
    }
    bind();
  }

  void upload(StepBuffer buffer, const void* data, std::size_t bytes, std::size_t offset) override {
    auto& storage = storage_[static_cast<std::size_t>(buffer)];
    if (offset + bytes > storage.size() * sizeof(device::float4)) {
      throw std::out_of_range("contact step upload exceeds its buffer");
    }
    std::memcpy(reinterpret_cast<std::byte*>(storage.data()) + offset, data, bytes);
  }

  void download(StepBuffer buffer, void* data, std::size_t bytes) override {
    const auto& storage = storage_[static_cast<std::size_t>(buffer)];
    if (bytes > storage.size() * sizeof(device::float4)) {
      throw std::out_of_range("contact step download exceeds its buffer");
    }
    std::memcpy(data, storage.data(), bytes);
  }

  void run(StepKernel kernel, std::uint32_t count,
           const device::ContactStepParams& params) override {
    switch (kernel) {
#define MS_CPU_STEP_CASE(name, extent)                                                         \
  case StepKernel::name:                                                                       \
    run_elements(device::step_live_count(extent, bindings_, params, count),                    \
                 [&](std::uint32_t index) { device::step_##name(index, bindings_, params); }); \
    return;
      MS_CONTACT_STEP_KERNELS(MS_CPU_STEP_CASE)
#undef MS_CPU_STEP_CASE
      case StepKernel::count:
        break;
    }
    throw std::invalid_argument("contact step kernel is invalid");
  }

  void scan(const device::ContactStepParams& params) override {
    if (device::step_gated(bindings_)) {
      return;
    }
    const auto scan = device::step_scan(bindings_, params);
    std::inclusive_scan(scan.input, scan.input + scan.count, scan.output);
    device::step_scan_total(bindings_, params, scan);
  }

  void solve(const device::ContactStepParams& params, std::uint32_t /*rows_hint*/) override {
    detail::solve_resident_contacts_cpu(bindings_, params);
  }

 private:
  // Runs `body` for elements 0, ..., count - 1, on this thread when they are
  // too few to share.
  template <typename Body>
  static void run_elements(std::uint32_t count, const Body& body) {
    if (count <= element_grain) {
      for (std::uint32_t index = 0; index < count; ++index) {
        body(index);
      }
      return;
    }
    detail::parallel_for(count, element_grain, [&](std::size_t begin, std::size_t end) {
      for (auto index = begin; index < end; ++index) {
        body(static_cast<std::uint32_t>(index));
      }
    });
  }

  void bind(){
#define MS_CPU_STEP_BIND(name, type)                                                   \
  bindings_.name = reinterpret_cast<std::remove_pointer_t<decltype(bindings_.name)>*>( \
      storage_[static_cast<std::size_t>(StepBuffer::name)].data());
      MS_CONTACT_STEP_BUFFERS(MS_CPU_STEP_BIND)
#undef MS_CPU_STEP_BIND
  }

  std::array<std::vector<device::float4>, detail::step_buffer_count> storage_;
  device::ContactStepBindings bindings_{};
};

}  // namespace

std::unique_ptr<detail::ContactStepDevice> make_cpu_contact_step_device() {
  return std::make_unique<CpuContactStepDevice>();
}

}  // namespace microsimulator
