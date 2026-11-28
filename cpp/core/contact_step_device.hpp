#pragma once

#include <cstddef>
#include <cstdint>

#include "core/device_types.hpp"

namespace microsimulator::detail {

// The buffers of a contact dynamics substep, as shared/contact_step_layout.h
// names them.
enum class StepBuffer : std::uint32_t {
#define MS_STEP_BUFFER_ENTRY(name, type) name,
  MS_CONTACT_STEP_BUFFERS(MS_STEP_BUFFER_ENTRY)
#undef MS_STEP_BUFFER_ENTRY
      count
};

// The per-element stages of a substep, as shared/contact_step_layout.h
// names them.
enum class StepKernel : std::uint32_t {
#define MS_STEP_KERNEL_ENTRY(name, extent) name,
  MS_CONTACT_STEP_KERNELS(MS_STEP_KERNEL_ENTRY)
#undef MS_STEP_KERNEL_ENTRY
      count
};

inline constexpr auto step_buffer_count = static_cast<std::size_t>(StepBuffer::count);
inline constexpr auto step_kernel_count = static_cast<std::size_t>(StepKernel::count);

// A backend's storage for a substep and its means of running the stages of
// shared/contact_step_layout.h on it. Work is enqueued in order; a device
// backend runs it without returning to the host until something is read back.
// Every method is pure virtual, so a backend that leaves one out does not
// compile.
class ContactStepDevice {
 public:
  virtual ~ContactStepDevice() = default;

  // Holds at least `bytes` in `buffer`, keeping its first `bytes` of contents
  // with `preserve`.
  virtual void reserve(StepBuffer buffer, std::size_t bytes, bool preserve) = 0;
  // Copies host data into `buffer` from byte `offset`, after the work already
  // enqueued.
  virtual void upload(StepBuffer buffer, const void* data, std::size_t bytes,
                      std::size_t offset) = 0;
  // Waits for the enqueued work and copies the first `bytes` of `buffer`.
  virtual void download(StepBuffer buffer, void* data, std::size_t bytes) = 0;
  // Runs `kernel` for elements 0, ..., count - 1.
  virtual void run(StepKernel kernel, std::uint32_t count,
                   const device::ContactStepParams& params) = 0;
  // Runs the scan `params.scan` names and records its total.
  virtual void scan(const device::ContactStepParams& params) = 0;
  // Solves the substep's rows from `multipliers` into `solve_state`, the
  // multipliers in pair `iterations % 2` and their response in `response`.
  // `rows_hint` is the number of rows the host expects, which lets a device
  // choose between solving in one group and solving a step at a time.
  virtual void solve(const device::ContactStepParams& params, std::uint32_t rows_hint) = 0;
};

}  // namespace microsimulator::detail
