#pragma once

#include <cuda_runtime_api.h>

#include <memory>

#include "core/contact_step_device.hpp"

namespace microsimulator::cuda {

// The stages of a contact dynamics substep on a CUDA device, launched on
// `stream` and run before the host reads a result back.
[[nodiscard]] std::unique_ptr<detail::ContactStepDevice> make_contact_step_device(
    cudaStream_t stream);

}  // namespace microsimulator::cuda
