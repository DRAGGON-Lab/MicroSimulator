#pragma once

#import <Metal/Metal.h>

#include <memory>

#include "core/contact_step_device.hpp"

namespace microsimulator::metal {

// The stages of a contact dynamics substep on a Metal device, enqueued on
// `queue` and run when the host reads a result back.
[[nodiscard]] std::unique_ptr<detail::ContactStepDevice> make_contact_step_device(
    id<MTLDevice> device, id<MTLCommandQueue> queue);

}  // namespace microsimulator::metal
