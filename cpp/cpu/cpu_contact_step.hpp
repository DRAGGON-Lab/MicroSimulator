#pragma once

#include <memory>

#include "core/contact_step_device.hpp"

namespace microsimulator {

// The stages of a contact dynamics substep on the host.
[[nodiscard]] std::unique_ptr<detail::ContactStepDevice> make_cpu_contact_step_device();

}  // namespace microsimulator
