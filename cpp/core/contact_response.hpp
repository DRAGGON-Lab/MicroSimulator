#pragma once

#include <span>

#include "core/device_types.hpp"
#include "microsimulator/contact_dynamics.hpp"

namespace microsimulator::detail {

// A contact response as the device kernels store it, one ContactResponse of
// shared/device_types.h per cell, unpacked into the host's form.
[[nodiscard]] inline ContactResponse unpack_contact_response(
    std::span<const device::ContactResponse> packed) {
  ContactResponse response;
  response.velocities.resize(packed.size());
  response.loads.resize(packed.size());
  response.length_rates.resize(packed.size());
  for (std::size_t body = 0; body < packed.size(); ++body) {
    const auto& linear = packed[body].linear_length;
    const auto& rotation = packed[body].rotation;
    response.velocities[body] = {linear.x, linear.y, linear.z, rotation.x, rotation.y, rotation.z};
    response.loads[body] = rotation.w;
    response.length_rates[body] = linear.w;
  }
  return response;
}

}  // namespace microsimulator::detail
