#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <string>

#include "cm/simulation.hpp"

namespace {

int fail(const std::string& message) {
  std::cerr << "CUDA runtime gate failed: " << message << '\n';

  return 1;
}

}  // namespace

int main() {
  const auto device_count = cm::backend_device_count(cm::BackendKind::cuda);

  if (device_count == 0) {
    return fail("the CUDA-enabled build did not discover an NVIDIA device");
  }

  try {
    for (std::size_t index = 0; index < device_count; ++index) {
      if (index > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        return fail("the CUDA device index exceeds the backend index type");
      }

      const auto device_index = static_cast<std::uint32_t>(index);

      if (!cm::backend_available(cm::BackendKind::cuda, device_index)) {
        return fail("an enumerated CUDA device is not available through the backend");
      }

      cm::Simulation simulation(cm::BackendKind::cuda, 0, device_index);
      const auto info = simulation.backend_info();

      if (info.kind != cm::BackendKind::cuda || !info.native || info.device_index != device_index ||
          info.name != "cuda" || info.device.empty()) {
        return fail("the constructed backend did not identify the selected native CUDA device");
      }

      simulation.step(0.0F);
      simulation.validate();
    }
  } catch (const std::exception& error) {
    return fail(error.what());
  }

  std::cout << "validated " << device_count << " native CUDA device(s)\n";

  return 0;
}
