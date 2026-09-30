#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "cm/backend.hpp"

namespace cm::test {

template <typename Function>
void for_each_backend_device(Function&& function) {
  constexpr std::array backends{BackendKind::cpu, BackendKind::metal, BackendKind::cuda};

  for (const auto backend : backends) {
    const auto device_count = backend_device_count(backend);

    for (std::size_t index = 0; index < device_count; ++index) {
      if (index > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::overflow_error("backend device index exceeds uint32");
      }

      function(backend, static_cast<std::uint32_t>(index));
    }
  }
}

}  // namespace cm::test
