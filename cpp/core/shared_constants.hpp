#pragma once

#include <cstdint>

// Numeric constants shared with the CUDA kernels and the Metal shaders. The
// values live in shared/constants.def.
namespace microsimulator::shared {

#define MICROSIMULATOR_SHARED_FLOAT(name, value) inline constexpr float name = value##F;
#define MICROSIMULATOR_SHARED_UINT(name, value) inline constexpr std::uint32_t name = value##U;
#include "shared/constants.def"
#undef MICROSIMULATOR_SHARED_FLOAT
#undef MICROSIMULATOR_SHARED_UINT

}  // namespace microsimulator::shared
