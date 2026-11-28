#pragma once

#include <cuda_runtime.h>

#include "core/shared_constants.hpp"

// CUDA view of the structures in shared/device_types.h. CUDA's vector types
// have the size and alignment of the Metal types of the same name, so the
// structures lay out as they do on the host and in Metal.
namespace microsimulator::cuda::device {

using uint = unsigned int;
using ulong = unsigned long long;
using uchar = unsigned char;
using ::float2;
using ::float4;
using ::uint2;
using ::uint4;

#include "shared/device_types.h"

static_assert(sizeof(DepositGrid) == 80);
static_assert(sizeof(DepositBox) == 32);
static_assert(sizeof(ShallowCouplingData) == 32);
static_assert(sizeof(BrinkmanDragData) == 32);

}  // namespace microsimulator::cuda::device
