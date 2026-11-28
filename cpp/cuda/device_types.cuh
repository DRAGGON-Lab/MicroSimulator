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
using ::float3;
using ::float4;
using ::uint2;
using ::uint4;
using microsimulator::shared::contact_reduction_width;
using microsimulator::shared::float_epsilon;
using microsimulator::shared::growth_load_law_exponential;
using microsimulator::shared::growth_load_law_linear;
using microsimulator::shared::inverse_sqrt_two;
using microsimulator::shared::pi;
using microsimulator::shared::segment_minimization_iterations;

#include "shared/device_types.h"

static_assert(sizeof(DepositGrid) == 80);
static_assert(sizeof(DepositBox) == 32);
static_assert(sizeof(ShallowCouplingData) == 32);
static_assert(sizeof(BrinkmanDragData) == 32);
static_assert(sizeof(ExternalConstraint) == 48);
static_assert(sizeof(ExternalContactData) == 32);
static_assert(sizeof(ExternalContactRecord) == 48);
static_assert(sizeof(FlowDriftData) == 64);
static_assert(sizeof(CellContactData) == 32);
static_assert(sizeof(CellContactRecord) == 48);

}  // namespace microsimulator::cuda::device
