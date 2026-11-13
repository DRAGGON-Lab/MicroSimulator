#pragma once

#include <cstddef>
#include <cstdint>

// Host view of the structures in shared/device_types.h. The scalar and vector
// names below reproduce the size and alignment of the Metal types of the same
// name, so the shared definitions lay out identically on both sides.
namespace microsimulator::device {

using uint = std::uint32_t;
using ulong = std::uint64_t;

struct float2 {
  float x;
  float y;
};

struct uint2 {
  std::uint32_t x;
  std::uint32_t y;
};

struct alignas(16) float4 {
  float x;
  float y;
  float z;
  float w;
};

struct alignas(16) uint4 {
  std::uint32_t x;
  std::uint32_t y;
  std::uint32_t z;
  std::uint32_t w;
};

#include "shared/device_types.h"

static_assert(sizeof(float2) == 8 && alignof(float2) == 4);
static_assert(sizeof(uint2) == 8 && alignof(uint2) == 4);
static_assert(sizeof(float4) == 16 && alignof(float4) == 16);
static_assert(sizeof(uint4) == 16 && alignof(uint4) == 16);
static_assert(sizeof(RateInstruction) == 20);
static_assert(sizeof(MechanicsDofs) == 32);
static_assert(sizeof(ContactRowData) == 80 && offsetof(ContactRowData, first_slot) == 64);
static_assert(sizeof(ContactBodyData) == 32);
static_assert(sizeof(ContactSolveData) == 32);
static_assert(sizeof(ContactSolveState) == 16);
static_assert(sizeof(ExternalConstraint) == 48 && offsetof(ExternalConstraint, geometry) == 16);
static_assert(sizeof(GridShape) == 16);
static_assert(sizeof(FlowGridParameters) == 96 && offsetof(FlowGridParameters, flow_axis) == 80);
static_assert(sizeof(DepthFlowScales) == 32);

}  // namespace microsimulator::device
