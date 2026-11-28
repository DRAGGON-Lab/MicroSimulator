#pragma once

#include <cstddef>
#include <cstdint>

// Host view of the structures in shared/device_types.h. The scalar and vector
// names below reproduce the size and alignment of the Metal types of the same
// name, so the shared definitions lay out identically on both sides.
namespace microsimulator::device {

using uint = std::uint32_t;
using ulong = std::uint64_t;
using uchar = std::uint8_t;

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

#define MICROSIMULATOR_DEVICE_POINTER(type) type*
#include "shared/contact_step_layout.h"

static_assert(sizeof(float2) == 8 && alignof(float2) == 4);
static_assert(sizeof(uint2) == 8 && alignof(uint2) == 4);
static_assert(sizeof(float4) == 16 && alignof(float4) == 16);
static_assert(sizeof(uint4) == 16 && alignof(uint4) == 16);
static_assert(sizeof(RateInstruction) == 20);
static_assert(sizeof(ContactResponse) == 32);
static_assert(sizeof(ContactRowData) == 80 && offsetof(ContactRowData, first_slot) == 64);
static_assert(sizeof(ContactBodyData) == 64);
static_assert(sizeof(ContactSolveData) == 32);
static_assert(sizeof(ContactSolveState) == 16);
static_assert(sizeof(ExternalConstraint) == 48 && offsetof(ExternalConstraint, geometry) == 16);
static_assert(sizeof(GridShape) == 16);
static_assert(sizeof(FlowGridParameters) == 96 && offsetof(FlowGridParameters, site_count) == 80);
static_assert(sizeof(DepthFlowScales) == 32);
static_assert(sizeof(DepositGrid) == 80 && offsetof(DepositGrid, cell_count) == 64);
static_assert(sizeof(DepositBox) == 32);
static_assert(sizeof(ShallowCouplingData) == 32);
static_assert(sizeof(BrinkmanDragData) == 32);
static_assert(sizeof(ExternalContactData) == 32);
static_assert(sizeof(ExternalContactRecord) == 48);
static_assert(sizeof(FlowDriftData) == 64);
static_assert(sizeof(CellContactData) == 32);
static_assert(sizeof(CellContactRecord) == 48);
static_assert(sizeof(ContactTouch) == 48);
static_assert(sizeof(SignalSweepState) == 16);

}  // namespace microsimulator::device
