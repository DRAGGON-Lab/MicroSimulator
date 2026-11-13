#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "mechanics.cuh"

namespace microsimulator::cuda {

// One row of the contact dynamics solve. The w components carry the row's
// scalars: the first and second length levers ride on the linear parts, the
// offset on the first angular part, and the compliance on the second.
struct alignas(16) ContactRowGpu {
  float4 first_linear;
  float4 first_angular;
  float4 second_linear;
  float4 second_angular;
  std::uint32_t first_slot;
  std::uint32_t second_slot;
  std::uint32_t padding0;
  std::uint32_t padding1;
};

// Cell axis with the translational mobility in w; the transverse mobility,
// axial mobility, and nominal elongation rate; then the velocity and angular
// velocity the cell has without contact.
struct alignas(16) ContactBodyGpu {
  float4 axis_mobility;
  float4 rates;
  float4 drift_linear;
  float4 drift_angular;
};

struct ContactSolveGpu {
  float dt;
  std::uint32_t law;
  float sensitivity;
  float free_load;
  float stall_load;
  std::uint32_t row_count;
  std::uint32_t body_count;
};

static_assert(sizeof(ContactRowGpu) == 80);
static_assert(sizeof(ContactBodyGpu) == 64);

// next = max(multipliers - step_length * gaps, 0).
void launch_contact_project(const float* multipliers, const float* gaps, float* next,
                            float step_length, std::uint32_t row_count, cudaStream_t stream);

// The velocity, angular velocity, elongation rate, and axial load of every
// cell under a set of multipliers. An incidence entry is a row index shifted
// left by one, with the low bit set when the cell is the row's second cell.
// The response holds the velocity and elongation rate in `linear_length` and
// the angular velocity and load in `rotation`.
void launch_contact_respond(const ContactRowGpu* rows, const ContactBodyGpu* bodies,
                            const float* multipliers, const std::uint32_t* incidence_offsets,
                            const std::uint32_t* incidence_entries, MechanicsDofsGpu* response,
                            ContactSolveGpu solve, cudaStream_t stream);

// The gap of every row under the multipliers whose response is given, and the
// terms of the step that led to them: step . step, step . change,
// change . change, and the row's residual. The initial evaluation has no
// previous step and writes only the residual.
void launch_contact_gaps(const ContactRowGpu* rows, const MechanicsDofsGpu* response,
                         const float* multipliers, float* gaps, const float* previous_multipliers,
                         const float* previous_gaps, float4* terms, ContactSolveGpu solve,
                         bool initial, cudaStream_t stream);

// Sums the first three components of neighboring elements and keeps the larger
// fourth.
void launch_contact_reduce(const float4* input, float4* output, std::uint32_t element_count,
                           cudaStream_t stream);

}  // namespace microsimulator::cuda
