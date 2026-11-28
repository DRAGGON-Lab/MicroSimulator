#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace microsimulator::cuda {

// What a set of contact multipliers does to one cell: its velocity with the
// elongation rate in w, and its angular velocity with the axial load in w.
struct alignas(16) ContactResponseGpu {
  float4 linear_length;
  float4 rotation;
};

static_assert(sizeof(ContactResponseGpu) == 32);

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

// The progress of a solve on the device: the next step length, the residual
// of the current multipliers, the steps taken, and whether the solve is done.
struct ContactSolveStateGpu {
  float step_length;
  float residual;
  std::uint32_t iterations;
  std::uint32_t done;
};

static_assert(sizeof(ContactRowGpu) == 80);
static_assert(sizeof(ContactBodyGpu) == 64);

// The velocity, angular velocity, elongation rate, and axial load of every
// cell under a set of multipliers. An incidence entry is a row index shifted
// left by one, with the low bit set when the cell is the row's second cell.
// The response holds the velocity and elongation rate in `linear_length` and
// the angular velocity and load in `rotation`.
void launch_contact_respond(const ContactRowGpu* rows, const ContactBodyGpu* bodies,
                            const float* multipliers, const std::uint32_t* incidence_offsets,
                            const std::uint32_t* incidence_entries, ContactResponseGpu* response,
                            ContactSolveGpu solve, cudaStream_t stream);

// The gap of every row under the multipliers whose response is given, and the
// terms of the step that led to them: step . step, step . change,
// change . change, and the row's residual. The initial evaluation has no
// previous step and writes only the residual.
void launch_contact_gaps(const ContactRowGpu* rows, const ContactResponseGpu* response,
                         const float* multipliers, float* gaps, const float* previous_multipliers,
                         const float* previous_gaps, float4* terms, ContactSolveGpu solve,
                         bool initial, cudaStream_t stream);

// The whole projected Barzilai-Borwein solve in one block of 1024 threads.
// The multipliers start in `multipliers_a`; the state ends with the
// iterations taken and the residual of the multipliers in array
// `iterations % 2`, whose response `response` holds.
void launch_contact_solve(const ContactRowGpu* rows, const ContactBodyGpu* bodies,
                          float* multipliers_a, float* multipliers_b, float* gaps_a, float* gaps_b,
                          const std::uint32_t* incidence_offsets,
                          const std::uint32_t* incidence_entries, ContactResponseGpu* response,
                          ContactSolveStateGpu* state, ContactSolveGpu solve, float tolerance,
                          std::uint32_t max_iterations, float first_step_length,
                          cudaStream_t stream);

// Takes `count` projected Barzilai-Borwein steps from the multipliers and
// gaps in pair `current`, alternating pairs, without returning to the host.
// Each step is three launches: the response to the projected multipliers,
// the step with its gaps and per-block terms in `partials`, and one block
// that sums the terms and records the step in `state`. Steps after the one
// that finishes the solve do nothing. Returns the pair that holds the last
// step `count` steps would write; `state->iterations` counts those taken.
void launch_contact_steps(std::uint32_t count, std::uint32_t current, const ContactRowGpu* rows,
                          const ContactBodyGpu* bodies, float* const multipliers[2],
                          float* const gaps[2], const std::uint32_t* incidence_offsets,
                          const std::uint32_t* incidence_entries, ContactResponseGpu* response,
                          float4* partials, ContactSolveStateGpu* state, ContactSolveGpu solve,
                          float tolerance, cudaStream_t stream);

// Sums the first three components of neighboring elements and keeps the larger
// fourth.
void launch_contact_reduce(const float4* input, float4* output, std::uint32_t element_count,
                           cudaStream_t stream);

}  // namespace microsimulator::cuda
