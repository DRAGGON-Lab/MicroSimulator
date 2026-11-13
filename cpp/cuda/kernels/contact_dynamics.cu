#include <cmath>

#include "contact_dynamics.cuh"
#include "core/shared_constants.hpp"

namespace microsimulator::cuda {
namespace {

using shared::growth_load_law_exponential;
using shared::growth_load_law_linear;

constexpr std::uint32_t threads_per_block = 256;
constexpr std::uint32_t no_second_slot = 0xffffffffU;

__device__ float dot3(float4 left, float4 right) {
  return left.x * right.x + left.y * right.y + left.z * right.z;
}

__device__ float contact_growth_factor(const ContactSolveGpu& solve, float load) {
  if (solve.law == growth_load_law_exponential) {
    return expf(-solve.sensitivity * fmaxf(load, 0.0F));
  }
  if (solve.law == growth_load_law_linear) {
    const float factor = 1.0F - (load - solve.free_load) / (solve.stall_load - solve.free_load);
    return fminf(fmaxf(factor, 0.0F), 1.0F);
  }
  return 1.0F;
}

__global__ void contact_project(const float* multipliers, const float* gaps, float* next,
                                float step_length, std::uint32_t row_count) {
  const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= row_count) {
    return;
  }
  next[row] = fmaxf(multipliers[row] - step_length * gaps[row], 0.0F);
}

__global__ void contact_respond(const ContactRowGpu* rows, const ContactBodyGpu* bodies,
                                const float* multipliers, const std::uint32_t* incidence_offsets,
                                const std::uint32_t* incidence_entries, MechanicsDofsGpu* response,
                                ContactSolveGpu solve) {
  const std::uint32_t body = blockIdx.x * blockDim.x + threadIdx.x;
  if (body >= solve.body_count) {
    return;
  }
  float force_x = 0.0F;
  float force_y = 0.0F;
  float force_z = 0.0F;
  float torque_x = 0.0F;
  float torque_y = 0.0F;
  float torque_z = 0.0F;
  float load = 0.0F;
  for (std::uint32_t offset = incidence_offsets[body]; offset < incidence_offsets[body + 1];
       ++offset) {
    const std::uint32_t entry = incidence_entries[offset];
    const std::uint32_t row = entry >> 1U;
    const float multiplier = multipliers[row];
    const bool second = (entry & 1U) != 0U;
    const float4 linear = second ? rows[row].second_linear : rows[row].first_linear;
    const float4 angular = second ? rows[row].second_angular : rows[row].first_angular;
    force_x += linear.x * multiplier;
    force_y += linear.y * multiplier;
    force_z += linear.z * multiplier;
    torque_x += angular.x * multiplier;
    torque_y += angular.y * multiplier;
    torque_z += angular.z * multiplier;
    load += linear.w * multiplier;
  }
  const float4 axis = bodies[body].axis_mobility;
  const float translational = axis.w;
  const float transverse = bodies[body].rates.x;
  const float axial = bodies[body].rates.y;
  const float nominal = bodies[body].rates.z;
  const float along =
      (axial - transverse) * (axis.x * torque_x + axis.y * torque_y + axis.z * torque_z);
  const float rate = nominal > 0.0F ? nominal * contact_growth_factor(solve, load) : nominal;
  response[body].linear_length =
      make_float4(force_x * translational, force_y * translational, force_z * translational, rate);
  response[body].rotation =
      make_float4(torque_x * transverse + axis.x * along, torque_y * transverse + axis.y * along,
                  torque_z * transverse + axis.z * along, load);
}

__global__ void contact_gaps(const ContactRowGpu* rows, const MechanicsDofsGpu* response,
                             const float* multipliers, float* gaps,
                             const float* previous_multipliers, const float* previous_gaps,
                             float4* terms, ContactSolveGpu solve, bool initial) {
  const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= solve.row_count) {
    return;
  }
  const std::uint32_t first = rows[row].first_slot;
  const std::uint32_t second = rows[row].second_slot;
  float rate = dot3(rows[row].first_linear, response[first].linear_length) +
               dot3(rows[row].first_angular, response[first].rotation) -
               rows[row].first_linear.w * response[first].linear_length.w;
  if (second != no_second_slot) {
    rate += dot3(rows[row].second_linear, response[second].linear_length) +
            dot3(rows[row].second_angular, response[second].rotation) -
            rows[row].second_linear.w * response[second].linear_length.w;
  }
  const float multiplier = multipliers[row];
  const float gap =
      rows[row].first_angular.w + solve.dt * rate + rows[row].second_angular.w * multiplier;
  gaps[row] = gap;
  float residual = multiplier > 0.0F ? fabsf(gap) : fmaxf(0.0F, -gap);
  if (!isfinite(gap) || !isfinite(multiplier)) {
    residual = HUGE_VALF;
  }
  if (initial) {
    terms[row] = make_float4(0.0F, 0.0F, 0.0F, residual);
    return;
  }
  const float step = multiplier - previous_multipliers[row];
  const float change = gap - previous_gaps[row];
  terms[row] = make_float4(step * step, step * change, change * change, residual);
}

__global__ void contact_reduce(const float4* input, float4* output, std::uint32_t element_count) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t first = index * 2U;
  if (first >= element_count) {
    return;
  }
  float4 value = input[first];
  if (first + 1U < element_count) {
    const float4 other = input[first + 1U];
    value = make_float4(value.x + other.x, value.y + other.y, value.z + other.z,
                        fmaxf(value.w, other.w));
  }
  output[index] = value;
}

std::uint32_t block_count(std::uint32_t count) {
  return ((count - 1) / threads_per_block) + 1;
}

}  // namespace

void launch_contact_project(const float* multipliers, const float* gaps, float* next,
                            float step_length, std::uint32_t row_count, cudaStream_t stream) {
  contact_project<<<block_count(row_count), threads_per_block, 0, stream>>>(multipliers, gaps, next,
                                                                            step_length, row_count);
}

void launch_contact_respond(const ContactRowGpu* rows, const ContactBodyGpu* bodies,
                            const float* multipliers, const std::uint32_t* incidence_offsets,
                            const std::uint32_t* incidence_entries, MechanicsDofsGpu* response,
                            ContactSolveGpu solve, cudaStream_t stream) {
  contact_respond<<<block_count(solve.body_count), threads_per_block, 0, stream>>>(
      rows, bodies, multipliers, incidence_offsets, incidence_entries, response, solve);
}

void launch_contact_gaps(const ContactRowGpu* rows, const MechanicsDofsGpu* response,
                         const float* multipliers, float* gaps, const float* previous_multipliers,
                         const float* previous_gaps, float4* terms, ContactSolveGpu solve,
                         bool initial, cudaStream_t stream) {
  contact_gaps<<<block_count(solve.row_count), threads_per_block, 0, stream>>>(
      rows, response, multipliers, gaps, previous_multipliers, previous_gaps, terms, solve,
      initial);
}

void launch_contact_reduce(const float4* input, float4* output, std::uint32_t element_count,
                           cudaStream_t stream) {
  const auto output_count = (element_count + 1) / 2;
  contact_reduce<<<block_count(output_count), threads_per_block, 0, stream>>>(input, output,
                                                                              element_count);
}

}  // namespace microsimulator::cuda
