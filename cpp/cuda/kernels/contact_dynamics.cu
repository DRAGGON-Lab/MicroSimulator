#include <cmath>

#include "contact_dynamics.cuh"
#include "core/shared_constants.hpp"

namespace microsimulator::cuda {
namespace {

using shared::growth_load_law_exponential;
using shared::growth_load_law_linear;

constexpr std::uint32_t threads_per_block = 256;
constexpr std::uint32_t no_second_slot = 0xffffffffU;
constexpr std::uint32_t contact_solve_block = 1024;

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

// The response of one cell to a set of multipliers: its velocity and
// elongation rate in `linear_length`, its angular velocity and axial load in
// `rotation`.
__device__ ContactResponseGpu respond_body(const ContactRowGpu* rows, const ContactBodyGpu* bodies,
                                           const float* multipliers,
                                           const std::uint32_t* incidence_offsets,
                                           const std::uint32_t* incidence_entries,
                                           const ContactSolveGpu& solve, std::uint32_t body) {
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
  const float4 drift_linear = bodies[body].drift_linear;
  const float4 drift_angular = bodies[body].drift_angular;
  ContactResponseGpu result;
  result.linear_length = make_float4(drift_linear.x + force_x * translational,
                                     drift_linear.y + force_y * translational,
                                     drift_linear.z + force_z * translational, rate);
  result.rotation = make_float4(drift_angular.x + torque_x * transverse + axis.x * along,
                                drift_angular.y + torque_y * transverse + axis.y * along,
                                drift_angular.z + torque_z * transverse + axis.z * along, load);
  return result;
}

__global__ void contact_respond(const ContactRowGpu* rows, const ContactBodyGpu* bodies,
                                const float* multipliers, const std::uint32_t* incidence_offsets,
                                const std::uint32_t* incidence_entries,
                                ContactResponseGpu* response, ContactSolveGpu solve) {
  const std::uint32_t body = blockIdx.x * blockDim.x + threadIdx.x;
  if (body >= solve.body_count) {
    return;
  }
  response[body] =
      respond_body(rows, bodies, multipliers, incidence_offsets, incidence_entries, solve, body);
}

// The gap of one row under the multipliers whose response is given.
__device__ float row_gap(const ContactRowGpu* rows, const ContactResponseGpu* response,
                         float multiplier, const ContactSolveGpu& solve, std::uint32_t row) {
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
  return rows[row].first_angular.w + solve.dt * rate + rows[row].second_angular.w * multiplier;
}

__device__ float row_residual(float multiplier, float gap) {
  float residual = multiplier > 0.0F ? fabsf(gap) : fmaxf(0.0F, -gap);
  if (!isfinite(gap) || !isfinite(multiplier)) {
    residual = HUGE_VALF;
  }
  return residual;
}

__global__ void contact_gaps(const ContactRowGpu* rows, const ContactResponseGpu* response,
                             const float* multipliers, float* gaps,
                             const float* previous_multipliers, const float* previous_gaps,
                             float4* terms, ContactSolveGpu solve, bool initial) {
  const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= solve.row_count) {
    return;
  }
  const float multiplier = multipliers[row];
  const float gap = row_gap(rows, response, multiplier, solve, row);
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

// Combines the partial sums of every thread of the block into element zero:
// the first three components add and the fourth takes the larger.
__device__ void reduce_block(float4* partial) {
  for (std::uint32_t stride = contact_solve_block / 2U; stride > 0U; stride /= 2U) {
    if (threadIdx.x < stride) {
      const float4 other = partial[threadIdx.x + stride];
      const float4 value = partial[threadIdx.x];
      partial[threadIdx.x] = make_float4(value.x + other.x, value.y + other.y, value.z + other.z,
                                         fmaxf(value.w, other.w));
    }
    __syncthreads();
  }
}

// The whole projected Barzilai-Borwein solve of a contact system in one
// block, as contact_respond_step, contact_gaps_step, and contact_advance_step
// compute it a step at a time over the whole device. The multipliers start in `multipliers_a` and
// each step writes the other array. The state ends with the iterations taken and the residual of
// the multipliers in array `iterations % 2`, whose response `response` holds.
__global__ void __launch_bounds__(contact_solve_block)
    contact_solve(const ContactRowGpu* rows, const ContactBodyGpu* bodies, float* multipliers_a,
                  float* multipliers_b, float* gaps_a, float* gaps_b,
                  const std::uint32_t* incidence_offsets, const std::uint32_t* incidence_entries,
                  ContactResponseGpu* response, ContactSolveStateGpu* state, ContactSolveGpu solve,
                  float tolerance, std::uint32_t max_iterations, float first_step_length) {
  __shared__ float4 partial[contact_solve_block];
  __shared__ float step_length;
  __shared__ bool done;
  float* multipliers[2] = {multipliers_a, multipliers_b};
  float* gaps[2] = {gaps_a, gaps_b};
  const std::uint32_t lane = threadIdx.x;
  const std::uint32_t lanes = blockDim.x;

  for (std::uint32_t body = lane; body < solve.body_count; body += lanes) {
    response[body] = respond_body(rows, bodies, multipliers_a, incidence_offsets, incidence_entries,
                                  solve, body);
  }
  __syncthreads();
  float4 local = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
  for (std::uint32_t row = lane; row < solve.row_count; row += lanes) {
    const float multiplier = multipliers_a[row];
    const float gap = row_gap(rows, response, multiplier, solve, row);
    gaps_a[row] = gap;
    local.w = fmaxf(local.w, row_residual(multiplier, gap));
  }
  partial[lane] = local;
  for (std::uint32_t index = lane + lanes; index < contact_solve_block; index += lanes) {
    partial[index] = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
  }
  __syncthreads();
  reduce_block(partial);
  if (lane == 0U) {
    const float residual = partial[0].w;
    state->step_length = first_step_length;
    state->residual = residual;
    state->iterations = 0U;
    done = !(residual > tolerance) || isinf(residual);
    step_length = first_step_length;
  }
  __syncthreads();

  std::uint32_t current = 0U;
  for (std::uint32_t iteration = 1U; iteration <= max_iterations && !done; ++iteration) {
    const std::uint32_t next = 1U - current;
    const float length = step_length;
    for (std::uint32_t row = lane; row < solve.row_count; row += lanes) {
      multipliers[next][row] = fmaxf(multipliers[current][row] - length * gaps[current][row], 0.0F);
    }
    __syncthreads();
    for (std::uint32_t body = lane; body < solve.body_count; body += lanes) {
      response[body] = respond_body(rows, bodies, multipliers[next], incidence_offsets,
                                    incidence_entries, solve, body);
    }
    __syncthreads();
    float4 terms = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    for (std::uint32_t row = lane; row < solve.row_count; row += lanes) {
      const float multiplier = multipliers[next][row];
      const float gap = row_gap(rows, response, multiplier, solve, row);
      gaps[next][row] = gap;
      const float step = multiplier - multipliers[current][row];
      const float change = gap - gaps[current][row];
      terms = make_float4(terms.x + step * step, terms.y + step * change, terms.z + change * change,
                          fmaxf(terms.w, row_residual(multiplier, gap)));
    }
    partial[lane] = terms;
    __syncthreads();
    reduce_block(partial);
    if (lane == 0U) {
      const float4 sums = partial[0];
      state->iterations = iteration;
      state->residual = sums.w;
      if (!(sums.w > tolerance) || isinf(sums.w)) {
        done = true;
      } else if (sums.y > 0.0F && sums.z > 0.0F) {
        const float candidate = (iteration & 1U) != 0U ? sums.x / sums.y : sums.y / sums.z;
        if (isfinite(candidate) && candidate > 0.0F) {
          step_length = candidate;
        }
      }
      state->step_length = step_length;
    }
    __syncthreads();
    current = next;
  }
}

// The threadgroup width of contact_gaps_step.
constexpr std::uint32_t contact_step_block = 256;

__device__ float projected_multiplier(const float* multipliers, const float* gaps,
                                      float step_length, std::uint32_t row) {
  return fmaxf(multipliers[row] - step_length * gaps[row], 0.0F);
}

// The response of every cell to the multipliers one projected step from
// `multipliers`, computing each projection where it is read.
__global__ void contact_respond_step(const ContactRowGpu* rows, const ContactBodyGpu* bodies,
                                     const float* multipliers, const float* gaps,
                                     const std::uint32_t* incidence_offsets,
                                     const std::uint32_t* incidence_entries,
                                     ContactResponseGpu* response,
                                     const ContactSolveStateGpu* state, ContactSolveGpu solve) {
  const std::uint32_t body = blockIdx.x * blockDim.x + threadIdx.x;
  if (body >= solve.body_count || state->done != 0U) {
    return;
  }
  const float length = state->step_length;
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
    const float multiplier = projected_multiplier(multipliers, gaps, length, row);
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
  const float4 drift_linear = bodies[body].drift_linear;
  const float4 drift_angular = bodies[body].drift_angular;
  response[body].linear_length = make_float4(drift_linear.x + force_x * translational,
                                             drift_linear.y + force_y * translational,
                                             drift_linear.z + force_z * translational, rate);
  response[body].rotation =
      make_float4(drift_angular.x + torque_x * transverse + axis.x * along,
                  drift_angular.y + torque_y * transverse + axis.y * along,
                  drift_angular.z + torque_z * transverse + axis.z * along, load);
}

// Takes the projected step into `next_multipliers` and `next_gaps`, and sums
// each block's step terms into `partials`.
__global__ void contact_gaps_step(const ContactRowGpu* rows, const ContactResponseGpu* response,
                                  const float* multipliers, const float* gaps,
                                  float* next_multipliers, float* next_gaps, float4* partials,
                                  const ContactSolveStateGpu* state, ContactSolveGpu solve) {
  __shared__ float4 partial[contact_step_block];
  if (state->done != 0U) {
    return;
  }
  const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  float4 terms = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
  if (row < solve.row_count) {
    const float multiplier = projected_multiplier(multipliers, gaps, state->step_length, row);
    const float gap = row_gap(rows, response, multiplier, solve, row);
    next_multipliers[row] = multiplier;
    next_gaps[row] = gap;
    const float step = multiplier - multipliers[row];
    const float change = gap - gaps[row];
    terms = make_float4(step * step, step * change, change * change, row_residual(multiplier, gap));
  }
  partial[threadIdx.x] = terms;
  __syncthreads();
  for (std::uint32_t stride = contact_step_block / 2U; stride > 0U; stride /= 2U) {
    if (threadIdx.x < stride) {
      const float4 other = partial[threadIdx.x + stride];
      const float4 value = partial[threadIdx.x];
      partial[threadIdx.x] = make_float4(value.x + other.x, value.y + other.y, value.z + other.z,
                                         fmaxf(value.w, other.w));
    }
    __syncthreads();
  }
  if (threadIdx.x == 0U) {
    partials[blockIdx.x] = partial[0];
  }
}

// Sums the block partials of contact_gaps_step in a fixed order and records
// the step: counts it, finishes the solve when its residual is within the
// tolerance or not finite, and otherwise sets the next Barzilai-Borwein step
// length.
__global__ void __launch_bounds__(contact_solve_block)
    contact_advance_step(const float4* partials, std::uint32_t block_count,
                         ContactSolveStateGpu* state, float tolerance) {
  __shared__ float4 partial[contact_solve_block];
  if (state->done != 0U) {
    return;
  }
  float4 local = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
  for (std::uint32_t block = threadIdx.x; block < block_count; block += blockDim.x) {
    const float4 value = partials[block];
    local = make_float4(local.x + value.x, local.y + value.y, local.z + value.z,
                        fmaxf(local.w, value.w));
  }
  partial[threadIdx.x] = local;
  __syncthreads();
  reduce_block(partial);
  if (threadIdx.x != 0U) {
    return;
  }
  const float4 sums = partial[0];
  state->iterations += 1U;
  state->residual = sums.w;
  if (!(sums.w > tolerance) || isinf(sums.w)) {
    state->done = 1U;
    return;
  }
  if (sums.y > 0.0F && sums.z > 0.0F) {
    const float next = (state->iterations & 1U) != 0U ? sums.x / sums.y : sums.y / sums.z;
    if (isfinite(next) && next > 0.0F) {
      state->step_length = next;
    }
  }
}

std::uint32_t block_count(std::uint32_t count) {
  return ((count - 1) / threads_per_block) + 1;
}

}  // namespace

void launch_contact_respond(const ContactRowGpu* rows, const ContactBodyGpu* bodies,
                            const float* multipliers, const std::uint32_t* incidence_offsets,
                            const std::uint32_t* incidence_entries, ContactResponseGpu* response,
                            ContactSolveGpu solve, cudaStream_t stream) {
  contact_respond<<<block_count(solve.body_count), threads_per_block, 0, stream>>>(
      rows, bodies, multipliers, incidence_offsets, incidence_entries, response, solve);
}

void launch_contact_gaps(const ContactRowGpu* rows, const ContactResponseGpu* response,
                         const float* multipliers, float* gaps, const float* previous_multipliers,
                         const float* previous_gaps, float4* terms, ContactSolveGpu solve,
                         bool initial, cudaStream_t stream) {
  contact_gaps<<<block_count(solve.row_count), threads_per_block, 0, stream>>>(
      rows, response, multipliers, gaps, previous_multipliers, previous_gaps, terms, solve,
      initial);
}

void launch_contact_solve(const ContactRowGpu* rows, const ContactBodyGpu* bodies,
                          float* multipliers_a, float* multipliers_b, float* gaps_a, float* gaps_b,
                          const std::uint32_t* incidence_offsets,
                          const std::uint32_t* incidence_entries, ContactResponseGpu* response,
                          ContactSolveStateGpu* state, ContactSolveGpu solve, float tolerance,
                          std::uint32_t max_iterations, float first_step_length,
                          cudaStream_t stream) {
  contact_solve<<<1, contact_solve_block, 0, stream>>>(
      rows, bodies, multipliers_a, multipliers_b, gaps_a, gaps_b, incidence_offsets,
      incidence_entries, response, state, solve, tolerance, max_iterations, first_step_length);
}

void launch_contact_steps(std::uint32_t count, std::uint32_t current, const ContactRowGpu* rows,
                          const ContactBodyGpu* bodies, float* const multipliers[2],
                          float* const gaps[2], const std::uint32_t* incidence_offsets,
                          const std::uint32_t* incidence_entries, ContactResponseGpu* response,
                          float4* partials, ContactSolveStateGpu* state, ContactSolveGpu solve,
                          float tolerance, cudaStream_t stream) {
  const auto row_blocks = (solve.row_count + contact_step_block - 1) / contact_step_block;
  for (std::uint32_t index = 0; index < count; ++index) {
    const auto next = 1U - current;
    contact_respond_step<<<block_count(solve.body_count), threads_per_block, 0, stream>>>(
        rows, bodies, multipliers[current], gaps[current], incidence_offsets, incidence_entries,
        response, state, solve);
    contact_gaps_step<<<row_blocks, contact_step_block, 0, stream>>>(
        rows, response, multipliers[current], gaps[current], multipliers[next], gaps[next],
        partials, state, solve);
    contact_advance_step<<<1, contact_solve_block, 0, stream>>>(partials, row_blocks, state,
                                                                tolerance);
    current = next;
  }
}

void launch_contact_reduce(const float4* input, float4* output, std::uint32_t element_count,
                           cudaStream_t stream) {
  const auto output_count = (element_count + 1) / 2;
  contact_reduce<<<block_count(output_count), threads_per_block, 0, stream>>>(input, output,
                                                                              element_count);
}

}  // namespace microsimulator::cuda
