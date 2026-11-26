#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

#include "core/shared_constants.hpp"

namespace microsimulator::cuda {

struct alignas(16) FlowGridParameters {
  std::uint32_t dimensions[4];
  float spacing[4];
  float inverse_spacing_squared[4];
  std::uint32_t face_offsets[4];
  std::uint32_t face_counts[4];
  std::uint32_t site_count;
  std::uint32_t total_face_count;
  std::uint32_t padding0;
  std::uint32_t padding1;
};

static_assert(sizeof(FlowGridParameters) == 96);

// Block partials one thread adds when a dot product is folded.
inline constexpr std::uint32_t depth_fold_width = 64;

// Where the depth-averaged kernels keep a solve's scalars and counters, so
// that a batch of iterations runs without the host.
enum DepthScalar : std::uint32_t {
  depth_rho = 0,
  depth_curvature = 1,
  depth_residual_squared = 2,
  depth_next_rho = 3,
  depth_alpha = 4,
  depth_beta = 5,
  depth_scalar_count = 6,
};
enum DepthCounter : std::uint32_t {
  depth_iterations = 0,
  depth_stopped = 1,
  depth_converged = 2,
  depth_failure = 3,
  depth_counter_count = 4,
};

// The inverse square spacing along each axis.
struct DepthFlowScales {
  double inverse_square[3];
};

// The operator over the active sites of a depth-averaged system. Each has six
// neighbors, lower then upper along x, y and z, named by their place among
// the active sites, with the conductance to each.
void launch_depth_flow_operator_double(const double* input, const std::uint32_t* neighbors,
                                       const float* conductance, const float* boundary_conductance,
                                       double* output, DepthFlowScales scales, std::uint32_t count,
                                       cudaStream_t stream);
// Writes the pressures of the active sites to their sites of the grid.
void launch_depth_flow_scatter(const double* active, const std::uint32_t* sites, double* pressure,
                               std::uint32_t count, cudaStream_t stream);
void launch_depth_flow_velocity_double(const double* pressure, const float* mobility,
                                       const float* face_conductance, const float* face_pressure,
                                       float* velocity, const FlowGridParameters& grid,
                                       cudaStream_t stream);
void launch_depth_flow_pcg_initialize(const double* right_hand_side, const float* diagonal,
                                      double* solution, double* residual, double* preconditioned,
                                      double* direction, std::uint32_t count, cudaStream_t stream);
// solution += alpha direction and residual -= alpha transformed, with alpha
// read from the solve's scalars.
void launch_depth_flow_pcg_update(double* solution, double* residual, const double* direction,
                                  const double* transformed, const double* scalars,
                                  std::uint32_t count, cudaStream_t stream);
void launch_depth_flow_pcg_precondition(const double* residual, const float* diagonal,
                                        double* preconditioned, std::uint32_t count,
                                        cudaStream_t stream);
// direction = preconditioned + beta direction, with beta read from the solve's
// scalars.
void launch_depth_flow_pcg_direction(const double* preconditioned, double* direction,
                                     const double* scalars, std::uint32_t count,
                                     cudaStream_t stream);
// Adds the block partials of a dot product, in groups and then together, and
// keeps the sum among the solve's scalars at `place`. `folded` holds one value
// for every `depth_fold_width` partials.
void launch_depth_flow_sum(const double* partials, double* folded, double* scalars,
                           std::uint32_t partial_count, std::uint32_t place, cudaStream_t stream);
// Begins an iteration: the step length from the curvature. A solve that has
// stopped takes a step of no length, which changes nothing.
void launch_depth_flow_step_length(double* scalars, std::uint32_t* counters, cudaStream_t stream);
// Ends an iteration: stops the solve if its residual meets the threshold, and
// otherwise leaves the weight of the next direction.
void launch_depth_flow_step_weight(double* scalars, std::uint32_t* counters, double threshold,
                                   cudaStream_t stream);
void launch_depth_flow_dot_partial(const double* left, const double* right, double* partials,
                                   std::uint32_t count, cudaStream_t stream);
void launch_resolved_flow_momentum(const float* input, const std::uint8_t* active,
                                   const std::uint8_t* exists, const float* face_drag,
                                   float* output, const FlowGridParameters& grid,
                                   cudaStream_t stream);
void launch_resolved_flow_gradient(const float* pressure, const std::uint8_t* fluid,
                                   const std::uint8_t* active, float* gradient,
                                   const FlowGridParameters& grid, cudaStream_t stream);
void launch_resolved_flow_divergence(const float* velocity, const std::uint8_t* fluid,
                                     float* divergence, const FlowGridParameters& grid,
                                     cudaStream_t stream);
void launch_flow_pcg_initialize(const float* right_hand_side, const float* diagonal,
                                float* solution, float* residual, float* preconditioned,
                                float* direction, std::uint32_t count, cudaStream_t stream);
void launch_flow_pcg_update(float* solution, float* residual, const float* direction,
                            const float* transformed, float alpha, std::uint32_t count,
                            cudaStream_t stream);
void launch_flow_pcg_precondition(const float* residual, const float* diagonal,
                                  float* preconditioned, std::uint32_t count, cudaStream_t stream);
void launch_flow_pcg_direction(const float* preconditioned, float* direction, float beta,
                               std::uint32_t count, cudaStream_t stream);
void launch_flow_vector_negate(const float* input, float* output, std::uint32_t count,
                               cudaStream_t stream);
void launch_flow_vector_combine(const float* source, float* target, float alpha, float beta,
                                std::uint32_t count, cudaStream_t stream);
void launch_flow_vector_subtract(const float* left, const float* right, float* output,
                                 std::uint32_t count, cudaStream_t stream);
void launch_flow_dot_partial(const float* left, const float* right, float* partials,
                             std::uint32_t count, cudaStream_t stream);

using shared::flow_reduction_width;

}  // namespace microsimulator::cuda
