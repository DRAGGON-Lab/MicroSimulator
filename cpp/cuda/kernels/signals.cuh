#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

#include "core/shared_constants.hpp"

namespace microsimulator::cuda {

struct SignalGridShapeGpu {
  std::uint32_t x;
  std::uint32_t y;
  std::uint32_t z;
  std::uint32_t sites;
};

struct SignalGridBoundariesGpu {
  std::uint32_t x_lower;
  std::uint32_t x_upper;
  std::uint32_t y_lower;
  std::uint32_t y_upper;
  std::uint32_t z_lower;
  std::uint32_t z_upper;
  // Where each face's own block of fixed values begins, in GridFace order;
  // zero for a face that carries the same values everywhere.
  std::uint32_t offsets[6];
  // The grid's advection scheme, as SignalAdvectionKind numbers it.
  std::uint32_t advection_scheme;
};

// Grid geometry and transport helpers shared by the signal and coupled-rate
// kernels. Whole-program compilation gives each translation unit its own copy,
// so a device-inline definition in this header needs no device linking.

__device__ inline std::uint32_t site_index(SignalGridShapeGpu shape, std::uint32_t x,
                                           std::uint32_t y, std::uint32_t z) {
  return x * shape.y * shape.z + y * shape.z + z;
}

// The fluid sites of a grid: the indices of the sites that are not obstacles,
// in ascending order. Transport threads number the fluid sites of each signal
// in turn, so obstacle sites cost nothing and keep the zero they hold.
struct SignalFluidGpu {
  const std::uint32_t* sites;
  std::uint32_t count;
};

// The level a transport thread works on.
__device__ inline std::uint32_t fluid_level(SignalFluidGpu fluid, std::uint32_t grid_sites,
                                            std::uint32_t entry) {
  const auto signal = entry / fluid.count;
  return signal * grid_sites + fluid.sites[entry - signal * fluid.count];
}

__device__ inline float grid_level(const float* levels, SignalGridShapeGpu shape,
                                   std::uint32_t signal, std::uint32_t x, std::uint32_t y,
                                   std::uint32_t z) {
  return levels[signal * shape.sites + site_index(shape, x, y, z)];
}

// The exterior stencil value beyond a boundary site. `plane` indexes the site
// on its face, with the first in-plane axis varying slowest.
__device__ inline float exterior_value(std::uint32_t kind, const float* fixed_values,
                                       std::uint32_t face, std::uint32_t offset,
                                       std::uint32_t plane, std::uint32_t signal,
                                       std::uint32_t signal_count, float current, float periodic) {
  if (kind == 0) {
    return current;
  }
  if (kind == 1) {
    return periodic;
  }
  if (offset == 0) {
    return fixed_values[face * signal_count + signal];
  }
  return fixed_values[offset + plane * signal_count + signal];
}

// What the van Leer scheme adds to the upwind site's concentration at a face:
// half the limited slope, 0.5 * psi(r) * (downwind - upwind) with
// r = (upwind - far) / (downwind - upwind) and psi(r) = (r + |r|) / (1 + |r|).
// It is zero at an extremum, where the two differences disagree in sign.
__device__ inline float van_leer_increment(float upwind, float downwind, float far) {
  const auto behind = upwind - far;
  const auto ahead = downwind - upwind;
  return behind * ahead > 0.0F ? behind * ahead / (behind + ahead) : 0.0F;
}

// The limited corrections to the advective fluxes through a site's two faces
// on one axis, as the upper face's less the lower face's. A face takes a
// correction only when its upwind site, its downwind site and the site two
// upwind are all fluid sites of the grid joined by open faces; otherwise it
// keeps the upwind flux. `lower` and `upper` are the site's neighbors on the
// axis as transport reads them, and `periodic_kind` is the axis's lower
// boundary kind.
__device__ inline float van_leer_correction(const float* levels, const std::uint8_t* obstacles,
                                            SignalGridShapeGpu shape, std::uint32_t periodic_kind,
                                            std::uint32_t signal, std::uint32_t x, std::uint32_t y,
                                            std::uint32_t z, std::uint32_t axis, float current,
                                            float lower, float upper, bool closed_lower,
                                            bool closed_upper, float lower_velocity,
                                            float upper_velocity) {
  const std::uint32_t dimensions[3]{shape.x, shape.y, shape.z};
  const std::uint32_t coordinate[3]{x, y, z};
  const auto count = dimensions[axis];
  const auto along = coordinate[axis];
  const bool periodic = periodic_kind == 1 && count >= 3;
  const bool lower_site = !closed_lower && (along > 0 || periodic);
  const bool upper_site = !closed_upper && (along + 1 < count || periodic);
  float lower_correction = 0.0F;
  float upper_correction = 0.0F;
  if (lower_site) {
    if (lower_velocity >= 0.0F) {
      if (along >= 2 || periodic) {
        // The site two below, where it is fluid in the grid.
        std::uint32_t site[3]{x, y, z};
        site[axis] = (along + count - 2) % count;
        if (obstacles[site_index(shape, site[0], site[1], site[2])] == 0) {
          lower_correction =
              lower_velocity *
              van_leer_increment(lower, current,
                                 grid_level(levels, shape, signal, site[0], site[1], site[2]));
        }
      }
    } else if (upper_site) {
      lower_correction = lower_velocity * van_leer_increment(current, lower, upper);
    }
  }
  if (upper_site) {
    if (upper_velocity >= 0.0F) {
      if (lower_site) {
        upper_correction = upper_velocity * van_leer_increment(current, upper, lower);
      }
    } else if (along + 2 < count || periodic) {
      std::uint32_t site[3]{x, y, z};
      site[axis] = (along + 2) % count;
      if (obstacles[site_index(shape, site[0], site[1], site[2])] == 0) {
        upper_correction =
            upper_velocity *
            van_leer_increment(upper, current,
                               grid_level(levels, shape, signal, site[0], site[1], site[2]));
      }
    }
  }
  return upper_correction - lower_correction;
}

// Whether each of a site's six faces is closed to transport, and the velocity
// it carries. A face is closed by a no-flux boundary, by a periodic boundary
// that wraps onto a solid site, or by a solid neighbour.
struct GridFaceState {
  bool closed_lower[3];
  bool closed_upper[3];
  float lower[3];
  float upper[3];
};

__device__ inline GridFaceState grid_face_state(SignalGridShapeGpu shape,
                                                SignalGridBoundariesGpu boundaries,
                                                const std::uint8_t* obstacles, const float* x_faces,
                                                const float* y_faces, const float* z_faces,
                                                std::uint32_t has_velocity_field, float4 advection,
                                                std::uint32_t x, std::uint32_t y, std::uint32_t z) {
  GridFaceState faces{};
  faces.closed_lower[0] =
      x == 0 ? (boundaries.x_lower == 0 ||
                (boundaries.x_lower == 1 && obstacles[site_index(shape, shape.x - 1, y, z)] != 0))
             : obstacles[site_index(shape, x - 1, y, z)] != 0;
  faces.closed_upper[0] =
      x + 1 == shape.x ? (boundaries.x_upper == 0 ||
                          (boundaries.x_upper == 1 && obstacles[site_index(shape, 0, y, z)] != 0))
                       : obstacles[site_index(shape, x + 1, y, z)] != 0;
  faces.closed_lower[1] =
      y == 0 ? (boundaries.y_lower == 0 ||
                (boundaries.y_lower == 1 && obstacles[site_index(shape, x, shape.y - 1, z)] != 0))
             : obstacles[site_index(shape, x, y - 1, z)] != 0;
  faces.closed_upper[1] =
      y + 1 == shape.y ? (boundaries.y_upper == 0 ||
                          (boundaries.y_upper == 1 && obstacles[site_index(shape, x, 0, z)] != 0))
                       : obstacles[site_index(shape, x, y + 1, z)] != 0;
  faces.closed_lower[2] =
      z == 0 ? (boundaries.z_lower == 0 ||
                (boundaries.z_lower == 1 && obstacles[site_index(shape, x, y, shape.z - 1)] != 0))
             : obstacles[site_index(shape, x, y, z - 1)] != 0;
  faces.closed_upper[2] =
      z + 1 == shape.z ? (boundaries.z_upper == 0 ||
                          (boundaries.z_upper == 1 && obstacles[site_index(shape, x, y, 0)] != 0))
                       : obstacles[site_index(shape, x, y, z + 1)] != 0;
  if (has_velocity_field != 0) {
    faces.lower[0] = x_faces[x * shape.y * shape.z + y * shape.z + z];
    faces.upper[0] = x_faces[(x + 1) * shape.y * shape.z + y * shape.z + z];
    faces.lower[1] = y_faces[x * (shape.y + 1) * shape.z + y * shape.z + z];
    faces.upper[1] = y_faces[x * (shape.y + 1) * shape.z + (y + 1) * shape.z + z];
    faces.lower[2] = z_faces[x * shape.y * (shape.z + 1) + y * (shape.z + 1) + z];
    faces.upper[2] = z_faces[x * shape.y * (shape.z + 1) + y * (shape.z + 1) + z + 1];
  } else {
    const float velocity[3]{advection.x, advection.y, advection.z};
    for (std::uint32_t axis = 0; axis < 3; ++axis) {
      faces.lower[axis] = velocity[axis];
      faces.upper[axis] = velocity[axis];
    }
  }
  return faces;
}

void launch_advance_signal_grid(
    const float* levels, float* output, const float* diffusion, const float4* advection,
    const float* fixed_values, const float* reaction_source, const float* reaction_loss,
    const std::uint8_t* obstacles, const float* x_faces, const float* y_faces, const float* z_faces,
    std::uint32_t has_velocity_field, std::uint32_t* error, SignalGridBoundariesGpu boundaries,
    SignalGridShapeGpu shape, float4 spacing, float dt, std::uint32_t signal_count,
    SignalFluidGpu fluid, std::uint32_t crank_nicolson, cudaStream_t stream);

// Writes the squares of a field's fluid levels, one term per fluid level.
void launch_signal_square_terms(const float* input, float* terms, std::uint32_t signal_count,
                                std::uint32_t grid_sites, SignalFluidGpu fluid,
                                cudaStream_t stream);
void launch_signal_crank_nicolson_jacobi(
    const float* current, float* output, const float* right_hand_side, const float* diffusion,
    const float4* advection, const float* fixed_values, const float* reaction_source,
    const float* reaction_loss, const std::uint8_t* obstacles, const float* x_faces,
    const float* y_faces, const float* z_faces, std::uint32_t has_velocity_field,
    std::uint32_t* error, SignalGridBoundariesGpu boundaries, SignalGridShapeGpu shape,
    float4 spacing, float half_dt, std::uint32_t signal_count, SignalFluidGpu fluid,
    cudaStream_t stream, const std::uint32_t* gate = nullptr);
// Writes one squared residual per fluid level.
void launch_signal_crank_nicolson_residual_terms(
    const float* current, const float* right_hand_side, float* terms, const float* diffusion,
    const float4* advection, const float* fixed_values, const float* reaction_source,
    const float* reaction_loss, const std::uint8_t* obstacles, const float* x_faces,
    const float* y_faces, const float* z_faces, std::uint32_t has_velocity_field,
    SignalGridBoundariesGpu boundaries, SignalGridShapeGpu shape, float4 spacing, float half_dt,
    std::uint32_t signal_count, SignalFluidGpu fluid, cudaStream_t stream,
    const std::uint32_t* gate = nullptr);

// Sums neighboring elements of `input` into `output`, halving the count.
void launch_reduce_sum_pairs(const float* input, float* output, std::uint32_t element_count,
                             cudaStream_t stream, const std::uint32_t* gate = nullptr);

// Progress of a batch of implicit signal sweeps on the device. Sweeps after
// the one that finishes the solve do nothing.
struct SignalSweepStateGpu {
  std::uint32_t done;
  std::uint32_t iterations;
  float residual_rms;
  std::uint32_t invalid;
};

// Records the sweep whose residual terms `reduction` sums, and finishes the
// batch when its residual RMS is within `threshold`, is not finite, or the
// sweep was invalid. A Jacobi kernel, residual kernel, or reduction given
// `&state->done` as its gate does nothing once the batch is finished.
void launch_signal_sweep_check(const float* reduction, const std::uint32_t* error,
                               SignalSweepStateGpu* state, float threshold,
                               std::uint32_t level_count, std::uint32_t fluid_level_count,
                               cudaStream_t stream);

}  // namespace microsimulator::cuda
