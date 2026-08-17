#include <cstdint>

#include "contacts.cuh"

namespace cm::cuda {
namespace {

constexpr float inverse_sqrt_two = 0.7071067811865475F;
constexpr float float_epsilon = 1.1920928955078125e-7F;

struct Capsule {
  std::uint64_t id;
  std::uint32_t slot;
  float3 center;
  float3 axis;
  float length;
  float radius;
};

struct PointPair {
  float3 first;
  float3 second;
};

struct PairPoints {
  PointPair values[2];
  std::uint32_t count;
};

struct ExternalEvaluation {
  float3 centerline_points[2];
  float3 normals[2];
  float separations[2];
  bool active[2];
  std::uint32_t active_count;
};

__device__ float3 add(float3 left, float3 right) {
  return make_float3(left.x + right.x, left.y + right.y, left.z + right.z);
}

__device__ float3 subtract(float3 left, float3 right) {
  return make_float3(left.x - right.x, left.y - right.y, left.z - right.z);
}

__device__ float3 multiply(float3 value, float scale) {
  return make_float3(value.x * scale, value.y * scale, value.z * scale);
}

__device__ float dot_product(float3 left, float3 right) {
  return left.x * right.x + left.y * right.y + left.z * right.z;
}

__device__ float3 cross_product(float3 left, float3 right) {
  return make_float3(left.y * right.z - left.z * right.y, left.z * right.x - left.x * right.z,
                     left.x * right.y - left.y * right.x);
}

__device__ float magnitude(float3 value) { return sqrtf(dot_product(value, value)); }

__device__ float3 normalized_vector(float3 value) {
  return multiply(value, 1.0F / magnitude(value));
}

__device__ float clamp_value(float value, float minimum, float maximum) {
  return fminf(fmaxf(value, minimum), maximum);
}

__device__ Capsule load_capsule(const std::uint64_t* ids, const float4* centers, const float4* axes,
                                const float4* geometry, std::uint32_t slot) {
  const auto center = centers[slot];
  const auto axis = axes[slot];
  return {
      ids[slot],
      slot,
      make_float3(center.x, center.y, center.z),
      make_float3(axis.x, axis.y, axis.z),
      geometry[slot].x,
      geometry[slot].y,
  };
}

__device__ void canonicalize(Capsule& first, Capsule& second) {
  if (second.id < first.id) {
    const auto temporary = first;
    first = second;
    second = temporary;
  }
}

__device__ PointPair closest_points(const Capsule& first, const Capsule& second, float epsilon) {
  const auto first_half = first.length * 0.5F;
  const auto second_half = second.length * 0.5F;
  const auto first_start = subtract(first.center, multiply(first.axis, first_half));
  const auto first_end = add(first.center, multiply(first.axis, first_half));
  const auto second_start = subtract(second.center, multiply(second.axis, second_half));
  const auto second_end = add(second.center, multiply(second.axis, second_half));
  const auto first_delta = subtract(first_end, first_start);
  const auto second_delta = subtract(second_end, second_start);
  const auto between_starts = subtract(first_start, second_start);
  const auto first_length_squared = dot_product(first_delta, first_delta);
  const auto second_length_squared = dot_product(second_delta, second_delta);
  const auto second_projection = dot_product(second_delta, between_starts);
  const auto epsilon_squared = epsilon * epsilon;

  float first_parameter = 0.0F;
  float second_parameter = 0.0F;
  if (first_length_squared <= epsilon_squared && second_length_squared <= epsilon_squared) {
    return {first_start, second_start};
  }
  if (first_length_squared <= epsilon_squared) {
    second_parameter = clamp_value(second_projection / second_length_squared, 0.0F, 1.0F);
  } else {
    const auto first_projection = dot_product(first_delta, between_starts);
    if (second_length_squared <= epsilon_squared) {
      first_parameter = clamp_value(-first_projection / first_length_squared, 0.0F, 1.0F);
    } else {
      const auto cross_projection = dot_product(first_delta, second_delta);
      const auto denominator =
          first_length_squared * second_length_squared - cross_projection * cross_projection;
      const auto parallel_tolerance = float_epsilon * first_length_squared * second_length_squared;
      if (denominator > parallel_tolerance) {
        first_parameter = clamp_value(
            (cross_projection * second_projection - first_projection * second_length_squared) /
                denominator,
            0.0F, 1.0F);
      }
      second_parameter =
          (cross_projection * first_parameter + second_projection) / second_length_squared;
      if (second_parameter < 0.0F) {
        second_parameter = 0.0F;
        first_parameter = clamp_value(-first_projection / first_length_squared, 0.0F, 1.0F);
      } else if (second_parameter > 1.0F) {
        second_parameter = 1.0F;
        first_parameter =
            clamp_value((cross_projection - first_projection) / first_length_squared, 0.0F, 1.0F);
      }
    }
  }

  return {
      add(first_start, multiply(first_delta, first_parameter)),
      add(second_start, multiply(second_delta, second_parameter)),
  };
}

__device__ PairPoints contact_points(const Capsule& first, const Capsule& second,
                                     ContactParametersGpu parameters) {
  const auto axis_dot = clamp_value(dot_product(first.axis, second.axis), -1.0F, 1.0F);
  const auto sine = sqrtf(fmaxf(0.0F, 1.0F - axis_dot * axis_dot));
  PairPoints result{};
  if (sine > parameters.parallel_sine_threshold || first.length <= parameters.degeneracy_epsilon ||
      second.length <= parameters.degeneracy_epsilon) {
    result.values[0] = closest_points(first, second, parameters.degeneracy_epsilon);
    result.count = 1;
    return result;
  }

  const auto first_half = first.length * 0.5F;
  const auto second_half = second.length * 0.5F;
  const auto center_coordinate = dot_product(subtract(second.center, first.center), first.axis);
  const auto projected_second_half = second_half * fabsf(axis_dot);
  const auto overlap_begin = fmaxf(-first_half, center_coordinate - projected_second_half);
  const auto overlap_end = fminf(first_half, center_coordinate + projected_second_half);
  if (overlap_end - overlap_begin <= parameters.degeneracy_epsilon) {
    result.values[0] = closest_points(first, second, parameters.degeneracy_epsilon);
    result.count = 1;
    return result;
  }

  const float first_parameters[2] = {overlap_begin, overlap_end};
  for (std::uint32_t index = 0; index < 2; ++index) {
    const auto point_on_first = add(first.center, multiply(first.axis, first_parameters[index]));
    const auto second_parameter =
        clamp_value(dot_product(subtract(point_on_first, second.center), second.axis), -second_half,
                    second_half);
    result.values[index] = {
        point_on_first,
        add(second.center, multiply(second.axis, second_parameter)),
    };
  }
  result.count = 2;
  return result;
}

__device__ float3 deterministic_normal(const Capsule& first, const Capsule& second,
                                       const PointPair& points, float epsilon) {
  const auto point_delta = subtract(points.second, points.first);
  if (magnitude(point_delta) > epsilon) {
    return normalized_vector(point_delta);
  }

  const auto axes_cross = cross_product(first.axis, second.axis);
  if (magnitude(axes_cross) > epsilon) {
    return normalized_vector(axes_cross);
  }

  const auto center_delta = subtract(second.center, first.center);
  const auto transverse_center_delta =
      subtract(center_delta, multiply(first.axis, dot_product(center_delta, first.axis)));
  if (magnitude(transverse_center_delta) > epsilon) {
    return normalized_vector(transverse_center_delta);
  }

  const auto absolute_axis =
      make_float3(fabsf(first.axis.x), fabsf(first.axis.y), fabsf(first.axis.z));
  float3 basis{};
  if (absolute_axis.x <= absolute_axis.y && absolute_axis.x <= absolute_axis.z) {
    basis = make_float3(1.0F, 0.0F, 0.0F);
  } else if (absolute_axis.y <= absolute_axis.z) {
    basis = make_float3(0.0F, 1.0F, 0.0F);
  } else {
    basis = make_float3(0.0F, 0.0F, 1.0F);
  }
  return normalized_vector(cross_product(first.axis, basis));
}

__device__ ExternalEvaluation
evaluate_external_constraint(const Capsule& cell, const ExternalConstraintGpu& constraint,
                             ConstraintContactParametersGpu contact_parameters) {
  ExternalEvaluation result{};
  const auto half_length = cell.length * 0.5F;
  result.centerline_points[0] = subtract(cell.center, multiply(cell.axis, half_length));
  result.centerline_points[1] = add(cell.center, multiply(cell.axis, half_length));
  for (std::uint32_t endpoint = 0; endpoint < 2; ++endpoint) {
    const auto point = result.centerline_points[endpoint];
    if (constraint.kind == 0) {
      const auto inward_normal =
          make_float3(constraint.parameters.x, constraint.parameters.y, constraint.parameters.z);
      const auto plane_point =
          make_float3(constraint.geometry.x, constraint.geometry.y, constraint.geometry.z);
      result.separations[endpoint] =
          dot_product(subtract(point, plane_point), inward_normal) - cell.radius;
      result.normals[endpoint] = multiply(inward_normal, -1.0F);
    } else if (constraint.kind == 1) {
      const auto sphere_center =
          make_float3(constraint.geometry.x, constraint.geometry.y, constraint.geometry.z);
      const auto center_delta = subtract(point, sphere_center);
      const auto distance = magnitude(center_delta);
      const auto radial = distance > contact_parameters.degeneracy_epsilon
                              ? multiply(center_delta, 1.0F / distance)
                              : make_float3(1.0F, 0.0F, 0.0F);
      if (constraint.allowed_region == 0) {
        result.separations[endpoint] = distance - constraint.geometry.w - cell.radius;
        result.normals[endpoint] = multiply(radial, -1.0F);
      } else {
        result.separations[endpoint] = constraint.geometry.w - distance - cell.radius;
        result.normals[endpoint] = radial;
      }
    } else if (constraint.kind == 2) {
      const auto box_center =
          make_float3(constraint.geometry.x, constraint.geometry.y, constraint.geometry.z);
      const auto half_extents =
          make_float3(constraint.parameters.x, constraint.parameters.y, constraint.parameters.z);
      const auto delta = subtract(point, box_center);
      const auto outside_vector =
          make_float3(delta.x - fminf(fmaxf(delta.x, -half_extents.x), half_extents.x),
                      delta.y - fminf(fmaxf(delta.y, -half_extents.y), half_extents.y),
                      delta.z - fminf(fmaxf(delta.z, -half_extents.z), half_extents.z));
      const auto outside_distance = magnitude(outside_vector);
      float signed_distance{};
      float3 outward{};
      if (outside_distance > contact_parameters.degeneracy_epsilon) {
        signed_distance = outside_distance;
        outward = multiply(outside_vector, 1.0F / outside_distance);
      } else {
        const auto clearances = make_float3(half_extents.x - fabsf(delta.x),
                                            half_extents.y - fabsf(delta.y),
                                            half_extents.z - fabsf(delta.z));
        if (clearances.x <= clearances.y && clearances.x <= clearances.z) {
          signed_distance = -clearances.x;
          outward = make_float3(delta.x >= 0.0F ? 1.0F : -1.0F, 0.0F, 0.0F);
        } else if (clearances.y <= clearances.z) {
          signed_distance = -clearances.y;
          outward = make_float3(0.0F, delta.y >= 0.0F ? 1.0F : -1.0F, 0.0F);
        } else {
          signed_distance = -clearances.z;
          outward = make_float3(0.0F, 0.0F, delta.z >= 0.0F ? 1.0F : -1.0F);
        }
      }
      if (constraint.allowed_region == 0) {
        result.separations[endpoint] = signed_distance - cell.radius;
        result.normals[endpoint] = multiply(outward, -1.0F);
      } else {
        result.separations[endpoint] = -signed_distance - cell.radius;
        result.normals[endpoint] = outward;
      }
    } else {
      const auto delta =
          make_float3(point.x - constraint.geometry.x, point.y - constraint.geometry.y, 0.0F);
      const auto radial_distance = magnitude(delta);
      const auto radial = radial_distance > contact_parameters.degeneracy_epsilon
                              ? multiply(delta, 1.0F / radial_distance)
                              : make_float3(1.0F, 0.0F, 0.0F);
      const auto z_offset = point.z - constraint.geometry.z;
      const auto axial = make_float3(0.0F, 0.0F, z_offset >= 0.0F ? 1.0F : -1.0F);
      const auto radial_excess = radial_distance - constraint.geometry.w;
      const auto axial_excess = fabsf(z_offset) - constraint.parameters.x;
      float signed_distance{};
      float3 outward{};
      if (radial_excess > 0.0F && axial_excess > 0.0F) {
        signed_distance =
            sqrtf((radial_excess * radial_excess) + (axial_excess * axial_excess));
        outward = multiply(add(multiply(radial, radial_excess), multiply(axial, axial_excess)),
                           1.0F / signed_distance);
      } else if (radial_excess > 0.0F) {
        signed_distance = radial_excess;
        outward = radial;
      } else if (axial_excess > 0.0F) {
        signed_distance = axial_excess;
        outward = axial;
      } else if (-radial_excess <= -axial_excess) {
        signed_distance = radial_excess;
        outward = radial;
      } else {
        signed_distance = axial_excess;
        outward = axial;
      }
      if (constraint.allowed_region == 0) {
        result.separations[endpoint] = signed_distance - cell.radius;
        result.normals[endpoint] = multiply(outward, -1.0F);
      } else {
        result.separations[endpoint] = -signed_distance - cell.radius;
        result.normals[endpoint] = outward;
      }
    }
    result.active[endpoint] = result.separations[endpoint] < contact_parameters.activation_margin;
    result.active_count += result.active[endpoint] ? 1U : 0U;
  }
  return result;
}

__global__ void count_cell_contacts(const std::uint64_t* ids, const float4* centers,
                                    const float4* axes, const float4* geometry,
                                    const uint2* candidates, std::uint32_t* counts,
                                    ContactParametersGpu parameters,
                                    std::uint32_t candidate_count) {
  const auto pair_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair_index >= candidate_count) {
    return;
  }
  const auto first_slot = candidates[pair_index].x;
  const auto second_slot = candidates[pair_index].y;

  auto first = load_capsule(ids, centers, axes, geometry, first_slot);
  auto second = load_capsule(ids, centers, axes, geometry, second_slot);
  canonicalize(first, second);
  const auto points = contact_points(first, second, parameters);
  std::uint32_t active_count = 0;
  for (std::uint32_t ordinal = 0; ordinal < points.count; ++ordinal) {
    const auto separation =
        magnitude(subtract(points.values[ordinal].second, points.values[ordinal].first)) -
        (first.radius + second.radius);
    active_count += separation < parameters.activation_margin ? 1U : 0U;
  }
  counts[pair_index] = active_count;
}

__global__ void inclusive_scan_step(const std::uint32_t* input, std::uint32_t* output,
                                    std::uint32_t offset, std::uint32_t element_count) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }
  auto value = input[index];
  if (index >= offset) {
    value += input[index - offset];
  }
  output[index] = value;
}

__global__ void fill_cell_contacts(
    const std::uint64_t* ids, const float4* centers, const float4* axes, const float4* geometry,
    const uint2* candidates, const std::uint32_t* counts,
    const std::uint32_t* inclusive_counts, std::uint64_t* first_ids,
    std::uint64_t* second_ids, std::uint32_t* first_slots, std::uint32_t* second_slots,
    std::uint32_t* ordinals, float4* points_on_first, float4* normals, float* separations,
    float* weights, ContactParametersGpu parameters, std::uint32_t candidate_count) {
  const auto pair_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair_index >= candidate_count) {
    return;
  }
  const auto first_slot = candidates[pair_index].x;
  const auto second_slot = candidates[pair_index].y;
  const auto pair_contact_count = counts[pair_index];
  if (pair_contact_count == 0) {
    return;
  }

  auto first = load_capsule(ids, centers, axes, geometry, first_slot);
  auto second = load_capsule(ids, centers, axes, geometry, second_slot);
  canonicalize(first, second);
  const auto points = contact_points(first, second, parameters);
  const auto weight = points.count == 2 ? inverse_sqrt_two : 1.0F;
  auto output_index = inclusive_counts[pair_index] - pair_contact_count;
  for (std::uint32_t ordinal = 0; ordinal < points.count; ++ordinal) {
    const auto point_delta = subtract(points.values[ordinal].second, points.values[ordinal].first);
    const auto separation = magnitude(point_delta) - (first.radius + second.radius);
    if (separation >= parameters.activation_margin) {
      continue;
    }
    const auto normal =
        deterministic_normal(first, second, points.values[ordinal], parameters.degeneracy_epsilon);
    const auto surface_point = add(points.values[ordinal].first, multiply(normal, first.radius));
    first_ids[output_index] = first.id;
    second_ids[output_index] = second.id;
    first_slots[output_index] = first.slot;
    second_slots[output_index] = second.slot;
    ordinals[output_index] = ordinal;
    points_on_first[output_index] =
        make_float4(surface_point.x, surface_point.y, surface_point.z, 0.0F);
    normals[output_index] = make_float4(normal.x, normal.y, normal.z, 0.0F);
    separations[output_index] = separation;
    weights[output_index] = weight;
    ++output_index;
  }
}

__global__ void count_external_contacts(const std::uint64_t* ids, const float4* centers,
                                        const float4* axes, const float4* geometry,
                                        const ExternalConstraintGpu* constraints,
                                        std::uint32_t* counts,
                                        ConstraintContactParametersGpu parameters,
                                        std::uint32_t cell_count, std::uint32_t constraint_count) {
  const auto constraint_index = blockIdx.x * blockDim.x + threadIdx.x;
  const auto cell_slot = blockIdx.y * blockDim.y + threadIdx.y;
  if (cell_slot >= cell_count || constraint_index >= constraint_count) {
    return;
  }
  const auto pair_index = cell_slot * constraint_count + constraint_index;
  const auto cell = load_capsule(ids, centers, axes, geometry, cell_slot);
  counts[pair_index] =
      evaluate_external_constraint(cell, constraints[constraint_index], parameters).active_count;
}

__global__ void fill_external_contacts(
    const std::uint64_t* ids, const float4* centers, const float4* axes, const float4* geometry,
    const ExternalConstraintGpu* constraints, const std::uint32_t* counts,
    const std::uint32_t* inclusive_counts, std::uint64_t* cell_ids, std::uint64_t* constraint_ids,
    std::uint32_t* cell_slots, std::uint32_t* constraint_kinds, std::uint32_t* endpoints,
    float4* points_on_cell, float4* normals, float* separations, float* weights,
    ConstraintContactParametersGpu parameters, std::uint32_t cell_count,
    std::uint32_t constraint_count) {
  const auto constraint_index = blockIdx.x * blockDim.x + threadIdx.x;
  const auto cell_slot = blockIdx.y * blockDim.y + threadIdx.y;
  if (cell_slot >= cell_count || constraint_index >= constraint_count) {
    return;
  }
  const auto pair_index = cell_slot * constraint_count + constraint_index;
  const auto pair_contact_count = counts[pair_index];
  if (pair_contact_count == 0) {
    return;
  }

  const auto cell = load_capsule(ids, centers, axes, geometry, cell_slot);
  const auto constraint = constraints[constraint_index];
  const auto evaluation = evaluate_external_constraint(cell, constraint, parameters);
  const auto weight =
      constraint.parameters.w * (evaluation.active_count == 2 ? inverse_sqrt_two : 1.0F);
  auto output_index = inclusive_counts[pair_index] - pair_contact_count;
  for (std::uint32_t endpoint = 0; endpoint < 2; ++endpoint) {
    if (!evaluation.active[endpoint]) {
      continue;
    }
    const auto point = add(evaluation.centerline_points[endpoint],
                           multiply(evaluation.normals[endpoint], cell.radius));
    const auto normal = evaluation.normals[endpoint];
    cell_ids[output_index] = cell.id;
    constraint_ids[output_index] = constraint.id;
    cell_slots[output_index] = cell.slot;
    constraint_kinds[output_index] = constraint.kind;
    endpoints[output_index] = endpoint;
    points_on_cell[output_index] = make_float4(point.x, point.y, point.z, 0.0F);
    normals[output_index] = make_float4(normal.x, normal.y, normal.z, 0.0F);
    separations[output_index] = evaluation.separations[endpoint];
    weights[output_index] = weight;
    ++output_index;
  }
}

}  // namespace

void launch_contact_count(const std::uint64_t* ids, const float4* centers, const float4* axes,
                          const float4* geometry, const uint2* candidates, std::uint32_t* counts,
                          ContactParametersGpu parameters, std::uint32_t candidate_count,
                          cudaStream_t stream) {
  constexpr std::uint32_t threads = 256;
  const auto blocks = ((candidate_count - 1) / threads) + 1;
  count_cell_contacts<<<blocks, threads, 0, stream>>>(ids, centers, axes, geometry, candidates,
                                                      counts, parameters, candidate_count);
}

void launch_inclusive_scan_step(const std::uint32_t* input, std::uint32_t* output,
                                std::uint32_t offset, std::uint32_t element_count,
                                cudaStream_t stream) {
  constexpr std::uint32_t threads = 256;
  const auto blocks = ((element_count - 1) / threads) + 1;
  inclusive_scan_step<<<blocks, threads, 0, stream>>>(input, output, offset, element_count);
}

void launch_contact_fill(const std::uint64_t* ids, const float4* centers, const float4* axes,
                         const float4* geometry, const uint2* candidates,
                         const std::uint32_t* counts,
                         const std::uint32_t* inclusive_counts, std::uint64_t* first_ids,
                         std::uint64_t* second_ids, std::uint32_t* first_slots,
                         std::uint32_t* second_slots, std::uint32_t* ordinals,
                         float4* points_on_first, float4* normals, float* separations,
                         float* weights, ContactParametersGpu parameters,
                         std::uint32_t candidate_count, cudaStream_t stream) {
  constexpr std::uint32_t threads = 256;
  const auto blocks = ((candidate_count - 1) / threads) + 1;
  fill_cell_contacts<<<blocks, threads, 0, stream>>>(
      ids, centers, axes, geometry, candidates, counts, inclusive_counts, first_ids, second_ids,
      first_slots, second_slots, ordinals, points_on_first, normals, separations, weights,
      parameters, candidate_count);
}

void launch_external_contact_count(const std::uint64_t* ids, const float4* centers,
                                   const float4* axes, const float4* geometry,
                                   const ExternalConstraintGpu* constraints, std::uint32_t* counts,
                                   ConstraintContactParametersGpu parameters,
                                   std::uint32_t cell_count, std::uint32_t constraint_count,
                                   cudaStream_t stream) {
  constexpr dim3 threads(16, 16);
  const dim3 blocks((constraint_count + threads.x - 1) / threads.x,
                    (cell_count + threads.y - 1) / threads.y);
  count_external_contacts<<<blocks, threads, 0, stream>>>(
      ids, centers, axes, geometry, constraints, counts, parameters, cell_count, constraint_count);
}

void launch_external_contact_fill(
    const std::uint64_t* ids, const float4* centers, const float4* axes, const float4* geometry,
    const ExternalConstraintGpu* constraints, const std::uint32_t* counts,
    const std::uint32_t* inclusive_counts, std::uint64_t* cell_ids, std::uint64_t* constraint_ids,
    std::uint32_t* cell_slots, std::uint32_t* constraint_kinds, std::uint32_t* endpoints,
    float4* points_on_cell, float4* normals, float* separations, float* weights,
    ConstraintContactParametersGpu parameters, std::uint32_t cell_count,
    std::uint32_t constraint_count, cudaStream_t stream) {
  constexpr dim3 threads(16, 16);
  const dim3 blocks((constraint_count + threads.x - 1) / threads.x,
                    (cell_count + threads.y - 1) / threads.y);
  fill_external_contacts<<<blocks, threads, 0, stream>>>(
      ids, centers, axes, geometry, constraints, counts, inclusive_counts, cell_ids, constraint_ids,
      cell_slots, constraint_kinds, endpoints, points_on_cell, normals, separations, weights,
      parameters, cell_count, constraint_count);
}

}  // namespace cm::cuda
