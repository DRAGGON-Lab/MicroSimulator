#include <metal_stdlib>

using namespace metal;

constant float inverse_sqrt_two = 0.7071067811865475f;
constant float float_epsilon = 1.1920928955078125e-7f;
constant uint segment_minimization_iterations = 40;
constant uint interior_location = 2;

struct Capsule {
  ulong id;
  uint slot;
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
  uint count;
};

struct ExternalConstraint {
  ulong id;
  uint kind;
  uint allowed_region;
  float4 geometry;
  float4 parameters;
};

struct ExternalEvaluation {
  float3 centerline_points[2];
  float3 normals[2];
  float separations[2];
  uint locations[2];
  uint active_count;
};

struct SurfacePoint {
  float signed_distance;
  float3 outward;
};

struct CenterlineMinimum {
  float3 point;
  SurfacePoint surface;
};

Capsule load_capsule(device const ulong* ids, device const float4* centers,
                     device const float4* axes, device const float4* geometry, uint slot) {
  Capsule result;
  result.id = ids[slot];
  result.slot = slot;
  result.center = centers[slot].xyz;
  result.axis = axes[slot].xyz;
  result.length = geometry[slot].x;
  result.radius = geometry[slot].y;
  return result;
}

void canonicalize(thread Capsule& first, thread Capsule& second) {
  if (second.id < first.id) {
    Capsule temporary = first;
    first = second;
    second = temporary;
  }
}

PointPair closest_points(const Capsule first, const Capsule second, float epsilon) {
  float first_half = first.length * 0.5f;
  float second_half = second.length * 0.5f;
  float3 first_start = first.center - first.axis * first_half;
  float3 first_end = first.center + first.axis * first_half;
  float3 second_start = second.center - second.axis * second_half;
  float3 second_end = second.center + second.axis * second_half;
  float3 first_delta = first_end - first_start;
  float3 second_delta = second_end - second_start;
  float3 between_starts = first_start - second_start;
  float first_length_squared = dot(first_delta, first_delta);
  float second_length_squared = dot(second_delta, second_delta);
  float second_projection = dot(second_delta, between_starts);
  float epsilon_squared = epsilon * epsilon;

  float first_parameter = 0.0f;
  float second_parameter = 0.0f;
  if (first_length_squared <= epsilon_squared && second_length_squared <= epsilon_squared) {
    return {first_start, second_start};
  }
  if (first_length_squared <= epsilon_squared) {
    second_parameter = clamp(second_projection / second_length_squared, 0.0f, 1.0f);
  } else {
    float first_projection = dot(first_delta, between_starts);
    if (second_length_squared <= epsilon_squared) {
      first_parameter = clamp(-first_projection / first_length_squared, 0.0f, 1.0f);
    } else {
      float cross_projection = dot(first_delta, second_delta);
      float denominator =
          first_length_squared * second_length_squared - cross_projection * cross_projection;
      float parallel_tolerance = float_epsilon * first_length_squared * second_length_squared;
      if (denominator > parallel_tolerance) {
        first_parameter = clamp(
            (cross_projection * second_projection - first_projection * second_length_squared) /
                denominator,
            0.0f, 1.0f);
      }
      second_parameter =
          (cross_projection * first_parameter + second_projection) / second_length_squared;
      if (second_parameter < 0.0f) {
        second_parameter = 0.0f;
        first_parameter = clamp(-first_projection / first_length_squared, 0.0f, 1.0f);
      } else if (second_parameter > 1.0f) {
        second_parameter = 1.0f;
        first_parameter =
            clamp((cross_projection - first_projection) / first_length_squared, 0.0f, 1.0f);
      }
    }
  }

  return {
      first_start + first_delta * first_parameter,
      second_start + second_delta * second_parameter,
  };
}

PairPoints contact_points(const Capsule first, const Capsule second, float4 parameters) {
  float epsilon = parameters.z;
  float axis_dot = clamp(dot(first.axis, second.axis), -1.0f, 1.0f);
  float sine = sqrt(max(0.0f, 1.0f - axis_dot * axis_dot));
  PairPoints result;
  if (sine > parameters.y || first.length <= epsilon || second.length <= epsilon) {
    result.values[0] = closest_points(first, second, epsilon);
    result.count = 1;
    return result;
  }

  float first_half = first.length * 0.5f;
  float second_half = second.length * 0.5f;
  float center_coordinate = dot(second.center - first.center, first.axis);
  float projected_second_half = second_half * abs(axis_dot);
  float overlap_begin = max(-first_half, center_coordinate - projected_second_half);
  float overlap_end = min(first_half, center_coordinate + projected_second_half);
  if (overlap_end - overlap_begin <= epsilon) {
    result.values[0] = closest_points(first, second, epsilon);
    result.count = 1;
    return result;
  }

  float first_parameters[2] = {overlap_begin, overlap_end};
  for (uint index = 0; index < 2; ++index) {
    float3 point_on_first = first.center + first.axis * first_parameters[index];
    float second_parameter =
        clamp(dot(point_on_first - second.center, second.axis), -second_half, second_half);
    result.values[index] = {point_on_first, second.center + second.axis * second_parameter};
  }
  result.count = 2;
  return result;
}

float3 deterministic_normal(const Capsule first, const Capsule second, const PointPair points,
                            float epsilon) {
  float3 point_delta = points.second - points.first;
  if (length(point_delta) > epsilon) {
    return normalize(point_delta);
  }

  float3 axes_cross = cross(first.axis, second.axis);
  if (length(axes_cross) > epsilon) {
    return normalize(axes_cross);
  }

  float3 center_delta = second.center - first.center;
  float3 transverse_center_delta = center_delta - first.axis * dot(center_delta, first.axis);
  if (length(transverse_center_delta) > epsilon) {
    return normalize(transverse_center_delta);
  }

  float3 absolute_axis = abs(first.axis);
  float3 basis;
  if (absolute_axis.x <= absolute_axis.y && absolute_axis.x <= absolute_axis.z) {
    basis = float3(1.0f, 0.0f, 0.0f);
  } else if (absolute_axis.y <= absolute_axis.z) {
    basis = float3(0.0f, 1.0f, 0.0f);
  } else {
    basis = float3(0.0f, 0.0f, 1.0f);
  }
  return normalize(cross(first.axis, basis));
}

SurfacePoint external_surface(float3 point, const ExternalConstraint constraint, float epsilon) {
  if (constraint.kind == 0) {
    float3 inward_normal = constraint.parameters.xyz;
    return {dot(point - constraint.geometry.xyz, inward_normal), inward_normal};
  }
  if (constraint.kind == 1) {
    float3 delta = point - constraint.geometry.xyz;
    float distance = length(delta);
    float3 outward = distance > epsilon ? delta / distance : float3(1.0f, 0.0f, 0.0f);
    return {distance - constraint.geometry.w, outward};
  }
  if (constraint.kind == 2) {
    float3 half_extents = constraint.parameters.xyz;
    float3 delta = point - constraint.geometry.xyz;
    float3 outside_vector = delta - clamp(delta, -half_extents, half_extents);
    float outside_distance = length(outside_vector);
    if (outside_distance > epsilon) {
      return {outside_distance, outside_vector / outside_distance};
    }
    float3 clearances = half_extents - abs(delta);
    if (clearances.x <= clearances.y && clearances.x <= clearances.z) {
      float sign = fabs(delta.x) <= epsilon || delta.x >= 0.0f ? 1.0f : -1.0f;
      return {-clearances.x, float3(sign, 0.0f, 0.0f)};
    }
    if (clearances.y <= clearances.z) {
      float sign = fabs(delta.y) <= epsilon || delta.y >= 0.0f ? 1.0f : -1.0f;
      return {-clearances.y, float3(0.0f, sign, 0.0f)};
    }
    float sign = fabs(delta.z) <= epsilon || delta.z >= 0.0f ? 1.0f : -1.0f;
    return {-clearances.z, float3(0.0f, 0.0f, sign)};
  }

  float3 delta = float3(point.xy - constraint.geometry.xy, 0.0f);
  float radial_distance = length(delta);
  float3 radial = radial_distance > epsilon ? delta / radial_distance : float3(1.0f, 0.0f, 0.0f);
  float z_offset = point.z - constraint.geometry.z;
  float3 axial = float3(0.0f, 0.0f, z_offset >= 0.0f ? 1.0f : -1.0f);
  float radial_excess = radial_distance - constraint.geometry.w;
  float axial_excess = fabs(z_offset) - constraint.parameters.x;
  if (radial_excess > 0.0f && axial_excess > 0.0f) {
    float distance = sqrt(radial_excess * radial_excess + axial_excess * axial_excess);
    return {distance, (radial * radial_excess + axial * axial_excess) / distance};
  }
  if (radial_excess > 0.0f) {
    return {radial_excess, radial};
  }
  if (axial_excess > 0.0f) {
    return {axial_excess, axial};
  }
  if (-radial_excess <= -axial_excess) {
    return {radial_excess, radial};
  }
  return {axial_excess, axial};
}

bool segment_intersects_bounds(float3 start, float3 end, float3 lower, float3 upper) {
  float3 delta = end - start;
  float starts[3] = {start.x, start.y, start.z};
  float deltas[3] = {delta.x, delta.y, delta.z};
  float lowers[3] = {lower.x, lower.y, lower.z};
  float uppers[3] = {upper.x, upper.y, upper.z};
  float entry = 0.0f;
  float exit = 1.0f;
  for (uint axis = 0; axis < 3; ++axis) {
    if (deltas[axis] == 0.0f) {
      if (starts[axis] < lowers[axis] || starts[axis] > uppers[axis]) {
        return false;
      }
      continue;
    }
    float first = (lowers[axis] - starts[axis]) / deltas[axis];
    float second = (uppers[axis] - starts[axis]) / deltas[axis];
    if (first > second) {
      float temporary = first;
      first = second;
      second = temporary;
    }
    entry = max(entry, first);
    exit = min(exit, second);
    if (entry > exit) {
      return false;
    }
  }
  return true;
}

CenterlineMinimum minimize_surface_on_segment(float3 start, float3 end,
                                              const ExternalConstraint constraint, float epsilon) {
  float3 delta = end - start;
  float lower = 0.0f;
  float upper = 1.0f;
  for (uint iteration = 0; iteration < segment_minimization_iterations; ++iteration) {
    float first_parameter = lower + (upper - lower) / 3.0f;
    float second_parameter = upper - (upper - lower) / 3.0f;
    SurfacePoint first = external_surface(start + delta * first_parameter, constraint, epsilon);
    SurfacePoint second = external_surface(start + delta * second_parameter, constraint, epsilon);
    if (first.signed_distance < second.signed_distance) {
      upper = second_parameter;
    } else if (second.signed_distance < first.signed_distance) {
      lower = first_parameter;
    } else {
      lower = first_parameter;
      upper = second_parameter;
    }
  }

  CenterlineMinimum result = {start, external_surface(start, constraint, epsilon)};
  float candidates[5] = {1.0f, 0.5f, lower, (lower + upper) * 0.5f, upper};
  for (uint index = 0; index < 5; ++index) {
    float3 point = start + delta * candidates[index];
    SurfacePoint surface = external_surface(point, constraint, epsilon);
    if (surface.signed_distance < result.surface.signed_distance) {
      result = {point, surface};
    }
  }
  if (constraint.kind == 3) {
    if (fabs(delta.z) > epsilon) {
      float parameter = clamp((constraint.geometry.z - start.z) / delta.z, 0.0f, 1.0f);
      float3 point = start + delta * parameter;
      SurfacePoint surface = external_surface(point, constraint, epsilon);
      if (surface.signed_distance <= result.surface.signed_distance) {
        result = {point, surface};
      }
    }
    float radial_length_squared = delta.x * delta.x + delta.y * delta.y;
    if (radial_length_squared > epsilon * epsilon) {
      float parameter = clamp(-((start.x - constraint.geometry.x) * delta.x +
                                (start.y - constraint.geometry.y) * delta.y) /
                                  radial_length_squared,
                              0.0f, 1.0f);
      float3 point = start + delta * parameter;
      SurfacePoint surface = external_surface(point, constraint, epsilon);
      if (surface.signed_distance <= result.surface.signed_distance) {
        result = {point, surface};
      }
    }
  }
  return result;
}

CenterlineMinimum sphere_minimum(float3 start, float3 end, const ExternalConstraint constraint,
                                 float epsilon) {
  float3 delta = end - start;
  float length_squared = dot(delta, delta);
  float parameter =
      length_squared > epsilon * epsilon
          ? clamp(-dot(start - constraint.geometry.xyz, delta) / length_squared, 0.0f, 1.0f)
          : 0.0f;
  float3 point = start + delta * parameter;
  return {point, external_surface(point, constraint, epsilon)};
}

void add_external_contact(thread ExternalEvaluation& result, uint location, float3 centerline_point,
                          const SurfacePoint surface, float radius, bool outside) {
  uint index = result.active_count++;
  result.centerline_points[index] = centerline_point;
  result.normals[index] = outside ? -surface.outward : surface.outward;
  result.separations[index] =
      (outside ? surface.signed_distance : -surface.signed_distance) - radius;
  result.locations[index] = location;
}

ExternalEvaluation evaluate_external_constraint(const Capsule cell,
                                                const ExternalConstraint constraint,
                                                float2 contact_parameters) {
  ExternalEvaluation result;
  result.active_count = 0;
  float half_length = cell.length * 0.5f;
  float3 endpoints[2] = {
      cell.center - cell.axis * half_length,
      cell.center + cell.axis * half_length,
  };
  SurfacePoint endpoint_surfaces[2] = {
      external_surface(endpoints[0], constraint, contact_parameters.y),
      external_surface(endpoints[1], constraint, contact_parameters.y),
  };

  bool finite_outside = constraint.kind != 0 && constraint.allowed_region == 0;
  if (finite_outside) {
    if (constraint.kind >= 2) {
      float reach = cell.radius + contact_parameters.x;
      float3 lower;
      float3 upper;
      if (constraint.kind == 2) {
        lower = constraint.geometry.xyz - constraint.parameters.xyz - reach;
        upper = constraint.geometry.xyz + constraint.parameters.xyz + reach;
      } else {
        float3 extents =
            float3(constraint.geometry.w, constraint.geometry.w, constraint.parameters.x);
        lower = constraint.geometry.xyz - extents - reach;
        upper = constraint.geometry.xyz + extents + reach;
      }
      if (!segment_intersects_bounds(endpoints[0], endpoints[1], lower, upper)) {
        return result;
      }
    }

    CenterlineMinimum minimum =
        constraint.kind == 1
            ? sphere_minimum(endpoints[0], endpoints[1], constraint, contact_parameters.y)
            : minimize_surface_on_segment(endpoints[0], endpoints[1], constraint,
                                          contact_parameters.y);
    if (minimum.surface.signed_distance - cell.radius >= contact_parameters.x) {
      return result;
    }
    for (uint endpoint = 0; endpoint < 2; ++endpoint) {
      float separation = endpoint_surfaces[endpoint].signed_distance - cell.radius;
      if (separation < contact_parameters.x &&
          fabs(endpoint_surfaces[endpoint].signed_distance - minimum.surface.signed_distance) <=
              contact_parameters.y) {
        add_external_contact(result, endpoint, endpoints[endpoint], endpoint_surfaces[endpoint],
                             cell.radius, true);
      }
    }
    if (result.active_count == 0) {
      add_external_contact(result, interior_location, minimum.point, minimum.surface, cell.radius,
                           true);
    }
    return result;
  }

  bool outside = constraint.allowed_region == 0;
  for (uint endpoint = 0; endpoint < 2; ++endpoint) {
    float separation = (outside ? endpoint_surfaces[endpoint].signed_distance
                                : -endpoint_surfaces[endpoint].signed_distance) -
                       cell.radius;
    if (separation < contact_parameters.x) {
      add_external_contact(result, endpoint, endpoints[endpoint], endpoint_surfaces[endpoint],
                           cell.radius, outside);
    }
  }
  return result;
}

kernel void count_cell_contacts(
    device const ulong* ids [[buffer(0)]], device const float4* centers [[buffer(1)]],
    device const float4* axes [[buffer(2)]], device const float4* geometry [[buffer(3)]],
    device uint* counts [[buffer(4)]], constant float4& parameters [[buffer(5)]],
    constant uint& candidate_count [[buffer(6)]], device const uint2* candidates [[buffer(7)]],
    uint pair_index [[thread_position_in_grid]]) {
  if (pair_index >= candidate_count) {
    return;
  }
  uint first_slot = candidates[pair_index].x;
  uint second_slot = candidates[pair_index].y;

  Capsule first = load_capsule(ids, centers, axes, geometry, first_slot);
  Capsule second = load_capsule(ids, centers, axes, geometry, second_slot);
  canonicalize(first, second);
  PairPoints points = contact_points(first, second, parameters);
  uint active_count = 0;
  for (uint ordinal = 0; ordinal < points.count; ++ordinal) {
    float separation = length(points.values[ordinal].second - points.values[ordinal].first) -
                       (first.radius + second.radius);
    active_count += separation < parameters.x;
  }
  counts[pair_index] = active_count;
}

kernel void inclusive_scan_step(device const uint* input [[buffer(0)]],
                                device uint* output [[buffer(1)]],
                                constant uint& offset [[buffer(2)]],
                                constant uint& element_count [[buffer(3)]],
                                uint index [[thread_position_in_grid]]) {
  if (index >= element_count) {
    return;
  }
  uint value = input[index];
  if (index >= offset) {
    value += input[index - offset];
  }
  output[index] = value;
}

kernel void fill_cell_contacts(
    device const ulong* ids [[buffer(0)]], device const float4* centers [[buffer(1)]],
    device const float4* axes [[buffer(2)]], device const float4* geometry [[buffer(3)]],
    device const uint* counts [[buffer(4)]], device const uint* inclusive_counts [[buffer(5)]],
    device ulong* first_ids [[buffer(6)]], device ulong* second_ids [[buffer(7)]],
    device uint* first_slots [[buffer(8)]], device uint* second_slots [[buffer(9)]],
    device uint* ordinals [[buffer(10)]], device float4* points_on_first [[buffer(11)]],
    device float4* normals [[buffer(12)]], device float* separations [[buffer(13)]],
    device float* weights [[buffer(14)]], constant float4& parameters [[buffer(15)]],
    constant uint& candidate_count [[buffer(16)]], device const uint2* candidates [[buffer(17)]],
    uint pair_index [[thread_position_in_grid]]) {
  if (pair_index >= candidate_count) {
    return;
  }
  uint first_slot = candidates[pair_index].x;
  uint second_slot = candidates[pair_index].y;
  uint pair_contact_count = counts[pair_index];
  if (pair_contact_count == 0) {
    return;
  }

  Capsule first = load_capsule(ids, centers, axes, geometry, first_slot);
  Capsule second = load_capsule(ids, centers, axes, geometry, second_slot);
  canonicalize(first, second);
  PairPoints points = contact_points(first, second, parameters);
  float weight = points.count == 2 ? inverse_sqrt_two : 1.0f;
  uint output_index = inclusive_counts[pair_index] - pair_contact_count;
  for (uint ordinal = 0; ordinal < points.count; ++ordinal) {
    float3 point_delta = points.values[ordinal].second - points.values[ordinal].first;
    float separation = length(point_delta) - (first.radius + second.radius);
    if (separation >= parameters.x) {
      continue;
    }
    float3 normal = deterministic_normal(first, second, points.values[ordinal], parameters.z);
    first_ids[output_index] = first.id;
    second_ids[output_index] = second.id;
    first_slots[output_index] = first.slot;
    second_slots[output_index] = second.slot;
    ordinals[output_index] = ordinal;
    points_on_first[output_index] =
        float4(points.values[ordinal].first + normal * first.radius, 0.0f);
    normals[output_index] = float4(normal, 0.0f);
    separations[output_index] = separation;
    weights[output_index] = weight;
    ++output_index;
  }
}

kernel void count_external_contacts(
    device const ulong* ids [[buffer(0)]], device const float4* centers [[buffer(1)]],
    device const float4* axes [[buffer(2)]], device const float4* geometry [[buffer(3)]],
    device const ExternalConstraint* constraints [[buffer(4)]], device uint* counts [[buffer(5)]],
    constant float2& parameters [[buffer(6)]], constant uint& cell_count [[buffer(7)]],
    constant uint& constraint_count [[buffer(8)]], uint2 position [[thread_position_in_grid]]) {
  if (position.x >= constraint_count || position.y >= cell_count) {
    return;
  }
  uint cell_slot = position.y;
  uint constraint_index = position.x;
  uint pair_index = cell_slot * constraint_count + constraint_index;
  Capsule cell = load_capsule(ids, centers, axes, geometry, cell_slot);
  ExternalEvaluation evaluation =
      evaluate_external_constraint(cell, constraints[constraint_index], parameters);
  counts[pair_index] = evaluation.active_count;
}

kernel void fill_external_contacts(
    device const ulong* ids [[buffer(0)]], device const float4* centers [[buffer(1)]],
    device const float4* axes [[buffer(2)]], device const float4* geometry [[buffer(3)]],
    device const ExternalConstraint* constraints [[buffer(4)]],
    device const uint* counts [[buffer(5)]], device const uint* inclusive_counts [[buffer(6)]],
    device ulong* cell_ids [[buffer(7)]], device ulong* constraint_ids [[buffer(8)]],
    device uint* cell_slots [[buffer(9)]], device uint* constraint_kinds [[buffer(10)]],
    device uint* locations [[buffer(11)]], device float4* points_on_cell [[buffer(12)]],
    device float4* normals [[buffer(13)]], device float* separations [[buffer(14)]],
    device float* weights [[buffer(15)]], constant float2& parameters [[buffer(16)]],
    constant uint& cell_count [[buffer(17)]], constant uint& constraint_count [[buffer(18)]],
    uint2 position [[thread_position_in_grid]]) {
  if (position.x >= constraint_count || position.y >= cell_count) {
    return;
  }
  uint cell_slot = position.y;
  uint constraint_index = position.x;
  uint pair_index = cell_slot * constraint_count + constraint_index;
  uint pair_contact_count = counts[pair_index];
  if (pair_contact_count == 0) {
    return;
  }

  Capsule cell = load_capsule(ids, centers, axes, geometry, cell_slot);
  ExternalConstraint constraint = constraints[constraint_index];
  ExternalEvaluation evaluation = evaluate_external_constraint(cell, constraint, parameters);
  float weight = constraint.parameters.w * (evaluation.active_count == 2 ? inverse_sqrt_two : 1.0f);
  uint output_index = inclusive_counts[pair_index] - pair_contact_count;
  for (uint contact = 0; contact < evaluation.active_count; ++contact) {
    cell_ids[output_index] = cell.id;
    constraint_ids[output_index] = constraint.id;
    cell_slots[output_index] = cell.slot;
    constraint_kinds[output_index] = constraint.kind;
    locations[output_index] = evaluation.locations[contact];
    points_on_cell[output_index] = float4(
        evaluation.centerline_points[contact] + evaluation.normals[contact] * cell.radius, 0.0f);
    normals[output_index] = float4(evaluation.normals[contact], 0.0f);
    separations[output_index] = evaluation.separations[contact];
    weights[output_index] = weight;
    ++output_index;
  }
}
