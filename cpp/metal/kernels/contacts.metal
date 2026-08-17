#include <metal_stdlib>

using namespace metal;

constant float inverse_sqrt_two = 0.7071067811865475f;
constant float float_epsilon = 1.1920928955078125e-7f;

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
  bool active[2];
  uint active_count;
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
      float denominator = first_length_squared * second_length_squared -
                          cross_projection * cross_projection;
      float parallel_tolerance =
          float_epsilon * first_length_squared * second_length_squared;
      if (denominator > parallel_tolerance) {
        first_parameter = clamp(
            (cross_projection * second_projection -
             first_projection * second_length_squared) /
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

ExternalEvaluation evaluate_external_constraint(const Capsule cell,
                                                const ExternalConstraint constraint,
                                                float2 contact_parameters) {
  ExternalEvaluation result;
  float half_length = cell.length * 0.5f;
  result.centerline_points[0] = cell.center - cell.axis * half_length;
  result.centerline_points[1] = cell.center + cell.axis * half_length;
  result.active_count = 0;
  for (uint endpoint = 0; endpoint < 2; ++endpoint) {
    float3 point = result.centerline_points[endpoint];
    if (constraint.kind == 0) {
      float3 inward_normal = constraint.parameters.xyz;
      result.separations[endpoint] =
          dot(point - constraint.geometry.xyz, inward_normal) - cell.radius;
      result.normals[endpoint] = -inward_normal;
    } else if (constraint.kind == 1) {
      float3 center_delta = point - constraint.geometry.xyz;
      float distance = length(center_delta);
      float3 radial =
          distance > contact_parameters.y ? center_delta / distance : float3(1.0f, 0.0f, 0.0f);
      if (constraint.allowed_region == 0) {
        result.separations[endpoint] = distance - constraint.geometry.w - cell.radius;
        result.normals[endpoint] = -radial;
      } else {
        result.separations[endpoint] = constraint.geometry.w - distance - cell.radius;
        result.normals[endpoint] = radial;
      }
    } else {
      float3 half_extents = constraint.parameters.xyz;
      float3 delta = point - constraint.geometry.xyz;
      float3 outside_vector = delta - clamp(delta, -half_extents, half_extents);
      float outside_distance = length(outside_vector);
      float signed_distance;
      float3 outward;
      if (outside_distance > contact_parameters.y) {
        signed_distance = outside_distance;
        outward = outside_vector / outside_distance;
      } else {
        float3 clearances = half_extents - abs(delta);
        if (clearances.x <= clearances.y && clearances.x <= clearances.z) {
          signed_distance = -clearances.x;
          outward = float3(delta.x >= 0.0f ? 1.0f : -1.0f, 0.0f, 0.0f);
        } else if (clearances.y <= clearances.z) {
          signed_distance = -clearances.y;
          outward = float3(0.0f, delta.y >= 0.0f ? 1.0f : -1.0f, 0.0f);
        } else {
          signed_distance = -clearances.z;
          outward = float3(0.0f, 0.0f, delta.z >= 0.0f ? 1.0f : -1.0f);
        }
      }
      if (constraint.allowed_region == 0) {
        result.separations[endpoint] = signed_distance - cell.radius;
        result.normals[endpoint] = -outward;
      } else {
        result.separations[endpoint] = -signed_distance - cell.radius;
        result.normals[endpoint] = outward;
      }
    }
    result.active[endpoint] = result.separations[endpoint] < contact_parameters.x;
    result.active_count += result.active[endpoint];
  }
  return result;
}

kernel void count_cell_contacts(device const ulong* ids [[buffer(0)]],
                                device const float4* centers [[buffer(1)]],
                                device const float4* axes [[buffer(2)]],
                                device const float4* geometry [[buffer(3)]],
                                device uint* counts [[buffer(4)]],
                                constant float4& parameters [[buffer(5)]],
                                constant uint& candidate_count [[buffer(6)]],
                                device const uint2* candidates [[buffer(7)]],
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
    constant uint& candidate_count [[buffer(16)]],
    device const uint2* candidates [[buffer(17)]],
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
    points_on_first[output_index] = float4(points.values[ordinal].first + normal * first.radius, 0.0f);
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
    device uint* endpoints [[buffer(11)]], device float4* points_on_cell [[buffer(12)]],
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
  for (uint endpoint = 0; endpoint < 2; ++endpoint) {
    if (!evaluation.active[endpoint]) {
      continue;
    }
    cell_ids[output_index] = cell.id;
    constraint_ids[output_index] = constraint.id;
    cell_slots[output_index] = cell.slot;
    constraint_kinds[output_index] = constraint.kind;
    endpoints[output_index] = endpoint;
    points_on_cell[output_index] = float4(
        evaluation.centerline_points[endpoint] + evaluation.normals[endpoint] * cell.radius, 0.0f);
    normals[output_index] = float4(evaluation.normals[endpoint], 0.0f);
    separations[output_index] = evaluation.separations[endpoint];
    weights[output_index] = weight;
    ++output_index;
  }
}
