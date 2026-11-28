#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>

#include "core/device_types.hpp"
#include "core/shared_constants.hpp"
#include "cpu/cpu_vector.hpp"

namespace microsimulator::device {

using uchar = std::uint8_t;
using namespace shared;

// Contacts between rod cells and external constraints, one (cell,
// constraint) pair at a time.
//
// A pair index is cell * constraint_count + constraint, over the constraints
// in ascending id order. Detection runs in three passes:
//
//   1. external_pair_count (per pair): how many contacts the pair makes.
//   2. std::inclusive_scan (per pair): the inclusive prefix sum of
//      the counts, which gives each pair its range of records.
//   3. external_pair_fill (per pair): the pair writes its contacts in order.
//
// A cell is the capsule of radius r about the segment between its two
// endpoints. A constraint of kind plane holds cells on the side its inward
// normal points to; a sphere, box, or cylinder holds them inside or outside
// itself. Every endpoint within the activation margin of the surface makes a
// row. A cell outside a finite body instead touches it where its centerline
// comes nearest: at its active endpoints when the nearest point is an
// endpoint, and at that interior point otherwise. With `every_face`, a cell
// inside a box or cylinder makes one row for every face an endpoint is near,
// so a cell in a corner is held by each wall that meets there.

// The constraint kinds and regions, as ExternalConstraintKind and
// ConstraintRegion number them.
inline constexpr uint constraint_plane = 0u;
inline constexpr uint constraint_sphere = 1u;
inline constexpr uint constraint_box = 2u;
inline constexpr uint constraint_cylinder = 3u;
inline constexpr uint region_outside = 0u;
// RodContactLocation::interior; the endpoints are 0 (negative) and 1
// (positive).
inline constexpr uint location_interior = 2u;

struct ExternalSurfacePoint {
  float signed_distance;
  Vec3f outward;
};

inline ExternalSurfacePoint external_surface_point(float signed_distance, Vec3f outward) {
  ExternalSurfacePoint result;
  result.signed_distance = signed_distance;
  result.outward = outward;
  return result;
}

// The signed distance from `point` to the constraint's surface, positive
// outside the body (or on the inward side of a plane), and the outward
// normal of the surface there.
inline ExternalSurfacePoint external_surface(Vec3f point, ExternalConstraint constraint,
                                             float epsilon) {
  const Vec3f origin = Vec3f{constraint.geometry.x, constraint.geometry.y, constraint.geometry.z};
  if (constraint.kind == constraint_plane) {
    const Vec3f inward =
        Vec3f{constraint.parameters.x, constraint.parameters.y, constraint.parameters.z};
    return external_surface_point(dot((point - origin), inward), inward);
  }
  if (constraint.kind == constraint_sphere) {
    const Vec3f delta = (point - origin);
    const float distance = length(delta);
    const Vec3f outward =
        distance > epsilon ? (delta * (1.0f / distance)) : Vec3f{1.0f, 0.0f, 0.0f};
    return external_surface_point(distance - constraint.geometry.w, outward);
  }
  if (constraint.kind == constraint_box) {
    const float half_extents[3] = {constraint.parameters.x, constraint.parameters.y,
                                   constraint.parameters.z};
    const Vec3f delta = (point - origin);
    const float offsets[3] = {delta.x, delta.y, delta.z};
    const Vec3f outside = Vec3f{delta.x - std::clamp(delta.x, -half_extents[0], half_extents[0]),
                                delta.y - std::clamp(delta.y, -half_extents[1], half_extents[1]),
                                delta.z - std::clamp(delta.z, -half_extents[2], half_extents[2])};
    const float outside_distance = length(outside);
    if (outside_distance > epsilon) {
      return external_surface_point(outside_distance, (outside * (1.0f / outside_distance)));
    }
    uint nearest = 0u;
    float clearance = half_extents[0] - std::fabs(offsets[0]);
    for (uint axis = 1u; axis < 3u; ++axis) {
      const float candidate = half_extents[axis] - std::fabs(offsets[axis]);
      if (candidate < clearance) {
        clearance = candidate;
        nearest = axis;
      }
    }
    const float sign =
        std::fabs(offsets[nearest]) <= epsilon || offsets[nearest] >= 0.0f ? 1.0f : -1.0f;
    return external_surface_point(-clearance,
                                  Vec3f{nearest == 0u ? sign : 0.0f, nearest == 1u ? sign : 0.0f,
                                        nearest == 2u ? sign : 0.0f});
  }
  const Vec3f delta = Vec3f{point.x - origin.x, point.y - origin.y, 0.0f};
  const float radial_distance = length(delta);
  const Vec3f radial =
      radial_distance > epsilon ? (delta * (1.0f / radial_distance)) : Vec3f{1.0f, 0.0f, 0.0f};
  const float z_offset = point.z - origin.z;
  const Vec3f axial = Vec3f{0.0f, 0.0f, z_offset >= 0.0f ? 1.0f : -1.0f};
  const float radial_excess = radial_distance - constraint.geometry.w;
  const float axial_excess = std::fabs(z_offset) - constraint.parameters.x;
  if (radial_excess > 0.0f && axial_excess > 0.0f) {
    const float distance = std::sqrt(radial_excess * radial_excess + axial_excess * axial_excess);
    return external_surface_point(
        distance, (((radial * radial_excess) + (axial * axial_excess)) * (1.0f / distance)));
  }
  if (radial_excess > 0.0f) {
    return external_surface_point(radial_excess, radial);
  }
  if (axial_excess > 0.0f) {
    return external_surface_point(axial_excess, axial);
  }
  if (-radial_excess <= -axial_excess) {
    return external_surface_point(radial_excess, radial);
  }
  return external_surface_point(axial_excess, axial);
}

// Whether the segment from `start` to `end` meets the axis-aligned box from
// `lower` to `upper`.
inline bool external_segment_meets_bounds(Vec3f start, Vec3f end, Vec3f lower, Vec3f upper) {
  const float starts[3] = {start.x, start.y, start.z};
  const float deltas[3] = {end.x - start.x, end.y - start.y, end.z - start.z};
  const float lowers[3] = {lower.x, lower.y, lower.z};
  const float uppers[3] = {upper.x, upper.y, upper.z};
  float entry = 0.0f;
  float exit = 1.0f;
  for (uint axis = 0u; axis < 3u; ++axis) {
    if (deltas[axis] == 0.0f) {
      if (starts[axis] < lowers[axis] || starts[axis] > uppers[axis]) {
        return false;
      }
      continue;
    }
    float first = (lowers[axis] - starts[axis]) / deltas[axis];
    float second = (uppers[axis] - starts[axis]) / deltas[axis];
    if (first > second) {
      const float swapped = first;
      first = second;
      second = swapped;
    }
    entry = first > entry ? first : entry;
    exit = second < exit ? second : exit;
    if (entry > exit) {
      return false;
    }
  }
  return true;
}

struct ExternalMinimum {
  Vec3f point;
  ExternalSurfacePoint surface;
};

inline ExternalMinimum external_minimum(Vec3f point, ExternalSurfacePoint surface) {
  ExternalMinimum result;
  result.point = point;
  result.surface = surface;
  return result;
}

// The point of the segment nearest the surface of a sphere, box, or
// cylinder. A sphere's is found exactly; a box's and a cylinder's by ternary
// search, then by comparing candidates the search can miss.
inline ExternalMinimum external_segment_minimum(Vec3f start, Vec3f end,
                                                ExternalConstraint constraint, float epsilon) {
  const Vec3f delta = (end - start);
  if (constraint.kind == constraint_sphere) {
    const Vec3f center = Vec3f{constraint.geometry.x, constraint.geometry.y, constraint.geometry.z};
    const float length_squared = dot(delta, delta);
    const float parameter =
        length_squared > epsilon * epsilon
            ? std::clamp(-dot((start - center), delta) / length_squared, 0.0f, 1.0f)
            : 0.0f;
    const Vec3f point = (start + (delta * parameter));
    return external_minimum(point, external_surface(point, constraint, epsilon));
  }
  float lower = 0.0f;
  float upper = 1.0f;
  for (uint iteration = 0u; iteration < segment_minimization_iterations; ++iteration) {
    const float first_parameter = lower + (upper - lower) / 3.0f;
    const float second_parameter = upper - (upper - lower) / 3.0f;
    const ExternalSurfacePoint first =
        external_surface((start + (delta * first_parameter)), constraint, epsilon);
    const ExternalSurfacePoint second =
        external_surface((start + (delta * second_parameter)), constraint, epsilon);
    if (first.signed_distance < second.signed_distance) {
      upper = second_parameter;
    } else if (second.signed_distance < first.signed_distance) {
      lower = first_parameter;
    } else {
      lower = first_parameter;
      upper = second_parameter;
    }
  }
  ExternalMinimum result = external_minimum(start, external_surface(start, constraint, epsilon));
  const float candidates[5] = {1.0f, 0.5f, lower, (lower + upper) * 0.5f, upper};
  for (uint index = 0u; index < 5u; ++index) {
    const Vec3f point = (start + (delta * candidates[index]));
    const ExternalSurfacePoint surface = external_surface(point, constraint, epsilon);
    if (surface.signed_distance < result.surface.signed_distance) {
      result = external_minimum(point, surface);
    }
  }
  if (constraint.kind == constraint_cylinder) {
    // The cylinder's mid-plane and the point nearest its axis.
    float parameters[2] = {-1.0f, -1.0f};
    if (std::fabs(delta.z) > epsilon) {
      parameters[0] = (constraint.geometry.z - start.z) / delta.z;
    }
    const float radial_length_squared = delta.x * delta.x + delta.y * delta.y;
    if (radial_length_squared > epsilon * epsilon) {
      parameters[1] = -((start.x - constraint.geometry.x) * delta.x +
                        (start.y - constraint.geometry.y) * delta.y) /
                      radial_length_squared;
    }
    const bool present[2] = {std::fabs(delta.z) > epsilon,
                             radial_length_squared > epsilon * epsilon};
    for (uint index = 0u; index < 2u; ++index) {
      if (!present[index]) {
        continue;
      }
      const Vec3f point = (start + (delta * (std::clamp(parameters[index], 0.0f, 1.0f))));
      const ExternalSurfacePoint surface = external_surface(point, constraint, epsilon);
      if (surface.signed_distance <= result.surface.signed_distance) {
        result = external_minimum(point, surface);
      }
    }
  }
  return result;
}

// One contact: the centerline point it acts at, the contact normal pointing
// out of the constraint into the cell, its separation, and its location.
struct ExternalCandidate {
  Vec3f centerline_point;
  Vec3f normal;
  float separation;
  uint location;
};

// The contacts of one pair and the weight each carries.
struct ExternalEvaluation {
  ExternalCandidate contacts[external_pair_capacity];
  uint count;
  float weight;
};

inline void external_add(ExternalEvaluation& evaluation, uint location, Vec3f centerline_point,
                         Vec3f normal, float separation) {
  const uint index = evaluation.count++;
  evaluation.contacts[index].centerline_point = centerline_point;
  evaluation.contacts[index].normal = normal;
  evaluation.contacts[index].separation = separation;
  evaluation.contacts[index].location = location;
}

// Every face of a box or cylinder container that an endpoint is near, one
// row each, at the constraint's full weight.
inline void external_container_faces(ExternalEvaluation& evaluation, Vec3f negative, Vec3f positive,
                                     float radius, ExternalConstraint constraint, float margin,
                                     float epsilon) {
  const Vec3f endpoints[2] = {negative, positive};
  const Vec3f origin = Vec3f{constraint.geometry.x, constraint.geometry.y, constraint.geometry.z};
  for (uint endpoint = 0u; endpoint < 2u; ++endpoint) {
    const Vec3f delta = (endpoints[endpoint] - origin);
    if (constraint.kind == constraint_box) {
      const float half_extents[3] = {constraint.parameters.x, constraint.parameters.y,
                                     constraint.parameters.z};
      const float offsets[3] = {delta.x, delta.y, delta.z};
      for (uint axis = 0u; axis < 3u; ++axis) {
        for (uint side = 0u; side < 2u; ++side) {
          const float sign = side == 0u ? -1.0f : 1.0f;
          const float separation = half_extents[axis] - sign * offsets[axis] - radius;
          if (separation >= margin) {
            continue;
          }
          external_add(
              evaluation, endpoint, endpoints[endpoint],
              Vec3f{axis == 0u ? sign : 0.0f, axis == 1u ? sign : 0.0f, axis == 2u ? sign : 0.0f},
              separation);
        }
      }
      continue;
    }
    const Vec3f radial_delta = Vec3f{delta.x, delta.y, 0.0f};
    const float radial_distance = length(radial_delta);
    const Vec3f radial = radial_distance > epsilon ? (radial_delta * (1.0f / radial_distance))
                                                   : Vec3f{1.0f, 0.0f, 0.0f};
    const float barrel = constraint.geometry.w - radial_distance - radius;
    if (barrel < margin) {
      external_add(evaluation, endpoint, endpoints[endpoint], radial, barrel);
    }
    for (uint side = 0u; side < 2u; ++side) {
      const float sign = side == 0u ? -1.0f : 1.0f;
      const float separation = constraint.parameters.x - sign * delta.z - radius;
      if (separation < margin) {
        external_add(evaluation, endpoint, endpoints[endpoint], Vec3f{0.0f, 0.0f, sign},
                     separation);
      }
    }
  }
  evaluation.weight = constraint.parameters.w;
}

// The contacts between a cell, its center holding its length in w and its
// axis its radius in w, and a constraint.
inline ExternalEvaluation external_evaluate(float4 center, float4 axis,
                                            ExternalConstraint constraint,
                                            ExternalContactData data) {
  ExternalEvaluation evaluation;
  evaluation.count = 0u;
  evaluation.weight = 0.0f;
  const float margin = data.activation_margin;
  const float epsilon = data.degeneracy_epsilon;
  const float radius = axis.w;
  const Vec3f middle = Vec3f{center.x, center.y, center.z};
  const Vec3f direction = Vec3f{axis.x, axis.y, axis.z};
  const float half_length = center.w * 0.5f;
  const Vec3f endpoints[2] = {
      (middle - (direction * half_length)),
      (middle + (direction * half_length)),
  };
  const bool finite = constraint.kind != constraint_plane;
  const bool outside =
      constraint.kind == constraint_plane || constraint.allowed_region == region_outside;
  if (data.every_face != 0u && !outside &&
      (constraint.kind == constraint_box || constraint.kind == constraint_cylinder)) {
    external_container_faces(evaluation, endpoints[0], endpoints[1], radius, constraint, margin,
                             epsilon);
    return evaluation;
  }
  const ExternalSurfacePoint surfaces[2] = {
      external_surface(endpoints[0], constraint, epsilon),
      external_surface(endpoints[1], constraint, epsilon),
  };
  if (finite && outside) {
    if (constraint.kind != constraint_sphere) {
      const float reach = radius + margin;
      Vec3f extent =
          constraint.kind == constraint_box
              ? Vec3f{constraint.parameters.x, constraint.parameters.y, constraint.parameters.z}
              : Vec3f{constraint.geometry.w, constraint.geometry.w, constraint.parameters.x};
      extent = Vec3f{extent.x + reach, extent.y + reach, extent.z + reach};
      const Vec3f origin =
          Vec3f{constraint.geometry.x, constraint.geometry.y, constraint.geometry.z};
      if (!external_segment_meets_bounds(endpoints[0], endpoints[1], (origin - extent),
                                         (origin + extent))) {
        return evaluation;
      }
    }
    const ExternalMinimum minimum =
        external_segment_minimum(endpoints[0], endpoints[1], constraint, epsilon);
    if (minimum.surface.signed_distance - radius >= margin) {
      return evaluation;
    }
    // A minimum at an endpoint leaves the other endpoint free to be inside
    // the surface too. Each active endpoint then needs its own row: a single
    // row lets the solve answer with a rotation about the rod's center that
    // drives the unconstrained end deeper.
    const float nearest = surfaces[0].signed_distance < surfaces[1].signed_distance
                              ? surfaces[0].signed_distance
                              : surfaces[1].signed_distance;
    const bool at_endpoint = nearest - minimum.surface.signed_distance <= epsilon;
    for (uint endpoint = 0u; at_endpoint && endpoint < 2u; ++endpoint) {
      const float separation = surfaces[endpoint].signed_distance - radius;
      if (separation < margin) {
        external_add(evaluation, endpoint, endpoints[endpoint],
                     (surfaces[endpoint].outward * -1.0f), separation);
      }
    }
    if (evaluation.count == 0u) {
      external_add(evaluation, location_interior, minimum.point, (minimum.surface.outward * -1.0f),
                   minimum.surface.signed_distance - radius);
    }
  } else {
    for (uint endpoint = 0u; endpoint < 2u; ++endpoint) {
      const float separation =
          (outside ? surfaces[endpoint].signed_distance : -surfaces[endpoint].signed_distance) -
          radius;
      if (separation < margin) {
        external_add(evaluation, endpoint, endpoints[endpoint],
                     outside ? (surfaces[endpoint].outward * -1.0f) : surfaces[endpoint].outward,
                     separation);
      }
    }
  }
  evaluation.weight = constraint.parameters.w * (evaluation.count == 2u ? inverse_sqrt_two : 1.0f);
  return evaluation;
}

// Pass 1. The number of contacts pair `pair` makes.
inline uint external_pair_count(uint pair, ExternalContactData data, const float4* centers,
                                const float4* axes, const ExternalConstraint* constraints) {
  const uint cell = pair / data.constraint_count;
  const uint constraint = pair % data.constraint_count;
  return external_evaluate(centers[cell], axes[cell], constraints[constraint], data).count;
}

// Pass 3. Writes the contacts of pair `pair` from the start of its range.
// `ends` is the inclusive prefix sum of the counts.
inline void external_pair_fill(uint pair, ExternalContactData data, const float4* centers,
                               const float4* axes, const ExternalConstraint* constraints,
                               const uint* ends, ExternalContactRecord* records) {
  const uint cell = pair / data.constraint_count;
  const uint constraint = pair % data.constraint_count;
  const float4 axis = axes[cell];
  const ExternalEvaluation evaluation =
      external_evaluate(centers[cell], axis, constraints[constraint], data);
  uint output = pair == 0u ? 0u : ends[pair - 1u];
  for (uint index = 0u; index < evaluation.count; ++index) {
    const ExternalCandidate contact = evaluation.contacts[index];
    const Vec3f point = (contact.centerline_point + (contact.normal * axis.w));
    ExternalContactRecord* record = records + output;
    record->point.x = point.x;
    record->point.y = point.y;
    record->point.z = point.z;
    record->point.w = contact.separation;
    record->normal.x = contact.normal.x;
    record->normal.y = contact.normal.y;
    record->normal.z = contact.normal.z;
    record->normal.w = evaluation.weight;
    record->cell = cell;
    record->constraint = constraint;
    record->location = contact.location;
    record->padding = 0u;
    ++output;
  }
}

}  // namespace microsimulator::device
