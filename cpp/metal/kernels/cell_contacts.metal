// Contacts between pairs of rod cells.
//
// A cell is the capsule of radius r about the segment of its length centered
// on its position along its axis. Detection runs in two phases, each a count,
// an inclusive prefix sum of the counts, and a fill:
//
//   Broad phase, over cells. Each cell's axis-aligned box, padded by its
//   radius and half the activation margin, goes into a hashed uniform grid
//   whose boxes are at least as wide as the widest cell box. A cell's
//   candidates are the cells of higher id in the 27 grid boxes about its own
//   whose boxes overlap its box, in ascending id order.
//
//   Narrow phase, over candidates. A pair makes a contact at each of its one
//   or two contact points whose separation is below the activation margin.
//
// Candidates follow cell slots and then partner ids, and contacts follow
// candidates, so a run gives the same output every time.

// A cell as the narrow phase reads it.
struct ContactCapsule {
  float3 center;
  float3 axis;
  float length;
  float radius;
};

inline ContactCapsule contact_capsule(device const float4* centers, device const float4* axes,
                                      uint slot) {
  float4 center = centers[slot];
  float4 axis = axes[slot];
  ContactCapsule result;
  result.center = float3(center.x, center.y, center.z);
  result.axis = float3(axis.x, axis.y, axis.z);
  result.length = center.w;
  result.radius = axis.w;
  return result;
}

struct ContactPointPair {
  float3 first;
  float3 second;
};

inline ContactPointPair contact_point_pair(float3 first, float3 second) {
  ContactPointPair result;
  result.first = first;
  result.second = second;
  return result;
}

// The closest points of the two centerline segments.
inline ContactPointPair capsule_closest_points(ContactCapsule first, ContactCapsule second,
                                               float epsilon) {
  float first_half = first.length * 0.5f;
  float second_half = second.length * 0.5f;
  float3 first_start = (first.center - (first.axis * first_half));
  float3 first_end = (first.center + (first.axis * first_half));
  float3 second_start = (second.center - (second.axis * second_half));
  float3 second_end = (second.center + (second.axis * second_half));
  float3 first_delta = (first_end - first_start);
  float3 second_delta = (second_end - second_start);
  float3 between_starts = (first_start - second_start);
  float first_length_squared = dot(first_delta, first_delta);
  float second_length_squared = dot(second_delta, second_delta);
  float second_projection = dot(second_delta, between_starts);
  float epsilon_squared = epsilon * epsilon;

  float first_parameter = 0.0f;
  float second_parameter = 0.0f;
  if (first_length_squared <= epsilon_squared && second_length_squared <= epsilon_squared) {
    return contact_point_pair(first_start, second_start);
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
          (first_length_squared * second_length_squared) - (cross_projection * cross_projection);
      float parallel_tolerance = float_epsilon * first_length_squared * second_length_squared;
      if (denominator > parallel_tolerance) {
        first_parameter = clamp(
            ((cross_projection * second_projection) - (first_projection * second_length_squared)) /
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
  return contact_point_pair((first_start + (first_delta * first_parameter)),
                            (second_start + (second_delta * second_parameter)));
}

// The one or two contact points of a pair. Nearly parallel cells whose
// projections overlap touch at both ends of the overlap.
struct ContactPointPairs {
  ContactPointPair pairs[2];
  uint count;
};

inline ContactPointPairs capsule_contact_points(ContactCapsule first, ContactCapsule second,
                                                float parallel_sine_threshold, float epsilon) {
  ContactPointPairs result;
  float axis_dot = clamp(dot(first.axis, second.axis), -1.0f, 1.0f);
  float sine_squared = 1.0f - (axis_dot * axis_dot);
  float sine = sqrt(sine_squared > 0.0f ? sine_squared : 0.0f);
  if (sine > parallel_sine_threshold || first.length <= epsilon || second.length <= epsilon) {
    result.pairs[0] = capsule_closest_points(first, second, epsilon);
    result.count = 1u;
    return result;
  }
  float first_half = first.length * 0.5f;
  float second_half = second.length * 0.5f;
  float center_coordinate = dot((second.center - first.center), first.axis);
  float projected_second_half = second_half * fabs(axis_dot);
  float begin_candidate = center_coordinate - projected_second_half;
  float end_candidate = center_coordinate + projected_second_half;
  float overlap_begin = -first_half < begin_candidate ? begin_candidate : -first_half;
  float overlap_end = end_candidate < first_half ? end_candidate : first_half;
  if (overlap_end - overlap_begin <= epsilon) {
    result.pairs[0] = capsule_closest_points(first, second, epsilon);
    result.count = 1u;
    return result;
  }
  result.count = 2u;
  for (uint ordinal = 0u; ordinal < 2u; ++ordinal) {
    float first_parameter = ordinal == 0u ? overlap_begin : overlap_end;
    float3 point_on_first = (first.center + (first.axis * first_parameter));
    float second_parameter =
        clamp(dot((point_on_first - second.center), second.axis), -second_half, second_half);
    result.pairs[ordinal] =
        contact_point_pair(point_on_first, (second.center + (second.axis * second_parameter)));
  }
  return result;
}

inline float3 contact_normalized(float3 value) {
  return (value * (1.0f / length(value)));
}

// The unit normal from the first cell to the second at a contact point pair.
// Coincident points take the normal of the plane of the two axes, then the
// transverse direction between the centers, then the coordinate axis least
// aligned with the first cell's.
inline float3 capsule_contact_normal(ContactCapsule first, ContactCapsule second,
                                     ContactPointPair points, float epsilon) {
  float3 point_delta = (points.second - points.first);
  if (length(point_delta) > epsilon) {
    return contact_normalized(point_delta);
  }
  float3 axes_cross = cross(first.axis, second.axis);
  if (length(axes_cross) > epsilon) {
    return contact_normalized(axes_cross);
  }
  float3 center_delta = (second.center - first.center);
  float3 transverse = (center_delta - (first.axis * dot(center_delta, first.axis)));
  if (length(transverse) > epsilon) {
    return contact_normalized(transverse);
  }
  float3 least = float3(1.0f, 0.0f, 0.0f);
  float least_alignment = fabs(dot(first.axis, least));
  for (uint axis = 1u; axis < 3u; ++axis) {
    float3 basis = float3(0.0f, axis == 1u ? 1.0f : 0.0f, axis == 2u ? 1.0f : 0.0f);
    float alignment = fabs(dot(first.axis, basis));
    if (alignment < least_alignment) {
      least = basis;
      least_alignment = alignment;
    }
  }
  return contact_normalized(cross(first.axis, least));
}

// The contacts of one candidate pair whose separation is below the margin.
struct CellPairEvaluation {
  CellContactRecord records[2];
  uint count;
};

inline CellPairEvaluation cell_pair_evaluate(CellContactData data, device const float4* centers,
                                             device const float4* axes, uint2 candidate) {
  ContactCapsule first = contact_capsule(centers, axes, candidate.x);
  ContactCapsule second = contact_capsule(centers, axes, candidate.y);
  ContactPointPairs points =
      capsule_contact_points(first, second, data.parallel_sine_threshold, data.degeneracy_epsilon);
  float weight = points.count == 2u ? inverse_sqrt_two : 1.0f;
  CellPairEvaluation result;
  result.count = 0u;
  for (uint ordinal = 0u; ordinal < points.count; ++ordinal) {
    ContactPointPair pair = points.pairs[ordinal];
    float separation = length((pair.second - pair.first)) - (first.radius + second.radius);
    if (!(separation < data.activation_margin)) {
      continue;
    }
    float3 normal = capsule_contact_normal(first, second, pair, data.degeneracy_epsilon);
    float3 point = (pair.first + (normal * first.radius));
    CellContactRecord record;
    record.point = float4(point.x, point.y, point.z, separation);
    record.normal = float4(normal.x, normal.y, normal.z, weight);
    record.first = candidate.x;
    record.second = candidate.y;
    record.ordinal = ordinal;
    record.padding = 0u;
    result.records[result.count] = record;
    result.count += 1u;
  }
  return result;
}

// ---------------------------------------------------------------------------
// Broad phase.

// A cell's box: its center with the box's half extent added to each side. A
// relative pad covers the rounding of the box in float, so the box holds the
// exact capsule grown by half the margin.
inline void cell_bounds(uint cell, CellContactData data, device const float4* centers,
                        device const float4* axes, device float4* lower, device float4* upper) {
  float4 center = centers[cell];
  float4 axis = axes[cell];
  float half_length = center.w * 0.5f;
  float padding = axis.w + data.activation_margin * 0.5f;
  float extent_x = fabs(axis.x) * half_length + padding;
  float extent_y = fabs(axis.y) * half_length + padding;
  float extent_z = fabs(axis.z) * half_length + padding;
  extent_x += 1.0e-6f * (fabs(center.x) + extent_x);
  extent_y += 1.0e-6f * (fabs(center.y) + extent_y);
  extent_z += 1.0e-6f * (fabs(center.z) + extent_z);
  lower[cell] = float4(center.x - extent_x, center.y - extent_y, center.z - extent_z, 0.0f);
  upper[cell] = float4(center.x + extent_x, center.y + extent_y, center.z + extent_z, 0.0f);
}

inline float cell_larger(float left, float right) {
  return left < right ? right : left;
}

// One level of the reduction of box widths to their largest along each axis:
// element `index` of the output takes the larger of input elements 2 index
// and 2 index + 1. The first level reads the boxes themselves.
inline void cell_width_leaf(uint index, uint count, device const float4* lower,
                            device const float4* upper, device float4* output) {
  float4 result = float4(0.0f, 0.0f, 0.0f, 0.0f);
  for (uint member = 2u * index; member < 2u * index + 2u && member < count; ++member) {
    result.x = cell_larger(result.x, upper[member].x - lower[member].x);
    result.y = cell_larger(result.y, upper[member].y - lower[member].y);
    result.z = cell_larger(result.z, upper[member].z - lower[member].z);
  }
  output[index] = result;
}

inline void cell_width_pair(uint index, uint count, device const float4* input,
                            device float4* output) {
  float4 result = input[2u * index];
  if (2u * index + 1u < count) {
    float4 other = input[2u * index + 1u];
    result.x = cell_larger(result.x, other.x);
    result.y = cell_larger(result.y, other.y);
    result.z = cell_larger(result.z, other.z);
  }
  output[index] = result;
}

// The width of a grid box along one axis, from the widest cell box. The
// margin keeps rounding from placing overlapping boxes two grid boxes apart.
inline float cell_grid_width(float widest) {
  float width = widest * 1.001f;
  return width > 1.0e-30f ? width : 1.0e-30f;
}

inline int cell_grid_coordinate(float lower, float upper, float width) {
  float position = floor((0.5f * (lower + upper)) / width);
  float limit = 1073741824.0f;
  position = position < -limit ? -limit : (position > limit ? limit : position);
  return int(position);
}

inline uint cell_bucket(int x, int y, int z, uint bucket_count) {
  return ((uint(x) * 73856093u) ^ (uint(y) * 19349663u) ^ (uint(z) * 83492791u)) &
         (bucket_count - 1u);
}

// A cell's grid box, stored with its bucket in w, and its count in that
// bucket.
inline void cell_grid_place(uint cell, CellContactData data, device const float4* lower,
                            device const float4* upper, device const float4* widest,
                            device uint4* coordinates, device atomic_uint* bucket_counts) {
  float4 width = widest[0];
  int x = cell_grid_coordinate(lower[cell].x, upper[cell].x, cell_grid_width(width.x));
  int y = cell_grid_coordinate(lower[cell].y, upper[cell].y, cell_grid_width(width.y));
  int z = cell_grid_coordinate(lower[cell].z, upper[cell].z, cell_grid_width(width.z));
  uint bucket = cell_bucket(x, y, z, data.bucket_count);
  coordinates[cell] = uint4(uint(x), uint(y), uint(z), bucket);
  atomic_fetch_add_explicit(&bucket_counts[bucket], 1u, memory_order_relaxed);
}

// Files a cell under its bucket. Members of a bucket land in any order; the
// candidates do not depend on it.
inline void cell_grid_fill(uint cell, device const uint4* coordinates,
                           device const uint* bucket_ends, device atomic_uint* bucket_cursors,
                           device uint* members) {
  uint bucket = coordinates[cell].w;
  uint start = bucket == 0u ? 0u : bucket_ends[bucket - 1u];
  members[start + atomic_fetch_add_explicit(&bucket_cursors[bucket], 1u, memory_order_relaxed)] =
      cell;
}

// Boxes that touch count as overlapping.
inline bool cell_boxes_overlap(float4 left_lower, float4 left_upper, float4 right_lower,
                               float4 right_upper) {
  return left_upper.x >= right_lower.x && right_upper.x >= left_lower.x &&
         left_upper.y >= right_lower.y && right_upper.y >= left_lower.y &&
         left_upper.z >= right_lower.z && right_upper.z >= left_lower.z;
}

// Visits the candidates of `cell`, and with `write` stores them in `output`
// from `start`. A member is taken only from the
// bucket of its own grid box, so a bucket that two neighboring grid boxes
// share yields each member once.
inline uint cell_candidates_visit(uint cell, CellContactData data, device const ulong* ids,
                                  device const float4* lower, device const float4* upper,
                                  device const uint4* coordinates, device const uint* bucket_ends,
                                  device const uint* members, device uint2* output, uint start,
                                  bool write) {
  uint4 own = coordinates[cell];
  ulong id = ids[cell];
  float4 own_lower = lower[cell];
  float4 own_upper = upper[cell];
  uint found = 0u;
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dz = -1; dz <= 1; ++dz) {
        uint x = uint(int(own.x) + dx);
        uint y = uint(int(own.y) + dy);
        uint z = uint(int(own.z) + dz);
        uint bucket = cell_bucket(int(x), int(y), int(z), data.bucket_count);
        uint begin = bucket == 0u ? 0u : bucket_ends[bucket - 1u];
        uint end = bucket_ends[bucket];
        for (uint entry = begin; entry < end; ++entry) {
          uint other = members[entry];
          uint4 at = coordinates[other];
          if (at.x != x || at.y != y || at.z != z || !(ids[other] > id) ||
              !cell_boxes_overlap(own_lower, own_upper, lower[other], upper[other])) {
            continue;
          }
          if (write) {
            output[start + found] = uint2(cell, other);
          }
          found += 1u;
        }
      }
    }
  }
  return found;
}

// Pass 1 of the broad phase. The number of candidates of `cell`; `candidates`
// is not written.
inline uint cell_candidate_count(uint cell, CellContactData data, device const ulong* ids,
                                 device const float4* lower, device const float4* upper,
                                 device const uint4* coordinates, device const uint* bucket_ends,
                                 device const uint* members, device uint2* candidates) {
  return cell_candidates_visit(cell, data, ids, lower, upper, coordinates, bucket_ends, members,
                               candidates, 0u, false);
}

// Pass 3 of the broad phase. Writes the candidates of `cell` in ascending
// partner id. `ends` is the inclusive prefix sum of the counts; a cell whose
// range passes the capacity writes nothing.
inline void cell_candidate_fill(uint cell, CellContactData data, device const ulong* ids,
                                device const float4* lower, device const float4* upper,
                                device const uint4* coordinates, device const uint* bucket_ends,
                                device const uint* members, device const uint* ends,
                                device uint2* candidates) {
  uint start = cell == 0u ? 0u : ends[cell - 1u];
  uint finish = ends[cell];
  if (finish > data.candidate_capacity) {
    return;
  }
  cell_candidates_visit(cell, data, ids, lower, upper, coordinates, bucket_ends, members,
                        candidates, start, true);
  for (uint entry = start + 1u; entry < finish; ++entry) {
    uint2 value = candidates[entry];
    ulong key = ids[value.y];
    uint position = entry;
    while (position > start && ids[candidates[position - 1u].y] > key) {
      candidates[position] = candidates[position - 1u];
      position -= 1u;
    }
    candidates[position] = value;
  }
}

// Pass 1 of the narrow phase. The number of contacts of candidate
// `candidate`, of `total`.
inline uint cell_pair_count(uint candidate, uint total, CellContactData data,
                            device const float4* centers, device const float4* axes,
                            device const uint2* candidates) {
  if (candidate >= total) {
    return 0u;
  }
  return cell_pair_evaluate(data, centers, axes, candidates[candidate]).count;
}

// Pass 3 of the narrow phase. Writes the contacts of candidate `candidate`
// from the start of its range; `ends` is the inclusive prefix sum of the
// counts. A candidate whose range passes the capacity writes nothing.
inline void cell_pair_fill(uint candidate, uint total, CellContactData data,
                           device const float4* centers, device const float4* axes,
                           device const uint2* candidates, device const uint* ends,
                           device CellContactRecord* records) {
  if (candidate >= total || ends[candidate] > data.contact_capacity) {
    return;
  }
  CellPairEvaluation evaluation = cell_pair_evaluate(data, centers, axes, candidates[candidate]);
  uint start = candidate == 0u ? 0u : ends[candidate - 1u];
  for (uint index = 0u; index < evaluation.count; ++index) {
    records[start + index] = evaluation.records[index];
  }
}

kernel void zero_values(device uint* values [[buffer(0)]], constant uint& count [[buffer(1)]],
                        uint index [[thread_position_in_grid]]) {
  if (index < count) {
    values[index] = 0u;
  }
}

kernel void cell_bounds_kernel(device const float4* centers [[buffer(0)]],
                               device const float4* axes [[buffer(1)]],
                               device float4* lower [[buffer(2)]],
                               device float4* upper [[buffer(3)]],
                               constant CellContactData& data [[buffer(4)]],
                               uint cell [[thread_position_in_grid]]) {
  if (cell >= data.cell_count) {
    return;
  }
  cell_bounds(cell, data, centers, axes, lower, upper);
}

kernel void cell_width_leaf_kernel(device const float4* lower [[buffer(0)]],
                                   device const float4* upper [[buffer(1)]],
                                   device float4* output [[buffer(2)]],
                                   constant uint& count [[buffer(3)]],
                                   uint index [[thread_position_in_grid]]) {
  if (index >= (count + 1u) / 2u) {
    return;
  }
  cell_width_leaf(index, count, lower, upper, output);
}

kernel void cell_width_pair_kernel(device const float4* input [[buffer(0)]],
                                   device float4* output [[buffer(1)]],
                                   constant uint& count [[buffer(2)]],
                                   uint index [[thread_position_in_grid]]) {
  if (index >= (count + 1u) / 2u) {
    return;
  }
  cell_width_pair(index, count, input, output);
}

kernel void cell_grid_place_kernel(device const float4* lower [[buffer(0)]],
                                   device const float4* upper [[buffer(1)]],
                                   device const float4* widest [[buffer(2)]],
                                   device uint4* coordinates [[buffer(3)]],
                                   device atomic_uint* bucket_counts [[buffer(4)]],
                                   constant CellContactData& data [[buffer(5)]],
                                   uint cell [[thread_position_in_grid]]) {
  if (cell >= data.cell_count) {
    return;
  }
  cell_grid_place(cell, data, lower, upper, widest, coordinates, bucket_counts);
}

kernel void cell_grid_fill_kernel(device const uint4* coordinates [[buffer(0)]],
                                  device const uint* bucket_ends [[buffer(1)]],
                                  device atomic_uint* bucket_cursors [[buffer(2)]],
                                  device uint* members [[buffer(3)]],
                                  constant CellContactData& data [[buffer(4)]],
                                  uint cell [[thread_position_in_grid]]) {
  if (cell >= data.cell_count) {
    return;
  }
  cell_grid_fill(cell, coordinates, bucket_ends, bucket_cursors, members);
}

kernel void cell_candidate_count_kernel(
    device const ulong* ids [[buffer(0)]], device const float4* lower [[buffer(1)]],
    device const float4* upper [[buffer(2)]], device const uint4* coordinates [[buffer(3)]],
    device const uint* bucket_ends [[buffer(4)]], device const uint* members [[buffer(5)]],
    device uint2* candidates [[buffer(6)]], device uint* counts [[buffer(7)]],
    constant CellContactData& data [[buffer(8)]], uint cell [[thread_position_in_grid]]) {
  if (cell >= data.cell_count) {
    return;
  }
  counts[cell] = cell_candidate_count(cell, data, ids, lower, upper, coordinates, bucket_ends,
                                      members, candidates);
}

kernel void cell_candidate_fill_kernel(
    device const ulong* ids [[buffer(0)]], device const float4* lower [[buffer(1)]],
    device const float4* upper [[buffer(2)]], device const uint4* coordinates [[buffer(3)]],
    device const uint* bucket_ends [[buffer(4)]], device const uint* members [[buffer(5)]],
    device const uint* ends [[buffer(6)]], device uint2* candidates [[buffer(7)]],
    constant CellContactData& data [[buffer(8)]], uint cell [[thread_position_in_grid]]) {
  if (cell >= data.cell_count) {
    return;
  }
  cell_candidate_fill(cell, data, ids, lower, upper, coordinates, bucket_ends, members, ends,
                      candidates);
}

// The candidates the broad phase wrote, of its capacity.
uint cell_candidate_total(device const uint* candidate_ends, constant CellContactData& data) {
  uint total = data.cell_count == 0u ? 0u : candidate_ends[data.cell_count - 1u];
  return min(total, data.candidate_capacity);
}

kernel void cell_pair_count_kernel(device const float4* centers [[buffer(0)]],
                                   device const float4* axes [[buffer(1)]],
                                   device const uint2* candidates [[buffer(2)]],
                                   device const uint* candidate_ends [[buffer(3)]],
                                   device uint* counts [[buffer(4)]],
                                   constant CellContactData& data [[buffer(5)]],
                                   uint candidate [[thread_position_in_grid]]) {
  if (candidate >= data.candidate_capacity) {
    return;
  }
  counts[candidate] = cell_pair_count(candidate, cell_candidate_total(candidate_ends, data), data,
                                      centers, axes, candidates);
}

kernel void cell_pair_fill_kernel(
    device const float4* centers [[buffer(0)]], device const float4* axes [[buffer(1)]],
    device const uint2* candidates [[buffer(2)]], device const uint* candidate_ends [[buffer(3)]],
    device const uint* ends [[buffer(4)]], device CellContactRecord* records [[buffer(5)]],
    constant CellContactData& data [[buffer(6)]], uint candidate [[thread_position_in_grid]]) {
  if (candidate >= data.candidate_capacity) {
    return;
  }
  cell_pair_fill(candidate, cell_candidate_total(candidate_ends, data), data, centers, axes,
                 candidates, ends, records);
}
