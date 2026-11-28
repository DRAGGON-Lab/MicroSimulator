#pragma once

#include "cuda/device_types.cuh"
#include "cuda/kernels/cell_contacts.cuh"
#include "cuda/kernels/contact_solve.cuh"
#include "cuda/kernels/external_contacts.cuh"
#include "cuda/kernels/flow_drift.cuh"
#include "cuda/kernels/vector_math.cuh"

namespace microsimulator::cuda::device {

using namespace microsimulator::shared;

// The stages of one contact dynamics substep on the device that holds it:
// every stage of core/contact_dynamics.cpp's substep as a function of one
// cell, contact, or row, over the buffers of shared/contact_step_layout.h.

__device__ inline bool step_gated(const ContactStepBindings& b) {
  return b.status[MS_STEP_FAILED] != 0u || b.status[MS_STEP_OVERFLOW] != 0u;
}

__device__ inline void step_raise(const ContactStepBindings& b, uint word, float value) {
  atomicMax((&b.status[word]), __float_as_uint(value));
}

__device__ inline float step_word_float(const ContactStepBindings& b, uint word) {
  return __uint_as_float(b.status[word]);
}

__device__ inline uint step_min(uint left, uint right) {
  return left < right ? left : right;
}

__device__ inline float4* step_centers(const ContactStepBindings& b, ContactStepParams p) {
  return p.geometry == 0u ? b.base_centers : b.trial_centers;
}

__device__ inline float4* step_axes(const ContactStepBindings& b, ContactStepParams p) {
  return p.geometry == 0u ? b.base_axes : b.trial_axes;
}

__device__ inline CellContactData step_cell_data(ContactStepParams p) {
  CellContactData data;
  data.cell_count = p.cell_count;
  data.bucket_count = p.bucket_count;
  data.candidate_capacity = p.candidate_capacity;
  data.contact_capacity = 2u * p.candidate_capacity;
  data.activation_margin = p.margin;
  data.parallel_sine_threshold = p.parallel_sine_threshold;
  data.degeneracy_epsilon = p.contact_degeneracy;
  data.padding0 = 0.0f;
  return data;
}

__device__ inline ExternalContactData step_wall_data(ContactStepParams p) {
  ExternalContactData data;
  data.cell_count = p.cell_count;
  data.constraint_count = p.constraint_count;
  data.every_face = 1u;
  data.padding0 = 0u;
  data.activation_margin = p.margin;
  data.degeneracy_epsilon = p.constraint_degeneracy;
  data.padding1 = 0.0f;
  data.padding2 = 0.0f;
  return data;
}

// The solve's data with the substep's current rows.
__device__ inline ContactSolveData step_solve_data(const ContactStepBindings& b,
                                                   ContactStepParams p) {
  ContactSolveData data = p.solve;
  data.row_count = b.status[MS_STEP_ROWS];
  data.body_count = p.cell_count;
  return data;
}

// ---------------------------------------------------------------------------
// The step and its substeps.

// Clears what an attempt at a substep accumulates.
__device__ inline void step_substep_begin(uint index, const ContactStepBindings& b,
                                          ContactStepParams p) {
  (void)p;
  if (index != 0u) {
    return;
  }
  b.status[MS_STEP_FAILED] = 0u;
  b.status[MS_STEP_OVERFLOW] = 0u;
  b.status[MS_STEP_SOLVE_ERROR] = 0u;
  b.status[MS_STEP_ROTATION] = 0u;
  b.status[MS_STEP_ITERATIONS] = 0u;
  b.status[MS_STEP_PROBE_ITERATIONS] = 0u;
  b.status[MS_STEP_STARTING_OVERLAP] = 0u;
  b.status[MS_STEP_ROWS] = 0u;
}

// Clears what one refinement round accumulates.
__device__ inline void step_round_begin(uint index, const ContactStepBindings& b,
                                        ContactStepParams p) {
  (void)p;
  if (index != 0u || step_gated(b)) {
    return;
  }
  b.status[MS_STEP_SOLVE_STATUS] = MS_STEP_CONVERGED;
  b.status[MS_STEP_PROBE_ROTATION] = 0u;
  b.status[MS_STEP_MOVE_ROTATION] = 0u;
  b.status[MS_STEP_MOVE_INVALID] = 0u;
  b.status[MS_STEP_MAX_OVERLAP] = 0u;
  b.status[MS_STEP_FOUND] = 0u;
  b.status[MS_STEP_OVERLAPPING] = 0u;
}

// The velocity then angular velocity the flow gives a cell of the base
// geometry over `p.drift.dt`.
__device__ inline void step_drift(uint cell, const ContactStepBindings& b, ContactStepParams p) {
  if (cell >= p.cell_count || step_gated(b)) {
    return;
  }
  uint result = flow_drift_rates_cell(cell, p.drift, b.base_centers, b.base_axes, b.fixed_cells,
                                      b.solid, b.x_faces, b.y_faces, b.z_faces, b.drift);
  if (result != drift_ok) {
    atomicMax((&b.status[MS_STEP_DRIFT_ERROR]), result);
  }
}

// The largest growth of a cell end over the step, in radii, and the fastest
// drift, which together bound the first substep's length.
__device__ inline void step_begin_bounds(uint cell, const ContactStepBindings& b,
                                         ContactStepParams p) {
  if (cell >= p.cell_count) {
    return;
  }
  float4 rates = b.cell_rates[cell];
  float growth = fabsf(rates.x) * b.base_centers[cell].w * 0.5f * p.step_dt / b.base_axes[cell].w;
  step_raise(b, MS_STEP_GROWTH, isfinite(growth) ? growth : INFINITY);
  if (p.has_drift != 0u) {
    const float* rate = b.drift + 6u * cell;
    float speed = length(make_float3(rate[0], rate[1], rate[2]));
    step_raise(b, MS_STEP_DRIFT_SPEED, isfinite(speed) ? speed : INFINITY);
  }
}

// A cell's mobilities from the base geometry and its elongation rate from the
// length it had at the start of the step, so that the elongation of an
// unloaded cell does not depend on how the step is divided.
__device__ inline void step_bodies(uint cell, const ContactStepBindings& b, ContactStepParams p) {
  if (cell >= p.cell_count || step_gated(b)) {
    return;
  }
  float4 center = b.base_centers[cell];
  float4 axis = b.base_axes[cell];
  float4 rates = b.cell_rates[cell];
  ContactBodyData body;
  body.axis_mobility = make_float4(axis.x, axis.y, axis.z, 0.0f);
  body.rates = make_float4(0.0f, 0.0f, rates.x * rates.y, 0.0f);
  body.drift_linear = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  body.drift_angular = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  if (p.has_drift != 0u) {
    const float* rate = b.drift + 6u * cell;
    body.drift_linear = make_float4(rate[0], rate[1], rate[2], 0.0f);
    body.drift_angular = make_float4(rate[3], rate[4], rate[5], 0.0f);
  }
  if (b.fixed_cells[cell] == 0u) {
    float radius = axis.w;
    float total_length = center.w + 2.0f * radius;
    float translational = p.drag * total_length;
    float axial = 0.5f * translational * radius * radius;
    float transverse =
        translational * ((total_length * total_length) + 3.0f * radius * radius) / 12.0f;
    body.axis_mobility.w = 1.0f / translational;
    body.rates.x = 1.0f / transverse;
    body.rates.y = 1.0f / axial;
  }
  b.bodies[cell] = body;
}

// ---------------------------------------------------------------------------
// Detection of the contacts of one geometry, as the cell and external
// contact kernels find them.

__device__ inline void step_detect_clear(uint index, const ContactStepBindings& b,
                                         ContactStepParams p) {
  if (step_gated(b)) {
    return;
  }
  if (index < p.bucket_count) {
    b.bucket_counts[index] = 0u;
    b.bucket_cursors[index] = 0u;
  }
  if (index == 0u) {
    b.status[MS_STEP_WIDTH] = 0u;
    b.status[MS_STEP_WIDTH + 1u] = 0u;
    b.status[MS_STEP_WIDTH + 2u] = 0u;
    b.status[MS_STEP_CANDIDATES] = 0u;
    b.status[MS_STEP_CELL_CONTACTS] = 0u;
    b.status[MS_STEP_WALL_CONTACTS] = 0u;
  }
}

__device__ inline void step_bounds(uint cell, const ContactStepBindings& b, ContactStepParams p) {
  if (cell >= p.cell_count || step_gated(b)) {
    return;
  }
  cell_bounds(cell, step_cell_data(p), step_centers(b, p), step_axes(b, p), b.lower, b.upper);
}

__device__ inline void step_width(uint cell, const ContactStepBindings& b, ContactStepParams p) {
  if (cell >= p.cell_count || step_gated(b)) {
    return;
  }
  step_raise(b, MS_STEP_WIDTH, b.upper[cell].x - b.lower[cell].x);
  step_raise(b, MS_STEP_WIDTH + 1u, b.upper[cell].y - b.lower[cell].y);
  step_raise(b, MS_STEP_WIDTH + 2u, b.upper[cell].z - b.lower[cell].z);
}

__device__ inline void step_place(uint cell, const ContactStepBindings& b, ContactStepParams p) {
  if (cell >= p.cell_count || step_gated(b)) {
    return;
  }
  int x = cell_grid_coordinate(b.lower[cell].x, b.upper[cell].x,
                               cell_grid_width(step_word_float(b, MS_STEP_WIDTH)));
  int y = cell_grid_coordinate(b.lower[cell].y, b.upper[cell].y,
                               cell_grid_width(step_word_float(b, MS_STEP_WIDTH + 1u)));
  int z = cell_grid_coordinate(b.lower[cell].z, b.upper[cell].z,
                               cell_grid_width(step_word_float(b, MS_STEP_WIDTH + 2u)));
  uint bucket = cell_bucket(x, y, z, p.bucket_count);
  b.coordinates[cell] = make_uint4(uint(x), uint(y), uint(z), bucket);
  atomicAdd((&b.bucket_counts[bucket]), 1u);
}

__device__ inline void step_grid_fill(uint cell, const ContactStepBindings& b,
                                      ContactStepParams p) {
  if (cell >= p.cell_count || step_gated(b)) {
    return;
  }
  cell_grid_fill(cell, b.coordinates, b.bucket_ends, (b.bucket_cursors), b.members);
}

__device__ inline void step_candidate_count(uint cell, const ContactStepBindings& b,
                                            ContactStepParams p) {
  if (cell >= p.cell_count || step_gated(b)) {
    return;
  }
  b.candidate_counts[cell] =
      cell_candidate_count(cell, step_cell_data(p), b.ids, b.lower, b.upper, b.coordinates,
                           b.bucket_ends, b.members, b.candidates);
}

__device__ inline void step_candidate_fill(uint cell, const ContactStepBindings& b,
                                           ContactStepParams p) {
  if (cell >= p.cell_count || step_gated(b)) {
    return;
  }
  cell_candidate_fill(cell, step_cell_data(p), b.ids, b.lower, b.upper, b.coordinates,
                      b.bucket_ends, b.members, b.candidate_ends, b.candidates);
}

__device__ inline void step_pair_count(uint candidate, const ContactStepBindings& b,
                                       ContactStepParams p) {
  if (candidate >= p.candidate_capacity || step_gated(b)) {
    return;
  }
  uint total = step_min(b.status[MS_STEP_CANDIDATES], p.candidate_capacity);
  b.pair_counts[candidate] = cell_pair_count(candidate, total, step_cell_data(p),
                                             step_centers(b, p), step_axes(b, p), b.candidates);
}

__device__ inline void step_pair_fill(uint candidate, const ContactStepBindings& b,
                                      ContactStepParams p) {
  if (candidate >= p.candidate_capacity || step_gated(b)) {
    return;
  }
  uint total = step_min(b.status[MS_STEP_CANDIDATES], p.candidate_capacity);
  cell_pair_fill(candidate, total, step_cell_data(p), step_centers(b, p), step_axes(b, p),
                 b.candidates, b.pair_ends, b.cell_records);
}

__device__ inline void step_wall_count(uint pair, const ContactStepBindings& b,
                                       ContactStepParams p) {
  if (pair >= p.cell_count * p.constraint_count || step_gated(b)) {
    return;
  }
  b.wall_counts[pair] = external_pair_count(pair, step_wall_data(p), step_centers(b, p),
                                            step_axes(b, p), b.constraints);
}

__device__ inline void step_wall_fill(uint pair, const ContactStepBindings& b,
                                      ContactStepParams p) {
  if (pair >= p.cell_count * p.constraint_count || step_gated(b) ||
      b.wall_ends[pair] > p.wall_capacity) {
    return;
  }
  external_pair_fill(pair, step_wall_data(p), step_centers(b, p), step_axes(b, p), b.constraints,
                     b.wall_ends, b.wall_records);
}

__device__ inline void step_touches_total(uint index, const ContactStepBindings& b,
                                          ContactStepParams p) {
  if (index != 0u || step_gated(b)) {
    return;
  }
  b.status[p.output == 0u ? MS_STEP_TOUCHES : MS_STEP_FOUND] =
      b.status[MS_STEP_CELL_CONTACTS] + b.status[MS_STEP_WALL_CONTACTS];
}

// Touch `index` of the detection: the cell contacts in candidate order, then
// the wall contacts in pair order.
__device__ inline void step_touches_fill(uint index, const ContactStepBindings& b,
                                         ContactStepParams p) {
  if (index >= p.touch_capacity || step_gated(b)) {
    return;
  }
  uint cells = b.status[MS_STEP_CELL_CONTACTS];
  uint walls = b.status[MS_STEP_WALL_CONTACTS];
  if (index >= cells + walls) {
    return;
  }
  ContactTouch touch;
  if (index < cells) {
    CellContactRecord record = b.cell_records[index];
    touch.point = record.point;
    touch.normal = record.normal;
    touch.first = record.first;
    touch.second = record.second;
    touch.partner_low = record.second;
    touch.partner_high = 0u;
  } else {
    ExternalContactRecord record = b.wall_records[index - cells];
    ulong id = b.constraints[record.constraint].id;
    touch.point = record.point;
    touch.normal = record.normal;
    touch.first = record.cell;
    touch.second = MS_STEP_NO_SECOND;
    touch.partner_low = uint(id & 0xffffffffu);
    touch.partner_high = uint(id >> 32);
  }
  ContactTouch* output = p.output == 0u ? b.touches : b.found;
  output[index] = touch;
}

// ---------------------------------------------------------------------------
// The rows of a substep.

__device__ inline uint4 step_key(ContactTouch touch) {
  return make_uint4(touch.first, touch.second == MS_STEP_NO_SECOND ? 1u : 0u, touch.partner_low,
                    touch.partner_high);
}

__device__ inline bool step_same_key(uint4 left, uint4 right) {
  return left.x == right.x && left.y == right.y && left.z == right.z && left.w == right.w;
}

__device__ inline float step_compliance(ContactTouch touch, ContactStepParams p) {
  return touch.second == MS_STEP_NO_SECOND ? p.wall_compliance : p.contact_compliance;
}

// The fraction of a cell's elongation that closes a contact whose centerline
// point is `point` and whose normal points out of the cell. The cylinder
// stretches uniformly about its center and the caps ride on its ends.
__device__ inline float step_length_lever(float4 center, float4 axis, float3 point,
                                          float3 outward) {
  float length = center.w;
  if (length <= 1.0e-6f) {
    return 0.0f;
  }
  float3 direction = make_float3(axis.x, axis.y, axis.z);
  float3 position = make_float3(center.x, center.y, center.z);
  float coordinate = dot(direction, (point - position)) / length;
  float lever = fminf(fmaxf(coordinate, -0.5f), 0.5f) * dot(direction, outward);
  return lever > 0.0f ? lever : 0.0f;
}

// The row of a touch at the geometry of `centers` and `axes`. A contact
// deeper than the recovery limit asks for only that much separation, and is
// stiffened in proportion so that a pair unable to separate still meets the
// full contact force of its overlap.
__device__ inline ContactRowData step_make_row(ContactTouch touch, const float4* centers,
                                               const float4* axes, ContactStepParams p) {
  float separation = touch.point.w;
  float compliance = step_compliance(touch, p);
  float3 point = make_float3(touch.point.x, touch.point.y, touch.point.z);
  float3 normal = make_float3(touch.normal.x, touch.normal.y, touch.normal.z);
  float4 first_center = centers[touch.first];
  float4 first_axis = axes[touch.first];
  ContactRowData row;
  row.first_slot = touch.first;
  row.second_slot = touch.second;
  row.padding0 = 0u;
  row.padding1 = 0u;
  float offset = separation > -p.recovery ? separation : -p.recovery;
  float row_compliance =
      separation < -p.recovery ? compliance * p.recovery / -separation : compliance;
  float3 first_arm =
      cross((point - make_float3(first_center.x, first_center.y, first_center.z)), normal);
  float3 first_centerline = (point - (normal * first_axis.w));
  float first_lever = step_length_lever(first_center, first_axis, first_centerline, normal);
  row.first_linear = make_float4(-normal.x, -normal.y, -normal.z, first_lever);
  row.first_angular = make_float4(-first_arm.x, -first_arm.y, -first_arm.z, offset);
  row.second_linear = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  row.second_angular = make_float4(0.0f, 0.0f, 0.0f, row_compliance);
  if (touch.second != MS_STEP_NO_SECOND) {
    float4 second_center = centers[touch.second];
    float4 second_axis = axes[touch.second];
    float3 second_arm =
        cross((point - make_float3(second_center.x, second_center.y, second_center.z)), normal);
    float3 second_centerline = (point + (normal * (separation + second_axis.w)));
    float second_lever =
        step_length_lever(second_center, second_axis, second_centerline, (normal * -1.0f));
    row.second_linear = make_float4(normal.x, normal.y, normal.z, second_lever);
    row.second_angular = make_float4(second_arm.x, second_arm.y, second_arm.z, row_compliance);
  }
  return row;
}

// The rows of the substep's touches at the base geometry.
__device__ inline void step_rows_build(uint index, const ContactStepBindings& b,
                                       ContactStepParams p) {
  if (index >= p.touch_capacity || step_gated(b) || index >= b.status[MS_STEP_TOUCHES]) {
    return;
  }
  ContactTouch touch = b.touches[index];
  b.rows[index] = step_make_row(touch, b.base_centers, b.base_axes, p);
  b.keys[index] = step_key(touch);
  float depth = -touch.point.w;
  if (depth > 0.0f) {
    step_raise(b, MS_STEP_STARTING_OVERLAP, depth);
  }
}

__device__ inline void step_rows_commit(uint index, const ContactStepBindings& b,
                                        ContactStepParams p) {
  (void)p;
  if (index != 0u || step_gated(b)) {
    return;
  }
  b.status[MS_STEP_ROWS] = b.status[MS_STEP_TOUCHES];
}

// Clears the row indexes before they are rebuilt for the rows a round
// solves.
__device__ inline void step_rows_clear(uint index, const ContactStepBindings& b,
                                       ContactStepParams p) {
  if (step_gated(b)) {
    return;
  }
  if (index < p.cell_count) {
    b.keyed_counts[index] = 0u;
    b.keyed_cursors[index] = 0u;
    b.incidence_counts[index] = 0u;
    b.incidence_cursors[index] = 0u;
  }
  if (index == 0u) {
    b.keyed_offsets[0] = 0u;
    b.incidence_offsets[0] = 0u;
    b.status[MS_STEP_BOUND] = 0u;
  }
}

// The rows filed under the first cell of each, in row order.
__device__ inline void step_keyed_count(uint row, const ContactStepBindings& b,
                                        ContactStepParams p) {
  if (row >= p.row_capacity || step_gated(b) || row >= b.status[MS_STEP_ROWS]) {
    return;
  }
  atomicAdd((&b.keyed_counts[b.keys[row].x]), 1u);
}

__device__ inline void step_keyed_fill(uint row, const ContactStepBindings& b,
                                       ContactStepParams p) {
  if (row >= p.row_capacity || step_gated(b) || row >= b.status[MS_STEP_ROWS]) {
    return;
  }
  uint first = b.keys[row].x;
  uint position = b.keyed_offsets[first] + atomicAdd((&b.keyed_cursors[first]), 1u);
  b.keyed_rows[position] = row;
}

__device__ inline void step_sort_range(uint* values, uint begin, uint end) {
  for (uint entry = begin + 1u; entry < end; ++entry) {
    uint value = values[entry];
    uint position = entry;
    while (position > begin && values[position - 1u] > value) {
      values[position] = values[position - 1u];
      position -= 1u;
    }
    values[position] = value;
  }
}

__device__ inline void step_keyed_sort(uint cell, const ContactStepBindings& b,
                                       ContactStepParams p) {
  if (cell >= p.cell_count || step_gated(b)) {
    return;
  }
  step_sort_range(b.keyed_rows, b.keyed_offsets[cell], b.keyed_offsets[cell + 1u]);
}

// A row's starting multiplier: the force of the same contact in the last
// substep. A capsule pair with two contact points has two rows under one key,
// matched in row order.
__device__ inline void step_warm_start(uint row, const ContactStepBindings& b,
                                       ContactStepParams p) {
  if (row >= p.row_capacity || step_gated(b) || row >= b.status[MS_STEP_ROWS]) {
    return;
  }
  uint4 key = b.keys[row];
  uint occurrence = 0u;
  for (uint entry = b.keyed_offsets[key.x]; entry < b.keyed_offsets[key.x + 1u]; ++entry) {
    uint other = b.keyed_rows[entry];
    if (other >= row) {
      break;
    }
    occurrence += step_same_key(b.keys[other], key) ? 1u : 0u;
  }
  float force = 0.0f;
  if (key.x < p.cell_count && b.status[MS_STEP_REMEMBERED_ROWS] != 0u) {
    for (uint entry = b.remembered_offsets[key.x]; entry < b.remembered_offsets[key.x + 1u];
         ++entry) {
      uint other = b.remembered_rows[entry];
      if (step_same_key(b.remembered_keys[other], key)) {
        if (occurrence == 0u) {
          force = b.remembered_multipliers[other];
          break;
        }
        occurrence -= 1u;
      }
    }
  }
  b.multipliers[row] = force;
}

// The rows that touch each cell, in row order, as the solve reads them.
__device__ inline void step_incidence_count(uint row, const ContactStepBindings& b,
                                            ContactStepParams p) {
  if (row >= p.row_capacity || step_gated(b) || row >= b.status[MS_STEP_ROWS]) {
    return;
  }
  ContactRowData data = b.rows[row];
  atomicAdd((&b.incidence_counts[data.first_slot]), 1u);
  if (data.second_slot != MS_STEP_NO_SECOND) {
    atomicAdd((&b.incidence_counts[data.second_slot]), 1u);
  }
}

__device__ inline void step_incidence_fill(uint row, const ContactStepBindings& b,
                                           ContactStepParams p) {
  if (row >= p.row_capacity || step_gated(b) || row >= b.status[MS_STEP_ROWS]) {
    return;
  }
  ContactRowData data = b.rows[row];
  uint first = data.first_slot;
  b.incidence_entries[b.incidence_offsets[first] + atomicAdd((&b.incidence_cursors[first]), 1u)] =
      row << 1;
  if (data.second_slot != MS_STEP_NO_SECOND) {
    uint second = data.second_slot;
    b.incidence_entries[b.incidence_offsets[second] +
                        atomicAdd((&b.incidence_cursors[second]), 1u)] = (row << 1) | 1u;
  }
}

__device__ inline void step_incidence_sort(uint cell, const ContactStepBindings& b,
                                           ContactStepParams p) {
  if (cell >= p.cell_count || step_gated(b)) {
    return;
  }
  step_sort_range(b.incidence_entries, b.incidence_offsets[cell], b.incidence_offsets[cell + 1u]);
}

__device__ inline void step_row_bound(uint row, const ContactStepBindings& b, ContactStepParams p) {
  if (row >= p.row_capacity || step_gated(b) || row >= b.status[MS_STEP_ROWS]) {
    return;
  }
  float bound = contact_row_bound(row, b.rows, b.bodies, step_solve_data(b, p));
  step_raise(b, MS_STEP_BOUND, isfinite(bound) && bound >= 0.0f ? bound : INFINITY);
}

// The first step length of the round's solve, and in the first round the
// overlap a substep may accept: the tolerance, plus whatever starting overlap
// exceeds what one substep recovers.
__device__ inline void step_solve_prepare(uint index, const ContactStepBindings& b,
                                          ContactStepParams p) {
  if (index != 0u || step_gated(b)) {
    return;
  }
  if (p.round == 0u) {
    float excess = step_word_float(b, MS_STEP_STARTING_OVERLAP) - p.recovery;
    b.status[MS_STEP_SLACK] = __float_as_uint(p.tolerance + (excess > 0.0f ? excess : 0.0f));
  }
  if (b.status[MS_STEP_ROWS] == 0u) {
    b.status[MS_STEP_FIRST_STEP] = __float_as_uint(1.0f);
    return;
  }
  float bound = step_word_float(b, MS_STEP_BOUND);
  if (!isfinite(bound) || !(bound > 0.0f)) {
    b.status[MS_STEP_SOLVE_ERROR] = MS_STEP_SOLVE_NO_DIAGONAL;
    b.status[MS_STEP_FAILED] = MS_STEP_FAILED_SOLVE;
    return;
  }
  b.status[MS_STEP_FIRST_STEP] = __float_as_uint(1.0f / bound);
}

// The iterations a solve may take: what the probe left in the round it ran
// in, and all of them otherwise.
__device__ inline uint step_solve_iterations(const ContactStepBindings& b, ContactStepParams p) {
  uint used = p.probe == 0u && p.round == 0u ? b.status[MS_STEP_PROBE_ITERATIONS] : 0u;
  return p.max_iterations > used ? p.max_iterations - used : 0u;
}

// Records a solve that ended: its iterations, and whether the attempt goes
// on. A probe that breaks down ends the attempt, and one that has not
// converged has its rotation read. A full solve that does not converge ends
// the attempt.
__device__ inline void step_solve_finish(uint index, const ContactStepBindings& b,
                                         ContactStepParams p) {
  if (index != 0u || step_gated(b)) {
    return;
  }
  ContactSolveState state = b.solve_state[0];
  b.status[MS_STEP_ITERATIONS] += state.iterations;
  uint status = MS_STEP_CONVERGED;
  if (!isfinite(state.residual)) {
    status = MS_STEP_BREAKDOWN;
  } else if (!(state.residual <= p.solve.tolerance)) {
    status = MS_STEP_ITERATION_LIMIT;
  }
  b.status[MS_STEP_SOLVE_STATUS] = status;
  if (p.probe != 0u) {
    b.status[MS_STEP_PROBE_ITERATIONS] = state.iterations;
    if (status == MS_STEP_BREAKDOWN) {
      b.status[MS_STEP_FAILED] = MS_STEP_FAILED_SOLVE;
    }
  } else if (status != MS_STEP_CONVERGED) {
    b.status[MS_STEP_FAILED] = MS_STEP_FAILED_SOLVE;
  }
}

// Leaves the solve's multipliers in `multipliers`, where the solve writes
// pair `iterations % 2`.
__device__ inline void step_solve_copy(uint row, const ContactStepBindings& b,
                                       ContactStepParams p) {
  if (row >= p.row_capacity || step_gated(b) || row >= b.status[MS_STEP_ROWS]) {
    return;
  }
  if ((b.solve_state[0].iterations & 1u) != 0u) {
    b.multipliers[row] = b.multipliers_next[row];
  }
}

// The angle a cell turns through in `dt` under the response, or infinity
// when it is not finite.
__device__ inline float step_turn(ContactResponse response, float dt) {
  float angle = length(
      make_float3(response.rotation.x * dt, response.rotation.y * dt, response.rotation.z * dt));
  return isfinite(angle) ? angle : INFINITY;
}

__device__ inline void step_probe_rotation(uint cell, const ContactStepBindings& b,
                                           ContactStepParams p) {
  if (cell >= p.cell_count || step_gated(b) ||
      b.status[MS_STEP_SOLVE_STATUS] == MS_STEP_CONVERGED) {
    return;
  }
  step_raise(b, MS_STEP_PROBE_ROTATION, step_turn(b.response[cell], p.dt));
}

__device__ inline void step_probe_decide(uint index, const ContactStepBindings& b,
                                         ContactStepParams p) {
  if (index != 0u || step_gated(b) || b.status[MS_STEP_SOLVE_STATUS] == MS_STEP_CONVERGED) {
    return;
  }
  b.status[MS_STEP_ROTATION] = b.status[MS_STEP_PROBE_ROTATION];
  if (step_word_float(b, MS_STEP_PROBE_ROTATION) > p.max_rotation) {
    b.status[MS_STEP_FAILED] = MS_STEP_FAILED_ROTATION;
  }
}

// sin and cos of an angle in [0, 1/2] from + and * alone, so that every
// backend rounds them alike.
__device__ inline float step_sine(float x) {
  float s = x * x;
  float p = -2.50521084e-8f;
  p = 2.75573192e-6f + s * p;
  p = -1.98412698e-4f + s * p;
  p = 8.33333333e-3f + s * p;
  p = -1.66666667e-1f + s * p;
  return x + x * (s * p);
}

__device__ inline float step_cosine(float x) {
  float s = x * x;
  float p = 2.08767570e-9f;
  p = -2.75573192e-7f + s * p;
  p = 2.48015873e-5f + s * p;
  p = -1.38888889e-3f + s * p;
  p = 4.16666667e-2f + s * p;
  p = -0.5f + s * p;
  return 1.0f + s * p;
}

__device__ inline bool step_finite(float3 value) {
  return isfinite(value.x) && isfinite(value.y) && isfinite(value.z);
}

// Moves every cell for `dt` under the response into the trial geometry. A
// rotation larger than the limit, or a geometry that is not finite, makes the
// move invalid; the largest turn of a finite rotation is recorded either way.
__device__ inline void step_move(uint cell, const ContactStepBindings& b, ContactStepParams p) {
  if (cell >= p.cell_count || step_gated(b)) {
    return;
  }
  ContactResponse response = b.response[cell];
  float4 center = b.base_centers[cell];
  float4 axis = b.base_axes[cell];
  float3 rotation = make_float3(response.rotation.x * p.dt, response.rotation.y * p.dt,
                                response.rotation.z * p.dt);
  if (!step_finite(rotation)) {
    b.status[MS_STEP_MOVE_INVALID] = 1u;
    return;
  }
  float angle = length(rotation);
  step_raise(b, MS_STEP_MOVE_ROTATION, angle);
  if (angle > p.max_rotation) {
    b.status[MS_STEP_MOVE_INVALID] = 1u;
    return;
  }
  float3 position =
      (make_float3(center.x, center.y, center.z) +
       (make_float3(response.linear_length.x, response.linear_length.y, response.linear_length.z) *
        p.dt));
  float3 direction = make_float3(axis.x, axis.y, axis.z);
  if (angle > 1.0e-12f && p.max_rotation != 0.0f) {
    float turn = angle < p.max_rotation ? angle : p.max_rotation;
    float3 about = (rotation * (1.0f / angle));
    float cosine = step_cosine(turn);
    float sine = step_sine(turn);
    float3 turned = (((direction * cosine) + (cross(about, direction) * sine)) +
                     (about * (dot(about, direction) * (1.0f - cosine))));
    float magnitude = length(turned);
    if (!isfinite(magnitude) || !(magnitude > 0.0f)) {
      b.status[MS_STEP_MOVE_INVALID] = 1u;
      return;
    }
    direction = (turned * (1.0f / magnitude));
  }
  float length = center.w + response.linear_length.w * p.dt;
  if (!step_finite(position) || !step_finite(direction) || !isfinite(length) || length < 0.0f) {
    b.status[MS_STEP_MOVE_INVALID] = 1u;
    return;
  }
  b.trial_centers[cell] = make_float4(position.x, position.y, position.z, length);
  b.trial_axes[cell] = make_float4(direction.x, direction.y, direction.z, axis.w);
}

__device__ inline void step_move_check(uint index, const ContactStepBindings& b,
                                       ContactStepParams p) {
  (void)p;
  if (index != 0u || step_gated(b)) {
    return;
  }
  b.status[MS_STEP_ROTATION] = b.status[MS_STEP_MOVE_ROTATION];
  if (b.status[MS_STEP_MOVE_INVALID] != 0u) {
    b.status[MS_STEP_FAILED] = MS_STEP_FAILED_MOVE;
  }
}

// Flags each contact at the trial geometry that overlaps by more than the
// slack plus what its own force presses in.
__device__ inline void step_overlap_flags(uint index, const ContactStepBindings& b,
                                          ContactStepParams p) {
  if (index >= p.touch_capacity || step_gated(b)) {
    return;
  }
  if (index >= b.status[MS_STEP_FOUND]) {
    b.found_flags[index] = 0u;
    return;
  }
  ContactTouch touch = b.found[index];
  uint4 key = step_key(touch);
  float force = 0.0f;
  for (uint entry = b.keyed_offsets[key.x]; entry < b.keyed_offsets[key.x + 1u]; ++entry) {
    uint row = b.keyed_rows[entry];
    if (step_same_key(b.keys[row], key)) {
      force += b.multipliers[row];
    }
  }
  float depth = -touch.point.w;
  if (depth > 0.0f) {
    step_raise(b, MS_STEP_MAX_OVERLAP, depth);
  }
  float allowed = step_word_float(b, MS_STEP_SLACK) + force * step_compliance(touch, p);
  b.found_flags[index] = touch.point.w < -allowed ? 1u : 0u;
}

// The rows of the overlapping contacts go after the rows already solved.
__device__ inline void step_append_check(uint index, const ContactStepBindings& b,
                                         ContactStepParams p) {
  if (index != 0u || step_gated(b)) {
    return;
  }
  if (b.status[MS_STEP_ROWS] + b.status[MS_STEP_OVERLAPPING] > p.row_capacity) {
    b.status[MS_STEP_OVERFLOW] = MS_STEP_STAGE_APPEND;
  }
}

// Each new row is linearized where it was found, its offset set so that the
// multipliers that produced the move predict the gap the move left.
__device__ inline void step_append(uint index, const ContactStepBindings& b, ContactStepParams p) {
  if (index >= p.touch_capacity || step_gated(b) || index >= b.status[MS_STEP_FOUND] ||
      b.found_flags[index] == 0u) {
    return;
  }
  uint row = b.status[MS_STEP_ROWS] + b.found_ends[index] - 1u;
  ContactTouch touch = b.found[index];
  ContactRowData data = step_make_row(touch, b.trial_centers, b.trial_axes, p);
  float rate =
      contact_side_rate(data.first_linear, data.first_angular, b.response[data.first_slot]);
  if (data.second_slot != MS_STEP_NO_SECOND) {
    rate +=
        contact_side_rate(data.second_linear, data.second_angular, b.response[data.second_slot]);
  }
  data.first_angular.w -= p.dt * rate;
  b.rows[row] = data;
  b.keys[row] = step_key(touch);
  b.multipliers[row] = 0.0f;
}

__device__ inline void step_append_commit(uint index, const ContactStepBindings& b,
                                          ContactStepParams p) {
  (void)p;
  if (index != 0u || step_gated(b)) {
    return;
  }
  b.status[MS_STEP_ROWS] += b.status[MS_STEP_OVERLAPPING];
}

// Makes the trial geometry the base, keeps the response, and remembers each
// row's force for the next substep's warm start.
__device__ inline void step_accept(uint index, const ContactStepBindings& b, ContactStepParams p) {
  uint rows = b.status[MS_STEP_ROWS];
  if (index < p.cell_count) {
    b.base_centers[index] = b.trial_centers[index];
    b.base_axes[index] = b.trial_axes[index];
    b.accepted_response[index] = b.response[index];
  }
  if (index <= p.cell_count) {
    b.remembered_offsets[index] = b.keyed_offsets[index];
  }
  if (index < rows) {
    b.remembered_keys[index] = b.keys[index];
    b.remembered_multipliers[index] = b.multipliers[index];
    b.remembered_rows[index] = b.keyed_rows[index];
  }
}

__device__ inline void step_accept_commit(uint index, const ContactStepBindings& b,
                                          ContactStepParams p) {
  (void)p;
  if (index != 0u) {
    return;
  }
  b.status[MS_STEP_REMEMBERED_ROWS] = b.status[MS_STEP_ROWS];
}

// The inputs, output, and length of a scan, and where its total goes.
struct StepScan {
  uint* input;
  uint* output;
  uint count;
  uint total_word;
  uint capacity;
};

__device__ inline StepScan step_scan(const ContactStepBindings& b, ContactStepParams p) {
  StepScan scan;
  scan.total_word = MS_STEP_WORDS;
  scan.capacity = 0xffffffffu;
  if (p.scan == MS_STEP_SCAN_BUCKETS) {
    scan.input = b.bucket_counts;
    scan.output = b.bucket_ends;
    scan.count = p.bucket_count;
  } else if (p.scan == MS_STEP_SCAN_CANDIDATES) {
    scan.input = b.candidate_counts;
    scan.output = b.candidate_ends;
    scan.count = p.cell_count;
    scan.total_word = MS_STEP_CANDIDATES;
    scan.capacity = p.candidate_capacity;
  } else if (p.scan == MS_STEP_SCAN_PAIRS) {
    scan.input = b.pair_counts;
    scan.output = b.pair_ends;
    scan.count = step_min(b.status[MS_STEP_CANDIDATES], p.candidate_capacity);
    scan.total_word = MS_STEP_CELL_CONTACTS;
  } else if (p.scan == MS_STEP_SCAN_WALLS) {
    scan.input = b.wall_counts;
    scan.output = b.wall_ends;
    scan.count = p.cell_count * p.constraint_count;
    scan.total_word = MS_STEP_WALL_CONTACTS;
    scan.capacity = p.wall_capacity;
  } else if (p.scan == MS_STEP_SCAN_FOUND) {
    scan.input = b.found_flags;
    scan.output = b.found_ends;
    scan.count = b.status[MS_STEP_FOUND];
    scan.total_word = MS_STEP_OVERLAPPING;
  } else if (p.scan == MS_STEP_SCAN_KEYED) {
    scan.input = b.keyed_counts;
    scan.output = b.keyed_offsets + 1;
    scan.count = p.cell_count;
  } else {
    scan.input = b.incidence_counts;
    scan.output = b.incidence_offsets + 1;
    scan.count = p.cell_count;
  }
  return scan;
}

// Records the total of a finished scan, and an overflow when it passes the
// room its stage has.
__device__ inline void step_scan_total(const ContactStepBindings& b, ContactStepParams p,
                                       StepScan scan) {
  if (scan.total_word == MS_STEP_WORDS) {
    return;
  }
  uint total = scan.count == 0u ? 0u : scan.output[scan.count - 1u];
  b.status[scan.total_word] = total;
  if (total > scan.capacity) {
    b.status[MS_STEP_OVERFLOW] = p.stage;
  }
}

// The elements of `extent` with work, of the `dispatched` enqueued. The
// elements beyond them return at once.
__device__ inline uint step_live_count(uint extent, const ContactStepBindings& b,
                                       ContactStepParams p, uint dispatched) {
  uint live = dispatched;
  if (extent == MS_STEP_EXTENT_ONE) {
    live = 1u;
  } else if (extent == MS_STEP_EXTENT_CELLS) {
    live = p.cell_count;
  } else if (extent == MS_STEP_EXTENT_CANDIDATES) {
    live = step_min(b.status[MS_STEP_CANDIDATES], p.candidate_capacity);
  } else if (extent == MS_STEP_EXTENT_DETECTED) {
    live = b.status[MS_STEP_CELL_CONTACTS] + b.status[MS_STEP_WALL_CONTACTS];
  } else if (extent == MS_STEP_EXTENT_ROWS) {
    live = b.status[MS_STEP_ROWS];
  } else if (extent == MS_STEP_EXTENT_FOUND) {
    live = b.status[MS_STEP_FOUND];
  } else if (extent == MS_STEP_EXTENT_ACCEPT) {
    uint rows = b.status[MS_STEP_ROWS];
    live = rows > p.cell_count + 1u ? rows : p.cell_count + 1u;
  }
  return step_min(live, dispatched);
}

}  // namespace microsimulator::cuda::device
