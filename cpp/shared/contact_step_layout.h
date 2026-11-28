// The layout of a contact dynamics substep that the host and every backend's
// substep kernels share: the buffers of ContactStepBindings, the words of
// `status`, the parameters of a stage, and the stages themselves.
//
// The substep's state lives in the buffers of ContactStepBindings, and its
// scalars in the words of `status`. A stage that fails or runs out of room
// records it in status, and every later gated stage then does nothing, so the
// host enqueues a whole refinement round and reads status once.
//
// An includer defines MICROSIMULATOR_DEVICE_POINTER(type) as its spelling of
// a pointer to device memory. core/device_types.hpp and cuda/device_types.cuh
// include this file, and the Metal build concatenates it ahead of the
// substep's kernels.

// The buffers of a substep: X(name, element type).
#define MS_CONTACT_STEP_BUFFERS(X)       \
  X(base_centers, float4)                \
  X(base_axes, float4)                   \
  X(trial_centers, float4)               \
  X(trial_axes, float4)                  \
  X(cell_rates, float4)                  \
  X(fixed_cells, uchar)                  \
  X(ids, ulong)                          \
  X(bodies, ContactBodyData)             \
  X(drift, float)                        \
  X(response, ContactResponse)           \
  X(accepted_response, ContactResponse)  \
  X(solid, uchar)                        \
  X(x_faces, float)                      \
  X(y_faces, float)                      \
  X(z_faces, float)                      \
  X(constraints, ExternalConstraint)     \
  X(lower, float4)                       \
  X(upper, float4)                       \
  X(coordinates, uint4)                  \
  X(bucket_counts, uint)                 \
  X(bucket_ends, uint)                   \
  X(bucket_cursors, uint)                \
  X(members, uint)                       \
  X(candidate_counts, uint)              \
  X(candidate_ends, uint)                \
  X(candidates, uint2)                   \
  X(pair_counts, uint)                   \
  X(pair_ends, uint)                     \
  X(cell_records, CellContactRecord)     \
  X(wall_counts, uint)                   \
  X(wall_ends, uint)                     \
  X(wall_records, ExternalContactRecord) \
  X(touches, ContactTouch)               \
  X(found, ContactTouch)                 \
  X(found_flags, uint)                   \
  X(found_ends, uint)                    \
  X(rows, ContactRowData)                \
  X(keys, uint4)                         \
  X(multipliers, float)                  \
  X(multipliers_next, float)             \
  X(gaps, float)                         \
  X(gaps_next, float)                    \
  X(terms, float4)                       \
  X(keyed_counts, uint)                  \
  X(keyed_offsets, uint)                 \
  X(keyed_cursors, uint)                 \
  X(keyed_rows, uint)                    \
  X(incidence_counts, uint)              \
  X(incidence_offsets, uint)             \
  X(incidence_cursors, uint)             \
  X(incidence_entries, uint)             \
  X(remembered_keys, uint4)              \
  X(remembered_multipliers, float)       \
  X(remembered_offsets, uint)            \
  X(remembered_rows, uint)               \
  X(status, uint)                        \
  X(solve_state, ContactSolveState)

#define MS_STEP_BINDING_FIELD(name, type) MICROSIMULATOR_DEVICE_POINTER(type) name;
struct ContactStepBindings {
  MS_CONTACT_STEP_BUFFERS(MS_STEP_BINDING_FIELD)
};
#undef MS_STEP_BINDING_FIELD

// The words of `status`. Floating-point words hold the bits of a
// non-negative float, so an atomic maximum of the bits is the maximum of the
// values.
#define MS_STEP_FAILED 0u
#define MS_STEP_OVERFLOW 1u
#define MS_STEP_DRIFT_ERROR 2u
#define MS_STEP_SOLVE_ERROR 3u
#define MS_STEP_CANDIDATES 4u
#define MS_STEP_CELL_CONTACTS 5u
#define MS_STEP_WALL_CONTACTS 6u
#define MS_STEP_TOUCHES 7u
#define MS_STEP_FOUND 8u
#define MS_STEP_ROWS 9u
#define MS_STEP_OVERLAPPING 10u
#define MS_STEP_REMEMBERED_ROWS 11u
#define MS_STEP_WIDTH 12u
#define MS_STEP_STARTING_OVERLAP 15u
#define MS_STEP_BOUND 16u
#define MS_STEP_SLACK 17u
#define MS_STEP_FIRST_STEP 18u
#define MS_STEP_ITERATIONS 19u
#define MS_STEP_PROBE_ITERATIONS 20u
#define MS_STEP_SOLVE_STATUS 21u
#define MS_STEP_ROTATION 22u
#define MS_STEP_MOVE_INVALID 23u
#define MS_STEP_MAX_OVERLAP 24u
#define MS_STEP_GROWTH 25u
#define MS_STEP_DRIFT_SPEED 26u
#define MS_STEP_PROBE_ROTATION 27u
#define MS_STEP_MOVE_ROTATION 28u
#define MS_STEP_WORDS 32u

// Why an attempt at a substep failed: its solve did not converge, a cell
// turned too far for the attempt to be worth finishing, or the move was not a
// valid geometry.
#define MS_STEP_FAILED_SOLVE 1u
#define MS_STEP_FAILED_ROTATION 2u
#define MS_STEP_FAILED_MOVE 3u
// A solve whose rows have no positive diagonal.
#define MS_STEP_SOLVE_NO_DIAGONAL 1u
// The solve status of MS_STEP_SOLVE_STATUS.
#define MS_STEP_CONVERGED 0u
#define MS_STEP_ITERATION_LIMIT 1u
#define MS_STEP_BREAKDOWN 2u
// The stage that ran out of room.
#define MS_STEP_STAGE_DETECT_BASE 1u
#define MS_STEP_STAGE_APPEND 2u
#define MS_STEP_STAGE_DETECT_TRIAL 3u
// The scans of a substep.
#define MS_STEP_SCAN_BUCKETS 0u
#define MS_STEP_SCAN_CANDIDATES 1u
#define MS_STEP_SCAN_PAIRS 2u
#define MS_STEP_SCAN_WALLS 3u
#define MS_STEP_SCAN_FOUND 4u
#define MS_STEP_SCAN_KEYED 5u
#define MS_STEP_SCAN_INCIDENCE 6u

#define MS_STEP_NO_SECOND 0xffffffffu

// What a stage is told besides the buffers.
struct ContactStepParams {
  uint cell_count;
  uint constraint_count;
  uint bucket_count;
  // Zero for the base geometry, one for the trial geometry.
  uint geometry;
  // Zero to detect into `touches`, one into `found`.
  uint output;
  uint candidate_capacity;
  uint wall_capacity;
  uint touch_capacity;
  uint row_capacity;
  uint scan;
  uint stage;
  uint max_iterations;
  uint probe;
  uint has_drift;
  uint round;
  uint project;
  float margin;
  float parallel_sine_threshold;
  float contact_degeneracy;
  float constraint_degeneracy;
  float dt;
  float contact_compliance;
  float wall_compliance;
  float recovery;
  float tolerance;
  float max_rotation;
  float step_dt;
  float drag;
  ContactSolveData solve;
  FlowDriftData drift;
};

// The elements a stage has work for, which a backend that reads status
// between stages may launch in place of all that the host enqueues: one, the
// cells, the candidate pairs, the contacts of the detection being made, the
// rows, the contacts found at the trial geometry, the larger of the cells and
// rows, or every element enqueued.
#define MS_STEP_EXTENT_ONE 0u
#define MS_STEP_EXTENT_CELLS 1u
#define MS_STEP_EXTENT_CANDIDATES 2u
#define MS_STEP_EXTENT_DETECTED 3u
#define MS_STEP_EXTENT_ROWS 4u
#define MS_STEP_EXTENT_FOUND 5u
#define MS_STEP_EXTENT_ACCEPT 6u
#define MS_STEP_EXTENT_DISPATCHED 7u

// The kernels over elements, in the order a backend lists them, with the
// extent of each: X(name, extent).
#define MS_CONTACT_STEP_KERNELS(X)           \
  X(substep_begin, MS_STEP_EXTENT_ONE)       \
  X(round_begin, MS_STEP_EXTENT_ONE)         \
  X(drift, MS_STEP_EXTENT_CELLS)             \
  X(begin_bounds, MS_STEP_EXTENT_CELLS)      \
  X(bodies, MS_STEP_EXTENT_CELLS)            \
  X(detect_clear, MS_STEP_EXTENT_DISPATCHED) \
  X(bounds, MS_STEP_EXTENT_CELLS)            \
  X(width, MS_STEP_EXTENT_CELLS)             \
  X(place, MS_STEP_EXTENT_CELLS)             \
  X(grid_fill, MS_STEP_EXTENT_CELLS)         \
  X(candidate_count, MS_STEP_EXTENT_CELLS)   \
  X(candidate_fill, MS_STEP_EXTENT_CELLS)    \
  X(pair_count, MS_STEP_EXTENT_CANDIDATES)   \
  X(pair_fill, MS_STEP_EXTENT_CANDIDATES)    \
  X(wall_count, MS_STEP_EXTENT_DISPATCHED)   \
  X(wall_fill, MS_STEP_EXTENT_DISPATCHED)    \
  X(touches_total, MS_STEP_EXTENT_ONE)       \
  X(touches_fill, MS_STEP_EXTENT_DETECTED)   \
  X(rows_build, MS_STEP_EXTENT_DETECTED)     \
  X(rows_commit, MS_STEP_EXTENT_ONE)         \
  X(rows_clear, MS_STEP_EXTENT_DISPATCHED)   \
  X(keyed_count, MS_STEP_EXTENT_ROWS)        \
  X(keyed_fill, MS_STEP_EXTENT_ROWS)         \
  X(keyed_sort, MS_STEP_EXTENT_CELLS)        \
  X(warm_start, MS_STEP_EXTENT_ROWS)         \
  X(incidence_count, MS_STEP_EXTENT_ROWS)    \
  X(incidence_fill, MS_STEP_EXTENT_ROWS)     \
  X(incidence_sort, MS_STEP_EXTENT_CELLS)    \
  X(row_bound, MS_STEP_EXTENT_ROWS)          \
  X(solve_prepare, MS_STEP_EXTENT_ONE)       \
  X(solve_finish, MS_STEP_EXTENT_ONE)        \
  X(solve_copy, MS_STEP_EXTENT_ROWS)         \
  X(probe_rotation, MS_STEP_EXTENT_CELLS)    \
  X(probe_decide, MS_STEP_EXTENT_ONE)        \
  X(move, MS_STEP_EXTENT_CELLS)              \
  X(move_check, MS_STEP_EXTENT_ONE)          \
  X(overlap_flags, MS_STEP_EXTENT_FOUND)     \
  X(append_check, MS_STEP_EXTENT_ONE)        \
  X(append, MS_STEP_EXTENT_FOUND)            \
  X(append_commit, MS_STEP_EXTENT_ONE)       \
  X(accept, MS_STEP_EXTENT_ACCEPT)           \
  X(accept_commit, MS_STEP_EXTENT_ONE)
