// Structures passed between the host and the Metal shaders.
//
// This file is written in the subset common to C++ and the Metal Shading
// Language. core/device_types.hpp includes it for the host, with uint, ulong,
// float4, and uint4 defined to match the shader types, and the Metal build
// concatenates it ahead of every shader source, so each layout has one
// definition.

struct RateInstruction {
  uint operation;
  uint first;
  uint second;
  uint third;
  float value;
};

struct MechanicsDofs {
  float4 linear_length;
  float4 rotation;
};

// One row of the contact dynamics solve. The w components carry the row's
// scalars.
struct ContactRowData {
  // Force on the first cell per unit multiplier; w is its length lever.
  float4 first_linear;
  // Torque on the first cell per unit multiplier; w is the row offset.
  float4 first_angular;
  // Force on the second cell per unit multiplier; w is its length lever.
  float4 second_linear;
  // Torque on the second cell per unit multiplier; w is the row compliance.
  float4 second_angular;
  uint first_slot;
  uint second_slot;
  uint padding0;
  uint padding1;
};

struct ContactBodyData {
  // Cell axis; w is the translational mobility.
  float4 axis_mobility;
  // Transverse mobility, axial mobility, and nominal elongation rate.
  float4 rates;
};

struct ContactSolveData {
  float dt;
  float tolerance;
  uint law;
  float sensitivity;
  float free_load;
  float stall_load;
  uint row_count;
  uint body_count;
};

// Progress of the projected-gradient iteration, kept on the device so that a
// batch of steps runs without returning to the host.
struct ContactSolveState {
  float step_length;
  float residual;
  uint iterations;
  uint done;
};

struct ExternalConstraint {
  ulong id;
  uint kind;
  uint allowed_region;
  float4 geometry;
  float4 parameters;
};

struct GridShape {
  uint x;
  uint y;
  uint z;
  uint sites;
};

struct FlowGridParameters {
  uint4 dimensions;
  float4 spacing;
  float4 inverse_spacing_squared;
  uint4 face_offsets;
  uint4 face_counts;
  uint flow_axis;
  uint site_count;
  uint total_face_count;
  uint padding;
};

// 1 / spacing^2 per axis as a double-word value.
struct DepthFlowScales {
  float4 inverse_square_high;
  float4 inverse_square_low;
};
