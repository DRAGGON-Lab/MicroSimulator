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

// What a set of contact multipliers does to one cell: its velocity with the
// elongation rate in w, and its angular velocity with the axial load in w.
struct ContactResponse {
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
  // Velocity and angular velocity the cell has without contact.
  float4 drift_linear;
  float4 drift_angular;
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
  uint site_count;
  uint total_face_count;
  uint padding0;
  uint padding1;
};

// 1 / spacing^2 per axis as a double-word value.
struct DepthFlowScales {
  float4 inverse_square_high;
  float4 inverse_square_low;
};

// The lattice and kernel of a biomass deposit. A cell spreads its biomass
// over the sites of a box about its center, by the integrals over each site
// of a separable tent of radius `origin.w`.
struct DepositGrid {
  // Site counts along x, y, and z; w is the site count.
  uint4 shape;
  // Lower corner of the first site's center; w is the tent radius.
  float4 origin;
  // Site spacing; w is the volume of one site.
  float4 spacing;
  // The largest box extent along x, y, and z; w is the weights each cell
  // holds, their product.
  uint4 box;
  uint cell_count;
  uint channel_count;
  uint has_obstacles;
  uint padding;
};

// The sites one cell deposits over: the box from `lower` spanning `extent`.
// lower.w is one when the cell lies inside the grid and zero when it lies
// outside and deposits nothing; extent.w is the box's site count.
struct DepositBox {
  uint4 lower;
  uint4 extent;
};

// The closure of shallow flow over a deposit, one column at a time.
struct ShallowCouplingData {
  // Site counts along x, y, and z; w is the site count.
  uint4 shape;
  float drag_coefficient;
  float max_volume_fraction;
  // One when a single base mobility holds at every site.
  uint uniform_base;
  // One when the deposit carries the in-plane velocity of its biomass.
  uint has_velocity;
};

// The resolved-flow Brinkman drag of a deposit, one site at a time.
struct BrinkmanDragData {
  // Site counts along x, y, and z; w is the site count.
  uint4 shape;
  float drag_coefficient;
  float max_volume_fraction;
  uint padding0;
  uint padding1;
};
