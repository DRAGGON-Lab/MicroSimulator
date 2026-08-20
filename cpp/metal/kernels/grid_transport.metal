// Grid geometry and transport helpers shared by the signal and coupled-rate
// kernels. A Metal library is compiled from source at runtime, with no include
// path, so this fragment is concatenated ahead of each kernel source at build
// time rather than included by it.

#include <metal_stdlib>

using namespace metal;

struct GridShape {
  uint x;
  uint y;
  uint z;
  uint sites;
};

uint site_index(GridShape shape, uint x, uint y, uint z) {
  return x * shape.y * shape.z + y * shape.z + z;
}

float grid_level(device const float* levels, GridShape shape, uint signal, uint x, uint y, uint z) {
  return levels[signal * shape.sites + site_index(shape, x, y, z)];
}

float exterior_value(uint kind, device const float* fixed_values, uint face, uint signal,
                     uint signal_count, float current, float periodic) {
  if (kind == 0u) {
    return current;
  }
  if (kind == 1u) {
    return periodic;
  }
  return fixed_values[face * signal_count + signal];
}

// Whether each of a site's six faces is closed to transport, and the velocity
// it carries. A face is closed by a no-flux boundary, by a periodic boundary
// that wraps onto a solid site, or by a solid neighbour.
struct GridFaceState {
  bool3 closed_lower;
  bool3 closed_upper;
  float lower[3];
  float upper[3];
};

GridFaceState grid_face_state(GridShape shape, constant uint* boundary_kinds,
                              device const uchar* obstacles, device const float* x_faces,
                              device const float* y_faces, device const float* z_faces,
                              uint has_velocity_field, float4 advection, uint x, uint y, uint z) {
  GridFaceState faces;
  faces.closed_lower.x =
      x == 0u
          ? (boundary_kinds[0] == 0u ||
             (boundary_kinds[0] == 1u && obstacles[site_index(shape, shape.x - 1u, y, z)] != 0u))
          : obstacles[site_index(shape, x - 1u, y, z)] != 0u;
  faces.closed_upper.x =
      x + 1u == shape.x
          ? (boundary_kinds[1] == 0u ||
             (boundary_kinds[1] == 1u && obstacles[site_index(shape, 0u, y, z)] != 0u))
          : obstacles[site_index(shape, x + 1u, y, z)] != 0u;
  faces.closed_lower.y =
      y == 0u
          ? (boundary_kinds[2] == 0u ||
             (boundary_kinds[2] == 1u && obstacles[site_index(shape, x, shape.y - 1u, z)] != 0u))
          : obstacles[site_index(shape, x, y - 1u, z)] != 0u;
  faces.closed_upper.y =
      y + 1u == shape.y
          ? (boundary_kinds[3] == 0u ||
             (boundary_kinds[3] == 1u && obstacles[site_index(shape, x, 0u, z)] != 0u))
          : obstacles[site_index(shape, x, y + 1u, z)] != 0u;
  faces.closed_lower.z =
      z == 0u
          ? (boundary_kinds[4] == 0u ||
             (boundary_kinds[4] == 1u && obstacles[site_index(shape, x, y, shape.z - 1u)] != 0u))
          : obstacles[site_index(shape, x, y, z - 1u)] != 0u;
  faces.closed_upper.z =
      z + 1u == shape.z
          ? (boundary_kinds[5] == 0u ||
             (boundary_kinds[5] == 1u && obstacles[site_index(shape, x, y, 0u)] != 0u))
          : obstacles[site_index(shape, x, y, z + 1u)] != 0u;
  if (has_velocity_field != 0u) {
    faces.lower[0] = x_faces[x * shape.y * shape.z + y * shape.z + z];
    faces.upper[0] = x_faces[(x + 1u) * shape.y * shape.z + y * shape.z + z];
    faces.lower[1] = y_faces[x * (shape.y + 1u) * shape.z + y * shape.z + z];
    faces.upper[1] = y_faces[x * (shape.y + 1u) * shape.z + (y + 1u) * shape.z + z];
    faces.lower[2] = z_faces[x * shape.y * (shape.z + 1u) + y * (shape.z + 1u) + z];
    faces.upper[2] = z_faces[x * shape.y * (shape.z + 1u) + y * (shape.z + 1u) + z + 1u];
  } else {
    float3 velocity = advection.xyz;
    for (uint axis = 0; axis < 3u; ++axis) {
      faces.lower[axis] = velocity[axis];
      faces.upper[axis] = velocity[axis];
    }
  }
  return faces;
}
