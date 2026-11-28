// Flow drift of rod cells in a face-staggered velocity field, one cell at a
// time.
//
// A free cell moves with the fluid velocity interpolated at its center and
// turns by a finite-aspect Jeffery closure: the fluid's spin plus the part of
// its strain along the rod, weighted by (a^2 - 1) / (a^2 + 1) for aspect
// ratio a. This treats the rod as an equivalent spheroid; it is not
// cell-resolved hydrodynamics. A fixed cell does not drift.
//
// A cell's center holds its length in w, and its axis its radius in w.

// The most substeps one cell's drift over a step may take.
constant uint drift_substep_limit = 100000u;

// The unit vector along `value`, or the zero vector when it has no
// direction.
inline float3 drift_normalized(float3 value) {
  const float magnitude = length(value);
  if (!(magnitude > 0.0f) || !(magnitude == magnitude) || magnitude * 0.0f != 0.0f) {
    return float3(0.0f, 0.0f, 0.0f);
  }
  return (value * (1.0f / magnitude));
}

inline bool drift_finite(float3 value) {
  return value.x * 0.0f == 0.0f && value.y * 0.0f == 0.0f && value.z * 0.0f == 0.0f;
}

// The interpolation of one axis: the lower site and the weight of the site
// above it, with the position drawn in to the lattice of site centers.
struct DriftAxis {
  uint lower;
  uint count;
  float fraction;
};

inline DriftAxis drift_axis(float position, float origin, float spacing, uint dimension) {
  DriftAxis axis;
  axis.lower = 0u;
  axis.count = 1u;
  axis.fraction = 0.0f;
  if (dimension == 1u) {
    return axis;
  }
  float coordinate = (position - origin) / spacing;
  coordinate = clamp(coordinate, 0.0f, float(dimension - 1u));
  axis.lower = uint(floor(coordinate));
  if (axis.lower >= dimension - 1u) {
    axis.lower = dimension - 1u;
    return axis;
  }
  axis.count = 2u;
  axis.fraction = coordinate - float(axis.lower);
  return axis;
}

// The fluid velocity at `point`, interpolated from the site-centered means of
// the face velocities. Where the grid has obstacles, only the face-connected
// fluid region of the strongest weight contributes, renormalized; a point
// surrounded by solid samples zero.
inline float3 drift_velocity(float3 point, FlowDriftData data, device const uchar* solid,
                             device const float* x_faces, device const float* y_faces,
                             device const float* z_faces) {
  const uint nx = data.shape.x;
  const uint ny = data.shape.y;
  const uint nz = data.shape.z;
  const DriftAxis axes[3] = {
      drift_axis(point.x, data.origin.x, data.spacing.x, nx),
      drift_axis(point.y, data.origin.y, data.spacing.y, ny),
      drift_axis(point.z, data.origin.z, data.spacing.z, nz),
  };
  uint coordinates[8][3];
  float weights[8];
  uint count = 0u;
  for (uint xi = 0u; xi < axes[0].count; ++xi) {
    const float wx = xi == 0u ? 1.0f - axes[0].fraction : axes[0].fraction;
    for (uint yi = 0u; yi < axes[1].count; ++yi) {
      const float wy = yi == 0u ? 1.0f - axes[1].fraction : axes[1].fraction;
      for (uint zi = 0u; zi < axes[2].count; ++zi) {
        const float wz = zi == 0u ? 1.0f - axes[2].fraction : axes[2].fraction;
        coordinates[count][0] = axes[0].lower + xi;
        coordinates[count][1] = axes[1].lower + yi;
        coordinates[count][2] = axes[2].lower + zi;
        weights[count] = wx * wy * wz;
        ++count;
      }
    }
  }
  if (data.shape.w != 0u) {
    uint seed = count;
    for (uint i = 0u; i < count; ++i) {
      const uint site = (coordinates[i][0] * ny + coordinates[i][1]) * nz + coordinates[i][2];
      if (solid[site] != 0u) {
        weights[i] = 0.0f;
      }
      if (weights[i] > 0.0f && (seed == count || weights[i] > weights[seed])) {
        seed = i;
      }
    }
    if (seed == count) {
      return float3(0.0f, 0.0f, 0.0f);
    }
    bool connected[8] = {false, false, false, false, false, false, false, false};
    connected[seed] = true;
    // The region grows by at least one site per pass until it is whole.
    bool grew = true;
    while (grew) {
      grew = false;
      for (uint i = 0u; i < count; ++i) {
        if (weights[i] <= 0.0f || connected[i]) {
          continue;
        }
        for (uint j = 0u; j < count; ++j) {
          uint distance = 0u;
          for (uint axis = 0u; axis < 3u; ++axis) {
            const uint a = coordinates[i][axis];
            const uint b = coordinates[j][axis];
            distance += a > b ? a - b : b - a;
          }
          if (connected[j] && distance == 1u) {
            connected[i] = true;
            grew = true;
            break;
          }
        }
      }
    }
    float total = 0.0f;
    for (uint i = 0u; i < count; ++i) {
      if (!connected[i]) {
        weights[i] = 0.0f;
      }
      total += weights[i];
    }
    for (uint i = 0u; i < count; ++i) {
      weights[i] = weights[i] / total;
    }
  }
  float3 result = float3(0.0f, 0.0f, 0.0f);
  for (uint i = 0u; i < count; ++i) {
    const float weight = weights[i];
    if (weight == 0.0f) {
      continue;
    }
    const uint x = coordinates[i][0];
    const uint y = coordinates[i][1];
    const uint z = coordinates[i][2];
    const uint x_lower = (x * ny + y) * nz + z;
    const uint x_upper = ((x + 1u) * ny + y) * nz + z;
    const uint y_lower = (x * (ny + 1u) + y) * nz + z;
    const uint y_upper = (x * (ny + 1u) + y + 1u) * nz + z;
    const uint z_lower = (x * ny + y) * (nz + 1u) + z;
    const uint z_upper = z_lower + 1u;
    result.x += weight * 0.5f * (x_faces[x_lower] + x_faces[x_upper]);
    result.y += weight * 0.5f * (y_faces[y_lower] + y_faces[y_upper]);
    result.z += weight * 0.5f * (z_faces[z_lower] + z_faces[z_upper]);
  }
  return result;
}

// The fluid velocity at `point`, and the rate at which a rod there along
// `axis` turns: the fluid's spin plus `lambda` times its strain along the
// rod. The velocity gradient is a central difference over one spacing.
struct DriftDerivative {
  float3 velocity;
  float3 orientation;
};

inline DriftDerivative drift_derivative(float3 point, float3 axis, float lambda, bool rotates,
                                        FlowDriftData data, device const uchar* solid,
                                        device const float* x_faces, device const float* y_faces,
                                        device const float* z_faces) {
  const float spacings[3] = {data.spacing.x, data.spacing.y, data.spacing.z};
  const uint dimensions[3] = {data.shape.x, data.shape.y, data.shape.z};
  float3 gradient[3] = {float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f),
                        float3(0.0f, 0.0f, 0.0f)};
  for (uint k = 0u; k < 3u; ++k) {
    if (dimensions[k] == 1u) {
      continue;
    }
    const float half_step = spacings[k] * 0.5f;
    const float3 offset =
        float3(k == 0u ? half_step : 0.0f, k == 1u ? half_step : 0.0f, k == 2u ? half_step : 0.0f);
    const float3 upper = drift_velocity((point + offset), data, solid, x_faces, y_faces, z_faces);
    const float3 lower = drift_velocity((point - offset), data, solid, x_faces, y_faces, z_faces);
    gradient[k] = ((upper - lower) * (1.0f / spacings[k]));
  }
  DriftDerivative result;
  result.velocity = drift_velocity(point, data, solid, x_faces, y_faces, z_faces);
  result.orientation = float3(0.0f, 0.0f, 0.0f);
  if (rotates) {
    const float3 along =
        (((gradient[0] * axis.x) + (gradient[1] * axis.y)) + (gradient[2] * axis.z));
    const float3 across =
        float3(dot(gradient[0], axis), dot(gradient[1], axis), dot(gradient[2], axis));
    const float3 strain = ((along + across) * 0.5f);
    const float3 spin = ((along - across) * 0.5f);
    result.orientation = (spin + ((strain - (axis * dot(axis, strain))) * lambda));
  }
  return result;
}

// The Jeffery shape factor of a rod of length L and radius r, whose
// equivalent spheroid has aspect ratio (L + 2r) / 2r.
inline float drift_shape_factor(float length, float radius) {
  const float aspect = (length + 2.0f * radius) / (2.0f * radius);
  return (aspect * aspect - 1.0f) / (aspect * aspect + 1.0f);
}

// The velocity then angular velocity of `cell` at the midpoint of its free
// motion over `data.dt`, six floats per cell in `rates`. Returns a status.
inline uint flow_drift_rates_cell(uint cell, FlowDriftData data, device const float4* centers,
                                  device const float4* axes, device const uchar* fixed_cells,
                                  device const uchar* solid, device const float* x_faces,
                                  device const float* y_faces, device const float* z_faces,
                                  device float* rates) {
  device float* out = rates + 6u * cell;
  for (uint index = 0u; index < 6u; ++index) {
    out[index] = 0.0f;
  }
  if (fixed_cells[cell] != 0u) {
    return drift_ok;
  }
  const float4 center = centers[cell];
  const float4 rod = axes[cell];
  const float3 position = float3(center.x, center.y, center.z);
  const float3 direction = float3(rod.x, rod.y, rod.z);
  const float lambda = drift_shape_factor(center.w, rod.w);
  const DriftDerivative start =
      drift_derivative(position, direction, lambda, true, data, solid, x_faces, y_faces, z_faces);
  const float half_dt = data.dt * 0.5f;
  const float3 midpoint = (position + (start.velocity * half_dt));
  const float3 mid_direction = drift_normalized((direction + (start.orientation * half_dt)));
  if (dot(mid_direction, mid_direction) == 0.0f) {
    return drift_degenerate_direction;
  }
  const DriftDerivative middle = drift_derivative(midpoint, mid_direction, lambda, true, data,
                                                  solid, x_faces, y_faces, z_faces);
  // A direction turning at rate d is a rotation at angular velocity p x d.
  const float3 angular = cross(direction, middle.orientation);
  out[0] = middle.velocity.x;
  out[1] = middle.velocity.y;
  out[2] = middle.velocity.z;
  out[3] = angular.x;
  out[4] = angular.y;
  out[5] = angular.z;
  return drift_ok;
}

// Carries `cell` with the flow over `data.dt` by midpoint substeps that move
// it no further than `data.spacing.w` and turn it no more than
// `data.origin.w`, and writes its new center and axis to `moved_centers` and
// `moved_axes`. Returns a status.
inline uint flow_drift_advance_cell(uint cell, FlowDriftData data, device const float4* centers,
                                    device const float4* axes, device const uchar* fixed_cells,
                                    device const uchar* solid, device const float* x_faces,
                                    device const float* y_faces, device const float* z_faces,
                                    device float4* moved_centers, device float4* moved_axes) {
  const float4 center = centers[cell];
  const float4 rod = axes[cell];
  moved_centers[cell] = center;
  moved_axes[cell] = rod;
  if (fixed_cells[cell] != 0u) {
    return drift_ok;
  }
  float3 position = float3(center.x, center.y, center.z);
  float3 direction = float3(rod.x, rod.y, rod.z);
  const float lambda = drift_shape_factor(center.w, rod.w);
  const float max_rotation = data.origin.w;
  const bool rotates = max_rotation != 0.0f;
  const float spatial_step = data.spacing.w;
  float remaining = data.dt;
  uint steps = 0u;
  while (remaining > 0.0f) {
    if (++steps > drift_substep_limit) {
      return drift_too_many_substeps;
    }
    const DriftDerivative start = drift_derivative(position, direction, lambda, rotates, data,
                                                   solid, x_faces, y_faces, z_faces);
    float step = remaining;
    const float speed = length(start.velocity);
    if (speed > 0.0f && spatial_step / speed < step) {
      step = spatial_step / speed;
    }
    const float turning = length(start.orientation);
    if (turning > 0.0f && max_rotation / turning < step) {
      step = max_rotation / turning;
    }
    DriftDerivative middle;
    while (true) {
      const float half_step = step * 0.5f;
      const float3 midpoint = (position + (start.velocity * half_step));
      const float3 mid_direction = drift_normalized((direction + (start.orientation * half_step)));
      if (dot(mid_direction, mid_direction) == 0.0f) {
        return drift_degenerate_direction;
      }
      middle = drift_derivative(midpoint, mid_direction, lambda, rotates, data, solid, x_faces,
                                y_faces, z_faces);
      if (step * length(middle.velocity) <= spatial_step * 1.001f &&
          (!rotates || step * length(middle.orientation) <= max_rotation * 1.001f)) {
        break;
      }
      step *= 0.5f;
      if (step <= 0.0f) {
        return drift_substep_underflow;
      }
    }
    if (!(step > 0.0f) || step * 0.0f != 0.0f) {
      return drift_invalid_substep;
    }
    position = (position + (middle.velocity * step));
    direction = drift_normalized((direction + (middle.orientation * step)));
    if (dot(direction, direction) == 0.0f) {
      return drift_degenerate_direction;
    }
    remaining = remaining - step > 0.0f ? remaining - step : 0.0f;
  }
  if (!drift_finite(position)) {
    return drift_non_finite;
  }
  device float4* moved_center = moved_centers + cell;
  moved_center->x = position.x;
  moved_center->y = position.y;
  moved_center->z = position.z;
  device float4* moved_axis = moved_axes + cell;
  moved_axis->x = direction.x;
  moved_axis->y = direction.y;
  moved_axis->z = direction.z;
  return drift_ok;
}

kernel void drift_rates(
    device const float4* centers [[buffer(0)]], device const float4* axes [[buffer(1)]],
    device const uchar* fixed_cells [[buffer(2)]], device const uchar* solid [[buffer(3)]],
    device const float* x_faces [[buffer(4)]], device const float* y_faces [[buffer(5)]],
    device const float* z_faces [[buffer(6)]], device float* rates [[buffer(7)]],
    device uint* status [[buffer(8)]], constant FlowDriftData& data [[buffer(9)]],
    uint cell [[thread_position_in_grid]]) {
  if (cell >= data.cell_count) {
    return;
  }
  status[cell] = flow_drift_rates_cell(cell, data, centers, axes, fixed_cells, solid, x_faces,
                                       y_faces, z_faces, rates);
}

kernel void drift_advance(
    device const float4* centers [[buffer(0)]], device const float4* axes [[buffer(1)]],
    device const uchar* fixed_cells [[buffer(2)]], device const uchar* solid [[buffer(3)]],
    device const float* x_faces [[buffer(4)]], device const float* y_faces [[buffer(5)]],
    device const float* z_faces [[buffer(6)]], device float4* moved_centers [[buffer(7)]],
    device float4* moved_axes [[buffer(8)]], device uint* status [[buffer(9)]],
    constant FlowDriftData& data [[buffer(10)]], uint cell [[thread_position_in_grid]]) {
  if (cell >= data.cell_count) {
    return;
  }
  status[cell] = flow_drift_advance_cell(cell, data, centers, axes, fixed_cells, solid, x_faces,
                                         y_faces, z_faces, moved_centers, moved_axes);
}
