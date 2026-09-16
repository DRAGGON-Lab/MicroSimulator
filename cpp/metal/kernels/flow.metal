#include <metal_stdlib>

using namespace metal;

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

struct FaceCoordinate {
  uint component;
  uint3 coordinate;
  uint3 dimensions;
};

inline uint site_index(uint3 coordinate, constant FlowGridParameters& grid) {
  return (coordinate.x * grid.dimensions.y + coordinate.y) * grid.dimensions.z + coordinate.z;
}

inline uint3 site_coordinate(uint index, constant FlowGridParameters& grid) {
  const uint z = index % grid.dimensions.z;
  index /= grid.dimensions.z;
  const uint y = index % grid.dimensions.y;
  return uint3(index / grid.dimensions.y, y, z);
}

inline FaceCoordinate face_coordinate(uint index, constant FlowGridParameters& grid) {
  uint component = index < grid.face_offsets.y ? 0 : (index < grid.face_offsets.z ? 1 : 2);
  const uint offset = component == 0 ? grid.face_offsets.x
                                     : (component == 1 ? grid.face_offsets.y : grid.face_offsets.z);
  uint local = index - offset;
  uint3 dimensions = grid.dimensions.xyz;
  dimensions[component] += 1;
  const uint z = local % dimensions.z;
  local /= dimensions.z;
  const uint y = local % dimensions.y;
  return {component, uint3(local / dimensions.y, y, z), dimensions};
}

inline uint face_index(uint component, uint3 coordinate, constant FlowGridParameters& grid) {
  const uint offset = component == 0 ? grid.face_offsets.x
                                     : (component == 1 ? grid.face_offsets.y : grid.face_offsets.z);
  if (component == 0) {
    return offset + (coordinate.x * grid.dimensions.y + coordinate.y) * grid.dimensions.z +
           coordinate.z;
  }
  if (component == 1) {
    return offset + (coordinate.x * (grid.dimensions.y + 1) + coordinate.y) * grid.dimensions.z +
           coordinate.z;
  }
  return offset + (coordinate.x * grid.dimensions.y + coordinate.y) * (grid.dimensions.z + 1) +
         coordinate.z;
}

inline float harmonic_mean(float first, float second) {
  const float sum = first + second;
  return sum > 0.0f ? 2.0f * first * second / sum : 0.0f;
}

kernel void depth_flow_operator(device const float* input [[buffer(0)]],
                                device const float* mobility [[buffer(1)]],
                                device const float* diagonal [[buffer(2)]],
                                device float* output [[buffer(3)]],
                                constant FlowGridParameters& grid [[buffer(4)]],
                                uint index [[thread_position_in_grid]]) {
  if (index >= grid.site_count) {
    return;
  }
  if (diagonal[index] == 0.0f) {
    output[index] = 0.0f;
    return;
  }
  const uint3 coordinate = site_coordinate(index, grid);
  float result = 0.0f;
  const float boundary = 2.0f * mobility[index] * grid.inverse_spacing_squared[grid.flow_axis];
  if (coordinate[grid.flow_axis] == 0) result += boundary * input[index];
  if (coordinate[grid.flow_axis]+1 == grid.dimensions[grid.flow_axis]) result += boundary * input[index];
  for (uint axis = 0; axis < 3; ++axis) {
    if (coordinate[axis] > 0) {
      uint3 neighbor = coordinate;
      neighbor[axis] -= 1;
      const uint neighbor_index = site_index(neighbor, grid);
      result += harmonic_mean(mobility[index], mobility[neighbor_index]) *
                grid.inverse_spacing_squared[axis] * (input[index] - input[neighbor_index]);
    }
    if (coordinate[axis] + 1 < grid.dimensions[axis]) {
      uint3 neighbor = coordinate;
      neighbor[axis] += 1;
      const uint neighbor_index = site_index(neighbor, grid);
      result += harmonic_mean(mobility[index], mobility[neighbor_index]) *
                grid.inverse_spacing_squared[axis] * (input[index] - input[neighbor_index]);
    }
  }
  output[index] = result;
}

kernel void depth_flow_velocity(device const float* pressure [[buffer(0)]],
                                device const float* mobility [[buffer(1)]],
                                device float* velocity [[buffer(2)]],
                                constant FlowGridParameters& grid [[buffer(3)]],
                                uint index [[thread_position_in_grid]]) {
  if (index >= grid.total_face_count) {
    return;
  }
  const FaceCoordinate face = face_coordinate(index, grid);
  const uint component = face.component;
  const bool has_lower = face.coordinate[component] > 0;
  const bool has_upper = face.coordinate[component] < grid.dimensions[component];
  uint3 lower_coordinate = face.coordinate;
  if (has_lower) {
    lower_coordinate[component] -= 1;
  }
  const uint lower = has_lower ? site_index(lower_coordinate, grid) : 0;
  const uint upper = has_upper ? site_index(face.coordinate, grid) : 0;
  float value = 0.0f;
  if (has_lower && has_upper) {
    value = -harmonic_mean(mobility[lower], mobility[upper]) * (pressure[upper] - pressure[lower]) /
            grid.spacing[component];
  } else if (component == grid.flow_axis && has_upper) {
    value = 2.0f * mobility[upper] * (1.0f - pressure[upper]) / grid.spacing[component];
  } else if (component == grid.flow_axis && has_lower) {
    value = 2.0f * mobility[lower] * pressure[lower] / grid.spacing[component];
  }
  velocity[index] = value;
}

kernel void resolved_flow_momentum(device const float* input [[buffer(0)]],
                                   device const uchar* active [[buffer(1)]],
                                   device const uchar* exists [[buffer(2)]],
                                   device const float* face_drag [[buffer(3)]],
                                   device float* output [[buffer(4)]],
                                   constant FlowGridParameters& grid [[buffer(5)]],
                                   uint index [[thread_position_in_grid]]) {
  if (index >= grid.total_face_count) {
    return;
  }
  if (active[index] == 0) {
    output[index] = 0.0f;
    return;
  }
  const FaceCoordinate face = face_coordinate(index, grid);
  float result = face_drag[index] * input[index];
  for (uint axis = 0; axis < 3; ++axis) {
    if (grid.dimensions[axis] == 1) {
      continue;
    }
    for (int offset = -1; offset <= 1; offset += 2) {
      const bool in_bounds = offset < 0 ? face.coordinate[axis] > 0
                                        : face.coordinate[axis] + 1 < face.dimensions[axis];
      float neighbor = 0.0f;
      uint neighbor_index = 0;
      if (in_bounds) {
        uint3 coordinate = face.coordinate;
        if (offset < 0) {
          coordinate[axis] -= 1;
        } else {
          coordinate[axis] += 1;
        }
        neighbor_index = face_index(face.component, coordinate, grid);
      }
      if (axis == face.component) {
        neighbor = in_bounds ? input[neighbor_index] : input[index];
      } else if (in_bounds && exists[neighbor_index] != 0) {
        neighbor = input[neighbor_index];
      } else {
        neighbor = -input[index];
      }
      result -= (neighbor - input[index]) * grid.inverse_spacing_squared[axis];
    }
  }
  output[index] = result;
}

kernel void resolved_flow_gradient(device const float* pressure [[buffer(0)]],
                                   device const uchar* fluid [[buffer(1)]],
                                   device const uchar* active [[buffer(2)]],
                                   device float* gradient [[buffer(3)]],
                                   constant FlowGridParameters& grid [[buffer(4)]],
                                   uint index [[thread_position_in_grid]]) {
  if (index >= grid.total_face_count) {
    return;
  }
  if (active[index] == 0) {
    gradient[index] = 0.0f;
    return;
  }
  const FaceCoordinate face = face_coordinate(index, grid);
  const uint component = face.component;
  const bool has_lower = face.coordinate[component] > 0;
  const bool has_upper = face.coordinate[component] < grid.dimensions[component];
  uint3 lower_coordinate = face.coordinate;
  if (has_lower) {
    lower_coordinate[component] -= 1;
  }
  const uint lower = has_lower ? site_index(lower_coordinate, grid) : 0;
  const uint upper = has_upper ? site_index(face.coordinate, grid) : 0;
  const float lower_value = has_lower && fluid[lower] != 0 ? pressure[lower] : 0.0f;
  const float upper_value = has_upper && fluid[upper] != 0 ? pressure[upper] : 0.0f;
  gradient[index] = (upper_value - lower_value) / grid.spacing[component];
}

kernel void resolved_flow_divergence(device const float* velocity [[buffer(0)]],
                                     device const uchar* fluid [[buffer(1)]],
                                     device float* divergence [[buffer(2)]],
                                     constant FlowGridParameters& grid [[buffer(3)]],
                                     uint index [[thread_position_in_grid]]) {
  if (index >= grid.site_count) {
    return;
  }
  if (fluid[index] == 0) {
    divergence[index] = 0.0f;
    return;
  }
  const uint3 coordinate = site_coordinate(index, grid);
  float result = 0.0f;
  for (uint component = 0; component < 3; ++component) {
    uint3 upper = coordinate;
    upper[component] += 1;
    result += (velocity[face_index(component, upper, grid)] -
               velocity[face_index(component, coordinate, grid)]) /
              grid.spacing[component];
  }
  divergence[index] = result;
}

kernel void flow_pcg_initialize(
    device const float* right_hand_side [[buffer(0)]], device const float* diagonal [[buffer(1)]],
    device float* solution [[buffer(2)]], device float* residual [[buffer(3)]],
    device float* preconditioned [[buffer(4)]], device float* direction [[buffer(5)]],
    constant uint& count [[buffer(6)]], uint index [[thread_position_in_grid]]) {
  if (index >= count) {
    return;
  }
  const float value = right_hand_side[index];
  const float scaled = diagonal[index] > 0.0f ? value / diagonal[index] : 0.0f;
  solution[index] = 0.0f;
  residual[index] = value;
  preconditioned[index] = scaled;
  direction[index] = scaled;
}

kernel void flow_pcg_update(device float* solution [[buffer(0)]],
                            device float* residual [[buffer(1)]],
                            device const float* direction [[buffer(2)]],
                            device const float* transformed [[buffer(3)]],
                            constant float& alpha [[buffer(4)]], constant uint& count [[buffer(5)]],
                            uint index [[thread_position_in_grid]]) {
  if (index < count) {
    solution[index] += alpha * direction[index];
    residual[index] -= alpha * transformed[index];
  }
}

kernel void flow_pcg_precondition(device const float* residual [[buffer(0)]],
                                  device const float* diagonal [[buffer(1)]],
                                  device float* preconditioned [[buffer(2)]],
                                  constant uint& count [[buffer(3)]],
                                  uint index [[thread_position_in_grid]]) {
  if (index < count) {
    preconditioned[index] = diagonal[index] > 0.0f ? residual[index] / diagonal[index] : 0.0f;
  }
}

kernel void flow_pcg_direction(device const float* preconditioned [[buffer(0)]],
                               device float* direction [[buffer(1)]],
                               constant float& beta [[buffer(2)]],
                               constant uint& count [[buffer(3)]],
                               uint index [[thread_position_in_grid]]) {
  if (index < count) {
    direction[index] = preconditioned[index] + beta * direction[index];
  }
}

kernel void flow_vector_negate(device const float* input [[buffer(0)]],
                               device float* output [[buffer(1)]],
                               constant uint& count [[buffer(2)]],
                               uint index [[thread_position_in_grid]]) {
  if (index < count) {
    output[index] = -input[index];
  }
}

kernel void flow_vector_subtract(device const float* left [[buffer(0)]],
                                 device const float* right [[buffer(1)]],
                                 device float* output [[buffer(2)]],
                                 constant uint& count [[buffer(3)]],
                                 uint index [[thread_position_in_grid]]) {
  if (index < count) {
    output[index] = left[index] - right[index];
  }
}

kernel void flow_dot_partial(device const float* left [[buffer(0)]],
                             device const float* right [[buffer(1)]],
                             device float* partials [[buffer(2)]],
                             constant uint& count [[buffer(3)]],
                             uint index [[thread_position_in_grid]],
                             uint local_index [[thread_index_in_threadgroup]],
                             uint group_index [[threadgroup_position_in_grid]]) {
  threadgroup float values[64];
  values[local_index] = index < count ? left[index] * right[index] : 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 32; stride > 0; stride >>= 1) {
    if (local_index < stride) {
      values[local_index] += values[local_index + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (local_index == 0) {
    partials[group_index] = values[0];
  }
}

kernel void flow_vector_combine(device const float* source [[buffer(0)]],
                                device float* target [[buffer(1)]],
                                constant float& alpha [[buffer(2)]],
                                constant float& beta [[buffer(3)]],
                                constant uint& count [[buffer(4)]],
                                uint i [[thread_position_in_grid]]) {
  if (i < count) target[i] = alpha*source[i] + beta*target[i];
}
