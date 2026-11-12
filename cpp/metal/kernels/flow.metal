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
  threadgroup float values[flow_reduction_width];
  values[local_index] = index < count ? left[index] * right[index] : 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = flow_reduction_width / 2u; stride > 0; stride >>= 1) {
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
  if (i < count) target[i] = alpha * source[i] + beta * target[i];
}
