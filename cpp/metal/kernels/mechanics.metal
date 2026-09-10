#include <metal_stdlib>

using namespace metal;

struct MechanicsDofs {
  float4 linear_length;
  float4 rotation;
};

MechanicsDofs zero_dofs() {
  MechanicsDofs result;
  result.linear_length = 0.0f;
  result.rotation = 0.0f;
  return result;
}

float dof_dot(const MechanicsDofs left, const MechanicsDofs right) {
  return dot(left.linear_length, right.linear_length) + dot(left.rotation, right.rotation);
}

MechanicsDofs scaled(const MechanicsDofs value, float scale) {
  MechanicsDofs result;
  result.linear_length = value.linear_length * scale;
  result.rotation = value.rotation * scale;
  return result;
}

MechanicsDofs added(const MechanicsDofs left, const MechanicsDofs right) {
  MechanicsDofs result;
  result.linear_length = left.linear_length + right.linear_length;
  result.rotation = left.rotation + right.rotation;
  return result;
}

MechanicsDofs contact_jacobian(float3 normal, float3 arm, float3 axis, float total_length,
                               float weight) {
  MechanicsDofs result;
  result.linear_length =
      float4(weight * normal, weight * dot(axis, arm) * dot(axis, normal) / total_length);
  result.rotation = float4(weight * cross(arm, normal), 0.0f);
  return result;
}

kernel void build_mechanics_rows(
    device const float4* centers [[buffer(0)]], device const float4* axes [[buffer(1)]],
    device const float4* geometry [[buffer(2)]], device const uint* first_slots [[buffer(3)]],
    device const uint* second_slots [[buffer(4)]], device const float4* points [[buffer(5)]],
    device const float4* normals [[buffer(6)]], device const float* separations [[buffer(7)]],
    device const float* weights [[buffer(8)]], device MechanicsDofs* first_rows [[buffer(9)]],
    device MechanicsDofs* second_rows [[buffer(10)]], device float* right_hand_side [[buffer(11)]],
    constant uint& contact_count [[buffer(12)]], uint index [[thread_position_in_grid]]) {
  if (index >= contact_count) {
    return;
  }
  uint first = first_slots[index];
  uint second = second_slots[index];
  float weight = weights[index];
  float3 normal = normals[index].xyz;
  float3 point = points[index].xyz;
  first_rows[index] = contact_jacobian(normal, point - centers[first].xyz, axes[first].xyz,
                                       geometry[first].x + 2.0f * geometry[first].y, weight);
  second_rows[index] =
      second == 0xffffffffu
          ? zero_dofs()
          : contact_jacobian(normal, point - centers[second].xyz, axes[second].xyz,
                             geometry[second].x + 2.0f * geometry[second].y, weight);
  right_hand_side[index] = weight * separations[index];
}

kernel void apply_mechanics_b(device const MechanicsDofs* first_rows [[buffer(0)]],
                              device const MechanicsDofs* second_rows [[buffer(1)]],
                              device const uint* first_slots [[buffer(2)]],
                              device const uint* second_slots [[buffer(3)]],
                              device const MechanicsDofs* input [[buffer(4)]],
                              device float* row_values [[buffer(5)]],
                              constant uint& contact_count [[buffer(6)]],
                              device const uchar* fixed [[buffer(7)]],
                              uint index [[thread_position_in_grid]]) {
  if (index >= contact_count) {
    return;
  }
  uint first = first_slots[index];
  uint second = second_slots[index];
  MechanicsDofs first_input = fixed[first] == 0 ? input[first] : zero_dofs();
  row_values[index] = dof_dot(first_rows[index], first_input);
  if (second != 0xffffffffu) {
    MechanicsDofs second_input = fixed[second] == 0 ? input[second] : zero_dofs();
    row_values[index] -= dof_dot(second_rows[index], second_input);
  }
}

kernel void apply_mechanics_transpose(device const MechanicsDofs* first_rows [[buffer(0)]],
                                      device const MechanicsDofs* second_rows [[buffer(1)]],
                                      device const float* row_values [[buffer(2)]],
                                      device const uint* incidence_offsets [[buffer(3)]],
                                      device const uint* incidence_indices [[buffer(4)]],
                                      device const uint* first_slots [[buffer(5)]],
                                      device MechanicsDofs* output [[buffer(6)]],
                                      constant uint& cell_count [[buffer(7)]],
                                      uint cell [[thread_position_in_grid]]) {
  if (cell >= cell_count) {
    return;
  }
  MechanicsDofs result = zero_dofs();
  for (uint offset = incidence_offsets[cell]; offset < incidence_offsets[cell + 1]; ++offset) {
    uint row = incidence_indices[offset];
    bool is_first = first_slots[row] == cell;
    MechanicsDofs jacobian = is_first ? first_rows[row] : second_rows[row];
    float sign = is_first ? 1.0f : -1.0f;
    result = added(result, scaled(jacobian, sign * row_values[row]));
  }
  output[cell] = result;
}

kernel void add_mechanics_regularizer(
    device const float4* axes [[buffer(0)]], device const float4* geometry [[buffer(1)]],
    device const MechanicsDofs* input [[buffer(2)]], device MechanicsDofs* output [[buffer(3)]],
    constant float4& parameters [[buffer(4)]], constant uint& cell_count [[buffer(5)]],
    device const uchar* fixed [[buffer(6)]], uint cell [[thread_position_in_grid]]) {
  if (cell >= cell_count) {
    return;
  }
  if (fixed[cell] != 0) {
    output[cell] = input[cell];
    return;
  }
  float mu_a = parameters.x;
  float gamma = parameters.y;
  float total_length = geometry[cell].x + 2.0f * geometry[cell].y;
  float radius = geometry[cell].y;
  float mass = mu_a * total_length;
  float axial_inertia = 0.5f * mass * radius * radius;
  float transverse_inertia = mass * (total_length * total_length + 3.0f * radius * radius) / 12.0f;
  float3 axis = axes[cell].xyz;
  float3 rotation = input[cell].rotation.xyz;
  float3 inertia_rotation = rotation * transverse_inertia +
                            axis * ((axial_inertia - transverse_inertia) * dot(axis, rotation));
  float regularization = 1.0f / gamma;

  MechanicsDofs result = output[cell];
  result.linear_length.xyz += regularization * mass * input[cell].linear_length.xyz;
  result.linear_length.w += input[cell].linear_length.w;
  result.rotation.xyz += regularization * inertia_rotation;
  output[cell] = result;
}

kernel void initialize_mechanics_vectors(device MechanicsDofs* right_hand_side [[buffer(0)]],
                                         device MechanicsDofs* solution [[buffer(1)]],
                                         device MechanicsDofs* residual [[buffer(2)]],
                                         device MechanicsDofs* search_direction [[buffer(3)]],
                                         constant uint& cell_count [[buffer(4)]],
                                         device const uchar* fixed [[buffer(5)]],
                                         uint cell [[thread_position_in_grid]]) {
  if (cell >= cell_count) {
    return;
  }
  solution[cell] = zero_dofs();
  MechanicsDofs projected_rhs = fixed[cell] == 0 ? right_hand_side[cell] : zero_dofs();
  right_hand_side[cell] = projected_rhs;
  residual[cell] = projected_rhs;
  search_direction[cell] = projected_rhs;
}

kernel void update_mechanics_solution_residual(
    device MechanicsDofs* solution [[buffer(0)]], device MechanicsDofs* residual [[buffer(1)]],
    device const MechanicsDofs* search_direction [[buffer(2)]],
    device const MechanicsDofs* applied [[buffer(3)]], constant float& alpha [[buffer(4)]],
    constant uint& cell_count [[buffer(5)]], uint cell [[thread_position_in_grid]]) {
  if (cell >= cell_count) {
    return;
  }
  solution[cell] = added(solution[cell], scaled(search_direction[cell], alpha));
  residual[cell] = added(residual[cell], scaled(applied[cell], -alpha));
}

kernel void update_mechanics_search_direction(device const MechanicsDofs* residual [[buffer(0)]],
                                              device MechanicsDofs* search_direction [[buffer(1)]],
                                              constant float& beta [[buffer(2)]],
                                              constant uint& cell_count [[buffer(3)]],
                                              uint cell [[thread_position_in_grid]]) {
  if (cell >= cell_count) {
    return;
  }
  search_direction[cell] = added(residual[cell], scaled(search_direction[cell], beta));
}

kernel void subtract_mechanics_vectors(device const MechanicsDofs* left [[buffer(0)]],
                                       device const MechanicsDofs* right [[buffer(1)]],
                                       device MechanicsDofs* output [[buffer(2)]],
                                       constant uint& cell_count [[buffer(3)]],
                                       uint cell [[thread_position_in_grid]]) {
  if (cell >= cell_count) {
    return;
  }
  output[cell] = added(left[cell], scaled(right[cell], -1.0f));
}

kernel void mechanics_dot_terms(device const MechanicsDofs* left [[buffer(0)]],
                                device const MechanicsDofs* right [[buffer(1)]],
                                device float* terms [[buffer(2)]],
                                constant uint& cell_count [[buffer(3)]],
                                uint cell [[thread_position_in_grid]]) {
  if (cell >= cell_count) {
    return;
  }
  terms[cell] = dof_dot(left[cell], right[cell]);
}

kernel void reduce_sum_pairs(device const float* input [[buffer(0)]],
                             device float* output [[buffer(1)]],
                             constant uint& element_count [[buffer(2)]],
                             uint index [[thread_position_in_grid]]) {
  uint first = index * 2;
  if (first >= element_count) {
    return;
  }
  float value = input[first];
  if (first + 1 < element_count) {
    value += input[first + 1];
  }
  output[index] = value;
}
