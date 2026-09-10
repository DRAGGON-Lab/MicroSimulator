#include "mechanics.cuh"

namespace cm::cuda {
namespace {

constexpr std::uint32_t threads_per_block = 256;

__device__ float3 add(float3 left, float3 right) {
  return make_float3(left.x + right.x, left.y + right.y, left.z + right.z);
}

__device__ float3 subtract(float3 left, float3 right) {
  return make_float3(left.x - right.x, left.y - right.y, left.z - right.z);
}

__device__ float3 multiply(float3 value, float scale) {
  return make_float3(value.x * scale, value.y * scale, value.z * scale);
}

__device__ float dot_product(float3 left, float3 right) {
  return left.x * right.x + left.y * right.y + left.z * right.z;
}

__device__ float3 cross_product(float3 left, float3 right) {
  return make_float3(left.y * right.z - left.z * right.y, left.z * right.x - left.x * right.z,
                     left.x * right.y - left.y * right.x);
}

__device__ MechanicsDofsGpu zero_dofs() {
  return {
      make_float4(0.0F, 0.0F, 0.0F, 0.0F),
      make_float4(0.0F, 0.0F, 0.0F, 0.0F),
  };
}

__device__ float dof_dot(const MechanicsDofsGpu& left, const MechanicsDofsGpu& right) {
  return left.linear_length.x * right.linear_length.x +
         left.linear_length.y * right.linear_length.y +
         left.linear_length.z * right.linear_length.z +
         left.linear_length.w * right.linear_length.w + left.rotation.x * right.rotation.x +
         left.rotation.y * right.rotation.y + left.rotation.z * right.rotation.z;
}

__device__ MechanicsDofsGpu scaled(const MechanicsDofsGpu& value, float scale) {
  return {
      make_float4(value.linear_length.x * scale, value.linear_length.y * scale,
                  value.linear_length.z * scale, value.linear_length.w * scale),
      make_float4(value.rotation.x * scale, value.rotation.y * scale, value.rotation.z * scale,
                  0.0F),
  };
}

__device__ MechanicsDofsGpu added(const MechanicsDofsGpu& left, const MechanicsDofsGpu& right) {
  return {
      make_float4(left.linear_length.x + right.linear_length.x,
                  left.linear_length.y + right.linear_length.y,
                  left.linear_length.z + right.linear_length.z,
                  left.linear_length.w + right.linear_length.w),
      make_float4(left.rotation.x + right.rotation.x, left.rotation.y + right.rotation.y,
                  left.rotation.z + right.rotation.z, 0.0F),
  };
}

__device__ MechanicsDofsGpu contact_jacobian(float3 normal, float3 arm, float3 axis,
                                             float total_length, float weight) {
  const auto angular = cross_product(arm, normal);
  return {
      make_float4(weight * normal.x, weight * normal.y, weight * normal.z,
                  weight * dot_product(axis, arm) * dot_product(axis, normal) / total_length),
      make_float4(weight * angular.x, weight * angular.y, weight * angular.z, 0.0F),
  };
}

__global__ void build_mechanics_rows(const float4* centers, const float4* axes,
                                     const float4* geometry, const std::uint32_t* first_slots,
                                     const std::uint32_t* second_slots, const float4* points,
                                     const float4* normals, const float* separations,
                                     const float* weights, MechanicsDofsGpu* first_rows,
                                     MechanicsDofsGpu* second_rows, float* right_hand_side,
                                     std::uint32_t contact_count) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= contact_count) {
    return;
  }
  const auto first = first_slots[index];
  const auto second = second_slots[index];
  const auto point = make_float3(points[index].x, points[index].y, points[index].z);
  const auto normal = make_float3(normals[index].x, normals[index].y, normals[index].z);
  const auto first_center = make_float3(centers[first].x, centers[first].y, centers[first].z);
  const auto first_axis = make_float3(axes[first].x, axes[first].y, axes[first].z);
  const auto weight = weights[index];
  first_rows[index] = contact_jacobian(normal, subtract(point, first_center), first_axis,
                                       geometry[first].x + 2.0F * geometry[first].y, weight);
  if (second == 0xffffffffU) {
    second_rows[index] = zero_dofs();
  } else {
    const auto second_center = make_float3(centers[second].x, centers[second].y, centers[second].z);
    const auto second_axis = make_float3(axes[second].x, axes[second].y, axes[second].z);
    second_rows[index] = contact_jacobian(normal, subtract(point, second_center), second_axis,
                                          geometry[second].x + 2.0F * geometry[second].y, weight);
  }
  right_hand_side[index] = weight * separations[index];
}

__global__ void apply_mechanics_b(const MechanicsDofsGpu* first_rows,
                                  const MechanicsDofsGpu* second_rows,
                                  const std::uint32_t* first_slots,
                                  const std::uint32_t* second_slots, const MechanicsDofsGpu* input,
                                  const std::uint8_t* fixed, float* row_values,
                                  std::uint32_t contact_count) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= contact_count) {
    return;
  }
  const auto first = first_slots[index];
  const auto second = second_slots[index];
  const auto first_input = fixed[first] == 0 ? input[first] : zero_dofs();
  row_values[index] = dof_dot(first_rows[index], first_input);
  if (second != 0xffffffffU) {
    const auto second_input = fixed[second] == 0 ? input[second] : zero_dofs();
    row_values[index] -= dof_dot(second_rows[index], second_input);
  }
}

__global__ void apply_mechanics_transpose(const MechanicsDofsGpu* first_rows,
                                          const MechanicsDofsGpu* second_rows,
                                          const float* row_values,
                                          const std::uint32_t* incidence_offsets,
                                          const std::uint32_t* incidence_indices,
                                          const std::uint32_t* first_slots,
                                          MechanicsDofsGpu* output, std::uint32_t cell_count) {
  const auto cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= cell_count) {
    return;
  }
  auto result = zero_dofs();
  for (auto offset = incidence_offsets[cell]; offset < incidence_offsets[cell + 1]; ++offset) {
    const auto row = incidence_indices[offset];
    const auto is_first = first_slots[row] == cell;
    const auto jacobian = is_first ? first_rows[row] : second_rows[row];
    result = added(result, scaled(jacobian, (is_first ? 1.0F : -1.0F) * row_values[row]));
  }
  output[cell] = result;
}

__global__ void add_mechanics_regularizer(const float4* axes, const float4* geometry,
                                          const MechanicsDofsGpu* input, MechanicsDofsGpu* output,
                                          const std::uint8_t* fixed, float mu_a, float gamma,
                                          std::uint32_t cell_count) {
  const auto cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= cell_count) {
    return;
  }
  if (fixed[cell] != 0) {
    output[cell] = input[cell];
    return;
  }
  const auto total_length = geometry[cell].x + 2.0F * geometry[cell].y;
  const auto radius = geometry[cell].y;
  const auto mass = mu_a * total_length;
  const auto axial_inertia = 0.5F * mass * radius * radius;
  const auto transverse_inertia =
      mass * (total_length * total_length + 3.0F * radius * radius) / 12.0F;
  const auto axis = make_float3(axes[cell].x, axes[cell].y, axes[cell].z);
  const auto rotation =
      make_float3(input[cell].rotation.x, input[cell].rotation.y, input[cell].rotation.z);
  const auto inertia_rotation =
      add(multiply(rotation, transverse_inertia),
          multiply(axis, (axial_inertia - transverse_inertia) * dot_product(axis, rotation)));
  const auto regularization = 1.0F / gamma;

  output[cell].linear_length.x += regularization * mass * input[cell].linear_length.x;
  output[cell].linear_length.y += regularization * mass * input[cell].linear_length.y;
  output[cell].linear_length.z += regularization * mass * input[cell].linear_length.z;
  output[cell].linear_length.w += input[cell].linear_length.w;
  output[cell].rotation.x += regularization * inertia_rotation.x;
  output[cell].rotation.y += regularization * inertia_rotation.y;
  output[cell].rotation.z += regularization * inertia_rotation.z;
}

__global__ void initialize_mechanics_vectors(MechanicsDofsGpu* right_hand_side,
                                             MechanicsDofsGpu* solution, MechanicsDofsGpu* residual,
                                             MechanicsDofsGpu* search_direction,
                                             const std::uint8_t* fixed, std::uint32_t cell_count) {
  const auto cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= cell_count) {
    return;
  }
  solution[cell] = zero_dofs();
  const auto projected_rhs = fixed[cell] == 0 ? right_hand_side[cell] : zero_dofs();
  right_hand_side[cell] = projected_rhs;
  residual[cell] = projected_rhs;
  search_direction[cell] = projected_rhs;
}

__global__ void update_mechanics_solution_residual(MechanicsDofsGpu* solution,
                                                   MechanicsDofsGpu* residual,
                                                   const MechanicsDofsGpu* search_direction,
                                                   const MechanicsDofsGpu* applied, float alpha,
                                                   std::uint32_t cell_count) {
  const auto cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= cell_count) {
    return;
  }
  solution[cell] = added(solution[cell], scaled(search_direction[cell], alpha));
  residual[cell] = added(residual[cell], scaled(applied[cell], -alpha));
}

__global__ void update_mechanics_search_direction(const MechanicsDofsGpu* residual,
                                                  MechanicsDofsGpu* search_direction, float beta,
                                                  std::uint32_t cell_count) {
  const auto cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= cell_count) {
    return;
  }
  search_direction[cell] = added(residual[cell], scaled(search_direction[cell], beta));
}

__global__ void subtract_mechanics_vectors(const MechanicsDofsGpu* left,
                                           const MechanicsDofsGpu* right, MechanicsDofsGpu* output,
                                           std::uint32_t cell_count) {
  const auto cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= cell_count) {
    return;
  }
  output[cell] = added(left[cell], scaled(right[cell], -1.0F));
}

__global__ void mechanics_dot_terms(const MechanicsDofsGpu* left, const MechanicsDofsGpu* right,
                                    float* terms, std::uint32_t cell_count) {
  const auto cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= cell_count) {
    return;
  }
  terms[cell] = dof_dot(left[cell], right[cell]);
}

__global__ void reduce_sum_pairs(const float* input, float* output, std::uint32_t element_count) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const auto first = index * 2;
  if (first >= element_count) {
    return;
  }
  auto value = input[first];
  if (first + 1 < element_count) {
    value += input[first + 1];
  }
  output[index] = value;
}

std::uint32_t block_count(std::uint32_t count) { return ((count - 1) / threads_per_block) + 1; }

}  // namespace

void launch_build_mechanics_rows(const float4* centers, const float4* axes, const float4* geometry,
                                 const std::uint32_t* first_slots,
                                 const std::uint32_t* second_slots, const float4* points,
                                 const float4* normals, const float* separations,
                                 const float* weights, MechanicsDofsGpu* first_rows,
                                 MechanicsDofsGpu* second_rows, float* right_hand_side,
                                 std::uint32_t contact_count, cudaStream_t stream) {
  build_mechanics_rows<<<block_count(contact_count), threads_per_block, 0, stream>>>(
      centers, axes, geometry, first_slots, second_slots, points, normals, separations, weights,
      first_rows, second_rows, right_hand_side, contact_count);
}

void launch_apply_mechanics_b(const MechanicsDofsGpu* first_rows,
                              const MechanicsDofsGpu* second_rows, const std::uint32_t* first_slots,
                              const std::uint32_t* second_slots, const MechanicsDofsGpu* input,
                              const std::uint8_t* fixed, float* row_values,
                              std::uint32_t contact_count, cudaStream_t stream) {
  apply_mechanics_b<<<block_count(contact_count), threads_per_block, 0, stream>>>(
      first_rows, second_rows, first_slots, second_slots, input, fixed, row_values, contact_count);
}

void launch_apply_mechanics_transpose(const MechanicsDofsGpu* first_rows,
                                      const MechanicsDofsGpu* second_rows, const float* row_values,
                                      const std::uint32_t* incidence_offsets,
                                      const std::uint32_t* incidence_indices,
                                      const std::uint32_t* first_slots, MechanicsDofsGpu* output,
                                      std::uint32_t cell_count, cudaStream_t stream) {
  apply_mechanics_transpose<<<block_count(cell_count), threads_per_block, 0, stream>>>(
      first_rows, second_rows, row_values, incidence_offsets, incidence_indices, first_slots,
      output, cell_count);
}

void launch_add_mechanics_regularizer(const float4* axes, const float4* geometry,
                                      const MechanicsDofsGpu* input, MechanicsDofsGpu* output,
                                      const std::uint8_t* fixed, float mu_a, float gamma,
                                      std::uint32_t cell_count, cudaStream_t stream) {
  add_mechanics_regularizer<<<block_count(cell_count), threads_per_block, 0, stream>>>(
      axes, geometry, input, output, fixed, mu_a, gamma, cell_count);
}

void launch_initialize_mechanics_vectors(MechanicsDofsGpu* right_hand_side,
                                         MechanicsDofsGpu* solution, MechanicsDofsGpu* residual,
                                         MechanicsDofsGpu* search_direction,
                                         const std::uint8_t* fixed, std::uint32_t cell_count,
                                         cudaStream_t stream) {
  initialize_mechanics_vectors<<<block_count(cell_count), threads_per_block, 0, stream>>>(
      right_hand_side, solution, residual, search_direction, fixed, cell_count);
}

void launch_update_mechanics_solution_residual(MechanicsDofsGpu* solution,
                                               MechanicsDofsGpu* residual,
                                               const MechanicsDofsGpu* search_direction,
                                               const MechanicsDofsGpu* applied, float alpha,
                                               std::uint32_t cell_count, cudaStream_t stream) {
  update_mechanics_solution_residual<<<block_count(cell_count), threads_per_block, 0, stream>>>(
      solution, residual, search_direction, applied, alpha, cell_count);
}

void launch_update_mechanics_search_direction(const MechanicsDofsGpu* residual,
                                              MechanicsDofsGpu* search_direction, float beta,
                                              std::uint32_t cell_count, cudaStream_t stream) {
  update_mechanics_search_direction<<<block_count(cell_count), threads_per_block, 0, stream>>>(
      residual, search_direction, beta, cell_count);
}

void launch_subtract_mechanics_vectors(const MechanicsDofsGpu* left, const MechanicsDofsGpu* right,
                                       MechanicsDofsGpu* output, std::uint32_t cell_count,
                                       cudaStream_t stream) {
  subtract_mechanics_vectors<<<block_count(cell_count), threads_per_block, 0, stream>>>(
      left, right, output, cell_count);
}

void launch_mechanics_dot_terms(const MechanicsDofsGpu* left, const MechanicsDofsGpu* right,
                                float* terms, std::uint32_t cell_count, cudaStream_t stream) {
  mechanics_dot_terms<<<block_count(cell_count), threads_per_block, 0, stream>>>(left, right, terms,
                                                                                 cell_count);
}

void launch_reduce_sum_pairs(const float* input, float* output, std::uint32_t element_count,
                             cudaStream_t stream) {
  const auto output_count = (element_count + 1) / 2;
  reduce_sum_pairs<<<block_count(output_count), threads_per_block, 0, stream>>>(input, output,
                                                                                element_count);
}

}  // namespace cm::cuda
