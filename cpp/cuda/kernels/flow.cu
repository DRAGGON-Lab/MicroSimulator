#include "flow.cuh"

namespace cm::cuda {
namespace {

constexpr std::uint32_t threads_per_block = 256;

struct Coordinate {
  std::uint32_t values[3];
};

struct FaceCoordinate {
  std::uint32_t component;
  Coordinate coordinate;
  Coordinate dimensions;
};

__device__ std::uint32_t site_index(const Coordinate& coordinate, const FlowGridParameters& grid) {
  return (coordinate.values[0] * grid.dimensions[1] + coordinate.values[1]) * grid.dimensions[2] +
         coordinate.values[2];
}

__device__ Coordinate site_coordinate(std::uint32_t index, const FlowGridParameters& grid) {
  Coordinate result{};
  result.values[2] = index % grid.dimensions[2];
  index /= grid.dimensions[2];
  result.values[1] = index % grid.dimensions[1];
  result.values[0] = index / grid.dimensions[1];
  return result;
}

__device__ FaceCoordinate face_coordinate(std::uint32_t index, const FlowGridParameters& grid) {
  const auto component =
      index < grid.face_offsets[1] ? 0U : (index < grid.face_offsets[2] ? 1U : 2U);
  auto local = index - grid.face_offsets[component];
  FaceCoordinate result{.component = component};
  for (std::uint32_t axis = 0; axis < 3; ++axis) {
    result.dimensions.values[axis] = grid.dimensions[axis] + (axis == component ? 1U : 0U);
  }
  result.coordinate.values[2] = local % result.dimensions.values[2];
  local /= result.dimensions.values[2];
  result.coordinate.values[1] = local % result.dimensions.values[1];
  result.coordinate.values[0] = local / result.dimensions.values[1];
  return result;
}

__device__ std::uint32_t face_index(std::uint32_t component, const Coordinate& coordinate,
                                    const FlowGridParameters& grid) {
  if (component == 0) {
    return grid.face_offsets[0] +
           (coordinate.values[0] * grid.dimensions[1] + coordinate.values[1]) * grid.dimensions[2] +
           coordinate.values[2];
  }
  if (component == 1) {
    return grid.face_offsets[1] +
           (coordinate.values[0] * (grid.dimensions[1] + 1) + coordinate.values[1]) *
               grid.dimensions[2] +
           coordinate.values[2];
  }
  return grid.face_offsets[2] +
         (coordinate.values[0] * grid.dimensions[1] + coordinate.values[1]) *
             (grid.dimensions[2] + 1) +
         coordinate.values[2];
}

__device__ float harmonic_mean(float first, float second) {
  const auto sum = first + second;
  return sum > 0.0F ? 2.0F * first * second / sum : 0.0F;
}

__global__ void depth_flow_operator(const float* input, const float* mobility,
                                    const float* diagonal, float* output, FlowGridParameters grid) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= grid.site_count) {
    return;
  }
  if (diagonal[index] == 0.0F) {
    output[index] = 0.0F;
    return;
  }
  const auto coordinate = site_coordinate(index, grid);
  auto result = diagonal[index] * input[index];
  for (std::uint32_t axis = 0; axis < 3; ++axis) {
    if (coordinate.values[axis] > 0) {
      auto neighbor = coordinate;
      --neighbor.values[axis];
      const auto neighbor_index = site_index(neighbor, grid);
      result += harmonic_mean(mobility[index], mobility[neighbor_index]) *
                grid.inverse_spacing_squared[axis] * (input[index] - input[neighbor_index]);
    }
    if (coordinate.values[axis] + 1 < grid.dimensions[axis]) {
      auto neighbor = coordinate;
      ++neighbor.values[axis];
      const auto neighbor_index = site_index(neighbor, grid);
      result += harmonic_mean(mobility[index], mobility[neighbor_index]) *
                grid.inverse_spacing_squared[axis] * (input[index] - input[neighbor_index]);
    }
  }
  output[index] = result;
}

__global__ void depth_flow_velocity(const float* pressure, const float* mobility, float* velocity,
                                    FlowGridParameters grid) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= grid.total_face_count) {
    return;
  }
  const auto face = face_coordinate(index, grid);
  const auto component = face.component;
  const auto has_lower = face.coordinate.values[component] > 0;
  const auto has_upper = face.coordinate.values[component] < grid.dimensions[component];
  auto lower_coordinate = face.coordinate;
  if (has_lower) {
    --lower_coordinate.values[component];
  }
  const auto lower = has_lower ? site_index(lower_coordinate, grid) : 0;
  const auto upper = has_upper ? site_index(face.coordinate, grid) : 0;
  auto value = 0.0F;
  if (has_lower && has_upper) {
    value = -harmonic_mean(mobility[lower], mobility[upper]) * (pressure[upper] - pressure[lower]) /
            grid.spacing[component];
  } else if (component == grid.flow_axis && has_upper) {
    value = 2.0F * mobility[upper] * (1.0F - pressure[upper]) / grid.spacing[component];
  } else if (component == grid.flow_axis && has_lower) {
    value = 2.0F * mobility[lower] * pressure[lower] / grid.spacing[component];
  }
  velocity[index] = value;
}

__global__ void resolved_flow_momentum(const float* input, const std::uint8_t* active,
                                       const std::uint8_t* exists, const float* face_drag,
                                       float* output, FlowGridParameters grid) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= grid.total_face_count) {
    return;
  }
  if (active[index] == 0) {
    output[index] = 0.0F;
    return;
  }
  const auto face = face_coordinate(index, grid);
  auto result = face_drag[index] * input[index];
  for (std::uint32_t axis = 0; axis < 3; ++axis) {
    if (grid.dimensions[axis] == 1) {
      continue;
    }
    for (int offset = -1; offset <= 1; offset += 2) {
      const auto in_bounds = offset < 0
                                 ? face.coordinate.values[axis] > 0
                                 : face.coordinate.values[axis] + 1 < face.dimensions.values[axis];
      std::uint32_t neighbor_index = 0;
      if (in_bounds) {
        auto coordinate = face.coordinate;
        if (offset < 0) {
          --coordinate.values[axis];
        } else {
          ++coordinate.values[axis];
        }
        neighbor_index = face_index(face.component, coordinate, grid);
      }
      float neighbor = 0.0F;
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

__global__ void resolved_flow_gradient(const float* pressure, const std::uint8_t* fluid,
                                       const std::uint8_t* active, float* gradient,
                                       FlowGridParameters grid) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= grid.total_face_count) {
    return;
  }
  if (active[index] == 0) {
    gradient[index] = 0.0F;
    return;
  }
  const auto face = face_coordinate(index, grid);
  const auto component = face.component;
  const auto has_lower = face.coordinate.values[component] > 0;
  const auto has_upper = face.coordinate.values[component] < grid.dimensions[component];
  auto lower_coordinate = face.coordinate;
  if (has_lower) {
    --lower_coordinate.values[component];
  }
  const auto lower = has_lower ? site_index(lower_coordinate, grid) : 0;
  const auto upper = has_upper ? site_index(face.coordinate, grid) : 0;
  const auto lower_value = has_lower && fluid[lower] != 0 ? pressure[lower] : 0.0F;
  const auto upper_value = has_upper && fluid[upper] != 0 ? pressure[upper] : 0.0F;
  gradient[index] = (upper_value - lower_value) / grid.spacing[component];
}

__global__ void resolved_flow_divergence(const float* velocity, const std::uint8_t* fluid,
                                         float* divergence, FlowGridParameters grid) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= grid.site_count) {
    return;
  }
  if (fluid[index] == 0) {
    divergence[index] = 0.0F;
    return;
  }
  const auto coordinate = site_coordinate(index, grid);
  auto result = 0.0F;
  for (std::uint32_t component = 0; component < 3; ++component) {
    auto upper = coordinate;
    ++upper.values[component];
    result += (velocity[face_index(component, upper, grid)] -
               velocity[face_index(component, coordinate, grid)]) /
              grid.spacing[component];
  }
  divergence[index] = result;
}

__global__ void flow_pcg_initialize(const float* right_hand_side, const float* diagonal,
                                    float* solution, float* residual, float* preconditioned,
                                    float* direction, std::uint32_t count) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const auto value = right_hand_side[index];
  const auto scaled = diagonal[index] > 0.0F ? value / diagonal[index] : 0.0F;
  solution[index] = 0.0F;
  residual[index] = value;
  preconditioned[index] = scaled;
  direction[index] = scaled;
}

__global__ void flow_pcg_update(float* solution, float* residual, const float* direction,
                                const float* transformed, float alpha, std::uint32_t count) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    solution[index] += alpha * direction[index];
    residual[index] -= alpha * transformed[index];
  }
}

__global__ void flow_pcg_precondition(const float* residual, const float* diagonal,
                                      float* preconditioned, std::uint32_t count) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    preconditioned[index] = diagonal[index] > 0.0F ? residual[index] / diagonal[index] : 0.0F;
  }
}

__global__ void flow_pcg_direction(const float* preconditioned, float* direction, float beta,
                                   std::uint32_t count) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    direction[index] = preconditioned[index] + beta * direction[index];
  }
}

__global__ void flow_vector_negate(const float* input, float* output, std::uint32_t count) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = -input[index];
  }
}

__global__ void flow_vector_combine(const float* source, float* target, float alpha, float beta,
                                    std::uint32_t count) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) target[i] = alpha * source[i] + beta * target[i];
}

__global__ void flow_vector_subtract(const float* left, const float* right, float* output,
                                     std::uint32_t count) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = left[index] - right[index];
  }
}

__global__ void flow_dot_partial(const float* left, const float* right, float* partials,
                                 std::uint32_t count) {
  __shared__ float values[flow_reduction_width];
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  values[threadIdx.x] = index < count ? left[index] * right[index] : 0.0F;
  __syncthreads();
  for (std::uint32_t stride = flow_reduction_width / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      values[threadIdx.x] += values[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = values[0];
  }
}

std::uint32_t block_count(std::uint32_t count, std::uint32_t width = threads_per_block) {
  return ((count - 1) / width) + 1;
}

}  // namespace

void launch_depth_flow_operator(const float* input, const float* mobility, const float* diagonal,
                                float* output, const FlowGridParameters& grid,
                                cudaStream_t stream) {
  depth_flow_operator<<<block_count(grid.site_count), threads_per_block, 0, stream>>>(
      input, mobility, diagonal, output, grid);
}

void launch_depth_flow_velocity(const float* pressure, const float* mobility, float* velocity,
                                const FlowGridParameters& grid, cudaStream_t stream) {
  depth_flow_velocity<<<block_count(grid.total_face_count), threads_per_block, 0, stream>>>(
      pressure, mobility, velocity, grid);
}

void launch_resolved_flow_momentum(const float* input, const std::uint8_t* active,
                                   const std::uint8_t* exists, const float* face_drag,
                                   float* output, const FlowGridParameters& grid,
                                   cudaStream_t stream) {
  resolved_flow_momentum<<<block_count(grid.total_face_count), threads_per_block, 0, stream>>>(
      input, active, exists, face_drag, output, grid);
}

void launch_resolved_flow_gradient(const float* pressure, const std::uint8_t* fluid,
                                   const std::uint8_t* active, float* gradient,
                                   const FlowGridParameters& grid, cudaStream_t stream) {
  resolved_flow_gradient<<<block_count(grid.total_face_count), threads_per_block, 0, stream>>>(
      pressure, fluid, active, gradient, grid);
}

void launch_resolved_flow_divergence(const float* velocity, const std::uint8_t* fluid,
                                     float* divergence, const FlowGridParameters& grid,
                                     cudaStream_t stream) {
  resolved_flow_divergence<<<block_count(grid.site_count), threads_per_block, 0, stream>>>(
      velocity, fluid, divergence, grid);
}

void launch_flow_pcg_initialize(const float* right_hand_side, const float* diagonal,
                                float* solution, float* residual, float* preconditioned,
                                float* direction, std::uint32_t count, cudaStream_t stream) {
  flow_pcg_initialize<<<block_count(count), threads_per_block, 0, stream>>>(
      right_hand_side, diagonal, solution, residual, preconditioned, direction, count);
}

void launch_flow_pcg_update(float* solution, float* residual, const float* direction,
                            const float* transformed, float alpha, std::uint32_t count,
                            cudaStream_t stream) {
  flow_pcg_update<<<block_count(count), threads_per_block, 0, stream>>>(
      solution, residual, direction, transformed, alpha, count);
}

void launch_flow_pcg_precondition(const float* residual, const float* diagonal,
                                  float* preconditioned, std::uint32_t count, cudaStream_t stream) {
  flow_pcg_precondition<<<block_count(count), threads_per_block, 0, stream>>>(
      residual, diagonal, preconditioned, count);
}

void launch_flow_pcg_direction(const float* preconditioned, float* direction, float beta,
                               std::uint32_t count, cudaStream_t stream) {
  flow_pcg_direction<<<block_count(count), threads_per_block, 0, stream>>>(preconditioned,
                                                                           direction, beta, count);
}

void launch_flow_vector_negate(const float* input, float* output, std::uint32_t count,
                               cudaStream_t stream) {
  flow_vector_negate<<<block_count(count), threads_per_block, 0, stream>>>(input, output, count);
}

void launch_flow_vector_combine(const float* source, float* target, float alpha, float beta,
                                std::uint32_t count, cudaStream_t stream) {
  flow_vector_combine<<<block_count(count), threads_per_block, 0, stream>>>(source, target, alpha,
                                                                            beta, count);
}

void launch_flow_vector_subtract(const float* left, const float* right, float* output,
                                 std::uint32_t count, cudaStream_t stream) {
  flow_vector_subtract<<<block_count(count), threads_per_block, 0, stream>>>(left, right, output,
                                                                             count);
}

void launch_flow_dot_partial(const float* left, const float* right, float* partials,
                             std::uint32_t count, cudaStream_t stream) {
  flow_dot_partial<<<block_count(count, flow_reduction_width), flow_reduction_width, 0, stream>>>(
      left, right, partials, count);
}

}  // namespace cm::cuda
