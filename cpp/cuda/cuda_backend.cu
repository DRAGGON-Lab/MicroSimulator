#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "cm/backend.hpp"
#include "kernels/contacts.cuh"
#include "kernels/coupled_rates.cuh"
#include "kernels/growth.cuh"
#include "kernels/mechanics.cuh"
#include "kernels/signals.cuh"
#include "kernels/species.cuh"

namespace cm {
namespace {

void check_cuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

template <typename T>
class CudaBuffer {
 public:
  CudaBuffer() = default;
  CudaBuffer(const CudaBuffer&) = delete;
  CudaBuffer& operator=(const CudaBuffer&) = delete;

  ~CudaBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  void reserve(std::size_t count, const char* description) {
    if (count <= capacity_) {
      return;
    }
    const auto new_capacity = std::bit_ceil(count);
    if (new_capacity > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::overflow_error(std::string("CUDA buffer size overflow for ") + description);
    }

    T* replacement = nullptr;
    const auto byte_count = new_capacity * sizeof(T);
    const auto allocation_operation = std::string("failed to allocate CUDA ") + description;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&replacement), byte_count),
               allocation_operation.c_str());
    if (data_ != nullptr) {
      const auto release_result = cudaFree(data_);
      if (release_result != cudaSuccess) {
        cudaFree(replacement);
        const auto release_operation = std::string("failed to release old CUDA ") + description;
        check_cuda(release_result, release_operation.c_str());
      }
    }
    data_ = replacement;
    capacity_ = new_capacity;
  }

  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }

 private:
  T* data_{nullptr};
  std::size_t capacity_{0};
};

class CudaBackend final : public ComputeBackend {
 public:
  explicit CudaBackend(std::uint32_t device_index) {
    if (device_index > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
      throw std::out_of_range("CUDA device index exceeds the runtime index space");
    }
    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "failed to enumerate CUDA devices");
    if (device_index >= static_cast<std::uint32_t>(device_count)) {
      throw std::out_of_range("CUDA device index is unavailable");
    }
    device_index_ = static_cast<int>(device_index);
    check_cuda(cudaSetDevice(device_index_), "failed to select the CUDA device");
    check_cuda(cudaGetDeviceProperties(&device_properties_, device_index_),
               "failed to inspect the CUDA device");
    check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
               "failed to create a CUDA stream");
  }

  ~CudaBackend() override {
    static_cast<void>(cudaSetDevice(device_index_));
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }

  [[nodiscard]] BackendInfo info() const override {
    return {
        .kind = BackendKind::cuda,
        .name = "cuda",
        .device = device_properties_.name,
        .device_index = static_cast<std::uint32_t>(device_index_),
        .native = true,
    };
  }

  [[nodiscard]] bool supports(BackendFeature feature) const noexcept override {
    return feature == BackendFeature::growth || feature == BackendFeature::species ||
           feature == BackendFeature::cell_contacts ||
           feature == BackendFeature::external_constraints ||
           feature == BackendFeature::cell_mechanics || feature == BackendFeature::signals ||
           feature == BackendFeature::coupled_rates;
  }

  void advance_growth(WorldState& state, float dt) override {
    activate_device();
    auto view = state.growth_state();
    if (view.lengths.empty()) {
      return;
    }
    if (view.lengths.size() > std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("CUDA growth launch exceeds the uint32 index space");
    }
    lengths_.reserve(view.lengths.size(), "growth lengths");
    growth_rates_.reserve(view.growth_rates.size(), "growth rates");

    const auto byte_count = view.lengths.size_bytes();
    check_cuda(cudaMemcpyAsync(lengths_.data(), view.lengths.data(), byte_count,
                               cudaMemcpyHostToDevice, stream_),
               "failed to upload CUDA growth lengths");
    check_cuda(cudaMemcpyAsync(growth_rates_.data(), view.growth_rates.data(), byte_count,
                               cudaMemcpyHostToDevice, stream_),
               "failed to upload CUDA growth rates");

    cuda::launch_growth(lengths_.data(), growth_rates_.data(), dt,
                        static_cast<std::uint32_t>(view.lengths.size()), stream_);
    check_cuda(cudaGetLastError(), "failed to launch the CUDA growth kernel");
    check_cuda(cudaMemcpyAsync(view.lengths.data(), lengths_.data(), byte_count,
                               cudaMemcpyDeviceToHost, stream_),
               "failed to download CUDA growth lengths");
    check_cuda(cudaStreamSynchronize(stream_), "CUDA growth execution failed");
  }

  void advance_species(WorldState& state, const SpeciesRatePlan& plan,
                       std::span<const float> previous_lengths, float dt) override {
    activate_device();
    if (!std::isfinite(dt) || dt < 0.0F) {
      throw std::invalid_argument("species time step must be finite and non-negative");
    }
    state.validate();
    plan.validate();
    if (plan.species_count() != state.species_count()) {
      throw std::invalid_argument("species rate plan and world state species counts disagree");
    }
    if (previous_lengths.size() != state.size()) {
      throw std::invalid_argument("previous cell lengths and world state cell counts disagree");
    }
    if (state.empty() || state.species_count() == 0) {
      return;
    }
    if (state.size() > std::numeric_limits<std::uint32_t>::max() ||
        state.species_count() > std::numeric_limits<std::uint32_t>::max() ||
        plan.instructions().size() > std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("CUDA species launch exceeds the uint32 index space");
    }
    if (!std::ranges::all_of(previous_lengths,
                             [](float value) { return std::isfinite(value) && value >= 0.0F; })) {
      throw std::invalid_argument("previous cell lengths must be finite and non-negative");
    }
    if (state.size() > std::numeric_limits<std::size_t>::max() / state.species_count() ||
        state.size() > std::numeric_limits<std::size_t>::max() / plan.instructions().size()) {
      throw std::overflow_error("CUDA species buffer size overflow");
    }

    const auto level_count = state.size() * state.species_count();
    const auto workspace_count = state.size() * plan.instructions().size();
    if (level_count > std::numeric_limits<std::uint32_t>::max() ||
        workspace_count > std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("CUDA flattened species storage exceeds the uint32 index space");
    }
    if (level_count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        workspace_count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        plan.instructions().size() >
            std::numeric_limits<std::size_t>::max() / sizeof(cuda::RateInstructionGpu)) {
      throw std::overflow_error("CUDA species allocation size overflow");
    }
    ensure_species_capacity(state.size(), level_count, plan.instructions().size(),
                            state.species_count(), workspace_count);

    const auto geometry = state.geometry_state();
    const auto attributes = state.cell_attributes();
    auto species_state = state.species_state();
    std::vector<float4> centers(state.size());
    std::vector<float4> shapes(state.size());
    for (std::size_t index = 0; index < state.size(); ++index) {
      centers[index] = make_float4(geometry.position_x[index], geometry.position_y[index],
                                   geometry.position_z[index], 0.0F);
      shapes[index] = make_float4(geometry.lengths[index], geometry.radii[index], 0.0F, 0.0F);
    }
    std::vector<cuda::RateInstructionGpu> instructions;
    instructions.reserve(plan.instructions().size());
    for (const auto& instruction : plan.instructions()) {
      instructions.push_back({
          .operation = static_cast<std::uint32_t>(instruction.operation),
          .first = instruction.first,
          .second = instruction.second,
          .third = instruction.third,
          .value = instruction.value,
      });
    }
    const std::vector<float> level_values(species_state.levels.begin(), species_state.levels.end());
    const std::vector<float> previous_values(previous_lengths.begin(), previous_lengths.end());
    const std::vector<float> growth_values(attributes.growth_rates.begin(),
                                           attributes.growth_rates.end());
    const std::vector<std::int32_t> cell_type_values(attributes.cell_types.begin(),
                                                     attributes.cell_types.end());
    const std::vector<std::uint32_t> output_values(plan.outputs().begin(), plan.outputs().end());

    copy_to_device(species_levels_, level_values, "failed to upload CUDA species levels");
    copy_to_device(species_previous_lengths_, previous_values,
                   "failed to upload CUDA previous cell lengths");
    copy_to_device(species_centers_, centers, "failed to upload CUDA species cell centers");
    copy_to_device(species_geometry_, shapes, "failed to upload CUDA species cell geometry");
    copy_to_device(species_growth_rates_, growth_values,
                   "failed to upload CUDA species growth rates");
    copy_to_device(species_cell_types_, cell_type_values,
                   "failed to upload CUDA species cell types");
    copy_to_device(species_instructions_, instructions,
                   "failed to upload CUDA species rate instructions");
    copy_to_device(species_outputs_, output_values, "failed to upload CUDA species rate outputs");
    check_cuda(cudaMemsetAsync(species_error_.data(), 0, sizeof(std::uint32_t), stream_),
               "failed to clear the CUDA species error flag");

    cuda::launch_advance_species(
        species_levels_.data(), species_previous_lengths_.data(), species_centers_.data(),
        species_geometry_.data(), species_growth_rates_.data(), species_cell_types_.data(),
        species_instructions_.data(), species_outputs_.data(), species_workspace_.data(),
        species_error_.data(), dt, static_cast<std::uint32_t>(state.species_count()),
        static_cast<std::uint32_t>(plan.instructions().size()),
        static_cast<std::uint32_t>(state.size()), stream_);
    check_cuda(cudaGetLastError(), "failed to launch the CUDA species kernel");

    std::uint32_t error = 0;
    check_cuda(cudaMemcpyAsync(&error, species_error_.data(), sizeof(error), cudaMemcpyDeviceToHost,
                               stream_),
               "failed to download the CUDA species error flag");
    check_cuda(cudaStreamSynchronize(stream_), "CUDA species execution failed");
    if (error != 0) {
      throw std::domain_error("CUDA species kernel produced a non-finite value");
    }
    check_cuda(cudaMemcpyAsync(species_state.levels.data(), species_levels_.data(),
                               species_state.levels.size_bytes(), cudaMemcpyDeviceToHost, stream_),
               "failed to download CUDA species levels");
    check_cuda(cudaStreamSynchronize(stream_), "CUDA species download failed");
  }

  SignalSolveReport advance_signal_grid(SignalGrid& grid, float dt) override {
    activate_device();
    grid.validate();
    grid.validate_step(dt);
    if (dt == 0.0F) {
      return {};
    }
    const auto& spec = grid.spec();
    const auto level_view = grid.levels();
    const std::vector<float> levels(level_view.begin(), level_view.end());
    const auto signal_count = spec.signal_count;
    const auto level_count = static_cast<std::uint32_t>(levels.size());

    signal_levels_.reserve(levels.size(), "signal-grid levels");
    signal_output_.reserve(levels.size(), "signal-grid output");
    signal_diffusion_.reserve(signal_count, "signal-grid diffusion");
    signal_advection_.reserve(signal_count, "signal-grid advection");
    signal_fixed_values_.reserve(static_cast<std::size_t>(6) * signal_count,
                                 "signal-grid boundary values");
    signal_reaction_source_.reserve(levels.size(), "signal-grid affine sources");
    signal_reaction_loss_.reserve(levels.size(), "signal-grid affine losses");
    signal_obstacles_.reserve(spec.site_count(), "signal-grid obstacles");
    signal_x_faces_.reserve(std::max<std::size_t>(spec.x_face_count(), 1), "signal x faces");
    signal_y_faces_.reserve(std::max<std::size_t>(spec.y_face_count(), 1), "signal y faces");
    signal_z_faces_.reserve(std::max<std::size_t>(spec.z_face_count(), 1), "signal z faces");
    signal_error_.reserve(1, "signal-grid error flag");

    std::vector<float4> advection;
    advection.reserve(signal_count);
    for (const auto velocity : spec.advection) {
      advection.push_back(make_float4(velocity.x, velocity.y, velocity.z, 0.0F));
    }
    const std::array<const GridBoundary*, 6> boundary_records{
        &spec.x_lower, &spec.x_upper, &spec.y_lower, &spec.y_upper, &spec.z_lower, &spec.z_upper,
    };
    std::vector<float> fixed_values(static_cast<std::size_t>(6) * signal_count, 0.0F);
    for (std::size_t face = 0; face < boundary_records.size(); ++face) {
      if (boundary_records[face]->kind == GridBoundaryKind::fixed) {
        std::copy(boundary_records[face]->values.begin(), boundary_records[face]->values.end(),
                  fixed_values.begin() + static_cast<std::ptrdiff_t>(face * signal_count));
      }
    }
    std::vector<float> reaction_source(levels.size(), 0.0F);
    std::vector<float> reaction_loss(levels.size(), 0.0F);
    if (spec.reaction.has_value()) {
      reaction_source = spec.reaction->source_rates;
      reaction_loss = spec.reaction->loss_rates;
    }
    std::vector<std::uint8_t> obstacles(spec.site_count(), 0);
    if (spec.has_obstacles()) {
      obstacles = spec.obstacles;
    }

    copy_to_device(signal_levels_, levels, "failed to upload CUDA signal-grid levels");
    copy_to_device(signal_diffusion_, spec.diffusion,
                   "failed to upload CUDA signal-grid diffusion");
    copy_to_device(signal_advection_, advection, "failed to upload CUDA signal-grid advection");
    copy_to_device(signal_fixed_values_, fixed_values,
                   "failed to upload CUDA signal-grid boundary values");
    copy_to_device(signal_reaction_source_, reaction_source,
                   "failed to upload CUDA signal-grid affine sources");
    copy_to_device(signal_reaction_loss_, reaction_loss,
                   "failed to upload CUDA signal-grid affine losses");
    copy_to_device(signal_obstacles_, obstacles, "failed to upload CUDA signal-grid obstacles");
    if (spec.velocity_field.has_value()) {
      copy_to_device(signal_x_faces_, spec.velocity_field->x_faces,
                     "failed to upload CUDA signal x faces");
      copy_to_device(signal_y_faces_, spec.velocity_field->y_faces,
                     "failed to upload CUDA signal y faces");
      copy_to_device(signal_z_faces_, spec.velocity_field->z_faces,
                     "failed to upload CUDA signal z faces");
    }
    const auto has_velocity_field =
        static_cast<std::uint32_t>(spec.velocity_field.has_value());
    check_cuda(cudaMemsetAsync(signal_error_.data(), 0, sizeof(std::uint32_t), stream_),
               "failed to clear the CUDA signal-grid error flag");

    const cuda::SignalGridBoundariesGpu boundaries{
        .x_lower = static_cast<std::uint32_t>(spec.x_lower.kind),
        .x_upper = static_cast<std::uint32_t>(spec.x_upper.kind),
        .y_lower = static_cast<std::uint32_t>(spec.y_lower.kind),
        .y_upper = static_cast<std::uint32_t>(spec.y_upper.kind),
        .z_lower = static_cast<std::uint32_t>(spec.z_lower.kind),
        .z_upper = static_cast<std::uint32_t>(spec.z_upper.kind),
    };
    const cuda::SignalGridShapeGpu shape{
        .x = spec.shape.x,
        .y = spec.shape.y,
        .z = spec.shape.z,
        .sites = static_cast<std::uint32_t>(spec.site_count()),
    };
    const auto crank_nicolson = spec.integration == SignalIntegrationKind::crank_nicolson;
    cuda::launch_advance_signal_grid(
        signal_levels_.data(), signal_output_.data(), signal_diffusion_.data(),
        signal_advection_.data(), signal_fixed_values_.data(), signal_reaction_source_.data(),
        signal_reaction_loss_.data(), signal_obstacles_.data(), signal_x_faces_.data(),
        signal_y_faces_.data(), signal_z_faces_.data(), has_velocity_field, signal_error_.data(),
        boundaries, shape,
        make_float4(spec.spacing.x, spec.spacing.y, spec.spacing.z, 0.0F), dt, signal_count,
        level_count, crank_nicolson, stream_);
    check_cuda(cudaGetLastError(), "failed to launch the CUDA signal-grid kernel");

    std::uint32_t error = 0;
    check_cuda(cudaMemcpyAsync(&error, signal_error_.data(), sizeof(error), cudaMemcpyDeviceToHost,
                               stream_),
               "failed to download the CUDA signal-grid error flag");
    check_cuda(cudaStreamSynchronize(stream_), "CUDA signal-grid execution failed");
    if (error != 0) {
      throw std::domain_error(
          "CUDA signal-grid kernel produced a non-finite or negative concentration");
    }
    const float* result_device = signal_output_.data();
    SignalSolveReport report;
    if (crank_nicolson) {
      const auto solve = solve_signal_crank_nicolson(
          signal_levels_.data(), signal_output_.data(), signal_diffusion_.data(),
          signal_advection_.data(), signal_fixed_values_.data(), signal_reaction_source_.data(),
          signal_reaction_loss_.data(), signal_obstacles_.data(), signal_x_faces_.data(),
          signal_y_faces_.data(), signal_z_faces_.data(), has_velocity_field,
          signal_error_.data(), boundaries, shape,
          make_float4(spec.spacing.x, spec.spacing.y, spec.spacing.z, 0.0F), dt, signal_count,
          level_count, spec.solver);
      result_device = solve.first;
      report = solve.second;
      if (!report.converged) {
        throw std::runtime_error("CUDA Crank-Nicolson signal solve did not converge after " +
                                 std::to_string(report.iterations) + " iterations");
      }
    }
    std::vector<float> output(levels.size());
    check_cuda(cudaMemcpyAsync(output.data(), result_device, output.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_),
               "failed to download CUDA signal-grid levels");
    check_cuda(cudaStreamSynchronize(stream_), "CUDA signal-grid download failed");
    grid.replace_levels(std::move(output));
    return report;
  }

  SignalSolveReport advance_coupled(WorldState& state, SignalGrid& grid,
                                    const CoupledRatePlan& plan,
                                    std::span<const float> previous_lengths, float dt) override {
    activate_device();
    validate_coupled_step(state, grid, plan, previous_lengths, dt);
    const auto checked_product = [](std::size_t left, std::size_t right, const char* name) {
      if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::overflow_error(std::string("CUDA coupled ") + name + " size overflow");
      }
      return left * right;
    };
    const auto cell_count_size = state.size();
    const auto species_count_size = state.species_count();
    const auto signal_count_size = plan.signal_count();
    const auto instruction_count_size = plan.instructions().size();
    const auto species_level_count =
        checked_product(cell_count_size, species_count_size, "species level");
    const auto workspace_count =
        checked_product(cell_count_size, instruction_count_size, "workspace");
    const auto cell_signal_count =
        checked_product(cell_count_size, signal_count_size, "cell signal");
    const auto grid_level_count = grid.levels().size();
    const auto& spec = grid.spec();
    for (const auto count :
         {cell_count_size, species_count_size, signal_count_size, instruction_count_size,
          species_level_count, workspace_count, cell_signal_count, grid_level_count}) {
      if (count > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("CUDA coupled launch exceeds the uint32 index space");
      }
    }

    coupled_species_levels_.reserve(species_level_count, "coupled species levels");
    coupled_previous_lengths_.reserve(cell_count_size, "coupled previous lengths");
    coupled_centers_.reserve(cell_count_size, "coupled cell centers");
    coupled_geometry_.reserve(cell_count_size, "coupled cell geometry");
    coupled_growth_rates_.reserve(cell_count_size, "coupled growth rates");
    coupled_cell_types_.reserve(cell_count_size, "coupled cell types");
    coupled_instructions_.reserve(instruction_count_size, "coupled rate instructions");
    coupled_species_outputs_.reserve(species_count_size, "coupled species outputs");
    coupled_signal_outputs_.reserve(signal_count_size, "coupled signal outputs");
    coupled_workspace_.reserve(workspace_count, "coupled rate workspace");
    coupled_cell_signal_rates_.reserve(cell_signal_count, "coupled cell signal rates");
    coupled_grid_levels_.reserve(grid_level_count, "coupled grid levels");
    coupled_grid_output_.reserve(grid_level_count, "coupled grid output");
    coupled_diffusion_.reserve(signal_count_size, "coupled diffusion");
    coupled_advection_.reserve(signal_count_size, "coupled advection");
    coupled_fixed_values_.reserve(6 * signal_count_size, "coupled boundary values");
    coupled_reaction_source_.reserve(grid_level_count, "coupled affine sources");
    coupled_reaction_loss_.reserve(grid_level_count, "coupled affine losses");
    coupled_obstacles_.reserve(spec.site_count(), "coupled grid obstacles");
    coupled_x_faces_.reserve(std::max<std::size_t>(spec.x_face_count(), 1), "coupled x faces");
    coupled_y_faces_.reserve(std::max<std::size_t>(spec.y_face_count(), 1), "coupled y faces");
    coupled_z_faces_.reserve(std::max<std::size_t>(spec.z_face_count(), 1), "coupled z faces");
    coupled_error_.reserve(1, "coupled error flag");

    const auto geometry = state.geometry_state();
    const auto attributes = state.cell_attributes();
    auto species_state = state.species_state();
    std::vector<float4> centers(cell_count_size);
    std::vector<float4> shapes(cell_count_size);
    for (std::size_t index = 0; index < cell_count_size; ++index) {
      centers[index] = make_float4(geometry.position_x[index], geometry.position_y[index],
                                   geometry.position_z[index], 0.0F);
      shapes[index] = make_float4(geometry.lengths[index], geometry.radii[index], 0.0F, 0.0F);
    }
    std::vector<cuda::RateInstructionGpu> instructions;
    instructions.reserve(instruction_count_size);
    for (const auto& instruction : plan.instructions()) {
      instructions.push_back({
          .operation = static_cast<std::uint32_t>(instruction.operation),
          .first = instruction.first,
          .second = instruction.second,
          .third = instruction.third,
          .value = instruction.value,
      });
    }
    std::vector<float4> advection;
    advection.reserve(signal_count_size);
    for (const auto velocity : spec.advection) {
      advection.push_back(make_float4(velocity.x, velocity.y, velocity.z, 0.0F));
    }
    const std::array<const GridBoundary*, 6> boundary_records{
        &spec.x_lower, &spec.x_upper, &spec.y_lower, &spec.y_upper, &spec.z_lower, &spec.z_upper,
    };
    std::vector<float> fixed_values(6 * signal_count_size, 0.0F);
    for (std::size_t face = 0; face < boundary_records.size(); ++face) {
      if (boundary_records[face]->kind == GridBoundaryKind::fixed) {
        std::copy(boundary_records[face]->values.begin(), boundary_records[face]->values.end(),
                  fixed_values.begin() + static_cast<std::ptrdiff_t>(face * signal_count_size));
      }
    }
    std::vector<float> reaction_source(grid_level_count, 0.0F);
    std::vector<float> reaction_loss(grid_level_count, 0.0F);
    if (spec.reaction.has_value()) {
      reaction_source = spec.reaction->source_rates;
      reaction_loss = spec.reaction->loss_rates;
    }
    const std::vector<float> species_levels(species_state.levels.begin(),
                                            species_state.levels.end());
    const std::vector<float> previous_values(previous_lengths.begin(), previous_lengths.end());
    const std::vector<float> growth_values(attributes.growth_rates.begin(),
                                           attributes.growth_rates.end());
    const std::vector<std::int32_t> cell_type_values(attributes.cell_types.begin(),
                                                     attributes.cell_types.end());
    const std::vector<std::uint32_t> species_outputs(plan.species_outputs().begin(),
                                                     plan.species_outputs().end());
    const std::vector<std::uint32_t> signal_outputs(plan.signal_outputs().begin(),
                                                    plan.signal_outputs().end());
    const std::vector<float> grid_levels(grid.levels().begin(), grid.levels().end());

    if (!species_levels.empty()) {
      copy_to_device(coupled_species_levels_, species_levels,
                     "failed to upload CUDA coupled species levels");
    }
    if (!previous_values.empty()) {
      copy_to_device(coupled_previous_lengths_, previous_values,
                     "failed to upload CUDA coupled previous lengths");
      copy_to_device(coupled_centers_, centers, "failed to upload CUDA coupled cell centers");
      copy_to_device(coupled_geometry_, shapes, "failed to upload CUDA coupled cell geometry");
      copy_to_device(coupled_growth_rates_, growth_values,
                     "failed to upload CUDA coupled growth rates");
      copy_to_device(coupled_cell_types_, cell_type_values,
                     "failed to upload CUDA coupled cell types");
    }
    copy_to_device(coupled_instructions_, instructions,
                   "failed to upload CUDA coupled instructions");
    if (!species_outputs.empty()) {
      copy_to_device(coupled_species_outputs_, species_outputs,
                     "failed to upload CUDA coupled species outputs");
    }
    copy_to_device(coupled_signal_outputs_, signal_outputs,
                   "failed to upload CUDA coupled signal outputs");
    copy_to_device(coupled_grid_levels_, grid_levels, "failed to upload CUDA coupled grid levels");
    copy_to_device(coupled_diffusion_, spec.diffusion, "failed to upload CUDA coupled diffusion");
    copy_to_device(coupled_advection_, advection, "failed to upload CUDA coupled advection");
    copy_to_device(coupled_fixed_values_, fixed_values,
                   "failed to upload CUDA coupled boundary values");
    copy_to_device(coupled_reaction_source_, reaction_source,
                   "failed to upload CUDA coupled affine sources");
    copy_to_device(coupled_reaction_loss_, reaction_loss,
                   "failed to upload CUDA coupled affine losses");
    {
      std::vector<std::uint8_t> obstacles(spec.site_count(), 0);
      if (spec.has_obstacles()) {
        obstacles = spec.obstacles;
      }
      copy_to_device(coupled_obstacles_, obstacles, "failed to upload CUDA coupled obstacles");
    }
    if (spec.velocity_field.has_value()) {
      copy_to_device(coupled_x_faces_, spec.velocity_field->x_faces,
                     "failed to upload CUDA coupled x faces");
      copy_to_device(coupled_y_faces_, spec.velocity_field->y_faces,
                     "failed to upload CUDA coupled y faces");
      copy_to_device(coupled_z_faces_, spec.velocity_field->z_faces,
                     "failed to upload CUDA coupled z faces");
    }
    const auto has_velocity_field =
        static_cast<std::uint32_t>(spec.velocity_field.has_value());
    check_cuda(cudaMemsetAsync(coupled_error_.data(), 0, sizeof(std::uint32_t), stream_),
               "failed to clear the CUDA coupled error flag");

    const cuda::SignalGridBoundariesGpu boundaries{
        .x_lower = static_cast<std::uint32_t>(spec.x_lower.kind),
        .x_upper = static_cast<std::uint32_t>(spec.x_upper.kind),
        .y_lower = static_cast<std::uint32_t>(spec.y_lower.kind),
        .y_upper = static_cast<std::uint32_t>(spec.y_upper.kind),
        .z_lower = static_cast<std::uint32_t>(spec.z_lower.kind),
        .z_upper = static_cast<std::uint32_t>(spec.z_upper.kind),
    };
    const cuda::SignalGridShapeGpu shape{
        .x = spec.shape.x,
        .y = spec.shape.y,
        .z = spec.shape.z,
        .sites = static_cast<std::uint32_t>(spec.site_count()),
    };
    const auto crank_nicolson = spec.integration == SignalIntegrationKind::crank_nicolson;
    check_cuda(
        cuda::launch_advance_coupled(
            coupled_species_levels_.data(), coupled_previous_lengths_.data(),
            coupled_centers_.data(), coupled_geometry_.data(), coupled_growth_rates_.data(),
            coupled_cell_types_.data(), coupled_instructions_.data(),
            coupled_species_outputs_.data(), coupled_signal_outputs_.data(),
            coupled_workspace_.data(), coupled_grid_levels_.data(), coupled_grid_output_.data(),
            coupled_diffusion_.data(), coupled_advection_.data(), coupled_fixed_values_.data(),
            coupled_reaction_source_.data(), coupled_reaction_loss_.data(),
            coupled_obstacles_.data(), coupled_x_faces_.data(), coupled_y_faces_.data(),
            coupled_z_faces_.data(), has_velocity_field, coupled_cell_signal_rates_.data(),
            coupled_error_.data(), boundaries, shape,
            make_float4(spec.origin.x, spec.origin.y, spec.origin.z, 0.0F),
            make_float4(spec.spacing.x, spec.spacing.y, spec.spacing.z, 0.0F), dt,
            static_cast<std::uint32_t>(species_count_size),
            static_cast<std::uint32_t>(signal_count_size),
            static_cast<std::uint32_t>(instruction_count_size),
            static_cast<std::uint32_t>(cell_count_size),
            static_cast<std::uint32_t>(grid_level_count), crank_nicolson, stream_),
        "failed to launch the CUDA coupled kernels");

    std::uint32_t error = 0;
    check_cuda(cudaMemcpyAsync(&error, coupled_error_.data(), sizeof(error), cudaMemcpyDeviceToHost,
                               stream_),
               "failed to download the CUDA coupled error flag");
    check_cuda(cudaStreamSynchronize(stream_), "CUDA coupled execution failed");
    if (error != 0) {
      throw std::domain_error("CUDA coupled kernels produced an invalid value");
    }

    const float* result_device = coupled_grid_output_.data();
    SignalSolveReport report;
    if (crank_nicolson) {
      const auto solve = solve_signal_crank_nicolson(
          coupled_grid_levels_.data(), coupled_grid_output_.data(), coupled_diffusion_.data(),
          coupled_advection_.data(), coupled_fixed_values_.data(), coupled_reaction_source_.data(),
          coupled_reaction_loss_.data(), coupled_obstacles_.data(), coupled_x_faces_.data(),
          coupled_y_faces_.data(), coupled_z_faces_.data(), has_velocity_field,
          coupled_error_.data(), boundaries, shape,
          make_float4(spec.spacing.x, spec.spacing.y, spec.spacing.z, 0.0F), dt,
          static_cast<std::uint32_t>(signal_count_size),
          static_cast<std::uint32_t>(grid_level_count), spec.solver);
      result_device = solve.first;
      report = solve.second;
      if (!report.converged) {
        throw std::runtime_error(
            "CUDA Crank-Nicolson coupled signal solve did not converge after " +
            std::to_string(report.iterations) + " iterations");
      }
    }

    std::vector<float> next_species(species_level_count);
    std::vector<float> next_grid(grid_level_count);
    if (!next_species.empty()) {
      copy_to_host(next_species, coupled_species_levels_,
                   "failed to download CUDA coupled species levels");
    }
    check_cuda(cudaMemcpyAsync(next_grid.data(), result_device, next_grid.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_),
               "failed to download CUDA coupled grid levels");
    check_cuda(cudaStreamSynchronize(stream_), "CUDA coupled download failed");
    SignalGridCheckpoint{.spec = spec, .levels = next_grid}.validate();
    std::ranges::copy(next_species, species_state.levels.begin());
    grid.replace_levels(std::move(next_grid));
    return report;
  }

  [[nodiscard]] ContactGraph find_cell_contacts(const WorldState& state,
                                                const ContactParameters& parameters) override {
    activate_device();
    validate_contact_parameters(parameters);
    const auto geometry = state.geometry_state();
    if (geometry.size() == 0) {
      return ContactGraph{};
    }
    if (geometry.size() > std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("CUDA contact launch exceeds the uint32 cell index space");
    }
    const auto candidates = find_cell_contact_candidates(state, parameters);
    if (candidates.empty()) {
      return ContactGraph(geometry.size(), {});
    }
    if (candidates.size() > std::numeric_limits<std::uint32_t>::max() / 2) {
      throw std::overflow_error("CUDA contact candidates exceed the uint32 scan space");
    }

    ensure_contact_cell_capacity(geometry.size());
    ensure_contact_candidate_capacity(candidates.size());
    ensure_contact_pair_capacity(candidates.size());
    upload_contact_cells(geometry);
    upload_contact_candidates(candidates);
    const auto candidate_count = static_cast<std::uint32_t>(candidates.size());
    const auto contact_count = count_contacts(candidate_count, parameters);
    if (contact_count == 0) {
      return ContactGraph(geometry.size(), {});
    }

    ensure_contact_output_capacity(contact_count);
    fill_contacts(candidate_count, parameters);
    return download_contacts(geometry.size(), contact_count);
  }

  [[nodiscard]] ExternalContactGraph find_external_contacts(
      const WorldState& state, const ConstraintSet& constraints,
      const ConstraintContactParameters& parameters) override {
    activate_device();
    validate_constraint_contact_parameters(parameters);
    state.validate();
    const auto geometry = state.geometry_state();
    if (geometry.size() == 0 || constraints.empty()) {
      return ExternalContactGraph(geometry.size(), {});
    }
    if (geometry.size() > std::numeric_limits<std::uint32_t>::max() ||
        constraints.size() > std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("CUDA external-contact launch exceeds the uint32 index space");
    }
    if (geometry.size() > std::numeric_limits<std::size_t>::max() / constraints.size()) {
      throw std::overflow_error("CUDA external-contact pair count overflow");
    }
    const auto pair_count = geometry.size() * constraints.size();
    if (pair_count > std::numeric_limits<std::uint32_t>::max() / 2) {
      throw std::overflow_error("CUDA external-contact staging exceeds the uint32 scan space");
    }

    ensure_contact_cell_capacity(geometry.size());
    ensure_external_constraint_capacity(constraints.size());
    ensure_contact_pair_capacity(pair_count);
    upload_contact_cells(geometry);
    upload_external_constraints(constraints);
    const auto contact_count = count_external_contacts(
        static_cast<std::uint32_t>(geometry.size()), static_cast<std::uint32_t>(constraints.size()),
        static_cast<std::uint32_t>(pair_count), parameters);
    if (contact_count == 0) {
      return ExternalContactGraph(geometry.size(), {});
    }

    ensure_contact_output_capacity(contact_count);
    fill_external_contacts(static_cast<std::uint32_t>(geometry.size()),
                           static_cast<std::uint32_t>(constraints.size()), parameters);
    return download_external_contacts(geometry.size(), contact_count);
  }

  [[nodiscard]] MechanicsSolveResult solve_cell_mechanics(
      const WorldState& state, const ContactGraph& contacts,
      const ExternalContactGraph& external_contacts,
      const MechanicsParameters& parameters) override {
    activate_device();
    validate_mechanics_parameters(parameters);
    state.validate();
    const auto geometry = state.geometry_state();
    if (contacts.cell_count() != geometry.size()) {
      throw std::invalid_argument("contact graph and world state cell counts disagree");
    }
    if (external_contacts.cell_count() != geometry.size()) {
      throw std::invalid_argument("external contact graph and world state cell counts disagree");
    }
    if (external_contacts.size() > std::numeric_limits<std::size_t>::max() - contacts.size()) {
      throw std::overflow_error("CUDA mechanics row count overflow");
    }
    const auto row_count = contacts.size() + external_contacts.size();
    if (geometry.size() > std::numeric_limits<std::uint32_t>::max() ||
        row_count > std::numeric_limits<std::uint32_t>::max() / 2) {
      throw std::overflow_error("CUDA mechanics exceeds the uint32 index space");
    }

    MechanicsSolveResult result;
    result.corrections.resize(geometry.size());
    if (geometry.size() == 0 || row_count == 0) {
      return result;
    }

    validate_mechanics_contacts(geometry, contacts);
    validate_external_mechanics_contacts(geometry, external_contacts);
    ensure_contact_cell_capacity(geometry.size());
    ensure_contact_output_capacity(row_count);
    ensure_mechanics_capacity(geometry.size(), row_count);
    upload_contact_cells(geometry);
    upload_mechanics_fixed(state.cell_attributes().fixed);
    upload_mechanics_contacts(contacts, external_contacts);
    upload_mechanics_incidence(contacts, external_contacts);

    const auto cell_count = static_cast<std::uint32_t>(geometry.size());
    const auto contact_count = static_cast<std::uint32_t>(row_count);
    auto residual_squared = initialize_mechanics(cell_count, contact_count);
    result.report.initial_residual_rms =
        std::sqrt(residual_squared / static_cast<float>(cell_count));
    result.report.final_residual_rms = result.report.initial_residual_rms;
    if (!std::isfinite(result.report.initial_residual_rms)) {
      result.report.status = SolverStatus::breakdown;
      result.report.breakdown = SolverBreakdown::non_finite_residual;
      return result;
    }
    if (result.report.initial_residual_rms <= parameters.residual_rms_tolerance) {
      return result;
    }

    result.report.status = SolverStatus::iteration_limit;
    const auto maximum_iterations = mechanics_iteration_limit(parameters, geometry.size());
    for (std::uint32_t iteration = 0; iteration < maximum_iterations; ++iteration) {
      const auto curvature = apply_search_direction(cell_count, contact_count, parameters);
      if (!std::isfinite(curvature)) {
        result.report.status = SolverStatus::breakdown;
        result.report.breakdown = SolverBreakdown::non_finite_curvature;
        break;
      }
      if (curvature <= 0.0F) {
        result.report.status = SolverStatus::breakdown;
        result.report.breakdown = SolverBreakdown::non_positive_curvature;
        break;
      }

      const auto alpha = residual_squared / curvature;
      const auto next_residual_squared = update_solution_residual(cell_count, alpha);
      result.report.iterations = iteration + 1;
      const auto recurrence_rms = std::sqrt(next_residual_squared / static_cast<float>(cell_count));
      if (!std::isfinite(recurrence_rms)) {
        result.report.status = SolverStatus::breakdown;
        result.report.breakdown = SolverBreakdown::non_finite_residual;
        break;
      }

      if (recurrence_rms <= parameters.residual_rms_tolerance) {
        residual_squared = recompute_residual(cell_count, contact_count, parameters);
        const auto recomputed_rms = std::sqrt(residual_squared / static_cast<float>(cell_count));
        if (!std::isfinite(recomputed_rms)) {
          result.report.status = SolverStatus::breakdown;
          result.report.breakdown = SolverBreakdown::non_finite_residual;
          break;
        }
        if (recomputed_rms <= parameters.residual_rms_tolerance) {
          result.report.status = SolverStatus::converged;
          break;
        }
        update_search_direction(cell_count, 0.0F);
        continue;
      }

      const auto beta = next_residual_squared / residual_squared;
      update_search_direction(cell_count, beta);
      residual_squared = next_residual_squared;
    }

    residual_squared = recompute_residual(cell_count, contact_count, parameters);
    result.report.final_residual_rms = std::sqrt(residual_squared / static_cast<float>(cell_count));
    if (!std::isfinite(result.report.final_residual_rms) &&
        result.report.status != SolverStatus::breakdown) {
      result.report.status = SolverStatus::breakdown;
      result.report.breakdown = SolverBreakdown::non_finite_residual;
    }
    result.corrections = download_mechanics_solution(geometry.size());
    return result;
  }

 private:
  void activate_device() {
    check_cuda(cudaSetDevice(device_index_), "failed to activate the CUDA device");
  }

  void ensure_species_capacity(std::size_t cell_count, std::size_t level_count,
                               std::size_t instruction_count, std::size_t species_count,
                               std::size_t workspace_count) {
    species_levels_.reserve(level_count, "species levels");
    species_previous_lengths_.reserve(cell_count, "species previous lengths");
    species_centers_.reserve(cell_count, "species cell centers");
    species_geometry_.reserve(cell_count, "species cell geometry");
    species_growth_rates_.reserve(cell_count, "species growth rates");
    species_cell_types_.reserve(cell_count, "species cell types");
    species_instructions_.reserve(instruction_count, "species rate instructions");
    species_outputs_.reserve(species_count, "species rate outputs");
    species_workspace_.reserve(workspace_count, "species rate workspace");
    species_error_.reserve(1, "species error flag");
  }

  void ensure_contact_cell_capacity(std::size_t count) {
    contact_ids_.reserve(count, "contact cell IDs");
    contact_centers_.reserve(count, "contact cell centers");
    contact_axes_.reserve(count, "contact cell axes");
    contact_geometry_.reserve(count, "contact cell geometry");
  }

  void ensure_contact_pair_capacity(std::size_t count) {
    contact_counts_.reserve(count, "contact counts");
    contact_scan_a_.reserve(count, "contact scan A");
    contact_scan_b_.reserve(count, "contact scan B");
  }

  void ensure_contact_candidate_capacity(std::size_t count) {
    contact_candidates_.reserve(count, "contact candidates");
  }

  void ensure_external_constraint_capacity(std::size_t count) {
    external_constraints_.reserve(count, "external constraints");
  }

  void ensure_contact_output_capacity(std::size_t count) {
    contact_first_ids_.reserve(count, "contact first IDs");
    contact_second_ids_.reserve(count, "contact second IDs");
    contact_first_slots_.reserve(count, "contact first slots");
    contact_second_slots_.reserve(count, "contact second slots");
    contact_ordinals_.reserve(count, "contact ordinals");
    contact_points_.reserve(count, "contact points");
    contact_normals_.reserve(count, "contact normals");
    contact_separations_.reserve(count, "contact separations");
    contact_weights_.reserve(count, "contact weights");
  }

  void upload_contact_cells(const CellGeometryView& geometry) {
    std::vector<float4> centers(geometry.size());
    std::vector<float4> axes(geometry.size());
    std::vector<float4> shapes(geometry.size());
    for (std::size_t index = 0; index < geometry.size(); ++index) {
      centers[index] = make_float4(geometry.position_x[index], geometry.position_y[index],
                                   geometry.position_z[index], 0.0F);
      axes[index] = make_float4(geometry.direction_x[index], geometry.direction_y[index],
                                geometry.direction_z[index], 0.0F);
      shapes[index] = make_float4(geometry.lengths[index], geometry.radii[index], 0.0F, 0.0F);
    }

    check_cuda(cudaMemcpy(contact_ids_.data(), geometry.ids.data(), geometry.ids.size_bytes(),
                          cudaMemcpyHostToDevice),
               "failed to upload CUDA contact cell IDs");
    check_cuda(cudaMemcpy(contact_centers_.data(), centers.data(), centers.size() * sizeof(float4),
                          cudaMemcpyHostToDevice),
               "failed to upload CUDA contact cell centers");
    check_cuda(cudaMemcpy(contact_axes_.data(), axes.data(), axes.size() * sizeof(float4),
                          cudaMemcpyHostToDevice),
               "failed to upload CUDA contact cell axes");
    check_cuda(cudaMemcpy(contact_geometry_.data(), shapes.data(), shapes.size() * sizeof(float4),
                          cudaMemcpyHostToDevice),
               "failed to upload CUDA contact cell geometry");
  }

  void upload_contact_candidates(std::span<const ContactCandidate> candidates) {
    std::vector<uint2> values;
    values.reserve(candidates.size());
    for (const auto& candidate : candidates) {
      values.push_back(make_uint2(candidate.first_slot, candidate.second_slot));
    }
    check_cuda(cudaMemcpy(contact_candidates_.data(), values.data(), values.size() * sizeof(uint2),
                          cudaMemcpyHostToDevice),
               "failed to upload CUDA contact candidates");
  }

  void upload_external_constraints(const ConstraintSet& constraints) {
    std::vector<cuda::ExternalConstraintGpu> values;
    values.reserve(constraints.size());
    for (const auto& plane : constraints.planes()) {
      values.push_back({
          .id = plane.id,
          .kind = static_cast<std::uint32_t>(ExternalConstraintKind::plane),
          .allowed_region = 0,
          .geometry = make_float4(plane.point.x, plane.point.y, plane.point.z, 0.0F),
          .parameters = make_float4(plane.inward_normal.x, plane.inward_normal.y,
                                    plane.inward_normal.z, plane.coefficient),
      });
    }
    for (const auto& sphere : constraints.spheres()) {
      values.push_back({
          .id = sphere.id,
          .kind = static_cast<std::uint32_t>(ExternalConstraintKind::sphere),
          .allowed_region = static_cast<std::uint32_t>(sphere.allowed_region),
          .geometry = make_float4(sphere.center.x, sphere.center.y, sphere.center.z, sphere.radius),
          .parameters = make_float4(0.0F, 0.0F, 0.0F, sphere.coefficient),
      });
    }
    for (const auto& box : constraints.boxes()) {
      values.push_back({
          .id = box.id,
          .kind = static_cast<std::uint32_t>(ExternalConstraintKind::box),
          .allowed_region = static_cast<std::uint32_t>(box.allowed_region),
          .geometry = make_float4(box.center.x, box.center.y, box.center.z, 0.0F),
          .parameters = make_float4(box.half_extents.x, box.half_extents.y, box.half_extents.z,
                                    box.coefficient),
      });
    }
    for (const auto& cylinder : constraints.cylinders()) {
      values.push_back({
          .id = cylinder.id,
          .kind = static_cast<std::uint32_t>(ExternalConstraintKind::cylinder),
          .allowed_region = static_cast<std::uint32_t>(cylinder.allowed_region),
          .geometry = make_float4(cylinder.center.x, cylinder.center.y, cylinder.center.z,
                                  cylinder.radius),
          .parameters = make_float4(cylinder.half_height, 0.0F, 0.0F, cylinder.coefficient),
      });
    }
    std::ranges::sort(values, {}, &cuda::ExternalConstraintGpu::id);
    copy_to_device(external_constraints_, values, "failed to upload CUDA external constraints");
  }

  void scan_contact_counts(std::uint32_t element_count) {
    const std::uint32_t* scan_input = contact_counts_.data();
    std::uint32_t* scan_output = contact_scan_a_.data();
    std::uint32_t offset = 1;
    while (offset < element_count) {
      cuda::launch_inclusive_scan_step(scan_input, scan_output, offset, element_count, stream_);
      check_cuda(cudaGetLastError(), "failed to launch the CUDA contact-scan kernel");
      scan_input = scan_output;
      scan_output =
          scan_output == contact_scan_a_.data() ? contact_scan_b_.data() : contact_scan_a_.data();
      if (offset > element_count / 2) {
        break;
      }
      offset *= 2;
    }
    contact_inclusive_counts_ = scan_input;
  }

  [[nodiscard]] std::uint32_t download_contact_count(std::uint32_t pair_count,
                                                     const char* operation) {
    std::uint32_t contact_count = 0;
    check_cuda(cudaMemcpyAsync(&contact_count, contact_inclusive_counts_ + pair_count - 1,
                               sizeof(contact_count), cudaMemcpyDeviceToHost, stream_),
               "failed to download the CUDA contact count");
    check_cuda(cudaStreamSynchronize(stream_), operation);
    return contact_count;
  }

  [[nodiscard]] std::uint32_t count_contacts(std::uint32_t candidate_count,
                                             const ContactParameters& parameters) {
    const cuda::ContactParametersGpu gpu_parameters{
        .activation_margin = parameters.activation_margin,
        .parallel_sine_threshold = parameters.parallel_sine_threshold,
        .degeneracy_epsilon = parameters.degeneracy_epsilon,
    };
    cuda::launch_contact_count(contact_ids_.data(), contact_centers_.data(), contact_axes_.data(),
                               contact_geometry_.data(), contact_candidates_.data(),
                               contact_counts_.data(), gpu_parameters, candidate_count, stream_);
    check_cuda(cudaGetLastError(), "failed to launch the CUDA contact-count kernel");
    scan_contact_counts(candidate_count);
    return download_contact_count(candidate_count, "CUDA contact count or scan failed");
  }

  void fill_contacts(std::uint32_t candidate_count, const ContactParameters& parameters) {
    const cuda::ContactParametersGpu gpu_parameters{
        .activation_margin = parameters.activation_margin,
        .parallel_sine_threshold = parameters.parallel_sine_threshold,
        .degeneracy_epsilon = parameters.degeneracy_epsilon,
    };
    cuda::launch_contact_fill(
        contact_ids_.data(), contact_centers_.data(), contact_axes_.data(),
        contact_geometry_.data(), contact_candidates_.data(), contact_counts_.data(),
        contact_inclusive_counts_, contact_first_ids_.data(), contact_second_ids_.data(),
        contact_first_slots_.data(), contact_second_slots_.data(), contact_ordinals_.data(),
        contact_points_.data(), contact_normals_.data(), contact_separations_.data(),
        contact_weights_.data(), gpu_parameters, candidate_count, stream_);
    check_cuda(cudaGetLastError(), "failed to launch the CUDA contact-fill kernel");
  }

  [[nodiscard]] ContactGraph download_contacts(std::size_t cell_count,
                                               std::uint32_t contact_count) {
    std::vector<std::uint64_t> first_ids(contact_count);
    std::vector<std::uint64_t> second_ids(contact_count);
    std::vector<std::uint32_t> first_slots(contact_count);
    std::vector<std::uint32_t> second_slots(contact_count);
    std::vector<std::uint32_t> ordinals(contact_count);
    std::vector<float4> points(contact_count);
    std::vector<float4> normals(contact_count);
    std::vector<float> separations(contact_count);
    std::vector<float> weights(contact_count);

    copy_to_host(first_ids, contact_first_ids_, "failed to download CUDA contact first IDs");
    copy_to_host(second_ids, contact_second_ids_, "failed to download CUDA contact second IDs");
    copy_to_host(first_slots, contact_first_slots_, "failed to download CUDA contact first slots");
    copy_to_host(second_slots, contact_second_slots_,
                 "failed to download CUDA contact second slots");
    copy_to_host(ordinals, contact_ordinals_, "failed to download CUDA contact ordinals");
    copy_to_host(points, contact_points_, "failed to download CUDA contact points");
    copy_to_host(normals, contact_normals_, "failed to download CUDA contact normals");
    copy_to_host(separations, contact_separations_, "failed to download CUDA contact separations");
    copy_to_host(weights, contact_weights_, "failed to download CUDA contact weights");
    check_cuda(cudaStreamSynchronize(stream_), "CUDA contact fill or download failed");

    std::vector<CellContact> contacts;
    contacts.reserve(contact_count);
    for (std::uint32_t index = 0; index < contact_count; ++index) {
      if (ordinals[index] > 1) {
        throw std::runtime_error("CUDA contact kernel produced an invalid ordinal");
      }
      contacts.push_back({
          .first_id = first_ids[index],
          .second_id = second_ids[index],
          .first_slot = first_slots[index],
          .second_slot = second_slots[index],
          .ordinal = static_cast<std::uint8_t>(ordinals[index]),
          .point_on_first = {points[index].x, points[index].y, points[index].z},
          .normal = {normals[index].x, normals[index].y, normals[index].z},
          .signed_separation = separations[index],
          .weight = weights[index],
      });
    }
    std::ranges::sort(contacts, {}, [](const CellContact& contact) {
      return std::tuple{contact.first_id, contact.second_id, contact.ordinal};
    });
    return ContactGraph(cell_count, std::move(contacts));
  }

  [[nodiscard]] std::uint32_t count_external_contacts(
      std::uint32_t cell_count, std::uint32_t constraint_count, std::uint32_t pair_count,
      const ConstraintContactParameters& parameters) {
    const cuda::ConstraintContactParametersGpu gpu_parameters{
        .activation_margin = parameters.activation_margin,
        .degeneracy_epsilon = parameters.degeneracy_epsilon,
    };
    cuda::launch_external_contact_count(contact_ids_.data(), contact_centers_.data(),
                                        contact_axes_.data(), contact_geometry_.data(),
                                        external_constraints_.data(), contact_counts_.data(),
                                        gpu_parameters, cell_count, constraint_count, stream_);
    check_cuda(cudaGetLastError(), "failed to launch the CUDA external-contact-count kernel");
    scan_contact_counts(pair_count);
    return download_contact_count(pair_count, "CUDA external contact count or scan failed");
  }

  void fill_external_contacts(std::uint32_t cell_count, std::uint32_t constraint_count,
                              const ConstraintContactParameters& parameters) {
    const cuda::ConstraintContactParametersGpu gpu_parameters{
        .activation_margin = parameters.activation_margin,
        .degeneracy_epsilon = parameters.degeneracy_epsilon,
    };
    cuda::launch_external_contact_fill(
        contact_ids_.data(), contact_centers_.data(), contact_axes_.data(),
        contact_geometry_.data(), external_constraints_.data(), contact_counts_.data(),
        contact_inclusive_counts_, contact_first_ids_.data(), contact_second_ids_.data(),
        contact_first_slots_.data(), contact_second_slots_.data(), contact_ordinals_.data(),
        contact_points_.data(), contact_normals_.data(), contact_separations_.data(),
        contact_weights_.data(), gpu_parameters, cell_count, constraint_count, stream_);
    check_cuda(cudaGetLastError(), "failed to launch the CUDA external-contact-fill kernel");
  }

  [[nodiscard]] ExternalContactGraph download_external_contacts(std::size_t cell_count,
                                                                std::uint32_t contact_count) {
    std::vector<std::uint64_t> cell_ids(contact_count);
    std::vector<std::uint64_t> constraint_ids(contact_count);
    std::vector<std::uint32_t> cell_slots(contact_count);
    std::vector<std::uint32_t> constraint_kinds(contact_count);
    std::vector<std::uint32_t> endpoints(contact_count);
    std::vector<float4> points(contact_count);
    std::vector<float4> normals(contact_count);
    std::vector<float> separations(contact_count);
    std::vector<float> weights(contact_count);

    copy_to_host(cell_ids, contact_first_ids_, "failed to download CUDA external cell IDs");
    copy_to_host(constraint_ids, contact_second_ids_,
                 "failed to download CUDA external constraint IDs");
    copy_to_host(cell_slots, contact_first_slots_, "failed to download CUDA external cell slots");
    copy_to_host(constraint_kinds, contact_second_slots_,
                 "failed to download CUDA external constraint kinds");
    copy_to_host(endpoints, contact_ordinals_, "failed to download CUDA external endpoints");
    copy_to_host(points, contact_points_, "failed to download CUDA external contact points");
    copy_to_host(normals, contact_normals_, "failed to download CUDA external contact normals");
    copy_to_host(separations, contact_separations_,
                 "failed to download CUDA external contact separations");
    copy_to_host(weights, contact_weights_, "failed to download CUDA external contact weights");
    check_cuda(cudaStreamSynchronize(stream_), "CUDA external contact fill or download failed");

    std::vector<ExternalContact> contacts;
    contacts.reserve(contact_count);
    for (std::uint32_t index = 0; index < contact_count; ++index) {
      if (constraint_kinds[index] >
              static_cast<std::uint32_t>(ExternalConstraintKind::cylinder) ||
          endpoints[index] > static_cast<std::uint32_t>(RodEndpoint::positive)) {
        throw std::runtime_error("CUDA external-contact kernel produced an invalid tag");
      }
      contacts.push_back({
          .cell_id = cell_ids[index],
          .cell_slot = cell_slots[index],
          .constraint_id = constraint_ids[index],
          .constraint_kind = static_cast<ExternalConstraintKind>(constraint_kinds[index]),
          .endpoint = static_cast<RodEndpoint>(endpoints[index]),
          .point_on_cell = {points[index].x, points[index].y, points[index].z},
          .normal = {normals[index].x, normals[index].y, normals[index].z},
          .signed_separation = separations[index],
          .weight = weights[index],
      });
    }
    std::ranges::sort(contacts, {}, [](const ExternalContact& contact) {
      return std::tuple{contact.cell_id, contact.constraint_id, contact.endpoint};
    });
    return ExternalContactGraph(cell_count, std::move(contacts));
  }

  static void validate_mechanics_contacts(const CellGeometryView& geometry,
                                          const ContactGraph& contacts) {
    for (const auto& contact : contacts.contacts()) {
      const auto first = static_cast<std::size_t>(contact.first_slot);
      const auto second = static_cast<std::size_t>(contact.second_slot);
      if (geometry.ids[first] != contact.first_id || geometry.ids[second] != contact.second_id) {
        throw std::invalid_argument("contact graph identifiers do not match current state slots");
      }
    }
  }

  static void validate_external_mechanics_contacts(const CellGeometryView& geometry,
                                                   const ExternalContactGraph& contacts) {
    for (const auto& contact : contacts.contacts()) {
      const auto cell = static_cast<std::size_t>(contact.cell_slot);
      if (geometry.ids[cell] != contact.cell_id) {
        throw std::invalid_argument(
            "external contact graph identifiers do not match current state slots");
      }
    }
  }

  static std::uint32_t mechanics_iteration_limit(const MechanicsParameters& parameters,
                                                 std::size_t cell_count) {
    if (parameters.max_iterations != 0) {
      return parameters.max_iterations;
    }
    constexpr std::size_t degrees_of_freedom = 7;
    if (cell_count > std::numeric_limits<std::uint32_t>::max() / degrees_of_freedom) {
      throw std::overflow_error("default mechanics iteration limit exceeds uint32");
    }
    return static_cast<std::uint32_t>(cell_count * degrees_of_freedom);
  }

  void ensure_mechanics_capacity(std::size_t cell_count, std::size_t contact_count) {
    mechanics_incidence_offsets_.reserve(cell_count + 1, "mechanics incidence offsets");
    mechanics_fixed_.reserve(cell_count, "mechanics fixed flags");
    mechanics_solution_.reserve(cell_count, "mechanics solution");
    mechanics_rhs_.reserve(cell_count, "mechanics right-hand side");
    mechanics_residual_.reserve(cell_count, "mechanics residual");
    mechanics_search_.reserve(cell_count, "mechanics search direction");
    mechanics_applied_.reserve(cell_count, "mechanics applied vector");
    mechanics_dot_terms_.reserve(cell_count, "mechanics dot terms");
    mechanics_reduce_a_.reserve(cell_count, "mechanics reduction A");
    mechanics_reduce_b_.reserve(cell_count, "mechanics reduction B");
    mechanics_first_rows_.reserve(contact_count, "mechanics first Jacobian rows");
    mechanics_second_rows_.reserve(contact_count, "mechanics second Jacobian rows");
    mechanics_row_values_.reserve(contact_count, "mechanics row values");
    mechanics_row_rhs_.reserve(contact_count, "mechanics row right-hand side");
    mechanics_incidence_indices_.reserve(contact_count * 2, "mechanics incidence indices");
  }

  void upload_mechanics_fixed(std::span<const std::uint8_t> fixed) {
    check_cuda(cudaMemcpy(mechanics_fixed_.data(), fixed.data(), fixed.size_bytes(),
                          cudaMemcpyHostToDevice),
               "failed to upload CUDA mechanics fixed flags");
  }

  void upload_mechanics_contacts(const ContactGraph& contacts,
                                 const ExternalContactGraph& external_contacts) {
    const auto row_count = contacts.size() + external_contacts.size();
    std::vector<std::uint32_t> first_slots(row_count);
    std::vector<std::uint32_t> second_slots(row_count);
    std::vector<float4> points(row_count);
    std::vector<float4> normals(row_count);
    std::vector<float> separations(row_count);
    std::vector<float> weights(row_count);
    for (std::size_t index = 0; index < contacts.size(); ++index) {
      const auto& contact = contacts.contacts()[index];
      first_slots[index] = contact.first_slot;
      second_slots[index] = contact.second_slot;
      points[index] = make_float4(contact.point_on_first.x, contact.point_on_first.y,
                                  contact.point_on_first.z, 0.0F);
      normals[index] = make_float4(contact.normal.x, contact.normal.y, contact.normal.z, 0.0F);
      separations[index] = contact.signed_separation;
      weights[index] = contact.weight;
    }
    for (std::size_t index = 0; index < external_contacts.size(); ++index) {
      const auto output_index = contacts.size() + index;
      const auto& contact = external_contacts.contacts()[index];
      first_slots[output_index] = contact.cell_slot;
      second_slots[output_index] = invalid_slot;
      points[output_index] = make_float4(contact.point_on_cell.x, contact.point_on_cell.y,
                                         contact.point_on_cell.z, 0.0F);
      normals[output_index] =
          make_float4(contact.normal.x, contact.normal.y, contact.normal.z, 0.0F);
      separations[output_index] = contact.signed_separation;
      weights[output_index] = contact.weight;
    }
    copy_to_device(contact_first_slots_, first_slots,
                   "failed to upload CUDA mechanics first slots");
    copy_to_device(contact_second_slots_, second_slots,
                   "failed to upload CUDA mechanics second slots");
    copy_to_device(contact_points_, points, "failed to upload CUDA mechanics contact points");
    copy_to_device(contact_normals_, normals, "failed to upload CUDA mechanics contact normals");
    copy_to_device(contact_separations_, separations,
                   "failed to upload CUDA mechanics contact separations");
    copy_to_device(contact_weights_, weights, "failed to upload CUDA mechanics contact weights");
  }

  void upload_mechanics_incidence(const ContactGraph& contacts,
                                  const ExternalContactGraph& external_contacts) {
    std::vector<std::uint32_t> offsets(contacts.cell_count() + 1);
    std::vector<std::uint32_t> indices(contacts.size() * 2 + external_contacts.size());
    std::uint32_t cursor = 0;
    for (std::size_t slot = 0; slot < contacts.cell_count(); ++slot) {
      offsets[slot] = cursor;
      for (const auto contact_index : contacts.incident_contact_indices(static_cast<Slot>(slot))) {
        indices[cursor++] = static_cast<std::uint32_t>(contact_index);
      }
      for (const auto contact_index :
           external_contacts.incident_contact_indices(static_cast<Slot>(slot))) {
        indices[cursor++] = static_cast<std::uint32_t>(contacts.size() + contact_index);
      }
    }
    offsets[contacts.cell_count()] = cursor;
    if (cursor != contacts.size() * 2 + external_contacts.size()) {
      throw std::logic_error("contact incidence size is inconsistent");
    }
    copy_to_device(mechanics_incidence_offsets_, offsets,
                   "failed to upload CUDA mechanics incidence offsets");
    copy_to_device(mechanics_incidence_indices_, indices,
                   "failed to upload CUDA mechanics incidence indices");
  }

  void apply_mechanics_operator(const cuda::MechanicsDofsGpu* input, cuda::MechanicsDofsGpu* output,
                                std::uint32_t cell_count, std::uint32_t contact_count,
                                const MechanicsParameters& parameters) {
    cuda::launch_apply_mechanics_b(mechanics_first_rows_.data(), mechanics_second_rows_.data(),
                                   contact_first_slots_.data(), contact_second_slots_.data(), input,
                                   mechanics_fixed_.data(), mechanics_row_values_.data(),
                                   contact_count, stream_);
    check_cuda(cudaGetLastError(), "failed to launch the CUDA mechanics-B kernel");
    cuda::launch_apply_mechanics_transpose(
        mechanics_first_rows_.data(), mechanics_second_rows_.data(), mechanics_row_values_.data(),
        mechanics_incidence_offsets_.data(), mechanics_incidence_indices_.data(),
        contact_first_slots_.data(), output, cell_count, stream_);
    check_cuda(cudaGetLastError(), "failed to launch the CUDA mechanics-transpose kernel");
    cuda::launch_add_mechanics_regularizer(contact_axes_.data(), contact_geometry_.data(), input,
                                           output, mechanics_fixed_.data(), parameters.mu_a,
                                           parameters.gamma, cell_count, stream_);
    check_cuda(cudaGetLastError(), "failed to launch the CUDA mechanics-regularizer kernel");
  }

  void ensure_signal_solve_capacity(std::uint32_t level_count) {
    signal_cn_a_.reserve(level_count, "signal Jacobi field A");
    signal_cn_b_.reserve(level_count, "signal Jacobi field B");
    signal_cn_terms_.reserve(level_count, "signal residual terms");
    signal_cn_reduce_a_.reserve(level_count, "signal reduction A");
    signal_cn_reduce_b_.reserve(level_count, "signal reduction B");
  }

  [[nodiscard]] float reduce_signal_terms(std::uint32_t level_count, const char* operation) {
    const float* input = signal_cn_terms_.data();
    float* output = signal_cn_reduce_a_.data();
    auto element_count = level_count;
    while (element_count > 1) {
      cuda::launch_reduce_sum_pairs(input, output, element_count, stream_);
      check_cuda(cudaGetLastError(), "failed to launch the CUDA signal-reduction kernel");
      input = output;
      output = output == signal_cn_reduce_a_.data() ? signal_cn_reduce_b_.data()
                                                    : signal_cn_reduce_a_.data();
      element_count = (element_count + 1) / 2;
    }
    float result = 0.0F;
    check_cuda(cudaMemcpyAsync(&result, input, sizeof(result), cudaMemcpyDeviceToHost, stream_),
               "failed to download a CUDA signal reduction");
    check_cuda(cudaStreamSynchronize(stream_), operation);
    return result;
  }

  [[nodiscard]] float signal_rhs_rms(const float* right_hand_side, std::uint32_t level_count) {
    cuda::launch_signal_square_terms(right_hand_side, signal_cn_terms_.data(), level_count,
                                     stream_);
    check_cuda(cudaGetLastError(), "failed to launch the CUDA signal-norm kernel");
    return std::sqrt(reduce_signal_terms(level_count, "CUDA signal norm failed") /
                     static_cast<float>(level_count));
  }

  [[nodiscard]] float signal_residual_rms(const float* current, const float* right_hand_side,
                                          const float* diffusion, const float4* advection,
                                          const float* fixed_values, const float* reaction_source,
                                          const float* reaction_loss,
                                          const std::uint8_t* obstacles, const float* x_faces,
                                          const float* y_faces, const float* z_faces,
                                          std::uint32_t has_velocity_field,
                                          cuda::SignalGridBoundariesGpu boundaries,
                                          cuda::SignalGridShapeGpu shape, float4 spacing,
                                          float half_dt, std::uint32_t signal_count,
                                          std::uint32_t level_count) {
    cuda::launch_signal_crank_nicolson_residual_terms(
        current, right_hand_side, signal_cn_terms_.data(), diffusion, advection, fixed_values,
        reaction_source, reaction_loss, obstacles, x_faces, y_faces, z_faces, has_velocity_field,
        boundaries, shape, spacing, half_dt, signal_count, level_count, stream_);
    check_cuda(cudaGetLastError(), "failed to launch the CUDA signal-residual kernel");
    return std::sqrt(reduce_signal_terms(level_count, "CUDA signal residual failed") /
                     static_cast<float>(level_count));
  }

  [[nodiscard]] std::pair<const float*, SignalSolveReport> solve_signal_crank_nicolson(
      const float* initial, const float* right_hand_side, const float* diffusion,
      const float4* advection, const float* fixed_values, const float* reaction_source,
      const float* reaction_loss, const std::uint8_t* obstacles, const float* x_faces,
      const float* y_faces, const float* z_faces, std::uint32_t has_velocity_field,
      std::uint32_t* error, cuda::SignalGridBoundariesGpu boundaries,
      cuda::SignalGridShapeGpu shape, float4 spacing, float dt, std::uint32_t signal_count,
      std::uint32_t level_count, const SignalSolveParameters& parameters) {
    ensure_signal_solve_capacity(level_count);
    const auto half_dt = 0.5F * dt;
    const auto right_hand_side_rms = signal_rhs_rms(right_hand_side, level_count);
    const auto threshold =
        parameters.absolute_tolerance + (parameters.relative_tolerance * right_hand_side_rms);
    SignalSolveReport report;
    report.residual_rms = signal_residual_rms(
        initial, right_hand_side, diffusion, advection, fixed_values, reaction_source,
        reaction_loss, obstacles, x_faces, y_faces, z_faces, has_velocity_field, boundaries,
        shape, spacing, half_dt, signal_count, level_count);
    if (std::isfinite(report.residual_rms) && report.residual_rms <= threshold) {
      return {initial, report};
    }
    if (!std::isfinite(report.residual_rms) || !std::isfinite(threshold)) {
      report.converged = false;
      return {initial, report};
    }

    check_cuda(cudaMemsetAsync(error, 0, sizeof(std::uint32_t), stream_),
               "failed to clear the CUDA signal solver error flag");
    const float* current = initial;
    for (std::uint32_t iteration = 1; iteration <= parameters.max_iterations; ++iteration) {
      float* output = current == signal_cn_a_.data() ? signal_cn_b_.data() : signal_cn_a_.data();
      cuda::launch_signal_crank_nicolson_jacobi(current, output, right_hand_side, diffusion,
                                                advection, fixed_values, reaction_source,
                                                reaction_loss, obstacles, x_faces, y_faces,
                                                z_faces, has_velocity_field, error, boundaries,
                                                shape, spacing, half_dt, signal_count,
                                                level_count, stream_);
      check_cuda(cudaGetLastError(), "failed to launch the CUDA signal Jacobi kernel");
      report.residual_rms = signal_residual_rms(
          output, right_hand_side, diffusion, advection, fixed_values, reaction_source,
          reaction_loss, obstacles, x_faces, y_faces, z_faces, has_velocity_field, boundaries,
          shape, spacing, half_dt, signal_count, level_count);
      report.iterations = iteration;
      current = output;

      std::uint32_t error_value = 0;
      check_cuda(cudaMemcpyAsync(&error_value, error, sizeof(error_value), cudaMemcpyDeviceToHost,
                                 stream_),
                 "failed to download the CUDA signal solver error flag");
      check_cuda(cudaStreamSynchronize(stream_), "CUDA signal solver error check failed");
      if (error_value != 0 || !std::isfinite(report.residual_rms)) {
        report.converged = false;
        return {current, report};
      }
      if (report.residual_rms <= threshold) {
        report.converged = true;
        return {current, report};
      }
    }
    report.converged = false;
    return {current, report};
  }

  [[nodiscard]] float reduce_mechanics_dot(const cuda::MechanicsDofsGpu* left,
                                           const cuda::MechanicsDofsGpu* right,
                                           std::uint32_t cell_count, const char* operation) {
    cuda::launch_mechanics_dot_terms(left, right, mechanics_dot_terms_.data(), cell_count, stream_);
    check_cuda(cudaGetLastError(), "failed to launch the CUDA mechanics-dot kernel");
    const float* input = mechanics_dot_terms_.data();
    float* output = mechanics_reduce_a_.data();
    auto element_count = cell_count;
    while (element_count > 1) {
      cuda::launch_reduce_sum_pairs(input, output, element_count, stream_);
      check_cuda(cudaGetLastError(), "failed to launch the CUDA mechanics-reduction kernel");
      input = output;
      output = output == mechanics_reduce_a_.data() ? mechanics_reduce_b_.data()
                                                    : mechanics_reduce_a_.data();
      element_count = (element_count + 1) / 2;
    }
    float result = 0.0F;
    check_cuda(cudaMemcpyAsync(&result, input, sizeof(result), cudaMemcpyDeviceToHost, stream_),
               "failed to download a CUDA mechanics reduction");
    check_cuda(cudaStreamSynchronize(stream_), operation);
    return result;
  }

  [[nodiscard]] float initialize_mechanics(std::uint32_t cell_count, std::uint32_t contact_count) {
    cuda::launch_build_mechanics_rows(
        contact_centers_.data(), contact_axes_.data(), contact_geometry_.data(),
        contact_first_slots_.data(), contact_second_slots_.data(), contact_points_.data(),
        contact_normals_.data(), contact_separations_.data(), contact_weights_.data(),
        mechanics_first_rows_.data(), mechanics_second_rows_.data(), mechanics_row_rhs_.data(),
        contact_count, stream_);
    check_cuda(cudaGetLastError(), "failed to launch the CUDA mechanics-row kernel");
    cuda::launch_apply_mechanics_transpose(
        mechanics_first_rows_.data(), mechanics_second_rows_.data(), mechanics_row_rhs_.data(),
        mechanics_incidence_offsets_.data(), mechanics_incidence_indices_.data(),
        contact_first_slots_.data(), mechanics_rhs_.data(), cell_count, stream_);
    check_cuda(cudaGetLastError(), "failed to launch the CUDA mechanics-RHS kernel");
    cuda::launch_initialize_mechanics_vectors(mechanics_rhs_.data(), mechanics_solution_.data(),
                                              mechanics_residual_.data(), mechanics_search_.data(),
                                              mechanics_fixed_.data(), cell_count, stream_);
    check_cuda(cudaGetLastError(), "failed to launch the CUDA mechanics-initialize kernel");
    return reduce_mechanics_dot(mechanics_residual_.data(), mechanics_residual_.data(), cell_count,
                                "CUDA mechanics initialization failed");
  }

  [[nodiscard]] float apply_search_direction(std::uint32_t cell_count, std::uint32_t contact_count,
                                             const MechanicsParameters& parameters) {
    apply_mechanics_operator(mechanics_search_.data(), mechanics_applied_.data(), cell_count,
                             contact_count, parameters);
    return reduce_mechanics_dot(mechanics_search_.data(), mechanics_applied_.data(), cell_count,
                                "CUDA mechanics operator application failed");
  }

  [[nodiscard]] float update_solution_residual(std::uint32_t cell_count, float alpha) {
    cuda::launch_update_mechanics_solution_residual(
        mechanics_solution_.data(), mechanics_residual_.data(), mechanics_search_.data(),
        mechanics_applied_.data(), alpha, cell_count, stream_);
    check_cuda(cudaGetLastError(), "failed to launch the CUDA mechanics-update kernel");
    return reduce_mechanics_dot(mechanics_residual_.data(), mechanics_residual_.data(), cell_count,
                                "CUDA mechanics update failed");
  }

  void update_search_direction(std::uint32_t cell_count, float beta) {
    cuda::launch_update_mechanics_search_direction(
        mechanics_residual_.data(), mechanics_search_.data(), beta, cell_count, stream_);
    check_cuda(cudaGetLastError(), "failed to launch the CUDA mechanics-search kernel");
    check_cuda(cudaStreamSynchronize(stream_), "CUDA mechanics search update failed");
  }

  [[nodiscard]] float recompute_residual(std::uint32_t cell_count, std::uint32_t contact_count,
                                         const MechanicsParameters& parameters) {
    apply_mechanics_operator(mechanics_solution_.data(), mechanics_applied_.data(), cell_count,
                             contact_count, parameters);
    cuda::launch_subtract_mechanics_vectors(mechanics_rhs_.data(), mechanics_applied_.data(),
                                            mechanics_residual_.data(), cell_count, stream_);
    check_cuda(cudaGetLastError(), "failed to launch the CUDA mechanics-residual kernel");
    return reduce_mechanics_dot(mechanics_residual_.data(), mechanics_residual_.data(), cell_count,
                                "CUDA mechanics residual recomputation failed");
  }

  [[nodiscard]] std::vector<CellCorrection> download_mechanics_solution(std::size_t cell_count) {
    std::vector<cuda::MechanicsDofsGpu> values(cell_count);
    copy_to_host(values, mechanics_solution_, "failed to download CUDA mechanics solution");
    check_cuda(cudaStreamSynchronize(stream_), "CUDA mechanics solution download failed");
    std::vector<CellCorrection> result;
    result.reserve(cell_count);
    for (const auto& value : values) {
      result.push_back({
          .translation = {value.linear_length.x, value.linear_length.y, value.linear_length.z},
          .rotation = {value.rotation.x, value.rotation.y, value.rotation.z},
          .length = value.linear_length.w,
      });
    }
    return result;
  }

  template <typename T>
  void copy_to_device(CudaBuffer<T>& destination, const std::vector<T>& source,
                      const char* operation) {
    check_cuda(cudaMemcpy(destination.data(), source.data(), source.size() * sizeof(T),
                          cudaMemcpyHostToDevice),
               operation);
  }

  template <typename T>
  void copy_to_host(std::vector<T>& destination, const CudaBuffer<T>& source,
                    const char* operation) {
    check_cuda(cudaMemcpyAsync(destination.data(), source.data(), destination.size() * sizeof(T),
                               cudaMemcpyDeviceToHost, stream_),
               operation);
  }

  int device_index_{0};
  cudaDeviceProp device_properties_{};
  cudaStream_t stream_{nullptr};

  CudaBuffer<float> lengths_;
  CudaBuffer<float> growth_rates_;

  CudaBuffer<float> species_levels_;
  CudaBuffer<float> species_previous_lengths_;
  CudaBuffer<float4> species_centers_;
  CudaBuffer<float4> species_geometry_;
  CudaBuffer<float> species_growth_rates_;
  CudaBuffer<std::int32_t> species_cell_types_;
  CudaBuffer<cuda::RateInstructionGpu> species_instructions_;
  CudaBuffer<std::uint32_t> species_outputs_;
  CudaBuffer<float> species_workspace_;
  CudaBuffer<std::uint32_t> species_error_;

  CudaBuffer<float> signal_levels_;
  CudaBuffer<float> signal_output_;
  CudaBuffer<float> signal_diffusion_;
  CudaBuffer<float4> signal_advection_;
  CudaBuffer<float> signal_fixed_values_;
  CudaBuffer<float> signal_reaction_source_;
  CudaBuffer<float> signal_reaction_loss_;
  CudaBuffer<std::uint8_t> signal_obstacles_;
  CudaBuffer<float> signal_x_faces_;
  CudaBuffer<float> signal_y_faces_;
  CudaBuffer<float> signal_z_faces_;
  CudaBuffer<std::uint32_t> signal_error_;
  CudaBuffer<float> signal_cn_a_;
  CudaBuffer<float> signal_cn_b_;
  CudaBuffer<float> signal_cn_terms_;
  CudaBuffer<float> signal_cn_reduce_a_;
  CudaBuffer<float> signal_cn_reduce_b_;

  CudaBuffer<float> coupled_species_levels_;
  CudaBuffer<float> coupled_previous_lengths_;
  CudaBuffer<float4> coupled_centers_;
  CudaBuffer<float4> coupled_geometry_;
  CudaBuffer<float> coupled_growth_rates_;
  CudaBuffer<std::int32_t> coupled_cell_types_;
  CudaBuffer<cuda::RateInstructionGpu> coupled_instructions_;
  CudaBuffer<std::uint32_t> coupled_species_outputs_;
  CudaBuffer<std::uint32_t> coupled_signal_outputs_;
  CudaBuffer<float> coupled_workspace_;
  CudaBuffer<float> coupled_cell_signal_rates_;
  CudaBuffer<float> coupled_grid_levels_;
  CudaBuffer<float> coupled_grid_output_;
  CudaBuffer<float> coupled_diffusion_;
  CudaBuffer<float4> coupled_advection_;
  CudaBuffer<float> coupled_fixed_values_;
  CudaBuffer<float> coupled_reaction_source_;
  CudaBuffer<float> coupled_reaction_loss_;
  CudaBuffer<std::uint8_t> coupled_obstacles_;
  CudaBuffer<float> coupled_x_faces_;
  CudaBuffer<float> coupled_y_faces_;
  CudaBuffer<float> coupled_z_faces_;
  CudaBuffer<std::uint32_t> coupled_error_;

  CudaBuffer<std::uint64_t> contact_ids_;
  CudaBuffer<float4> contact_centers_;
  CudaBuffer<float4> contact_axes_;
  CudaBuffer<float4> contact_geometry_;
  CudaBuffer<uint2> contact_candidates_;
  CudaBuffer<cuda::ExternalConstraintGpu> external_constraints_;

  CudaBuffer<std::uint32_t> contact_counts_;
  CudaBuffer<std::uint32_t> contact_scan_a_;
  CudaBuffer<std::uint32_t> contact_scan_b_;
  const std::uint32_t* contact_inclusive_counts_{nullptr};

  CudaBuffer<std::uint64_t> contact_first_ids_;
  CudaBuffer<std::uint64_t> contact_second_ids_;
  CudaBuffer<std::uint32_t> contact_first_slots_;
  CudaBuffer<std::uint32_t> contact_second_slots_;
  CudaBuffer<std::uint32_t> contact_ordinals_;
  CudaBuffer<float4> contact_points_;
  CudaBuffer<float4> contact_normals_;
  CudaBuffer<float> contact_separations_;
  CudaBuffer<float> contact_weights_;

  CudaBuffer<cuda::MechanicsDofsGpu> mechanics_first_rows_;
  CudaBuffer<cuda::MechanicsDofsGpu> mechanics_second_rows_;
  CudaBuffer<float> mechanics_row_values_;
  CudaBuffer<float> mechanics_row_rhs_;
  CudaBuffer<std::uint32_t> mechanics_incidence_offsets_;
  CudaBuffer<std::uint32_t> mechanics_incidence_indices_;
  CudaBuffer<std::uint8_t> mechanics_fixed_;
  CudaBuffer<cuda::MechanicsDofsGpu> mechanics_solution_;
  CudaBuffer<cuda::MechanicsDofsGpu> mechanics_rhs_;
  CudaBuffer<cuda::MechanicsDofsGpu> mechanics_residual_;
  CudaBuffer<cuda::MechanicsDofsGpu> mechanics_search_;
  CudaBuffer<cuda::MechanicsDofsGpu> mechanics_applied_;
  CudaBuffer<float> mechanics_dot_terms_;
  CudaBuffer<float> mechanics_reduce_a_;
  CudaBuffer<float> mechanics_reduce_b_;
};

}  // namespace

std::unique_ptr<ComputeBackend> make_cuda_backend(std::uint32_t device_index) {
  return std::make_unique<CudaBackend>(device_index);
}

std::size_t cuda_backend_device_count() noexcept {
  int device_count = 0;
  const auto result = cudaGetDeviceCount(&device_count);
  if (result != cudaSuccess) {
    static_cast<void>(cudaGetLastError());
    return 0;
  }
  return device_count > 0 ? static_cast<std::size_t>(device_count) : 0;
}

}  // namespace cm
