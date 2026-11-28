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
#include <type_traits>
#include <utility>
#include <vector>

#include "core/boundary_values.hpp"
#include "core/contact_response.hpp"
#include "core/contact_solver.hpp"
#include "core/signal_jacobi.hpp"
#include "cuda_cell_kernels.cuh"
#include "cuda_flow.cuh"
#include "kernels/contact_dynamics.cuh"
#include "kernels/contacts.cuh"
#include "kernels/coupled_rates.cuh"
#include "kernels/growth.cuh"
#include "kernels/signals.cuh"
#include "kernels/species.cuh"
#include "microsimulator/backend.hpp"

namespace microsimulator {
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

  [[nodiscard]] T* data() noexcept {
    return data_;
  }
  [[nodiscard]] const T* data() const noexcept {
    return data_;
  }

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

  void advance_growth(WorldState& state, float dt) override {
    activate_device();
    if (!std::isfinite(dt) || dt < 0.0F) {
      throw std::invalid_argument("time step must be finite and non-negative");
    }
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
    grid.validate_step(dt);
    if (dt == 0.0F) {
      return {};
    }
    const auto& spec = grid.spec();
    const auto level_view = grid.levels();
    const std::vector<float> levels(level_view.begin(), level_view.end());
    const auto signal_count = spec.signal_count;
    const auto level_count = static_cast<std::uint32_t>(levels.size());

    const auto& statics = sync_grid(grid);
    const auto& boundaries = statics.boundaries;
    const auto& shape = statics.shape;
    const auto has_velocity_field = statics.has_velocity_field;
    const auto fluid = statics.fluid;
    signal_levels_.reserve(levels.size(), "signal-grid levels");
    signal_output_.reserve(levels.size(), "signal-grid output");
    signal_error_.reserve(1, "signal-grid error flag");
    clear_obstacle_levels(signal_output_, signal_output_cleared_, levels.size());
    copy_to_device(signal_levels_, levels, "failed to upload CUDA signal-grid levels");
    check_cuda(cudaMemsetAsync(signal_error_.data(), 0, sizeof(std::uint32_t), stream_),
               "failed to clear the CUDA signal-grid error flag");

    const auto crank_nicolson = static_cast<std::uint32_t>(spec.integration);
    cuda::launch_advance_signal_grid(
        signal_levels_.data(), signal_output_.data(), signal_diffusion_.data(),
        signal_advection_.data(), signal_fixed_values_.data(), signal_reaction_source_.data(),
        signal_reaction_loss_.data(), signal_obstacles_.data(), signal_x_faces_.data(),
        signal_y_faces_.data(), signal_z_faces_.data(), has_velocity_field, signal_error_.data(),
        boundaries, shape, make_float4(spec.spacing.x, spec.spacing.y, spec.spacing.z, 0.0F), dt,
        signal_count, fluid, crank_nicolson, stream_);
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
          signal_y_faces_.data(), signal_z_faces_.data(), has_velocity_field, signal_error_.data(),
          boundaries, shape, make_float4(spec.spacing.x, spec.spacing.y, spec.spacing.z, 0.0F),
          (crank_nicolson == 2 ? dt : 0.5F * dt), signal_count, level_count, spec.solver,
          grid.layout().implicit_operator_scale(crank_nicolson == 2 ? dt : 0.5F * dt));
      result_device = solve.first;
      report = solve.second;
      if (!report.converged) {
        throw std::runtime_error(
            signal_solve_failure("CUDA Implicit signal solve", spec, report.iterations));
      }
    }
    std::vector<float> output(levels.size());
    check_cuda(cudaMemcpyAsync(output.data(), result_device, output.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_),
               "failed to download CUDA signal-grid levels");
    check_cuda(cudaStreamSynchronize(stream_), "CUDA signal-grid download failed");
    grid.accept_step(std::move(output));
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
    const auto scatter_entry_count =
        checked_product(cell_count_size, cuda::scatter_stencil_sites, "scatter entry");
    for (const auto count : {cell_count_size, species_count_size, signal_count_size,
                             instruction_count_size, species_level_count, workspace_count,
                             cell_signal_count, grid_level_count, scatter_entry_count}) {
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
    const auto& statics = sync_grid(grid);
    const auto& boundaries = statics.boundaries;
    const auto& shape = statics.shape;
    const auto has_velocity_field = statics.has_velocity_field;
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
    check_cuda(cudaMemsetAsync(coupled_error_.data(), 0, sizeof(std::uint32_t), stream_),
               "failed to clear the CUDA coupled error flag");

    const auto crank_nicolson = static_cast<std::uint32_t>(spec.integration);
    const auto origin = make_float4(spec.origin.x, spec.origin.y, spec.origin.z, 0.0F);
    const auto spacing = make_float4(spec.spacing.x, spec.spacing.y, spec.spacing.z, 0.0F);
    const std::uint32_t* scatter_ends = nullptr;
    if (cell_count_size != 0) {
      coupled_scatter_stencils_.reserve(cell_count_size, "coupled scatter stencils");
      coupled_scatter_cells_.reserve(scatter_entry_count, "coupled scatter cell lists");
      coupled_scatter_counts_.reserve(spec.site_count(), "coupled scatter counts");
      coupled_scatter_cursors_.reserve(spec.site_count(), "coupled scatter cursors");
      coupled_scatter_scan_a_.reserve(spec.site_count(), "coupled scatter scan A");
      coupled_scatter_scan_b_.reserve(spec.site_count(), "coupled scatter scan B");
      check_cuda(cuda::launch_coupled_scatter_lists(
                     coupled_centers_.data(), coupled_scatter_stencils_.data(),
                     coupled_scatter_counts_.data(), coupled_scatter_cursors_.data(),
                     coupled_scatter_scan_a_.data(), coupled_scatter_scan_b_.data(),
                     coupled_scatter_cells_.data(), shape, origin, spacing,
                     static_cast<std::uint32_t>(cell_count_size), stream_, &scatter_ends),
                 "failed to launch the CUDA coupled scatter-list kernels");
    }
    check_cuda(
        cuda::launch_advance_coupled(
            coupled_species_levels_.data(), coupled_previous_lengths_.data(),
            coupled_centers_.data(), coupled_geometry_.data(), coupled_growth_rates_.data(),
            coupled_cell_types_.data(), coupled_instructions_.data(),
            coupled_species_outputs_.data(), coupled_signal_outputs_.data(),
            coupled_workspace_.data(), coupled_grid_levels_.data(), coupled_grid_output_.data(),
            signal_diffusion_.data(), signal_advection_.data(), signal_fixed_values_.data(),
            signal_reaction_source_.data(), signal_reaction_loss_.data(), signal_obstacles_.data(),
            signal_x_faces_.data(), signal_y_faces_.data(), signal_z_faces_.data(),
            has_velocity_field, coupled_cell_signal_rates_.data(), coupled_error_.data(),
            boundaries, shape, origin, spacing, dt, static_cast<std::uint32_t>(species_count_size),
            static_cast<std::uint32_t>(signal_count_size),
            static_cast<std::uint32_t>(instruction_count_size),
            static_cast<std::uint32_t>(cell_count_size),
            static_cast<std::uint32_t>(grid_level_count), crank_nicolson, scatter_ends,
            coupled_scatter_cells_.data(), stream_),
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
          coupled_grid_levels_.data(), coupled_grid_output_.data(), signal_diffusion_.data(),
          signal_advection_.data(), signal_fixed_values_.data(), signal_reaction_source_.data(),
          signal_reaction_loss_.data(), signal_obstacles_.data(), signal_x_faces_.data(),
          signal_y_faces_.data(), signal_z_faces_.data(), has_velocity_field, coupled_error_.data(),
          boundaries, shape, make_float4(spec.spacing.x, spec.spacing.y, spec.spacing.z, 0.0F),
          (crank_nicolson == 2 ? dt : 0.5F * dt), static_cast<std::uint32_t>(signal_count_size),
          static_cast<std::uint32_t>(grid_level_count), spec.solver,
          grid.layout().implicit_operator_scale(crank_nicolson == 2 ? dt : 0.5F * dt));
      result_device = solve.first;
      report = solve.second;
      if (!report.converged) {
        throw std::runtime_error(
            signal_solve_failure("CUDA Implicit coupled signal solve", spec, report.iterations));
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
    // The grid refuses levels it cannot hold, so it takes its own before the
    // species take theirs: a refused step changes neither.
    grid.accept_step(std::move(next_grid));
    std::ranges::copy(next_species, species_state.levels.begin());
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
    return cuda::external_contacts(state, constraints, parameters, false, stream_);
  }

  [[nodiscard]] ExternalContactGraph find_external_contact_faces(
      const WorldState& state, const ConstraintSet& constraints,
      const ConstraintContactParameters& parameters) override {
    activate_device();
    return cuda::external_contacts(state, constraints, parameters, true, stream_);
  }

  [[nodiscard]] std::vector<std::array<float, 6>> flow_drift_rates(const WorldState& state,
                                                                   const SignalGrid& grid,
                                                                   float dt) override {
    activate_device();
    return cuda::flow_drift_rates(state, grid, dt, stream_);
  }

  void advance_flow_drift(WorldState& state, const SignalGrid& grid, float dt,
                          const FlowDriftParameters& parameters) override {
    activate_device();
    cuda::advance_flow_drift(state, grid, dt, parameters, stream_);
  }

  [[nodiscard]] ContactSolveResult solve_contact_forces(
      const ContactSystem& system, std::span<const float> initial_multipliers,
      const ContactSolveParameters& parameters) override {
    activate_device();
    validate_contact_system(system);
    if (initial_multipliers.size() != system.rows.size()) {
      throw std::invalid_argument("contact multiplier count does not match the contact rows");
    }
    for (const auto multiplier : initial_multipliers) {
      if (!std::isfinite(multiplier) || multiplier < 0.0F) {
        throw std::invalid_argument("contact multipliers must be finite and non-negative");
      }
    }
    if (!std::isfinite(parameters.tolerance) || parameters.tolerance <= 0.0F) {
      throw std::invalid_argument("contact solve tolerance must be finite and positive");
    }

    ContactSolveResult result;
    result.multipliers.assign(initial_multipliers.begin(), initial_multipliers.end());
    // An incidence entry packs a row index with one bit for the row's side.
    if (system.bodies.size() >= std::numeric_limits<std::uint32_t>::max() ||
        system.rows.size() > std::numeric_limits<std::uint32_t>::max() / 2) {
      throw std::overflow_error("CUDA contact dynamics exceeds the uint32 index space");
    }
    const cuda::ContactSolveGpu solve{
        .dt = system.dt,
        .law = static_cast<std::uint32_t>(system.growth.law),
        .sensitivity = system.growth.sensitivity,
        .free_load = system.growth.free_load,
        .stall_load = system.growth.stall_load,
        .row_count = static_cast<std::uint32_t>(system.rows.size()),
        .body_count = static_cast<std::uint32_t>(system.bodies.size()),
    };
    if (solve.body_count == 0) {
      return result;
    }
    if (solve.row_count == 0) {
      // Without contacts every cell moves with its drift and grows freely.
      ensure_contact_dynamics_capacity(solve.body_count, 1);
      upload_contact_system(system, result.multipliers);
      cuda::launch_contact_respond(
          contact_rows_.data(), contact_bodies_.data(), contact_multipliers_[0].data(),
          contact_incidence_offsets_.data(), contact_incidence_entries_.data(),
          contact_response_.data(), solve, stream_);
      check_cuda(cudaGetLastError(), "failed to launch the CUDA contact-respond kernel");
      result.response = download_contact_response(solve.body_count);
      return result;
    }
    const auto bound = contact_system_diagonal_bound(system);
    if (!std::isfinite(bound) || bound <= 0.0F) {
      throw std::invalid_argument("contact system has no positive diagonal");
    }

    ensure_contact_dynamics_capacity(solve.body_count, solve.row_count);
    upload_contact_system(system, result.multipliers);
    if (solve.row_count <= contact_solve_block_rows) {
      // One block runs the whole solve, so the host waits once.
      contact_state_.reserve(1, "contact solve state");
      cuda::launch_contact_solve(
          contact_rows_.data(), contact_bodies_.data(), contact_multipliers_[0].data(),
          contact_multipliers_[1].data(), contact_gaps_[0].data(), contact_gaps_[1].data(),
          contact_incidence_offsets_.data(), contact_incidence_entries_.data(),
          contact_response_.data(), contact_state_.data(), solve, parameters.tolerance,
          parameters.max_iterations, 1.0F / bound, stream_);
      check_cuda(cudaGetLastError(), "failed to launch the CUDA contact-solve kernel");
      cuda::ContactSolveStateGpu state{};
      check_cuda(cudaMemcpyAsync(&state, contact_state_.data(), sizeof(state),
                                 cudaMemcpyDeviceToHost, stream_),
                 "failed to download the CUDA contact solve state");
      check_cuda(cudaStreamSynchronize(stream_), "CUDA contact solve failed");
      result.report = detail::device_contact_solve_report(state.iterations, state.residual,
                                                          parameters.tolerance);
      copy_to_host(result.multipliers, contact_multipliers_[state.iterations % 2],
                   "failed to download CUDA contact multipliers");
      result.response = download_contact_response(solve.body_count);
      return result;
    }

    // The multipliers and their gaps live in the buffer pair `current` names;
    // each step writes the other pair. The solve's state stays on the device,
    // so the host waits once for each batch of steps.
    std::uint32_t current = 0;
    std::uint32_t taken_total = 0;
    contact_state_.reserve(1, "contact solve state");
    float* const multipliers[2] = {contact_multipliers_[0].data(), contact_multipliers_[1].data()};
    float* const gaps[2] = {contact_gaps_[0].data(), contact_gaps_[1].data()};
    const auto read_state = [&] {
      cuda::ContactSolveStateGpu state{};
      check_cuda(cudaMemcpyAsync(&state, contact_state_.data(), sizeof(state),
                                 cudaMemcpyDeviceToHost, stream_),
                 "failed to download the CUDA contact solve state");
      check_cuda(cudaStreamSynchronize(stream_), "CUDA contact solve failed");
      return state;
    };
    const detail::ContactSolverOperations operations{
        .initialize =
            [&](float first_step_length) {
              const auto residual = evaluate_contact_step(solve, 0, 0, true).residual;
              const cuda::ContactSolveStateGpu state{
                  .step_length = first_step_length,
                  .residual = residual,
                  .iterations = 0,
                  .done = !(residual > parameters.tolerance) || !std::isfinite(residual) ? 1U : 0U,
              };
              check_cuda(cudaMemcpyAsync(contact_state_.data(), &state, sizeof(state),
                                         cudaMemcpyHostToDevice, stream_),
                         "failed to upload the CUDA contact solve state");
              return residual;
            },
        .iterate =
            [&](std::uint32_t count) {
              cuda::launch_contact_steps(
                  count, current, contact_rows_.data(), contact_bodies_.data(), multipliers, gaps,
                  contact_incidence_offsets_.data(), contact_incidence_entries_.data(),
                  contact_response_.data(), contact_reduce_[0].data(), contact_state_.data(), solve,
                  parameters.tolerance, stream_);
              check_cuda(cudaGetLastError(), "failed to launch the CUDA contact step kernels");
              const auto state = read_state();
              const auto taken = state.iterations - taken_total;
              taken_total = state.iterations;
              current = (current + taken) % 2;
              return detail::ContactSolverProgress{.iterations = taken, .residual = state.residual};
            },
        .batch = contact_solve_batch,
    };
    result.report = detail::solve_contact_system(operations, 1.0F / bound, parameters);
    copy_to_host(result.multipliers, contact_multipliers_[current],
                 "failed to download CUDA contact multipliers");
    check_cuda(cudaStreamSynchronize(stream_), "CUDA contact multiplier download failed");
    // The respond kernel last ran on the multipliers the solve returns.
    result.response = download_contact_response(solve.body_count);
    return result;
  }

  [[nodiscard]] DepthAveragedFlowResult solve_depth_averaged_flow(
      const SignalGridSpec& spec, std::span<const float> mobility,
      const DepthAveragedFlowParameters& parameters) override {
    activate_device();
    return cuda::solve_depth_averaged_flow(spec, mobility, parameters, stream_);
  }

  [[nodiscard]] DepthAveragedPortFlowResult solve_depth_averaged_port_flow(
      const SignalGridSpec& spec, std::span<const FlowPort> ports, std::span<const float> mobility,
      const DepthAveragedPortFlowParameters& parameters) override {
    activate_device();
    return cuda::solve_depth_averaged_port_flow(spec, ports, mobility, parameters, stream_);
  }

  [[nodiscard]] ResolvedFlowResult solve_resolved_flow(
      const SignalGridSpec& spec, std::span<const float> drag,
      const ResolvedFlowParameters& parameters) override {
    activate_device();
    return cuda::solve_resolved_flow(spec, drag, parameters, stream_);
  }

  [[nodiscard]] ResolvedPortFlowResult solve_resolved_port_flow(
      const SignalGridSpec& spec, std::span<const FlowPort> ports, std::span<const float> drag,
      const ResolvedPortFlowParameters& parameters) override {
    activate_device();
    return cuda::solve_resolved_port_flow(spec, ports, drag, parameters, stream_);
  }

  [[nodiscard]] BiomassDeposit deposit_biomass(
      const SignalGridSpec& spec, const DepositCells& cells,
      const BiomassDepositParameters& parameters) override {
    activate_device();
    return cuda::deposit_biomass(spec, cells, parameters, stream_);
  }

  [[nodiscard]] ShallowFlowCoupling shallow_flow_coupling(
      const SignalGridSpec& spec, std::span<const float> base, const BiomassDeposit& deposit,
      const BiomassDragParameters& parameters) override {
    activate_device();
    return cuda::shallow_flow_coupling(spec, base, deposit, parameters, stream_);
  }

  [[nodiscard]] std::vector<float> brinkman_drag(const SignalGridSpec& spec,
                                                 const BiomassDeposit& deposit,
                                                 const BiomassDragParameters& parameters) override {
    activate_device();
    return cuda::brinkman_drag(spec, deposit, parameters, stream_);
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

    check_cuda(cudaMemcpyAsync(contact_ids_.data(), geometry.ids.data(), geometry.ids.size_bytes(),
                               cudaMemcpyHostToDevice, stream_),
               "failed to upload CUDA contact cell IDs");
    check_cuda(cudaMemcpyAsync(contact_centers_.data(), centers.data(),
                               centers.size() * sizeof(float4), cudaMemcpyHostToDevice, stream_),
               "failed to upload CUDA contact cell centers");
    check_cuda(cudaMemcpyAsync(contact_axes_.data(), axes.data(), axes.size() * sizeof(float4),
                               cudaMemcpyHostToDevice, stream_),
               "failed to upload CUDA contact cell axes");
    check_cuda(cudaMemcpyAsync(contact_geometry_.data(), shapes.data(),
                               shapes.size() * sizeof(float4), cudaMemcpyHostToDevice, stream_),
               "failed to upload CUDA contact cell geometry");
    check_cuda(cudaStreamSynchronize(stream_), "CUDA contact cell upload failed");
  }

  void upload_contact_candidates(std::span<const ContactCandidate> candidates) {
    std::vector<uint2> values;
    values.reserve(candidates.size());
    for (const auto& candidate : candidates) {
      values.push_back(make_uint2(candidate.first_slot, candidate.second_slot));
    }
    check_cuda(cudaMemcpyAsync(contact_candidates_.data(), values.data(),
                               values.size() * sizeof(uint2), cudaMemcpyHostToDevice, stream_),
               "failed to upload CUDA contact candidates");
    check_cuda(cudaStreamSynchronize(stream_), "CUDA contact candidate upload failed");
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

  // What the transport kernels are told of the grid on the device.
  struct GridStatics {
    cuda::SignalGridBoundariesGpu boundaries{};
    cuda::SignalGridShapeGpu shape{};
    std::uint32_t has_velocity_field{0};
    cuda::SignalFluidGpu fluid{};
  };

  // Which allocation a buffer was last cleared in, and for which obstacles.
  struct ObstacleClearance {
    const float* data{nullptr};
    std::uint64_t stamp{0};
  };

  // The transport kernels write fluid levels only, so a buffer they write
  // must hold zero wherever the grid has an obstacle. A buffer does once it
  // has been cleared for the obstacles the device holds.
  void clear_obstacle_levels(CudaBuffer<float>& buffer, ObstacleClearance& cleared,
                             std::size_t level_count) {
    if (cleared.data == buffer.data() && cleared.stamp == obstacle_stamp_) {
      return;
    }
    check_cuda(cudaMemsetAsync(buffer.data(), 0, level_count * sizeof(float), stream_),
               "failed to clear a CUDA signal-grid buffer");
    cleared = {.data = buffer.data(), .stamp = obstacle_stamp_};
  }

  // Puts on the device everything transport reads of a grid besides its
  // levels: coefficients, reaction field, obstacles, face velocities,
  // boundary values and the list of fluid sites. They change only with the
  // grid's specification, so a step whose grid the device already holds
  // uploads nothing.
  const GridStatics& sync_grid(const SignalGrid& grid) {
    if (grid.revision() == grid_revision_) {
      return grid_statics_;
    }
    grid_revision_ = 0;
    const auto& spec = grid.spec();
    const auto level_count = spec.level_count();
    const auto signal_count = spec.signal_count;
    signal_diffusion_.reserve(signal_count, "signal-grid diffusion");
    signal_advection_.reserve(signal_count, "signal-grid advection");
    signal_reaction_source_.reserve(level_count, "signal-grid affine sources");
    signal_reaction_loss_.reserve(level_count, "signal-grid affine losses");
    signal_obstacles_.reserve(spec.site_count(), "signal-grid obstacles");
    signal_x_faces_.reserve(std::max<std::size_t>(spec.x_face_count(), 1), "signal x faces");
    signal_y_faces_.reserve(std::max<std::size_t>(spec.y_face_count(), 1), "signal y faces");
    signal_z_faces_.reserve(std::max<std::size_t>(spec.z_face_count(), 1), "signal z faces");

    std::vector<float4> advection;
    advection.reserve(signal_count);
    for (const auto velocity : spec.advection) {
      advection.push_back(make_float4(velocity.x, velocity.y, velocity.z, 0.0F));
    }
    const auto boundary_table = detail::fixed_boundary_table(spec);
    signal_fixed_values_.reserve(boundary_table.values.size(), "signal-grid boundary values");
    std::vector<float> reaction_source(level_count, 0.0F);
    std::vector<float> reaction_loss(level_count, 0.0F);
    if (spec.reaction.has_value()) {
      reaction_source = spec.reaction->source_rates;
      reaction_loss = spec.reaction->loss_rates;
    }
    std::vector<std::uint8_t> obstacles(spec.site_count(), 0);
    if (spec.has_obstacles()) {
      obstacles = spec.obstacles;
    }
    const auto& fluid = grid.layout().fluid_sites;
    signal_fluid_sites_.reserve(std::max<std::size_t>(fluid.size(), 1), "signal-grid fluid sites");

    copy_to_device(signal_diffusion_, spec.diffusion,
                   "failed to upload CUDA signal-grid diffusion");
    copy_to_device(signal_advection_, advection, "failed to upload CUDA signal-grid advection");
    copy_to_device(signal_fixed_values_, boundary_table.values,
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
    if (!fluid.empty()) {
      copy_to_device(signal_fluid_sites_, fluid, "failed to upload CUDA signal-grid fluid sites");
    }
    // The uploads read host arrays that end with this call.
    check_cuda(cudaStreamSynchronize(stream_), "CUDA signal-grid upload failed");
    if (obstacles != grid_obstacles_) {
      grid_obstacles_ = std::move(obstacles);
      ++obstacle_stamp_;
    }

    grid_statics_.boundaries = {
        .x_lower = static_cast<std::uint32_t>(spec.x_lower.kind),
        .x_upper = static_cast<std::uint32_t>(spec.x_upper.kind),
        .y_lower = static_cast<std::uint32_t>(spec.y_lower.kind),
        .y_upper = static_cast<std::uint32_t>(spec.y_upper.kind),
        .z_lower = static_cast<std::uint32_t>(spec.z_lower.kind),
        .z_upper = static_cast<std::uint32_t>(spec.z_upper.kind),
        .offsets = {},
        .advection_scheme = static_cast<std::uint32_t>(spec.advection_scheme),
    };
    std::copy(boundary_table.offsets.begin(), boundary_table.offsets.end(),
              grid_statics_.boundaries.offsets);
    grid_statics_.shape = {
        .x = spec.shape.x,
        .y = spec.shape.y,
        .z = spec.shape.z,
        .sites = static_cast<std::uint32_t>(spec.site_count()),
    };
    grid_statics_.has_velocity_field = static_cast<std::uint32_t>(spec.velocity_field.has_value());
    grid_statics_.fluid = {.sites = signal_fluid_sites_.data(),
                           .count = static_cast<std::uint32_t>(fluid.size())};
    grid_revision_ = grid.revision();
    return grid_statics_;
  }

  void ensure_signal_solve_capacity(std::uint32_t level_count) {
    signal_cn_a_.reserve(level_count, "signal Jacobi field A");
    signal_cn_b_.reserve(level_count, "signal Jacobi field B");
    clear_obstacle_levels(signal_cn_a_, signal_cn_a_cleared_, level_count);
    clear_obstacle_levels(signal_cn_b_, signal_cn_b_cleared_, level_count);
    signal_cn_terms_.reserve(level_count, "signal residual terms");
    signal_cn_reduce_a_.reserve(level_count, "signal reduction A");
    signal_cn_reduce_b_.reserve(level_count, "signal reduction B");
  }

  // Sums the terms the last residual or norm kernel wrote, one per fluid level.
  [[nodiscard]] float reduce_signal_terms(std::uint32_t term_count, const char* operation) {
    if (term_count == 0) {
      check_cuda(cudaStreamSynchronize(stream_), operation);
      return 0.0F;
    }
    const float* input = signal_cn_terms_.data();
    float* output = signal_cn_reduce_a_.data();
    auto element_count = term_count;
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

  [[nodiscard]] float signal_residual_rms(
      const float* current, const float* right_hand_side, const float* diffusion,
      const float4* advection, const float* fixed_values, const float* reaction_source,
      const float* reaction_loss, const std::uint8_t* obstacles, const float* x_faces,
      const float* y_faces, const float* z_faces, std::uint32_t has_velocity_field,
      cuda::SignalGridBoundariesGpu boundaries, cuda::SignalGridShapeGpu shape, float4 spacing,
      float half_dt, std::uint32_t signal_count, std::uint32_t level_count) {
    cuda::launch_signal_crank_nicolson_residual_terms(
        current, right_hand_side, signal_cn_terms_.data(), diffusion, advection, fixed_values,
        reaction_source, reaction_loss, obstacles, x_faces, y_faces, z_faces, has_velocity_field,
        boundaries, shape, spacing, half_dt, signal_count, grid_statics_.fluid, stream_);
    check_cuda(cudaGetLastError(), "failed to launch the CUDA signal-residual kernel");
    return std::sqrt(reduce_signal_terms(signal_count * grid_statics_.fluid.count,
                                         "CUDA signal residual failed") /
                     static_cast<float>(level_count));
  }

  [[nodiscard]] float signal_rhs_rms(const float* right_hand_side, std::uint32_t signal_count,
                                     std::uint32_t level_count) {
    cuda::launch_signal_square_terms(right_hand_side, signal_cn_terms_.data(), signal_count,
                                     grid_statics_.shape.sites, grid_statics_.fluid, stream_);
    check_cuda(cudaGetLastError(), "failed to launch the CUDA signal-norm kernel");
    return std::sqrt(
        reduce_signal_terms(signal_count * grid_statics_.fluid.count, "CUDA signal norm failed") /
        static_cast<float>(level_count));
  }

  [[nodiscard]] std::pair<const float*, SignalSolveReport> solve_signal_crank_nicolson(
      const float* initial, const float* right_hand_side, const float* diffusion,
      const float4* advection, const float* fixed_values, const float* reaction_source,
      const float* reaction_loss, const std::uint8_t* obstacles, const float* x_faces,
      const float* y_faces, const float* z_faces, std::uint32_t has_velocity_field,
      std::uint32_t* error, cuda::SignalGridBoundariesGpu boundaries,
      cuda::SignalGridShapeGpu shape, float4 spacing, float dt, std::uint32_t signal_count,
      std::uint32_t level_count, const SignalSolveParameters& parameters, float operator_scale) {
    ensure_signal_solve_capacity(level_count);
    const auto half_dt = dt;
    const float* current = initial;
    const detail::SignalJacobiOperations operations{
        .initial_residual_rms =
            [&] {
              return signal_residual_rms(initial, right_hand_side, diffusion, advection,
                                         fixed_values, reaction_source, reaction_loss, obstacles,
                                         x_faces, y_faces, z_faces, has_velocity_field, boundaries,
                                         shape, spacing, half_dt, signal_count, level_count);
            },
        .right_hand_side_rms =
            [&] { return signal_rhs_rms(right_hand_side, signal_count, level_count); },
        .operator_scale = operator_scale,
        .begin_sweeps =
            [&] {
              check_cuda(cudaMemsetAsync(error, 0, sizeof(std::uint32_t), stream_),
                         "failed to clear the CUDA signal solver error flag");
            },
        .sweep_batch =
            [&](std::uint32_t count, float threshold) {
              signal_sweep_state_.reserve(1, "signal sweep state");
              auto* state = signal_sweep_state_.data();
              check_cuda(cudaMemsetAsync(state, 0, sizeof(cuda::SignalSweepStateGpu), stream_),
                         "failed to clear the CUDA signal sweep state");
              const auto* gate = &state->done;
              const auto term_count = signal_count * grid_statics_.fluid.count;
              // Sweeps write the two scratch iterates in turn, never the one the
              // solve started from.
              std::vector<const float*> iterates{current};
              for (std::uint32_t sweep = 0; sweep < count; ++sweep) {
                const float* from = iterates.back();
                float* to = from == signal_cn_a_.data() ? signal_cn_b_.data() : signal_cn_a_.data();
                iterates.push_back(to);
                cuda::launch_signal_crank_nicolson_jacobi(
                    from, to, right_hand_side, diffusion, advection, fixed_values, reaction_source,
                    reaction_loss, obstacles, x_faces, y_faces, z_faces, has_velocity_field, error,
                    boundaries, shape, spacing, half_dt, signal_count, grid_statics_.fluid, stream_,
                    gate);
                check_cuda(cudaGetLastError(), "failed to launch the CUDA signal Jacobi kernel");
                cuda::launch_signal_crank_nicolson_residual_terms(
                    to, right_hand_side, signal_cn_terms_.data(), diffusion, advection,
                    fixed_values, reaction_source, reaction_loss, obstacles, x_faces, y_faces,
                    z_faces, has_velocity_field, boundaries, shape, spacing, half_dt, signal_count,
                    grid_statics_.fluid, stream_, gate);
                check_cuda(cudaGetLastError(), "failed to launch the CUDA signal-residual kernel");
                const float* input = signal_cn_terms_.data();
                float* output = signal_cn_reduce_a_.data();
                auto element_count = term_count;
                while (element_count > 1) {
                  cuda::launch_reduce_sum_pairs(input, output, element_count, stream_, gate);
                  check_cuda(cudaGetLastError(),
                             "failed to launch the CUDA signal-reduction kernel");
                  input = output;
                  output = output == signal_cn_reduce_a_.data() ? signal_cn_reduce_b_.data()
                                                                : signal_cn_reduce_a_.data();
                  element_count = (element_count + 1) / 2;
                }
                cuda::launch_signal_sweep_check(input, error, state, threshold, level_count,
                                                term_count, stream_);
                check_cuda(cudaGetLastError(), "failed to launch the CUDA sweep-check kernel");
              }
              cuda::SignalSweepStateGpu result{};
              check_cuda(
                  cudaMemcpyAsync(&result, state, sizeof(result), cudaMemcpyDeviceToHost, stream_),
                  "failed to download the CUDA signal sweep state");
              check_cuda(cudaStreamSynchronize(stream_), "CUDA signal Jacobi batch failed");
              current = iterates[result.iterations];
              return detail::SignalSweepBatch{
                  .iterations = result.iterations,
                  .residual_rms = result.residual_rms,
                  .valid = result.invalid == 0,
              };
            },
        .batch = signal_sweep_batch,
    };
    const auto report = detail::solve_signal_jacobi(operations, parameters);
    return {current, report};
  }

  void ensure_contact_dynamics_capacity(std::size_t body_count, std::size_t row_count) {
    contact_bodies_.reserve(body_count, "contact bodies");
    contact_response_.reserve(body_count, "contact response");
    contact_incidence_offsets_.reserve(body_count + 1, "contact incidence offsets");
    contact_rows_.reserve(row_count, "contact rows");
    contact_incidence_entries_.reserve(row_count * 2, "contact incidence entries");
    contact_terms_.reserve(row_count, "contact step terms");
    for (std::size_t index = 0; index < 2; ++index) {
      contact_multipliers_[index].reserve(row_count, "contact multipliers");
      contact_gaps_[index].reserve(row_count, "contact gaps");
      contact_reduce_[index].reserve(row_count, "contact reduction");
    }
  }

  void upload_contact_system(const ContactSystem& system,
                             const std::vector<float>& initial_multipliers) {
    const auto body_count = system.bodies.size();
    std::vector<cuda::ContactRowGpu> rows(system.rows.size());
    std::vector<std::uint32_t> offsets(body_count + 1, 0U);
    for (std::size_t index = 0; index < system.rows.size(); ++index) {
      const auto& row = system.rows[index];
      const auto& first = row.first_jacobian;
      const auto& second = row.second_jacobian;
      rows[index] = {
          .first_linear = make_float4(first[0], first[1], first[2], row.first_length_lever),
          .first_angular = make_float4(first[3], first[4], first[5], row.offset),
          .second_linear = make_float4(second[0], second[1], second[2], row.second_length_lever),
          .second_angular = make_float4(second[3], second[4], second[5], row.compliance),
          .first_slot = row.first_slot,
          .second_slot = row.second_slot,
          .padding0 = 0,
          .padding1 = 0,
      };
      ++offsets[row.first_slot + 1];
      if (row.second_slot != invalid_slot) {
        ++offsets[row.second_slot + 1];
      }
    }
    for (std::size_t body = 0; body < body_count; ++body) {
      offsets[body + 1] += offsets[body];
    }
    std::vector<std::uint32_t> entries(offsets[body_count]);
    std::vector<std::uint32_t> cursors(offsets.begin(), offsets.end() - 1);
    for (std::size_t index = 0; index < system.rows.size(); ++index) {
      const auto& row = system.rows[index];
      const auto entry = static_cast<std::uint32_t>(index) << 1U;
      entries[cursors[row.first_slot]++] = entry;
      if (row.second_slot != invalid_slot) {
        entries[cursors[row.second_slot]++] = entry | 1U;
      }
    }
    std::vector<cuda::ContactBodyGpu> bodies(body_count);
    for (std::size_t body = 0; body < body_count; ++body) {
      const auto& source = system.bodies[body];
      bodies[body] = {
          .axis_mobility = make_float4(source.axis.x, source.axis.y, source.axis.z,
                                       source.translational_mobility),
          .rates = make_float4(source.transverse_mobility, source.axial_mobility,
                               source.length_rate, 0.0F),
          .drift_linear = make_float4(source.drift[0], source.drift[1], source.drift[2], 0.0F),
          .drift_angular = make_float4(source.drift[3], source.drift[4], source.drift[5], 0.0F),
      };
    }
    copy_to_device(contact_rows_, rows, "failed to upload CUDA contact rows");
    copy_to_device(contact_bodies_, bodies, "failed to upload CUDA contact bodies");
    copy_to_device(contact_incidence_offsets_, offsets,
                   "failed to upload CUDA contact incidence offsets");
    copy_to_device(contact_incidence_entries_, entries,
                   "failed to upload CUDA contact incidence entries");
    copy_to_device(contact_multipliers_[0], initial_multipliers,
                   "failed to upload CUDA contact multipliers");
    check_cuda(cudaStreamSynchronize(stream_), "CUDA contact system upload failed");
  }

  // Evaluates the response to the multipliers in pair `target`, then their
  // gaps and step terms against pair `previous`, and reduces the terms.
  [[nodiscard]] detail::ContactSolverSums evaluate_contact_step(const cuda::ContactSolveGpu& solve,
                                                                std::size_t target,
                                                                std::size_t previous,
                                                                bool initial) {
    cuda::launch_contact_respond(
        contact_rows_.data(), contact_bodies_.data(), contact_multipliers_[target].data(),
        contact_incidence_offsets_.data(), contact_incidence_entries_.data(),
        contact_response_.data(), solve, stream_);
    check_cuda(cudaGetLastError(), "failed to launch the CUDA contact-respond kernel");
    cuda::launch_contact_gaps(contact_rows_.data(), contact_response_.data(),
                              contact_multipliers_[target].data(), contact_gaps_[target].data(),
                              contact_multipliers_[previous].data(), contact_gaps_[previous].data(),
                              contact_terms_.data(), solve, initial, stream_);
    check_cuda(cudaGetLastError(), "failed to launch the CUDA contact-gaps kernel");
    const float4* input = contact_terms_.data();
    std::size_t output = 0;
    auto element_count = solve.row_count;
    while (element_count > 1) {
      cuda::launch_contact_reduce(input, contact_reduce_[output].data(), element_count, stream_);
      check_cuda(cudaGetLastError(), "failed to launch the CUDA contact-reduce kernel");
      input = contact_reduce_[output].data();
      output = 1 - output;
      element_count = (element_count + 1) / 2;
    }
    float4 reduced{};
    check_cuda(cudaMemcpyAsync(&reduced, input, sizeof(reduced), cudaMemcpyDeviceToHost, stream_),
               "failed to download a CUDA contact reduction");
    check_cuda(cudaStreamSynchronize(stream_), "CUDA contact step failed");
    return {
        .step_dot_step = reduced.x,
        .step_dot_change = reduced.y,
        .change_dot_change = reduced.z,
        .residual = reduced.w,
    };
  }

  template <typename T>
  void copy_to_device(CudaBuffer<T>& destination, const std::vector<T>& source,
                      const char* operation) {
    // The backend stream is nonblocking; a default-stream cudaMemcpy can race its kernels.
    check_cuda(cudaMemcpyAsync(destination.data(), source.data(), source.size() * sizeof(T),
                               cudaMemcpyHostToDevice, stream_),
               operation);
  }

  [[nodiscard]] ContactResponse download_contact_response(std::uint32_t body_count) {
    std::vector<device::ContactResponse> packed(body_count);
    static_assert(sizeof(device::ContactResponse) == sizeof(cuda::ContactResponseGpu));
    check_cuda(cudaMemcpyAsync(packed.data(), contact_response_.data(),
                               packed.size() * sizeof(cuda::ContactResponseGpu),
                               cudaMemcpyDeviceToHost, stream_),
               "failed to download the CUDA contact response");
    check_cuda(cudaStreamSynchronize(stream_), "CUDA contact response download failed");
    return detail::unpack_contact_response(packed);
  }

  template <typename T>
  void copy_to_host(std::vector<T>& destination, const CudaBuffer<T>& source,
                    const char* operation) {
    check_cuda(cudaMemcpyAsync(destination.data(), source.data(), destination.size() * sizeof(T),
                               cudaMemcpyDeviceToHost, stream_),
               operation);
  }

  // Systems of up to this many rows are solved in one block. Above it, the
  // solve spreads over the whole device a batch of steps at a time.
  static constexpr std::uint32_t contact_solve_block_rows = 4096;
  // Steps launched before the host waits, above that size.
  static constexpr std::uint32_t contact_solve_batch = 32;
  // Implicit signal sweeps launched before the host waits. A step takes about
  // a dozen.
  static constexpr std::uint32_t signal_sweep_batch = 16;

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
  CudaBuffer<std::uint32_t> signal_fluid_sites_;
  CudaBuffer<std::uint32_t> signal_error_;
  // The grid specification the transport buffers hold, zero for none, and
  // what the kernels are told of it.
  std::uint64_t grid_revision_{0};
  GridStatics grid_statics_;
  // The obstacle mask on the device, and a number that changes with it.
  std::vector<std::uint8_t> grid_obstacles_;
  std::uint64_t obstacle_stamp_{0};
  ObstacleClearance signal_output_cleared_;
  ObstacleClearance signal_cn_a_cleared_;
  ObstacleClearance signal_cn_b_cleared_;
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
  CudaBuffer<uint2> coupled_scatter_stencils_;
  CudaBuffer<std::uint32_t> coupled_scatter_cells_;
  CudaBuffer<std::uint32_t> coupled_scatter_counts_;
  CudaBuffer<std::uint32_t> coupled_scatter_cursors_;
  CudaBuffer<std::uint32_t> coupled_scatter_scan_a_;
  CudaBuffer<std::uint32_t> coupled_scatter_scan_b_;
  CudaBuffer<float> coupled_grid_levels_;
  CudaBuffer<float> coupled_grid_output_;
  CudaBuffer<std::uint32_t> coupled_error_;

  CudaBuffer<std::uint64_t> contact_ids_;
  CudaBuffer<float4> contact_centers_;
  CudaBuffer<float4> contact_axes_;
  CudaBuffer<float4> contact_geometry_;
  CudaBuffer<uint2> contact_candidates_;

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

  CudaBuffer<cuda::ContactRowGpu> contact_rows_;
  CudaBuffer<cuda::ContactBodyGpu> contact_bodies_;
  CudaBuffer<cuda::ContactResponseGpu> contact_response_;
  CudaBuffer<cuda::ContactSolveStateGpu> contact_state_;
  CudaBuffer<cuda::SignalSweepStateGpu> signal_sweep_state_;
  CudaBuffer<std::uint32_t> contact_incidence_offsets_;
  CudaBuffer<std::uint32_t> contact_incidence_entries_;
  CudaBuffer<float4> contact_terms_;
  std::array<CudaBuffer<float>, 2> contact_multipliers_;
  std::array<CudaBuffer<float>, 2> contact_gaps_;
  std::array<CudaBuffer<float4>, 2> contact_reduce_;
};

// A backend that leaves out any provider method is abstract.
static_assert(!std::is_abstract_v<CudaBackend>, "the CUDA backend must implement every provider");

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

}  // namespace microsimulator
