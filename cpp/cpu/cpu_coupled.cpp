#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ranges>
#include <stdexcept>
#include <string>
#include <vector>

#include "cm/coupled_rates.hpp"
#include "cm/signals.hpp"
#include "cm/world_state.hpp"

namespace cm {
namespace {

float evaluate_instruction(const RateInstruction& instruction, std::span<const float> workspace,
                           std::span<const float> species, std::span<const float> signals,
                           const CellGeometryView& geometry, const CellAttributeView& attributes,
                           std::size_t cell, float volume_change_rate) {
  switch (instruction.operation) {
    case RateOp::constant:
      return instruction.value;
    case RateOp::species:
      return species[instruction.first];
    case RateOp::signal:
      return signals[instruction.first];
    case RateOp::position_x:
      return geometry.position_x[cell];
    case RateOp::position_y:
      return geometry.position_y[cell];
    case RateOp::position_z:
      return geometry.position_z[cell];
    case RateOp::cell_length:
      return geometry.lengths[cell];
    case RateOp::cell_radius:
      return geometry.radii[cell];
    case RateOp::growth_rate:
      return attributes.growth_rates[cell];
    case RateOp::cell_type:
      return static_cast<float>(attributes.cell_types[cell]);
    case RateOp::cell_volume_change_rate:
      return volume_change_rate;
    case RateOp::cell_volume:
      return effective_cell_volume(geometry.lengths[cell], geometry.radii[cell]);
    case RateOp::cell_surface_area:
      return effective_cell_surface_area(geometry.lengths[cell], geometry.radii[cell]);
    case RateOp::add:
      return workspace[instruction.first] + workspace[instruction.second];
    case RateOp::subtract:
      return workspace[instruction.first] - workspace[instruction.second];
    case RateOp::multiply:
      return workspace[instruction.first] * workspace[instruction.second];
    case RateOp::divide:
      return workspace[instruction.first] / workspace[instruction.second];
    case RateOp::power:
      return std::pow(workspace[instruction.first], workspace[instruction.second]);
    case RateOp::minimum:
      return std::min(workspace[instruction.first], workspace[instruction.second]);
    case RateOp::maximum:
      return std::max(workspace[instruction.first], workspace[instruction.second]);
    case RateOp::negate:
      return -workspace[instruction.first];
    case RateOp::exponential:
      return std::exp(workspace[instruction.first]);
    case RateOp::logarithm:
      return std::log(workspace[instruction.first]);
    case RateOp::less:
      return workspace[instruction.first] < workspace[instruction.second] ? 1.0F : 0.0F;
    case RateOp::less_equal:
      return workspace[instruction.first] <= workspace[instruction.second] ? 1.0F : 0.0F;
    case RateOp::greater:
      return workspace[instruction.first] > workspace[instruction.second] ? 1.0F : 0.0F;
    case RateOp::greater_equal:
      return workspace[instruction.first] >= workspace[instruction.second] ? 1.0F : 0.0F;
    case RateOp::equal:
      return workspace[instruction.first] == workspace[instruction.second] ? 1.0F : 0.0F;
    case RateOp::select:
      return workspace[instruction.first] != 0.0F ? workspace[instruction.second]
                                                  : workspace[instruction.third];
  }
  throw std::logic_error("unknown coupled rate operation");
}

}  // namespace

void validate_coupled_step(const WorldState& state, const SignalGrid& grid,
                           const CoupledRatePlan& plan, std::span<const float> previous_lengths,
                           float dt) {
  if (!std::isfinite(dt) || dt < 0.0F) {
    throw std::invalid_argument("coupled time step must be finite and non-negative");
  }
  state.validate();
  grid.validate();
  grid.validate_step(dt);
  plan.validate();
  if (plan.species_count() != state.species_count()) {
    throw std::invalid_argument("coupled rate plan and world state species counts disagree");
  }
  if (plan.signal_count() != grid.spec().signal_count) {
    throw std::invalid_argument("coupled rate plan and signal grid counts disagree");
  }
  if (previous_lengths.size() != state.size()) {
    throw std::invalid_argument("previous cell lengths and world state cell counts disagree");
  }
  if (!std::ranges::all_of(previous_lengths,
                           [](float value) { return std::isfinite(value) && value >= 0.0F; })) {
    throw std::invalid_argument("previous cell lengths must be finite and non-negative");
  }
  const auto geometry = state.geometry_state();
  for (std::size_t cell = 0; cell < state.size(); ++cell) {
    const auto stencil = signal_grid_stencil(
        grid.spec(),
        {geometry.position_x[cell], geometry.position_y[cell], geometry.position_z[cell]});
    if (stencil.entirely_solid) {
      throw std::invalid_argument("signal sample position is inside a grid obstacle");
    }
  }
}

SignalSolveReport advance_coupled_cpu(WorldState& state, SignalGrid& grid,
                                      const CoupledRatePlan& plan,
                                      std::span<const float> previous_lengths, float dt) {
  validate_coupled_step(state, grid, plan, previous_lengths, dt);

  const auto geometry = state.geometry_state();
  const auto attributes = state.cell_attributes();
  auto species_state = state.species_state();
  const auto& grid_spec = grid.spec();
  const auto old_grid = grid.levels();
  const auto site_count = grid_spec.site_count();

  std::vector<SignalGridStencil> stencils;
  stencils.reserve(state.size());
  std::vector<float> sampled(state.size() * plan.signal_count(), 0.0F);
  for (std::size_t cell = 0; cell < state.size(); ++cell) {
    const auto stencil = signal_grid_stencil(
        grid_spec,
        {geometry.position_x[cell], geometry.position_y[cell], geometry.position_z[cell]});
    stencils.push_back(stencil);
    for (std::size_t entry = 0; entry < stencil.count; ++entry) {
      for (std::size_t signal = 0; signal < plan.signal_count(); ++signal) {
        sampled[(cell * plan.signal_count()) + signal] +=
            stencil.weights[entry] * old_grid[(signal * site_count) + stencil.sites[entry]];
      }
    }
  }

  std::vector<float> next_species(species_state.levels.begin(), species_state.levels.end());
  for (std::size_t cell = 0; cell < state.size(); ++cell) {
    const auto previous_volume =
        effective_cell_volume(previous_lengths[cell], geometry.radii[cell]);
    const auto current_volume = effective_cell_volume(geometry.lengths[cell], geometry.radii[cell]);
    const auto dilution = previous_volume / current_volume;
    const auto offset = cell * state.species_count();
    for (std::size_t species = 0; species < state.species_count(); ++species) {
      next_species[offset + species] *= dilution;
    }
  }

  std::vector<float> signal_sources(grid_spec.level_count(), 0.0F);
  std::vector<float> workspace(plan.instructions().size());
  for (std::size_t cell = 0; cell < state.size(); ++cell) {
    const auto species_offset = cell * state.species_count();
    const auto cell_species =
        std::span<const float>(next_species).subspan(species_offset, state.species_count());
    const auto cell_signals =
        std::span<const float>(sampled).subspan(cell * plan.signal_count(), plan.signal_count());
    for (std::size_t index = 0; index < plan.instructions().size(); ++index) {
      workspace[index] = evaluate_instruction(
          plan.instructions()[index], workspace, cell_species, cell_signals, geometry, attributes,
          cell,
          dt == 0.0F ? 0.0F
                     : (effective_cell_volume(geometry.lengths[cell], geometry.radii[cell]) -
                        effective_cell_volume(previous_lengths[cell], geometry.radii[cell])) /
                           dt);
      if (!std::isfinite(workspace[index])) {
        throw std::domain_error("coupled rate instruction " + std::to_string(index) +
                                " produced a non-finite value");
      }
    }
    for (std::size_t species = 0; species < state.species_count(); ++species) {
      next_species[species_offset + species] += dt * workspace[plan.species_outputs()[species]];
      if (!std::isfinite(next_species[species_offset + species])) {
        throw std::domain_error("coupled species update produced a non-finite level");
      }
    }
    const auto& stencil = stencils[cell];
    for (std::size_t signal = 0; signal < plan.signal_count(); ++signal) {
      const auto concentration_rate =
          workspace[plan.signal_outputs()[signal]] / grid_spec.voxel_volume();
      for (std::size_t entry = 0; entry < stencil.count; ++entry) {
        signal_sources[(signal * site_count) + stencil.sites[entry]] +=
            stencil.weights[entry] * concentration_rate;
      }
    }
  }

  SignalSolveReport signal_report;
  std::vector<float> next_grid;
  if (grid_spec.integration == SignalIntegrationKind::crank_nicolson) {
    auto result = signal_grid_crank_nicolson_candidate(grid, dt, signal_sources);
    signal_report = result.report;
    if (!signal_report.converged) {
      throw std::runtime_error("Crank-Nicolson coupled signal solve did not converge after " +
                               std::to_string(signal_report.iterations) + " iterations");
    }
    next_grid = std::move(result.levels);
  } else {
    next_grid = signal_grid_forward_euler_candidate(grid, dt);
    for (std::size_t index = 0; index < next_grid.size(); ++index) {
      next_grid[index] += dt * signal_sources[index];
    }
  }
  SignalGridCheckpoint{.spec = grid_spec, .levels = next_grid}.validate();
  std::ranges::copy(next_species, species_state.levels.begin());
  grid.replace_levels(std::move(next_grid));
  return signal_report;
}

}  // namespace cm
