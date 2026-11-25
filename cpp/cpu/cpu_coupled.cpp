#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ranges>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/parallel.hpp"
#include "microsimulator/coupled_rates.hpp"
#include "microsimulator/signals.hpp"
#include "microsimulator/world_state.hpp"

namespace microsimulator {
namespace {

// Cells handed to one thread at a time.
constexpr std::size_t cell_grain = 256;

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
  detail::parallel_for(state.size(), cell_grain, [&](std::size_t begin, std::size_t end) {
    for (std::size_t cell = begin; cell < end; ++cell) {
      const auto stencil = signal_grid_stencil(
          grid.spec(),
          {geometry.position_x[cell], geometry.position_y[cell], geometry.position_z[cell]});
      if (stencil.entirely_solid) {
        throw std::invalid_argument("signal sample position is inside a grid obstacle");
      }
    }
  });
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

  const auto species_count = state.species_count();
  const auto signal_count = plan.signal_count();
  const auto instructions = plan.instructions();
  std::vector<SignalGridStencil> stencils(state.size());
  std::vector<float> next_species(species_state.levels.begin(), species_state.levels.end());
  // What each cell exchanges with the grid, as a concentration rate of every
  // signal.
  std::vector<float> exchange(state.size() * signal_count);
  detail::parallel_for(state.size(), cell_grain, [&](std::size_t begin, std::size_t end) {
    std::vector<float> sampled(signal_count);
    std::vector<float> workspace(instructions.size());
    for (std::size_t cell = begin; cell < end; ++cell) {
      const auto stencil = signal_grid_stencil(
          grid_spec,
          {geometry.position_x[cell], geometry.position_y[cell], geometry.position_z[cell]});
      stencils[cell] = stencil;
      std::ranges::fill(sampled, 0.0F);
      for (std::size_t entry = 0; entry < stencil.count; ++entry) {
        for (std::size_t signal = 0; signal < signal_count; ++signal) {
          sampled[signal] +=
              stencil.weights[entry] * old_grid[(signal * site_count) + stencil.sites[entry]];
        }
      }

      const auto previous_volume =
          effective_cell_volume(previous_lengths[cell], geometry.radii[cell]);
      const auto current_volume =
          effective_cell_volume(geometry.lengths[cell], geometry.radii[cell]);
      const auto dilution = previous_volume / current_volume;
      const auto cell_species =
          std::span(next_species).subspan(cell * species_count, species_count);
      for (auto& level : cell_species) {
        level *= dilution;
      }

      const auto volume_change_rate = dt == 0.0F ? 0.0F : (current_volume - previous_volume) / dt;
      for (std::size_t index = 0; index < instructions.size(); ++index) {
        workspace[index] =
            evaluate_instruction(instructions[index], workspace, cell_species, sampled, geometry,
                                 attributes, cell, volume_change_rate);
        if (!std::isfinite(workspace[index])) {
          throw std::domain_error("coupled rate instruction " + std::to_string(index) +
                                  " produced a non-finite value");
        }
      }
      for (std::size_t species = 0; species < species_count; ++species) {
        cell_species[species] += dt * workspace[plan.species_outputs()[species]];
        if (!std::isfinite(cell_species[species])) {
          throw std::domain_error("coupled species update produced a non-finite level");
        }
      }
      for (std::size_t signal = 0; signal < signal_count; ++signal) {
        exchange[(cell * signal_count) + signal] =
            workspace[plan.signal_outputs()[signal]] / grid_spec.voxel_volume();
      }
    }
  });

  // Cells share grid sites, so the sources are summed one cell at a time.
  std::vector<float> signal_sources(grid_spec.level_count(), 0.0F);
  for (std::size_t cell = 0; cell < state.size(); ++cell) {
    const auto& stencil = stencils[cell];
    for (std::size_t signal = 0; signal < signal_count; ++signal) {
      const auto concentration_rate = exchange[(cell * signal_count) + signal];
      for (std::size_t entry = 0; entry < stencil.count; ++entry) {
        signal_sources[(signal * site_count) + stencil.sites[entry]] +=
            stencil.weights[entry] * concentration_rate;
      }
    }
  }

  SignalSolveReport signal_report;
  std::vector<float> next_grid;
  if (grid_spec.integration != SignalIntegrationKind::forward_euler) {
    auto result = signal_grid_crank_nicolson_candidate(grid, dt, signal_sources);
    signal_report = result.report;
    if (!signal_report.converged) {
      throw std::runtime_error("Implicit coupled signal solve did not converge after " +
                               std::to_string(signal_report.iterations) + " iterations");
    }
    next_grid = std::move(result.levels);
  } else {
    next_grid = signal_grid_forward_euler_candidate(grid, dt);
    detail::parallel_for(next_grid.size(), 16384, [&](std::size_t begin, std::size_t end) {
      for (std::size_t index = begin; index < end; ++index) {
        next_grid[index] += dt * signal_sources[index];
      }
    });
  }
  SignalGridCheckpoint{.spec = grid_spec, .levels = next_grid}.validate();
  std::ranges::copy(next_species, species_state.levels.begin());
  grid.replace_levels(std::move(next_grid));
  return signal_report;
}

}  // namespace microsimulator
