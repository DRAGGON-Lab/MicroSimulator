#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/parallel.hpp"
#include "microsimulator/species.hpp"
#include "microsimulator/world_state.hpp"

namespace microsimulator {
namespace {

// Cells handed to one thread at a time.
constexpr std::size_t cell_grain = 256;

float evaluate_instruction(const RateInstruction& instruction, std::span<const float> workspace,
                           std::span<const float> species, const CellGeometryView& geometry,
                           const CellAttributeView& attributes, std::size_t cell,
                           float volume_change_rate) {
  switch (instruction.operation) {
    case RateOp::constant:
      return instruction.value;
    case RateOp::species:
      return species[instruction.first];
    case RateOp::signal:
      throw std::logic_error("standalone species rate plans cannot sample signals");
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
  throw std::logic_error("unknown species rate operation");
}

}  // namespace

void advance_species_cpu(WorldState& state, const SpeciesRatePlan& plan,
                         std::span<const float> previous_lengths, float dt) {
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
  if (!std::ranges::all_of(previous_lengths,
                           [](float value) { return std::isfinite(value) && value >= 0.0F; })) {
    throw std::invalid_argument("previous cell lengths must be finite and non-negative");
  }

  const auto geometry = state.geometry_state();
  const auto attributes = state.cell_attributes();
  auto species_state = state.species_state();
  const auto species_count = state.species_count();
  const auto instructions = plan.instructions();
  std::vector<float> next_levels(species_state.levels.begin(), species_state.levels.end());
  detail::parallel_for(state.size(), cell_grain, [&](std::size_t begin, std::size_t end) {
    std::vector<float> workspace(instructions.size());
    for (std::size_t cell = begin; cell < end; ++cell) {
      const auto radius = geometry.radii[cell];
      const auto previous_volume = effective_cell_volume(previous_lengths[cell], radius);
      const auto current_volume = effective_cell_volume(geometry.lengths[cell], radius);
      const auto dilution = previous_volume / current_volume;
      const auto cell_species = std::span(next_levels).subspan(cell * species_count, species_count);
      for (auto& level : cell_species) {
        level *= dilution;
      }

      const auto volume_change_rate = dt == 0.0F ? 0.0F : (current_volume - previous_volume) / dt;
      for (std::size_t index = 0; index < instructions.size(); ++index) {
        workspace[index] = evaluate_instruction(instructions[index], workspace, cell_species,
                                                geometry, attributes, cell, volume_change_rate);
        if (!std::isfinite(workspace[index])) {
          throw std::domain_error("species rate instruction " + std::to_string(index) +
                                  " produced a non-finite value");
        }
      }
      for (std::size_t species = 0; species < species_count; ++species) {
        cell_species[species] += dt * workspace[plan.outputs()[species]];
        if (!std::isfinite(cell_species[species])) {
          throw std::domain_error("species Euler update produced a non-finite level");
        }
      }
    }
  });
  std::ranges::copy(next_levels, species_state.levels.begin());
}

}  // namespace microsimulator
