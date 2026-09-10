#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "cm/species.hpp"
#include "cm/world_state.hpp"

namespace cm {
namespace {

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

  const auto geometry = state.geometry_state();
  const auto attributes = state.cell_attributes();
  auto species_state = state.species_state();
  std::vector<float> next_levels(species_state.levels.begin(), species_state.levels.end());
  for (std::size_t cell = 0; cell < state.size(); ++cell) {
    if (!std::isfinite(previous_lengths[cell]) || previous_lengths[cell] < 0.0F) {
      throw std::invalid_argument("previous cell lengths must be finite and non-negative");
    }
    const auto radius = geometry.radii[cell];
    const auto previous_volume = effective_cell_volume(previous_lengths[cell], radius);
    const auto current_volume = effective_cell_volume(geometry.lengths[cell], radius);
    const auto dilution = previous_volume / current_volume;
    const auto offset = cell * state.species_count();
    for (std::size_t species = 0; species < state.species_count(); ++species) {
      next_levels[offset + species] *= dilution;
    }
  }

  std::vector<float> workspace(plan.instructions().size());
  for (std::size_t cell = 0; cell < state.size(); ++cell) {
    const auto offset = cell * state.species_count();
    const auto cell_species =
        std::span<const float>(next_levels).subspan(offset, state.species_count());
    for (std::size_t index = 0; index < plan.instructions().size(); ++index) {
      workspace[index] = evaluate_instruction(
          plan.instructions()[index], workspace, cell_species, geometry, attributes, cell,
          dt == 0.0F ? 0.0F
                     : (effective_cell_volume(geometry.lengths[cell], geometry.radii[cell]) -
                        effective_cell_volume(previous_lengths[cell], geometry.radii[cell])) /
                           dt);
      if (!std::isfinite(workspace[index])) {
        throw std::domain_error("species rate instruction " + std::to_string(index) +
                                " produced a non-finite value");
      }
    }
    for (std::size_t species = 0; species < state.species_count(); ++species) {
      const auto rate = workspace[plan.outputs()[species]];
      next_levels[offset + species] += dt * rate;
      if (!std::isfinite(next_levels[offset + species])) {
        throw std::domain_error("species Euler update produced a non-finite level");
      }
    }
  }
  std::ranges::copy(next_levels, species_state.levels.begin());
}

}  // namespace cm
