#include <array>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "cm/simulation.hpp"

namespace {

bool close(float left, float right, float tolerance = 1.0e-6F) {
  return std::abs(left - right) <= tolerance;
}

cm::RateInstruction constant(float value) {
  return {.operation = cm::RateOp::constant, .value = value};
}

cm::RateInstruction species(std::uint32_t index) {
  return {.operation = cm::RateOp::species, .first = index};
}

void test_dilution_precedes_simultaneous_euler_update() {
  cm::Simulation simulation(cm::BackendKind::cpu, 1, 2);
  cm::CellInit cell;
  cell.length = 2.0F;
  cell.radius = 0.5F;
  cell.growth_rate = 0.5F;
  cell.species = {4.0F, 2.0F};
  const auto id = simulation.add_cell(cell);

  std::vector<cm::RateInstruction> instructions;
  instructions.push_back(constant(2.0F));
  instructions.push_back(species(0));
  instructions.push_back(constant(-0.5F));
  instructions.push_back({
      .operation = cm::RateOp::multiply,
      .first = 1,
      .second = 2,
  });
  simulation.set_species_rate_plan(cm::SpeciesRatePlan(2, std::move(instructions), {0, 3}));
  simulation.step(0.25F);

  const auto result = simulation.cell(id);
  const auto dilution = 3.0F / 3.25F;
  assert(close(result.length, 2.25F));
  assert(close(result.species[0], 4.0F * dilution + 0.5F));
  assert(close(result.species[1], 2.0F * dilution - 0.5F * (4.0F * dilution) * 0.25F));
  assert(simulation.supports(cm::BackendFeature::species));
  simulation.validate();
}

void test_division_preserves_concentration_and_schema() {
  cm::Simulation simulation(cm::BackendKind::cpu, 0, 2);
  cm::CellInit cell;
  cell.length = 4.0F;
  cell.species = {1.25F, -0.5F};
  const auto parent = simulation.add_cell(cell);
  const auto [first, second] = simulation.divide_equal(parent);

  assert(simulation.species_count() == 2);
  assert(simulation.cell(first).species == cell.species);
  assert(simulation.cell(second).species == cell.species);
  const std::array updated_species{2.0F, 3.0F};
  simulation.set_species(first, updated_species);
  assert(simulation.cell(first).species == std::vector<float>({2.0F, 3.0F}));
  assert(simulation.cell(second).species == cell.species);
  simulation.validate();
}

void test_invalid_species_contracts_fail_explicitly() {
  cm::Simulation simulation(cm::BackendKind::cpu, 0, 2);
  cm::CellInit cell;
  cell.species = {1.0F};
  bool rejected = false;

  try {
    static_cast<void>(simulation.add_cell(cell));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }

  assert(rejected);

  rejected = false;

  try {
    const std::vector<cm::RateInstruction> instructions{species(0)};
    static_cast<void>(cm::SpeciesRatePlan(2, instructions, {1, 0}));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }

  assert(rejected);
}

}  // namespace

int main() {
  test_dilution_precedes_simultaneous_euler_update();
  test_division_preserves_concentration_and_schema();
  test_invalid_species_contracts_fail_explicitly();

  return 0;
}
