#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "microsimulator/simulation.hpp"

namespace {

bool close(float left, float right, float tolerance = 1.0e-6F) {
  return std::abs(left - right) <= tolerance;
}

microsimulator::SignalGridSpec grid_spec(microsimulator::GridShape shape,
                                         microsimulator::Vec3 spacing = {1.0F, 1.0F, 1.0F},
                                         float diffusion = 0.0F) {
  microsimulator::SignalGridSpec spec;
  spec.signal_count = 1;
  spec.shape = shape;
  spec.spacing = spacing;
  spec.diffusion = {diffusion};
  spec.advection = {{0.0F, 0.0F, 0.0F}};
  return spec;
}

microsimulator::RateInstruction constant(float value) {
  return {.operation = microsimulator::RateOp::constant, .value = value};
}

microsimulator::RateInstruction signal(std::uint32_t index) {
  return {.operation = microsimulator::RateOp::signal, .first = index};
}

template <typename Exception, typename Function>
void assert_throws(Function&& function) {
  bool rejected = false;
  try {
    function();
  } catch (const Exception&) {
    rejected = true;
  }
  assert(rejected);
}

void test_sample_and_scatter_share_trilinear_weights() {
  microsimulator::Simulation simulation(microsimulator::BackendKind::cpu, 1, 1);
  simulation.configure_signal_grid(grid_spec({2, 2, 2}, {2.0F, 1.0F, 1.0F}),
                                   std::vector<float>(8, 4.0F));
  microsimulator::CellInit cell;
  cell.position = {1.0F, 0.5F, 0.5F};
  cell.growth_rate = 0.0F;
  cell.species = {10.0F};
  const auto id = simulation.add_cell(cell);
  simulation.set_coupled_rate_plan(
      microsimulator::CoupledRatePlan(1, 1, {signal(0), constant(-2.0F)}, {0}, {1}));

  simulation.step(0.5F);

  assert(close(simulation.cell(id).species[0], 12.0F));
  float grid_amount = 0.0F;
  for (const auto level : simulation.signal_levels()) {
    assert(close(level, 3.9375F));
    grid_amount += level * 2.0F;
  }
  assert(close(grid_amount, 63.0F));
  assert(simulation.supports(microsimulator::BackendFeature::coupled_rates));
}

void test_old_grid_sampling_follows_post_growth_dilution() {
  microsimulator::Simulation simulation(microsimulator::BackendKind::cpu, 1, 1);
  simulation.configure_signal_grid(grid_spec({3, 1, 1}, {1.0F, 1.0F, 1.0F}, 1.0F),
                                   {0.0F, 2.0F, 0.0F});
  microsimulator::CellInit cell;
  cell.position = {1.0F, 0.0F, 0.0F};
  cell.length = 2.0F;
  cell.radius = 0.5F;
  cell.growth_rate = 1.0F;
  cell.species = {4.0F};
  const auto id = simulation.add_cell(cell);
  simulation.set_coupled_rate_plan(
      microsimulator::CoupledRatePlan(1, 1, {signal(0), constant(0.0F)}, {0}, {1}));

  simulation.step(0.25F);

  assert(close(simulation.cell(id).length, 2.5F));
  assert(close(simulation.cell(id).species[0], (4.0F * 3.0F / 3.5F) + 0.5F));
  const auto levels = simulation.signal_levels();
  assert(close(levels[0], 0.5F));
  assert(close(levels[1], 1.0F));
  assert(close(levels[2], 0.5F));
}

void test_crank_nicolson_includes_coupled_sources() {
  microsimulator::Simulation simulation(microsimulator::BackendKind::cpu, 1, 0);
  auto spec = grid_spec({1, 1, 1});
  spec.integration = microsimulator::SignalIntegrationKind::crank_nicolson;
  simulation.configure_signal_grid(spec, {1.0F});
  microsimulator::CellInit cell;
  cell.growth_rate = 0.0F;
  simulation.add_cell(cell);
  simulation.set_coupled_rate_plan(
      microsimulator::CoupledRatePlan(0, 1, {constant(2.0F)}, {}, {0}));

  simulation.step(1.0F);

  assert(close(simulation.signal_levels()[0], 3.0F));
  assert(simulation.last_signal_solve_report().has_value());
  assert(simulation.last_signal_solve_report()->converged);
}

void test_invalid_position_is_rejected_before_growth() {
  microsimulator::Simulation simulation(microsimulator::BackendKind::cpu, 1, 1);
  simulation.configure_signal_grid(grid_spec({2, 1, 1}), {1.0F, 1.0F});
  microsimulator::CellInit cell;
  cell.position = {2.0F, 0.0F, 0.0F};
  cell.length = 2.0F;
  cell.growth_rate = 1.0F;
  cell.species = {3.0F};
  const auto id = simulation.add_cell(cell);
  simulation.set_coupled_rate_plan(
      microsimulator::CoupledRatePlan(1, 1, {constant(0.0F)}, {0}, {0}));

  assert_throws<std::out_of_range>([&] { simulation.step(0.25F); });
  assert(close(simulation.cell(id).length, 2.0F));
  assert(close(simulation.cell(id).species[0], 3.0F));
  assert(simulation.signal_levels() == std::vector<float>({1.0F, 1.0F}));
  assert(simulation.time() == 0.0);
}

void test_coupled_plan_is_exact_checkpoint_state() {
  microsimulator::Simulation simulation(microsimulator::BackendKind::cpu, 1, 1);
  simulation.configure_signal_grid(grid_spec({1, 1, 1}), {2.0F});
  microsimulator::CellInit cell;
  cell.growth_rate = 0.0F;
  cell.species = {1.0F};
  simulation.add_cell(cell);
  simulation.set_coupled_rate_plan(
      microsimulator::CoupledRatePlan(1, 1, {signal(0), constant(0.0F)}, {0}, {1}));

  const auto checkpoint = simulation.checkpoint();
  assert(checkpoint.coupled_rate_plan.has_value());
  microsimulator::Simulation restored(microsimulator::BackendKind::cpu, checkpoint);
  assert(restored.has_coupled_rate_plan());
  restored.step(0.5F);
  assert(close(restored.cells()[0].species[0], 2.0F));
  restored.clear_coupled_rate_plan();
  assert(!restored.has_coupled_rate_plan());
}

void test_signal_inputs_are_reserved_for_coupled_plans() {
  assert_throws<std::invalid_argument>(
      [&] { static_cast<void>(microsimulator::SpeciesRatePlan(1, {signal(0)}, {0})); });
  assert_throws<std::invalid_argument>(
      [&] { static_cast<void>(microsimulator::CoupledRatePlan(1, 1, {signal(1)}, {0}, {0})); });
}

}  // namespace

int main() {
  test_sample_and_scatter_share_trilinear_weights();
  test_old_grid_sampling_follows_post_growth_dilution();
  test_crank_nicolson_includes_coupled_sources();
  test_invalid_position_is_rejected_before_growth();
  test_coupled_plan_is_exact_checkpoint_state();
  test_signal_inputs_are_reserved_for_coupled_plans();
  return 0;
}
