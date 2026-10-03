#include <cassert>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "backend_devices.hpp"
#include "cm/simulation.hpp"

namespace {

constexpr float absolute_tolerance = 1.0e-6F;
constexpr float relative_tolerance = 1.0e-6F;

bool close(float actual, float expected) {
  const auto tolerance = absolute_tolerance + (relative_tolerance * std::abs(expected));

  return std::abs(actual - expected) <= tolerance;
}

void assert_missing(const cm::Simulation& simulation, cm::CellId id) {
  bool rejected = false;

  try {
    static_cast<void>(simulation.cell(id));
  } catch (const std::out_of_range&) {
    rejected = true;
  }

  assert(rejected);
}

void run_lifecycle_scenario(cm::BackendKind backend, std::uint32_t device_index) {
  cm::Simulation simulation(backend, 0, 0, device_index);
  cm::CellInit initial;
  initial.position = {2.0F, 3.0F, 1.0F};
  initial.direction = {2.0F, 0.0F, 0.0F};
  initial.length = 4.0F;
  initial.radius = 0.5F;
  initial.growth_rate = 0.25F;
  initial.cell_type = 7;

  const auto parent = simulation.add_cell(initial);
  simulation.step(0.5F);
  const auto [first, second] = simulation.divide_equal(parent);
  assert(parent == 1);
  assert(first == 2);
  assert(second == 3);
  assert_missing(simulation, parent);

  auto first_cell = simulation.cell(first);
  auto second_cell = simulation.cell(second);
  assert(first_cell.slot == 0);
  assert(second_cell.slot == 1);
  assert(close(first_cell.length, 1.75F));
  assert(close(second_cell.length, 1.75F));
  assert(close(first_cell.position.x, 0.625F));
  assert(close(second_cell.position.x, 3.375F));
  assert(simulation.lineage_parent(first) == parent);
  assert(simulation.lineage_parent(second) == parent);

  simulation.step(0.2F);
  const auto [third, fourth] = simulation.divide_equal(second);
  assert(third == 4);
  assert(fourth == 5);
  assert_missing(simulation, second);
  assert(simulation.lineage_parent(third) == second);
  assert(simulation.lineage_parent(fourth) == second);
  assert(simulation.lineage_parent(second) == parent);

  simulation.step(0.1F);
  const auto cells = simulation.cells();
  assert(cells.size() == 3);
  assert(cells[0].id == first);
  assert(cells[1].id == third);
  assert(cells[2].id == fourth);
  assert(cells[0].slot == 0);
  assert(cells[1].slot == 1);
  assert(cells[2].slot == 2);
  assert(close(cells[0].length, 1.8834375F));
  assert(close(cells[1].length, 0.42921875F));
  assert(close(cells[2].length, 0.42921875F));
  assert(close(cells[1].position.x, 2.665625F));
  assert(close(cells[2].position.x, 4.084375F));

  for (const auto& cell : cells) {
    assert(cell.cell_type == 7);
    assert(cell.growth_rate == 0.25F);
    assert(close(cell.direction.x, 1.0F));
    assert(close(cell.direction.y, 0.0F));
    assert(close(cell.direction.z, 0.0F));
  }

  const auto expected_time =
      static_cast<double>(0.5F) + static_cast<double>(0.2F) + static_cast<double>(0.1F);
  assert(std::abs(simulation.time() - expected_time) <= 1.0e-12);
  assert(simulation.backend_info().kind == backend);
  simulation.validate();

  cm::Simulation asymmetric(backend, 0, 0, device_index);
  initial.length = 6.0F;
  initial.radius = 0.5F;
  const auto asymmetric_parent = asymmetric.add_cell(initial);
  const auto [short_daughter, long_daughter] = asymmetric.divide(asymmetric_parent, 0.25F);
  assert(close(asymmetric.cell(short_daughter).length, 1.25F));
  assert(close(asymmetric.cell(long_daughter).length, 3.75F));
  assert(close(asymmetric.cell(short_daughter).position.x, -0.375F));
  assert(close(asymmetric.cell(long_daughter).position.x, 3.125F));
  assert(asymmetric.lineage_parent(short_daughter) == asymmetric_parent);
  assert(asymmetric.lineage_parent(long_daughter) == asymmetric_parent);
  asymmetric.validate();
}

}  // namespace

int main() {
  cm::test::for_each_backend_device(run_lifecycle_scenario);

  return 0;
}
