#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "backend_devices.hpp"
#include "cm/simulation.hpp"

namespace {

constexpr float absolute_tolerance = 1.0e-6F;
constexpr float relative_tolerance = 1.0e-6F;
constexpr std::size_t cell_count = 513;
constexpr std::array time_steps{0.01F, 0.025F, 0.1F, 0.04F};

bool close(float actual, float expected) {
  const auto tolerance = absolute_tolerance + (relative_tolerance * std::abs(expected));

  return std::abs(actual - expected) <= tolerance;
}

void run_growth_scenario(cm::BackendKind backend, std::uint32_t device_index) {
  cm::Simulation simulation(backend, cell_count, 0, device_index);
  std::vector<cm::CellId> ids;
  std::vector<float> expected_lengths;
  std::vector<float> growth_rates;
  ids.reserve(cell_count);
  expected_lengths.reserve(cell_count);
  growth_rates.reserve(cell_count);

  simulation.step(0.0F);

  for (std::size_t index = 0; index < cell_count; ++index) {
    cm::CellInit cell;
    cell.position = {
        static_cast<float>(index) * 0.25F,
        static_cast<float>(index % 5) * -0.125F,
        static_cast<float>(index % 3) * 0.5F,
    };
    cell.length = 1.0F + static_cast<float>(index % 17) * 0.1F;
    cell.radius = 0.25F + static_cast<float>(index % 3) * 0.05F;
    cell.growth_rate = static_cast<float>(index % 11) * 0.025F;
    cell.cell_type = static_cast<std::int32_t>(index % 4);

    const auto id = simulation.add_cell(cell);
    assert(id == static_cast<cm::CellId>(index + 1));
    ids.push_back(id);
    expected_lengths.push_back(cell.length);
    growth_rates.push_back(cell.growth_rate);
  }

  double expected_time = 0.0;

  for (const auto dt : time_steps) {
    simulation.step(dt);
    expected_time += static_cast<double>(dt);

    for (std::size_t index = 0; index < cell_count; ++index) {
      expected_lengths[index] += growth_rates[index] * expected_lengths[index] * dt;
    }
  }

  assert(simulation.cell_count() == cell_count);
  assert(std::abs(simulation.time() - expected_time) <= 1.0e-12);

  for (std::size_t index = 0; index < cell_count; ++index) {
    const auto cell = simulation.cell(ids[index]);
    assert(cell.id == ids[index]);
    assert(cell.slot == static_cast<cm::Slot>(index));
    assert(close(cell.length, expected_lengths[index]));
    assert(cell.growth_rate == growth_rates[index]);
    assert(cell.cell_type == static_cast<std::int32_t>(index % 4));
  }

  const auto info = simulation.backend_info();
  assert(info.kind == backend);
  assert(info.device_index == device_index);
  assert(info.native);
  assert(!info.name.empty());
  assert(!info.device.empty());
  simulation.validate();
}

}  // namespace

int main() {
  cm::test::for_each_backend_device(run_growth_scenario);

  return 0;
}
