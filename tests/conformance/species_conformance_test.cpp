#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "backend_devices.hpp"
#include "cm/simulation.hpp"

namespace {

constexpr std::size_t cell_count = 513;
constexpr std::size_t species_count = 3;
constexpr std::array time_steps{0.0F, 0.01F, 0.025F, 0.1F};

bool close(float actual, float expected) {
  constexpr float absolute_tolerance = 2.0e-5F;
  constexpr float relative_tolerance = 2.0e-5F;

  return std::abs(actual - expected) <=
         absolute_tolerance + relative_tolerance * std::abs(expected);
}

cm::SpeciesRatePlan make_plan() {
  using enum cm::RateOp;

  return cm::SpeciesRatePlan(species_count,
                             {
                                 {.operation = species, .first = 0},
                                 {.operation = species, .first = 1},
                                 {.operation = species, .first = 2},
                                 {.operation = constant, .value = 0.25F},
                                 {.operation = position_x},
                                 {.operation = position_y},
                                 {.operation = position_z},
                                 {.operation = cell_length},
                                 {.operation = cell_radius},
                                 {.operation = growth_rate},
                                 {.operation = cell_type},
                                 {.operation = cell_volume},
                                 {.operation = cell_surface_area},
                                 {.operation = add, .first = 0, .second = 1},
                                 {.operation = subtract, .first = 13, .second = 2},
                                 {.operation = multiply, .first = 14, .second = 3},
                                 {.operation = constant, .value = 2.0F},
                                 {.operation = divide, .first = 15, .second = 16},
                                 {.operation = power, .first = 8, .second = 16},
                                 {.operation = minimum, .first = 17, .second = 18},
                                 {.operation = maximum, .first = 19, .second = 3},
                                 {.operation = negate, .first = 9},
                                 {.operation = exponential, .first = 21},
                                 {.operation = constant, .value = 1.0F},
                                 {.operation = add, .first = 0, .second = 23},
                                 {.operation = logarithm, .first = 24},
                                 {.operation = less, .first = 4, .second = 7},
                                 {.operation = less_equal, .first = 5, .second = 6},
                                 {.operation = greater, .first = 7, .second = 8},
                                 {.operation = greater_equal, .first = 10, .second = 3},
                                 {.operation = equal, .first = 10, .second = 16},
                                 {.operation = select, .first = 30, .second = 22, .third = 25},
                                 {.operation = add, .first = 20, .second = 31},
                                 {.operation = select, .first = 26, .second = 27, .third = 28},
                                 {.operation = select, .first = 29, .second = 33, .third = 30},
                                 {.operation = divide, .first = 12, .second = 11},
                                 {.operation = add, .first = 35, .second = 4},
                                 {.operation = add, .first = 36, .second = 5},
                                 {.operation = add, .first = 37, .second = 6},
                                 {.operation = add, .first = 38, .second = 18},
                             },
                             {32, 34, 39});
}

void populate(cm::Simulation& simulation) {
  simulation.set_species_rate_plan(make_plan());

  for (std::size_t index = 0; index < cell_count; ++index) {
    cm::CellInit cell;
    cell.position = {static_cast<float>(index % 19) * 0.03F, static_cast<float>(index % 7) * -0.02F,
                     static_cast<float>(index % 5) * 0.01F};
    cell.length = 1.0F + static_cast<float>(index % 13) * 0.1F;
    cell.radius = 0.25F + static_cast<float>(index % 3) * 0.05F;
    cell.growth_rate = static_cast<float>(index % 11) * 0.025F;
    cell.cell_type = static_cast<std::int32_t>(index % 4);
    cell.species = {
        0.1F + static_cast<float>(index % 17) * 0.02F,
        0.2F + static_cast<float>(index % 23) * 0.01F,
        static_cast<float>(index % 9) * -0.03F,
    };
    assert(simulation.add_cell(cell) == static_cast<cm::CellId>(index + 1));
  }
}

void compare(const cm::Simulation& actual, const cm::Simulation& expected) {
  assert(actual.cell_count() == expected.cell_count());
  assert(actual.species_count() == expected.species_count());
  assert(std::abs(actual.time() - expected.time()) <= 1.0e-12);
  const auto actual_cells = actual.cells();
  const auto expected_cells = expected.cells();

  for (std::size_t cell = 0; cell < expected_cells.size(); ++cell) {
    assert(actual_cells[cell].id == expected_cells[cell].id);
    assert(actual_cells[cell].slot == expected_cells[cell].slot);
    assert(close(actual_cells[cell].length, expected_cells[cell].length));
    assert(actual_cells[cell].species.size() == species_count);

    for (std::size_t species = 0; species < species_count; ++species) {
      assert(close(actual_cells[cell].species[species], expected_cells[cell].species[species]));
    }
  }
}

void run_scenario(cm::BackendKind backend, std::uint32_t device_index) {
  cm::Simulation reference(cm::BackendKind::cpu, cell_count, species_count);
  cm::Simulation candidate(backend, cell_count, species_count, device_index);
  populate(reference);
  populate(candidate);

  for (const auto dt : time_steps) {
    reference.step(dt);
    candidate.step(dt);
  }

  compare(candidate, reference);
  candidate.validate();
}

void run_non_finite_rejection(cm::BackendKind backend, std::uint32_t device_index) {
  cm::Simulation simulation(backend, 1, 1, device_index);
  cm::CellInit cell;
  cell.growth_rate = 0.0F;
  cell.species = {1.0F};
  const auto id = simulation.add_cell(cell);
  using enum cm::RateOp;
  simulation.set_species_rate_plan(
      cm::SpeciesRatePlan(1,
                          {
                              {.operation = constant, .value = 0.0F},
                              {.operation = divide, .first = 0, .second = 0},
                          },
                          {1}));
  bool rejected = false;

  try {
    simulation.step(0.1F);
  } catch (const std::domain_error&) {
    rejected = true;
  }

  assert(rejected);
  assert(simulation.time() == 0.0);
  assert(simulation.cell(id).species == cell.species);
}

}  // namespace

int main() {
  cm::test::for_each_backend_device([](cm::BackendKind backend, std::uint32_t device_index) {
    cm::Simulation probe(backend, 0, 0, device_index);

    if (probe.supports(cm::BackendFeature::species)) {
      run_scenario(backend, device_index);
      run_non_finite_rejection(backend, device_index);
    }
  });

  return 0;
}
