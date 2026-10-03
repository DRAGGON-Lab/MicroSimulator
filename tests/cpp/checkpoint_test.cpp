#include "cm/checkpoint.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "backend_devices.hpp"
#include "cm/simulation.hpp"

namespace {

cm::SpeciesRatePlan make_rate_plan() {
  using enum cm::RateOp;

  return cm::SpeciesRatePlan(2,
                             {
                                 {.operation = species, .first = 0},
                                 {.operation = species, .first = 1},
                                 {.operation = constant, .value = 0.125F},
                                 {.operation = add, .first = 0, .second = 2},
                                 {.operation = negate, .first = 1},
                             },
                             {3, 4});
}

void assert_cells_equal(const cm::CellSnapshot& actual, const cm::CellSnapshot& expected) {
  assert(actual.id == expected.id);
  assert(actual.slot == expected.slot);
  assert(actual.position.x == expected.position.x);
  assert(actual.position.y == expected.position.y);
  assert(actual.position.z == expected.position.z);
  assert(actual.direction.x == expected.direction.x);
  assert(actual.direction.y == expected.direction.y);
  assert(actual.direction.z == expected.direction.z);
  assert(actual.length == expected.length);
  assert(actual.radius == expected.radius);
  assert(actual.growth_rate == expected.growth_rate);
  assert(actual.cell_type == expected.cell_type);
  assert(actual.species == expected.species);
}

void assert_rate_plans_equal(const cm::SpeciesRatePlan& actual,
                             const cm::SpeciesRatePlan& expected) {
  assert(actual.species_count() == expected.species_count());
  assert(actual.outputs().size() == expected.outputs().size());

  for (std::size_t index = 0; index < actual.outputs().size(); ++index) {
    assert(actual.outputs()[index] == expected.outputs()[index]);
  }

  assert(actual.instructions().size() == expected.instructions().size());

  for (std::size_t index = 0; index < actual.instructions().size(); ++index) {
    const auto& left = actual.instructions()[index];
    const auto& right = expected.instructions()[index];
    assert(left.operation == right.operation);
    assert(left.first == right.first);
    assert(left.second == right.second);
    assert(left.third == right.third);
    assert(left.value == right.value);
  }
}

void assert_checkpoints_equal(const cm::SimulationCheckpoint& actual,
                              const cm::SimulationCheckpoint& expected) {
  assert(actual.schema_version == expected.schema_version);
  assert(actual.time == expected.time);
  assert(actual.world.species_count == expected.world.species_count);
  assert(actual.world.next_id == expected.world.next_id);
  assert(actual.world.lineage.size() == expected.world.lineage.size());

  for (std::size_t index = 0; index < actual.world.lineage.size(); ++index) {
    assert(actual.world.lineage[index].child == expected.world.lineage[index].child);
    assert(actual.world.lineage[index].parent == expected.world.lineage[index].parent);
  }

  assert(actual.world.cells.size() == expected.world.cells.size());

  for (std::size_t index = 0; index < actual.world.cells.size(); ++index) {
    assert_cells_equal(actual.world.cells[index], expected.world.cells[index]);
  }

  assert(actual.constraints.next_id == expected.constraints.next_id);
  assert(actual.constraints.planes.size() == expected.constraints.planes.size());
  assert(actual.constraints.spheres.size() == expected.constraints.spheres.size());

  for (std::size_t index = 0; index < actual.constraints.planes.size(); ++index) {
    const auto& left = actual.constraints.planes[index];
    const auto& right = expected.constraints.planes[index];
    assert(left.id == right.id);
    assert(left.point.x == right.point.x);
    assert(left.point.y == right.point.y);
    assert(left.point.z == right.point.z);
    assert(left.inward_normal.x == right.inward_normal.x);
    assert(left.inward_normal.y == right.inward_normal.y);
    assert(left.inward_normal.z == right.inward_normal.z);
    assert(left.coefficient == right.coefficient);
  }

  for (std::size_t index = 0; index < actual.constraints.spheres.size(); ++index) {
    const auto& left = actual.constraints.spheres[index];
    const auto& right = expected.constraints.spheres[index];
    assert(left.id == right.id);
    assert(left.center.x == right.center.x);
    assert(left.center.y == right.center.y);
    assert(left.center.z == right.center.z);
    assert(left.radius == right.radius);
    assert(left.coefficient == right.coefficient);
    assert(left.allowed_region == right.allowed_region);
  }

  assert_rate_plans_equal(actual.species_rate_plan, expected.species_rate_plan);
}

void assert_resumed_step_close(const cm::Simulation& actual, const cm::Simulation& expected) {
  assert(actual.time() == expected.time());
  const auto actual_cells = actual.cells();
  const auto expected_cells = expected.cells();
  assert(actual_cells.size() == expected_cells.size());

  for (std::size_t index = 0; index < actual_cells.size(); ++index) {
    assert(actual_cells[index].id == expected_cells[index].id);
    assert(actual_cells[index].slot == expected_cells[index].slot);
    assert(std::abs(actual_cells[index].length - expected_cells[index].length) <= 1.0e-6F);
    assert(actual_cells[index].species.size() == expected_cells[index].species.size());

    for (std::size_t species = 0; species < actual_cells[index].species.size(); ++species) {
      assert(std::abs(actual_cells[index].species[species] -
                      expected_cells[index].species[species]) <= 1.0e-5F);
    }
  }
}

template <typename Function>
void assert_invalid(Function&& function) {
  bool rejected = false;

  try {
    function();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }

  assert(rejected);
}

}  // namespace

int main() {
  cm::Simulation original(cm::BackendKind::cpu, 8, 2);
  original.set_species_rate_plan(make_rate_plan());

  cm::CellInit first;
  first.position = {1.25F, -2.5F, 0.75F};
  first.direction = {0.0F, 1.0F, 0.0F};
  first.length = 4.5F;
  first.radius = 0.4F;
  first.growth_rate = 0.2F;
  first.cell_type = 7;
  first.species = {3.0F, 1.5F};
  const auto first_id = original.add_cell(first);

  cm::CellInit second;
  second.position = {-0.5F, 1.0F, 2.0F};
  second.direction = {1.0F, 0.0F, 0.0F};
  second.length = 2.75F;
  second.radius = 0.3F;
  second.growth_rate = 0.05F;
  second.cell_type = -2;
  second.species = {0.25F, 4.0F};
  original.add_cell(second);

  cm::PlaneConstraintInit plane;
  plane.point = {0.0F, -3.0F, 0.0F};
  plane.inward_normal = {0.0F, 2.0F, 0.0F};
  plane.coefficient = 1.25F;
  assert(original.add_plane_constraint(plane) == 1);
  cm::SphereConstraintInit sphere;
  sphere.center = {1.0F, 2.0F, 3.0F};
  sphere.radius = 8.0F;
  sphere.coefficient = 0.75F;
  sphere.allowed_region = cm::SphereRegion::inside;
  assert(original.add_sphere_constraint(sphere) == 2);

  original.step(0.125F);
  const auto [daughter_a, daughter_b] = original.divide_equal(first_id);
  original.step(0.03125F);

  const auto saved = original.checkpoint();
  assert(saved.schema_version == cm::checkpoint_schema_version);
  assert(saved.world.next_id == 5);
  assert(saved.constraints.next_id == 3);
  assert(saved.world.lineage.size() == 2);

  cm::Simulation restored(cm::BackendKind::cpu, saved);
  assert_checkpoints_equal(restored.checkpoint(), saved);
  assert(restored.lineage_parent(daughter_a) == first_id);
  assert(restored.lineage_parent(daughter_b) == first_id);

  cm::test::for_each_backend_device([&](cm::BackendKind backend, std::uint32_t device_index) {
    cm::Simulation candidate(backend, saved, device_index);
    assert_checkpoints_equal(candidate.checkpoint(), saved);
    cm::Simulation reference(cm::BackendKind::cpu, saved);
    candidate.step(0.02F);
    reference.step(0.02F);
    assert_resumed_step_close(candidate, reference);
  });

  cm::CellInit added;
  added.species = {0.5F, 0.75F};
  assert(original.add_cell(added) == restored.add_cell(added));
  assert(original.add_plane_constraint(plane) == restored.add_plane_constraint(plane));
  original.step(0.0625F);
  restored.step(0.0625F);
  assert_checkpoints_equal(restored.checkpoint(), original.checkpoint());

  auto invalid_version = saved;
  invalid_version.schema_version += 1;
  assert_invalid([&] {
    cm::Simulation rejected(cm::BackendKind::cpu, invalid_version);
  });

  auto invalid_slot = saved;
  invalid_slot.world.cells.front().slot = 1;
  assert_invalid([&] {
    cm::Simulation rejected(cm::BackendKind::cpu, invalid_slot);
  });

  auto invalid_next_id = saved;
  invalid_next_id.world.next_id = daughter_b;
  assert_invalid([&] {
    cm::Simulation rejected(cm::BackendKind::cpu, invalid_next_id);
  });

  auto invalid_plan = saved;
  invalid_plan.species_rate_plan = cm::SpeciesRatePlan::zero(1);
  assert_invalid([&] {
    cm::Simulation rejected(cm::BackendKind::cpu, invalid_plan);
  });
}
