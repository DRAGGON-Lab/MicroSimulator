#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "cm/simulation.hpp"

namespace {

bool close(float left, float right, float tolerance = 1.0e-6F) {
  return std::abs(left - right) <= tolerance;
}

void test_growth_uses_stable_ids() {
  cm::Simulation simulation;
  cm::CellInit initial;
  initial.length = 4.0F;
  initial.radius = 0.5F;
  initial.growth_rate = 0.25F;

  const auto id = simulation.add_cell(initial);
  simulation.step(0.5F);

  assert(id == 1);
  assert(simulation.cell_count() == 1);
  assert(close(simulation.cell(id).length, 4.5F));
  assert(std::abs(simulation.time() - 0.5) <= 1.0e-12);
  simulation.validate();
}

void test_division_reuses_slot_but_not_identity() {
  cm::Simulation simulation;
  cm::CellInit initial;
  initial.position = {2.0F, 3.0F, 0.0F};
  initial.direction = {2.0F, 0.0F, 0.0F};
  initial.length = 4.0F;
  initial.radius = 0.5F;
  initial.growth_rate = 0.75F;
  initial.cell_type = 7;

  const auto parent = simulation.add_cell(initial);
  const auto [first, second] = simulation.divide_equal(parent);
  const auto first_cell = simulation.cell(first);
  const auto second_cell = simulation.cell(second);

  assert(simulation.cell_count() == 2);
  assert(first == 2);
  assert(second == 3);
  assert(first_cell.slot == 0);
  assert(second_cell.slot == 1);
  assert(close(first_cell.length, 1.5F));
  assert(close(second_cell.length, 1.5F));
  assert(close(first_cell.position.x, 0.75F));
  assert(close(second_cell.position.x, 3.25F));
  assert(first_cell.cell_type == 7);
  assert(close(first_cell.growth_rate, 0.75F));
  assert(simulation.lineage_parent(first) == parent);
  assert(simulation.lineage_parent(second) == parent);

  bool parent_is_gone = false;
  try {
    static_cast<void>(simulation.cell(parent));
  } catch (const std::out_of_range&) {
    parent_is_gone = true;
  }
  assert(parent_is_gone);
  simulation.validate();
}

void test_asymmetric_division_preserves_capsule_extent() {
  cm::Simulation simulation(cm::BackendKind::cpu, 0, 2);
  cm::CellInit initial;
  initial.position = {2.0F, 3.0F, 0.0F};
  initial.direction = {2.0F, 0.0F, 0.0F};
  initial.length = 6.0F;
  initial.radius = 0.5F;
  initial.species = {2.0F, 3.0F};

  const auto parent = simulation.add_cell(initial);
  const auto [first, second] = simulation.divide(parent, 0.25F);
  const auto first_cell = simulation.cell(first);
  const auto second_cell = simulation.cell(second);

  assert(close(first_cell.length, 1.25F));
  assert(close(second_cell.length, 3.75F));
  assert(close(first_cell.position.x, -0.375F));
  assert(close(second_cell.position.x, 3.125F));
  assert(first_cell.species == initial.species);
  assert(second_cell.species == initial.species);
  assert(close(first_cell.position.x - (first_cell.length * 0.5F), -1.0F));
  assert(close(second_cell.position.x + (second_cell.length * 0.5F), 5.0F));
  assert(close(
      (second_cell.position.x - (second_cell.length * 0.5F)) -
          (first_cell.position.x + (first_cell.length * 0.5F)),
      1.0F));
  assert(simulation.lineage_parent(first) == parent);
  assert(simulation.lineage_parent(second) == parent);
  simulation.validate();
}

void test_invalid_division_fraction_is_atomic() {
  cm::Simulation simulation;
  cm::CellInit initial;
  initial.length = 6.0F;
  const auto parent = simulation.add_cell(initial);

  for (const auto fraction : {0.0F, 1.0F, -0.25F, 1.25F,
                              std::numeric_limits<float>::quiet_NaN()}) {
    bool rejected = false;
    try {
      static_cast<void>(simulation.divide(parent, fraction));
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
    assert(simulation.cell_count() == 1);
    assert(simulation.cell(parent).slot == 0);
    assert(close(simulation.cell(parent).length, 6.0F));
  }
  simulation.validate();
}

void test_invalid_state_fails_explicitly() {
  cm::Simulation simulation;
  cm::CellInit invalid;
  invalid.radius = 0.0F;

  bool rejected = false;
  try {
    static_cast<void>(simulation.add_cell(invalid));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);

  rejected = false;
  try {
    simulation.step(-0.1F);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
}

void test_mutable_cell_attributes_keep_stable_identity() {
  cm::Simulation simulation;
  const auto id = simulation.add_cell(cm::CellInit{});
  simulation.set_cell_attributes(id, 2.5F, 7);

  const auto updated = simulation.cell(id);
  assert(updated.id == id);
  assert(close(updated.growth_rate, 2.5F));
  assert(updated.cell_type == 7);

  bool rejected = false;
  try {
    simulation.set_cell_attributes(id, std::numeric_limits<float>::quiet_NaN(), 8);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
  assert(close(simulation.cell(id).growth_rate, 2.5F));
  assert(simulation.cell(id).cell_type == 7);
}

void test_geometry_can_be_updated_by_stable_id() {
  cm::Simulation simulation;
  const auto first = simulation.add_cell(cm::CellInit{});
  const auto second = simulation.add_cell(cm::CellInit{});
  simulation.set_cell_geometry(second, {1.0F, 2.0F, 3.0F}, {0.0F, 2.0F, 0.0F}, 4.0F);

  assert(simulation.cell(first).slot == 0);
  const auto updated = simulation.cell(second);
  assert(updated.id == second);
  assert(updated.slot == 1);
  assert(close(updated.position.x, 1.0F));
  assert(close(updated.position.y, 2.0F));
  assert(close(updated.position.z, 3.0F));
  assert(close(updated.direction.y, 1.0F));
  assert(close(updated.length, 4.0F));
}

void test_fixed_state_is_mutable_persistent_and_inherited() {
  cm::Simulation simulation;
  cm::CellInit initial;
  initial.length = 4.0F;
  initial.growth_rate = 0.25F;
  initial.fixed = true;
  const auto parent = simulation.add_cell(initial);

  assert(simulation.cell(parent).fixed);
  simulation.set_cell_fixed(parent, false);
  assert(!simulation.cell(parent).fixed);
  simulation.set_cell_fixed(parent, true);
  simulation.step(0.5F);
  assert(close(simulation.cell(parent).length, 4.5F));

  const auto [first, second] = simulation.divide_equal(parent);
  assert(simulation.cell(first).fixed);
  assert(simulation.cell(second).fixed);
  const auto checkpoint = simulation.checkpoint();
  cm::Simulation restored(cm::BackendKind::cpu, checkpoint);
  assert(restored.cell(first).fixed);
  assert(restored.cell(second).fixed);
  restored.validate();
}

void test_removal_backfills_slots_and_keeps_lineage() {
  cm::WorldState state(0, 2);
  cm::CellInit cell;
  cell.species = {1.0F, 2.0F};
  const auto first = state.add_cell(cell);
  cell.position = {5.0F, 0.0F, 0.0F};
  cell.species = {3.0F, 4.0F};
  const auto second = state.add_cell(cell);
  cell.position = {9.0F, 0.0F, 0.0F};
  cell.species = {5.0F, 6.0F};
  const auto third = state.add_cell(cell);
  const auto daughters = state.divide_equal(first);

  state.remove_cell(second);
  assert(state.size() == 3);
  assert(!state.contains(second));
  assert(state.contains(third));
  const auto moved = state.cell(daughters.second);
  assert(moved.slot == 1);
  const auto untouched = state.cell(third);
  assert(untouched.slot == 2);
  assert(untouched.position.x == 9.0F);
  assert(untouched.species[0] == 5.0F);
  assert(state.lineage_parent(daughters.first) == first);

  const auto checkpoint = state.checkpoint();
  checkpoint.validate();
  const cm::WorldState restored(checkpoint);
  assert(restored.size() == 3);
  assert(restored.lineage_parent(daughters.second) == first);

  bool rejected = false;
  try {
    state.remove_cell(second);
  } catch (const std::exception&) {
    rejected = true;
  }
  assert(rejected);
  state.validate();
}

void test_unavailable_backends_do_not_fall_back() {
  assert(cm::backend_device_count(cm::BackendKind::cpu) == 1);
  assert(cm::backend_available(cm::BackendKind::cpu, 0));
  assert(!cm::backend_available(cm::BackendKind::cpu, 1));
  bool invalid_cpu_device_rejected = false;
  try {
    cm::Simulation simulation(cm::BackendKind::cpu, 0, 0, 1);
  } catch (const std::out_of_range&) {
    invalid_cpu_device_rejected = true;
  }
  assert(invalid_cpu_device_rejected);

  for (const auto backend : {cm::BackendKind::metal, cm::BackendKind::cuda}) {
    if (cm::backend_available(backend)) {
      cm::Simulation simulation(backend);
      assert(simulation.backend_info().kind == backend);
      assert(simulation.backend_info().device_index == 0);
      assert(cm::backend_device_count(backend) >= 1);
      continue;
    }
    bool rejected = false;
    try {
      cm::Simulation simulation(backend);
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    assert(rejected);
  }
}

}  // namespace

int main() {
  test_growth_uses_stable_ids();
  test_division_reuses_slot_but_not_identity();
  test_asymmetric_division_preserves_capsule_extent();
  test_invalid_division_fraction_is_atomic();
  test_invalid_state_fails_explicitly();
  test_mutable_cell_attributes_keep_stable_identity();
  test_geometry_can_be_updated_by_stable_id();
  test_fixed_state_is_mutable_persistent_and_inherited();
  test_removal_backfills_slots_and_keeps_lineage();
  test_unavailable_backends_do_not_fall_back();
  return 0;
}
