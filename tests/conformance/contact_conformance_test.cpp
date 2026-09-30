#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "backend_devices.hpp"
#include "cm/simulation.hpp"

namespace {

constexpr float absolute_tolerance = 2.0e-5F;
constexpr float relative_tolerance = 2.0e-5F;

bool close(float actual, float expected) {
  const auto tolerance = absolute_tolerance + relative_tolerance * std::abs(expected);

  return std::abs(actual - expected) <= tolerance;
}

void add_capsule(cm::Simulation& simulation, cm::Vec3 center, cm::Vec3 axis, float length = 2.0F,
                 float radius = 0.5F) {
  cm::CellInit cell;
  cell.position = center;
  cell.direction = axis;
  cell.length = length;
  cell.radius = radius;
  simulation.add_cell(cell);
}

void compare_graphs(const cm::ContactGraph& actual, const cm::ContactGraph& expected) {
  assert(actual.cell_count() == expected.cell_count());
  assert(actual.size() == expected.size());

  for (std::size_t index = 0; index < expected.size(); ++index) {
    const auto& actual_contact = actual.contacts()[index];
    const auto& expected_contact = expected.contacts()[index];
    assert(actual_contact.first_id == expected_contact.first_id);
    assert(actual_contact.second_id == expected_contact.second_id);
    assert(actual_contact.first_slot == expected_contact.first_slot);
    assert(actual_contact.second_slot == expected_contact.second_slot);
    assert(actual_contact.ordinal == expected_contact.ordinal);
    assert(close(actual_contact.point_on_first.x, expected_contact.point_on_first.x));
    assert(close(actual_contact.point_on_first.y, expected_contact.point_on_first.y));
    assert(close(actual_contact.point_on_first.z, expected_contact.point_on_first.z));
    assert(close(actual_contact.normal.x, expected_contact.normal.x));
    assert(close(actual_contact.normal.y, expected_contact.normal.y));
    assert(close(actual_contact.normal.z, expected_contact.normal.z));
    assert(close(actual_contact.signed_separation, expected_contact.signed_separation));
    assert(close(actual_contact.weight, expected_contact.weight));
  }

  for (std::size_t slot = 0; slot < expected.cell_count(); ++slot) {
    const auto actual_indices = actual.incident_contact_indices(static_cast<cm::Slot>(slot));
    const auto expected_indices = expected.incident_contact_indices(static_cast<cm::Slot>(slot));
    assert(actual_indices.size() == expected_indices.size());

    for (std::size_t index = 0; index < expected_indices.size(); ++index) {
      assert(actual_indices[index] == expected_indices[index]);
    }
  }
}

void populate_mixed_geometry(cm::Simulation& simulation) {
  add_capsule(simulation, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}, 4.0F);
  add_capsule(simulation, {0.0F, 0.8F, 0.0F}, {-1.0F, 0.0F, 0.0F}, 4.0F);
  add_capsule(simulation, {4.9F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}, 4.0F);
  add_capsule(simulation, {10.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F});
  add_capsule(simulation, {10.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F});
  add_capsule(simulation, {15.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}, 0.0F);
  add_capsule(simulation, {15.0F, 0.8F, 0.0F}, {1.0F, 0.0F, 0.0F}, 0.0F);
}

void run_mixed_geometry(cm::BackendKind backend, std::uint32_t device_index) {
  cm::Simulation reference(cm::BackendKind::cpu);
  cm::Simulation candidate(backend, 0, 0, device_index);
  populate_mixed_geometry(reference);
  populate_mixed_geometry(candidate);
  compare_graphs(candidate.find_cell_contacts(), reference.find_cell_contacts());
}

void run_empty_and_single_cell(cm::BackendKind backend, std::uint32_t device_index) {
  cm::Simulation empty(backend, 0, 0, device_index);
  const auto empty_graph = empty.find_cell_contacts();
  assert(empty_graph.cell_count() == 0);
  assert(empty_graph.empty());

  cm::Simulation single(backend, 0, 0, device_index);
  add_capsule(single, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F});
  const auto single_graph = single.find_cell_contacts();
  assert(single_graph.cell_count() == 1);
  assert(single_graph.empty());
}

void run_dense_geometry(cm::BackendKind backend, std::uint32_t device_index) {
  cm::Simulation reference(cm::BackendKind::cpu, 31);
  cm::Simulation candidate(backend, 31, 0, device_index);

  for (std::size_t index = 0; index < 31; ++index) {
    add_capsule(reference, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F});
    add_capsule(candidate, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F});
  }

  const auto expected = reference.find_cell_contacts();
  const auto actual = candidate.find_cell_contacts();
  assert(actual.size() == 930);
  assert(actual.incident_contact_indices(0).size() == 60);
  compare_graphs(actual, expected);
}

void run_parameters_and_buffer_reuse(cm::BackendKind backend, std::uint32_t device_index) {
  cm::Simulation reference(cm::BackendKind::cpu);
  cm::Simulation candidate(backend, 0, 0, device_index);

  for (auto* simulation : {&reference, &candidate}) {
    add_capsule(*simulation, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}, 0.0F);
    add_capsule(*simulation, {1.005F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}, 0.0F);
  }

  cm::ContactParameters strict;
  strict.activation_margin = 0.0F;
  compare_graphs(candidate.find_cell_contacts(strict), reference.find_cell_contacts(strict));
  assert(candidate.find_cell_contacts(strict).empty());
  compare_graphs(candidate.find_cell_contacts(), reference.find_cell_contacts());

  add_capsule(reference, {0.5F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}, 0.0F);
  add_capsule(candidate, {0.5F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}, 0.0F);
  compare_graphs(candidate.find_cell_contacts(), reference.find_cell_contacts());
}

void run_compacted_identity_geometry(cm::BackendKind backend, std::uint32_t device_index) {
  cm::Simulation reference(cm::BackendKind::cpu);
  cm::Simulation candidate(backend, 0, 0, device_index);

  for (auto* simulation : {&reference, &candidate}) {
    cm::CellInit first;
    first.length = 4.0F;
    const auto parent = simulation->add_cell(first);
    cm::CellInit second = first;
    second.position.y = 0.2F;
    simulation->add_cell(second);
    simulation->divide_equal(parent);
  }

  compare_graphs(candidate.find_cell_contacts(), reference.find_cell_contacts());
}

}  // namespace

int main() {
  cm::test::for_each_backend_device([](cm::BackendKind backend, std::uint32_t device_index) {
    cm::Simulation capability_probe(backend, 0, 0, device_index);

    if (!capability_probe.supports(cm::BackendFeature::cell_contacts)) {
      return;
    }

    run_empty_and_single_cell(backend, device_index);
    run_mixed_geometry(backend, device_index);
    run_dense_geometry(backend, device_index);
    run_parameters_and_buffer_reuse(backend, device_index);
    run_compacted_identity_geometry(backend, device_index);
  });

  return 0;
}
