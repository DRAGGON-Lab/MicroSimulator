#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <tuple>

#include "cm/contact_graph.hpp"
#include "cm/simulation.hpp"

namespace {

bool close(float actual, float expected, float tolerance = 1.0e-5F) {
  return std::abs(actual - expected) <= tolerance;
}

cm::CellId add_capsule(cm::WorldState& state, cm::Vec3 center, cm::Vec3 axis, float length = 2.0F,
                       float radius = 0.5F) {
  cm::CellInit cell;
  cell.position = center;
  cell.direction = axis;
  cell.length = length;
  cell.radius = radius;

  return state.add_cell(cell);
}

void test_separated_capsules_have_no_contact() {
  cm::WorldState state;
  add_capsule(state, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F});
  add_capsule(state, {4.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F});

  const auto graph = cm::find_cell_contacts_cpu(state);
  assert(graph.cell_count() == 2);
  assert(graph.empty());
  assert(graph.incident_contact_indices(0).empty());
  assert(graph.incident_contact_indices(1).empty());
}

void test_end_on_contact_has_signed_separation() {
  cm::WorldState state;
  add_capsule(state, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F});
  add_capsule(state, {2.9F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F});

  const auto graph = cm::find_cell_contacts_cpu(state);
  assert(graph.size() == 1);
  const auto& contact = graph.contacts().front();
  assert(contact.first_id == 1);
  assert(contact.second_id == 2);
  assert(contact.ordinal == 0);
  assert(close(contact.signed_separation, -0.1F));
  assert(close(contact.normal.x, 1.0F));
  assert(close(contact.point_on_first.x, 1.5F));
  assert(close(contact.weight, 1.0F));
}

void test_parallel_overlap_uses_two_weighted_contacts() {
  cm::WorldState state;
  add_capsule(state, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}, 4.0F);
  add_capsule(state, {0.0F, 0.8F, 0.0F}, {1.0F, 0.0F, 0.0F}, 4.0F);

  const auto graph = cm::find_cell_contacts_cpu(state);
  assert(graph.size() == 2);
  const auto contacts = graph.contacts();
  assert(contacts[0].ordinal == 0);
  assert(contacts[1].ordinal == 1);
  assert(close(contacts[0].point_on_first.x, -2.0F));
  assert(close(contacts[1].point_on_first.x, 2.0F));

  for (const auto& contact : contacts) {
    assert(close(contact.signed_separation, -0.2F));
    assert(close(contact.normal.y, 1.0F));
    assert(close(contact.weight, std::sqrt(0.5F)));
  }

  assert(graph.incident_contact_indices(0).size() == 2);
  assert(graph.incident_contact_indices(1).size() == 2);
  assert(graph.neighbor_ids(0).size() == 1);
  assert(graph.neighbor_ids(0).front() == 2);
  assert(graph.neighbor_ids(1).size() == 1);
  assert(graph.neighbor_ids(1).front() == 1);
}

void test_neighbors_use_sorted_stable_ids_after_division() {
  cm::WorldState state;
  const auto parent = add_capsule(state, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}, 4.0F);
  const auto neighbor = add_capsule(state, {0.0F, 0.8F, 0.0F}, {1.0F, 0.0F, 0.0F}, 4.0F);
  const auto [first_daughter, second_daughter] = state.divide_equal(parent);

  const auto graph = cm::find_cell_contacts_cpu(state);
  assert(state.cell(first_daughter).slot == 0);
  assert(first_daughter > neighbor);
  assert(second_daughter > first_daughter);
  assert(graph.neighbor_ids(0).size() == 2);
  assert(graph.neighbor_ids(0)[0] == neighbor);
  assert(graph.neighbor_ids(0)[1] == second_daughter);

  const auto neighbor_slot = state.cell(neighbor).slot;
  assert(graph.neighbor_ids(neighbor_slot).size() == 2);
  assert(graph.neighbor_ids(neighbor_slot)[0] == first_daughter);
  assert(graph.neighbor_ids(neighbor_slot)[1] == second_daughter);
}

void test_neighbor_lookup_checks_slot_bounds() {
  const cm::ContactGraph graph(0, {});
  bool rejected = false;

  try {
    static_cast<void>(graph.neighbor_ids(0));
  } catch (const std::out_of_range&) {
    rejected = true;
  }

  assert(rejected);
}

void test_skew_and_coincident_contacts_have_finite_normals() {
  cm::WorldState skew;
  add_capsule(skew, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F});
  add_capsule(skew, {0.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F});
  const auto skew_graph = cm::find_cell_contacts_cpu(skew);
  const auto skew_contact = skew_graph.contacts().front();
  assert(close(skew_contact.normal.z, 1.0F));
  assert(close(cm::norm(skew_contact.normal), 1.0F));

  cm::WorldState coincident;
  add_capsule(coincident, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}, 4.0F);
  add_capsule(coincident, {0.0F, 0.0F, 0.0F}, {-1.0F, 0.0F, 0.0F}, 4.0F);
  const auto graph = cm::find_cell_contacts_cpu(coincident);
  const auto contacts = graph.contacts();
  assert(contacts.size() == 2);

  for (const auto& contact : contacts) {
    assert(std::isfinite(contact.normal.x));
    assert(std::isfinite(contact.normal.y));
    assert(std::isfinite(contact.normal.z));
    assert(close(cm::norm(contact.normal), 1.0F));
    assert(close(contact.signed_separation, -1.0F));
  }
}

void test_contact_graph_has_no_per_cell_limit() {
  cm::WorldState state;
  constexpr std::size_t capsule_count = 31;

  for (std::size_t index = 0; index < capsule_count; ++index) {
    add_capsule(state, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F});
  }

  const auto graph = cm::find_cell_contacts_cpu(state);
  const auto pair_count = capsule_count * (capsule_count - 1) / 2;
  assert(graph.size() == pair_count * 2);
  assert(graph.incident_contact_indices(0).size() == (capsule_count - 1) * 2);
  assert(graph.incident_contact_indices(0).size() > 24);
}

void test_sweep_and_prune_stages_only_overlapping_bounds() {
  cm::WorldState sparse;
  constexpr std::size_t capsule_count = 2048;

  for (std::size_t index = 0; index < capsule_count; ++index) {
    add_capsule(sparse, {static_cast<float>(index) * 10.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F});
  }

  assert(cm::find_cell_contact_candidates(sparse).empty());
  assert(cm::find_cell_contacts_cpu(sparse).empty());

  cm::WorldState dense;

  for (std::size_t index = 0; index < 31; ++index) {
    add_capsule(dense, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F});
  }

  const auto candidates = cm::find_cell_contact_candidates(dense);
  const auto geometry = dense.geometry_state();
  assert(candidates.size() == 31 * 30 / 2);

  for (const auto& candidate : candidates) {
    assert(geometry.ids[candidate.first_slot] < geometry.ids[candidate.second_slot]);
  }
}

void test_sweep_and_prune_matches_exhaustive_oracle() {
  cm::WorldState state;
  constexpr std::size_t capsule_count = 257;

  for (std::size_t index = 0; index < capsule_count; ++index) {
    const auto x = static_cast<float>((index * 37) % 101) * 0.21F;
    const auto y = static_cast<float>((index * 53) % 89) * 0.19F;
    const auto z = static_cast<float>((index * 29) % 47) * 0.17F;
    const auto axis_x = static_cast<float>((index * 7) % 13) + 1.0F;
    const auto axis_y = static_cast<float>((index * 11) % 17) + 1.0F;
    const auto axis_z = static_cast<float>((index * 5) % 19) + 1.0F;
    const auto length = 0.25F + static_cast<float>(index % 11) * 0.3F;
    const auto radius = 0.15F + static_cast<float>(index % 5) * 0.07F;
    add_capsule(state, {x, y, z}, {axis_x, axis_y, axis_z}, length, radius);
  }

  for (const auto margin : {0.0F, 0.01F, 0.5F}) {
    cm::ContactParameters parameters;
    parameters.activation_margin = margin;
    const auto actual = cm::find_cell_contacts_cpu(state, parameters);
    const auto expected = cm::find_cell_contacts_cpu_exhaustive(state, parameters);
    assert(actual.size() == expected.size());

    for (std::size_t index = 0; index < expected.size(); ++index) {
      assert(actual.contacts()[index].first_id == expected.contacts()[index].first_id);
      assert(actual.contacts()[index].second_id == expected.contacts()[index].second_id);
      assert(actual.contacts()[index].ordinal == expected.contacts()[index].ordinal);
      assert(close(actual.contacts()[index].signed_separation,
                   expected.contacts()[index].signed_separation));
    }
  }
}

void test_contacts_are_sorted_by_stable_identity() {
  cm::WorldState state;
  const auto old = add_capsule(state, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}, 4.0F);
  add_capsule(state, {0.0F, 0.2F, 0.0F}, {1.0F, 0.0F, 0.0F}, 4.0F);
  static_cast<void>(state.divide_equal(old));

  const auto graph = cm::find_cell_contacts_cpu(state);
  const auto contacts = graph.contacts();
  auto previous = std::tuple{cm::invalid_cell_id, cm::invalid_cell_id, std::uint8_t{0}};

  for (const auto& contact : contacts) {
    const auto key = std::tuple{contact.first_id, contact.second_id, contact.ordinal};
    assert(contact.first_id < contact.second_id);
    assert(previous < key);
    previous = key;
  }
}

void test_pair_order_reverses_the_contact_normal() {
  cm::WorldState forward;
  add_capsule(forward, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}, 4.0F);
  add_capsule(forward, {0.0F, 0.8F, 0.0F}, {-1.0F, 0.0F, 0.0F}, 4.0F);
  const auto forward_graph = cm::find_cell_contacts_cpu(forward);

  cm::WorldState reversed;
  add_capsule(reversed, {0.0F, 0.8F, 0.0F}, {-1.0F, 0.0F, 0.0F}, 4.0F);
  add_capsule(reversed, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}, 4.0F);
  const auto reversed_graph = cm::find_cell_contacts_cpu(reversed);

  assert(forward_graph.size() == reversed_graph.size());

  for (std::size_t index = 0; index < forward_graph.size(); ++index) {
    const auto& forward_contact = forward_graph.contacts()[index];
    const auto& reversed_contact = reversed_graph.contacts()[index];
    assert(close(forward_contact.signed_separation, reversed_contact.signed_separation));
    assert(close(forward_contact.weight, reversed_contact.weight));
    assert(close(forward_contact.normal.x, -reversed_contact.normal.x));
    assert(close(forward_contact.normal.y, -reversed_contact.normal.y));
    assert(close(forward_contact.normal.z, -reversed_contact.normal.z));
  }
}

void test_invalid_parameters_are_rejected() {
  cm::WorldState state;
  cm::ContactParameters parameters;
  parameters.parallel_sine_threshold = 2.0F;
  bool rejected = false;

  try {
    static_cast<void>(cm::find_cell_contacts_cpu(state, parameters));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }

  assert(rejected);
}

void test_simulation_exposes_the_backend_contact_contract() {
  cm::Simulation simulation;
  cm::CellInit first;
  first.length = 4.0F;
  cm::CellInit second = first;
  second.position.y = 0.8F;
  simulation.add_cell(first);
  simulation.add_cell(second);

  const auto graph = simulation.find_cell_contacts();
  assert(graph.cell_count() == 2);
  assert(graph.size() == 2);
}

}  // namespace

int main() {
  test_separated_capsules_have_no_contact();
  test_end_on_contact_has_signed_separation();
  test_parallel_overlap_uses_two_weighted_contacts();
  test_neighbors_use_sorted_stable_ids_after_division();
  test_neighbor_lookup_checks_slot_bounds();
  test_skew_and_coincident_contacts_have_finite_normals();
  test_contact_graph_has_no_per_cell_limit();
  test_sweep_and_prune_stages_only_overlapping_bounds();
  test_sweep_and_prune_matches_exhaustive_oracle();
  test_contacts_are_sorted_by_stable_identity();
  test_pair_order_reverses_the_contact_normal();
  test_invalid_parameters_are_rejected();
  test_simulation_exposes_the_backend_contact_contract();

  return 0;
}
