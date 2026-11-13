#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "microsimulator/contact_dynamics.hpp"
#include "microsimulator/simulation.hpp"

namespace {

using microsimulator::BackendKind;
using microsimulator::CellInit;
using microsimulator::ContactDynamicsParameters;
using microsimulator::GrowthLoadLaw;
using microsimulator::Simulation;
using microsimulator::Vec3;

bool close(float actual, float expected, float tolerance = 1.0e-5F) {
  return std::abs(actual - expected) <= tolerance;
}

// Deterministic angles in (-scale, scale) for perturbing daughter directions.
class Jitter {
 public:
  float next(float scale) {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    const auto unit = static_cast<float>((state_ >> 40U) & 0xFFFFFFU) / 16777216.0F;
    return (2.0F * unit - 1.0F) * scale;
  }

 private:
  std::uint64_t state_{0x9E3779B97F4A7C15ULL};
};

// Divides every cell at least `division_length` long, turning each daughter by
// a small angle in the plane so that a line of cells can buckle.
void divide_long_cells(Simulation& simulation, float division_length, Jitter& jitter) {
  for (const auto& cell : simulation.cells()) {
    if (cell.length < division_length) {
      continue;
    }
    const auto [first, second] = simulation.divide_equal(cell.id);
    for (const auto id : {first, second}) {
      const auto daughter = simulation.cell(id);
      const auto angle = jitter.next(0.02F);
      const Vec3 direction{
          daughter.direction.x * std::cos(angle) - daughter.direction.y * std::sin(angle),
          daughter.direction.x * std::sin(angle) + daughter.direction.y * std::cos(angle),
          daughter.direction.z,
      };
      simulation.set_cell_geometry(id, daughter.position, direction, daughter.length);
    }
  }
}

float deepest_overlap(Simulation& simulation) {
  auto overlap = 0.0F;
  const auto pairs = simulation.find_cell_contacts();
  for (const auto& contact : pairs.contacts()) {
    overlap = std::max(overlap, -contact.signed_separation);
  }
  const auto walls = simulation.find_external_contacts();
  for (const auto& contact : walls.contacts()) {
    overlap = std::max(overlap, -contact.signed_separation);
  }
  return overlap;
}

float footprint(const Simulation& simulation) {
  auto area = 0.0F;
  for (const auto& cell : simulation.cells()) {
    area += std::numbers::pi_v<float> * cell.radius * cell.radius +
            2.0F * cell.radius * cell.length;
  }
  return area;
}

void test_growth_load_factor_follows_each_law() {
  microsimulator::GrowthLoadResponse response;
  assert(close(microsimulator::growth_load_factor(response, 100.0F), 1.0F));

  response.law = GrowthLoadLaw::exponential;
  response.sensitivity = 0.5F;
  assert(close(microsimulator::growth_load_factor(response, 0.0F), 1.0F));
  assert(close(microsimulator::growth_load_factor(response, 2.0F), std::exp(-1.0F)));

  response.law = GrowthLoadLaw::linear;
  response.free_load = 1.0F;
  response.stall_load = 3.0F;
  assert(close(microsimulator::growth_load_factor(response, 0.5F), 1.0F));
  assert(close(microsimulator::growth_load_factor(response, 2.0F), 0.5F));
  assert(close(microsimulator::growth_load_factor(response, 3.0F), 0.0F));
  assert(close(microsimulator::growth_load_factor(response, 30.0F), 0.0F));

  response.stall_load = 1.0F;
  bool rejected = false;
  try {
    microsimulator::validate_growth_load_response(response);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
}

void test_separated_cells_grow_without_moving() {
  Simulation simulation;
  CellInit cell;
  cell.length = 2.0F;
  const auto first = simulation.add_cell(cell);
  cell.position = {0.0F, 5.0F, 0.0F};
  const auto second = simulation.add_cell(cell);

  const auto result = simulation.step(0.1F, ContactDynamicsParameters{});
  assert(result.report.max_rows == 0);
  assert(close(simulation.cell(first).position.y, 0.0F));
  assert(close(simulation.cell(second).position.y, 5.0F));
  assert(close(simulation.cell(first).length, 2.2F));
  assert(close(result.length_increments[0], 0.2F));
  assert(close(result.axial_loads[0], 0.0F));
  assert(close(static_cast<float>(simulation.time()), 0.1F));
}

void test_end_to_end_cells_push_apart_and_keep_growing() {
  Simulation simulation;
  CellInit cell;
  cell.length = 2.0F;
  cell.position = {-1.5F, 0.0F, 0.0F};
  const auto first = simulation.add_cell(cell);
  cell.position = {1.5F, 0.0F, 0.0F};
  const auto second = simulation.add_cell(cell);

  ContactDynamicsParameters parameters;
  const auto result = simulation.step(0.1F, parameters);
  const auto left = simulation.cell(first);
  const auto right = simulation.cell(second);
  // Without a load response both cells elongate in full, and the drag is the
  // same on each, so they move apart symmetrically.
  assert(close(left.length, 2.2F));
  assert(close(right.length, 2.2F));
  assert(close(left.position.x, -right.position.x, 1.0e-4F));
  const auto gap = (right.position.x - left.position.x) - 0.5F * (left.length + right.length) -
                   (left.radius + right.radius);
  assert(gap > -parameters.overlap_tolerance);
  assert(gap < parameters.overlap_tolerance);
  // Each cell carries the contact force on one end.
  assert(result.axial_loads[0] > 0.0F);
  assert(close(result.axial_loads[0], result.axial_loads[1], 1.0e-3F));
}

void test_solution_satisfies_complementarity() {
  // Two crossed rods pressed together and a third clear of both.
  const std::vector<microsimulator::ContactBody> bodies(
      3, microsimulator::ContactBody{
             .axis = {1.0F, 0.0F, 0.0F},
             .translational_mobility = 0.5F,
             .transverse_mobility = 1.0F,
             .axial_mobility = 8.0F,
             .length_rate = 0.5F,
         });
  std::vector<microsimulator::ContactRow> rows(2);
  rows[0].first_slot = 0;
  rows[0].second_slot = 1;
  rows[0].first_jacobian = {0.0F, -1.0F, 0.0F, 0.0F, 0.0F, -0.3F};
  rows[0].second_jacobian = {0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.1F};
  rows[0].first_length_lever = 0.5F;
  rows[0].offset = -0.02F;
  rows[0].compliance = 1.0e-4F;
  rows[1].first_slot = 1;
  rows[1].second_slot = 2;
  rows[1].first_jacobian = {-1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  rows[1].second_jacobian = {1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  rows[1].offset = 0.5F;
  rows[1].compliance = 1.0e-4F;

  microsimulator::ContactSystem system{
      .rows = rows,
      .bodies = bodies,
      .dt = 0.05F,
  };
  system.growth.law = GrowthLoadLaw::exponential;
  system.growth.sensitivity = 0.2F;

  const std::vector<float> zero(rows.size(), 0.0F);
  const microsimulator::ContactSolveParameters parameters{.tolerance = 1.0e-6F};
  const auto solve = microsimulator::solve_contact_system_cpu(system, zero, parameters);
  assert(solve.report.status == microsimulator::ContactSolveStatus::converged);
  const auto gaps = microsimulator::evaluate_contact_gaps(system, solve.multipliers);
  assert(solve.multipliers[0] > 0.0F);
  assert(std::abs(gaps[0]) <= 1.0e-5F);
  assert(solve.multipliers[1] == 0.0F);
  assert(gaps[1] > 0.0F);

  // The pressed cell elongates more slowly than the free one.
  const auto response = microsimulator::evaluate_contact_response(system, solve.multipliers);
  assert(response.loads[0] > 0.0F);
  assert(response.length_rates[0] < 0.5F);
  assert(close(response.length_rates[2], 0.5F));

  // A solve started from its own solution has nothing left to do.
  const auto again =
      microsimulator::solve_contact_system_cpu(system, solve.multipliers, parameters);
  assert(again.report.iterations == 0);
}

void test_cell_between_walls_stalls_at_the_stall_load() {
  Simulation simulation;
  microsimulator::PlaneConstraintInit left;
  left.point = {-2.0F, 0.0F, 0.0F};
  left.inward_normal = {1.0F, 0.0F, 0.0F};
  simulation.add_plane_constraint(left);
  microsimulator::PlaneConstraintInit right;
  right.point = {2.0F, 0.0F, 0.0F};
  right.inward_normal = {-1.0F, 0.0F, 0.0F};
  simulation.add_plane_constraint(right);
  CellInit cell;
  cell.length = 2.0F;
  const auto id = simulation.add_cell(cell);

  ContactDynamicsParameters parameters;
  parameters.growth.law = GrowthLoadLaw::linear;
  parameters.growth.stall_load = 4.0F;
  microsimulator::ContactDynamicsResult result;
  for (int step = 0; step < 200; ++step) {
    result = simulation.step(0.05F, parameters);
  }
  // The walls leave room for a cylinder of length 3. The cell fills it, to
  // within the overlap its stall load presses into the two walls, and stops.
  const auto filled = simulation.cell(id);
  const auto wall_overlap = parameters.growth.stall_load / parameters.contact_stiffness;
  assert(filled.length >= 3.0F - 2.0F * parameters.overlap_tolerance);
  assert(filled.length <= 3.0F + 2.0F * (wall_overlap + parameters.overlap_tolerance));
  assert(close(filled.position.x, 0.0F, 1.0e-3F));
  assert(result.axial_loads[0] >= parameters.growth.stall_load * 0.99F);
  assert(result.axial_loads[0] <= parameters.growth.stall_load * 1.05F);
  const auto held = filled.length;
  for (int step = 0; step < 50; ++step) {
    std::ignore = simulation.step(0.05F, parameters);
  }
  assert(close(simulation.cell(id).length, held, 1.0e-4F));
}

void test_fixed_cell_holds_its_place() {
  Simulation simulation;
  CellInit cell;
  cell.length = 2.0F;
  cell.position = {-1.5F, 0.0F, 0.0F};
  cell.fixed = true;
  const auto anchor = simulation.add_cell(cell);
  cell.position = {1.5F, 0.0F, 0.0F};
  cell.fixed = false;
  const auto free = simulation.add_cell(cell);

  ContactDynamicsParameters parameters;
  std::ignore = simulation.step(0.1F, parameters);
  const auto held = simulation.cell(anchor);
  const auto pushed = simulation.cell(free);
  assert(close(held.position.x, -1.5F));
  assert(close(held.length, 2.2F));
  // Each cell's facing end advances by half its elongation, and the free cell
  // takes all of the separation.
  assert(close(pushed.position.x, 1.5F + 0.2F, 2.0F * parameters.overlap_tolerance));
}

void test_container_corner_reports_every_wall() {
  microsimulator::WorldState state;
  CellInit cell;
  cell.length = 1.0F;
  cell.radius = 0.4F;
  cell.position = {1.1F, 1.55F, 0.0F};
  state.add_cell(cell);
  microsimulator::ConstraintSet constraints;
  microsimulator::BoxConstraintInit box;
  box.half_extents = {2.0F, 2.0F, 0.6F};
  box.allowed_region = microsimulator::ConstraintRegion::inside;
  constraints.add_box(box);

  microsimulator::ConstraintContactParameters parameters;
  parameters.activation_margin = 0.1F;
  // Both ends lie 0.05 clear of the +y wall, and the positive end also touches
  // the +x wall. The nearest-face contacts hold each end against one wall only.
  const auto nearest = microsimulator::find_external_contacts_cpu(state, constraints, parameters);
  const auto faces = microsimulator::find_external_contact_faces(state, constraints, parameters);
  assert(nearest.size() == 2);
  assert(faces.size() == 3);
  bool x_wall = false;
  bool y_wall = false;
  for (const auto& contact : faces.contacts()) {
    if (contact.location != microsimulator::RodContactLocation::positive) {
      assert(close(contact.normal.y, 1.0F));
      continue;
    }
    if (close(contact.normal.x, 1.0F)) {
      x_wall = true;
      assert(close(contact.signed_separation, 0.0F, 1.0e-5F));
    }
    if (close(contact.normal.y, 1.0F)) {
      y_wall = true;
      assert(close(contact.signed_separation, 0.05F, 1.0e-5F));
    }
  }
  assert(x_wall && y_wall);
}

void test_closed_chamber_fills_and_holds() {
  Simulation simulation;
  microsimulator::BoxConstraintInit chamber;
  chamber.half_extents = {2.0F, 2.0F, 0.6F};
  chamber.allowed_region = microsimulator::ConstraintRegion::inside;
  simulation.add_box_constraint(chamber);
  CellInit founder;
  founder.length = 1.0F;
  founder.radius = 0.4F;
  simulation.add_cell(founder);

  ContactDynamicsParameters parameters;
  parameters.growth.law = GrowthLoadLaw::linear;
  parameters.growth.stall_load = 2.0F;
  Jitter jitter;
  const auto run = [&](int steps) {
    for (int step = 0; step < steps; ++step) {
      divide_long_cells(simulation, 2.0F, jitter);
      std::ignore = simulation.step(0.05F, parameters);
    }
  };
  run(300);
  const auto filled_count = simulation.cell_count();
  const auto filled_footprint = footprint(simulation);
  run(300);

  // The population and its footprint stop changing once the floor is full.
  assert(simulation.cell_count() == filled_count);
  assert(close(footprint(simulation), filled_footprint, 0.02F));
  const auto cover = footprint(simulation) / 16.0F;
  assert(cover > 0.6F);
  assert(cover < 1.0F);
  assert(deepest_overlap(simulation) < 0.01F);
  for (const auto& cell : simulation.cells()) {
    assert(std::abs(cell.position.x) + 0.0F <= 2.0F);
    assert(std::abs(cell.position.y) <= 2.0F);
    assert(close(cell.position.z, 0.0F, 1.0e-3F));
  }
}

void test_unconfined_colony_grows_in_full_without_overlap() {
  Simulation simulation(BackendKind::cpu, 512);
  CellInit founder;
  founder.length = 2.0F;
  simulation.add_cell(founder);
  // The same cells with no mechanics: growth and division alone.
  Simulation reference(BackendKind::cpu, 512);
  reference.add_cell(founder);

  ContactDynamicsParameters parameters;
  Jitter jitter;
  Jitter reference_jitter;
  auto worst = 0.0F;
  for (int step = 0; step < 110; ++step) {
    divide_long_cells(simulation, 4.0F, jitter);
    divide_long_cells(reference, 4.0F, reference_jitter);
    const auto result = simulation.step(0.05F, parameters);
    reference.step(0.05F);
    worst = std::max(worst, result.report.max_overlap);
  }
  assert(simulation.cell_count() == reference.cell_count());
  assert(simulation.cell_count() >= 32);
  auto length = 0.0F;
  auto reference_length = 0.0F;
  for (const auto& cell : simulation.cells()) {
    length += cell.length;
    assert(close(cell.position.z, 0.0F, 1.0e-4F));
  }
  for (const auto& cell : reference.cells()) {
    reference_length += cell.length;
  }
  assert(close(length, reference_length, 1.0e-3F * reference_length));
  // Overlap stays at what the contact forces press into the cells.
  assert(worst < 0.01F);
  assert(deepest_overlap(simulation) < 0.01F);
}

void test_failed_step_leaves_the_simulation_unchanged() {
  Simulation simulation;
  CellInit cell;
  cell.length = 2.0F;
  cell.position = {-1.5F, 0.0F, 0.0F};
  simulation.add_cell(cell);
  cell.position = {1.5F, 0.0F, 0.0F};
  simulation.add_cell(cell);
  cell.position = {4.5F, 0.0F, 0.0F};
  simulation.add_cell(cell);
  const auto before = simulation.cells();

  ContactDynamicsParameters parameters;
  // One iteration cannot resolve two coupled contacts at any substep length.
  parameters.max_iterations = 1;
  parameters.overlap_tolerance = 1.0e-7F;
  parameters.max_step_halvings = 2;
  bool failed = false;
  try {
    std::ignore = simulation.step(0.1F, parameters);
  } catch (const std::runtime_error&) {
    failed = true;
  }
  assert(failed);
  assert(simulation.time() == 0.0);
  const auto after = simulation.cells();
  for (std::size_t index = 0; index < before.size(); ++index) {
    assert(after[index].position.x == before[index].position.x);
    assert(after[index].length == before[index].length);
    assert(after[index].direction.x == before[index].direction.x);
  }

  parameters = ContactDynamicsParameters{};
  parameters.contact_stiffness = 0.0F;
  bool rejected = false;
  try {
    std::ignore = simulation.step(0.1F, parameters);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
}

}  // namespace

int main() {
  test_growth_load_factor_follows_each_law();
  test_separated_cells_grow_without_moving();
  test_end_to_end_cells_push_apart_and_keep_growing();
  test_solution_satisfies_complementarity();
  test_cell_between_walls_stalls_at_the_stall_load();
  test_fixed_cell_holds_its_place();
  test_container_corner_reports_every_wall();
  test_closed_chamber_fills_and_holds();
  test_unconfined_colony_grows_in_full_without_overlap();
  test_failed_step_leaves_the_simulation_unchanged();
  return 0;
}
