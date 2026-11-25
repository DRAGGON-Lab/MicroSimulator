#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <string_view>
#include <tuple>
#include <vector>

#include "backend_devices.hpp"
#include "microsimulator/simulation.hpp"

namespace {

using microsimulator::BackendKind;
using microsimulator::ContactBody;
using microsimulator::ContactRow;
using microsimulator::ContactSolveStatus;
using microsimulator::ContactSystem;
using microsimulator::GrowthLoadLaw;

void require_close(float actual, float expected, float tolerance, std::string_view scenario,
                   std::size_t index, std::string_view field) {
  if (std::abs(actual - expected) <= tolerance) {
    return;
  }
  std::cerr << scenario << ' ' << field << ' ' << index << ": actual=" << actual
            << " expected=" << expected << '\n';
  std::abort();
}

// A chain of `count` cells along x, each pressed into the next, with the
// first held against a wall, a fixed cell at the far end, and a cross
// contact that couples rotation.
struct Fixture {
  std::vector<ContactBody> bodies;
  std::vector<ContactRow> rows;

  [[nodiscard]] ContactSystem system(GrowthLoadLaw law) const {
    ContactSystem result{
        .rows = rows,
        .bodies = bodies,
        .dt = 0.05F,
    };
    result.growth.law = law;
    result.growth.sensitivity = 0.3F;
    result.growth.free_load = 0.2F;
    result.growth.stall_load = 3.0F;
    return result;
  }
};

Fixture make_chain(std::size_t count) {
  Fixture fixture;
  for (std::size_t index = 0; index < count; ++index) {
    const auto angle = 0.1F * static_cast<float>(index % 5);
    ContactBody body{
        .axis = {std::cos(angle), std::sin(angle), 0.0F},
        .translational_mobility = 1.0F / (3.0F + 0.1F * static_cast<float>(index % 7)),
        .transverse_mobility = 0.9F,
        .axial_mobility = 20.0F,
        .length_rate = index % 4 == 3 ? -0.2F : 0.8F + 0.05F * static_cast<float>(index % 3),
    };
    if (index + 1 == count) {
      body.translational_mobility = 0.0F;
      body.transverse_mobility = 0.0F;
      body.axial_mobility = 0.0F;
    }
    fixture.bodies.push_back(body);
  }
  // The wall on the first cell.
  ContactRow wall;
  wall.first_slot = 0;
  wall.first_jacobian = {1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.1F};
  wall.first_length_lever = 0.5F;
  wall.offset = -0.004F;
  wall.compliance = 1.0e-4F;
  fixture.rows.push_back(wall);
  for (std::size_t index = 0; index + 1 < count; ++index) {
    const auto tilt = 0.2F * static_cast<float>(index % 3) - 0.2F;
    const auto normal_x = std::cos(tilt);
    const auto normal_y = std::sin(tilt);
    ContactRow row;
    row.first_slot = static_cast<microsimulator::Slot>(index);
    row.second_slot = static_cast<microsimulator::Slot>(index + 1);
    row.first_jacobian = {-normal_x, -normal_y, 0.0F, 0.0F, 0.0F, -1.5F * normal_y};
    row.second_jacobian = {normal_x, normal_y, 0.0F, 0.0F, 0.0F, -1.5F * normal_y};
    row.first_length_lever = 0.5F * normal_x;
    row.second_length_lever = 0.5F * normal_x;
    row.offset = index % 3 == 0 ? -0.01F : 0.002F * static_cast<float>(index % 4);
    row.compliance = index % 6 == 0 ? 2.5e-5F : 1.0e-4F;
    fixture.rows.push_back(row);
  }
  // A row far from closing, which stays inactive.
  if (count > 2) {
    ContactRow distant;
    distant.first_slot = 0;
    distant.second_slot = static_cast<microsimulator::Slot>(count - 2);
    distant.first_jacobian = {0.0F, -1.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    distant.second_jacobian = {0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    distant.offset = 5.0F;
    distant.compliance = 1.0e-4F;
    fixture.rows.push_back(distant);
  }
  return fixture;
}

void run_solve_matches_reference(microsimulator::Simulation& backend, std::size_t count,
                                 GrowthLoadLaw law, std::string_view scenario) {
  const auto fixture = make_chain(count);
  const auto system = fixture.system(law);
  const std::vector<float> zero(fixture.rows.size(), 0.0F);
  const microsimulator::ContactSolveParameters parameters{.tolerance = 2.0e-6F,
                                                          .max_iterations = 200000};
  const auto expected = microsimulator::solve_contact_system_cpu(system, zero, parameters);
  const auto actual = backend.solve_contact_system(system, zero, parameters);
  assert(expected.report.status == ContactSolveStatus::converged);
  assert(actual.report.status == ContactSolveStatus::converged);
  assert(actual.report.residual <= parameters.tolerance);
  assert(actual.multipliers.size() == expected.multipliers.size());

  // The solution is unique, so both solves reach the same response.
  const auto gaps = microsimulator::evaluate_contact_gaps(system, actual.multipliers);
  for (std::size_t index = 0; index < gaps.size(); ++index) {
    assert(actual.multipliers[index] >= 0.0F);
    assert(gaps[index] >= -2.0F * parameters.tolerance);
    if (actual.multipliers[index] > 0.0F) {
      assert(std::abs(gaps[index]) <= 2.0F * parameters.tolerance);
    }
  }
  const auto actual_response =
      microsimulator::evaluate_contact_response(system, actual.multipliers);
  const auto expected_response =
      microsimulator::evaluate_contact_response(system, expected.multipliers);
  for (std::size_t body = 0; body < fixture.bodies.size(); ++body) {
    for (std::size_t component = 0; component < 6; ++component) {
      require_close(actual_response.velocities[body][component],
                    expected_response.velocities[body][component], 2.0e-3F, scenario, body,
                    "velocity");
    }
    require_close(actual_response.length_rates[body], expected_response.length_rates[body],
                  2.0e-3F, scenario, body, "length rate");
    require_close(actual_response.loads[body], expected_response.loads[body],
                  2.0e-2F * (1.0F + expected_response.loads[body]), scenario, body, "load");
  }

  // A solve from the solution has nothing left to do.
  const auto again = backend.solve_contact_system(system, actual.multipliers, parameters);
  assert(again.report.status == ContactSolveStatus::converged);
  assert(again.report.iterations == 0);
  for (std::size_t index = 0; index < again.multipliers.size(); ++index) {
    assert(again.multipliers[index] == actual.multipliers[index]);
  }
}

void run_empty_and_limited_solves(microsimulator::Simulation& backend) {
  const auto fixture = make_chain(6);
  auto system = fixture.system(GrowthLoadLaw::linear);
  const std::vector<float> zero(fixture.rows.size(), 0.0F);

  // The iteration limit is reported with the multipliers reached so far.
  const microsimulator::ContactSolveParameters limited{.tolerance = 1.0e-7F, .max_iterations = 3};
  const auto stopped = backend.solve_contact_system(system, zero, limited);
  assert(stopped.report.status == ContactSolveStatus::iteration_limit);
  assert(stopped.report.iterations == 3);
  assert(stopped.report.residual > limited.tolerance);
  assert(std::ranges::any_of(stopped.multipliers, [](float value) { return value > 0.0F; }));

  system.rows = {};
  const auto empty = backend.solve_contact_system(system, {}, limited);
  assert(empty.report.status == ContactSolveStatus::converged);
  assert(empty.multipliers.empty());
}

float footprint(const microsimulator::Simulation& simulation) {
  auto area = 0.0F;
  for (const auto& cell : simulation.cells()) {
    area += std::numbers::pi_v<float> * cell.radius * cell.radius +
            2.0F * cell.radius * cell.length;
  }
  return area;
}

// Divides cells at `division_length`, turning daughters by fixed alternating
// angles so that every backend sees the same plan.
void divide_long_cells(microsimulator::Simulation& simulation, float division_length,
                       std::uint32_t& counter) {
  for (const auto& cell : simulation.cells()) {
    if (cell.length < division_length) {
      continue;
    }
    const auto [first, second] = simulation.divide_equal(cell.id);
    for (const auto id : {first, second}) {
      const auto daughter = simulation.cell(id);
      const auto angle = 0.004F * static_cast<float>(counter++ % 7U) - 0.012F;
      simulation.set_cell_geometry(
          id, daughter.position,
          {daughter.direction.x * std::cos(angle) - daughter.direction.y * std::sin(angle),
           daughter.direction.x * std::sin(angle) + daughter.direction.y * std::cos(angle),
           daughter.direction.z},
          daughter.length);
    }
  }
}

void run_closed_chamber(BackendKind backend, std::uint32_t device_index) {
  microsimulator::Simulation simulation(backend, 0, 0, device_index);
  microsimulator::BoxConstraintInit chamber;
  chamber.half_extents = {2.0F, 2.0F, 0.6F};
  chamber.allowed_region = microsimulator::ConstraintRegion::inside;
  simulation.add_box_constraint(chamber);
  microsimulator::CellInit founder;
  founder.length = 1.0F;
  founder.radius = 0.4F;
  simulation.add_cell(founder);

  microsimulator::ContactDynamicsParameters parameters;
  parameters.growth.law = GrowthLoadLaw::linear;
  parameters.growth.stall_load = 2.0F;
  std::uint32_t counter = 0;
  const auto run = [&](int steps) {
    for (int step = 0; step < steps; ++step) {
      divide_long_cells(simulation, 2.0F, counter);
      std::ignore = simulation.step(0.05F, parameters);
    }
  };
  run(300);
  const auto filled_count = simulation.cell_count();
  const auto filled_footprint = footprint(simulation);
  run(200);
  assert(simulation.cell_count() == filled_count);
  assert(std::abs(footprint(simulation) - filled_footprint) < 0.02F);
  const auto cover = footprint(simulation) / 16.0F;
  assert(cover > 0.6F && cover < 1.0F);
  const auto contacts = simulation.find_cell_contacts();
  for (const auto& contact : contacts.contacts()) {
    assert(contact.signed_separation > -0.01F);
  }
  for (const auto& cell : simulation.cells()) {
    assert(std::abs(cell.position.x) <= 2.0F && std::abs(cell.position.y) <= 2.0F);
    assert(std::abs(cell.position.z) < 1.0e-3F);
  }
}

void run_free_colony(BackendKind backend, std::uint32_t device_index) {
  microsimulator::Simulation simulation(backend, 256, 0, device_index);
  microsimulator::Simulation reference(BackendKind::cpu, 256);
  microsimulator::CellInit founder;
  founder.length = 2.0F;
  simulation.add_cell(founder);
  reference.add_cell(founder);

  const microsimulator::ContactDynamicsParameters parameters;
  std::uint32_t counter = 0;
  std::uint32_t reference_counter = 0;
  auto worst = 0.0F;
  for (int step = 0; step < 100; ++step) {
    divide_long_cells(simulation, 4.0F, counter);
    divide_long_cells(reference, 4.0F, reference_counter);
    const auto result = simulation.step(0.05F, parameters);
    reference.step(0.05F);
    worst = std::max(worst, result.report.max_overlap);
  }
  // Growth without a load response is the growth of cells that never touch.
  assert(simulation.cell_count() == reference.cell_count());
  assert(simulation.cell_count() >= 16);
  auto length = 0.0F;
  auto reference_length = 0.0F;
  for (const auto& cell : simulation.cells()) {
    length += cell.length;
    assert(std::abs(cell.position.z) < 1.0e-3F);
  }
  for (const auto& cell : reference.cells()) {
    reference_length += cell.length;
  }
  assert(std::abs(length - reference_length) <= 1.0e-3F * reference_length);
  assert(worst < 0.01F);
}

}  // namespace

int main() {
  microsimulator::test::for_each_backend_device(
      [](BackendKind backend, std::uint32_t device_index) {
        microsimulator::Simulation solver_host(backend, 0, 0, device_index);
        if (!solver_host.supports(microsimulator::BackendFeature::contact_dynamics)) {
          return;
        }
        auto* solver = &solver_host;
        run_solve_matches_reference(*solver, 2, GrowthLoadLaw::none, "pair");
        run_solve_matches_reference(*solver, 9, GrowthLoadLaw::none, "chain without response");
        run_solve_matches_reference(*solver, 9, GrowthLoadLaw::exponential, "exponential chain");
        run_solve_matches_reference(*solver, 9, GrowthLoadLaw::linear, "linear chain");
        // A larger system after a smaller one grows the device buffers.
        run_solve_matches_reference(*solver, 300, GrowthLoadLaw::linear, "long chain");
        run_solve_matches_reference(*solver, 5, GrowthLoadLaw::exponential, "short chain");
        run_empty_and_limited_solves(*solver);
        run_closed_chamber(backend, device_index);
        run_free_colony(backend, device_index);
      });
  return 0;
}
