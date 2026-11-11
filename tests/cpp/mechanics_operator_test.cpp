#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "microsimulator/mechanics.hpp"
#include "microsimulator/simulation.hpp"

namespace {

bool close(float actual, float expected, float tolerance = 1.0e-5F) {
  return std::abs(actual - expected) <= tolerance;
}

microsimulator::CellId add_capsule(microsimulator::WorldState& state, microsimulator::Vec3 center,
                                   microsimulator::Vec3 axis, float length = 4.0F,
                                   float radius = 0.5F) {
  microsimulator::CellInit cell;
  cell.position = center;
  cell.direction = axis;
  cell.length = length;
  cell.radius = radius;
  return state.add_cell(cell);
}

float correction_dot(const std::vector<microsimulator::CellCorrection>& left,
                     const std::vector<microsimulator::CellCorrection>& right) {
  float result = 0.0F;
  for (std::size_t index = 0; index < left.size(); ++index) {
    result += microsimulator::dot(left[index].translation, right[index].translation);
    result += microsimulator::dot(left[index].rotation, right[index].rotation);
    result += left[index].length * right[index].length;
  }
  return result;
}

std::vector<microsimulator::CellCorrection> subtract(
    const std::vector<microsimulator::CellCorrection>& left,
    const std::vector<microsimulator::CellCorrection>& right) {
  auto result = left;
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index].translation = left[index].translation - right[index].translation;
    result[index].rotation = left[index].rotation - right[index].rotation;
    result[index].length = left[index].length - right[index].length;
  }
  return result;
}

float residual_rms(const std::vector<microsimulator::CellCorrection>& residual) {
  if (residual.empty()) {
    return 0.0F;
  }
  return std::sqrt(correction_dot(residual, residual) / static_cast<float>(residual.size()));
}

void assert_correction_close(const microsimulator::CellCorrection& actual,
                             const microsimulator::CellCorrection& expected) {
  assert(close(actual.translation.x, expected.translation.x));
  assert(close(actual.translation.y, expected.translation.y));
  assert(close(actual.translation.z, expected.translation.z));
  assert(close(actual.rotation.x, expected.rotation.x));
  assert(close(actual.rotation.y, expected.rotation.y));
  assert(close(actual.rotation.z, expected.rotation.z));
  assert(close(actual.length, expected.length));
}

void test_contact_free_operator_is_declared_regularizer() {
  microsimulator::WorldState state;
  add_capsule(state, {}, {1.0F, 0.0F, 0.0F});
  const microsimulator::ContactGraph contacts(1, {});
  microsimulator::MechanicsParameters parameters;
  parameters.mu_a = 2.0F;
  parameters.gamma = 4.0F;
  const std::vector input{microsimulator::CellCorrection{
      .translation = {1.0F, 2.0F, 3.0F},
      .rotation = {1.0F, 2.0F, 3.0F},
      .length = 4.0F,
  }};

  const auto output =
      microsimulator::apply_mechanics_operator_cpu(state, contacts, input, parameters);
  const auto mass = 10.0F;
  const auto axial_inertia = 0.5F * mass * 0.5F * 0.5F;
  const auto transverse_inertia = mass * (25.0F + 3.0F * 0.25F) / 12.0F;
  assert(close(output[0].translation.x, 0.25F * mass));
  assert(close(output[0].translation.y, 0.25F * mass * 2.0F));
  assert(close(output[0].translation.z, 0.25F * mass * 3.0F));
  assert(close(output[0].rotation.x, 0.25F * axial_inertia));
  assert(close(output[0].rotation.y, 0.25F * transverse_inertia * 2.0F));
  assert(close(output[0].rotation.z, 0.25F * transverse_inertia * 3.0F));
  assert(close(output[0].length, 4.0F));
}

void test_operator_is_symmetric_and_positive_definite() {
  microsimulator::WorldState state;
  add_capsule(state, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F});
  add_capsule(state, {0.2F, 0.8F, 0.0F}, {1.0F, 0.2F, 0.0F});
  add_capsule(state, {-0.3F, 0.1F, 0.7F}, {0.0F, 1.0F, 0.3F});
  const auto contacts = microsimulator::find_cell_contacts_cpu(state);
  assert(!contacts.empty());

  const std::vector first{
      microsimulator::CellCorrection{{0.2F, -0.4F, 0.3F}, {0.1F, 0.2F, -0.2F}, 0.05F},
      microsimulator::CellCorrection{{-0.1F, 0.7F, -0.2F}, {0.4F, -0.3F, 0.1F}, -0.02F},
      microsimulator::CellCorrection{{0.5F, 0.1F, -0.6F}, {-0.2F, 0.1F, 0.3F}, 0.08F},
  };
  const std::vector second{
      microsimulator::CellCorrection{{-0.3F, 0.2F, 0.1F}, {0.5F, -0.1F, 0.2F}, -0.04F},
      microsimulator::CellCorrection{{0.6F, -0.2F, 0.4F}, {-0.3F, 0.4F, 0.2F}, 0.03F},
      microsimulator::CellCorrection{{-0.2F, 0.8F, 0.1F}, {0.2F, 0.3F, -0.4F}, 0.06F},
  };

  const auto applied_first = microsimulator::apply_mechanics_operator_cpu(state, contacts, first);
  const auto applied_second = microsimulator::apply_mechanics_operator_cpu(state, contacts, second);
  const auto left = correction_dot(first, applied_second);
  const auto right = correction_dot(applied_first, second);
  assert(close(left, right, 2.0e-5F));
  assert(correction_dot(first, applied_first) > 0.0F);
}

void test_solver_converges_and_reports_recomputed_residual() {
  microsimulator::WorldState state;
  add_capsule(state, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F});
  add_capsule(state, {0.0F, 0.8F, 0.0F}, {1.0F, 0.0F, 0.0F});
  const auto contacts = microsimulator::find_cell_contacts_cpu(state);
  microsimulator::MechanicsParameters parameters;
  parameters.residual_rms_tolerance = 1.0e-6F;

  const auto solution = microsimulator::solve_cell_mechanics_cpu(state, contacts, parameters);
  assert(solution.report.status == microsimulator::SolverStatus::converged);
  assert(solution.report.breakdown == microsimulator::SolverBreakdown::none);
  assert(solution.report.iterations > 0);
  assert(solution.corrections[0].translation.y < 0.0F);
  assert(solution.corrections[1].translation.y > 0.0F);

  const auto applied = microsimulator::apply_mechanics_operator_cpu(
      state, contacts, solution.corrections, parameters);
  const auto rhs = microsimulator::build_mechanics_rhs_cpu(state, contacts, parameters);
  const auto recomputed = residual_rms(subtract(rhs, applied));
  assert(close(solution.report.final_residual_rms, recomputed, 1.0e-7F));
  assert(recomputed <= parameters.residual_rms_tolerance);
}

void test_external_rows_contribute_to_operator_rhs_and_solve() {
  microsimulator::WorldState state;
  add_capsule(state, {0.0F, 0.4F, 0.0F}, {1.0F, 0.0F, 0.0F}, 2.0F);
  const microsimulator::ContactGraph contacts(1, {});
  microsimulator::ConstraintSet constraints;
  microsimulator::PlaneConstraintInit plane;
  plane.inward_normal = {0.0F, 1.0F, 0.0F};
  constraints.add_plane(plane);
  const auto external_contacts = microsimulator::find_external_contacts_cpu(state, constraints);
  assert(external_contacts.size() == 2);

  const std::vector input{microsimulator::CellCorrection{
      .translation = {0.0F, 1.0F, 0.0F},
  }};
  const auto applied =
      microsimulator::apply_mechanics_operator_cpu(state, contacts, external_contacts, input);
  assert(close(applied[0].translation.y, 1.3F));
  assert(close(applied[0].rotation.z, 0.0F));
  assert(close(applied[0].length, 0.0F));

  const auto rhs = microsimulator::build_mechanics_rhs_cpu(state, contacts, external_contacts);
  assert(close(rhs[0].translation.y, 0.1F));
  assert(close(rhs[0].rotation.z, 0.0F));
  assert(close(rhs[0].length, 0.0F));

  microsimulator::MechanicsParameters parameters;
  parameters.residual_rms_tolerance = 1.0e-6F;
  const auto solution =
      microsimulator::solve_cell_mechanics_cpu(state, contacts, external_contacts, parameters);
  assert(solution.report.status == microsimulator::SolverStatus::converged);
  assert(solution.corrections[0].translation.y > 0.0F);
}

void test_sphere_rows_drive_cells_toward_the_allowed_region() {
  microsimulator::Simulation outside;
  microsimulator::CellInit outside_cell;
  outside_cell.position = {1.2F, 0.0F, 0.0F};
  outside_cell.length = 0.0F;
  outside_cell.radius = 0.5F;
  outside.add_cell(outside_cell);
  microsimulator::SphereConstraintInit outside_sphere;
  outside_sphere.radius = 1.0F;
  outside.add_sphere_constraint(outside_sphere);
  const auto outside_result = outside.solve_cell_mechanics();
  assert(outside_result.report.status == microsimulator::SolverStatus::converged);
  assert(outside_result.corrections[0].translation.x > 0.0F);

  microsimulator::Simulation inside;
  microsimulator::CellInit inside_cell;
  inside_cell.position = {4.8F, 0.0F, 0.0F};
  inside_cell.length = 0.0F;
  inside_cell.radius = 0.5F;
  inside.add_cell(inside_cell);
  microsimulator::SphereConstraintInit inside_sphere;
  inside_sphere.radius = 5.0F;
  inside_sphere.allowed_region = microsimulator::SphereRegion::inside;
  inside.add_sphere_constraint(inside_sphere);
  const auto inside_result = inside.solve_cell_mechanics();
  assert(inside_result.report.status == microsimulator::SolverStatus::converged);
  assert(inside_result.corrections[0].translation.x < 0.0F);
}

void test_iteration_limit_and_breakdown_are_diagnosed() {
  microsimulator::WorldState limited_state;
  add_capsule(limited_state, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F});
  add_capsule(limited_state, {0.2F, 0.8F, 0.0F}, {1.0F, 0.2F, 0.0F});
  add_capsule(limited_state, {-0.3F, 0.1F, 0.7F}, {0.0F, 1.0F, 0.3F});
  const auto limited_contacts = microsimulator::find_cell_contacts_cpu(limited_state);
  microsimulator::MechanicsParameters limited_parameters;
  limited_parameters.residual_rms_tolerance = 0.0F;
  limited_parameters.max_iterations = 1;
  const auto limited =
      microsimulator::solve_cell_mechanics_cpu(limited_state, limited_contacts, limited_parameters);
  assert(limited.report.status == microsimulator::SolverStatus::iteration_limit);
  assert(limited.report.breakdown == microsimulator::SolverBreakdown::none);
  assert(limited.report.iterations == 1);

  microsimulator::WorldState overflow_state;
  const auto first = add_capsule(overflow_state, {}, {1.0F, 0.0F, 0.0F}, 1.0e20F, 1.0e20F);
  const auto second =
      add_capsule(overflow_state, {0.0F, 1.0F, 0.0F}, {1.0F, 0.0F, 0.0F}, 1.0e20F, 1.0e20F);
  microsimulator::CellContact contact{
      .first_id = first,
      .second_id = second,
      .first_slot = 0,
      .second_slot = 1,
      .point_on_first = {},
      .normal = {0.0F, 1.0F, 0.0F},
      .signed_separation = -1.0F,
      .weight = 1.0F,
  };
  const microsimulator::ContactGraph overflow_contacts(2, {contact});
  const auto broken = microsimulator::solve_cell_mechanics_cpu(overflow_state, overflow_contacts);
  assert(broken.report.status == microsimulator::SolverStatus::breakdown);
  assert(broken.report.breakdown == microsimulator::SolverBreakdown::non_finite_curvature);
}

void test_invalid_inputs_are_rejected() {
  microsimulator::WorldState state;
  add_capsule(state, {}, {1.0F, 0.0F, 0.0F});
  microsimulator::MechanicsParameters parameters;
  parameters.gamma = 0.0F;
  bool rejected = false;
  try {
    static_cast<void>(microsimulator::solve_cell_mechanics_cpu(
        state, microsimulator::ContactGraph(1, {}), parameters));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
}

void test_fixed_cells_are_projected_out_of_cpu_mechanics() {
  microsimulator::WorldState state;
  const auto fixed_id =
      add_capsule(state, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F});
  add_capsule(state, {0.0F, 0.8F, 0.0F}, {1.0F, 0.0F, 0.0F});
  state.set_cell_fixed(fixed_id, true);
  const auto contacts = microsimulator::find_cell_contacts_cpu(state);
  assert(!contacts.empty());

  const microsimulator::CellCorrection fixed_input{
      .translation = {0.4F, -0.3F, 0.2F},
      .rotation = {-0.1F, 0.5F, 0.7F},
      .length = 0.25F,
  };
  const microsimulator::CellCorrection free_input{
      .translation = {-0.2F, 0.6F, 0.3F},
      .rotation = {0.8F, -0.4F, 0.1F},
      .length = -0.15F,
  };
  const std::vector with_fixed_input{fixed_input, free_input};
  const std::vector with_zero_fixed{microsimulator::CellCorrection{}, free_input};
  const auto applied =
      microsimulator::apply_mechanics_operator_cpu(state, contacts, with_fixed_input);
  const auto applied_zero =
      microsimulator::apply_mechanics_operator_cpu(state, contacts, with_zero_fixed);
  assert_correction_close(applied[0], fixed_input);
  assert_correction_close(applied[1], applied_zero[1]);

  const auto rhs = microsimulator::build_mechanics_rhs_cpu(state, contacts);
  assert_correction_close(rhs[0], microsimulator::CellCorrection{});
  const auto solution = microsimulator::solve_cell_mechanics_cpu(state, contacts);
  assert(solution.report.status == microsimulator::SolverStatus::converged);
  assert_correction_close(solution.corrections[0], microsimulator::CellCorrection{});
  assert(solution.corrections[1].translation.y > 0.0F);
}

void test_simulation_exposes_cpu_mechanics_capability() {
  microsimulator::Simulation simulation;
  microsimulator::CellInit first;
  first.length = 4.0F;
  microsimulator::CellInit second = first;
  second.position.y = 0.8F;
  simulation.add_cell(first);
  simulation.add_cell(second);
  assert(simulation.supports(microsimulator::BackendFeature::cell_mechanics));
  const auto result = simulation.solve_cell_mechanics();
  assert(result.report.status == microsimulator::SolverStatus::converged);
  assert(result.corrections.size() == 2);
}

}  // namespace

int main() {
  test_contact_free_operator_is_declared_regularizer();
  test_operator_is_symmetric_and_positive_definite();
  test_solver_converges_and_reports_recomputed_residual();
  test_external_rows_contribute_to_operator_rhs_and_solve();
  test_sphere_rows_drive_cells_toward_the_allowed_region();
  test_iteration_limit_and_breakdown_are_diagnosed();
  test_invalid_inputs_are_rejected();
  test_fixed_cells_are_projected_out_of_cpu_mechanics();
  test_simulation_exposes_cpu_mechanics_capability();
  return 0;
}
