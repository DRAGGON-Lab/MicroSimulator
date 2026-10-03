#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include "backend_devices.hpp"
#include "cm/stokes_flow.hpp"
#include "core/capsule_bodies.hpp"

namespace {
void check(bool value, const char* message) {
  if (!value) {
    throw std::runtime_error(message);
  }
}

void kernel_moments() {
  const double K = 59.0 / 60 - std::sqrt(29.0) / 20;
  double square = 0;

  for (unsigned j = 0; j <= 100; ++j) {
    const double r = double(j) / 100;
    const auto weights = cm::detail::immersed_delta_weights(r);
    std::array<double, 4> moment{};
    double even = 0, squared = 0;

    for (unsigned i = 0; i < 6; ++i) {
      check(weights[i] >= -1e-15, "IB kernel must be nonnegative");
      squared += weights[i] * weights[i];

      if (i % 2 == 0) {
        even += weights[i];
      }

      for (unsigned p = 0; p < 4; ++p) {
        moment[p] += std::pow(r - 3 + i, p) * weights[i];
      }
    }

    if (j == 0) {
      square = squared;
    }

    check(std::abs(moment[0] - 1) < 1e-14 && std::abs(moment[1]) < 1e-14 &&
              std::abs(moment[2] - K) < 1e-14 && std::abs(moment[3]) < 1e-14 &&
              std::abs(even - 0.5) < 1e-14 && std::abs(squared - square) < 1e-14,
          "IB kernel moments and sum of squares");
  }
}

void run(cm::BackendKind backend, std::uint32_t index) {
  cm::FluidGridSpec grid;
  grid.shape = {12, 12, 12};
  cm::CapsuleBody body;
  body.position = {5.5, 5.5, 5.5};
  body.radius = 2;
  body.force_n = {1e-14, 0, 0};
  cm::StokesFlowSolver solver(backend, index);
  cm::LinearSolveParameters parameters;
  parameters.relative_tolerance = 1e-5;
  const auto result = solver.solve_bodies(grid, {}, {}, {body}, parameters);
  const auto& response = result.bodies[0];
  std::cout << "body iterations " << result.report.iterations << ", velocity "
            << response.velocity[0] << ", slip " << response.no_slip_rms_m_s << std::endl;
  check(response.velocity[0] > 0, "force-driven body translation");
  check(
      std::abs(response.velocity[1]) + std::abs(response.velocity[2]) < 1e-3 * response.velocity[0],
      "sphere transverse symmetry");
  check(std::abs(response.hydrodynamic_force_n[0] + 1e-14) < 1e-18, "rigid-body force balance");
  check(response.no_slip_rms_m_s < 1e-4 * response.velocity[0] * grid.length_unit_m,
        "rigid-body no-slip residual");

  if (backend != cm::BackendKind::cpu) {
    const auto oracle = cm::StokesFlowSolver{}.solve_bodies(grid, {}, {}, {body}, parameters);
    check(std::abs(response.velocity[0] - oracle.bodies[0].velocity[0]) <
              2e-4 * oracle.bodies[0].velocity[0],
          "rigid-body CPU/GPU parity");
  }

  body.fixed = true;
  auto fixed =
      solver.solve_bodies(grid, {}, {{.name = "in", .value = 0.01}, {.name = "out", .upper = true}},
                          {body}, parameters);
  check(fixed.bodies[0].velocity[0] == 0 && fixed.bodies[0].velocity[1] == 0, "fixed body motion");
  check(fixed.bodies[0].hydrodynamic_force_n[1] > 0, "fixed-body drag direction");
  body.length_rate = 0.01;
  auto growing = solver.solve_bodies(grid, {}, {{.name = "in"}, {.name = "out", .upper = true}},
                                     {body}, parameters);
  const double rate = std::numbers::pi * body.radius * body.radius * body.length_rate * 1e-18;
  check(std::abs(growing.report.net_flow_rate_m3_s - rate) < 1e-4 * rate,
        "growth displaces medium");
  check(std::abs(growing.report.source_volume_rate_m3_s - rate) < 1e-12 * rate,
        "growth source volume");
  check(growing.report.continuity_rms_per_s < 1e-4 * growing.report.max_speed_m_s / 1e-6,
        "growth continuity residual");
  bool rejected = false;

  try {
    (void)solver.solve_bodies(grid, {}, {}, {body}, parameters);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }

  check(rejected, "sealed incompressible chamber must reject growth");
  body.fixed = false;
  body.length_rate = 0;
  body.position = {1.61, 5.5, 5.5};
  body.force_n = {-1e-14, 0, 0};
  cm::FluidBodyStepParameters stepping;
  stepping.minimum_gap_m = 0.1e-6;
  const auto step = solver.propose_body_step(grid, {}, {}, {body}, 100, parameters, stepping);
  std::cout << "contact dt " << step.accepted_dt << " x " << step.bodies[0].position[0] << " vx "
            << step.flow.bodies[0].velocity[0] << " count " << step.contacts.size() << std::endl;
  check(step.bodies[0].position[0] - body.radius + 0.5 >= 0.1, "wall contact must not penetrate");
  check(body.position[0] == 1.61, "body proposal changed caller-owned inputs");
  check(!step.contacts.empty() && step.contacts[0].normal_force_n > 0,
        "compressive contact reaction");
  check(std::abs(step.flow.bodies[0].hydrodynamic_force_n[0] + step.contacts[0].normal_force_n -
                 1e-14) < 1e-18,
        "contact and hydrodynamic force balance");
  grid.shape = {16, 12, 12};
  cm::CapsuleBody first, second;
  first.id = 1;
  second.id = 2;
  first.radius = second.radius = 2;
  first.position = {5.45, 5.5, 5.5};
  second.position = {9.55002, 5.5, 5.5};
  first.force_n = {1e-14, 0, 0};
  second.force_n = {-1e-14, 0, 0};
  const auto pair =
      solver.propose_body_step(grid, {}, {}, {first, second}, 1, parameters, stepping);
  check(pair.bodies[1].position[0] - pair.bodies[0].position[0] - 4 >= 0.1,
        "pair contact must preserve minimum gap");
  check(pair.contacts.size() == 1 && pair.contacts[0].normal_force_n > 0,
        "compressive pair contact reaction");
  const double reaction = pair.contacts[0].normal_force_n;
  check(std::abs(pair.flow.bodies[0].hydrodynamic_force_n[0] + 1e-14 - reaction) < 1e-18 &&
            std::abs(pair.flow.bodies[1].hydrodynamic_force_n[0] - 1e-14 + reaction) < 1e-18,
        "pair contact equal and opposite force balance");
  std::cout << "media body backend " << static_cast<int>(backend) << " device " << index
            << " passed\n";
}
}  // namespace

int main() {
  kernel_moments();
  cm::test::for_each_backend_device(run);
}
