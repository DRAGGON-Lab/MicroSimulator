#include <cmath>
#include <iostream>
#include <numbers>
#include <numeric>
#include <stdexcept>

#include "backend_devices.hpp"
#include "cm/solute_transport.hpp"

namespace {
void check(bool value, const char* message) {
  if (!value) {
    throw std::runtime_error(message);
  }
}

cm::SignalGridVelocityField zero_velocity(const cm::FluidGridSpec& g) {
  cm::SignalGridVelocityField f;
  f.x_faces.resize((g.shape.x + 1) * g.shape.y * g.shape.z);
  f.y_faces.resize(g.shape.x * (g.shape.y + 1) * g.shape.z);
  f.z_faces.resize(g.shape.x * g.shape.y * (g.shape.z + 1));

  return f;
}

void run(cm::BackendKind backend, std::uint32_t device) {
  cm::FluidGridSpec grid;
  grid.shape = {8, 8, 8};
  cm::FluidGeometry geometry(grid, {});
  cm::SoluteTransportSolver solver(backend, device);
  const auto velocity = zero_velocity(grid);
  std::vector<double> initial;

  for (const auto& f : geometry.fragments()) {
    initial.push_back(f.volume *
                      (2 + 0.5 * std::cos(std::numbers::pi * (f.centroid[0] + 0.5) / 8)));
  }

  const double diffusion = 0.7, dt = 1.3;
  const auto result =
      solver.step(geometry, geometry, velocity, {}, {{"tracer", diffusion}}, {}, initial, dt);
  const double damping =
      1 / (1 + dt * 4 * diffusion * std::pow(std::sin(std::numbers::pi / 16), 2));

  for (std::size_t i = 0; i < initial.size(); ++i) {
    const double exact =
        2 + 0.5 * damping *
                std::cos(std::numbers::pi * (geometry.fragments()[i].centroid[0] + 0.5) / 8);
    check(std::abs(result.concentrations[i] - exact) < 2e-4, "backward Euler diffusion eigenmode");
  }

  check(std::abs(result.report.mass_balance_error[0]) < 0.01, "closed diffusion amount ledger");
  // Compare timestep refinement against the exact semidiscrete diffusion
  // solution, independently of the backward-Euler formula checked above.
  const double eigenvalue = 4 * diffusion * std::pow(std::sin(std::numbers::pi / 16), 2);
  double previous_error = INFINITY;

  for (unsigned steps : {1, 2, 4, 8}) {
    auto amounts = initial;

    for (unsigned step = 0; step < steps; ++step) {
      amounts = solver
                    .step(geometry, geometry, velocity, {}, {{"tracer", diffusion}}, {}, amounts,
                          4.0 / steps)
                    .amounts;
    }

    double error = 0;

    for (std::size_t i = 0; i < amounts.size(); ++i) {
      const auto& fragment = geometry.fragments()[i];
      const double exact = 2 + 0.5 * std::exp(-4 * eigenvalue) *
                                   std::cos(std::numbers::pi * (fragment.centroid[0] + 0.5) / 8);
      error = std::max(error, std::abs(amounts[i] / fragment.volume - exact));
    }

    check(error < 0.65 * previous_error, "first-order transport timestep refinement");
    previous_error = error;
  }

  auto channel = velocity;
  std::fill(channel.x_faces.begin(), channel.x_faces.end(), 0.2F);
  std::vector<cm::FlowPort> ports{{.name = "in", .axis = cm::FlowAxis::x},
                                  {.name = "out", .axis = cm::FlowAxis::x, .upper = true}};
  std::vector<double> uniform(initial.size(), 2);
  const auto perfusion = solver.step(geometry, geometry, channel, ports, {{"tracer", 0.1}},
                                     {{"in", {2}}, {"out", {2}}}, uniform, dt);

  for (double c : perfusion.concentrations) {
    check(std::abs(c - 2) < 2e-4, "uniform perfusion concentration");
  }

  check(perfusion.reservoirs[0].amounts[0] > 0 && perfusion.reservoirs[1].amounts[0] < 0,
        "reservoir ledger signs");
  cm::CapsuleBody body;
  body.position = {3.31, 3.67, 3.48};
  body.radius = 1.2;
  body.length = 1.3;
  body.orientation = {std::cos(0.31), 0, std::sin(0.31), 0};
  const cm::FluidGeometry old(grid, {body});
  body.position[0] += 0.12;
  const cm::FluidGeometry moved(grid, {body});
  uniform.clear();

  for (const auto& f : old.fragments()) {
    uniform.push_back(2 * f.volume);
  }

  const auto moving = solver.step(old, moved, velocity, {}, {{"tracer", 0.1}}, {}, uniform, dt);
  double maximum = 0;

  for (double c : moving.concentrations) {
    maximum = std::max(maximum, std::abs(c - 2));
  }

  std::cout << "moving concentration error " << maximum << " GCL "
            << moving.report.maximum_volume_residual << std::endl;
  check(maximum < 4e-4, "moving capsule preserves uniform concentration");
  check(std::abs(moving.report.mass_balance_error[0]) < 0.02, "moving capsule amount ledger");
  const auto uptake = solver.step(old, old, velocity, {}, {{"tracer", 0.1}}, {}, uniform, dt,
                                  {{.body_id = body.id, .solute = 0, .uptake_velocity = 0.05}});
  const double removed = std::accumulate(uniform.begin(), uniform.end(), 0.0) -
                         std::accumulate(uptake.amounts.begin(), uptake.amounts.end(), 0.0);
  check(removed > 0 && uptake.cells[0].amounts[0] > 0, "surface uptake direction");
  check(std::abs(removed - uptake.cells[0].amounts[0]) < 0.01,
        "surface uptake paired amount ledger");

  for (double m : uptake.amounts) {
    check(m >= 0, "nonnegative uptake amount");
  }

  std::cout << "media transport backend " << static_cast<int>(backend) << " device " << device
            << " passed\n";
}
}  // namespace

int main() {
  cm::test::for_each_backend_device(run);
}
