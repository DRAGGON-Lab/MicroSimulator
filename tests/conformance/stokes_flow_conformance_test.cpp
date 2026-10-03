#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include "backend_devices.hpp"
#include "cm/stokes_flow.hpp"

namespace {
void check(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void close(double actual, double expected, double relative, double absolute, const char* label) {
  if (std::abs(actual - expected) > absolute + relative * std::abs(expected)) {
    std::cerr << label << ": " << actual << " != " << expected << '\n';
    throw std::runtime_error(label);
  }
}

template <class Function>
void rejects(Function&& fn) {
  try {
    fn();
  } catch (const std::invalid_argument&) {
    return;
  }

  throw std::runtime_error("expected invalid argument");
}

cm::FluidGridSpec grid() {
  cm::FluidGridSpec g;
  g.shape = {4, 8, 4};

  return g;
}

std::vector<cm::FlowPort> ports() {
  return {{.name = "in", .value = 0.01}, {.name = "out", .upper = true}};
}

void fields(const cm::FluidFlowResult& a, const cm::FluidFlowResult& b, double factor = 1) {
  for (auto [x, y] : {std::pair{&a.field.x_faces, &b.field.x_faces},
                      std::pair{&a.field.y_faces, &b.field.y_faces},
                      std::pair{&a.field.z_faces, &b.field.z_faces}}) {
    check(x->size() == y->size(), "field shape");

    for (std::size_t i = 0; i < x->size(); ++i) {
      close((*x)[i], factor * (*y)[i], 1e-3, 3e-5, "field parity");
    }
  }
}

void run(cm::BackendKind backend, std::uint32_t index) {
  auto g = grid();
  cm::FluidProperties medium;
  cm::StokesFlowSolver solver(backend, index), reference;
  cm::LinearSolveParameters parameters;
  parameters.relative_tolerance = 1e-6;
  parameters.absolute_tolerance = 1e-8;
  auto p = ports();
  const auto expected = reference.solve(g, medium, p, parameters);
  const auto result = solver.solve(g, medium, p, parameters);
  fields(result, expected);

  for (const auto axis : {cm::FlowAxis::x, cm::FlowAxis::z}) {
    auto rotated_grid = g;
    rotated_grid.shape = axis == cm::FlowAxis::x ? cm::GridShape{8, 4, 4} : cm::GridShape{4, 4, 8};
    auto rotated_ports = ports();

    for (auto& port : rotated_ports) {
      port.axis = axis;
    }

    const auto rotated = solver.solve(rotated_grid, medium, rotated_ports, parameters);
    close(rotated.ports[1].flow_rate_m3_s, result.ports[1].flow_rate_m3_s, 1e-4, 1e-25,
          "axis covariance");
  }

  check(result.ports[0].flow_rate_m3_s < 0 && result.ports[1].flow_rate_m3_s > 0, "port signs");
  check(result.report.relative_residual < 1.1e-6, "true residual");
  close(result.report.net_flow_rate_m3_s, 0, 0, std::abs(result.ports[0].flow_rate_m3_s) * 1e-4,
        "fluid conservation");
  check(result.report.divergence_rms_per_s < 1e-4 * result.report.max_speed_m_s / 1e-6,
        "continuity");

  for (auto& port : p) {
    port.value += 37;
  }

  const auto shifted = solver.solve(g, medium, p, parameters);
  fields(shifted, result);

  for (std::size_t i = 0; i < shifted.pressure_pa.size(); ++i) {
    close(shifted.pressure_pa[i], result.pressure_pa[i] + 37, 0, 1e-7, "pressure offset");
  }

  medium.viscosity_pa_s *= 2;
  fields(solver.solve(g, medium, ports(), parameters), result, 0.5);
  medium.viscosity_pa_s /= 2;
  p = ports();
  p[0].kind = cm::FlowPortKind::flow_rate;
  p[0].value = result.ports[0].flow_rate_m3_s;
  const auto pumped = solver.solve(g, medium, p, parameters);
  fields(pumped, result);
  close(pumped.ports[0].pressure_pa, 0.01, 1e-4, 1e-8, "pump pressure");
  close(pumped.ports[0].flow_rate_m3_s, p[0].value, 1e-4, 1e-25, "pump rate");
  p[1].kind = cm::FlowPortKind::flow_rate;
  p[1].value = -p[0].value;
  fields(solver.solve(g, medium, p, parameters), result);
  p[1].value *= 0.9;
  rejects([&] {
    (void)solver.solve(g, medium, p);
  });
  p = ports();
  p.push_back(p[0]);
  p.back().name = "duplicate";
  rejects([&] {
    (void)solver.solve(g, medium, p);
  });
  p = ports();
  p[0].sites = {4};
  rejects([&] {
    (void)solver.solve(g, medium, p);
  });
  auto zero = solver.solve(g, medium, {});
  check(zero.report.max_speed_m_s == 0, "sealed rest");
  p = ports();
  p[0].value = -p[0].value;
  fields(solver.solve(g, medium, p, parameters), result, -1);
  // The inlet is split into two independently prescribed reservoir patches.
  p = ports();
  cm::FlowPort second = p[0];
  second.name = "second";

  for (std::uint32_t x = 0; x < 4; ++x) {
    for (std::uint32_t z = 0; z < 4; ++z) {
      (x < 2 ? p[0].sites : second.sites).push_back(x * 8 * 4 + z);
    }
  }

  p.push_back(second);
  auto multi = solver.solve(g, medium, p, parameters);
  fields(multi, result);
  close(multi.ports[0].flow_rate_m3_s, multi.ports[2].flow_rate_m3_s, 1e-4, 1e-25,
        "junction symmetry");
  parameters.memory_limit_bytes = 1;
  rejects([&] {
    (void)solver.solve(g, medium, p, parameters);
  });
  std::cout << "media flow backend " << static_cast<int>(backend) << " device " << index
            << " passed\n";
}
}  // namespace

int main() {
  cm::test::for_each_backend_device(run);
}
