#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "../../cpp/core/flow_system.hpp"
#include "microsimulator/simulation.hpp"

namespace {

microsimulator::SignalGridSpec duct(std::uint32_t nx, std::uint32_t ny, std::uint32_t nz,
                                    microsimulator::Vec3 spacing = {1.0F, 1.0F, 1.0F}) {
  microsimulator::SignalGridSpec spec;
  spec.signal_count = 1;
  spec.shape = {.x = nx, .y = ny, .z = nz};
  spec.spacing = spacing;
  spec.diffusion = {0.0F};
  spec.advection = {{0.0F, 0.0F, 0.0F}};
  spec.y_lower = {.kind = microsimulator::GridBoundaryKind::fixed, .values = {0.0F}};
  spec.y_upper = {.kind = microsimulator::GridBoundaryKind::fixed, .values = {0.0F}};
  return spec;
}

std::size_t y_face(const microsimulator::SignalGridSpec& spec, std::uint32_t x, std::uint32_t y,
                   std::uint32_t z) {
  return (static_cast<std::size_t>(x) * (spec.shape.y + 1) + y) * spec.shape.z + z;
}

// A box of fluid with every side face a fixed boundary, so that a port may
// open any of them.
microsimulator::SignalGridSpec open_box(std::uint32_t nx, std::uint32_t ny, std::uint32_t nz,
                                        microsimulator::Vec3 spacing = {1.0F, 1.0F, 1.0F}) {
  auto spec = duct(nx, ny, nz, spacing);
  spec.x_lower = {.kind = microsimulator::GridBoundaryKind::fixed, .values = {0.0F}};
  spec.x_upper = {.kind = microsimulator::GridBoundaryKind::fixed, .values = {0.0F}};
  return spec;
}

microsimulator::FlowPort port(microsimulator::GridFace face, std::uint32_t begin,
                              std::uint32_t end, std::uint32_t depth,
                              microsimulator::FlowPortKind kind, float value) {
  return {.face = face,
          .patch = {.begin = {begin, 0}, .end = {end, depth}},
          .kind = kind,
          .value = value};
}

// The largest net volume leaving any fluid site per unit time.
float largest_divergence(const microsimulator::SignalGridSpec& spec,
                         const microsimulator::SignalGridVelocityField& field) {
  const auto nx = spec.shape.x, ny = spec.shape.y, nz = spec.shape.z;
  float largest = 0.0F;
  for (std::uint32_t x = 0; x < nx; ++x) {
    for (std::uint32_t y = 0; y < ny; ++y) {
      for (std::uint32_t z = 0; z < nz; ++z) {
        if (spec.solid_site((static_cast<std::size_t>(x) * ny + y) * nz + z)) continue;
        const auto x_low = (static_cast<std::size_t>(x) * ny + y) * nz + z;
        const auto x_high = (static_cast<std::size_t>(x + 1) * ny + y) * nz + z;
        const auto z_low = (static_cast<std::size_t>(x) * ny + y) * (nz + 1) + z;
        const auto net =
            (field.x_faces[x_high] - field.x_faces[x_low]) / spec.spacing.x +
            (field.y_faces[y_face(spec, x, y + 1, z)] - field.y_faces[y_face(spec, x, y, z)]) /
                spec.spacing.y +
            (field.z_faces[z_low + 1] - field.z_faces[z_low]) / spec.spacing.z;
        largest = std::max(largest, std::abs(net));
      }
    }
  }
  return largest;
}

template <typename Exception, typename Function>
void assert_throws(Function&& function) {
  bool rejected = false;
  try {
    function();
  } catch (const Exception&) {
    rejected = true;
  }
  assert(rejected);
}

}  // namespace

int main() {
  {
    auto spec = duct(4, 6, 3, {0.7F, 1.1F, 0.9F});
    spec.obstacles.assign(spec.site_count(), 0);
    spec.obstacles[22] = 1;
    std::vector<float> drag(spec.site_count(), 0.3F);
    const microsimulator::detail::ResolvedFlowSystem system(spec, drag,
                                                            microsimulator::FlowAxis::y);
    std::vector<double> u(system.active().size()), v(u.size()), p(spec.site_count());
    for (std::size_t i = 0; i < u.size(); ++i) {
      u[i] = system.active()[i] ? std::sin(static_cast<double>(i)) : 0;
      v[i] = system.active()[i] ? std::cos(static_cast<double>(i) * 0.7) : 0;
    }
    for (std::size_t i = 0; i < p.size(); ++i) p[i] = std::sin(static_cast<double>(i) * 0.3);
    const auto dot = [](const auto& a, const auto& b) {
      double sum = 0;
      for (std::size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
      return sum;
    };
    const auto gp = system.gradient(p), du = system.divergence(u);
    assert(std::abs(dot(u, gp) + dot(du, p)) < 1e-10);
    std::vector<double> au, av;
    system.apply_momentum(u, au);
    system.apply_momentum(v, av);
    assert(std::abs(dot(u, av) - dot(au, v)) < 1e-10);
    assert(dot(u, au) > 0);
  }
  {
    const auto spec = duct(4, 8, 3);
    microsimulator::Simulation simulation;
    const auto result = simulation.solve_depth_averaged_flow(
        spec, {}, {.mean_inlet_speed = 5.0F, .axis = microsimulator::FlowAxis::y});
    assert(simulation.supports(microsimulator::BackendFeature::depth_averaged_flow));
    assert(result.report.iterations > 0);
    assert(result.report.relative_residual <= 1.0e-6F);
    assert(std::ranges::all_of(result.field.y_faces,
                               [](float value) { return std::abs(value - 5.0F) <= 1.0e-5F; }));
    assert(std::ranges::all_of(result.field.x_faces,
                               [](float value) { return std::abs(value) <= 2.0e-5F; }));
    assert(std::ranges::all_of(result.field.z_faces,
                               [](float value) { return std::abs(value) <= 2.0e-5F; }));
  }

  {
    const auto spec = duct(2, 6, 1);
    std::vector<float> mobility(spec.site_count());
    for (std::uint32_t x = 0; x < spec.shape.x; ++x) {
      for (std::uint32_t y = 0; y < spec.shape.y; ++y) {
        mobility[(static_cast<std::size_t>(x) * spec.shape.y) + y] = x == 0 ? 1.0F : 3.0F;
      }
    }
    const auto result = microsimulator::solve_depth_averaged_flow_cpu(
        spec, mobility, {.mean_inlet_speed = 4.0F, .axis = microsimulator::FlowAxis::y});
    const auto slow = result.field.y_faces[y_face(spec, 0, 3, 0)];
    const auto fast = result.field.y_faces[y_face(spec, 1, 3, 0)];
    assert(std::abs((fast / slow) - 3.0F) <= 2.0e-5F);
  }

  {
    constexpr std::uint32_t nx = 8;
    const auto spec = duct(nx, 6, 1, {1.0F / static_cast<float>(nx), 0.25F, 1.0F});
    microsimulator::Simulation simulation;
    const auto result = simulation.solve_resolved_flow(
        spec, {}, {.mean_inlet_speed = 1.0F, .axis = microsimulator::FlowAxis::y});
    assert(simulation.supports(microsimulator::BackendFeature::resolved_flow));
    float max_error = 0.0F;
    for (std::uint32_t x = 0; x < nx; ++x) {
      const auto position = (static_cast<float>(x) + 0.5F) / static_cast<float>(nx);
      const auto exact = 6.0F * position * (1.0F - position);
      max_error =
          std::max(max_error, std::abs(result.field.y_faces[y_face(spec, x, 3, 0)] - exact));
    }
    assert(max_error / 1.5F < 0.02F);
    assert(result.report.outer_iterations > 0);
    assert(result.report.inner_iterations > 0);
    assert(result.report.divergence_rms < 2.0e-5F);
    assert(result.report.min_gap_voxels == nx);
  }

  using microsimulator::FlowPortKind;
  using microsimulator::GridFace;
  {
    // Two inlets on one face at different speeds feed one outlet. Each port
    // spans the grid's two layers of depth.
    auto spec = open_box(6, 8, 2, {0.5F, 1.0F, 0.25F});
    // The inlets' face holds a reservoir patch through its full depth, as a
    // device gives each inlet its own media.
    spec.y_lower.patches = {{.region = {.begin = {0, 0}, .end = {2, 2}}, .values = {1.0F}}};
    const std::vector<microsimulator::FlowPort> ports{
        port(GridFace::y_lower, 0, 2, 2, FlowPortKind::speed, 3.0F),
        port(GridFace::y_lower, 4, 6, 2, FlowPortKind::speed, 1.0F),
        port(GridFace::y_upper, 0, 6, 2, FlowPortKind::pressure, 0.0F),
    };
    microsimulator::Simulation simulation;
    const auto result = simulation.solve_depth_averaged_port_flow(spec, ports);
    assert(result.report.solves == 2);
    assert(result.report.iterations > 0);
    assert(result.report.relative_residual <= 1.0e-6F);
    assert(result.ports.size() == 3);
    // A face is 0.5 by 0.25, and each inlet opens four of them.
    assert(std::abs(result.ports[0].mean_speed - 3.0F) <= 1.0e-5F);
    assert(std::abs(result.ports[1].mean_speed - 1.0F) <= 1.0e-5F);
    assert(std::abs(result.ports[0].flux - 1.5F) <= 1.0e-5F);
    assert(std::abs(result.ports[1].flux - 0.5F) <= 1.0e-5F);
    assert(std::abs(result.ports[2].flux + 2.0F) <= 1.0e-5F);
    assert(std::abs(result.ports[2].mean_speed + (2.0F / 1.5F)) <= 1.0e-5F);
    // The faster inlet needs the higher pressure, and both sit above the outlet.
    assert(result.ports[0].pressure > result.ports[1].pressure);
    assert(result.ports[1].pressure > result.ports[2].pressure);
    assert(result.ports[2].pressure == 0.0F);
    assert(largest_divergence(spec, result.field) <= 2.0e-5F);
    // The lower face is closed between the inlets.
    assert(result.field.y_faces[y_face(spec, 2, 0, 0)] == 0.0F);
    assert(result.field.y_faces[y_face(spec, 3, 0, 1)] == 0.0F);
    assert(result.field.y_faces[y_face(spec, 0, 0, 0)] > 0.0F);
  }

  {
    // An inlet on a face perpendicular to the outlet's turns the flow.
    const auto spec = open_box(5, 7, 1);
    const std::vector<microsimulator::FlowPort> ports{
        port(GridFace::x_lower, 1, 3, 1, FlowPortKind::speed, 2.0F),
        port(GridFace::y_upper, 0, 5, 1, FlowPortKind::pressure, 0.0F),
    };
    const auto result = microsimulator::solve_depth_averaged_port_flow_cpu(spec, ports, {});
    assert(result.report.solves == 1);
    assert(std::abs(result.ports[0].flux - 4.0F) <= 1.0e-5F);
    assert(std::abs(result.ports[1].flux + 4.0F) <= 1.0e-5F);
    assert(largest_divergence(spec, result.field) <= 2.0e-5F);
    // Flow enters along +x and leaves along +y.
    assert(result.field.x_faces[1] > 0.0F && result.field.x_faces[2] > 0.0F);
    assert(result.field.x_faces[0] == 0.0F);
    assert(result.field.y_faces[y_face(spec, 2, 7, 0)] > 0.0F);

    // The same port leaving at that speed draws the flow the other way.
    auto reversed = ports;
    reversed[0].value = -2.0F;
    const auto drawn = microsimulator::solve_depth_averaged_port_flow_cpu(spec, reversed, {});
    assert(std::abs(drawn.ports[0].flux + 4.0F) <= 1.0e-5F);
    assert(drawn.ports[0].pressure < 0.0F);
    assert(drawn.field.y_faces[y_face(spec, 2, 7, 0)] < 0.0F);
  }

  {
    // Ports over the whole lower and upper faces of an axis state the flow the
    // single-axis solve does, here around an obstacle with varying mobility.
    auto spec = duct(5, 8, 2, {0.8F, 1.2F, 0.6F});
    spec.obstacles.assign(spec.site_count(), 0);
    std::vector<float> mobility(spec.site_count(), 1.0F);
    for (std::uint32_t x = 0; x < spec.shape.x; ++x) {
      for (std::uint32_t y = 0; y < spec.shape.y; ++y) {
        for (std::uint32_t z = 0; z < spec.shape.z; ++z) {
          const auto site = (static_cast<std::size_t>(x) * spec.shape.y + y) * spec.shape.z + z;
          spec.obstacles[site] = (x == 2 && y >= 3 && y <= 4) ? 1 : 0;
          mobility[site] = 0.5F + (0.25F * static_cast<float>(x));
        }
      }
    }
    const auto stated = microsimulator::solve_depth_averaged_flow_cpu(
        spec, mobility, {.mean_inlet_speed = 5.0F, .axis = microsimulator::FlowAxis::y});
    const std::vector<microsimulator::FlowPort> ports{
        port(GridFace::y_lower, 0, 5, 2, FlowPortKind::speed, 5.0F),
        port(GridFace::y_upper, 0, 5, 2, FlowPortKind::pressure, 0.0F),
    };
    const auto through = microsimulator::solve_depth_averaged_port_flow_cpu(spec, ports, mobility);
    const auto agree = [&](const std::vector<float>& actual, const std::vector<float>& expected) {
      assert(actual.size() == expected.size());
      for (std::size_t face = 0; face < actual.size(); ++face) {
        assert(std::abs(actual[face] - expected[face]) <= 2.0e-5F * stated.report.max_speed);
      }
    };
    agree(through.field.x_faces, stated.field.x_faces);
    agree(through.field.y_faces, stated.field.y_faces);
    agree(through.field.z_faces, stated.field.z_faces);
    assert(std::abs(through.report.max_speed - stated.report.max_speed) <=
           2.0e-5F * stated.report.max_speed);
  }

  {
    // Pressure ports alone drive the flow their difference draws. With unit
    // mobility a duct of length L carries the speed (p_in - p_out) / L.
    const auto spec = duct(3, 5, 1, {1.0F, 0.5F, 1.0F});
    const std::vector<microsimulator::FlowPort> ports{
        port(GridFace::y_lower, 0, 3, 1, FlowPortKind::pressure, 3.0F),
        port(GridFace::y_upper, 0, 3, 1, FlowPortKind::pressure, 1.0F),
    };
    const auto result = microsimulator::solve_depth_averaged_port_flow_cpu(spec, ports, {});
    assert(result.report.solves == 1);
    assert(std::ranges::all_of(result.field.y_faces,
                               [](float value) { return std::abs(value - 0.8F) <= 1.0e-5F; }));
    assert(result.ports[0].pressure == 3.0F && result.ports[1].pressure == 1.0F);
    assert(std::abs(result.ports[0].mean_speed - 0.8F) <= 1.0e-5F);
    assert(std::abs(result.ports[1].mean_speed + 0.8F) <= 1.0e-5F);
  }

  {
    // Ports that do not state a solvable flow are refused.
    const auto spec = open_box(4, 6, 2);
    const auto inlet = port(GridFace::y_lower, 0, 2, 2, FlowPortKind::speed, 1.0F);
    const auto outlet = port(GridFace::y_upper, 0, 4, 2, FlowPortKind::pressure, 0.0F);
    const auto solve = [&](const microsimulator::SignalGridSpec& grid,
                           std::vector<microsimulator::FlowPort> ports) {
      static_cast<void>(microsimulator::solve_depth_averaged_port_flow_cpu(grid, ports, {}));
    };
    solve(spec, {inlet, outlet});
    assert_throws<std::invalid_argument>([&] { solve(spec, {}); });
    assert_throws<std::invalid_argument>([&] { solve(spec, {inlet}); });
    assert_throws<std::invalid_argument>([&] { solve(spec, {outlet}); });
    assert_throws<std::invalid_argument>([&] {
      solve(spec, {inlet, port(GridFace::y_lower, 1, 3, 2, FlowPortKind::speed, 1.0F), outlet});
    });
    assert_throws<std::invalid_argument>(
        [&] { solve(spec, {port(GridFace::y_lower, 0, 5, 2, FlowPortKind::speed, 1.0F), outlet}); });
    assert_throws<std::invalid_argument>(
        [&] { solve(spec, {port(GridFace::y_lower, 2, 2, 2, FlowPortKind::speed, 1.0F), outlet}); });
    // A shallow-flow port spans its columns' depth and opens a side face.
    assert_throws<std::invalid_argument>(
        [&] { solve(spec, {port(GridFace::y_lower, 0, 2, 1, FlowPortKind::speed, 1.0F), outlet}); });
    auto capped = spec;
    capped.z_upper = {.kind = microsimulator::GridBoundaryKind::fixed, .values = {0.0F}};
    assert_throws<std::invalid_argument>([&] {
      solve(capped, {{.face = GridFace::z_upper,
                      .patch = {.begin = {0, 0}, .end = {2, 2}},
                      .kind = FlowPortKind::speed,
                      .value = 1.0F},
                     outlet});
    });
    // A port needs a fixed boundary to open, and fluid behind it.
    auto closed = spec;
    closed.x_lower = {};
    closed.x_upper = {};
    assert_throws<std::invalid_argument>([&] {
      solve(closed, {port(GridFace::x_lower, 0, 2, 2, FlowPortKind::speed, 1.0F), outlet});
    });
    auto blocked = spec;
    blocked.obstacles.assign(blocked.site_count(), 0);
    for (std::uint32_t z = 0; z < 2; ++z) {
      blocked.obstacles[(0 * 6 + 0) * 2 + z] = 1;
      blocked.obstacles[(1 * 6 + 0) * 2 + z] = 1;
    }
    assert_throws<std::invalid_argument>([&] { solve(blocked, {inlet, outlet}); });
    // A wall across the box leaves the inlet no way to the outlet.
    auto walled = spec;
    walled.obstacles.assign(walled.site_count(), 0);
    for (std::uint32_t x = 0; x < 4; ++x) {
      for (std::uint32_t z = 0; z < 2; ++z) {
        walled.obstacles[(static_cast<std::size_t>(x) * 6 + 3) * 2 + z] = 1;
      }
    }
    assert_throws<std::runtime_error>([&] { solve(walled, {inlet, outlet}); });
  }

  {
    // Resolved flow through a port pair across a channel's whole width is the
    // single-axis solve: plane Poiseuille flow at the stated mean speed.
    constexpr std::uint32_t nx = 8;
    const auto spec = duct(nx, 6, 1, {1.0F / static_cast<float>(nx), 0.25F, 1.0F});
    const std::vector<microsimulator::FlowPort> ports{
        port(GridFace::y_lower, 0, nx, 1, FlowPortKind::speed, 1.0F),
        port(GridFace::y_upper, 0, nx, 1, FlowPortKind::pressure, 0.0F),
    };
    microsimulator::Simulation simulation;
    const auto result = simulation.solve_resolved_port_flow(spec, ports);
    const auto stated = simulation.solve_resolved_flow(
        spec, {}, {.mean_inlet_speed = 1.0F, .axis = microsimulator::FlowAxis::y});
    float max_error = 0.0F;
    for (std::uint32_t x = 0; x < nx; ++x) {
      const auto position = (static_cast<float>(x) + 0.5F) / static_cast<float>(nx);
      const auto exact = 6.0F * position * (1.0F - position);
      const auto solved = result.field.y_faces[y_face(spec, x, 3, 0)];
      max_error = std::max(max_error, std::abs(solved - exact));
      assert(std::abs(solved - stated.field.y_faces[y_face(spec, x, 3, 0)]) <= 2.0e-4F);
    }
    assert(max_error / 1.5F < 0.02F);
    assert(result.report.solves == 1);
    assert(result.report.outer_iterations > 0 && result.report.inner_iterations > 0);
    assert(result.report.divergence_rms < 2.0e-5F);
    // The channel's width is a gap between walls; its length runs port to port.
    assert(result.report.min_gap_voxels == nx);
    assert(std::abs(result.ports[0].mean_speed - 1.0F) <= 1.0e-5F);
    assert(std::abs(result.ports[1].mean_speed + 1.0F) <= 1.0e-4F);
    assert(result.ports[0].pressure > 0.0F);
  }

  {
    // A side inlet joins a main channel: each inlet carries its stated speed
    // and the outlet carries their sum.
    const auto spec = open_box(8, 10, 1, {0.25F, 0.25F, 1.0F});
    const std::vector<microsimulator::FlowPort> ports{
        port(GridFace::y_lower, 0, 8, 1, FlowPortKind::speed, 1.0F),
        port(GridFace::x_lower, 3, 6, 1, FlowPortKind::speed, 0.5F),
        port(GridFace::y_upper, 0, 8, 1, FlowPortKind::pressure, 0.0F),
    };
    const auto result = microsimulator::solve_resolved_port_flow_cpu(spec, ports, {});
    assert(result.report.solves == 2);
    assert(std::abs(result.ports[0].mean_speed - 1.0F) <= 1.0e-4F);
    assert(std::abs(result.ports[1].mean_speed - 0.5F) <= 1.0e-4F);
    // Fluxes are per unit depth: 8 faces and 3 faces, each 0.25 wide.
    assert(std::abs(result.ports[0].flux - 2.0F) <= 2.0e-4F);
    assert(std::abs(result.ports[1].flux - 0.375F) <= 2.0e-4F);
    assert(std::abs(result.ports[2].flux + 2.375F) <= 5.0e-4F);
    assert(result.report.divergence_rms < 5.0e-5F);
    assert(largest_divergence(spec, result.field) <= 5.0e-4F);
    // No-slip walls: the closed parts of the side face carry no flow.
    assert(result.field.x_faces[0] == 0.0F);
    assert(result.field.x_faces[4] > 0.0F);
  }

  {
    // A resolved port may open a z face, here a hole in the lid of a box.
    auto spec = open_box(4, 6, 3, {0.5F, 0.5F, 0.5F});
    spec.z_upper = {.kind = microsimulator::GridBoundaryKind::fixed, .values = {0.0F}};
    const std::vector<microsimulator::FlowPort> ports{
        {.face = GridFace::z_upper,
         .patch = {.begin = {1, 1}, .end = {3, 3}},
         .kind = FlowPortKind::speed,
         .value = 0.8F},
        port(GridFace::y_upper, 0, 4, 3, FlowPortKind::pressure, 0.0F),
    };
    const auto result = microsimulator::solve_resolved_port_flow_cpu(spec, ports, {});
    assert(std::abs(result.ports[0].mean_speed - 0.8F) <= 1.0e-4F);
    assert(std::abs(result.ports[0].flux + result.ports[1].flux) <= 5.0e-4F);
    // Flow enters downward through the lid.
    const auto lid = [&](std::uint32_t x, std::uint32_t y) {
      return result.field.z_faces[(static_cast<std::size_t>(x) * 6 + y) * 4 + 3];
    };
    assert(lid(1, 1) < 0.0F && lid(2, 2) < 0.0F);
    assert(lid(0, 0) == 0.0F && lid(3, 4) == 0.0F);
    assert_throws<std::invalid_argument>(
        [&] { static_cast<void>(microsimulator::solve_resolved_port_flow_cpu(spec, {}, {})); });
  }

  {
    auto spec = duct(3, 4, 1);
    spec.obstacles.assign(spec.site_count(), 0);
    for (std::uint32_t x = 0; x < spec.shape.x; ++x) {
      spec.obstacles[(static_cast<std::size_t>(x) * spec.shape.y) + 2] = 1;
    }
    assert_throws<std::runtime_error>(
        [&] { static_cast<void>(microsimulator::solve_depth_averaged_flow_cpu(spec, {})); });
    assert_throws<std::runtime_error>(
        [&] { static_cast<void>(microsimulator::solve_resolved_flow_cpu(spec, {})); });
  }
}
