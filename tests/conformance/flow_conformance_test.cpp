#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

#include "backend_devices.hpp"
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

std::size_t site_index(const microsimulator::SignalGridSpec& spec, std::uint32_t x, std::uint32_t y,
                       std::uint32_t z) {
  return (static_cast<std::size_t>(x) * spec.shape.y + y) * spec.shape.z + z;
}

bool close(float actual, float expected) {
  constexpr float absolute_tolerance = 8.0e-4F;
  constexpr float relative_tolerance = 8.0e-4F;
  return std::abs(actual - expected) <=
         absolute_tolerance + relative_tolerance * std::abs(expected);
}

void compare_component(std::span<const float> actual, std::span<const float> expected,
                       std::string_view scenario, std::string_view component) {
  assert(actual.size() == expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (!close(actual[index], expected[index])) {
      std::cerr << scenario << ' ' << component << " face " << index << ": actual=" << actual[index]
                << " expected=" << expected[index] << '\n';
      std::abort();
    }
  }
}

void compare_fields(const microsimulator::SignalGridVelocityField& actual,
                    const microsimulator::SignalGridVelocityField& expected,
                    std::string_view scenario) {
  compare_component(actual.x_faces, expected.x_faces, scenario, "x");
  compare_component(actual.y_faces, expected.y_faces, scenario, "y");
  compare_component(actual.z_faces, expected.z_faces, scenario, "z");
}

void run_depth_case(microsimulator::BackendKind backend, std::uint32_t device_index) {
  auto spec = duct(5, 8, 2, {0.7F, 1.1F, 0.6F});
  spec.obstacles.assign(spec.site_count(), 0);
  for (std::uint32_t y = 3; y <= 4; ++y) {
    spec.obstacles[site_index(spec, 2, y, 1)] = 1;
  }
  std::vector<float> mobility(spec.site_count(), 1.0F);
  for (std::uint32_t x = 0; x < spec.shape.x; ++x) {
    for (std::uint32_t y = 0; y < spec.shape.y; ++y) {
      for (std::uint32_t z = 0; z < spec.shape.z; ++z) {
        mobility[site_index(spec, x, y, z)] =
            spec.solid_site(site_index(spec, x, y, z)) ? 0.0F : 0.4F + 0.1F * static_cast<float>(x);
      }
    }
  }
  microsimulator::DepthAveragedFlowParameters parameters;
  parameters.mean_inlet_speed = 3.5F;
  parameters.relative_tolerance = 1.0e-6F;
  microsimulator::Simulation reference;
  microsimulator::Simulation candidate(backend, 0, 0, device_index);
  const auto expected = reference.solve_depth_averaged_flow(spec, mobility, parameters);
  const auto actual = candidate.solve_depth_averaged_flow(spec, mobility, parameters);
  assert(actual.report.relative_residual <= 1.1e-6F);
  assert(expected.report.relative_residual <= 1.1e-6F);
  assert(std::abs(actual.report.relative_residual - expected.report.relative_residual) <= 1.0e-6F);
  compare_fields(actual.field, expected.field, "depth-averaged flow");
}

// A 64-by-64 sheet whose mobility spans three decades around square pillars.
// The system is ill-conditioned enough that single-precision conjugate
// gradient drifts well outside this tolerance, so the case holds every backend
// to the double-precision iteration.
void run_ill_conditioned_depth_case(microsimulator::BackendKind backend,
                                    std::uint32_t device_index) {
  constexpr std::uint32_t size = 64;
  auto spec = duct(size, size, 1);
  spec.obstacles.assign(spec.site_count(), 0);
  for (std::uint32_t pillar_x = 8; pillar_x < size; pillar_x += 16) {
    for (std::uint32_t pillar_y = 12; pillar_y + 12 < size; pillar_y += 16) {
      for (std::uint32_t x = pillar_x; x < pillar_x + 4; ++x) {
        for (std::uint32_t y = pillar_y; y < pillar_y + 4; ++y) {
          spec.obstacles[site_index(spec, x, y, 0)] = 1;
        }
      }
    }
  }
  std::vector<float> mobility(spec.site_count());
  std::uint32_t state = 12345;
  for (std::size_t site = 0; site < mobility.size(); ++site) {
    state = (state * 1664525U) + 1013904223U;
    const auto exponent = -3.0F * static_cast<float>(state >> 8) / 16777216.0F;
    mobility[site] = spec.solid_site(site) ? 0.0F : std::pow(10.0F, exponent);
  }
  microsimulator::DepthAveragedFlowParameters parameters;
  parameters.mean_inlet_speed = 1.0F;
  parameters.relative_tolerance = 1.0e-6F;
  microsimulator::Simulation reference;
  microsimulator::Simulation candidate(backend, 0, 0, device_index);
  const auto expected = reference.solve_depth_averaged_flow(spec, mobility, parameters);
  const auto actual = candidate.solve_depth_averaged_flow(spec, mobility, parameters);
  assert(actual.report.relative_residual <= 1.1e-6F);

  const auto tolerance = 1.0e-6F * expected.report.max_speed;
  const auto compare = [&](std::span<const float> actual_faces,
                           std::span<const float> expected_faces, std::string_view component) {
    assert(actual_faces.size() == expected_faces.size());
    for (std::size_t index = 0; index < actual_faces.size(); ++index) {
      if (std::abs(actual_faces[index] - expected_faces[index]) > tolerance) {
        std::cerr << "ill-conditioned depth-averaged flow " << component << " face " << index
                  << ": actual=" << actual_faces[index] << " expected=" << expected_faces[index]
                  << '\n';
        std::abort();
      }
    }
  };
  compare(actual.field.x_faces, expected.field.x_faces, "x");
  compare(actual.field.y_faces, expected.field.y_faces, "y");
  compare(actual.field.z_faces, expected.field.z_faces, "z");
}

void run_resolved_case(microsimulator::BackendKind backend, std::uint32_t device_index) {
  auto spec = duct(6, 7, 2, {0.2F, 0.35F, 0.3F});
  std::vector<float> drag(spec.site_count(), 0.0F);
  for (std::uint32_t x = 0; x < spec.shape.x; ++x) {
    for (std::uint32_t y = 0; y < spec.shape.y; ++y) {
      for (std::uint32_t z = 0; z < spec.shape.z; ++z) {
        if (x >= 3) {
          drag[site_index(spec, x, y, z)] = 12.0F;
        }
      }
    }
  }
  microsimulator::ResolvedFlowParameters parameters;
  parameters.mean_inlet_speed = 2.0F;
  parameters.relative_tolerance = 1.0e-6F;
  parameters.inner_relative_tolerance = 1.0e-6F;
  microsimulator::Simulation reference;
  microsimulator::Simulation candidate(backend, 0, 0, device_index);
  const auto expected = reference.solve_resolved_flow(spec, drag, parameters);
  const auto actual = candidate.solve_resolved_flow(spec, drag, parameters);
  assert(actual.report.divergence_rms < 5.0e-5F);
  assert(actual.report.min_gap_voxels == expected.report.min_gap_voxels);
  compare_fields(actual.field, expected.field, "resolved flow");
}

}  // namespace

int main(int argc, char** argv) {
  const bool resolved_only = argc == 2 && std::string_view(argv[1]) == "resolved";
  microsimulator::test::for_each_backend_device(
      [&](microsimulator::BackendKind backend, std::uint32_t device_index) {
        microsimulator::Simulation probe(backend, 0, 0, device_index);
        assert(probe.supports(microsimulator::BackendFeature::depth_averaged_flow));
        assert(probe.supports(microsimulator::BackendFeature::resolved_flow));
        if (!resolved_only) {
          run_depth_case(backend, device_index);
          run_ill_conditioned_depth_case(backend, device_index);
        }
        if (resolved_only) run_resolved_case(backend, device_index);
      });
}
