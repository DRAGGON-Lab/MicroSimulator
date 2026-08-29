#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "cm/simulation.hpp"

namespace {

cm::SignalGridSpec duct(std::uint32_t nx, std::uint32_t ny, std::uint32_t nz,
                        cm::Vec3 spacing = {1.0F, 1.0F, 1.0F}) {
  cm::SignalGridSpec spec;
  spec.signal_count = 1;
  spec.shape = {.x = nx, .y = ny, .z = nz};
  spec.spacing = spacing;
  spec.diffusion = {0.0F};
  spec.advection = {{0.0F, 0.0F, 0.0F}};
  spec.y_lower = {.kind = cm::GridBoundaryKind::fixed, .values = {0.0F}};
  spec.y_upper = {.kind = cm::GridBoundaryKind::fixed, .values = {0.0F}};
  return spec;
}

std::size_t y_face(const cm::SignalGridSpec& spec, std::uint32_t x, std::uint32_t y,
                   std::uint32_t z) {
  return (static_cast<std::size_t>(x) * (spec.shape.y + 1) + y) * spec.shape.z + z;
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
    const auto spec = duct(4, 8, 3);
    cm::Simulation simulation;
    const auto result = simulation.solve_depth_averaged_flow(
        spec, {}, {.mean_inlet_speed = 5.0F, .axis = cm::FlowAxis::y});
    assert(simulation.supports(cm::BackendFeature::depth_averaged_flow));
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
    const auto result = cm::solve_depth_averaged_flow_cpu(
        spec, mobility, {.mean_inlet_speed = 4.0F, .axis = cm::FlowAxis::y});
    const auto slow = result.field.y_faces[y_face(spec, 0, 3, 0)];
    const auto fast = result.field.y_faces[y_face(spec, 1, 3, 0)];
    assert(std::abs((fast / slow) - 3.0F) <= 2.0e-5F);
  }

  {
    constexpr std::uint32_t nx = 8;
    const auto spec = duct(nx, 6, 1, {1.0F / static_cast<float>(nx), 0.25F, 1.0F});
    cm::Simulation simulation;
    const auto result = simulation.solve_resolved_flow(
        spec, {}, {.mean_inlet_speed = 1.0F, .axis = cm::FlowAxis::y});
    assert(simulation.supports(cm::BackendFeature::resolved_flow));
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

  {
    auto spec = duct(3, 4, 1);
    spec.obstacles.assign(spec.site_count(), 0);
    for (std::uint32_t x = 0; x < spec.shape.x; ++x) {
      spec.obstacles[(static_cast<std::size_t>(x) * spec.shape.y) + 2] = 1;
    }
    assert_throws<std::runtime_error>(
        [&] { static_cast<void>(cm::solve_depth_averaged_flow_cpu(spec, {})); });
    assert_throws<std::runtime_error>(
        [&] { static_cast<void>(cm::solve_resolved_flow_cpu(spec, {})); });
  }
}
