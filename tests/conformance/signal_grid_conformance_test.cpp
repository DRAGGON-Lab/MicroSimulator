#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "backend_devices.hpp"
#include "cm/simulation.hpp"

namespace {

cm::SignalGridSpec make_spec() {
  cm::SignalGridSpec spec;
  spec.signal_count = 2;
  spec.shape = {.x = 9, .y = 7, .z = 5};
  spec.origin = {-4.0F, -3.0F, -2.0F};
  spec.spacing = {1.0F, 0.75F, 1.25F};
  spec.diffusion = {0.2F, 0.05F};
  spec.advection = {{0.1F, -0.2F, 0.05F}, {-0.05F, 0.03F, -0.08F}};
  spec.x_lower.kind = cm::GridBoundaryKind::periodic;
  spec.x_upper.kind = cm::GridBoundaryKind::periodic;
  spec.z_lower.kind = cm::GridBoundaryKind::fixed;
  spec.z_lower.values = {0.25F, 0.5F};
  spec.z_upper.kind = cm::GridBoundaryKind::fixed;
  spec.z_upper.values = {1.0F, 0.75F};
  cm::SignalGridAffineReaction reaction;
  reaction.source_rates.resize(spec.level_count());
  reaction.loss_rates.resize(spec.level_count());
  for (std::size_t index = 0; index < spec.level_count(); ++index) {
    reaction.source_rates[index] = 0.01F * static_cast<float>(index % 3);
    reaction.loss_rates[index] = 0.005F * static_cast<float>(index % 5);
  }
  spec.reaction = std::move(reaction);
  return spec;
}

cm::SignalGridSpec make_masked_spec() {
  auto spec = make_spec();
  const auto solid = [&](std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    const auto interior_block = x >= 3 && x <= 5 && y >= 2 && y <= 4 && z == 2;
    const auto periodic_edge = x == 0 && y == 1 && z == 1;
    return interior_block || periodic_edge;
  };
  std::vector<std::uint8_t> obstacles(spec.site_count(), 0);
  for (std::uint32_t x = 0; x < spec.shape.x; ++x) {
    for (std::uint32_t y = 0; y < spec.shape.y; ++y) {
      for (std::uint32_t z = 0; z < spec.shape.z; ++z) {
        const auto site = (static_cast<std::size_t>(x) * spec.shape.y * spec.shape.z) +
                          (static_cast<std::size_t>(y) * spec.shape.z) + z;
        obstacles[site] = solid(x, y, z) ? 1 : 0;
      }
    }
  }
  for (std::size_t signal = 0; signal < spec.signal_count; ++signal) {
    for (std::size_t site = 0; site < spec.site_count(); ++site) {
      if (obstacles[site] != 0) {
        spec.reaction->source_rates[(signal * spec.site_count()) + site] = 0.0F;
        spec.reaction->loss_rates[(signal * spec.site_count()) + site] = 0.0F;
      }
    }
  }
  spec.obstacles = std::move(obstacles);
  return spec;
}

cm::SignalGridSpec make_velocity_field_spec() {
  auto spec = make_spec();
  spec.advection = {{0.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.0F}};
  cm::SignalGridVelocityField field;
  field.x_faces.resize(spec.x_face_count(), 0.0F);
  field.y_faces.resize(spec.y_face_count(), 0.0F);
  field.z_faces.resize(spec.z_face_count(), 0.0F);
  for (std::uint32_t fx = 0; fx <= spec.shape.x; ++fx) {
    for (std::uint32_t y = 0; y < spec.shape.y; ++y) {
      for (std::uint32_t z = 0; z < spec.shape.z; ++z) {
        const auto index = (static_cast<std::size_t>(fx) * spec.shape.y * spec.shape.z) +
                           (static_cast<std::size_t>(y) * spec.shape.z) + z;
        field.x_faces[index] =
            0.05F + (0.01F * static_cast<float>(y)) - (0.008F * static_cast<float>(z));
      }
    }
  }
  spec.velocity_field = std::move(field);
  return spec;
}

std::vector<float> make_levels(const cm::SignalGridSpec& spec) {
  std::vector<float> levels(spec.level_count());
  for (std::size_t index = 0; index < levels.size(); ++index) {
    levels[index] = spec.solid_site(index % spec.site_count())
                        ? 0.0F
                        : 0.5F + (0.001F * static_cast<float>((index * 37) % 211));
  }
  return levels;
}

bool close(float actual, float expected) {
  constexpr float tolerance = 5.0e-6F;
  return std::abs(actual - expected) <=
         tolerance + (tolerance * std::max(std::abs(actual), std::abs(expected)));
}

void assert_matches(const cm::Simulation& actual, const cm::Simulation& expected) {
  assert(actual.signal_count() == expected.signal_count());
  assert(actual.signal_levels().size() == expected.signal_levels().size());
  const auto actual_levels = actual.signal_levels();
  const auto expected_levels = expected.signal_levels();
  for (std::size_t index = 0; index < actual_levels.size(); ++index) {
    assert(close(actual_levels[index], expected_levels[index]));
  }
  const auto actual_sample = actual.sample_signals({-0.25F, -0.5F, 1.1F});
  const auto expected_sample = expected.sample_signals({-0.25F, -0.5F, 1.1F});
  for (std::size_t signal = 0; signal < actual_sample.size(); ++signal) {
    assert(close(actual_sample[signal], expected_sample[signal]));
  }
}

enum class SpecKind { plain, masked, velocity_field };

void run_case(cm::SignalIntegrationKind integration, float dt, SpecKind kind = SpecKind::plain) {
  auto spec = kind == SpecKind::masked         ? make_masked_spec()
              : kind == SpecKind::velocity_field ? make_velocity_field_spec()
                                                 : make_spec();
  spec.integration = integration;
  const auto levels = make_levels(spec);
  cm::Simulation reference;
  reference.configure_signal_grid(spec, levels);
  reference.step(dt);

  cm::test::for_each_backend_device([&](cm::BackendKind backend, std::uint32_t device_index) {
    cm::Simulation candidate(backend, 0, 0, device_index);
    candidate.configure_signal_grid(spec, levels);
    if (!candidate.supports(cm::BackendFeature::signals)) {
      return;
    }
    candidate.step(dt);
    assert_matches(candidate, reference);
    assert(candidate.last_signal_solve_report().has_value());
    assert(candidate.last_signal_solve_report()->converged);
  });
}

}  // namespace

int main() {
  run_case(cm::SignalIntegrationKind::forward_euler, 0.02F);
  run_case(cm::SignalIntegrationKind::crank_nicolson, 0.5F);
  run_case(cm::SignalIntegrationKind::forward_euler, 0.02F, SpecKind::masked);
  run_case(cm::SignalIntegrationKind::crank_nicolson, 0.5F, SpecKind::masked);
  run_case(cm::SignalIntegrationKind::forward_euler, 0.02F, SpecKind::velocity_field);
  run_case(cm::SignalIntegrationKind::crank_nicolson, 0.5F, SpecKind::velocity_field);
}
