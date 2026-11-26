#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "backend_devices.hpp"
#include "microsimulator/simulation.hpp"

namespace {

microsimulator::SignalGridSpec make_spec() {
  microsimulator::SignalGridSpec spec;
  spec.signal_count = 2;
  spec.shape = {.x = 9, .y = 7, .z = 5};
  spec.origin = {-4.0F, -3.0F, -2.0F};
  spec.spacing = {1.0F, 0.75F, 1.25F};
  spec.diffusion = {0.2F, 0.05F};
  spec.advection = {{0.1F, -0.2F, 0.05F}, {-0.05F, 0.03F, -0.08F}};
  spec.x_lower.kind = microsimulator::GridBoundaryKind::periodic;
  spec.x_upper.kind = microsimulator::GridBoundaryKind::periodic;
  spec.z_lower.kind = microsimulator::GridBoundaryKind::fixed;
  spec.z_lower.values = {0.25F, 0.5F};
  spec.z_upper.kind = microsimulator::GridBoundaryKind::fixed;
  spec.z_upper.values = {1.0F, 0.75F};
  microsimulator::SignalGridAffineReaction reaction;
  reaction.source_rates.resize(spec.level_count());
  reaction.loss_rates.resize(spec.level_count());
  for (std::size_t index = 0; index < spec.level_count(); ++index) {
    reaction.source_rates[index] = 0.01F * static_cast<float>(index % 3);
    reaction.loss_rates[index] = 0.005F * static_cast<float>(index % 5);
  }
  spec.reaction = std::move(reaction);
  return spec;
}

// The plain grid with the sites `solid` names made obstacles.
template <typename Solid>
microsimulator::SignalGridSpec make_spec_with_obstacles(const Solid& solid) {
  auto spec = make_spec();
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

microsimulator::SignalGridSpec make_masked_spec() {
  return make_spec_with_obstacles([](std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    const auto interior_block = x >= 3 && x <= 5 && y >= 2 && y <= 4 && z == 2;
    const auto periodic_edge = x == 0 && y == 1 && z == 1;
    return interior_block || periodic_edge;
  });
}

microsimulator::SignalGridSpec make_velocity_field_spec() {
  auto spec = make_spec();
  spec.advection = {{0.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.0F}};
  microsimulator::SignalGridVelocityField field;
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

// Every face is a fixed reservoir, three of them with patches, and a velocity
// field crosses all six. Each of the three patched faces indexes its sites by
// a different pair of axes.
microsimulator::SignalGridSpec make_patched_spec() {
  auto spec = make_spec();
  spec.advection = {{0.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.0F}};
  for (auto* boundary : {&spec.x_lower, &spec.x_upper, &spec.y_lower, &spec.y_upper}) {
    boundary->kind = microsimulator::GridBoundaryKind::fixed;
    boundary->values = {0.4F, 0.6F};
  }
  spec.x_lower.patches = {
      {.region = {.begin = {1, 0}, .end = {4, 3}}, .values = {1.5F, 0.1F}},
      {.region = {.begin = {3, 2}, .end = {7, 5}}, .values = {0.0F, 2.0F}},
  };
  spec.y_upper.patches = {
      {.region = {.begin = {0, 1}, .end = {5, 4}}, .values = {0.9F, 0.2F}},
  };
  spec.z_lower.patches = {
      {.region = {.begin = {2, 0}, .end = {9, 2}}, .values = {1.2F, 0.3F}},
      {.region = {.begin = {6, 1}, .end = {8, 7}}, .values = {0.1F, 1.1F}},
  };
  microsimulator::SignalGridVelocityField field;
  field.x_faces.assign(spec.x_face_count(), 0.06F);
  field.y_faces.assign(spec.y_face_count(), -0.04F);
  field.z_faces.assign(spec.z_face_count(), 0.05F);
  spec.velocity_field = std::move(field);
  return spec;
}

std::vector<float> make_levels(const microsimulator::SignalGridSpec& spec) {
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

void assert_matches(const microsimulator::Simulation& actual,
                    const microsimulator::Simulation& expected) {
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

enum class SpecKind { plain, masked, velocity_field, patched };

void run_case(microsimulator::SignalIntegrationKind integration, float dt,
              SpecKind kind = SpecKind::plain,
              microsimulator::SignalAdvectionKind scheme =
                  microsimulator::SignalAdvectionKind::upwind) {
  auto spec = kind == SpecKind::masked           ? make_masked_spec()
              : kind == SpecKind::velocity_field ? make_velocity_field_spec()
              : kind == SpecKind::patched        ? make_patched_spec()
                                                 : make_spec();
  spec.integration = integration;
  spec.advection_scheme = scheme;
  const auto levels = make_levels(spec);
  microsimulator::Simulation reference;
  reference.configure_signal_grid(spec, levels);
  reference.step(dt);

  microsimulator::test::for_each_backend_device(
      [&](microsimulator::BackendKind backend, std::uint32_t device_index) {
        microsimulator::Simulation candidate(backend, 0, 0, device_index);
        candidate.configure_signal_grid(spec, levels);
        if (!candidate.supports(microsimulator::BackendFeature::signals)) {
          return;
        }
        candidate.step(dt);
        assert_matches(candidate, reference);
        assert(candidate.last_signal_solve_report().has_value());
        assert(candidate.last_signal_solve_report()->converged);
      });
}

// Steps a simulation and checks the step against one taken by a simulation
// that has held no other grid.
void assert_step_matches_fresh(microsimulator::Simulation& candidate,
                               const microsimulator::SignalGridSpec& spec, float dt) {
  microsimulator::Simulation reference;
  reference.configure_signal_grid(spec, candidate.signal_levels());
  reference.step(dt);
  candidate.step(dt);
  assert_matches(candidate, reference);
}

// A backend keeps what it derives from a grid between steps. Each change to
// the grid must reach the very next step: its velocity field, its reaction
// field, its boundary and patch values, and its levels.
void run_changing_grid_case(microsimulator::SignalIntegrationKind integration, float dt,
                            microsimulator::SignalAdvectionKind scheme) {
  microsimulator::test::for_each_backend_device(
      [&](microsimulator::BackendKind backend, std::uint32_t device_index) {
        microsimulator::Simulation candidate(backend, 0, 0, device_index);
        if (!candidate.supports(microsimulator::BackendFeature::signals)) {
          return;
        }
        auto spec = make_patched_spec();
        spec.integration = integration;
        spec.advection_scheme = scheme;
        candidate.configure_signal_grid(spec, make_levels(spec));
        assert_step_matches_fresh(candidate, spec, dt);
        assert_step_matches_fresh(candidate, spec, dt);

        auto field = *spec.velocity_field;
        std::ranges::fill(field.x_faces, -0.03F);
        std::ranges::fill(field.z_faces, 0.09F);
        spec.velocity_field = field;
        candidate.set_velocity_field(field);
        assert_step_matches_fresh(candidate, spec, dt);

        auto reaction = *spec.reaction;
        for (std::size_t index = 0; index < reaction.loss_rates.size(); ++index) {
          reaction.source_rates[index] = 0.02F * static_cast<float>(index % 4);
          reaction.loss_rates[index] = 0.01F * static_cast<float>(index % 3);
        }
        spec.reaction = reaction;
        candidate.set_signal_reaction(reaction);
        assert_step_matches_fresh(candidate, spec, dt);

        spec.x_upper.values = {2.5F, 0.0F};
        candidate.set_signal_boundary_values(microsimulator::GridFace::x_upper,
                                             spec.x_upper.values);
        assert_step_matches_fresh(candidate, spec, dt);

        spec.z_lower.patches[1].values = {3.0F, 0.05F};
        candidate.set_signal_boundary_patch_values(microsimulator::GridFace::z_lower, 1,
                                                   spec.z_lower.patches[1].values);
        assert_step_matches_fresh(candidate, spec, dt);

        spec.reaction.reset();
        candidate.set_signal_reaction(std::nullopt);
        assert_step_matches_fresh(candidate, spec, dt);

        // A grid with obstacles, whose reaction field and levels change.
        microsimulator::Simulation masked(backend, 0, 0, device_index);
        auto masked_spec = make_masked_spec();
        masked_spec.integration = integration;
        masked_spec.advection_scheme = scheme;
        masked.configure_signal_grid(masked_spec, make_levels(masked_spec));
        assert_step_matches_fresh(masked, masked_spec, dt);
        masked_spec.reaction.reset();
        masked.set_signal_reaction(std::nullopt);
        assert_step_matches_fresh(masked, masked_spec, dt);
        auto levels = make_levels(masked_spec);
        std::ranges::reverse(levels);
        for (std::size_t index = 0; index < levels.size(); ++index) {
          if (masked_spec.solid_site(index % masked_spec.site_count())) {
            levels[index] = 0.0F;
          }
        }
        masked.set_signal_levels(levels);
        assert_step_matches_fresh(masked, masked_spec, dt);
      });
}

}  // namespace

int main() {
  run_case(microsimulator::SignalIntegrationKind::forward_euler, 0.02F);
  run_case(microsimulator::SignalIntegrationKind::crank_nicolson, 0.5F);
  run_case(microsimulator::SignalIntegrationKind::backward_euler, 0.5F);
  run_case(microsimulator::SignalIntegrationKind::forward_euler, 0.02F, SpecKind::masked);
  run_case(microsimulator::SignalIntegrationKind::crank_nicolson, 0.5F, SpecKind::masked);
  run_case(microsimulator::SignalIntegrationKind::forward_euler, 0.02F, SpecKind::velocity_field);
  run_case(microsimulator::SignalIntegrationKind::crank_nicolson, 0.5F, SpecKind::velocity_field);
  run_case(microsimulator::SignalIntegrationKind::forward_euler, 0.02F, SpecKind::patched);
  run_case(microsimulator::SignalIntegrationKind::crank_nicolson, 0.5F, SpecKind::patched);
  run_case(microsimulator::SignalIntegrationKind::backward_euler, 0.5F, SpecKind::patched);
  // The van Leer scheme, on every kind of grid: constant and face velocities,
  // walls, periodic wrap, and patched reservoirs.
  const auto van_leer = microsimulator::SignalAdvectionKind::van_leer;
  for (const auto kind :
       {SpecKind::plain, SpecKind::masked, SpecKind::velocity_field, SpecKind::patched}) {
    run_case(microsimulator::SignalIntegrationKind::forward_euler, 0.02F, kind, van_leer);
    run_case(microsimulator::SignalIntegrationKind::crank_nicolson, 0.5F, kind, van_leer);
    run_case(microsimulator::SignalIntegrationKind::backward_euler, 0.5F, kind, van_leer);
  }
  for (const auto scheme : {microsimulator::SignalAdvectionKind::upwind, van_leer}) {
    run_changing_grid_case(microsimulator::SignalIntegrationKind::forward_euler, 0.02F, scheme);
    run_changing_grid_case(microsimulator::SignalIntegrationKind::crank_nicolson, 0.5F, scheme);
    run_changing_grid_case(microsimulator::SignalIntegrationKind::backward_euler, 0.5F, scheme);
  }
}
