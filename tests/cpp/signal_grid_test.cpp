#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "cm/signals.hpp"
#include "cm/simulation.hpp"

namespace {

cm::SignalGridSpec line_spec(std::uint32_t length) {
  cm::SignalGridSpec spec;
  spec.signal_count = 1;
  spec.shape = {.x = length, .y = 1, .z = 1};
  spec.diffusion = {1.0F};
  spec.advection = {{0.0F, 0.0F, 0.0F}};
  return spec;
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

void assert_close(float actual, float expected) { assert(std::abs(actual - expected) <= 1.0e-6F); }

}  // namespace

int main() {
  {
    auto spec = line_spec(1);
    spec.diffusion = {0.0F};
    spec.reaction = cm::SignalGridAffineReaction{
        .source_rates = {2.0F},
        .loss_rates = {0.5F},
    };
    cm::SignalGrid grid(spec, {1.0F});
    static_cast<void>(cm::advance_signal_grid_cpu(grid, 0.5F));
    assert_close(grid.levels()[0], 1.75F);

    auto unstable = spec;
    unstable.reaction->loss_rates = {2.0F};
    cm::SignalGrid unstable_grid(unstable, {1.0F});
    assert_throws<std::invalid_argument>(
        [&] { static_cast<void>(cm::advance_signal_grid_cpu(unstable_grid, 0.51F)); });
  }

  {
    auto spec = line_spec(1);
    spec.diffusion = {0.0F};
    spec.integration = cm::SignalIntegrationKind::crank_nicolson;
    spec.reaction = cm::SignalGridAffineReaction{
        .source_rates = {2.0F},
        .loss_rates = {0.5F},
    };
    cm::SignalGrid grid(spec, {1.0F});
    const auto report = cm::advance_signal_grid_cpu(grid, 1.0F);
    assert(report.converged);
    assert_close(grid.levels()[0], 2.2F);
  }

  {
    auto invalid = line_spec(2);
    invalid.reaction = cm::SignalGridAffineReaction{
        .source_rates = {1.0F},
        .loss_rates = {0.0F, 0.0F},
    };
    assert_throws<std::invalid_argument>([&] { invalid.validate(); });
    invalid.reaction->source_rates = {1.0F, -1.0F};
    assert_throws<std::invalid_argument>([&] { invalid.validate(); });
    invalid.reaction->source_rates = {1.0F, 1.0F};
    invalid.reaction->loss_rates = {0.0F, -1.0F};
    assert_throws<std::invalid_argument>([&] { invalid.validate(); });
  }

  {
    cm::SignalGrid grid(line_spec(3), {0.0F, 1.0F, 0.0F});
    static_cast<void>(cm::advance_signal_grid_cpu(grid, 0.25F));
    const auto levels = grid.levels();
    assert_close(levels[0], 0.25F);
    assert_close(levels[1], 0.5F);
    assert_close(levels[2], 0.25F);
    assert_close(levels[0] + levels[1] + levels[2], 1.0F);

    const std::vector<float> before(levels.begin(), levels.end());
    assert_throws<std::invalid_argument>(
        [&] { static_cast<void>(cm::advance_signal_grid_cpu(grid, 0.51F)); });
    assert(std::vector<float>(grid.levels().begin(), grid.levels().end()) == before);
  }

  {
    auto spec = line_spec(3);
    spec.integration = cm::SignalIntegrationKind::crank_nicolson;
    spec.solver.absolute_tolerance = 1.0e-7F;
    spec.solver.relative_tolerance = 1.0e-6F;
    cm::SignalGrid grid(spec, {0.0F, 1.0F, 0.0F});
    const auto report = cm::advance_signal_grid_cpu(grid, 1.0F);
    assert(report.converged);
    assert(report.iterations > 0);
    assert(report.residual_rms <= 2.0e-6F);
    assert_close(grid.levels()[0], 0.4F);
    assert_close(grid.levels()[1], 0.2F);
    assert_close(grid.levels()[2], 0.4F);
  }

  {
    auto spec = line_spec(9);
    spec.integration = cm::SignalIntegrationKind::crank_nicolson;
    spec.solver.max_iterations = 1;
    spec.solver.absolute_tolerance = 1.0e-12F;
    spec.solver.relative_tolerance = 0.0F;
    cm::SignalGrid grid(spec, {0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F});
    const auto result = cm::signal_grid_crank_nicolson_candidate(grid, 2.0F);
    assert(!result.report.converged);
    assert(result.report.iterations == 1);
    assert_throws<std::runtime_error>(
        [&] { static_cast<void>(cm::advance_signal_grid_cpu(grid, 2.0F)); });
  }

  {
    auto spec = line_spec(2);
    spec.x_lower.kind = cm::GridBoundaryKind::fixed;
    spec.x_lower.values = {2.0F};
    cm::SignalGrid grid(spec, {0.0F, 0.0F});
    static_cast<void>(cm::advance_signal_grid_cpu(grid, 0.25F));
    assert_close(grid.levels()[0], 0.5F);
    assert_close(grid.levels()[1], 0.0F);
  }

  {
    auto spec = line_spec(4);
    spec.diffusion = {0.0F};
    spec.advection = {{1.0F, 0.0F, 0.0F}};
    spec.x_lower.kind = cm::GridBoundaryKind::periodic;
    spec.x_upper.kind = cm::GridBoundaryKind::periodic;
    cm::SignalGrid grid(spec, {1.0F, 0.0F, 0.0F, 0.0F});
    static_cast<void>(cm::advance_signal_grid_cpu(grid, 0.5F));
    assert_close(grid.levels()[0], 0.5F);
    assert_close(grid.levels()[1], 0.5F);
    assert_close(grid.levels()[2], 0.0F);
    assert_close(grid.levels()[3], 0.0F);
  }

  {
    auto spec = line_spec(4);
    spec.diffusion = {0.0F};
    spec.advection = {{1.0F, 0.0F, 0.0F}};
    cm::SignalGrid grid(spec, {0.0F, 0.0F, 0.0F, 1.0F});
    static_cast<void>(cm::advance_signal_grid_cpu(grid, 0.5F));
    assert_close(grid.levels()[0], 0.0F);
    assert_close(grid.levels()[1], 0.0F);
    assert_close(grid.levels()[2], 0.0F);
    assert_close(grid.levels()[3], 1.0F);
  }

  {
    cm::SignalGridSpec spec;
    spec.signal_count = 1;
    spec.shape = {.x = 2, .y = 2, .z = 2};
    spec.diffusion = {0.0F};
    spec.advection = {{0.0F, 0.0F, 0.0F}};
    cm::SignalGrid grid(spec, {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F});
    assert_close(grid.sample({0.5F, 0.5F, 0.5F})[0], 3.5F);
    assert_close(grid.sample({1.0F, 1.0F, 1.0F})[0], 7.0F);
    assert_throws<std::out_of_range>([&] { static_cast<void>(grid.sample({1.01F, 0.0F, 0.0F})); });

    cm::SignalGrid reduced(line_spec(2), {2.0F, 4.0F});
    assert_close(reduced.sample({0.5F, 99.0F, -37.0F})[0], 3.0F);
  }

  {
    auto invalid = line_spec(3);
    invalid.x_lower.kind = cm::GridBoundaryKind::periodic;
    assert_throws<std::invalid_argument>([&] { invalid.validate(); });
    invalid.x_upper.kind = cm::GridBoundaryKind::periodic;
    invalid.diffusion = {-1.0F};
    assert_throws<std::invalid_argument>([&] { invalid.validate(); });
  }

  {
    cm::Simulation simulation;
    const auto spec = line_spec(3);
    simulation.configure_signal_grid(spec, {0.0F, 1.0F, 0.0F});
    assert(simulation.has_signal_grid());
    assert(simulation.signal_count() == 1);
    assert(simulation.supports(cm::BackendFeature::signals));
    simulation.step(0.25F);
    assert_close(simulation.signal_levels()[0], 0.25F);
    assert(simulation.time() == 0.25);
    const auto checkpoint = simulation.checkpoint();
    assert(checkpoint.signal_grid.has_value());
    assert(checkpoint.signal_grid->levels == simulation.signal_levels());

    cm::Simulation restored(cm::BackendKind::cpu, checkpoint);
    assert(restored.signal_levels() == simulation.signal_levels());
    assert_close(restored.sample_signals({1.0F, 0.0F, 0.0F})[0], 0.5F);

    cm::CellInit cell;
    restored.add_cell(cell);
    assert_throws<std::logic_error>([&] { restored.configure_signal_grid(spec); });
  }

  {
    auto spec = line_spec(3);
    spec.advection = {{0.5F, 0.0F, 0.0F}};
    spec.obstacles = {0, 1, 0};
    cm::SignalGrid grid(spec, {1.0F, 0.0F, 0.5F});
    static_cast<void>(cm::advance_signal_grid_cpu(grid, 0.25F));
    assert_close(grid.levels()[0], 1.0F);
    assert_close(grid.levels()[1], 0.0F);
    assert_close(grid.levels()[2], 0.5F);
  }

  {
    auto spec = line_spec(4);
    spec.obstacles = {0, 0, 1, 0};
    cm::SignalGrid grid(spec, {2.0F, 0.0F, 0.0F, 5.0F});
    static_cast<void>(cm::advance_signal_grid_cpu(grid, 0.25F));
    assert_close(grid.levels()[0], 1.5F);
    assert_close(grid.levels()[1], 0.5F);
    assert_close(grid.levels()[2], 0.0F);
    assert_close(grid.levels()[3], 5.0F);
    assert_close(grid.levels()[0] + grid.levels()[1], 2.0F);
  }

  {
    auto spec = line_spec(4);
    spec.integration = cm::SignalIntegrationKind::crank_nicolson;
    spec.obstacles = {0, 0, 1, 0};
    cm::SignalGrid grid(spec, {2.0F, 0.0F, 0.0F, 5.0F});
    const auto report = cm::advance_signal_grid_cpu(grid, 1.0F);
    assert(report.converged);
    assert(std::abs(grid.levels()[0] + grid.levels()[1] - 2.0F) <= 1.0e-4F);
    assert(grid.levels()[2] == 0.0F);
    assert(std::abs(grid.levels()[3] - 5.0F) <= 1.0e-4F);
  }

  {
    auto spec = line_spec(2);
    spec.obstacles = {0, 1};
    cm::SignalGrid grid(spec, {3.0F, 0.0F});
    assert_close(grid.sample({0.5F, 0.0F, 0.0F})[0], 3.0F);
    assert_throws<std::invalid_argument>(
        [&] { static_cast<void>(grid.sample({1.0F, 0.0F, 0.0F})); });
  }

  {
    auto invalid = line_spec(2);
    invalid.obstacles = {1};
    assert_throws<std::invalid_argument>([&] { invalid.validate(); });
    invalid.obstacles = {0, 2};
    assert_throws<std::invalid_argument>([&] { invalid.validate(); });
    invalid.obstacles = {0, 1};
    invalid.validate();
    assert_throws<std::invalid_argument>(
        [&] { cm::SignalGrid grid(invalid, {0.0F, 1.0F}); });
    invalid.reaction = cm::SignalGridAffineReaction{
        .source_rates = {0.0F, 1.0F},
        .loss_rates = {0.0F, 0.0F},
    };
    assert_throws<std::invalid_argument>([&] { invalid.validate(); });
  }

  {
    auto spec = line_spec(3);
    spec.diffusion = {0.0F};
    spec.x_lower.kind = cm::GridBoundaryKind::fixed;
    spec.x_lower.values = {2.0F};
    spec.x_upper.kind = cm::GridBoundaryKind::fixed;
    spec.x_upper.values = {0.0F};
    spec.velocity_field = cm::SignalGridVelocityField{
        .x_faces = {1.0F, 1.0F, 1.0F, 1.0F},
        .y_faces = std::vector<float>(6, 0.0F),
        .z_faces = std::vector<float>(6, 0.0F),
    };
    cm::SignalGrid grid(spec, {0.0F, 0.0F, 0.0F});
    static_cast<void>(cm::advance_signal_grid_cpu(grid, 0.5F));
    assert_close(grid.levels()[0], 1.0F);
    assert_close(grid.levels()[1], 0.0F);
    static_cast<void>(cm::advance_signal_grid_cpu(grid, 0.5F));
    assert_close(grid.levels()[0], 1.5F);
    assert_close(grid.levels()[1], 0.5F);
    assert_close(grid.levels()[2], 0.0F);
  }

  {
    cm::SignalGridSpec spec;
    spec.signal_count = 1;
    spec.shape = {.x = 3, .y = 3, .z = 1};
    spec.diffusion = {0.1F};
    spec.advection = {{0.0F, 0.0F, 0.0F}};
    spec.y_lower.kind = cm::GridBoundaryKind::periodic;
    spec.y_upper.kind = cm::GridBoundaryKind::periodic;
    std::vector<float> y_faces(12, 0.0F);
    for (std::uint32_t x = 0; x < 3; ++x) {
      for (std::uint32_t fy = 0; fy < 4; ++fy) {
        y_faces[(x * 4) + fy] = 0.5F * static_cast<float>(x + 1);
      }
    }
    spec.velocity_field = cm::SignalGridVelocityField{
        .x_faces = std::vector<float>(12, 0.0F),
        .y_faces = y_faces,
        .z_faces = std::vector<float>(18, 0.0F),
    };
    std::vector<float> levels(9);
    float total_before = 0.0F;
    for (std::size_t index = 0; index < levels.size(); ++index) {
      levels[index] = 0.5F + 0.1F * static_cast<float>(index);
      total_before += levels[index];
    }
    cm::SignalGrid grid(spec, levels);
    static_cast<void>(cm::advance_signal_grid_cpu(grid, 0.2F));
    float total_after = 0.0F;
    for (const auto level : grid.levels()) {
      total_after += level;
    }
    assert(std::abs(total_after - total_before) <= 1.0e-4F);
  }

  {
    auto spec = line_spec(3);
    spec.integration = cm::SignalIntegrationKind::crank_nicolson;
    spec.x_lower.kind = cm::GridBoundaryKind::fixed;
    spec.x_lower.values = {2.0F};
    spec.x_upper.kind = cm::GridBoundaryKind::fixed;
    spec.x_upper.values = {0.0F};
    spec.velocity_field = cm::SignalGridVelocityField{
        .x_faces = {1.0F, 1.0F, 1.0F, 1.0F},
        .y_faces = std::vector<float>(6, 0.0F),
        .z_faces = std::vector<float>(6, 0.0F),
    };
    cm::SignalGrid grid(spec, {0.0F, 0.0F, 0.0F});
    const auto report = cm::advance_signal_grid_cpu(grid, 1.0F);
    assert(report.converged);
    assert(grid.levels()[0] > grid.levels()[1]);
    assert(grid.levels()[1] > grid.levels()[2]);
  }

  {
    auto invalid = line_spec(3);
    invalid.velocity_field = cm::SignalGridVelocityField{
        .x_faces = {1.0F},
        .y_faces = std::vector<float>(6, 0.0F),
        .z_faces = std::vector<float>(6, 0.0F),
    };
    assert_throws<std::invalid_argument>([&] { invalid.validate(); });

    invalid.velocity_field = cm::SignalGridVelocityField{
        .x_faces = {1.0F, 1.0F, 1.0F, 1.0F},
        .y_faces = std::vector<float>(6, 0.0F),
        .z_faces = std::vector<float>(6, 0.0F),
    };
    assert_throws<std::invalid_argument>([&] { invalid.validate(); });

    invalid.x_lower.kind = cm::GridBoundaryKind::fixed;
    invalid.x_lower.values = {0.0F};
    invalid.x_upper.kind = cm::GridBoundaryKind::fixed;
    invalid.x_upper.values = {0.0F};
    invalid.validate();

    invalid.advection = {{0.5F, 0.0F, 0.0F}};
    assert_throws<std::invalid_argument>([&] { invalid.validate(); });
  }

  {
    auto spec = line_spec(3);
    spec.diffusion = {0.0F};
    spec.x_lower.kind = cm::GridBoundaryKind::fixed;
    spec.x_lower.values = {0.0F};
    spec.x_upper.kind = cm::GridBoundaryKind::fixed;
    spec.x_upper.values = {0.0F};

    cm::SignalGrid grid(spec);
    grid.set_velocity_field(cm::SignalGridVelocityField{
        .x_faces = std::vector<float>(4, 2.0F),
        .y_faces = std::vector<float>(6, 0.0F),
        .z_faces = std::vector<float>(6, 0.0F),
    });
    assert_throws<std::invalid_argument>([&grid] {
      grid.set_velocity_field(cm::SignalGridVelocityField{
          .x_faces = std::vector<float>(5, 0.0F),
          .y_faces = std::vector<float>(6, 0.0F),
          .z_faces = std::vector<float>(6, 0.0F),
      });
    });
    grid.set_velocity_field(std::nullopt);

    cm::Simulation simulation;
    simulation.configure_signal_grid(spec);
    simulation.set_velocity_field(cm::SignalGridVelocityField{
        .x_faces = std::vector<float>(4, 2.0F),
        .y_faces = std::vector<float>(6, 0.0F),
        .z_faces = std::vector<float>(6, 0.0F),
    });
    simulation.set_velocity_field(std::nullopt);

    cm::Simulation bare;
    assert_throws<std::logic_error>([&bare] { bare.set_velocity_field(std::nullopt); });
  }
}
