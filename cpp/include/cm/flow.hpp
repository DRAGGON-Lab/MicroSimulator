#pragma once

#include <cstdint>
#include <span>

#include "cm/signals.hpp"

namespace cm {

enum class FlowAxis : std::uint8_t {
  x,
  y,
  z,
};

struct DepthAveragedFlowParameters {
  float mean_inlet_speed{1.0F};
  FlowAxis axis{FlowAxis::y};
  float relative_tolerance{1.0e-6F};
  std::uint32_t max_iterations{50'000};

  void validate() const;
};

struct DepthAveragedFlowReport {
  std::uint32_t iterations{0};
  float relative_residual{0.0F};
  float mean_inlet_speed{0.0F};
  float max_speed{0.0F};
};

struct DepthAveragedFlowResult {
  SignalGridVelocityField field;
  DepthAveragedFlowReport report;
};

struct ResolvedFlowParameters {
  float mean_inlet_speed{1.0F};
  FlowAxis axis{FlowAxis::y};
  float relative_tolerance{1.0e-6F};
  std::uint32_t max_outer_iterations{500};
  float inner_relative_tolerance{1.0e-6F};
  std::uint32_t max_inner_iterations{50'000};

  void validate() const;
};

struct ResolvedFlowReport {
  std::uint32_t outer_iterations{0};
  std::uint64_t inner_iterations{0};
  float divergence_rms{0.0F};
  float mean_inlet_speed{0.0F};
  float max_speed{0.0F};
  std::uint32_t min_gap_voxels{0};
};

struct ResolvedFlowResult {
  SignalGridVelocityField field;
  ResolvedFlowReport report;
};

void validate_flow_grid(const SignalGridSpec& spec, FlowAxis axis);

[[nodiscard]] DepthAveragedFlowResult solve_depth_averaged_flow_cpu(
    const SignalGridSpec& spec, std::span<const float> mobility,
    const DepthAveragedFlowParameters& parameters = DepthAveragedFlowParameters{});

[[nodiscard]] ResolvedFlowResult solve_resolved_flow_cpu(
    const SignalGridSpec& spec, std::span<const float> drag,
    const ResolvedFlowParameters& parameters = ResolvedFlowParameters{});

}  // namespace cm
