#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "microsimulator/signals.hpp"

namespace microsimulator {

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

  // In-plane body force per column, in pressure-gradient units (x-major, y-minor).
  std::vector<std::array<float, 2>> body_force{};

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
  float relative_residual{0.0F};
  float momentum_relative_residual{0.0F};
  float divergence_rms{0.0F};
  float mean_inlet_speed{0.0F};
  float max_speed{0.0F};
  std::uint32_t min_gap_voxels{0};
};

struct ResolvedFlowResult {
  SignalGridVelocityField field;
  ResolvedFlowReport report;
};

// How a port is driven. A speed port carries a stated mean speed; a pressure
// port is held at a stated pressure and carries whatever flow that draws.
enum class FlowPortKind : std::uint8_t {
  speed,
  pressure,
};

// An opening in one face of the grid through which fluid enters or leaves.
// `patch` is the rectangle of boundary sites the opening spans. For a speed
// port, `value` is the mean speed normal to the face over the port's open
// area, positive into the grid and negative out of it. For a pressure port it
// is the pressure held just outside the face, in the solver's own units:
// only differences between ports matter, and with the solvers' relative
// mobility and unit viscosity they are not physical pressures.
struct FlowPort {
  GridFace face{GridFace::y_lower};
  GridFacePatch patch;
  FlowPortKind kind{FlowPortKind::pressure};
  float value{0.0F};
};

// What a solved field carries through one port: the pressure held outside it,
// the volume entering the grid per unit time (negative when leaving), and that
// flux over the port's open area.
struct FlowPortState {
  float pressure{0.0F};
  float flux{0.0F};
  float mean_speed{0.0F};
};

struct DepthAveragedPortFlowParameters {
  float relative_tolerance{1.0e-6F};
  std::uint32_t max_iterations{50'000};

  void validate() const;
};

// `iterations` sums, and `relative_residual` is the largest over, the
// `solves` pressure solves whose combination is the field.
struct DepthAveragedPortFlowReport {
  std::uint32_t solves{0};
  std::uint32_t iterations{0};
  float relative_residual{0.0F};
  float max_speed{0.0F};
};

struct DepthAveragedPortFlowResult {
  SignalGridVelocityField field;
  DepthAveragedPortFlowReport report;
  std::vector<FlowPortState> ports;
};

// A port solve has no linear profile to remove beforehand, as a single-axis
// solve does. It solves for the correction to the depth-averaged pressure
// where the geometry is one shallow flow describes, and for the whole
// pressure field otherwise, and a long channel takes more outer iterations
// than a single-axis solve of it.
struct ResolvedPortFlowParameters {
  float relative_tolerance{1.0e-6F};
  std::uint32_t max_outer_iterations{5000};
  float inner_relative_tolerance{1.0e-6F};
  std::uint32_t max_inner_iterations{50'000};

  void validate() const;
};

// Iteration counts sum, and residuals are the largest over, the `solves`
// block solves whose combination is the field. `divergence_rms` is that of
// the combined field.
struct ResolvedPortFlowReport {
  std::uint32_t solves{0};
  std::uint32_t outer_iterations{0};
  std::uint64_t inner_iterations{0};
  float relative_residual{0.0F};
  float momentum_relative_residual{0.0F};
  float divergence_rms{0.0F};
  float max_speed{0.0F};
  std::uint32_t min_gap_voxels{0};
};

struct ResolvedPortFlowResult {
  SignalGridVelocityField field;
  ResolvedPortFlowReport report;
  std::vector<FlowPortState> ports;
};

void validate_flow_grid(const SignalGridSpec& spec, FlowAxis axis);
// Checks that a grid and its ports state a solvable flow: the port faces are
// fixed boundaries, no axis is periodic, the ports lie on their faces without
// overlapping, each opens onto fluid, at least one holds a pressure, and every
// speed port reaches a pressure port through fluid.
void validate_flow_ports(const SignalGridSpec& spec, std::span<const FlowPort> ports);

[[nodiscard]] DepthAveragedFlowResult solve_depth_averaged_flow_cpu(
    const SignalGridSpec& spec, std::span<const float> mobility,
    const DepthAveragedFlowParameters& parameters = DepthAveragedFlowParameters{});

[[nodiscard]] DepthAveragedPortFlowResult solve_depth_averaged_port_flow_cpu(
    const SignalGridSpec& spec, std::span<const FlowPort> ports, std::span<const float> mobility,
    const DepthAveragedPortFlowParameters& parameters = DepthAveragedPortFlowParameters{});
[[nodiscard]] ResolvedFlowResult solve_resolved_flow_cpu(
    const SignalGridSpec& spec, std::span<const float> drag,
    const ResolvedFlowParameters& parameters = ResolvedFlowParameters{});
[[nodiscard]] ResolvedPortFlowResult solve_resolved_port_flow_cpu(
    const SignalGridSpec& spec, std::span<const FlowPort> ports, std::span<const float> drag,
    const ResolvedPortFlowParameters& parameters = ResolvedPortFlowParameters{});

}  // namespace microsimulator
