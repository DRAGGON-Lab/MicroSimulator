#pragma once

#include <cstdint>
#include <memory>
#include <span>

#include "microsimulator/flow.hpp"

namespace microsimulator::metal {

class FlowSolver {
 public:
  explicit FlowSolver(std::uint32_t device_index);
  ~FlowSolver();

  FlowSolver(const FlowSolver&) = delete;
  FlowSolver& operator=(const FlowSolver&) = delete;

  [[nodiscard]] DepthAveragedFlowResult solve_depth_averaged(
      const SignalGridSpec& spec, std::span<const float> mobility,
      const DepthAveragedFlowParameters& parameters);
  [[nodiscard]] DepthAveragedPortFlowResult solve_depth_averaged_ports(
      const SignalGridSpec& spec, std::span<const FlowPort> ports, std::span<const float> mobility,
      const DepthAveragedPortFlowParameters& parameters);
  [[nodiscard]] ResolvedFlowResult solve_resolved(const SignalGridSpec& spec,
                                                  std::span<const float> drag,
                                                  const ResolvedFlowParameters& parameters);
  [[nodiscard]] ResolvedPortFlowResult solve_resolved_ports(
      const SignalGridSpec& spec, std::span<const FlowPort> ports, std::span<const float> drag,
      const ResolvedPortFlowParameters& parameters);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace microsimulator::metal
