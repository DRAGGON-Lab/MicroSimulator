#pragma once

#include <cstdint>
#include <memory>
#include <span>

#include "cm/flow.hpp"

namespace cm::metal {

class FlowSolver {
 public:
  explicit FlowSolver(std::uint32_t device_index);
  ~FlowSolver();

  FlowSolver(const FlowSolver&) = delete;
  FlowSolver& operator=(const FlowSolver&) = delete;

  [[nodiscard]] DepthAveragedFlowResult solve_depth_averaged(
      const SignalGridSpec& spec, std::span<const float> mobility,
      const DepthAveragedFlowParameters& parameters);
  [[nodiscard]] ResolvedFlowResult solve_resolved(const SignalGridSpec& spec,
                                                  std::span<const float> drag,
                                                  const ResolvedFlowParameters& parameters);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace cm::metal
