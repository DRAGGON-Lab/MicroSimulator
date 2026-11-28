#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "microsimulator/deposition.hpp"

namespace microsimulator::metal {

// The shared deposition and shallow-flow closure bodies, run as Metal
// kernels on one device.
class DepositionKernels {
 public:
  explicit DepositionKernels(std::uint32_t device_index);
  ~DepositionKernels();

  DepositionKernels(const DepositionKernels&) = delete;
  DepositionKernels& operator=(const DepositionKernels&) = delete;

  [[nodiscard]] BiomassDeposit deposit(const SignalGridSpec& spec, const DepositCells& cells,
                                       const BiomassDepositParameters& parameters);
  [[nodiscard]] ShallowFlowCoupling shallow_flow_coupling(const SignalGridSpec& spec,
                                                          std::span<const float> base,
                                                          const BiomassDeposit& deposit,
                                                          const BiomassDragParameters& parameters);
  [[nodiscard]] std::vector<float> brinkman_drag(const SignalGridSpec& spec,
                                                 const BiomassDeposit& deposit,
                                                 const BiomassDragParameters& parameters);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace microsimulator::metal
