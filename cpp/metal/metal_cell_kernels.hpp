#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "microsimulator/constraints.hpp"
#include "microsimulator/deposition.hpp"
#include "microsimulator/mechanics.hpp"
#include "microsimulator/signals.hpp"
#include "microsimulator/world_state.hpp"

namespace microsimulator::metal {

// The per-cell Metal kernels of one device: biomass deposition and its flow
// closures, external contacts, and flow drift.
class CellKernels {
 public:
  explicit CellKernels(std::uint32_t device_index);
  ~CellKernels();

  CellKernels(const CellKernels&) = delete;
  CellKernels& operator=(const CellKernels&) = delete;

  [[nodiscard]] BiomassDeposit deposit(const SignalGridSpec& spec, const DepositCells& cells,
                                       const BiomassDepositParameters& parameters);
  [[nodiscard]] ShallowFlowCoupling shallow_flow_coupling(const SignalGridSpec& spec,
                                                          std::span<const float> base,
                                                          const BiomassDeposit& deposit,
                                                          const BiomassDragParameters& parameters);
  [[nodiscard]] std::vector<float> brinkman_drag(const SignalGridSpec& spec,
                                                 const BiomassDeposit& deposit,
                                                 const BiomassDragParameters& parameters);
  [[nodiscard]] ExternalContactGraph external_contacts(
      const WorldState& state, const ConstraintSet& constraints,
      const ConstraintContactParameters& parameters, bool every_face);
  [[nodiscard]] std::vector<std::array<float, 6>> flow_drift_rates(const WorldState& state,
                                                                   const SignalGrid& grid,
                                                                   float dt);
  void advance_flow_drift(WorldState& state, const SignalGrid& grid, float dt,
                          const FlowDriftParameters& parameters);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace microsimulator::metal
