#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include "microsimulator/constraints.hpp"
#include "microsimulator/contact_dynamics.hpp"
#include "microsimulator/contact_graph.hpp"
#include "microsimulator/coupled_rates.hpp"
#include "microsimulator/deposition.hpp"
#include "microsimulator/flow.hpp"
#include "microsimulator/mechanics.hpp"
#include "microsimulator/signals.hpp"
#include "microsimulator/species.hpp"
#include "microsimulator/types.hpp"
#include "microsimulator/world_state.hpp"

namespace microsimulator {

// The compute a backend provides, divided by the physics it serves. Every
// method is pure virtual, and each backend is a final class deriving from
// ComputeBackend, so a backend that leaves any of them out does not compile.
// Every stage that visits cells, sites, faces, or contacts during a step is a
// method here, so that the CPU, CUDA, and Metal backends each implement all
// of them.

// Growth, contact detection, and the contact force solve.
class MechanicsProvider {
 public:
  virtual ~MechanicsProvider() = default;

  virtual void advance_growth(WorldState& state, float dt) = 0;
  [[nodiscard]] virtual ContactGraph find_cell_contacts(const WorldState& state,
                                                        const ContactParameters& parameters) = 0;
  [[nodiscard]] virtual ExternalContactGraph find_external_contacts(
      const WorldState& state, const ConstraintSet& constraints,
      const ConstraintContactParameters& parameters) = 0;
  // Solves one substep's contact system from the given multipliers.
  [[nodiscard]] virtual ContactSolveResult solve_contact_forces(
      const ContactSystem& system, std::span<const float> initial_multipliers,
      const ContactSolveParameters& parameters) = 0;
};

// Intracellular species, extracellular transport, and exchange between them.
class TransportProvider {
 public:
  virtual ~TransportProvider() = default;

  virtual void advance_species(WorldState& state, const SpeciesRatePlan& plan,
                               std::span<const float> previous_lengths, float dt) = 0;
  [[nodiscard]] virtual SignalSolveReport advance_signal_grid(SignalGrid& grid, float dt) = 0;
  [[nodiscard]] virtual SignalSolveReport advance_coupled(WorldState& state, SignalGrid& grid,
                                                          const CoupledRatePlan& plan,
                                                          std::span<const float> previous_lengths,
                                                          float dt) = 0;
};

// Steady device flow.
class FlowProvider {
 public:
  virtual ~FlowProvider() = default;

  [[nodiscard]] virtual DepthAveragedFlowResult solve_depth_averaged_flow(
      const SignalGridSpec& spec, std::span<const float> mobility,
      const DepthAveragedFlowParameters& parameters) = 0;
  [[nodiscard]] virtual DepthAveragedPortFlowResult solve_depth_averaged_port_flow(
      const SignalGridSpec& spec, std::span<const FlowPort> ports, std::span<const float> mobility,
      const DepthAveragedPortFlowParameters& parameters) = 0;
  [[nodiscard]] virtual ResolvedFlowResult solve_resolved_flow(
      const SignalGridSpec& spec, std::span<const float> drag,
      const ResolvedFlowParameters& parameters) = 0;
  [[nodiscard]] virtual ResolvedPortFlowResult solve_resolved_port_flow(
      const SignalGridSpec& spec, std::span<const FlowPort> ports, std::span<const float> drag,
      const ResolvedPortFlowParameters& parameters) = 0;
};

// What cells contribute to the fields around them, and the closures that
// turn it into flow coefficients.
class CouplingProvider {
 public:
  virtual ~CouplingProvider() = default;

  [[nodiscard]] virtual BiomassDeposit deposit_biomass(
      const SignalGridSpec& spec, const DepositCells& cells,
      const BiomassDepositParameters& parameters) = 0;
  // `base` holds one mobility per site, or one for every site.
  [[nodiscard]] virtual ShallowFlowCoupling shallow_flow_coupling(
      const SignalGridSpec& spec, std::span<const float> base, const BiomassDeposit& deposit,
      const BiomassDragParameters& parameters) = 0;
  // The resolved-flow Brinkman drag of a deposit, one value per site.
  [[nodiscard]] virtual std::vector<float> brinkman_drag(
      const SignalGridSpec& spec, const BiomassDeposit& deposit,
      const BiomassDragParameters& parameters) = 0;
};

class ComputeBackend : public MechanicsProvider,
                       public TransportProvider,
                       public FlowProvider,
                       public CouplingProvider {
 public:
  ~ComputeBackend() override = default;

  [[nodiscard]] virtual BackendInfo info() const = 0;
  // Every backend implements every feature, which the interfaces above
  // enforce at compile time.
  [[nodiscard]] bool supports(BackendFeature /*feature*/) const noexcept {
    return true;
  }
};

[[nodiscard]] std::unique_ptr<ComputeBackend> make_cpu_backend(std::uint32_t device_index = 0);
[[nodiscard]] std::unique_ptr<ComputeBackend> make_metal_backend(std::uint32_t device_index = 0);
[[nodiscard]] std::unique_ptr<ComputeBackend> make_cuda_backend(std::uint32_t device_index = 0);
[[nodiscard]] std::size_t metal_backend_device_count() noexcept;
[[nodiscard]] std::size_t cuda_backend_device_count() noexcept;
[[nodiscard]] std::size_t backend_device_count(BackendKind kind) noexcept;
[[nodiscard]] bool backend_available(BackendKind kind, std::uint32_t device_index = 0) noexcept;

}  // namespace microsimulator
