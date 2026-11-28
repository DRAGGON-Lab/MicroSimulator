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

namespace detail {
class ContactStepDevice;
}  // namespace detail

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
  // One contact per wall a cell end is near; a box or cylinder container
  // counts as its nearest face.
  [[nodiscard]] virtual ExternalContactGraph find_external_contacts(
      const WorldState& state, const ConstraintSet& constraints,
      const ConstraintContactParameters& parameters) = 0;
  // The same contacts, except that a cell inside a box or cylinder makes one
  // for every face of it that an end is near, as a contact solve needs to
  // hold a cell in a corner.
  [[nodiscard]] virtual ExternalContactGraph find_external_contact_faces(
      const WorldState& state, const ConstraintSet& constraints,
      const ConstraintContactParameters& parameters) = 0;
  // The storage and stages of contact dynamics substeps on this backend's
  // device, which keep a step's geometry, contacts, and rows there. The
  // backend owns it, and one step uses it at a time.
  [[nodiscard]] virtual detail::ContactStepDevice& contact_step_device() = 0;
  // Solves one substep's contact system from the given multipliers, and
  // returns what the multipliers it reaches do to the cells.
  [[nodiscard]] virtual ContactSolveResult solve_contact_forces(
      const ContactSystem& system, std::span<const float> initial_multipliers,
      const ContactSolveParameters& parameters) = 0;
  // The velocity then angular velocity the grid's flow gives each cell over a
  // move of length `dt`, by slot. Fixed cells do not drift.
  [[nodiscard]] virtual std::vector<std::array<float, 6>> flow_drift_rates(const WorldState& state,
                                                                           const SignalGrid& grid,
                                                                           float dt) = 0;
  // Carries every free cell with the grid's flow over `dt`, in substeps that
  // move it at most a quarter of the smallest spacing and turn it at most the
  // parameters' rotation limit. The state is unchanged if this throws.
  virtual void advance_flow_drift(WorldState& state, const SignalGrid& grid, float dt,
                                  const FlowDriftParameters& parameters) = 0;
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

// The backend of the given kind on the given device. Throws when this build
// does not include it.
[[nodiscard]] std::unique_ptr<ComputeBackend> make_compute_backend(BackendKind kind,
                                                                   std::uint32_t device_index = 0);
[[nodiscard]] std::unique_ptr<ComputeBackend> make_cpu_backend(std::uint32_t device_index = 0);
[[nodiscard]] std::unique_ptr<ComputeBackend> make_metal_backend(std::uint32_t device_index = 0);
[[nodiscard]] std::unique_ptr<ComputeBackend> make_cuda_backend(std::uint32_t device_index = 0);
[[nodiscard]] std::size_t metal_backend_device_count() noexcept;
[[nodiscard]] std::size_t cuda_backend_device_count() noexcept;
[[nodiscard]] std::size_t backend_device_count(BackendKind kind) noexcept;
[[nodiscard]] bool backend_available(BackendKind kind, std::uint32_t device_index = 0) noexcept;

}  // namespace microsimulator
