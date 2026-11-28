#include <memory>
#include <stdexcept>
#include <type_traits>

#include "cpu/cpu_contact_step.hpp"
#include "cpu/cpu_deposition.hpp"
#include "cpu/cpu_flow_drift.hpp"
#include "microsimulator/backend.hpp"

namespace microsimulator {
namespace {

class CpuBackend final : public ComputeBackend {
 public:
  [[nodiscard]] BackendInfo info() const override {
    return {
        .kind = BackendKind::cpu,
        .name = "cpu-reference",
        .device = "host",
        .device_index = 0,
        .native = true,
    };
  }

  void advance_growth(WorldState& state, float dt) override {
    state.advance_growth(dt);
  }

  void advance_species(WorldState& state, const SpeciesRatePlan& plan,
                       std::span<const float> previous_lengths, float dt) override {
    advance_species_cpu(state, plan, previous_lengths, dt);
  }

  SignalSolveReport advance_signal_grid(SignalGrid& grid, float dt) override {
    return advance_signal_grid_cpu(grid, dt);
  }

  SignalSolveReport advance_coupled(WorldState& state, SignalGrid& grid,
                                    const CoupledRatePlan& plan,
                                    std::span<const float> previous_lengths, float dt) override {
    return advance_coupled_cpu(state, grid, plan, previous_lengths, dt);
  }

  [[nodiscard]] detail::ContactStepDevice& contact_step_device() override {
    if (!step_device_) {
      step_device_ = make_cpu_contact_step_device();
    }
    return *step_device_;
  }

  [[nodiscard]] ContactGraph find_cell_contacts(const WorldState& state,
                                                const ContactParameters& parameters) override {
    return find_cell_contacts_cpu(state, parameters);
  }

  [[nodiscard]] ExternalContactGraph find_external_contacts(
      const WorldState& state, const ConstraintSet& constraints,
      const ConstraintContactParameters& parameters) override {
    return find_external_contacts_cpu(state, constraints, parameters);
  }

  [[nodiscard]] ExternalContactGraph find_external_contact_faces(
      const WorldState& state, const ConstraintSet& constraints,
      const ConstraintContactParameters& parameters) override {
    return find_external_contact_faces_cpu(state, constraints, parameters);
  }

  [[nodiscard]] ContactSolveResult solve_contact_forces(
      const ContactSystem& system, std::span<const float> initial_multipliers,
      const ContactSolveParameters& parameters) override {
    return solve_contact_system_cpu(system, initial_multipliers, parameters);
  }

  [[nodiscard]] std::vector<std::array<float, 6>> flow_drift_rates(const WorldState& state,
                                                                   const SignalGrid& grid,
                                                                   float dt) override {
    return flow_drift_rates_cpu(state, grid, dt);
  }

  void advance_flow_drift(WorldState& state, const SignalGrid& grid, float dt,
                          const FlowDriftParameters& parameters) override {
    advance_flow_drift_cpu(state, grid, dt, parameters);
  }

  [[nodiscard]] DepthAveragedFlowResult solve_depth_averaged_flow(
      const SignalGridSpec& spec, std::span<const float> mobility,
      const DepthAveragedFlowParameters& parameters) override {
    return solve_depth_averaged_flow_cpu(spec, mobility, parameters);
  }

  [[nodiscard]] DepthAveragedPortFlowResult solve_depth_averaged_port_flow(
      const SignalGridSpec& spec, std::span<const FlowPort> ports, std::span<const float> mobility,
      const DepthAveragedPortFlowParameters& parameters) override {
    return solve_depth_averaged_port_flow_cpu(spec, ports, mobility, parameters);
  }

  [[nodiscard]] ResolvedFlowResult solve_resolved_flow(
      const SignalGridSpec& spec, std::span<const float> drag,
      const ResolvedFlowParameters& parameters) override {
    return solve_resolved_flow_cpu(spec, drag, parameters);
  }

  [[nodiscard]] ResolvedPortFlowResult solve_resolved_port_flow(
      const SignalGridSpec& spec, std::span<const FlowPort> ports, std::span<const float> drag,
      const ResolvedPortFlowParameters& parameters) override {
    return solve_resolved_port_flow_cpu(spec, ports, drag, parameters);
  }

  [[nodiscard]] BiomassDeposit deposit_biomass(
      const SignalGridSpec& spec, const DepositCells& cells,
      const BiomassDepositParameters& parameters) override {
    return deposit_biomass_cpu(spec, cells, parameters);
  }

  [[nodiscard]] ShallowFlowCoupling shallow_flow_coupling(
      const SignalGridSpec& spec, std::span<const float> base, const BiomassDeposit& deposit,
      const BiomassDragParameters& parameters) override {
    return shallow_flow_coupling_cpu(spec, base, deposit, parameters);
  }

  [[nodiscard]] std::vector<float> brinkman_drag(const SignalGridSpec& spec,
                                                 const BiomassDeposit& deposit,
                                                 const BiomassDragParameters& parameters) override {
    return brinkman_drag_cpu(spec, deposit, parameters);
  }

 private:
  std::unique_ptr<detail::ContactStepDevice> step_device_;
};

// A backend that leaves out any provider method is abstract.
static_assert(!std::is_abstract_v<CpuBackend>, "the CPU backend must implement every provider");

}  // namespace

std::unique_ptr<ComputeBackend> make_cpu_backend(std::uint32_t device_index) {
  if (device_index != 0) {
    throw std::out_of_range("CPU backend exposes only device index 0");
  }
  return std::make_unique<CpuBackend>();
}

}  // namespace microsimulator
