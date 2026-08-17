#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "cm/backend.hpp"
#include "cm/checkpoint.hpp"

namespace cm {

class Simulation {
 public:
  explicit Simulation(BackendKind backend = BackendKind::cpu, std::size_t reserved_capacity = 0,
                      std::size_t species_count = 0, std::uint32_t device_index = 0);
  Simulation(BackendKind backend, const SimulationCheckpoint& checkpoint,
             std::uint32_t device_index = 0);

  [[nodiscard]] BackendInfo backend_info() const;
  [[nodiscard]] bool supports(BackendFeature feature) const noexcept;
  [[nodiscard]] double time() const noexcept;
  [[nodiscard]] std::size_t cell_count() const noexcept;
  [[nodiscard]] std::size_t species_count() const noexcept;
  [[nodiscard]] std::size_t signal_count() const noexcept;
  [[nodiscard]] bool has_signal_grid() const noexcept;
  [[nodiscard]] std::optional<SignalSolveReport> last_signal_solve_report() const noexcept;
  [[nodiscard]] bool has_coupled_rate_plan() const noexcept;

  CellId add_cell(const CellInit& cell);
  ConstraintId add_plane_constraint(const PlaneConstraintInit& plane);
  ConstraintId add_sphere_constraint(const SphereConstraintInit& sphere);
  ConstraintId add_box_constraint(const BoxConstraintInit& box);
  void set_cell_geometry(CellId id, Vec3 position, Vec3 direction, float length);
  void set_cell_attributes(CellId id, float growth_rate, std::int32_t cell_type);
  void set_cell_fixed(CellId id, bool fixed);
  void set_species(CellId id, std::span<const float> levels);
  void set_species_rate_plan(const SpeciesRatePlan& plan);
  void set_coupled_rate_plan(const CoupledRatePlan& plan);
  void clear_coupled_rate_plan() noexcept;
  void configure_signal_grid(const SignalGridSpec& spec, std::vector<float> levels = {});
  void set_signal_levels(std::span<const float> levels);
  std::pair<CellId, CellId> divide(CellId parent_id, float first_fraction);
  std::pair<CellId, CellId> divide_equal(CellId parent_id);
  void step(float dt);
  [[nodiscard]] ContactGraph find_cell_contacts(
      const ContactParameters& parameters = ContactParameters{});
  [[nodiscard]] ExternalContactGraph find_external_contacts(
      const ConstraintContactParameters& parameters = ConstraintContactParameters{});
  [[nodiscard]] MechanicsSolveResult solve_cell_mechanics(
      const MechanicsParameters& mechanics_parameters = MechanicsParameters{},
      const ContactParameters& contact_parameters = ContactParameters{},
      const ConstraintContactParameters& constraint_parameters = ConstraintContactParameters{});
  [[nodiscard]] MechanicsSolveResult relax_cell_mechanics(
      const MechanicsParameters& mechanics_parameters = MechanicsParameters{},
      const ContactParameters& contact_parameters = ContactParameters{},
      const MechanicsIntegrationParameters& integration_parameters =
          MechanicsIntegrationParameters{},
      const ConstraintContactParameters& constraint_parameters = ConstraintContactParameters{});

  [[nodiscard]] CellSnapshot cell(CellId id) const;
  [[nodiscard]] std::vector<CellSnapshot> cells() const;
  [[nodiscard]] std::optional<CellId> lineage_parent(CellId id) const noexcept;
  [[nodiscard]] std::vector<float> signal_levels() const;
  [[nodiscard]] std::vector<float> sample_signals(Vec3 position) const;
  [[nodiscard]] SimulationCheckpoint checkpoint() const;
  void validate() const;

 private:
  WorldState state_;
  ConstraintSet constraints_;
  std::unique_ptr<ComputeBackend> backend_;
  SpeciesRatePlan species_rate_plan_;
  std::optional<SignalGrid> signal_grid_;
  std::optional<SignalSolveReport> last_signal_solve_report_;
  std::optional<CoupledRatePlan> coupled_rate_plan_;
  double time_{0.0};
};

}  // namespace cm
