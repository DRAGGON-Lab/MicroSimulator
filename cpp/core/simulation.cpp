#include "cm/simulation.hpp"

#include <cmath>
#include <stdexcept>

namespace cm {
namespace {

std::unique_ptr<ComputeBackend> make_backend(BackendKind kind, std::uint32_t device_index) {
  switch (kind) {
    case BackendKind::cpu:
      return make_cpu_backend(device_index);
    case BackendKind::metal:
#if CM_HAS_METAL
      return make_metal_backend(device_index);
#else
      throw std::runtime_error("Metal backend is not implemented in this build");
#endif
    case BackendKind::cuda:
#if CM_HAS_CUDA
      return make_cuda_backend(device_index);
#else
      throw std::runtime_error("CUDA backend is not implemented in this build");
#endif
  }
  throw std::runtime_error("unknown compute backend");
}

const SimulationCheckpoint& validated_checkpoint(const SimulationCheckpoint& checkpoint) {
  checkpoint.validate();
  return checkpoint;
}

}  // namespace

std::size_t backend_device_count(BackendKind kind) noexcept {
  if (kind == BackendKind::cpu) {
    return 1;
  }
#if CM_HAS_METAL
  if (kind == BackendKind::metal) {
    return metal_backend_device_count();
  }
#endif
#if CM_HAS_CUDA
  if (kind == BackendKind::cuda) {
    return cuda_backend_device_count();
  }
#endif
  return 0;
}

bool backend_available(BackendKind kind, std::uint32_t device_index) noexcept {
  return static_cast<std::size_t>(device_index) < backend_device_count(kind);
}

Simulation::Simulation(BackendKind backend, std::size_t reserved_capacity,
                       std::size_t species_count, std::uint32_t device_index)
    : state_(reserved_capacity, species_count),
      backend_(make_backend(backend, device_index)),
      species_rate_plan_(SpeciesRatePlan::zero(species_count)) {}

Simulation::Simulation(BackendKind backend, const SimulationCheckpoint& checkpoint,
                       std::uint32_t device_index)
    : state_(validated_checkpoint(checkpoint).world),
      constraints_(checkpoint.constraints),
      backend_(make_backend(backend, device_index)),
      species_rate_plan_(checkpoint.species_rate_plan),
      signal_grid_(checkpoint.signal_grid.has_value()
                       ? std::optional<SignalGrid>(SignalGrid(*checkpoint.signal_grid))
                       : std::nullopt),
      coupled_rate_plan_(checkpoint.coupled_rate_plan),
      time_(checkpoint.time) {
  validate();
}

BackendInfo Simulation::backend_info() const { return backend_->info(); }

bool Simulation::supports(BackendFeature feature) const noexcept {
  return backend_->supports(feature);
}

double Simulation::time() const noexcept { return time_; }

std::size_t Simulation::cell_count() const noexcept { return state_.size(); }

std::size_t Simulation::species_count() const noexcept { return state_.species_count(); }

std::size_t Simulation::signal_count() const noexcept {
  return signal_grid_.has_value() ? signal_grid_->spec().signal_count : 0;
}

bool Simulation::has_signal_grid() const noexcept { return signal_grid_.has_value(); }

std::optional<SignalSolveReport> Simulation::last_signal_solve_report() const noexcept {
  return last_signal_solve_report_;
}

bool Simulation::has_coupled_rate_plan() const noexcept { return coupled_rate_plan_.has_value(); }

CellId Simulation::add_cell(const CellInit& cell) { return state_.add_cell(cell); }

void Simulation::remove_cell(CellId id) { state_.remove_cell(id); }

ConstraintId Simulation::add_plane_constraint(const PlaneConstraintInit& plane) {
  return constraints_.add_plane(plane);
}

ConstraintId Simulation::add_sphere_constraint(const SphereConstraintInit& sphere) {
  return constraints_.add_sphere(sphere);
}

ConstraintId Simulation::add_box_constraint(const BoxConstraintInit& box) {
  return constraints_.add_box(box);
}

ConstraintId Simulation::add_cylinder_constraint(const CylinderConstraintInit& cylinder) {
  return constraints_.add_cylinder(cylinder);
}

void Simulation::set_cell_geometry(CellId id, Vec3 position, Vec3 direction, float length) {
  state_.set_cell_geometry(id, position, direction, length);
}

void Simulation::set_cell_attributes(CellId id, float growth_rate, std::int32_t cell_type) {
  state_.set_cell_attributes(id, growth_rate, cell_type);
}

void Simulation::set_cell_fixed(CellId id, bool fixed) { state_.set_cell_fixed(id, fixed); }

void Simulation::set_species(CellId id, std::span<const float> levels) {
  state_.set_species(id, levels);
}

void Simulation::set_species_rate_plan(const SpeciesRatePlan& plan) {
  plan.validate();
  if (plan.species_count() != state_.species_count()) {
    throw std::invalid_argument("species rate plan and simulation species counts disagree");
  }
  species_rate_plan_ = plan;
}

void Simulation::set_coupled_rate_plan(const CoupledRatePlan& plan) {
  plan.validate();
  if (!signal_grid_.has_value()) {
    throw std::logic_error("coupled rate plan requires a signal grid");
  }
  if (plan.species_count() != state_.species_count() ||
      plan.signal_count() != signal_grid_->spec().signal_count) {
    throw std::invalid_argument("coupled rate plan counts disagree with the simulation");
  }
  coupled_rate_plan_ = plan;
}

void Simulation::clear_coupled_rate_plan() noexcept { coupled_rate_plan_.reset(); }

void Simulation::configure_signal_grid(const SignalGridSpec& spec, std::vector<float> levels) {
  if (!state_.empty()) {
    throw std::logic_error("signal grid geometry must be configured before cells are added");
  }
  if (signal_grid_.has_value()) {
    throw std::logic_error("signal grid geometry is already configured");
  }
  signal_grid_.emplace(spec, std::move(levels));
}

void Simulation::set_signal_levels(std::span<const float> levels) {
  if (!signal_grid_.has_value()) {
    throw std::logic_error("simulation does not have a signal grid");
  }
  signal_grid_->set_levels(levels);
}

void Simulation::set_velocity_field(std::optional<SignalGridVelocityField> field) {
  if (!signal_grid_.has_value()) {
    throw std::logic_error("simulation does not have a signal grid");
  }
  signal_grid_->set_velocity_field(std::move(field));
}

std::pair<CellId, CellId> Simulation::divide(CellId parent_id, float first_fraction) {
  return state_.divide(parent_id, first_fraction);
}

std::pair<CellId, CellId> Simulation::divide_equal(CellId parent_id) {
  return state_.divide_equal(parent_id);
}

void Simulation::step(float dt) {
  if (!std::isfinite(dt) || dt < 0.0F) {
    throw std::invalid_argument("time step must be finite and non-negative");
  }
  const auto geometry = state_.geometry_state();
  const std::vector<float> previous_lengths(geometry.lengths.begin(), geometry.lengths.end());
  if (coupled_rate_plan_.has_value()) {
    if (!backend_->supports(BackendFeature::coupled_rates)) {
      throw std::runtime_error("selected backend does not implement coupled rates");
    }
    validate_coupled_step(state_, *signal_grid_, *coupled_rate_plan_, previous_lengths, dt);
  } else {
    if (state_.species_count() != 0 && !backend_->supports(BackendFeature::species)) {
      throw std::runtime_error("selected backend does not implement species integration");
    }
  }
  if (signal_grid_.has_value() && !coupled_rate_plan_.has_value()) {
    if (!backend_->supports(BackendFeature::signals)) {
      throw std::runtime_error("selected backend does not implement signal grid integration");
    }
    signal_grid_->validate_step(dt);
  }
  backend_->advance_growth(state_, dt);
  last_signal_solve_report_.reset();
  if (coupled_rate_plan_.has_value()) {
    last_signal_solve_report_ =
        backend_->advance_coupled(state_, *signal_grid_, *coupled_rate_plan_, previous_lengths, dt);
  } else {
    if (state_.species_count() != 0) {
      backend_->advance_species(state_, species_rate_plan_, previous_lengths, dt);
    }
    if (signal_grid_.has_value()) {
      last_signal_solve_report_ = backend_->advance_signal_grid(*signal_grid_, dt);
    }
  }
  time_ += static_cast<double>(dt);
}

ContactGraph Simulation::find_cell_contacts(const ContactParameters& parameters) {
  return backend_->find_cell_contacts(state_, parameters);
}

ExternalContactGraph Simulation::find_external_contacts(
    const ConstraintContactParameters& parameters) {
  if (!backend_->supports(BackendFeature::external_constraints)) {
    throw std::runtime_error("selected backend does not implement external constraints");
  }
  return backend_->find_external_contacts(state_, constraints_, parameters);
}

MechanicsSolveResult Simulation::solve_cell_mechanics(
    const MechanicsParameters& mechanics_parameters, const ContactParameters& contact_parameters,
    const ConstraintContactParameters& constraint_parameters) {
  if (!backend_->supports(BackendFeature::cell_mechanics)) {
    throw std::runtime_error("selected backend does not implement cell mechanics");
  }
  validate_constraint_contact_parameters(constraint_parameters);
  if (!constraints_.empty() && !backend_->supports(BackendFeature::external_constraints)) {
    throw std::runtime_error("selected backend does not implement external constraints");
  }
  const auto contacts = backend_->find_cell_contacts(state_, contact_parameters);
  ExternalContactGraph external_contacts(state_.size(), {});
  if (!constraints_.empty()) {
    external_contacts =
        backend_->find_external_contacts(state_, constraints_, constraint_parameters);
  }
  return backend_->solve_cell_mechanics(state_, contacts, external_contacts, mechanics_parameters);
}

MechanicsSolveResult Simulation::relax_cell_mechanics(
    const MechanicsParameters& mechanics_parameters, const ContactParameters& contact_parameters,
    const MechanicsIntegrationParameters& integration_parameters,
    const ConstraintContactParameters& constraint_parameters) {
  auto result =
      solve_cell_mechanics(mechanics_parameters, contact_parameters, constraint_parameters);
  integrate_mechanics_result(state_, result, integration_parameters);
  return result;
}

CellSnapshot Simulation::cell(CellId id) const { return state_.cell(id); }

std::vector<CellSnapshot> Simulation::cells() const { return state_.cells(); }

std::optional<CellId> Simulation::lineage_parent(CellId id) const noexcept {
  return state_.lineage_parent(id);
}

std::vector<float> Simulation::signal_levels() const {
  if (!signal_grid_.has_value()) {
    throw std::logic_error("simulation does not have a signal grid");
  }
  return std::vector<float>(signal_grid_->levels().begin(), signal_grid_->levels().end());
}

std::vector<float> Simulation::sample_signals(Vec3 position) const {
  if (!signal_grid_.has_value()) {
    throw std::logic_error("simulation does not have a signal grid");
  }
  return signal_grid_->sample(position);
}

SimulationCheckpoint Simulation::checkpoint() const {
  validate();
  SimulationCheckpoint result{
      .schema_version = checkpoint_schema_version,
      .time = time_,
      .world = state_.checkpoint(),
      .constraints = constraints_.checkpoint(),
      .species_rate_plan = species_rate_plan_,
      .signal_grid = signal_grid_.has_value()
                         ? std::optional<SignalGridCheckpoint>(signal_grid_->checkpoint())
                         : std::nullopt,
      .coupled_rate_plan = coupled_rate_plan_,
  };
  result.validate();
  return result;
}

void Simulation::validate() const {
  state_.validate();
  constraints_.validate();
  species_rate_plan_.validate();
  if (signal_grid_.has_value()) {
    signal_grid_->validate();
  }
  if (coupled_rate_plan_.has_value()) {
    coupled_rate_plan_->validate();
    if (!signal_grid_.has_value() ||
        coupled_rate_plan_->species_count() != state_.species_count() ||
        coupled_rate_plan_->signal_count() != signal_grid_->spec().signal_count) {
      throw std::logic_error("simulation coupled rate plan counts disagree with state");
    }
  }
  if (!std::isfinite(time_) || time_ < 0.0) {
    throw std::logic_error("simulation time must be finite and non-negative");
  }
  if (species_rate_plan_.species_count() != state_.species_count()) {
    throw std::logic_error("simulation rate plan and world species counts disagree");
  }
}

}  // namespace cm
