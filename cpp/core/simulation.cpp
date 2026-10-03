#include "cm/simulation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <tuple>

#include "culture_simulation_internal.hpp"

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
  if (checkpoint.culture) {
    culture_ =
        std::make_unique<detail::CultureSimulation>(state_, *checkpoint.culture, backend_->info());
  }

  validate();
}

Simulation::~Simulation() = default;
Simulation::Simulation(Simulation&&) noexcept = default;
Simulation& Simulation::operator=(Simulation&&) noexcept = default;

void Simulation::restore_checkpoint(const SimulationCheckpoint& checkpoint) {
  const auto info = backend_->info();
  Simulation candidate(info.kind, checkpoint, info.device_index);
  *this = std::move(candidate);
}

void Simulation::configure_culture(const CultureConfiguration& configuration,
                                   const std::vector<double>& concentrations,
                                   const std::vector<double>& biochemical_volumes) {
  if (culture_ || signal_grid_ || coupled_rate_plan_ || !constraints_.empty() || time_ != 0) {
    throw std::logic_error(
        "configure fluid flow once, after seeding cells and before stepping, without legacy "
        "signals or constraints");
  }

  auto candidate = std::make_unique<detail::CultureSimulation>(
      state_, configuration, concentrations, biochemical_volumes, backend_->info());
  culture_ = std::move(candidate);
}

bool Simulation::has_culture() const noexcept {
  return bool(culture_);
}

std::optional<CultureCheckpoint> Simulation::culture_checkpoint() const {
  return culture_ ? std::optional<CultureCheckpoint>(culture_->checkpoint()) : std::nullopt;
}

std::vector<FluidFragment> Simulation::fluid_fragments() const {
  return culture_ ? culture_->geometry().fragments() : std::vector<FluidFragment>{};
}

void Simulation::set_cell_force(CellId id, std::array<double, 3> force,
                                std::array<double, 3> torque) {
  if (!culture_) {
    throw std::logic_error("simulation does not have fluid flow");
  }

  culture_->set_force(id, force, torque);
}

std::vector<double> Simulation::cell_surface_concentrations(CellId id) const {
  if (!culture_) {
    throw std::logic_error("simulation does not have fluid flow");
  }

  (void)state_.cell(id);
  const auto& state = culture_->checkpoint();
  const auto& mesh = culture_->geometry();
  const auto ns = state.configuration.solutes.size();
  std::vector<double> result(ns);
  double area = 0;

  for (const auto& face : mesh.faces()) {
    if (face.body_id == id) {
      area += face.area;

      for (std::size_t s = 0; s < ns; ++s) {
        result[s] += face.area * state.extracellular_amounts[face.first * ns + s] /
                     mesh.fragments()[face.first].volume;
      }
    }
  }

  if (!(area > 0)) {
    throw std::runtime_error("culture cell has no resolved extracellular surface");
  }

  for (double& c : result) {
    c /= area;
  }

  return result;
}

BackendInfo Simulation::backend_info() const {
  return backend_->info();
}

bool Simulation::supports(BackendFeature feature) const noexcept {
  return backend_->supports(feature);
}

double Simulation::time() const noexcept {
  return time_;
}

std::size_t Simulation::cell_count() const noexcept {
  return state_.size();
}

std::size_t Simulation::species_count() const noexcept {
  return state_.species_count();
}

std::size_t Simulation::signal_count() const noexcept {
  if (culture_) {
    return culture_->checkpoint().configuration.solutes.size();
  }

  return signal_grid_.has_value() ? signal_grid_->spec().signal_count : 0;
}

bool Simulation::has_signal_grid() const noexcept {
  return signal_grid_.has_value();
}

std::optional<SignalSolveReport> Simulation::last_signal_solve_report() const noexcept {
  return last_signal_solve_report_;
}

bool Simulation::has_coupled_rate_plan() const noexcept {
  return coupled_rate_plan_.has_value();
}

CellId Simulation::add_cell(const CellInit& cell) {
  if (culture_) {
    throw std::logic_error("seed cells before configuring fluid flow");
  }

  return state_.add_cell(cell);
}

void Simulation::remove_cell(CellId id) {
  if (culture_) {
    throw std::logic_error("instantaneous cell removal has no conservative culture-volume model");
  }

  state_.remove_cell(id);
}

namespace {
struct DriftUpdate {
  Slot slot;
  Vec3 position;
  Vec3 direction;
  float length;
};

std::pair<Vec3, Vec3> drift_derivative(const SignalGrid& grid,
                                       const MechanicsIntegrationParameters& parameters,
                                       const std::array<float, 3>& h,
                                       const std::array<std::uint32_t, 3>& dims, float lambda,
                                       Vec3 point, Vec3 axis) {
  const auto sample = [&](Vec3 value) {
    return grid.sample_velocity(value, GridSampleBound::clamped);
  };
  // Central differences of the interpolated fluid velocity. This is an
  // equivalent-spheroid Jeffery closure, not cell-resolved hydrodynamics.
  std::array<Vec3, 3> gradient{};

  for (std::size_t k = 0; k < 3; ++k) {
    if (dims[k] == 1) {
      continue;
    }

    Vec3 offset{};

    if (k == 0) {
      offset.x = h[k] * 0.5F;
    }

    if (k == 1) {
      offset.y = h[k] * 0.5F;
    }

    if (k == 2) {
      offset.z = h[k] * 0.5F;
    }

    gradient[k] = (sample(point + offset) - sample(point - offset)) * (1 / h[k]);
  }

  const Vec3 ap = gradient[0] * axis.x + gradient[1] * axis.y + gradient[2] * axis.z;
  const Vec3 atp{dot(gradient[0], axis), dot(gradient[1], axis), dot(gradient[2], axis)};
  const Vec3 strain = (ap + atp) * 0.5F;
  const Vec3 spin = (ap - atp) * 0.5F;
  const auto orientation = parameters.max_rotation_radians == 0
                               ? Vec3{}
                               : spin + (strain - axis * dot(axis, strain)) * lambda;

  return std::pair{sample(point), orientation};
}

DriftUpdate propose_drift_update(const SignalGrid& grid, const CellGeometryView& geometry,
                                 std::size_t slot, float dt,
                                 const MechanicsIntegrationParameters& parameters,
                                 const std::array<float, 3>& h,
                                 const std::array<std::uint32_t, 3>& dims, float spatial_step) {
  Vec3 position{geometry.position_x[slot], geometry.position_y[slot], geometry.position_z[slot]};
  Vec3 direction{geometry.direction_x[slot], geometry.direction_y[slot],
                 geometry.direction_z[slot]};
  const float aspect =
      (geometry.lengths[slot] + 2 * geometry.radii[slot]) / (2 * geometry.radii[slot]);
  const float lambda = (aspect * aspect - 1) / (aspect * aspect + 1);
  const auto derivative = [&](Vec3 point, Vec3 axis) {
    return drift_derivative(grid, parameters, h, dims, lambda, point, axis);
  };
  double remaining = dt;
  std::uint32_t steps = 0;

  while (remaining > 0) {
    if (++steps > 100000) {
      throw std::runtime_error("flow drift needs too many substeps; reduce dt");
    }

    const auto [velocity, orientation] = derivative(position, direction);
    float step = static_cast<float>(remaining);

    if (norm(velocity) > 0) {
      step = std::min(step, spatial_step / norm(velocity));
    }

    if (norm(orientation) > 0) {
      step = std::min(step, parameters.max_rotation_radians / norm(orientation));
    }

    Vec3 mid_velocity, mid_orientation;

    while (true) {
      const auto midpoint = position + velocity * (step * 0.5F);
      const auto mid_direction = normalized(direction + orientation * (step * 0.5F));
      std::tie(mid_velocity, mid_orientation) = derivative(midpoint, mid_direction);

      if (step * norm(mid_velocity) <= spatial_step * 1.001F &&
          (parameters.max_rotation_radians == 0 ||
           step * norm(mid_orientation) <= parameters.max_rotation_radians * 1.001F)) {
        break;
      }

      step *= 0.5F;

      if (step <= 0) {
        throw std::runtime_error("flow drift substep underflow");
      }
    }

    if (!std::isfinite(step) || step <= 0) {
      throw std::runtime_error("invalid flow drift substep");
    }

    position = position + mid_velocity * step;
    direction = normalized(direction + mid_orientation * step);
    remaining = std::max(0.0, remaining - step);
  }

  if (!std::isfinite(position.x) || !std::isfinite(position.y) || !std::isfinite(position.z)) {
    throw std::runtime_error("flow drift produced non-finite geometry");
  }

  return {static_cast<Slot>(slot), position, direction, geometry.lengths[slot]};
}

}  // namespace

void Simulation::apply_flow_drift(float dt, const MechanicsIntegrationParameters& parameters) {
  if (culture_) {
    throw std::logic_error("fluid flow advances body poses during Simulation.step");
  }

  if (!std::isfinite(dt) || dt < 0.0F) {
    throw std::invalid_argument("time step must be finite and non-negative");
  }

  validate_mechanics_integration_parameters(parameters);

  if (!signal_grid_.has_value() || !signal_grid_->spec().velocity_field.has_value()) {
    throw std::logic_error("flow drift requires a signal grid with a velocity field");
  }

  if (dt == 0.0F || state_.empty()) {
    return;
  }

  const auto geometry = state_.geometry_state();
  const auto attributes = state_.cell_attributes();
  const auto spacing = signal_grid_->spec().spacing;
  const std::array<float, 3> h{spacing.x, spacing.y, spacing.z};
  const std::array<std::uint32_t, 3> dims{
      signal_grid_->spec().shape.x, signal_grid_->spec().shape.y, signal_grid_->spec().shape.z};
  const float spatial_step = 0.25F * *std::min_element(h.begin(), h.end());

  std::vector<DriftUpdate> updates;
  updates.reserve(geometry.size());

  for (std::size_t slot = 0; slot < geometry.size(); ++slot) {
    if (attributes.fixed[slot] != 0) {
      continue;
    }

    updates.push_back(
        propose_drift_update(*signal_grid_, geometry, slot, dt, parameters, h, dims, spatial_step));
  }

  for (const auto& update : updates) {
    state_.set_cell_geometry(update.slot, update.position, update.direction, update.length);
  }
}

ConstraintId Simulation::add_plane_constraint(const PlaneConstraintInit& plane) {
  if (culture_) {
    throw std::logic_error("culture walls belong to FluidGridSpec");
  }

  return constraints_.add_plane(plane);
}

ConstraintId Simulation::add_sphere_constraint(const SphereConstraintInit& sphere) {
  if (culture_) {
    throw std::logic_error("culture walls belong to FluidGridSpec");
  }

  return constraints_.add_sphere(sphere);
}

ConstraintId Simulation::add_box_constraint(const BoxConstraintInit& box) {
  if (culture_) {
    throw std::logic_error("culture walls belong to FluidGridSpec");
  }

  return constraints_.add_box(box);
}

ConstraintId Simulation::add_cylinder_constraint(const CylinderConstraintInit& cylinder) {
  if (culture_) {
    throw std::logic_error("culture walls belong to FluidGridSpec");
  }

  return constraints_.add_cylinder(cylinder);
}

void Simulation::set_cell_geometry(CellId id, Vec3 position, Vec3 direction, float length) {
  if (culture_) {
    throw std::logic_error(
        "culture poses change through physical stepping or conservative division");
  }

  state_.set_cell_geometry(id, position, direction, length);
}

void Simulation::set_cell_attributes(CellId id, float growth_rate, std::int32_t cell_type) {
  if (culture_ && !culture_->checkpoint().configuration.growth.empty() && growth_rate != 0) {
    throw std::invalid_argument("kinetic culture owns growth; prescribed elongation must be zero");
  }

  state_.set_cell_attributes(id, growth_rate, cell_type);
}

void Simulation::set_cell_fixed(CellId id, bool fixed) {
  state_.set_cell_fixed(id, fixed);
}

void Simulation::set_species(CellId id, std::span<const float> levels) {
  if (culture_) {
    culture_->set_species(id, levels);
  }

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
  if (culture_) {
    throw std::logic_error("culture exchange uses explicit surface transfer bindings");
  }

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

void Simulation::clear_coupled_rate_plan() noexcept {
  coupled_rate_plan_.reset();
}

void Simulation::configure_signal_grid(const SignalGridSpec& spec, std::vector<float> levels) {
  if (culture_) {
    throw std::logic_error("culture solutes belong to CultureConfiguration");
  }

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

void Simulation::set_signal_reaction(std::optional<SignalGridAffineReaction> reaction) {
  if (!signal_grid_.has_value()) {
    throw std::logic_error("simulation does not have a signal grid");
  }

  signal_grid_->set_reaction(std::move(reaction));
}

std::pair<CellId, CellId> Simulation::divide(CellId parent_id, float first_fraction) {
  if (culture_) {
    return culture_->divide(state_, parent_id, first_fraction);
  }

  return state_.divide(parent_id, first_fraction);
}

std::pair<CellId, CellId> Simulation::divide_equal(CellId parent_id) {
  if (culture_) {
    return culture_->divide(state_, parent_id, 0.5);
  }

  return state_.divide_equal(parent_id);
}

void Simulation::validate_step_backend(std::span<const float> previous_lengths, float dt) const {
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
}

void Simulation::step(float dt) {
  if (!std::isfinite(dt) || dt < 0.0F) {
    throw std::invalid_argument("time step must be finite and non-negative");
  }

  if (culture_) {
    if (!std::isfinite(time_ + dt)) {
      throw std::overflow_error("simulation time overflow");
    }

    culture_->step(state_, *backend_, species_rate_plan_, dt);
    time_ += dt;

    return;
  }

  const auto geometry = state_.geometry_state();
  const std::vector<float> previous_lengths(geometry.lengths.begin(), geometry.lengths.end());

  validate_step_backend(previous_lengths, dt);

  auto saved_state = state_;
  const auto saved_levels = signal_grid_.has_value() ? signal_levels() : std::vector<float>{};
  const auto saved_report = last_signal_solve_report_;

  try {
    backend_->advance_growth(state_, dt);
    last_signal_solve_report_.reset();

    if (coupled_rate_plan_.has_value()) {
      last_signal_solve_report_ = backend_->advance_coupled(
          state_, *signal_grid_, *coupled_rate_plan_, previous_lengths, dt);
    } else {
      if (state_.species_count() != 0) {
        backend_->advance_species(state_, species_rate_plan_, previous_lengths, dt);
      }

      if (signal_grid_.has_value()) {
        last_signal_solve_report_ = backend_->advance_signal_grid(*signal_grid_, dt);
      }
    }
  } catch (...) {
    state_ = std::move(saved_state);

    if (signal_grid_.has_value()) {
      signal_grid_->set_levels(saved_levels);
    }

    last_signal_solve_report_ = saved_report;
    throw;
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
  if (culture_) {
    throw std::logic_error("fluid flow includes hydrodynamic motion and contact constraints");
  }

  auto result =
      solve_cell_mechanics(mechanics_parameters, contact_parameters, constraint_parameters);
  integrate_mechanics_result(state_, result, integration_parameters);

  return result;
}

DepthAveragedFlowResult Simulation::solve_depth_averaged_flow(
    const SignalGridSpec& spec, std::span<const float> mobility,
    const DepthAveragedFlowParameters& parameters) {
  if (!backend_->supports(BackendFeature::depth_averaged_flow)) {
    throw std::runtime_error("selected backend does not implement depth-averaged flow");
  }

  return backend_->solve_depth_averaged_flow(spec, mobility, parameters);
}

ResolvedFlowResult Simulation::solve_resolved_flow(const SignalGridSpec& spec,
                                                   std::span<const float> drag,
                                                   const ResolvedFlowParameters& parameters) {
  if (!backend_->supports(BackendFeature::resolved_flow)) {
    throw std::runtime_error("selected backend does not implement resolved flow");
  }

  return backend_->solve_resolved_flow(spec, drag, parameters);
}

CellSnapshot Simulation::cell(CellId id) const {
  return state_.cell(id);
}

std::vector<CellSnapshot> Simulation::cells() const {
  return state_.cells();
}

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
      .culture = culture_ ? std::optional<CultureCheckpoint>(culture_->checkpoint()) : std::nullopt,
  };
  result.validate();

  return result;
}

void Simulation::validate() const {
  if (culture_) {
    culture_->checkpoint().validate(state_.checkpoint());
  }

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
