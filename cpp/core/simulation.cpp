#include "cm/simulation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <tuple>

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

void Simulation::apply_flow_drift(float dt, const MechanicsIntegrationParameters& parameters) {
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
  struct DriftUpdate {
    Slot slot;
    Vec3 position;
    Vec3 direction;
    float length;
  };
  std::vector<DriftUpdate> updates;
  updates.reserve(geometry.size());
  for (std::size_t slot = 0; slot < geometry.size(); ++slot) {
    if (attributes.fixed[slot] != 0) continue;
    Vec3 position{geometry.position_x[slot], geometry.position_y[slot], geometry.position_z[slot]};
    Vec3 direction{geometry.direction_x[slot], geometry.direction_y[slot],
                   geometry.direction_z[slot]};
    const float aspect =
        (geometry.lengths[slot] + 2 * geometry.radii[slot]) / (2 * geometry.radii[slot]);
    const float lambda = (aspect * aspect - 1) / (aspect * aspect + 1);
    const auto sample = [&](Vec3 point) {
      return signal_grid_->sample_velocity(point, GridSampleBound::clamped);
    };
    const auto derivative = [&](Vec3 point, Vec3 axis) {
      // Central differences of the interpolated fluid velocity. This is an
      // equivalent-spheroid Jeffery closure, not cell-resolved hydrodynamics.
      std::array<Vec3, 3> gradient{};
      for (std::size_t k = 0; k < 3; ++k) {
        if (dims[k] == 1) continue;
        Vec3 offset{};
        if (k == 0) offset.x = h[k] * 0.5F;
        if (k == 1) offset.y = h[k] * 0.5F;
        if (k == 2) offset.z = h[k] * 0.5F;
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
    };
    double remaining = dt;
    std::uint32_t steps = 0;
    while (remaining > 0) {
      if (++steps > 100000)
        throw std::runtime_error("flow drift needs too many substeps; reduce dt");
      const auto [velocity, orientation] = derivative(position, direction);
      float step = static_cast<float>(remaining);
      if (norm(velocity) > 0) step = std::min(step, spatial_step / norm(velocity));
      if (norm(orientation) > 0)
        step = std::min(step, parameters.max_rotation_radians / norm(orientation));
      Vec3 mid_velocity, mid_orientation;
      while (true) {
        const auto midpoint = position + velocity * (step * 0.5F);
        const auto mid_direction = normalized(direction + orientation * (step * 0.5F));
        std::tie(mid_velocity, mid_orientation) = derivative(midpoint, mid_direction);
        if (step * norm(mid_velocity) <= spatial_step * 1.001F &&
            (parameters.max_rotation_radians == 0 ||
             step * norm(mid_orientation) <= parameters.max_rotation_radians * 1.001F))
          break;
        step *= 0.5F;
        if (step <= 0) throw std::runtime_error("flow drift substep underflow");
      }
      if (!std::isfinite(step) || step <= 0) throw std::runtime_error("invalid flow drift substep");
      position = position + mid_velocity * step;
      direction = normalized(direction + mid_orientation * step);
      remaining = std::max(0.0, remaining - step);
    }
    if (!std::isfinite(position.x) || !std::isfinite(position.y) || !std::isfinite(position.z))
      throw std::runtime_error("flow drift produced non-finite geometry");
    updates.push_back({static_cast<Slot>(slot), position, direction, geometry.lengths[slot]});
  }
  for (const auto& update : updates) {
    state_.set_cell_geometry(update.slot, update.position, update.direction, update.length);
  }
}

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

void Simulation::set_signal_reaction(std::optional<SignalGridAffineReaction> reaction) {
  if (!signal_grid_.has_value()) {
    throw std::logic_error("simulation does not have a signal grid");
  }
  signal_grid_->set_reaction(std::move(reaction));
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
    if (signal_grid_.has_value()) signal_grid_->set_levels(saved_levels);
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
