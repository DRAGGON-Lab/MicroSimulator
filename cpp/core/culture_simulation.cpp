#include <map>
#include <set>

#include "capsule_bodies.hpp"
#include "culture_simulation_internal.hpp"

namespace cm {
namespace {
void require(bool ok, const char* message) {
  if (!ok) {
    throw std::invalid_argument(message);
  }
}

bool nonnegative(double x) {
  return std::isfinite(x) && x >= 0;
}

bool positive(double x) {
  return std::isfinite(x) && x > 0;
}

std::vector<CapsuleBody> bodies(const std::vector<CultureCellState>& cells) {
  std::vector<CapsuleBody> result;

  for (const auto& cell : cells) {
    result.push_back(cell.body);
  }

  return result;
}

std::array<double, 4> orientation(Vec3 direction) {
  const double norm =
      std::sqrt(double(direction.x) * direction.x + double(direction.y) * direction.y +
                double(direction.z) * direction.z);
  const double x = direction.x / norm, y = direction.y / norm, z = direction.z / norm;

  if (x < -1 + 1e-14) {
    return {0, 0, 1, 0};
  }

  std::array<double, 4> q{1 + x, 0, -z, y};
  double qnorm = 0;

  for (double a : q) {
    qnorm += a * a;
  }

  for (double& a : q) {
    a /= std::sqrt(qnorm);
  }

  return q;
}

std::vector<float> levels(const CultureCellState& cell) {
  std::vector<float> c;

  for (double amount : cell.species_amounts) {
    const float value = static_cast<float>(amount / cell.biochemical_volume);
    require(std::isfinite(value) && value >= 0,
            "culture intracellular concentration is not representable");
    c.push_back(value);
  }

  return c;
}

void sync(WorldState& world, const CultureCellState& cell) {
  const auto axis = detail::body_rotate(cell.body.orientation, {1, 0, 0});
  world.set_cell_geometry(
      cell.body.id,
      {float(cell.body.position[0]), float(cell.body.position[1]), float(cell.body.position[2])},
      {float(axis[0]), float(axis[1]), float(axis[2])}, float(cell.body.length));
  world.set_species(cell.body.id, levels(cell));
}
}  // namespace

namespace {
void validate_culture_events(const CultureConfiguration& config, std::size_t species_count) {
  double previous = 0;

  for (const auto& e : config.events) {
    require(std::isfinite(e.time) && e.time > previous,
            "culture events must be strictly increasing and positive");
    auto candidate = config;
    candidate.events.clear();
    candidate.ports = e.ports;
    candidate.reservoirs = e.reservoirs;
    candidate.validate(species_count);
    require(e.ports.size() == config.ports.size(), "event must preserve hydraulic ports");

    for (std::size_t i = 0; i < config.ports.size(); ++i) {
      require(e.ports[i].name == config.ports[i].name && e.ports[i].axis == config.ports[i].axis &&
                  e.ports[i].upper == config.ports[i].upper &&
                  e.ports[i].sites == config.ports[i].sites,
              "event cannot change port geometry");
    }

    previous = e.time;
  }
}

void validate_reserve_configuration(const CultureConfiguration& config, std::size_t species_count) {
  std::set<std::uint32_t> requirements;

  for (const auto& r : config.biomass_requirements) {
    require(r.species < species_count && positive(r.amount_per_biomass) &&
                requirements.insert(r.species).second,
            "invalid or duplicate culture biomass requirement");
  }

  std::set<std::pair<CellId, std::uint32_t>> unique_exchange;

  for (const auto& e : config.exchange) {
    require(e.body_id != 0 && e.solute < config.solutes.size() && e.species < species_count &&
                nonnegative(e.uptake_velocity) && nonnegative(e.secretion_rate),
            "invalid culture cell exchange");
    require(unique_exchange.insert({e.body_id, e.solute}).second,
            "duplicate body/solute culture exchange");
  }
}

void validate_chemical_boundaries(const CultureConfiguration& config) {
  std::set<std::string> names;

  for (const auto& s : config.solutes) {
    require(!s.name.empty() && names.insert(s.name).second && nonnegative(s.diffusion),
            "invalid culture solute");
  }

  names.clear();

  for (const auto& port : config.ports) {
    require(!port.name.empty() && names.insert(port.name).second && std::isfinite(port.value),
            "invalid culture port");
  }

  std::set<std::string> supplied;

  for (const auto& reservoir : config.reservoirs) {
    require(names.contains(reservoir.port) && supplied.insert(reservoir.port).second &&
                reservoir.concentrations.size() == config.solutes.size(),
            "invalid culture reservoir");

    for (double c : reservoir.concentrations) {
      require(nonnegative(c), "invalid culture reservoir concentration");
    }
  }

  require(config.solutes.empty() || supplied.size() == config.ports.size(),
          "every culture port needs reservoir concentrations");
}

void validate_culture_cell(const CultureCellState& cell, const CellSnapshot& physical,
                           std::size_t species_count, std::set<CellId>& ids) {
  cell.body.validate();
  require(cell.body.id == physical.id && ids.insert(cell.body.id).second &&
              positive(cell.biochemical_volume) && positive(float(cell.biochemical_volume)) &&
              cell.species_amounts.size() == species_count,
          "invalid culture cell state");
  require(float(cell.body.length) == physical.length &&
              float(cell.body.radius) == physical.radius &&
              float(cell.body.position[0]) == physical.position.x &&
              float(cell.body.position[1]) == physical.position.y &&
              float(cell.body.position[2]) == physical.position.z,
          "culture pose differs from world geometry");

  for (double m : cell.species_amounts) {
    require(nonnegative(m), "invalid culture intracellular amount");
  }

  require(levels(cell) == physical.species,
          "culture intracellular amounts differ from world concentrations");
}

void validate_culture_ledgers(const CultureCheckpoint& checkpoint) {
  for (const auto& cell : checkpoint.cells) {
    require(cell.uptake_totals.size() == checkpoint.configuration.solutes.size() &&
                nonnegative(cell.realized_specific_rate) && nonnegative(cell.biomass_produced),
            "invalid culture growth ledger");

    for (double amount : cell.uptake_totals) {
      require(nonnegative(amount), "invalid culture uptake total");
    }
  }

  for (double m : checkpoint.extracellular_amounts) {
    require(nonnegative(m), "invalid culture extracellular amount");
  }

  require(checkpoint.reservoir_totals.size() == checkpoint.configuration.ports.size(),
          "culture reservoir ledger size mismatch");

  for (std::size_t p = 0; p < checkpoint.reservoir_totals.size(); ++p) {
    const auto& t = checkpoint.reservoir_totals[p];
    require(t.port == checkpoint.configuration.ports[p].name && t.body_id == 0 &&
                t.amounts.size() == checkpoint.configuration.solutes.size(),
            "invalid culture reservoir ledger");

    for (double m : t.amounts) {
      require(std::isfinite(m), "nonfinite culture reservoir ledger");
    }
  }
}

}  // namespace

void CultureConfiguration::validate(std::size_t species_count) const {
  grid.validate();
  fluid.validate();
  solver.validate();
  stepping.validate();
  geometry.validate();
  require(positive(biomass_per_geometric_volume) && maximum_substeps > 0 && maximum_retries <= 64,
          "invalid culture integration parameters");
  require(positive(coupling_tolerance) && coupling_tolerance < 0.01 &&
              maximum_coupling_iterations > 0 && maximum_coupling_iterations <= 10000,
          "invalid culture coupling parameters");
  std::set<CellId> modeled;

  for (const auto& m : growth) {
    m.validate(solutes.size());
    require(modeled.insert(m.cell_id).second, "duplicate growth binding");
  }

  require(growth.empty() || (exchange.empty() && biomass_requirements.empty()),
          "kinetic growth cannot also use reserve-budget exchanges");
  validate_culture_events(*this, species_count);

  validate_reserve_configuration(*this, species_count);

  validate_chemical_boundaries(*this);
}

void CultureCheckpoint::validate(const WorldStateCheckpoint& world) const {
  configuration.validate(world.species_count);
  require(nonnegative(time) && event_index <= configuration.events.size(),
          "invalid culture event position");

  for (std::size_t i = 0; i < configuration.events.size(); ++i) {
    require((configuration.events[i].time <= time) == (i < event_index),
            "culture event cursor differs from time");
  }

  require(cells.size() == world.cells.size(), "culture cell count differs from world");
  std::set<CellId> ids;

  for (std::size_t i = 0; i < cells.size(); ++i) {
    validate_culture_cell(cells[i], world.cells[i], world.species_count, ids);
  }

  for (const auto& m : configuration.growth) {
    require(ids.contains(m.cell_id), "growth model refers to absent cell");
    const auto found = std::find_if(world.cells.begin(), world.cells.end(), [&](const auto& c) {
      return c.id == m.cell_id;
    });
    require(found->growth_rate == 0, "kinetic growth owns the cell growth rate");
  }

  for (const auto& e : configuration.exchange) {
    require(ids.contains(e.body_id), "culture exchange refers to an absent cell");
  }

  validate_culture_ledgers(*this);
}

namespace detail {
CultureSimulation::CultureSimulation(const WorldState& world,
                                     const CultureConfiguration& configuration,
                                     const std::vector<double>& concentrations,
                                     const std::vector<double>& biochemical_volumes,
                                     const BackendInfo& backend)
    : flow_(backend.kind, backend.device_index),
      transport_(backend.kind, backend.device_index),
      growth_(backend.kind, backend.device_index) {
  configuration.validate(world.species_count());
  data_.configuration = configuration;
  require(biochemical_volumes.empty() || biochemical_volumes.size() == world.size(),
          "biochemical volume count mismatch");
  require(concentrations.size() == configuration.solutes.size(),
          "initial culture concentration count mismatch");

  for (double c : concentrations) {
    require(nonnegative(c), "invalid initial culture concentration");
  }

  for (const auto& cell : world.cells()) {
    CultureCellState value;
    value.body.id = cell.id;
    value.body.position = {cell.position.x, cell.position.y, cell.position.z};
    value.body.orientation = orientation(cell.direction);
    value.body.length = cell.length;
    value.body.radius = cell.radius;
    value.body.fixed = cell.fixed;
    value.biochemical_volume =
        biochemical_volumes.empty()
            ? configuration.biomass_per_geometric_volume * value.body.geometric_volume()
            : biochemical_volumes[cell.slot];
    require(positive(value.biochemical_volume), "invalid initial biochemical volume");

    for (float c : cell.species) {
      require(nonnegative(c), "culture intracellular species must be nonnegative");
      value.species_amounts.push_back(c * value.biochemical_volume);
    }

    for (const auto& m : configuration.growth) {
      if (m.cell_id == cell.id && biochemical_volumes.empty()) {
        value.biochemical_volume = m.volume_ratio * value.body.geometric_volume();
      }
    }

    value.species_amounts.clear();

    for (float c : cell.species) {
      value.species_amounts.push_back(c * value.biochemical_volume);
    }

    value.uptake_totals.resize(configuration.solutes.size());
    data_.cells.push_back(std::move(value));
  }

  geometry_ = std::make_shared<FluidGeometry>(configuration.grid, bodies(data_.cells),
                                              configuration.geometry);

  for (const auto& f : geometry_->fragments()) {
    for (double c : concentrations) {
      data_.extracellular_amounts.push_back(c * f.volume);
    }
  }

  for (const auto& p : configuration.ports) {
    data_.reservoir_totals.push_back({p.name, 0, std::vector<double>(concentrations.size())});
  }

  data_.last_report.flow =
      flow_
          .solve_bodies(configuration.grid, configuration.fluid, configuration.ports,
                        bodies(data_.cells), configuration.solver)
          .report;
  data_.validate(world.checkpoint());
}

CultureSimulation::CultureSimulation(const WorldState& world, const CultureCheckpoint& checkpoint,
                                     const BackendInfo& backend)
    : data_(checkpoint),
      flow_(backend.kind, backend.device_index),
      transport_(backend.kind, backend.device_index),
      growth_(backend.kind, backend.device_index) {
  data_.validate(world.checkpoint());
  geometry_ = std::make_shared<FluidGeometry>(data_.configuration.grid, bodies(data_.cells),
                                              data_.configuration.geometry);
  require(data_.extracellular_amounts.size() ==
              geometry_->fragments().size() * data_.configuration.solutes.size(),
          "culture checkpoint fragment amount count mismatch");
}

void CultureSimulation::set_species(CellId id, std::span<const float> values) {
  auto found = std::find_if(data_.cells.begin(), data_.cells.end(), [&](const auto& c) {
    return c.body.id == id;
  });
  require(found != data_.cells.end() && values.size() == found->species_amounts.size(),
          "invalid culture cell species update");
  std::vector<double> amounts;

  for (float c : values) {
    require(nonnegative(c), "culture cell species must be nonnegative");
    amounts.push_back(c * found->biochemical_volume);
  }

  found->species_amounts = std::move(amounts);
}

void CultureSimulation::set_force(CellId id, std::array<double, 3> force,
                                  std::array<double, 3> torque) {
  auto found = std::find_if(data_.cells.begin(), data_.cells.end(), [&](const auto& c) {
    return c.body.id == id;
  });
  require(found != data_.cells.end(), "unknown culture cell id");
  auto candidate = found->body;
  candidate.force_n = force;
  candidate.torque_nm = torque;
  candidate.validate();
  found->body = candidate;
}

namespace {
void prepare_reserve_growth(std::vector<CultureCellState>& trial, const WorldState& trial_world,
                            const CultureConfiguration& config, double maximum_dt) {
  for (auto& cell : trial) {
    const auto attributes = trial_world.cell(cell.body.id);
    require(attributes.growth_rate >= 0, "culture growth rate must be nonnegative");
    cell.body.fixed = attributes.fixed;
    cell.body.length_rate = attributes.growth_rate * cell.body.length;
    double available = INFINITY;

    for (const auto& requirement : config.biomass_requirements) {
      available = std::min(available, (1 - 16 * std::numeric_limits<double>::epsilon()) *
                                          cell.species_amounts[requirement.species] /
                                          requirement.amount_per_biomass);
    }

    const double area = std::numbers::pi * cell.body.radius * cell.body.radius *
                        config.biomass_per_geometric_volume;
    cell.body.length_rate = std::min(cell.body.length_rate, available / (area * maximum_dt));
  }
}

void apply_reserve_growth(std::vector<CultureCellState>& trial, WorldState& trial_world,
                          const CultureConfiguration& config, const FluidBodyStep& proposal,
                          std::vector<float>& old_lengths, std::vector<float>& old_volumes,
                          std::vector<float>& new_volumes) {
  for (std::size_t i = 0; i < trial.size(); ++i) {
    auto& cell = trial[i];
    const double old_b = cell.biochemical_volume;
    const double increment = config.biomass_per_geometric_volume * std::numbers::pi *
                             cell.body.radius * cell.body.radius *
                             (proposal.bodies[i].length - cell.body.length);

    for (const auto& requirement : config.biomass_requirements) {
      cell.species_amounts[requirement.species] -= requirement.amount_per_biomass * increment;

      if (cell.species_amounts[requirement.species] < 0) {
        throw std::runtime_error("culture growth exceeded intracellular amount budget");
      }
    }

    // Species kinetics sees the true geometric pose and separate old/new
    // biochemical volumes; no geometry is altered to encode biomass.
    trial_world.set_species(cell.body.id, levels(cell));
    old_lengths.push_back(float(cell.body.length));
    old_volumes.push_back(float(old_b));
    cell.body = proposal.bodies[i];
    cell.biochemical_volume += increment;
    new_volumes.push_back(float(cell.biochemical_volume));
    const auto axis = body_rotate(cell.body.orientation, {1, 0, 0});
    trial_world.set_cell_geometry(
        cell.body.id,
        {float(cell.body.position[0]), float(cell.body.position[1]), float(cell.body.position[2])},
        {float(axis[0]), float(axis[1]), float(axis[2])}, float(cell.body.length));
  }
}

std::vector<SurfaceTransferLaw> reserve_exchange_laws(
    const CultureConfiguration& config, const std::vector<CultureCellState>& trial,
    const std::map<CellId, std::size_t>& cell_index, double step_dt) {
  std::map<std::pair<CellId, std::uint32_t>, double> requested_secretion;

  for (const auto& e : config.exchange) {
    requested_secretion[{e.body_id, e.species}] += e.secretion_rate;
  }

  std::vector<SurfaceTransferLaw> exchange;

  // Each body/solute pair has one intracellular destination, making the
  // transfer ledger unambiguous even when several species are present.
  for (const auto& e : config.exchange) {
    const auto& cell = trial.at(cell_index.at(e.body_id));
    const double demand = requested_secretion.at({e.body_id, e.species}) * step_dt;
    const double fraction =
        demand == 0 ? 1 : std::min(1.0, cell.species_amounts[e.species] / demand);
    exchange.push_back({e.body_id, e.solute, e.uptake_velocity, e.secretion_rate * fraction});
  }

  return exchange;
}

void apply_reserve_transfers(const CultureConfiguration& config,
                             std::vector<CultureCellState>& trial,
                             const std::map<CellId, std::size_t>& cell_index,
                             const SoluteTransportResult& transported) {
  for (const auto& transfer : transported.cells) {
    for (const auto& e : config.exchange) {
      if (e.body_id == transfer.body_id) {
        auto& m = trial[cell_index.at(e.body_id)].species_amounts[e.species];
        m += transfer.amounts[e.solute];

        if (!nonnegative(m)) {
          throw std::runtime_error("culture secretion exceeded intracellular amount budget");
        }
      }
    }
  }
}

void sample_growth_inputs(const CultureCheckpoint& staged, const FluidGeometry& old_geometry,
                          const std::map<CellId, std::size_t>& slots,
                          std::vector<GrowthInput>& input) {
  const auto& config = staged.configuration;

  for (std::size_t k = 0; k < config.growth.size(); ++k) {
    const auto& model = config.growth[k];
    auto& in = input[k];
    in.biochemical_volume = staged.cells[slots.at(model.cell_id)].biochemical_volume;
    in.concentrations.resize(model.requirements.size());
    in.uptake.resize(model.requirements.size());

    for (const auto& f : old_geometry.faces()) {
      if (f.body_id == model.cell_id) {
        in.surface_area += f.area;

        for (std::size_t j = 0; j < model.requirements.size(); ++j) {
          in.concentrations[j] += f.area *
                                  staged.extracellular_amounts[f.first * config.solutes.size() +
                                                               model.requirements[j].solute] /
                                  old_geometry.fragments()[f.first].volume;
        }
      }
    }

    require(in.surface_area > 0, "growth cell has no accessible surface");

    for (auto& c : in.concentrations) {
      c /= in.surface_area;
    }
  }
}

void average_growth_surfaces(const CultureConfiguration& config, const FluidGeometry& old_geometry,
                             const FluidGeometry& geometry, std::vector<GrowthInput>& input) {
  for (std::size_t k = 0; k < input.size(); ++k) {
    input[k].surface_area = 0;

    for (const auto* mesh : {&old_geometry, &geometry}) {
      for (const auto& f : mesh->faces()) {
        if (f.body_id == config.growth[k].cell_id) {
          input[k].surface_area += 0.5 * f.area;
        }
      }
    }
  }
}

std::vector<SurfaceTransferLaw> kinetic_exchange_laws(
    const CultureConfiguration& config, const std::vector<GrowthEvaluation>& linearized) {
  std::vector<SurfaceTransferLaw> exchange;

  for (std::size_t k = 0; k < config.growth.size(); ++k) {
    for (std::size_t j = 0; j < config.growth[k].requirements.size(); ++j) {
      exchange.push_back({config.growth[k].cell_id, config.growth[k].requirements[j].solute,
                          linearized[k].uptake_velocities[j], 0});
    }
  }

  return exchange;
}

std::map<CellId, std::size_t> update_growth_inputs(const CultureConfiguration& config,
                                                   const SoluteTransportResult& transported,
                                                   std::vector<GrowthInput>& input) {
  std::map<CellId, std::size_t> transfer_index;

  for (std::size_t i = 0; i < transported.cells.size(); ++i) {
    transfer_index[transported.cells[i].body_id] = i;
  }

  for (std::size_t k = 0; k < input.size(); ++k) {
    const auto& model = config.growth[k];
    const auto index = transfer_index.at(model.cell_id);

    for (std::size_t j = 0; j < model.requirements.size(); ++j) {
      const auto s = model.requirements[j].solute;
      input[k].concentrations[j] = transported.surfaces[index].concentrations[s];
      input[k].uptake[j] = transported.cells[index].amounts[s];
    }
  }

  return transfer_index;
}

void apply_growth_extents(const CultureConfiguration& config, const FluidGeometry& geometry,
                          std::vector<CultureCellState>& trial,
                          const std::map<CellId, std::size_t>& slots,
                          const std::map<CellId, std::size_t>& transfer_index,
                          const std::vector<GrowthInput>& input, const std::vector<double>& extents,
                          SoluteTransportResult& transported, double dt) {
  // Enforce the shared reaction extent exactly. Only the converged solver
  // residual/working-precision excess is returned, locally at that surface.
  // No unmatched substrate is lost or silently stored inside a cell.
  for (std::size_t k = 0; k < input.size(); ++k) {
    const auto& model = config.growth[k];
    auto& cell = trial[slots.at(model.cell_id)];
    const auto ti = transfer_index.at(model.cell_id);
    double area = 0;

    for (const auto& face : geometry.faces()) {
      if (face.body_id == model.cell_id) {
        area += face.area;
      }
    }

    require(area > 0, "growth cell has no final accessible surface");

    for (std::size_t j = 0; j < model.requirements.size(); ++j) {
      const auto& requirement = model.requirements[j];
      const auto s = requirement.solute;
      const double consumed = extents[k] / requirement.biomass_yield;
      const double returned = std::max(0.0, transported.cells[ti].amounts[s] - consumed);

      for (const auto& face : geometry.faces()) {
        if (face.body_id == model.cell_id) {
          transported.amounts[face.first * config.solutes.size() + s] +=
              returned * face.area / area;
        }
      }

      transported.cells[ti].amounts[s] = consumed;
      cell.uptake_totals[s] += consumed;
    }

    cell.biochemical_volume += extents[k] / model.biomass_density;
    cell.biomass_produced += extents[k];
    cell.realized_specific_rate =
        extents[k] / (input[k].biochemical_volume * model.biomass_density * dt);
  }
}

WorldState advance_kinetic_species(const WorldState& world, const CultureCheckpoint& staged,
                                   std::vector<CultureCellState>& trial, ComputeBackend& backend,
                                   const SpeciesRatePlan& plan, double dt) {
  auto candidate_world = world;
  std::vector<float> old_lengths, old_volumes, new_volumes;

  for (std::size_t i = 0; i < trial.size(); ++i) {
    old_lengths.push_back(float(staged.cells[i].body.length));
    old_volumes.push_back(float(staged.cells[i].biochemical_volume));
    new_volumes.push_back(float(trial[i].biochemical_volume));
    // Kinetics receives old concentration and explicit old/new biomass.
    candidate_world.set_species(trial[i].body.id, levels(staged.cells[i]));
    const auto axis = body_rotate(trial[i].body.orientation, {1, 0, 0});
    candidate_world.set_cell_geometry(
        trial[i].body.id,
        {float(trial[i].body.position[0]), float(trial[i].body.position[1]),
         float(trial[i].body.position[2])},
        {float(axis[0]), float(axis[1]), float(axis[2])}, float(trial[i].body.length));
  }

  const bool zero = std::all_of(plan.outputs().begin(), plan.outputs().end(), [&](auto o) {
    return plan.instructions()[o].operation == RateOp::constant &&
           plan.instructions()[o].value == 0;
  });

  if (!zero) {
    backend.advance_species(candidate_world, plan, old_lengths, float(dt),
                            {old_volumes, new_volumes});

    for (auto& cell : trial) {
      const auto snapshot = candidate_world.cell(cell.body.id);

      for (std::size_t s = 0; s < cell.species_amounts.size(); ++s) {
        cell.species_amounts[s] = snapshot.species[s] * cell.biochemical_volume;

        if (!nonnegative(cell.species_amounts[s])) {
          throw std::runtime_error("invalid intracellular reaction amount");
        }
      }
    }
  }

  for (const auto& cell : trial) {
    sync(candidate_world, cell);
  }

  return candidate_world;
}

void inherit_growth_models(CultureConfiguration& config, CellId parent, CellId first,
                           CellId second) {
  std::vector<CellGrowthModel> daughters;

  for (auto& model : config.growth) {
    if (model.cell_id == parent) {
      auto other = model;
      model.cell_id = first;
      other.cell_id = second;
      daughters.push_back(other);
    }
  }

  config.growth.insert(config.growth.end(), daughters.begin(), daughters.end());
}

void inherit_exchange_laws(CultureConfiguration& config, CellId parent, CellId first, CellId second,
                           double fraction) {
  std::vector<CellSurfaceExchange> inherited;

  for (auto& e : config.exchange) {
    if (e.body_id == parent) {
      auto other = e;
      e.body_id = first;
      other.body_id = second;
      e.secretion_rate *= fraction;
      other.secretion_rate -= e.secretion_rate;
      inherited.push_back(other);
    }
  }

  config.exchange.insert(config.exchange.end(), inherited.begin(), inherited.end());
}

}  // namespace

double CultureSimulation::reserve_step(CultureCheckpoint& staged,
                                       std::shared_ptr<FluidGeometry>& staged_geometry,
                                       WorldState& staged_world, ComputeBackend& backend,
                                       const SpeciesRatePlan& plan, double maximum_dt,
                                       double remaining, bool zero_plan) {
  const auto& config = staged.configuration;
  auto trial = staged.cells;
  auto trial_world = staged_world;

  prepare_reserve_growth(trial, trial_world, config, maximum_dt);

  const auto proposal =
      flow_.propose_body_step(config.grid, config.fluid, config.ports, bodies(trial), maximum_dt,
                              config.solver, config.stepping);
  const double step_dt = proposal.accepted_dt;
  require(step_dt > 0 && step_dt <= remaining && remaining - step_dt < remaining,
          "culture step cannot advance time");
  std::vector<float> old_lengths, old_volumes, new_volumes;

  apply_reserve_growth(trial, trial_world, config, proposal, old_lengths, old_volumes, new_volumes);

  if (!zero_plan) {
    backend.advance_species(trial_world, plan, old_lengths, float(step_dt),
                            {old_volumes, new_volumes});

    for (auto& cell : trial) {
      const auto snapshot = trial_world.cell(cell.body.id);

      for (std::size_t s = 0; s < cell.species_amounts.size(); ++s) {
        const double m = snapshot.species[s] * cell.biochemical_volume;

        if (!nonnegative(m)) {
          throw std::runtime_error("culture intracellular reaction produced an invalid amount");
        }

        cell.species_amounts[s] = m;
      }
    }
  }

  auto new_geometry = std::make_shared<FluidGeometry>(config.grid, bodies(trial), config.geometry);
  std::map<CellId, std::size_t> cell_index;

  for (std::size_t i = 0; i < trial.size(); ++i) {
    cell_index[trial[i].body.id] = i;
  }

  const auto exchange = reserve_exchange_laws(config, trial, cell_index, step_dt);

  const auto transported = transport_.step(
      *staged_geometry, *new_geometry, proposal.flow.field, config.ports, config.solutes,
      config.reservoirs, staged.extracellular_amounts, step_dt, exchange, config.solver);

  apply_reserve_transfers(config, trial, cell_index, transported);

  for (const auto& cell : trial) {
    sync(trial_world, cell);
  }

  staged.cells = std::move(trial);
  staged_world = std::move(trial_world);
  staged_geometry = std::move(new_geometry);
  staged.extracellular_amounts = transported.amounts;

  for (std::size_t p = 0; p < transported.reservoirs.size(); ++p) {
    for (std::size_t s = 0; s < config.solutes.size(); ++s) {
      staged.reservoir_totals[p].amounts[s] += transported.reservoirs[p].amounts[s];
    }
  }

  staged.last_report.flow = proposal.flow.report;
  staged.last_report.transport = transported.report;

  return step_dt;
}

namespace {
void apply_culture_events(CultureCheckpoint& staged) {
  const auto& config = staged.configuration;

  while (staged.event_index < config.events.size() &&
         config.events[staged.event_index].time <= staged.time) {
    const auto& e = config.events[staged.event_index++];
    staged.configuration.ports = e.ports;
    staged.configuration.reservoirs = e.reservoirs;
  }
}
}  // namespace

void CultureSimulation::step(WorldState& world, ComputeBackend& backend,
                             const SpeciesRatePlan& plan, double dt) {
  require(nonnegative(dt), "invalid culture step interval");

  if (dt == 0) {
    return;
  }

  auto staged = data_;
  auto staged_geometry = geometry_;
  auto staged_world = world;
  const auto& config = staged.configuration;
  staged.last_report = {};
  double remaining = dt;
  const bool zero_plan =
      std::all_of(plan.outputs().begin(), plan.outputs().end(), [&](auto output) {
        const auto& instruction = plan.instructions()[output];

        return instruction.operation == RateOp::constant && instruction.value == 0;
      });

  while (remaining > 0) {
    if (staged.last_report.substeps >= config.maximum_substeps) {
      throw std::runtime_error("culture maximum_substeps exhausted; entire step rejected");
    }

    apply_culture_events(staged);

    double maximum_dt = remaining;

    if (staged.event_index < config.events.size()) {
      maximum_dt = std::min(maximum_dt, config.events[staged.event_index].time - staged.time);
    }

    bool accepted = false;

    for (std::uint32_t retry = 0; retry <= config.maximum_retries && !accepted; ++retry) {
      try {
        if (!config.growth.empty()) {
          const double advanced =
              kinetic_step(staged, staged_geometry, staged_world, backend, plan, maximum_dt);
          staged.time += advanced;
          remaining -= advanced;
          ++staged.last_report.substeps;
          accepted = true;
          continue;
        }

        const double step_dt = reserve_step(staged, staged_geometry, staged_world, backend, plan,
                                            maximum_dt, remaining, zero_plan);
        ++staged.last_report.substeps;
        remaining -= step_dt;
        staged.time += step_dt;
        accepted = true;
      } catch (const std::runtime_error&) {
        if (retry == config.maximum_retries) {
          throw;
        }

        maximum_dt *= 0.5;
        ++staged.last_report.retries;
      }
    }
  }

  apply_culture_events(staged);

  staged.validate(staged_world.checkpoint());

  for (std::size_t i = 0; i < staged.cells.size(); ++i) {
    if (std::any_of(config.growth.begin(), config.growth.end(), [&](const auto& m) {
          return m.cell_id == staged.cells[i].body.id;
        })) {
      staged.cells[i].realized_specific_rate =
          (staged.cells[i].biochemical_volume - data_.cells[i].biochemical_volume) /
          (data_.cells[i].biochemical_volume * dt);
    }
  }

  world = std::move(staged_world);
  data_ = std::move(staged);
  geometry_ = std::move(staged_geometry);
}

double CultureSimulation::kinetic_step(CultureCheckpoint& staged,
                                       std::shared_ptr<FluidGeometry>& old_geometry,
                                       WorldState& world, ComputeBackend& backend,
                                       const SpeciesRatePlan& plan, double dt) {
  const auto& config = staged.configuration;
  std::map<CellId, std::size_t> slots;

  for (std::size_t i = 0; i < staged.cells.size(); ++i) {
    slots[staged.cells[i].body.id] = i;
    require(world.cell(staged.cells[i].body.id).growth_rate == 0,
            "kinetic culture requires growth models instead of prescribed elongation");
  }

  std::vector<GrowthInput> input(config.growth.size());
  std::vector<double> rates(staged.cells.size());

  sample_growth_inputs(staged, *old_geometry, slots, input);

  double previous_residual = INFINITY;

  for (std::uint32_t iteration = 0; iteration < config.maximum_coupling_iterations; ++iteration) {
    auto trial = staged.cells;

    for (std::size_t i = 0; i < trial.size(); ++i) {
      trial[i].body.fixed = world.cell(trial[i].body.id).fixed;
      trial[i].body.length_rate = rates[i];
    }

    const auto proposal = flow_.propose_body_step(
        config.grid, config.fluid, config.ports, bodies(trial), dt, config.solver, config.stepping);

    if (proposal.accepted_dt < dt) {
      dt = proposal.accepted_dt;
      require(dt > 0, "culture cannot advance time");
      std::fill(rates.begin(), rates.end(), 0);
      continue;
    }

    for (std::size_t i = 0; i < trial.size(); ++i) {
      trial[i].body = proposal.bodies[i];
    }

    auto geometry = std::make_shared<FluidGeometry>(config.grid, bodies(trial), config.geometry);

    average_growth_surfaces(config, *old_geometry, *geometry, input);

    const auto linearized = growth_.evaluate(config.growth, input, dt);
    const auto exchange = kinetic_exchange_laws(config, linearized);

    auto chemical_parameters = config.solver;
    chemical_parameters.relative_tolerance =
        std::min(chemical_parameters.relative_tolerance, 0.01 * config.coupling_tolerance);
    chemical_parameters.absolute_tolerance = 0;
    auto transported = transport_.step(
        *old_geometry, *geometry, proposal.flow.field, config.ports, config.solutes,
        config.reservoirs, staged.extracellular_amounts, dt, exchange, chemical_parameters);
    const auto transfer_index = update_growth_inputs(config, transported, input);

    const auto realized = growth_.evaluate(config.growth, input, dt);
    std::vector<double> extents(input.size()), next_rates = rates;
    double residual = 0;

    for (std::size_t k = 0; k < input.size(); ++k) {
      const auto& model = config.growth[k];
      const auto index = slots.at(model.cell_id);
      // Bound GPU roundoff by the actual double-precision chemical ledger.
      double extent = realized[k].biomass_gain;

      for (std::size_t j = 0; j < model.requirements.size(); ++j) {
        extent = std::min(extent, input[k].uptake[j] * model.requirements[j].biomass_yield);
      }

      extents[k] = extent;
      next_rates[index] =
          extent / (model.biomass_density * model.volume_ratio * dt * std::numbers::pi *
                    trial[index].body.radius * trial[index].body.radius);
      residual = std::max(residual, realized[k].stoichiometric_residual);
      const double rate_scale =
          std::max({std::abs(next_rates[index]), std::abs(rates[index]), 1e-30});
      residual = std::max(residual, std::abs(next_rates[index] - rates[index]) / rate_scale);

      for (std::size_t j = 0; j < model.requirements.size(); ++j) {
        const double a = linearized[k].uptake_velocities[j], b = realized[k].uptake_velocities[j];
        residual = std::max(residual, std::abs(a - b) / std::max({a, b, 1e-30}));
      }
    }

    if (residual <= config.coupling_tolerance) {
      apply_growth_extents(config, *geometry, trial, slots, transfer_index, input, extents,
                           transported, dt);

      auto candidate_world = advance_kinetic_species(world, staged, trial, backend, plan, dt);

      staged.cells = std::move(trial);
      staged.extracellular_amounts = std::move(transported.amounts);

      for (std::size_t p = 0; p < transported.reservoirs.size(); ++p) {
        for (std::size_t s = 0; s < config.solutes.size(); ++s) {
          staged.reservoir_totals[p].amounts[s] += transported.reservoirs[p].amounts[s];
        }
      }

      staged.last_report.flow = proposal.flow.report;
      staged.last_report.transport = transported.report;
      world = std::move(candidate_world);
      old_geometry = std::move(geometry);

      return dt;
    }

    const double damping = residual > previous_residual ? 0.5 : 1.0;

    for (std::size_t i = 0; i < rates.size(); ++i) {
      rates[i] += damping * (next_rates[i] - rates[i]);
    }

    previous_residual = residual;
  }

  throw std::runtime_error("culture uptake/growth/geometry iteration did not converge");
}

std::pair<CellId, CellId> CultureSimulation::divide(WorldState& world, CellId parent,
                                                    double fraction) {
  require(std::isfinite(fraction) && fraction > 0 && fraction < 1,
          "invalid culture division fraction");
  auto data = data_;
  auto checkpoint = world.checkpoint();
  auto found = std::find_if(data.cells.begin(), data.cells.end(), [&](const auto& c) {
    return c.body.id == parent;
  });
  require(found != data.cells.end(), "unknown culture division parent");
  const auto index = std::size_t(found - data.cells.begin());
  const auto original = *found;
  auto first = original, second = original;
  const double cap = 4 * original.body.radius / 3;
  first.body.length = fraction * (original.body.length + cap) - cap;
  second.body.length = (1 - fraction) * (original.body.length + cap) - cap;
  require(first.body.length >= 0 && second.body.length >= 0,
          "parent is too short for volume-conserving division");
  require(checkpoint.next_id < std::numeric_limits<CellId>::max() - 1,
          "culture cell identifier space exhausted");
  first.body.id = checkpoint.next_id++;
  second.body.id = checkpoint.next_id++;
  const double gap =
      data.configuration.stepping.minimum_gap_m / data.configuration.grid.length_unit_m +
      1e-5 * data.configuration.grid.spacing;
  const double separation =
      (first.body.length + second.body.length) / 2 + 2 * original.body.radius + gap;
  const auto axis = body_rotate(original.body.orientation, {1, 0, 0});

  for (unsigned d = 0; d < 3; ++d) {
    first.body.position[d] -= (1 - fraction) * separation * axis[d];
    second.body.position[d] += fraction * separation * axis[d];
    first.body.force_n[d] *= fraction;
    second.body.force_n[d] -= first.body.force_n[d];
    first.body.torque_nm[d] *= fraction;
    second.body.torque_nm[d] -= first.body.torque_nm[d];
  }

  first.biochemical_volume = original.biochemical_volume * fraction;
  second.biochemical_volume = original.biochemical_volume - first.biochemical_volume;

  for (std::size_t s = 0; s < first.species_amounts.size(); ++s) {
    first.species_amounts[s] *= fraction;
    second.species_amounts[s] -= first.species_amounts[s];
  }

  first.biomass_produced = original.biomass_produced * fraction;
  second.biomass_produced = original.biomass_produced - first.biomass_produced;

  for (std::size_t s = 0; s < first.uptake_totals.size(); ++s) {
    first.uptake_totals[s] = original.uptake_totals[s] * fraction;
    second.uptake_totals[s] = original.uptake_totals[s] - first.uptake_totals[s];
  }

  inherit_growth_models(data.configuration, parent, first.body.id, second.body.id);
  data.cells[index] = first;
  data.cells.push_back(second);
  auto new_geometry = std::make_shared<FluidGeometry>(data.configuration.grid, bodies(data.cells),
                                                      data.configuration.geometry);
  SignalGridVelocityField zero;
  const auto shape = data.configuration.grid.shape;
  zero.x_faces.resize((std::size_t(shape.x) + 1) * shape.y * shape.z);
  zero.y_faces.resize(std::size_t(shape.x) * (shape.y + 1) * shape.z);
  zero.z_faces.resize(std::size_t(shape.x) * shape.y * (shape.z + 1));
  auto tracers = data.configuration.solutes;

  for (auto& s : tracers) {
    s.diffusion = 0;
  }

  // A geometric remap has no physical duration, pressure history or reservoir
  // exchange. The unit interval below only parameterizes conservative remapping.
  const auto remap = transport_.step(*geometry_, *new_geometry, zero, {}, tracers, {},
                                     data.extracellular_amounts, 1, {}, data.configuration.solver);
  data.extracellular_amounts = remap.amounts;
  auto snapshot = checkpoint.cells[index];
  snapshot.id = first.body.id;
  snapshot.length = float(first.body.length);
  snapshot.position = {float(first.body.position[0]), float(first.body.position[1]),
                       float(first.body.position[2])};
  snapshot.species = levels(first);
  checkpoint.cells[index] = snapshot;
  snapshot.id = second.body.id;
  snapshot.slot = static_cast<Slot>(checkpoint.cells.size());
  snapshot.length = float(second.body.length);
  snapshot.position = {float(second.body.position[0]), float(second.body.position[1]),
                       float(second.body.position[2])};
  snapshot.species = levels(second);
  checkpoint.cells.push_back(snapshot);
  checkpoint.lineage.push_back({first.body.id, parent});
  checkpoint.lineage.push_back({second.body.id, parent});
  inherit_exchange_laws(data.configuration, parent, first.body.id, second.body.id, fraction);
  auto staged_world = WorldState(checkpoint);
  data.validate(checkpoint);
  world = std::move(staged_world);
  data_ = std::move(data);
  geometry_ = std::move(new_geometry);

  return {first.body.id, second.body.id};
}
}  // namespace detail
}  // namespace cm
