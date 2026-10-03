#include "cm/solute_transport.hpp"

#include <algorithm>
#include <cmath>
#include <set>

#include "numerics_device.hpp"
#include "transport_geometry_internal.hpp"

namespace cm {
namespace {
void require(bool ok, const char* message) {
  if (!ok) {
    throw std::invalid_argument(message);
  }
}
}  // namespace

struct SoluteTransportSolver::Impl {
  std::unique_ptr<detail::NumericsDevice> device;

  void validate_chemical_input(const std::vector<double>& amounts,
                               const std::vector<Solute>& solutes, std::size_t old_count) {
    const auto ns = solutes.size();
    require(amounts.size() == old_count * ns, "chemical amount dimensions mismatch");

    for (double a : amounts) {
      require(std::isfinite(a) && a >= 0, "chemical amounts must be finite and nonnegative");
    }

    std::set<std::string> names;

    for (const auto& s : solutes) {
      require(!s.name.empty() && names.insert(s.name).second && std::isfinite(s.diffusion) &&
                  s.diffusion >= 0,
              "invalid solute definition");
    }
  }

  void map_reservoirs(const std::vector<FlowPort>& ports,
                      const std::vector<ChemicalBoundary>& reservoirs, std::size_t ns,
                      std::vector<std::vector<double>>& reservoir,
                      std::vector<ChemicalBoundaryKind>& boundary_kind,
                      std::vector<bool>& backflow) {
    std::set<std::string> names;

    for (const auto& r : reservoirs) {
      const auto found = std::find_if(ports.begin(), ports.end(), [&](const auto& p) {
        return p.name == r.port;
      });
      require(found != ports.end() && names.insert(r.port).second && r.concentrations.size() == ns,
              "invalid culture reservoir");

      for (double c : r.concentrations) {
        require(std::isfinite(c) && c >= 0, "invalid reservoir concentration");
      }

      const auto index = std::size_t(found - ports.begin());
      require(r.kind == ChemicalBoundaryKind::reservoir_contact ||
                  r.kind == ChemicalBoundaryKind::advective ||
                  r.kind == ChemicalBoundaryKind::outflow,
              "invalid chemical boundary kind");
      reservoir[index] = r.concentrations;
      boundary_kind[index] = r.kind;
      backflow[index] = r.allow_backflow;
    }

    for (const auto& r : reservoir) {
      require(r.size() == ns, "missing culture reservoir concentrations");
    }
  }

  void validate_surface_exchange(const std::vector<SurfaceTransferLaw>& exchange,
                                 const std::map<CellId, double>& body_area, std::size_t ns) {
    for (const auto& e : exchange) {
      require(body_area.contains(e.body_id) && e.solute < ns && std::isfinite(e.uptake_velocity) &&
                  e.uptake_velocity >= 0 && std::isfinite(e.secretion_rate) &&
                  e.secretion_rate >= 0,
              "invalid surface transfer");
    }
  }

  detail::NumericsLinearResult solve_species(const TransportGeometry::Impl& g,
                                             const std::vector<Solute>& solutes,
                                             const std::vector<std::vector<double>>& reservoir,
                                             const std::vector<ChemicalBoundaryKind>& boundary_kind,
                                             const std::vector<SurfaceTransferLaw>& exchange,
                                             const std::vector<double>& initial, std::size_t s,
                                             double amount_scale,
                                             const LinearSolveParameters& parameters) {
    const auto n = g.group_count, ns = solutes.size();
    const double dt = g.dt, h = g.grid.spacing, vh = h * h * h;
    const auto& v1 = g.v1;
    const auto& edges = g.edges;
    const auto& membranes = g.membranes;
    const auto& body_area = g.body_area;
    detail::NumericsLinearRows system(n);
    std::vector<double> source(n);

    for (std::uint32_t i = 0; i < n; ++i) {
      system[i].emplace_back(i, v1[i] / vh);
      source[i] = initial[i * ns + s] / amount_scale;
    }

    for (const auto& e : edges) {
      const auto a = e.first, b = e.second;
      const double diffusion =
          (b == fluid_boundary && boundary_kind[e.port] != ChemicalBoundaryKind::reservoir_contact)
              ? 0
              : solutes[s].diffusion * e.area / (h * (b == fluid_boundary ? 0.5 : 1));
      const double out = dt * (diffusion + std::max(e.q, 0.0)) / vh;
      const double in = dt * (diffusion + std::max(-e.q, 0.0)) / vh;
      system[a].emplace_back(a, out);

      if (b == fluid_boundary) {
        source[a] += in * reservoir[e.port][s] * vh / amount_scale;
      } else {
        system[a].emplace_back(b, -in);
        system[b].emplace_back(a, -out);
        system[b].emplace_back(b, in);
      }
    }

    for (const auto& membrane : membranes) {
      for (const auto& e : exchange) {
        if (e.solute == s && e.body_id == membrane.body) {
          system[membrane.group].emplace_back(membrane.group,
                                              dt * e.uptake_velocity * membrane.area / vh);
          source[membrane.group] +=
              dt * e.secretion_rate * membrane.area / body_area.at(e.body_id) / amount_scale;
        }
      }
    }

    auto chemical_parameters = parameters;
    // Preserve relative accuracy even when a dilute/slow feed makes the
    // equilibrated right-hand side much smaller than the reservoir scale.
    chemical_parameters.absolute_tolerance = 0;
    const auto solved =
        detail::solve_numerics_linear(device.get(), std::move(system), source, chemical_parameters);
    std::vector<double> concentration(n);

    for (std::size_t i = 0; i < n; ++i) {
      concentration[i] = solved.solution[i] * amount_scale / vh;

      if (concentration[i] < 0) {
        throw std::runtime_error(
            "solute transport produced a negative concentration; candidate rejected");
      }
    }

    auto result = solved;
    result.solution = std::move(concentration);

    return result;
  }

  void record_species_ledger(const TransportGeometry::Impl& g, const std::vector<Solute>& solutes,
                             const std::vector<std::vector<double>>& reservoir,
                             const std::vector<ChemicalBoundaryKind>& boundary_kind,
                             const std::vector<SurfaceTransferLaw>& exchange,
                             const std::map<CellId, std::size_t>& cell_transfer,
                             const std::vector<double>& concentration, std::size_t s,
                             double total_initial, double amount_scale,
                             const LinearSolveParameters& parameters,
                             SoluteTransportResult& result) {
    const auto old_count = g.old_count, new_count = g.new_count, ns = solutes.size();
    const double dt = g.dt, h = g.grid.spacing;
    const auto& group = g.group;
    const auto& edges = g.edges;
    const auto& membranes = g.membranes;
    const auto& body_area = g.body_area;
    double total_final = 0, injected = 0, uptake = 0;

    for (std::size_t i = 0; i < new_count; ++i) {
      const double c = concentration[group[old_count + i]], m = c * g.final_volumes[i];
      result.amounts[i * ns + s] = m;
      result.concentrations[i * ns + s] = c;
      total_final += m;
    }

    for (const auto& e : edges) {
      if (e.second == fluid_boundary) {
        const double diffusion = boundary_kind[e.port] == ChemicalBoundaryKind::reservoir_contact
                                     ? solutes[s].diffusion * e.area / (h * 0.5)
                                     : 0;
        const double transfer = dt * ((diffusion + std::max(-e.q, 0.0)) * reservoir[e.port][s] -
                                      (diffusion + std::max(e.q, 0.0)) * concentration[e.first]);
        result.reservoirs[e.port].amounts[s] += transfer;
        injected += transfer;
      }
    }

    for (const auto& membrane : membranes) {
      for (const auto& e : exchange) {
        if (e.solute == s && e.body_id == membrane.body) {
          const double transfer = dt * membrane.area *
                                  (e.uptake_velocity * concentration[membrane.group] -
                                   e.secretion_rate / body_area.at(e.body_id));
          result.cells[cell_transfer.at(e.body_id)].amounts[s] += transfer;
          uptake += transfer;
        }
      }
    }

    for (const auto& membrane : membranes) {
      result.surfaces[cell_transfer.at(membrane.body)].concentrations[s] +=
          membrane.area * concentration[membrane.group] / body_area.at(membrane.body);
    }

    const double error = total_final - total_initial - injected + uptake;
    result.report.mass_balance_error[s] = error;
    const double budget = total_initial + std::abs(injected) + std::abs(uptake) + total_final;

    if (std::abs(error) > 10 * (parameters.relative_tolerance * budget +
                                parameters.absolute_tolerance * amount_scale)) {
      throw std::runtime_error("solute transport mass ledger failed");
    }
  }

  void validate_outlet_backflow(const TransportGeometry::Impl& g,
                                const std::vector<ChemicalBoundaryKind>& boundary_kind,
                                const std::vector<bool>& backflow) {
    const auto& edges = g.edges;
    const double h = g.grid.spacing, vh = h * h * h, dt = g.dt;

    for (const auto& e : edges) {
      if (e.second == fluid_boundary && boundary_kind[e.port] == ChemicalBoundaryKind::outflow &&
          !backflow[e.port] && e.q < -1e-10 * vh / dt) {
        throw std::runtime_error("chemical outlet backflow requires an external composition");
      }
    }
  }

  Impl(BackendKind kind, std::uint32_t index) : device(detail::make_numerics_device(kind, index)) {}
};

SoluteTransportSolver::SoluteTransportSolver(BackendKind kind, std::uint32_t index)
    : impl_(std::make_unique<Impl>(kind, index)) {}

SoluteTransportSolver::~SoluteTransportSolver() = default;

SoluteTransportResult SoluteTransportSolver::step(
    const FluidGeometry& before, const FluidGeometry& after, const MacVelocityField& velocity,
    const std::vector<FlowPort>& ports, const std::vector<Solute>& solutes,
    const std::vector<ChemicalBoundary>& reservoirs, const std::vector<double>& amounts, double dt,
    const std::vector<SurfaceTransferLaw>& exchange, const LinearSolveParameters& parameters) {
  const auto geometry = TransportGeometry::prepare(before, after, velocity, ports, dt, parameters,
                                                   impl_->device.get());

  return propose(geometry, solutes, reservoirs, amounts, exchange, parameters);
}

SoluteTransportResult SoluteTransportSolver::propose(
    const TransportGeometry& prepared, const std::vector<Solute>& solutes,
    const std::vector<ChemicalBoundary>& reservoirs, const std::vector<double>& amounts,
    const std::vector<SurfaceTransferLaw>& exchange, const LinearSolveParameters& parameters) {
  parameters.validate();
  const auto& g = *prepared.impl_;
  const auto& grid = g.grid;
  const auto& ports = g.ports;
  const double dt = g.dt, h = grid.spacing, vh = h * h * h;
  const auto old_count = g.old_count, new_count = g.new_count, n = g.group_count,
             ns = solutes.size();
  const auto& group = g.group;
  const auto& body_area = g.body_area;
  impl_->validate_chemical_input(amounts, solutes, old_count);

  std::vector<std::vector<double>> reservoir(ports.size());
  std::vector<ChemicalBoundaryKind> boundary_kind(ports.size());
  std::vector<bool> backflow(ports.size(), true);
  impl_->map_reservoirs(ports, reservoirs, ns, reservoir, boundary_kind, backflow);

  impl_->validate_surface_exchange(exchange, body_area, ns);

  std::vector<double> initial(n * ns);

  for (std::size_t i = 0; i < old_count; ++i) {
    for (std::size_t s = 0; s < ns; ++s) {
      initial[group[i] * ns + s] += amounts[i * ns + s];
    }
  }

  SoluteTransportResult result;
  result.report.projection_iterations = g.report.projection_iterations;
  result.report.maximum_volume_residual = g.report.maximum_volume_residual;

  impl_->validate_outlet_backflow(g, boundary_kind, backflow);

  result.amounts.resize(new_count * ns);
  result.concentrations.resize(new_count * ns);
  result.report.mass_balance_error.resize(ns);

  for (const auto& p : ports) {
    result.reservoirs.push_back({p.name, 0, std::vector<double>(ns)});
  }

  std::map<CellId, std::size_t> cell_transfer;

  for (const auto& [body, area] : body_area) {
    cell_transfer[body] = result.cells.size();
    result.cells.push_back({"", body, std::vector<double>(ns)});
    result.surfaces.push_back({body, area, std::vector<double>(ns)});
  }

  for (std::size_t s = 0; s < ns; ++s) {
    double amount_scale = 0, total_initial = 0;

    for (std::size_t i = 0; i < n; ++i) {
      amount_scale = std::max(amount_scale, initial[i * ns + s]);
      total_initial += initial[i * ns + s];
    }

    for (const auto& r : reservoir) {
      amount_scale = std::max(amount_scale, r[s] * vh);
    }

    for (const auto& e : exchange) {
      if (e.solute == s) {
        amount_scale = std::max(amount_scale, e.secretion_rate * dt);
      }
    }

    if (amount_scale == 0) {
      continue;
    }

    const auto solved = impl_->solve_species(g, solutes, reservoir, boundary_kind, exchange,
                                             initial, s, amount_scale, parameters);
    result.report.transport_iterations += solved.iterations;
    const auto& concentration = solved.solution;

    impl_->record_species_ledger(g, solutes, reservoir, boundary_kind, exchange, cell_transfer,
                                 concentration, s, total_initial, amount_scale, parameters, result);
  }

  return result;
}
}  // namespace cm
