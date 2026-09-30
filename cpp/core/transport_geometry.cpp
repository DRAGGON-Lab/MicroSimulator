#include <numeric>
#include <set>
#include <tuple>

#include "flow_system.hpp"
#include "numerics_device.hpp"
#include "transport_geometry_internal.hpp"

namespace cm {
namespace {
void require(bool ok, const char* message) {
  if (!ok) {
    throw std::invalid_argument(message);
  }
}

struct UnionFind {
  std::vector<std::uint32_t> root;

  explicit UnionFind(std::size_t n) : root(n) {
    std::iota(root.begin(), root.end(), 0);
  }

  std::uint32_t find(std::uint32_t i) {
    while (root[i] != i) {
      root[i] = root[root[i]];
      i = root[i];
    }

    return i;
  }

  void join(std::uint32_t i, std::uint32_t j) {
    root[find(i)] = find(j);
  }
};

std::vector<double> validated_face_velocities(const MacVelocityField& velocity) {
  std::vector<double> face_velocity;

  for (const auto* values : {&velocity.x_faces, &velocity.y_faces, &velocity.z_faces}) {
    for (float u : *values) {
      require(std::isfinite(u), "nonfinite culture velocity");
      face_velocity.push_back(u);
    }
  }

  return face_velocity;
}

std::vector<std::uint32_t> resolve_transport_port(const FluidGridSpec& grid,
                                                  const detail::FlowGridLayout& layout,
                                                  const FlowPort& port) {
  const auto& lattice = grid;
  const auto axis = static_cast<unsigned>(port.axis);
  auto sites = port.sites;

  if (sites.empty()) {
    for (std::uint32_t s = 0; s < grid.site_count(); ++s) {
      const auto c = layout.site_coordinates(s);

      if (!lattice.solid_site(s) && c[axis] == (port.upper ? layout.dimensions()[axis] - 1 : 0)) {
        sites.push_back(s);
      }
    }
  }

  return sites;
}

std::vector<std::uint32_t> transport_port_faces(const FluidGridSpec& grid,
                                                const detail::FlowGridLayout& layout,
                                                const std::vector<FlowPort>& ports) {
  const auto& lattice = grid;
  std::set<std::string> names;
  std::vector<std::uint32_t> port_face(layout.total_face_count(), fluid_boundary);

  for (std::uint32_t p = 0; p < ports.size(); ++p) {
    const auto& port = ports[p];
    const auto axis = static_cast<unsigned>(port.axis);
    require(axis < 3 && !port.name.empty() && names.insert(port.name).second &&
                std::isfinite(port.value) &&
                (port.kind == FlowPortKind::pressure || port.kind == FlowPortKind::flow_rate),
            "invalid culture port");
    const auto sites = resolve_transport_port(grid, layout, port);

    require(!sites.empty(), "culture port is empty");

    for (auto s : sites) {
      require(s < grid.site_count() && !lattice.solid_site(s),
              "culture port selects an invalid site");
      auto c = layout.site_coordinates(s);
      require(c[axis] == (port.upper ? layout.dimensions()[axis] - 1 : 0),
              "culture port site is not on its boundary");

      if (port.upper) {
        ++c[axis];
      }

      const auto f = layout.face_index(axis, c[0], c[1], c[2]);
      require(port_face[f] == fluid_boundary, "overlapping culture ports");
      port_face[f] = p;
    }
  }

  return port_face;
}

std::tuple<std::vector<std::uint32_t>, std::vector<double>, std::vector<double>> temporal_volumes(
    const FluidGeometry& before, const FluidGeometry& after,
    const std::vector<FluidOverlap>& overlaps, const LinearSolveParameters& parameters) {
  const auto old_count = before.fragments().size(), new_count = after.fragments().size();
  // Space-time control volumes are connected unions of old and new fragments
  // within ONE voxel. Only a positive geometric overlap joins the time levels.
  // Closing and opening fragments remain in the solve with zero final/initial
  // volume; no component-wide amount redistribution is performed.
  UnionFind temporal(old_count + new_count);

  for (const auto& o : overlaps) {
    temporal.join(o.first, static_cast<std::uint32_t>(old_count) + o.second);
  }

  std::map<std::uint32_t, std::uint32_t> group_ids;
  std::vector<std::uint32_t> group(old_count + new_count);

  for (std::uint32_t i = 0; i < group.size(); ++i) {
    const auto [it, inserted] =
        group_ids.try_emplace(temporal.find(i), static_cast<std::uint32_t>(group_ids.size()));
    group[i] = it->second;
  }

  const auto n = group_ids.size();
  require(n * std::uint64_t{12288} <= parameters.memory_limit_bytes,
          "solute transport memory limit exceeded");
  std::vector<double> v0(n), v1(n);

  for (std::size_t i = 0; i < old_count; ++i) {
    v0[group[i]] += before.fragments()[i].volume;
  }

  for (std::size_t i = 0; i < new_count; ++i) {
    v1[group[old_count + i]] += after.fragments()[i].volume;
  }

  return {std::move(group), std::move(v0), std::move(v1)};
}

void append_transport_face(
    const FluidFace& face, std::size_t offset, const std::vector<std::uint32_t>& group,
    const std::vector<std::uint32_t>& port_face, const std::vector<double>& face_velocity,
    std::map<std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>, TransportEdge>& edge_map,
    std::map<std::pair<std::uint32_t, CellId>, double>& membrane_map) {
  auto a = group[offset + face.first];

  if (face.body_id) {
    membrane_map[{a, face.body_id}] += 0.5 * face.area;

    return;
  }

  if (face.grid_face == fluid_boundary) {
    return;
  }

  auto b = face.second == fluid_boundary ? fluid_boundary : group[offset + face.second];
  const auto p = b == fluid_boundary ? port_face[face.grid_face] : fluid_boundary;

  if (b == fluid_boundary && p == fluid_boundary) {
    return;
  }

  if (a == b) {
    return;
  }

  double q = 0.5 * face.area * face.normal[static_cast<unsigned>(face.axis)] *
             face_velocity[face.grid_face];

  if (b != fluid_boundary && a > b) {
    std::swap(a, b);
    q = -q;
  }

  auto [it, inserted] = edge_map.try_emplace({a, b, p}, TransportEdge{a, b, p});
  auto& e = it->second;
  e.area += 0.5 * face.area;
  e.q += q;
}

std::tuple<std::vector<TransportEdge>, std::vector<TransportMembrane>, std::map<CellId, double>>
transport_faces(const FluidGeometry& before, const FluidGeometry& after,
                const std::vector<std::uint32_t>& group,
                const std::vector<std::uint32_t>& port_face,
                const std::vector<double>& face_velocity) {
  const auto old_count = before.fragments().size();
  std::map<std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>, TransportEdge> edge_map;
  std::map<std::pair<std::uint32_t, CellId>, double> membrane_map;

  for (bool newer : {false, true}) {
    const auto& geometry = newer ? after : before;
    const auto offset = newer ? old_count : 0;

    for (const auto& face : geometry.faces()) {
      append_transport_face(face, offset, group, port_face, face_velocity, edge_map, membrane_map);
    }
  }

  std::vector<TransportEdge> edges;

  for (const auto& [key, e] : edge_map) {
    edges.push_back(e);
  }

  std::vector<TransportMembrane> membranes;
  std::map<CellId, double> body_area;

  for (const auto& [key, area] : membrane_map) {
    membranes.push_back({key.first, key.second, area});
    body_area[key.second] += area;
  }

  return {std::move(edges), std::move(membranes), std::move(body_area)};
}

void fix_projection_gauge(detail::NumericsLinearRows& rows, std::vector<double>& rhs,
                          UnionFind& components, const std::vector<bool>& pressure_node,
                          const std::vector<double>& v0, const std::vector<double>& v1, double vh) {
  const auto np = rhs.size(), n = v0.size();
  std::map<std::uint32_t, bool> anchored;
  std::map<std::uint32_t, double> balance, budget;

  for (std::uint32_t i = 0; i < np; ++i) {
    const auto root = components.find(i);
    anchored[root] = anchored[root] || pressure_node[i];
    balance[root] += rhs[i];
    budget[root] += std::abs(rhs[i]);

    if (i < n) {
      budget[root] += (v0[i] + v1[i]) / vh;
    }
  }

  std::set<std::uint32_t> gauge;

  for (const auto& [root, has_pressure] : anchored) {
    if (!has_pressure) {
      require(std::abs(balance[root]) <= 1e-10 * std::max(1.0, budget[root]),
              "incompatible culture volume change and prescribed port rates");
      gauge.insert(root);
    }
  }

  for (std::uint32_t i = 0; i < np; ++i) {
    if (gauge.contains(i)) {
      rows[i] = {{i, 1}};
      rhs[i] = 0;
    } else {
      std::erase_if(rows[i], [&](const auto& a) {
        return gauge.contains(a.first);
      });
    }
  }
}

detail::NumericsLinearResult solve_volume_projection(detail::NumericsDevice* device,
                                                     const detail::NumericsLinearRows& rows,
                                                     const std::vector<double>& rhs,
                                                     const LinearSolveParameters& parameters,
                                                     std::size_t n) {
  auto projection = detail::solve_numerics_linear(device, rows, rhs, parameters);

  // A global Krylov norm does not control every cut-cell volume equation.
  // Refine with residuals of the original operator. Corrections are solved on
  // the selected backend; no host solve or tolerance relaxation is involved.
  for (unsigned refinement = 0; refinement < 4; ++refinement) {
    auto residual = rhs;
    double scale = 0;

    for (std::size_t i = 0; i < rows.size(); ++i) {
      for (const auto& [j, a] : rows[i]) {
        residual[i] -= a * projection.solution[j];
      }

      scale = std::max(scale, std::abs(residual[i]));
    }

    if (scale < 0.1 * std::max(parameters.relative_tolerance, parameters.absolute_tolerance) /
                    std::max(1.0, double(n))) {
      break;
    }

    for (double& r : residual) {
      r /= scale;
    }

    const auto correction = detail::solve_numerics_linear(device, rows, residual, parameters);
    projection.iterations += correction.iterations;

    for (std::size_t i = 0; i < rows.size(); ++i) {
      projection.solution[i] += scale * correction.solution[i];
    }
  }

  return projection;
}

GeometricFluxReport apply_volume_projection(std::vector<TransportEdge>& edges,
                                            const std::vector<std::uint32_t>& rate_node,
                                            const detail::NumericsLinearResult& projection,
                                            const std::vector<double>& v0,
                                            const std::vector<double>& v1, double vh, double dt,
                                            const LinearSolveParameters& parameters) {
  const auto n = v0.size();
  std::vector<double> volume_residual(n);

  for (std::size_t i = 0; i < n; ++i) {
    volume_residual[i] = v1[i] - v0[i];
  }

  for (auto& e : edges) {
    const auto b = e.second == fluid_boundary ? rate_node[e.port] : e.second;
    e.q += e.weight *
           (projection.solution[e.first] - (b == fluid_boundary ? 0 : projection.solution[b])) *
           vh / dt;
    volume_residual[e.first] += dt * e.q;

    if (e.second != fluid_boundary) {
      volume_residual[e.second] -= dt * e.q;
    }
  }

  GeometricFluxReport report;
  report.projection_iterations = projection.iterations;

  for (double r : volume_residual) {
    report.maximum_volume_residual = std::max(report.maximum_volume_residual, std::abs(r));
  }

  if (report.maximum_volume_residual >
      10 * vh * std::max(parameters.relative_tolerance, parameters.absolute_tolerance)) {
    throw std::runtime_error("culture geometric conservation residual failed");
  }

  return report;
}

GeometricFluxReport project_transport_flux(
    const FluidGridSpec& grid, const std::vector<FlowPort>& ports, double dt,
    const std::vector<double>& v0, const std::vector<double>& v1, std::vector<TransportEdge>& edges,
    const LinearSolveParameters& parameters, detail::NumericsDevice* device) {
  const auto n = v0.size();
  const double h = grid.spacing, vh = h * h * h;
  std::vector<std::uint32_t> rate_node(ports.size(), fluid_boundary);
  std::size_t np = n;

  for (std::size_t p = 0; p < ports.size(); ++p) {
    if (ports[p].kind == FlowPortKind::flow_rate) {
      rate_node[p] = static_cast<std::uint32_t>(np++);
    }
  }

  detail::NumericsLinearRows rows(np);
  std::vector<double> rhs(np), raw_port(ports.size());
  std::vector<bool> pressure_node(np, false);
  UnionFind components(np);

  for (std::size_t i = 0; i < n; ++i) {
    rhs[i] = (v0[i] - v1[i]) / vh;
  }

  for (auto& e : edges) {
    const auto a = e.first, b = e.second == fluid_boundary ? rate_node[e.port] : e.second;
    e.weight = e.area / (h * h) * (e.second == fluid_boundary ? 2 : 1);
    rows[a].emplace_back(a, e.weight);
    rhs[a] -= e.q * dt / vh;

    if (b != fluid_boundary) {
      rows[a].emplace_back(b, -e.weight);
      rows[b].emplace_back(a, -e.weight);
      rows[b].emplace_back(b, e.weight);
      rhs[b] += e.q * dt / vh;
      components.join(a, b);
    } else {
      pressure_node[a] = true;
    }

    if (e.second == fluid_boundary) {
      raw_port[e.port] += e.q;
    }
  }

  const double rate_to_model = grid.time_unit_s / std::pow(grid.length_unit_m, 3);

  for (std::size_t p = 0; p < ports.size(); ++p) {
    if (rate_node[p] != fluid_boundary) {
      rhs[rate_node[p]] -= ports[p].value * rate_to_model * dt / vh;
    }
  }

  fix_projection_gauge(rows, rhs, components, pressure_node, v0, v1, vh);

  const auto projection = solve_volume_projection(device, rows, rhs, parameters, n);

  const auto report =
      apply_volume_projection(edges, rate_node, projection, v0, v1, vh, dt, parameters);

  return report;
}

}  // namespace

TransportGeometry::TransportGeometry(std::shared_ptr<const Impl> value) : impl_(std::move(value)) {}

TransportGeometry::TransportGeometry(const FluidGeometry& before, const FluidGeometry& after,
                                     const MacVelocityField& velocity,
                                     const std::vector<FlowPort>& ports, double dt,
                                     BackendKind backend, std::uint32_t device_index,
                                     const LinearSolveParameters& parameters)
    : TransportGeometry(prepare(before, after, velocity, ports, dt, parameters,
                                detail::make_numerics_device(backend, device_index).get())) {}

const GeometricFluxReport& TransportGeometry::report() const {
  return impl_->report;
}

TransportGeometry TransportGeometry::prepare(const FluidGeometry& before,
                                             const FluidGeometry& after,
                                             const MacVelocityField& velocity,
                                             const std::vector<FlowPort>& ports, double dt,
                                             const LinearSolveParameters& parameters,
                                             detail::NumericsDevice* device) {
  parameters.validate();
  require(std::isfinite(dt) && dt > 0, "solute transport dt must be finite and positive");
  const auto& grid = before.grid();
  require(grid.length_unit_m == after.grid().length_unit_m &&
              grid.time_unit_s == after.grid().time_unit_s,
          "solute transport units changed during a step");
  const auto overlaps = before.overlaps(after);  // also validates the immutable device grid
  const auto old_count = before.fragments().size(), new_count = after.fragments().size();
  require(old_count + new_count < fluid_boundary, "too many solute transport fragments");
  const auto& lattice = grid;
  const detail::FlowGridLayout layout(lattice, FlowAxis::x);
  require(velocity.x_faces.size() == layout.face_count(0) &&
              velocity.y_faces.size() == layout.face_count(1) &&
              velocity.z_faces.size() == layout.face_count(2),
          "solute transport velocity dimensions mismatch");
  const auto face_velocity = validated_face_velocities(velocity);

  const auto port_face = transport_port_faces(grid, layout, ports);

  auto [group, v0, v1] = temporal_volumes(before, after, overlaps, parameters);
  const auto n = v0.size();

  auto [edges, membranes, body_area] =
      transport_faces(before, after, group, port_face, face_velocity);

  const auto report = project_transport_flux(grid, ports, dt, v0, v1, edges, parameters, device);

  auto value = std::make_shared<Impl>();
  value->grid = grid;
  value->ports = ports;
  value->dt = dt;
  value->old_count = old_count;
  value->new_count = new_count;
  value->group_count = n;
  value->group = std::move(group);
  value->v0 = std::move(v0);
  value->v1 = std::move(v1);

  for (const auto& f : after.fragments()) {
    value->final_volumes.push_back(f.volume);
  }

  value->edges = std::move(edges);
  value->membranes = std::move(membranes);
  value->body_area = std::move(body_area);
  value->report = report;

  return TransportGeometry(std::move(value));
}
}  // namespace cm
