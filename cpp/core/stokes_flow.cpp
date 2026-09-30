#include "cm/stokes_flow.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <utility>

#include "capsule_bodies.hpp"
#include "capsule_contacts.hpp"
#include "flexible_gmres.hpp"
#include "flow_system.hpp"
#include "fluid_multigrid.hpp"
#include "numerics_device.hpp"
#include "numerics_linear.hpp"

namespace cm {
namespace {
constexpr std::uint32_t absent = std::numeric_limits<std::uint32_t>::max();
using Coordinates = std::array<std::uint32_t, 3>;
using detail::norm2;
using detail::solve_cpu;
using detail::solve_gpu;
using detail::SparseMatrix;
using Row = std::vector<std::pair<std::uint32_t, double>>;

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::invalid_argument(message);
  }
}

bool positive(double x) {
  return std::isfinite(x) && x > 0;
}

std::uint32_t index32(std::size_t n) {
  if (n >= absent) {
    throw std::length_error("fluid flow exceeds uint32 indexing");
  }

  return static_cast<std::uint32_t>(n);
}

struct PortedSystem {
  FluidGridSpec lattice;
  detail::FlowGridLayout layout;
  std::vector<std::uint32_t> velocity, pressure, component, face_port, port_pressure;
  std::vector<std::vector<std::size_t>> port_faces;
  std::vector<double> reference_pressure;
  std::vector<double> volume_source_m3_s;
  std::vector<double> body_volume_rate_m3_s;
  std::vector<Row> rows;
  std::vector<double> rhs;
  double h_m, speed_scale, pressure_scale;
  std::uint64_t estimated_bytes;
  std::size_t velocity_count{0};
  std::vector<detail::BodyMarker> markers;
  std::vector<std::uint32_t> body_velocity;
  std::uint32_t marker_start{0};
  std::uint32_t contact_start{0};

  PortedSystem(const FluidGridSpec& grid, const FluidProperties& fluid,
               const std::vector<FlowPort>& ports, const LinearSolveParameters& parameters,
               const std::vector<CapsuleBody>& bodies,
               const std::vector<detail::CapsuleContact>& contacts)
      : lattice(grid),
        layout(lattice, FlowAxis::y),
        h_m(double(grid.spacing) * grid.length_unit_m),
        speed_scale(h_m / grid.time_unit_s),
        pressure_scale(0),
        estimated_bytes(0) {
    const auto n = layout.site_count(), nf = layout.total_face_count();
    // Includes peak host assembly and 2*120 Arnoldi vectors in CPU binary64.
    // Checked before allocating topology, not after a failed device allocation.
    estimated_bytes = (n + nf + ports.size() + contacts.size()) *
                      (bodies.empty() ? std::uint64_t{4096} : std::uint64_t{12288});

    validate_bodies(fluid, parameters, bodies);
    initialize_topology(ports.size());
    const auto nc = identify_components();
    const auto [component_source, component_source_scale] = distribute_growth_sources(bodies, nc);
    const auto anchored =
        initialize_ports(ports, fluid, nc, component_source, component_source_scale);

    allocate_velocity_dofs();
    allocate_constraint_dofs(ports, bodies, contacts.size(), anchored);

    assemble_normal_strain();
    assemble_shear_strain();
    assemble_port_tractions(ports);
    assemble_body_coupling(bodies);
    assemble_body_forces(fluid, bodies, contacts);
  }

 private:
  void validate_bodies(const FluidProperties& fluid, const LinearSolveParameters& parameters,
                       const std::vector<CapsuleBody>& bodies) {
    const auto& grid = lattice;
    const auto n = layout.site_count();
    std::set<CellId> body_ids;

    for (const auto& body : bodies) {
      body.validate();
      require(body_ids.insert(body.id).second, "duplicate fluid body id");
      require(body.radius >= 2 * grid.spacing,
              "fluid body radius requires at least two grid spacings");
      const double markers_estimate = 32 + 4 * std::numbers::pi * body.radius *
                                               (body.length + 2 * body.radius) /
                                               (double(grid.spacing) * grid.spacing);
      require(markers_estimate * 32768 <= double(parameters.memory_limit_bytes),
              "fluid body marker memory estimate exceeds memory_limit_bytes");
      const auto body_bytes = static_cast<std::uint64_t>(markers_estimate * 32768);
      require(body_bytes <= parameters.memory_limit_bytes &&
                  estimated_bytes <= parameters.memory_limit_bytes - body_bytes,
              "fluid body memory estimate exceeds memory_limit_bytes");
      estimated_bytes += body_bytes;
      const auto direction = detail::body_rotate(body.orientation, {1, 0, 0});
      std::array<double, 3> start{}, end{};
      const std::array<double, 3> origin{grid.origin.x, grid.origin.y, grid.origin.z};

      for (std::size_t a = 0; a < 3; ++a) {
        start[a] = body.position[a] - 0.5 * body.length * direction[a];
        end[a] = body.position[a] + 0.5 * body.length * direction[a];
        require(std::min(start[a], end[a]) - body.radius >= origin[a] - 0.5 * grid.spacing &&
                    std::max(start[a], end[a]) + body.radius <=
                        origin[a] + (double(layout.dimensions()[a]) - 0.5) * grid.spacing,
                "fluid body intersects the domain boundary");
      }

      speed_scale =
          std::max(speed_scale, std::abs(body.length_rate) * grid.length_unit_m / grid.time_unit_s);

      for (std::size_t s = 0; s < n; ++s) {
        if (lattice.solid_site(s)) {
          const auto x = layout.site_coordinates(s);
          std::array<double, 3> lo{}, hi{};

          for (std::size_t a = 0; a < 3; ++a) {
            lo[a] = origin[a] + (double(x[a]) - 0.5) * grid.spacing;
            hi[a] = lo[a] + grid.spacing;
          }

          require(detail::body_segment_box_distance_squared(start, end, lo, hi) >=
                      body.radius * body.radius,
                  "fluid body penetrates a wall voxel");
        }
      }

      for (std::size_t a = 0; a < 3; ++a) {
        speed_scale =
            std::max(speed_scale, std::abs(body.force_n[a]) / (fluid.viscosity_pa_s * h_m));
        speed_scale =
            std::max(speed_scale, std::abs(body.torque_nm[a]) / (fluid.viscosity_pa_s * h_m * h_m));
      }
    }

    require(estimated_bytes <= parameters.memory_limit_bytes,
            "fluid memory estimate exceeds memory_limit_bytes");

    for (std::size_t a = 0; a < bodies.size(); ++a) {
      for (std::size_t b = a + 1; b < bodies.size(); ++b) {
        const auto [a0, a1] = detail::body_endpoints(bodies[a]);
        const auto [b0, b1] = detail::body_endpoints(bodies[b]);
        const auto [pa, pb] = detail::body_closest_segments(a0, a1, b0, b1);
        double distance = 0;

        for (std::size_t d = 0; d < 3; ++d) {
          distance += std::pow(pa[d] - pb[d], 2);
        }

        require(std::sqrt(distance) >= bodies[a].radius + bodies[b].radius, "fluid bodies overlap");
      }
    }
  }

  void initialize_topology(std::size_t port_count) {
    const auto n = layout.site_count(), nf = layout.total_face_count();
    index32(n + nf + port_count);
    velocity.assign(nf, absent);
    pressure.assign(n, absent);
    component.assign(n, absent);
    face_port.assign(nf, absent);
    port_pressure.assign(port_count, absent);
    port_faces.resize(port_count);
  }

  std::uint32_t identify_components() {
    const auto n = layout.site_count();
    // Components are defined by face connectivity; sealed pockets have their
    // own pressure gauge and cannot exchange mass through a corner.
    std::uint32_t nc = 0;

    for (std::size_t seed = 0; seed < n; ++seed) {
      if (lattice.solid_site(seed) || component[seed] != absent) {
        continue;
      }

      std::vector<std::size_t> pending{seed};
      component[seed] = nc;

      for (std::size_t q = 0; q < pending.size(); ++q) {
        for (std::size_t a = 0; a < 3; ++a) {
          for (int sign : {-1, 1}) {
            auto neighbor = layout.neighbor_site(pending[q], a, sign);

            if (neighbor && !lattice.solid_site(*neighbor) && component[*neighbor] == absent) {
              component[*neighbor] = nc;
              pending.push_back(*neighbor);
            }
          }
        }
      }

      ++nc;
    }

    return nc;
  }

  std::pair<std::vector<double>, std::vector<double>> distribute_growth_sources(
      const std::vector<CapsuleBody>& bodies, std::uint32_t nc) {
    const auto& grid = lattice;
    const auto n = layout.site_count();
    reference_pressure.assign(nc, 0);
    volume_source_m3_s.resize(n, 0);
    std::vector<double> component_source(nc, 0), component_source_scale(nc, 0);

    for (const auto& body : bodies) {
      const double rate = std::numbers::pi * body.radius * body.radius * body.length_rate *
                          std::pow(grid.length_unit_m, 3) / grid.time_unit_s;
      require(std::isfinite(rate), "fluid body volume rate overflows physical units");
      body_volume_rate_m3_s.push_back(rate);

      if (rate == 0) {
        continue;
      }

      const auto axis = detail::body_rotate(body.orientation, {1, 0, 0});
      const std::array<double, 3> origin{grid.origin.x, grid.origin.y, grid.origin.z};
      std::vector<std::size_t> inside;

      for (std::size_t s = 0; s < n; ++s) {
        const auto x = layout.site_coordinates(s);
        std::array<double, 3> r{};
        double axial = 0;

        for (std::size_t a = 0; a < 3; ++a) {
          r[a] = origin[a] + double(x[a]) * grid.spacing - body.position[a];
          axial += r[a] * axis[a];
        }

        axial = std::clamp(axial, -body.length / 2, body.length / 2);
        double distance = 0;

        for (std::size_t a = 0; a < 3; ++a) {
          distance += std::pow(r[a] - axial * axis[a], 2);
        }

        if (distance < body.radius * body.radius) {
          inside.push_back(s);
        }
      }

      require(!inside.empty(), "growing fluid body has no interior source cells");
      const auto c = component[inside[0]];
      require(c != absent, "growing fluid body intersects a wall");

      for (auto s : inside) {
        require(component[s] == c, "growing fluid body spans disconnected fluid components");
        volume_source_m3_s[s] += rate / double(inside.size());
      }

      component_source[c] += rate;
      component_source_scale[c] += std::abs(rate);
    }

    return {std::move(component_source), std::move(component_source_scale)};
  }

  std::vector<std::uint32_t> resolve_port_sites(const FlowPort& port, std::size_t axis) const {
    const auto n = layout.site_count();
    auto sites = port.sites;

    if (sites.empty()) {
      for (std::size_t s = 0; s < n; ++s) {
        if (!lattice.solid_site(s) &&
            layout.site_coordinates(s)[axis] == (port.upper ? layout.dimensions()[axis] - 1 : 0)) {
          sites.push_back(index32(s));
        }
      }
    }

    require(!sites.empty(), "flow port is entirely blocked");

    return sites;
  }

  std::uint32_t register_port_faces(const FlowPort& port, std::size_t k, std::size_t axis,
                                    const std::vector<std::uint32_t>& sites) {
    const auto n = layout.site_count();
    std::uint32_t c = absent;

    for (auto site : sites) {
      require(site < n && !lattice.solid_site(site), "flow port site is not fluid");
      auto coordinates = layout.site_coordinates(site);
      require(coordinates[axis] == (port.upper ? layout.dimensions()[axis] - 1 : 0),
              "flow port site does not touch its declared boundary");

      if (c == absent) {
        c = component[site];
      }

      require(c == component[site], "a flow port must belong to one connected fluid component");

      if (port.upper) {
        ++coordinates[axis];
      }

      const auto face = layout.face_index(axis, coordinates[0], coordinates[1], coordinates[2]);
      require(face_port[face] == absent, "flow ports contain duplicate or overlapping faces");
      face_port[face] = index32(k);
      port_faces[k].push_back(face);
    }

    return c;
  }

  std::vector<bool> initialize_ports(const std::vector<FlowPort>& ports,
                                     const FluidProperties& fluid, std::uint32_t nc,
                                     const std::vector<double>& component_source,
                                     const std::vector<double>& component_source_scale) {
    std::vector<bool> anchored(nc, false);
    std::vector<double> net_rate(nc, 0), rate_scale(nc, 0);
    std::set<std::string> names;

    for (std::size_t k = 0; k < ports.size(); ++k) {
      const auto& port = ports[k];
      const auto axis = static_cast<std::size_t>(port.axis);
      require(axis < 3, "invalid flow port axis");
      require(!port.name.empty() && names.insert(port.name).second,
              "flow port names must be nonempty and unique");
      require(std::isfinite(port.value), "flow port value must be finite");
      require(port.kind == FlowPortKind::pressure || port.kind == FlowPortKind::flow_rate,
              "invalid flow port kind");
      const auto sites = resolve_port_sites(port, axis);

      const auto c = register_port_faces(port, k, axis, sites);

      if (port.kind == FlowPortKind::pressure) {
        if (!anchored[c]) {
          reference_pressure[c] = port.value;
        }

        anchored[c] = true;
      } else {
        net_rate[c] += port.value;
        rate_scale[c] += std::abs(port.value);
        speed_scale = std::max(
            speed_scale, std::abs(port.value) / (h_m * h_m * static_cast<double>(sites.size())));
      }
    }

    for (std::size_t c = 0; c < nc; ++c) {
      require(anchored[c] || std::abs(net_rate[c] - component_source[c]) <=
                                 1e-10 * (rate_scale[c] + component_source_scale[c]),
              "prescribed flow rates are incompatible in an unanchored fluid component");
    }

    for (std::size_t k = 0; k < ports.size(); ++k) {
      auto face = port_faces[k][0];
      auto [axis, x] = layout.face_coordinates(face);
      auto site = layout.adjacent_site(axis, x, ports[k].upper ? -1 : 1);

      if (ports[k].kind == FlowPortKind::pressure) {
        speed_scale =
            std::max(speed_scale, std::abs(ports[k].value - reference_pressure[component[*site]]) *
                                      h_m / fluid.viscosity_pa_s);
      }
    }

    pressure_scale = fluid.viscosity_pa_s * speed_scale / h_m;
    require(positive(pressure_scale) && positive(speed_scale), "hydraulic unit scales overflow");

    return anchored;
  }

  void allocate_velocity_dofs() {
    const auto nf = layout.total_face_count();

    for (std::size_t f = 0; f < nf; ++f) {
      auto [axis, x] = layout.face_coordinates(f);
      auto lo = layout.adjacent_site(axis, x, -1), hi = layout.adjacent_site(axis, x, 1);

      if ((lo && hi && !lattice.solid_site(*lo) && !lattice.solid_site(*hi)) ||
          face_port[f] != absent) {
        velocity[f] = index32(velocity_count++);
      }
    }
  }

  void allocate_constraint_dofs(const std::vector<FlowPort>& ports,
                                const std::vector<CapsuleBody>& bodies, std::size_t contact_count,
                                const std::vector<bool>& anchored) {
    const auto& grid = lattice;
    const auto n = layout.site_count();
    std::size_t size = velocity_count;
    std::vector<bool> gauged = anchored;

    for (std::size_t s = 0; s < n; ++s) {
      if (component[s] == absent) {
        continue;
      }

      if (!gauged[component[s]]) {
        gauged[component[s]] = true;
      } else {
        pressure[s] = index32(size++);
      }
    }

    for (std::size_t k = 0; k < ports.size(); ++k) {
      if (ports[k].kind == FlowPortKind::flow_rate) {
        port_pressure[k] = index32(size++);
      }
    }

    markers = detail::capsule_surface_markers(bodies, 2 * grid.spacing);
    marker_start = index32(size);
    size += 3 * markers.size();
    body_velocity.resize(bodies.size(), absent);

    for (std::size_t b = 0; b < bodies.size(); ++b) {
      if (!bodies[b].fixed) {
        body_velocity[b] = index32(size);
        size += 6;
      }
    }

    contact_start = index32(size);
    size += contact_count;
    index32(size);
    rows.resize(size);
    rhs.resize(size);
  }

  void add(std::uint32_t i, std::uint32_t j, double v) {
    if (i != absent && j != absent && v != 0) {
      rows[i].emplace_back(j, v);
    }
  }

  void energy(const Row& gradient, double weight) {
    for (auto [i, a] : gradient) {
      for (auto [j, b] : gradient) {
        add(i, j, weight * a * b);
      }
    }
  }

  std::uint32_t face_dof(std::size_t a, Coordinates x) {
    return velocity[layout.face_index(a, x[0], x[1], x[2])];
  }

  void assemble_normal_strain() {
    const auto n = layout.site_count();

    // 2 mu e:e. Normal strains live at cell centers and shear strains on
    // edges. This symmetric energy form gives pressure TRACTION at ports,
    // including the normal viscous stress, instead of rescaling a velocity.
    for (std::size_t s = 0; s < n; ++s) {
      if (lattice.solid_site(s)) {
        continue;
      }

      if (pressure[s] != absent) {
        rhs[pressure[s]] = -volume_source_m3_s[s] / (h_m * h_m * speed_scale);
      }

      auto x = layout.site_coordinates(s);

      for (std::size_t a = 0; a < 3; ++a) {
        auto y = x;
        ++y[a];
        auto lo = face_dof(a, x), hi = face_dof(a, y);
        energy({{lo, -1}, {hi, 1}}, 2);
        add(pressure[s], lo, 1);
        add(lo, pressure[s], 1);
        add(pressure[s], hi, -1);
        add(hi, pressure[s], -1);
      }
    }
  }

  unsigned edge_fluid_sites(std::size_t a, std::size_t b, Coordinates edge,
                            bool (&fluid)[2][2]) const {
    unsigned count = 0;

    for (unsigned i = 0; i < 2; ++i) {
      for (unsigned j = 0; j < 2; ++j) {
        if ((i == 0 && edge[a] == 0) || (j == 0 && edge[b] == 0)) {
          continue;
        }

        auto q = edge;
        q[a] -= 1 - i;
        q[b] -= 1 - j;

        if (q[a] >= layout.dimensions()[a] || q[b] >= layout.dimensions()[b]) {
          continue;
        }

        fluid[i][j] = !lattice.solid_site(layout.site_index(q[0], q[1], q[2]));
        count += fluid[i][j] ? 1U : 0U;
      }
    }

    return count;
  }

  bool assemble_wall_strain(std::size_t a, std::size_t b, Coordinates edge,
                            const bool (&fluid)[2][2], unsigned count, Row& strain) {
    bool flat = false;

    // A half control volume at a no-slip wall uses a reflected
    // tangential ghost. At an open port, tangential traction is zero.
    if (count == 2) {
      for (unsigned axis = 0; axis < 2; ++axis) {
        for (unsigned side = 0; side < 2; ++side) {
          bool pair =
              axis == 0 ? fluid[side][0] && fluid[side][1] : fluid[0][side] && fluid[1][side];

          if (!pair) {
            continue;
          }

          flat = true;
          const auto normal = axis == 0 ? a : b, tangent = axis == 0 ? b : a;
          auto q0 = edge, q1 = edge;
          --q0[tangent];
          const auto f0 = layout.face_index(normal, q0[0], q0[1], q0[2]);
          const auto f1 = layout.face_index(normal, q1[0], q1[1], q1[2]);

          if (face_port[f0] != absent && face_port[f1] != absent) {
            continue;
          }

          auto q = edge;
          q[normal] -= 1 - side;
          strain.emplace_back(face_dof(tangent, q), side == 1 ? 2 : -2);
          strain.emplace_back(velocity[f0], -1);
          strain.emplace_back(velocity[f1], 1);
          energy(strain, 0.5);
        }
      }
    }

    return flat;
  }

  void assemble_edge_strain(std::size_t a, std::size_t b, Coordinates edge) {
    bool fluid[2][2]{};
    const unsigned count = edge_fluid_sites(a, b, edge, fluid);

    if (count <= 1) {
      return;
    }

    Row strain;
    const bool flat = assemble_wall_strain(a, b, edge, fluid, count, strain);

    if (flat || count == 2) {
      return;
    }

    for (auto [normal, tangent] : {std::pair{a, b}, std::pair{b, a}}) {
      if (edge[normal] > 0) {
        auto q = edge;
        --q[normal];
        strain.emplace_back(face_dof(tangent, q), -1);
      }

      if (edge[normal] < layout.dimensions()[normal]) {
        strain.emplace_back(face_dof(tangent, edge), 1);
      }
    }

    energy(strain, double(count) / 4);
  }

  void assemble_shear_strain() {
    for (std::size_t a = 0; a < 3; ++a) {
      for (std::size_t b = a + 1; b < 3; ++b) {
        auto dims = layout.dimensions();
        ++dims[a];
        ++dims[b];

        for (std::uint32_t x = 0; x < dims[0]; ++x) {
          for (std::uint32_t y = 0; y < dims[1]; ++y) {
            for (std::uint32_t z = 0; z < dims[2]; ++z) {
              assemble_edge_strain(a, b, {x, y, z});
            }
          }
        }
      }
    }
  }

  void assemble_port_tractions(const std::vector<FlowPort>& ports) {
    for (std::size_t k = 0; k < ports.size(); ++k) {
      const auto& port = ports[k];
      const double sign = port.upper ? 1 : -1;

      for (auto f : port_faces[k]) {
        if (port.kind == FlowPortKind::pressure) {
          auto [a, q] = layout.face_coordinates(f);
          auto site = layout.adjacent_site(a, q, port.upper ? -1 : 1);
          rhs[velocity[f]] =
              -sign * (port.value - reference_pressure[component[*site]]) / pressure_scale;
        } else {
          add(velocity[f], port_pressure[k], sign);
          add(port_pressure[k], velocity[f], sign);
        }
      }

      if (port.kind == FlowPortKind::flow_rate) {
        rhs[port_pressure[k]] = port.value / (h_m * h_m * speed_scale);
      }
    }
  }

  void couple_marker_to_fluid(std::size_t a, std::uint32_t row, const std::array<int, 3>& base,
                              const std::array<std::array<double, 6>, 3>& weights) {
    for (unsigned i = 0; i < 6; ++i) {
      for (unsigned j = 0; j < 6; ++j) {
        for (unsigned k = 0; k < 6; ++k) {
          std::array<int, 3> q{base[0] + 3 - int(i), base[1] + 3 - int(j), base[2] + 3 - int(k)};
          const double weight = weights[0][i] * weights[1][j] * weights[2][k];

          if (std::abs(weight) < 1e-18) {
            continue;
          }

          // No-slip walls have zero extension. This is a bounded wall
          // regularization; unresolved lubrication is not added to the model.
          bool inside = true;

          for (std::size_t d = 0; d < 3; ++d) {
            inside = inside && q[d] >= 0 && q[d] < int(layout.dimensions()[d] + (a == d ? 1U : 0U));
          }

          if (!inside) {
            continue;
          }

          const auto face =
              layout.face_index(a, std::uint32_t(q[0]), std::uint32_t(q[1]), std::uint32_t(q[2]));
          require(face_port[face] == absent, "fluid body kernel intersects an open port");
          add(row, velocity[face], weight);
          add(velocity[face], row, weight);
        }
      }
    }
  }

  void assemble_body_coupling(const std::vector<CapsuleBody>& bodies) {
    const auto& grid = lattice;
    // Symmetric constrained-IB system: A u + G p + J^T lambda = f,
    // J u - K U = 0, -K^T lambda = F_external. Lambda is force ON the body.
    const std::array<double, 3> origin{grid.origin.x, grid.origin.y, grid.origin.z};

    for (std::size_t m = 0; m < markers.size(); ++m) {
      const auto& marker = markers[m];
      const auto& body = bodies[marker.body];
      const auto body_axis = detail::body_rotate(body.orientation, {1, 0, 0});

      for (std::size_t a = 0; a < 3; ++a) {
        const auto row = index32(marker_start + 3 * m + a);
        rhs[row] = body_axis[a] * marker.axial_fraction * body.length_rate * grid.length_unit_m /
                   (grid.time_unit_s * speed_scale);
        std::array<int, 3> base{};
        std::array<std::array<double, 6>, 3> weights{};

        for (std::size_t d = 0; d < 3; ++d) {
          const double p = (body.position[d] + marker.offset[d] - origin[d]) / grid.spacing;
          require(p > -0.5 && p < double(layout.dimensions()[d]) - 0.5,
                  "fluid body intersects the domain boundary");
          const double t = p + (a == d ? 0.5 : 0);
          base[d] = static_cast<int>(std::floor(t));
          weights[d] = detail::immersed_delta_weights(t - base[d]);
        }

        couple_marker_to_fluid(a, row, base, weights);

        if (body_velocity[marker.body] != absent) {
          const auto u = body_velocity[marker.body];
          add(row, u + std::uint32_t(a), -1);
          add(u + std::uint32_t(a), row, -1);

          for (std::size_t d = 0; d < 3; ++d) {
            std::array<double, 3> unit{};
            unit[d] = 1;
            const auto& r = marker.offset;
            const std::array<double, 3> cross{unit[1] * r[2] - unit[2] * r[1],
                                              unit[2] * r[0] - unit[0] * r[2],
                                              unit[0] * r[1] - unit[1] * r[0]};
            const double v = -cross[a] / grid.spacing;
            add(row, u + 3 + std::uint32_t(d), v);
            add(u + 3 + std::uint32_t(d), row, v);
          }
        }
      }
    }
  }

  void assemble_body_forces(const FluidProperties& fluid, const std::vector<CapsuleBody>& bodies,
                            const std::vector<detail::CapsuleContact>& contacts) {
    const auto& grid = lattice;
    const double force_scale = fluid.viscosity_pa_s * speed_scale * h_m;
    require(positive(force_scale) && positive(force_scale * h_m),
            "fluid force units overflow or underflow");

    for (std::size_t b = 0; b < bodies.size(); ++b) {
      if (body_velocity[b] != absent) {
        for (std::size_t a = 0; a < 3; ++a) {
          rhs[body_velocity[b] + a] = bodies[b].force_n[a] / force_scale;
          rhs[body_velocity[b] + 3 + a] = bodies[b].torque_nm[a] / (force_scale * h_m);
        }
      }
    }

    for (std::size_t c = 0; c < contacts.size(); ++c) {
      const auto row = contact_start + index32(c);
      bool movable = false;

      for (bool second : {false, true}) {
        const auto b = second ? contacts[c].second : contacts[c].first;

        if (b == detail::fluid_wall_body || body_velocity[b] == absent) {
          continue;
        }

        movable = true;
        const auto gradient = detail::contact_velocity_row(contacts[c], second, grid.spacing);

        for (std::uint32_t a = 0; a < 6; ++a) {
          add(row, body_velocity[b] + a, -gradient[a]);
          add(body_velocity[b] + a, row, -gradient[a]);
        }
      }

      require(movable, "fluid contact cannot be satisfied by fixed bodies");
      rhs[row] =
          -contacts[c].required_speed * grid.length_unit_m / (grid.time_unit_s * speed_scale);
    }
  }
};

}  // namespace

std::size_t FluidGridSpec::site_count() const {
  const auto xy = std::uint64_t(shape.x) * shape.y;

  if (shape.z && xy >= absent / shape.z) {
    throw std::length_error("fluid grid exceeds uint32 indexing");
  }

  return std::size_t(xy * shape.z);
}

void FluidGridSpec::validate() const {
  require(shape.x >= 2 && shape.y >= 2 && shape.z >= 2,
          "resolved fluid flow requires at least two sites along every axis");
  (void)site_count();
  require(positive(spacing) && positive(length_unit_m) && positive(time_unit_s),
          "fluid spacing and unit scales must be finite and positive");
  require(std::isfinite(origin.x) && std::isfinite(origin.y) && std::isfinite(origin.z),
          "fluid origin must be finite");
  const double h = double(spacing) * length_unit_m;
  require(positive(h * h * h) && positive(h * std::max({shape.x, shape.y, shape.z})),
          "physical fluid grid dimensions overflow or underflow");
  require(obstacles.empty() || obstacles.size() == site_count(), "fluid obstacle size mismatch");

  for (auto v : obstacles) {
    require(v <= 1, "fluid obstacles must be Boolean");
  }
}

void FluidProperties::validate() const {
  require(positive(viscosity_pa_s) && positive(density_kg_m3),
          "fluid viscosity and density must be finite and positive");
}

void CapsuleBody::validate() const {
  require(id != 0, "fluid body id must be nonzero");
  require(std::isfinite(length) && length >= 0 && positive(radius),
          "invalid fluid body dimensions");
  require(std::isfinite(length_rate), "nonfinite fluid body length rate");

  for (double x : position) {
    require(std::isfinite(x), "nonfinite fluid body position");
  }

  for (double x : force_n) {
    require(std::isfinite(x), "nonfinite fluid body force");
  }

  for (double x : torque_nm) {
    require(std::isfinite(x), "nonfinite fluid body torque");
  }

  double q = 0;

  for (double x : orientation) {
    require(std::isfinite(x), "nonfinite fluid orientation");
    q += x * x;
  }

  require(std::abs(q - 1) < 1e-10, "fluid orientation must be a unit quaternion");
}

double CapsuleBody::geometric_volume() const {
  return std::numbers::pi * radius * radius * (length + 4 * radius / 3);
}

struct StokesFlowSolver::Impl {
  BackendKind backend;
  std::unique_ptr<detail::NumericsDevice> device;

  explicit Impl(BackendKind kind, std::uint32_t index)
      : backend(kind), device(detail::make_numerics_device(kind, index)) {}
};

StokesFlowSolver::StokesFlowSolver(BackendKind backend, std::uint32_t device_index)
    : impl_(std::make_unique<Impl>(backend, device_index)) {}

StokesFlowSolver::~StokesFlowSolver() = default;

FluidFlowResult StokesFlowSolver::solve(const FluidGridSpec& grid, const FluidProperties& fluid,
                                        const std::vector<FlowPort>& ports,
                                        const LinearSolveParameters& parameters) {
  return solve_bodies(grid, fluid, ports, {}, parameters);
}

static std::vector<detail::MultigridLevel> velocity_hierarchy(const PortedSystem& system,
                                                              const SparseMatrix& matrix) {
  detail::CsrMatrix::Rows velocity_rows(system.velocity_count);

  for (std::size_t i = 0; i < system.velocity_count; ++i) {
    for (auto j = matrix.offsets[i]; j < matrix.offsets[i + 1]; ++j) {
      if (matrix.columns[j] < system.velocity_count) {
        velocity_rows[i][matrix.columns[j]] = matrix.values[j];
      }
    }
  }

  std::vector<detail::VelocityCoordinate> coordinates(system.velocity_count);

  for (std::size_t f = 0; f < system.velocity.size(); ++f) {
    if (system.velocity[f] != absent) {
      auto [axis, x] = system.layout.face_coordinates(f);
      coordinates[system.velocity[f]] = {static_cast<std::uint32_t>(axis), x[0], x[1], x[2]};
    }
  }

  return detail::build_velocity_hierarchy(detail::CsrMatrix(velocity_rows), coordinates);
}

static double reconstruct_body_results(const PortedSystem& system, const FluidGridSpec& grid,
                                       const FluidProperties& fluid,
                                       const std::vector<CapsuleBody>& bodies,
                                       const std::vector<double>& solution,
                                       FluidFlowResult& result) {
  result.bodies.resize(bodies.size());
  const double force_scale = fluid.viscosity_pa_s * system.speed_scale * system.h_m;

  for (std::size_t b = 0; b < bodies.size(); ++b) {
    auto& body = result.bodies[b];
    body.id = bodies[b].id;
    body.volume_change_rate_m3_s = system.body_volume_rate_m3_s[b];

    if (system.body_velocity[b] != absent) {
      for (std::size_t a = 0; a < 3; ++a) {
        body.velocity[a] = solution[system.body_velocity[b] + a] * system.speed_scale *
                           grid.time_unit_s / grid.length_unit_m;
        body.angular_velocity[a] = solution[system.body_velocity[b] + 3 + a] * system.speed_scale *
                                   grid.time_unit_s / system.h_m;
      }
    }
  }

  for (std::size_t m = 0; m < system.markers.size(); ++m) {
    const auto& marker = system.markers[m];
    auto& body = result.bodies[marker.body];
    ++body.marker_count;
    std::array<double, 3> force{};

    for (std::size_t a = 0; a < 3; ++a) {
      const auto row = system.marker_start + 3 * m + a;
      force[a] = solution[row] * force_scale;
      body.hydrodynamic_force_n[a] += force[a];
      double error = -system.rhs[row];

      for (auto [j, v] : system.rows[row]) {
        error += v * solution[j];
      }

      body.no_slip_rms_m_s += error * error * system.speed_scale * system.speed_scale;
    }

    const auto& r = marker.offset;
    body.hydrodynamic_torque_nm[0] += (r[1] * force[2] - r[2] * force[1]) * grid.length_unit_m;
    body.hydrodynamic_torque_nm[1] += (r[2] * force[0] - r[0] * force[2]) * grid.length_unit_m;
    body.hydrodynamic_torque_nm[2] += (r[0] * force[1] - r[1] * force[0]) * grid.length_unit_m;
  }

  for (auto& body : result.bodies) {
    body.no_slip_rms_m_s = std::sqrt(body.no_slip_rms_m_s / (3 * body.marker_count));
  }

  return force_scale;
}

static void reconstruct_port_results(const PortedSystem& system, const std::vector<FlowPort>& ports,
                                     const std::vector<double>& solution,
                                     const std::vector<double>& velocity, FluidFlowResult& result) {
  auto& report = result.report;

  for (std::size_t k = 0; k < ports.size(); ++k) {
    double q = 0;

    for (auto f : system.port_faces[k]) {
      q += velocity[f] * (ports[k].upper ? 1 : -1);
    }

    auto [a, x] = system.layout.face_coordinates(system.port_faces[k][0]);
    auto s = system.layout.adjacent_site(a, x, ports[k].upper ? -1 : 1);
    const double p = ports[k].kind == FlowPortKind::pressure
                         ? ports[k].value
                         : solution[system.port_pressure[k]] * system.pressure_scale +
                               system.reference_pressure[system.component[*s]];
    result.ports.push_back(
        {ports[k].name, p, q * system.h_m * system.h_m,
         static_cast<double>(system.port_faces[k].size()) * system.h_m * system.h_m});
    report.net_flow_rate_m3_s += result.ports.back().flow_rate_m3_s;
  }
}

static void reconstruct_fluid_results(const PortedSystem& system, const FluidGridSpec& grid,
                                      const FluidProperties& fluid,
                                      const std::vector<FlowPort>& ports,
                                      const std::vector<double>& solution,
                                      FluidFlowResult& result) {
  auto& report = result.report;
  std::vector<double> velocity(system.layout.total_face_count());

  for (std::size_t f = 0; f < velocity.size(); ++f) {
    if (system.velocity[f] != absent) {
      velocity[f] = solution[system.velocity[f]] * system.speed_scale;
    }
  }

  result.pressure_pa.resize(system.layout.site_count(), 0);
  double divergence_square = 0, continuity_square = 0;
  std::size_t fluid_count = 0;

  for (std::size_t s = 0; s < result.pressure_pa.size(); ++s) {
    if (system.component[s] == absent) {
      continue;
    }

    ++fluid_count;
    result.pressure_pa[s] =
        system.reference_pressure[system.component[s]] +
        (system.pressure[s] == absent ? 0 : solution[system.pressure[s]] * system.pressure_scale);
    auto x = system.layout.site_coordinates(s);
    double divergence = 0;

    for (std::size_t a = 0; a < 3; ++a) {
      auto y = x;
      ++y[a];
      divergence += (velocity[system.layout.face_index(a, y[0], y[1], y[2])] -
                     velocity[system.layout.face_index(a, x[0], x[1], x[2])]) /
                    system.h_m;
    }

    divergence_square += divergence * divergence;
    const double source = system.volume_source_m3_s[s] / std::pow(system.h_m, 3);
    continuity_square += (divergence - source) * (divergence - source);
    report.source_volume_rate_m3_s += system.volume_source_m3_s[s];
  }

  reconstruct_port_results(system, ports, solution, velocity, result);

  report.divergence_rms_per_s =
      fluid_count ? std::sqrt(divergence_square / double(fluid_count)) : 0;
  report.continuity_rms_per_s =
      fluid_count ? std::sqrt(continuity_square / double(fluid_count)) : 0;
  std::array<std::vector<float>*, 3> fields{&result.field.x_faces, &result.field.y_faces,
                                            &result.field.z_faces};

  for (std::size_t a = 0; a < 3; ++a) {
    fields[a]->resize(system.layout.face_counts()[a]);

    for (std::size_t j = 0; j < fields[a]->size(); ++j) {
      const double u = velocity[system.layout.face_offsets()[a] + j];
      report.max_speed_m_s = std::max(report.max_speed_m_s, std::abs(u));
      const float v = float(u * grid.time_unit_s / grid.length_unit_m);

      if (!std::isfinite(v)) {
        throw std::overflow_error("fluid model velocity overflow");
      }

      (*fields[a])[j] = v;
    }
  }

  const double length = system.h_m * std::max({grid.shape.x, grid.shape.y, grid.shape.z});
  report.reynolds_number =
      fluid.density_kg_m3 * report.max_speed_m_s * length / fluid.viscosity_pa_s;
  report.viscous_relaxation_time_s = fluid.density_kg_m3 * length * length / fluid.viscosity_pa_s;
}

static FluidFlowResult solve_stokes_system(detail::NumericsDevice* device,
                                           const FluidGridSpec& grid, const FluidProperties& fluid,
                                           const std::vector<FlowPort>& ports,
                                           const std::vector<CapsuleBody>& bodies,
                                           const LinearSolveParameters& parameters,
                                           const std::vector<detail::CapsuleContact>& contacts,
                                           std::vector<double>* contact_forces) {
  grid.validate();
  fluid.validate();
  parameters.validate();

  if (device) {
    require(parameters.relative_tolerance >= std::numeric_limits<float>::epsilon(),
            "GPU fluid relative tolerance must be at least float32 epsilon");
  }

  PortedSystem system(grid, fluid, ports, parameters, bodies, contacts);
  SparseMatrix matrix(system.rows);
  const auto hierarchy = velocity_hierarchy(system, matrix);

  auto rhs = system.rhs;

  for (std::size_t i = 0; i < rhs.size(); ++i) {
    rhs[i] *= matrix.scale[i];
  }

  const auto solved = device
                          ? solve_gpu(*device, matrix, rhs, parameters, hierarchy, !bodies.empty())
                          : solve_cpu(matrix, rhs, parameters, hierarchy, !bodies.empty());
  auto residual = rhs;
  matrix.apply(solved.solution, residual);

  for (std::size_t i = 0; i < rhs.size(); ++i) {
    residual[i] -= rhs[i];
  }

  const double absolute = norm2(residual), rhs_norm = norm2(rhs);

  if (!std::isfinite(absolute) ||
      absolute > 1.05 * std::max(parameters.absolute_tolerance,
                                 parameters.relative_tolerance * rhs_norm)) {
    throw std::runtime_error("fluid true residual failed; candidate rejected");
  }

  auto solution = solved.solution;

  for (std::size_t i = 0; i < solution.size(); ++i) {
    solution[i] *= matrix.scale[i];

    if (!std::isfinite(solution[i])) {
      throw std::runtime_error("nonfinite fluid solution");
    }
  }

  FluidFlowResult result;
  auto& report = result.report;
  report.iterations = solved.iterations;
  report.relative_residual = rhs_norm == 0 ? 0 : absolute / rhs_norm;
  report.absolute_residual = absolute;
  report.estimated_memory_bytes = system.estimated_bytes;
  reconstruct_fluid_results(system, grid, fluid, ports, solution, result);

  const double force_scale =
      reconstruct_body_results(system, grid, fluid, bodies, solution, result);

  if (contact_forces) {
    contact_forces->resize(contacts.size());

    for (std::size_t c = 0; c < contacts.size(); ++c) {
      (*contact_forces)[c] = solution[system.contact_start + c] * force_scale;
    }
  }

  return result;
}

FluidFlowResult StokesFlowSolver::solve_bodies(const FluidGridSpec& grid,
                                               const FluidProperties& fluid,
                                               const std::vector<FlowPort>& ports,
                                               const std::vector<CapsuleBody>& bodies,
                                               const LinearSolveParameters& parameters) {
  return solve_stokes_system(impl_->device.get(), grid, fluid, ports, bodies, parameters, {},
                             nullptr);
}

void FluidBodyStepParameters::validate() const {
  require(std::isfinite(minimum_gap_m) && minimum_gap_m >= 0, "invalid fluid minimum contact gap");
  require(positive(maximum_displacement_fraction) && maximum_displacement_fraction <= 0.25,
          "body displacement fraction must be in (0,0.25]");
  require(max_halvings <= 64 && max_contact_iterations > 0, "invalid fluid body step limits");
}

static bool remove_attractive_contact(std::vector<std::size_t>& active,
                                      const std::vector<double>& forces, double force_tolerance) {
  std::size_t remove = active.size();
  double most_negative = -force_tolerance;

  for (std::size_t c = 0; c < forces.size(); ++c) {
    if (forces[c] < most_negative) {
      most_negative = forces[c];
      remove = c;
    }
  }

  if (remove < active.size()) {
    active.erase(active.begin() + static_cast<std::ptrdiff_t>(remove));

    return true;
  }

  return false;
}

static bool add_violated_contact(std::vector<std::size_t>& active,
                                 const std::vector<detail::CapsuleContact>& candidates,
                                 const FluidFlowResult& flow, const FluidGridSpec& grid,
                                 double dt) {
  double worst = -1e-7 * grid.spacing / dt;
  std::size_t add = candidates.size();

  for (std::size_t c = 0; c < candidates.size(); ++c) {
    if (std::find(active.begin(), active.end(), c) != active.end()) {
      continue;
    }

    const double slack =
        detail::contact_separation_speed(candidates[c], flow) - candidates[c].required_speed;

    if (slack < worst) {
      worst = slack;
      add = c;
    }
  }

  if (add < candidates.size()) {
    active.push_back(add);

    return true;
  }

  return false;
}

static bool advance_body_positions(std::vector<CapsuleBody>& bodies, const FluidFlowResult& flow,
                                   double dt, const FluidGridSpec& grid,
                                   const FluidBodyStepParameters& step_parameters) {
  bool bounded = true;

  for (std::size_t b = 0; b < bodies.size(); ++b) {
    const auto& motion = flow.bodies[b];
    auto& body = bodies[b];
    const double speed = std::sqrt(detail::body_dot(motion.velocity, motion.velocity));
    const double spin =
        std::sqrt(detail::body_dot(motion.angular_velocity, motion.angular_velocity));
    const double displacement =
        dt * (speed + (body.length / 2 + body.radius) * spin + std::abs(body.length_rate) / 2);

    if (displacement > step_parameters.maximum_displacement_fraction * grid.spacing ||
        body.length + dt * body.length_rate < 0) {
      bounded = false;
      break;
    }

    for (std::size_t d = 0; d < 3; ++d) {
      body.position[d] += dt * motion.velocity[d];
    }

    body.length += dt * body.length_rate;

    if (spin > 0) {
      const double angle = dt * spin;
      const double s = std::sin(angle / 2) / spin;
      const std::array<double, 4> dq{std::cos(angle / 2), s * motion.angular_velocity[0],
                                     s * motion.angular_velocity[1],
                                     s * motion.angular_velocity[2]};
      const auto& q = body.orientation;
      std::array<double, 4> next{dq[0] * q[0] - dq[1] * q[1] - dq[2] * q[2] - dq[3] * q[3],
                                 dq[0] * q[1] + dq[1] * q[0] + dq[2] * q[3] - dq[3] * q[2],
                                 dq[0] * q[2] - dq[1] * q[3] + dq[2] * q[0] + dq[3] * q[1],
                                 dq[0] * q[3] + dq[1] * q[2] - dq[2] * q[1] + dq[3] * q[0]};
      double norm = 0;

      for (double x : next) {
        norm += x * x;
      }

      for (double& x : next) {
        x /= std::sqrt(norm);
      }

      body.orientation = next;
    }
  }

  return bounded;
}

static void record_active_contacts(const FluidGridSpec& grid,
                                   const std::vector<CapsuleBody>& bodies,
                                   const std::vector<detail::CapsuleContact>& candidates,
                                   const std::vector<std::size_t>& active,
                                   const std::vector<double>& forces, FluidBodyStep& candidate) {
  for (std::size_t c = 0; c < active.size(); ++c) {
    const auto& contact = candidates[active[c]];
    auto point = bodies[contact.first].position;

    for (std::size_t a = 0; a < 3; ++a) {
      point[a] += contact.first_offset[a];
    }

    candidate.contacts.push_back(
        {bodies[contact.first].id,
         contact.second == detail::fluid_wall_body ? 0 : bodies[contact.second].id, contact.normal,
         point, contact.gap * grid.length_unit_m, forces[c]});
  }
}

FluidBodyStep StokesFlowSolver::propose_body_step(
    const FluidGridSpec& grid, const FluidProperties& fluid, const std::vector<FlowPort>& ports,
    const std::vector<CapsuleBody>& bodies, double maximum_dt,
    const LinearSolveParameters& solve_parameters, const FluidBodyStepParameters& step_parameters) {
  step_parameters.validate();
  require(positive(maximum_dt) && positive(maximum_dt * grid.time_unit_s),
          "body step time must be finite and positive");
  const auto unconstrained = solve_bodies(grid, fluid, ports, bodies, solve_parameters);
  const double gap = step_parameters.minimum_gap_m / grid.length_unit_m;
  require(std::isfinite(gap), "fluid contact gap overflows model units");
  const double guarded_gap = gap + 1e-5 * grid.spacing;
  double dt = maximum_dt;
  std::string failure = "body displacement exceeds the geometric step limit";

  for (std::uint32_t halving = 0; halving <= step_parameters.max_halvings; ++halving, dt *= 0.5) {
    if (!positive(dt)) {
      break;
    }

    const auto candidates =
        detail::capsule_contacts(grid, bodies, guarded_gap, dt, 0.5 * grid.spacing);
    std::vector<std::size_t> active;
    std::vector<double> forces;
    auto flow = unconstrained;
    bool converged = false;
    std::uint32_t iteration = 0;

    for (; iteration < step_parameters.max_contact_iterations; ++iteration) {
      std::vector<detail::CapsuleContact> constraints;

      for (auto c : active) {
        constraints.push_back(candidates[c]);
      }

      if (!active.empty()) {
        flow = solve_stokes_system(impl_->device.get(), grid, fluid, ports, bodies,
                                   solve_parameters, constraints, &forces);
      } else {
        flow = unconstrained;
        forces.clear();
      }

      const double force_tolerance =
          std::max(1e-30, solve_parameters.relative_tolerance * fluid.viscosity_pa_s *
                              flow.report.max_speed_m_s * grid.spacing * grid.length_unit_m);

      if (remove_attractive_contact(active, forces, force_tolerance)) {
        continue;
      }

      if (add_violated_contact(active, candidates, flow, grid, dt)) {
        continue;
      }

      converged = true;
      break;
    }

    if (!converged) {
      throw std::runtime_error(
          "fluid contact complementarity did not converge; candidate rejected");
    }

    FluidBodyStep candidate;
    candidate.bodies = bodies;
    candidate.accepted_dt = dt;
    candidate.halvings = halving;
    candidate.contact_iterations = iteration + 1;
    const bool bounded = advance_body_positions(candidate.bodies, flow, dt, grid, step_parameters);

    if (!bounded) {
      continue;
    }

    try {
      const auto after = detail::capsule_contacts(grid, candidate.bodies, gap, dt, 0);
      bool separated = true;

      for (const auto& c : after) {
        separated = separated && c.gap >= gap;
      }

      if (!separated) {
        failure = "nonlinear contact gap failed after the proposed motion";
        continue;
      }

      // Check domain, ports, walls, resolution, and budgets before returning a
      // candidate. No second fluid solution is implied by this geometry check.
      (void)PortedSystem(grid, fluid, ports, solve_parameters, candidate.bodies, {});
    } catch (const std::invalid_argument& error) {
      failure = error.what();
      continue;
    }

    record_active_contacts(grid, bodies, candidates, active, forces, candidate);

    candidate.flow = std::move(flow);

    return candidate;
  }

  throw std::runtime_error("fluid body step exhausted substep halvings: " + failure);
}
}  // namespace cm
