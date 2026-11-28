#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend_devices.hpp"
#include "microsimulator/backend.hpp"
#include "microsimulator/simulation.hpp"
#include "microsimulator/threads.hpp"

namespace {

using microsimulator::BackendKind;
using microsimulator::BiomassDeposit;
using microsimulator::DepositCells;
using microsimulator::ShallowFlowCoupling;
using microsimulator::SignalGridSpec;
using microsimulator::Simulation;
using microsimulator::Vec3;

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::fprintf(stderr, "cell kernels conformance: %s\n", message.c_str());
    std::exit(1);
  }
}

SignalGridSpec box(std::uint32_t nx, std::uint32_t ny, std::uint32_t nz) {
  SignalGridSpec spec;
  spec.signal_count = 1;
  spec.shape = {.x = nx, .y = ny, .z = nz};
  spec.origin = {-0.5F * static_cast<float>(nx - 1), -0.5F * static_cast<float>(ny - 1), 0.0F};
  spec.spacing = {1.0F, 1.0F, 1.0F};
  spec.diffusion = {0.0F};
  spec.advection = {{0.0F, 0.0F, 0.0F}};
  return spec;
}

std::size_t site(const SignalGridSpec& spec, std::uint32_t x, std::uint32_t y, std::uint32_t z) {
  return (static_cast<std::size_t>(x) * spec.shape.y + y) * spec.shape.z + z;
}

double biomass(float length, float radius) {
  return std::numbers::pi * radius * radius * (length + 2.0 * radius);
}

// A colony scattered over the box, with in-plane velocities as its channels.
DepositCells colony(const SignalGridSpec& spec, std::size_t count) {
  DepositCells cells;
  cells.channel_count = 2;
  std::vector<float> vx;
  std::vector<float> vy;
  std::uint32_t state = 12345;
  const auto next = [&state] {
    state = state * 1664525U + 1013904223U;
    return static_cast<float>(state >> 8) / static_cast<float>(1U << 24);
  };
  const auto width = static_cast<float>(spec.shape.x - 1);
  const auto height = static_cast<float>(spec.shape.y - 1);
  for (std::size_t cell = 0; cell < count; ++cell) {
    cells.positions.push_back(
        {spec.origin.x + width * next(), spec.origin.y + height * next(),
         spec.origin.z + static_cast<float>(spec.shape.z - 1) * next()});
    cells.lengths.push_back(1.0F + 3.0F * next());
    cells.radii.push_back(0.5F);
    vx.push_back(4.0F * next() - 2.0F);
    vy.push_back(4.0F * next() - 2.0F);
  }
  cells.channels = vx;
  cells.channels.insert(cells.channels.end(), vy.begin(), vy.end());
  return cells;
}

double total(const std::vector<float>& values, std::size_t begin, std::size_t count) {
  double sum = 0.0;
  for (std::size_t index = begin; index < begin + count; ++index) {
    sum += values[index];
  }
  return sum;
}

void require_close(double actual, double expected, double tolerance, const std::string& what) {
  require(std::abs(actual - expected) <= tolerance * std::max(1.0, std::abs(expected)),
          what + ": " + std::to_string(actual) + " against " + std::to_string(expected));
}

// The sum over sites of density times site volume is the biomass of the cells
// inside the grid, and likewise for each channel.
void require_conserved(const SignalGridSpec& spec, const DepositCells& cells,
                       const BiomassDeposit& deposit, const std::vector<bool>& inside) {
  const auto sites = spec.site_count();
  double mass = 0.0;
  double momentum_x = 0.0;
  double momentum_y = 0.0;
  for (std::size_t cell = 0; cell < cells.size(); ++cell) {
    if (!inside[cell]) {
      continue;
    }
    const auto amount = biomass(cells.lengths[cell], cells.radii[cell]);
    mass += amount;
    momentum_x += amount * cells.channels[cell];
    momentum_y += amount * cells.channels[cells.size() + cell];
  }
  const auto volume = static_cast<double>(spec.voxel_volume());
  require_close(total(deposit.density, 0, sites) * volume, mass, 1.0e-5, "biomass conservation");
  require_close(total(deposit.channels, 0, sites) * volume, momentum_x, 1.0e-4,
                "x momentum conservation");
  require_close(total(deposit.channels, sites, sites) * volume, momentum_y, 1.0e-4,
                "y momentum conservation");
}

void test_conservation_and_truncation() {
  const auto spec = box(24, 30, 3);
  auto cells = colony(spec, 300);
  // One cell at a corner, whose tent the grid truncates, and one outside the
  // grid, which deposits nothing.
  cells.positions[0] = {spec.origin.x, spec.origin.y, 0.0F};
  cells.positions[1] = {spec.origin.x - 10.0F, 0.0F, 0.0F};
  std::vector<bool> inside(cells.size(), true);
  inside[1] = false;
  Simulation simulation(BackendKind::cpu);
  const auto deposit = simulation.deposit_biomass(spec, cells);
  require(deposit.density.size() == spec.site_count(), "density covers every site");
  require(deposit.channel_count == 2, "deposit keeps its channels");
  require_conserved(spec, cells, deposit, inside);
}

// A wall across the grid keeps a cell's biomass on its own side.
void test_wall_separates_support() {
  auto spec = box(20, 12, 1);
  spec.obstacles.assign(spec.site_count(), 0);
  for (std::uint32_t y = 0; y < spec.shape.y; ++y) {
    spec.obstacles[site(spec, 10, y, 0)] = 1;
  }
  DepositCells cells;
  // Site 9 is centered at x = -0.5; the tent reaches across the wall at site 10.
  cells.positions = {{-0.7F, 0.0F, 0.0F}};
  cells.lengths = {2.0F};
  cells.radii = {0.5F};
  Simulation simulation(BackendKind::cpu);
  const auto deposit = simulation.deposit_biomass(spec, cells);
  for (std::uint32_t x = 10; x < spec.shape.x; ++x) {
    for (std::uint32_t y = 0; y < spec.shape.y; ++y) {
      require(deposit.density[site(spec, x, y, 0)] == 0.0F, "no biomass past the wall");
    }
  }
  require_close(total(deposit.density, 0, spec.site_count()), biomass(2.0F, 0.5F), 1.0e-5,
                "conservation beside a wall");
}

void test_solid_support_is_rejected() {
  auto spec = box(30, 30, 1);
  spec.obstacles.assign(spec.site_count(), 1);
  spec.obstacles[site(spec, 0, 0, 0)] = 0;
  DepositCells cells;
  cells.positions = {{0.0F, 0.0F, 0.0F}};
  cells.lengths = {1.0F};
  cells.radii = {0.5F};
  Simulation simulation(BackendKind::cpu);
  bool rejected = false;
  try {
    static_cast<void>(simulation.deposit_biomass(spec, cells));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "a tent with no fluid support is rejected");
}

// The closure matches its formula evaluated in double precision.
void test_closure_formula() {
  const auto spec = box(16, 16, 2);
  const auto cells = colony(spec, 120);
  Simulation simulation(BackendKind::cpu);
  const auto deposit = simulation.deposit_biomass(spec, cells);
  const microsimulator::BiomassDragParameters parameters{.drag_coefficient = 2.0F,
                                                                  .max_volume_fraction = 0.5F};
  const std::vector<float> base{0.25F};
  const auto coupling = simulation.shallow_flow_coupling(spec, base, deposit, parameters);
  const auto sites = spec.site_count();
  for (std::uint32_t x = 0; x < spec.shape.x; ++x) {
    for (std::uint32_t y = 0; y < spec.shape.y; ++y) {
      double mass = 0.0;
      double mx = 0.0;
      double my = 0.0;
      for (std::uint32_t z = 0; z < spec.shape.z; ++z) {
        mass += deposit.density[site(spec, x, y, z)];
        mx += deposit.channels[site(spec, x, y, z)];
        my += deposit.channels[sites + site(spec, x, y, z)];
      }
      const auto fraction = std::min(mass / spec.shape.z, 0.5);
      const auto drag = 2.0 * fraction * fraction / std::pow(1.0 - fraction, 3);
      const auto mobility = 0.25 / (1.0 + 0.25 * drag);
      for (std::uint32_t z = 0; z < spec.shape.z; ++z) {
        require_close(coupling.mobility[site(spec, x, y, z)], mobility, 1.0e-5, "mobility");
      }
      const auto column = static_cast<std::size_t>(x) * spec.shape.y + y;
      const auto fx = mass > 0.0 ? drag * mx / mass : 0.0;
      const auto fy = mass > 0.0 ? drag * my / mass : 0.0;
      require_close(coupling.body_force[column][0], fx, 1.0e-4, "x body force");
      require_close(coupling.body_force[column][1], fy, 1.0e-4, "y body force");
    }
  }
}

struct Reference {
  BiomassDeposit deposit;
  ShallowFlowCoupling coupling;
  std::vector<float> drag;
};

// A walled device with an obstacle block, deposited and closed on a backend.
Reference run(BackendKind backend, std::uint32_t device_index) {
  auto spec = box(40, 48, 4);
  spec.obstacles.assign(spec.site_count(), 0);
  for (std::uint32_t x = 18; x < 22; ++x) {
    for (std::uint32_t y = 0; y < 30; ++y) {
      for (std::uint32_t z = 0; z < spec.shape.z; ++z) {
        spec.obstacles[site(spec, x, y, z)] = 1;
      }
    }
  }
  auto cells = colony(spec, 600);
  for (std::size_t cell = 0; cell < cells.size(); ++cell) {
    // Cells inside the block sit just beside it instead.
    auto& position = cells.positions[cell];
    if (position.x > -2.5F && position.x < 2.5F && position.y < 6.5F) {
      position.x = -3.0F;
    }
  }
  Simulation simulation(backend, 0, 0, device_index);
  Reference result;
  result.deposit = simulation.deposit_biomass(spec, cells);
  std::vector<float> base(spec.site_count());
  for (std::size_t index = 0; index < base.size(); ++index) {
    base[index] = spec.obstacles[index] != 0 ? 0.0F : 0.5F + 0.001F * static_cast<float>(index % 7);
  }
  result.coupling = simulation.shallow_flow_coupling(spec, base, result.deposit,
                                                     {.drag_coefficient = 2.0F,
                                                      .max_volume_fraction = 0.5F});
  result.drag = simulation.brinkman_drag(spec, result.deposit,
                                         {.drag_coefficient = 3.0F, .max_volume_fraction = 0.6F});
  return result;
}

// The largest distance between two equal-length arrays in units in the last
// place, with its index.
std::string largest_difference(const std::vector<float>& left, const std::vector<float>& right) {
  std::int64_t worst = 0;
  std::size_t at = 0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    const auto a = std::bit_cast<std::int32_t>(left[index]);
    const auto b = std::bit_cast<std::int32_t>(right[index]);
    const auto distance = std::abs(static_cast<std::int64_t>(a) - static_cast<std::int64_t>(b));
    if (distance > worst) {
      worst = distance;
      at = index;
    }
  }
  return std::to_string(worst) + " ulp at " + std::to_string(at) + " (" +
         std::to_string(left[at]) + " against " + std::to_string(right[at]) + ")";
}

bool same(const std::vector<float>& left, const std::vector<float>& right) {
  return left == right;
}

void require_same(const Reference& left, const Reference& right, const std::string& label) {
  require(left.deposit.density.size() == right.deposit.density.size(), label + ": density size");
  require(same(left.deposit.density, right.deposit.density),
          label + ": density differs by " +
              largest_difference(left.deposit.density, right.deposit.density));
  require(same(left.deposit.channels, right.deposit.channels), label + ": channels differ");
  require(same(left.coupling.mobility, right.coupling.mobility), label + ": mobility differs");
  require(left.coupling.body_force == right.coupling.body_force, label + ": body force differs");
  require(same(left.drag, right.drag), label + ": Brinkman drag differs");
}

// Agreement to `tolerance` relative to the largest magnitude of the
// reference, as two backends that round differently agree.
void require_near(const std::vector<float>& actual, const std::vector<float>& expected,
                  double tolerance, const std::string& label) {
  require(actual.size() == expected.size(), label + ": size differs");
  double scale = 0.0;
  for (const float value : expected) {
    scale = std::max(scale, std::abs(static_cast<double>(value)));
  }
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const double difference =
        std::abs(static_cast<double>(actual[index]) - static_cast<double>(expected[index]));
    require(difference <= tolerance * std::max(scale, 1e-30),
            label + ": differs at " + std::to_string(index) + " (" + std::to_string(actual[index]) +
                " against " + std::to_string(expected[index]) + ")");
  }
}

template <std::size_t N>
std::vector<float> flatten(const std::vector<std::array<float, N>>& rows) {
  std::vector<float> values;
  values.reserve(N * rows.size());
  for (const auto& row : rows) {
    values.insert(values.end(), row.begin(), row.end());
  }
  return values;
}

constexpr double backend_tolerance = 1e-5;

void require_close(const Reference& actual, const Reference& expected, const std::string& label) {
  require_near(actual.deposit.density, expected.deposit.density, backend_tolerance,
               label + " density");
  require_near(actual.deposit.channels, expected.deposit.channels, backend_tolerance,
               label + " channels");
  require_near(actual.coupling.mobility, expected.coupling.mobility, backend_tolerance,
               label + " mobility");
  require_near(flatten(actual.coupling.body_force), flatten(expected.coupling.body_force),
               backend_tolerance, label + " body force");
  require_near(actual.drag, expected.drag, backend_tolerance, label + " Brinkman drag");
}

// A colony in a box container with walls, posts, and a pillar of every kind,
// with cells pressed into the container's corners and against each wall.
struct ContactScene {
  microsimulator::WorldState state;
  microsimulator::ConstraintSet constraints;
};

ContactScene contact_scene() {
  ContactScene scene{microsimulator::WorldState(400), {}};
  auto& constraints = scene.constraints;
  constraints.add_box({.center = {0.0F, 0.0F, 0.0F},
                       .half_extents = {10.0F, 8.0F, 1.5F},
                       .coefficient = 1.0F,
                       .allowed_region = microsimulator::ConstraintRegion::inside});
  constraints.add_cylinder({.center = {20.0F, 0.0F, 0.0F},
                            .radius = 6.0F,
                            .half_height = 1.5F,
                            .coefficient = 0.5F,
                            .allowed_region = microsimulator::ConstraintRegion::inside});
  constraints.add_box({.center = {3.0F, 2.0F, 0.0F}, .half_extents = {1.0F, 1.0F, 2.0F}});
  constraints.add_cylinder({.center = {-4.0F, -3.0F, 0.0F}, .radius = 1.2F, .half_height = 2.0F});
  constraints.add_sphere({.center = {-6.0F, 4.0F, 0.0F}, .radius = 1.5F});
  constraints.add_plane({.point = {0.0F, -7.5F, 0.0F}, .inward_normal = {0.0F, 1.0F, 0.0F}});
  std::uint32_t state = 777;
  const auto next = [&state] {
    state = state * 1664525U + 1013904223U;
    return static_cast<float>(state >> 8) / static_cast<float>(1U << 24);
  };
  for (std::size_t cell = 0; cell < 300; ++cell) {
    const auto angle = 6.2831853F * next();
    const Vec3 direction{std::cos(angle), std::sin(angle), 0.2F * (next() - 0.5F)};
    const auto magnitude = microsimulator::norm(direction);
    Vec3 position{-9.5F + 19.0F * next(), -7.5F + 15.0F * next(), -1.0F + 2.0F * next()};
    if (cell % 5 == 0) {
      position = {20.0F + 5.0F * std::cos(angle), 5.0F * std::sin(angle), 0.9F};
    }
    if (cell % 17 == 0) {
      position = {cell % 2 == 0 ? 9.2F : -9.2F, cell % 3 == 0 ? 7.2F : -7.2F, 0.9F};
    }
    scene.state.add_cell({.position = position,
                          .direction = direction * (1.0F / magnitude),
                          .length = 1.0F + 2.0F * next(),
                          .radius = 0.5F});
  }
  return scene;
}

bool same_contacts(const microsimulator::ExternalContactGraph& left,
                   const microsimulator::ExternalContactGraph& right) {
  const auto a = left.contacts();
  const auto b = right.contacts();
  if (a.size() != b.size()) {
    return false;
  }
  for (std::size_t index = 0; index < a.size(); ++index) {
    const auto& x = a[index];
    const auto& y = b[index];
    if (x.cell_id != y.cell_id || x.cell_slot != y.cell_slot || x.constraint_id != y.constraint_id ||
        x.constraint_kind != y.constraint_kind || x.location != y.location ||
        x.point_on_cell.x != y.point_on_cell.x || x.point_on_cell.y != y.point_on_cell.y ||
        x.point_on_cell.z != y.point_on_cell.z || x.normal.x != y.normal.x ||
        x.normal.y != y.normal.y || x.normal.z != y.normal.z ||
        x.signed_separation != y.signed_separation || x.weight != y.weight) {
      return false;
    }
  }
  return true;
}

// The same contacts, with the same separations and weights within tolerance.
// Where a cell lies parallel to a face, every point of its centerline is
// equally near, and where it sits as deep behind two faces of a box, either
// face is nearest, so the point and the direction of the unit normal are not
// compared.
void require_close_contacts(const microsimulator::ExternalContactGraph& actual,
                            const microsimulator::ExternalContactGraph& expected,
                            const std::string& label) {
  const auto a = actual.contacts();
  const auto b = expected.contacts();
  require(a.size() == b.size(), label + ": contact count differs");
  for (std::size_t index = 0; index < a.size(); ++index) {
    const auto& x = a[index];
    const auto& y = b[index];
    const auto at = label + ": contact " + std::to_string(index);
    require(x.cell_id == y.cell_id && x.cell_slot == y.cell_slot &&
                x.constraint_id == y.constraint_id && x.constraint_kind == y.constraint_kind &&
                x.location == y.location,
            at + " differs");
    require(std::abs(x.signed_separation - y.signed_separation) <= 1e-4F, at + " separation");
    require(std::abs(x.weight - y.weight) <= 1e-5F * std::max(1.0F, std::abs(y.weight)),
            at + " weight");
    const float length = x.normal.x * x.normal.x + x.normal.y * x.normal.y + x.normal.z * x.normal.z;
    require(std::abs(length - 1.0F) <= 1e-5F, at + " normal is not a unit vector");
  }
}

struct ContactReference {
  microsimulator::ExternalContactGraph nearest{0, {}};
  microsimulator::ExternalContactGraph faces{0, {}};
};

ContactReference run_contacts(BackendKind backend, std::uint32_t device_index) {
  const auto scene = contact_scene();
  const microsimulator::ConstraintContactParameters parameters{.activation_margin = 0.3F,
                                                                .degeneracy_epsilon = 1.0e-6F};
  auto compute = microsimulator::make_compute_backend(backend, device_index);
  return {
      .nearest = compute->find_external_contacts(scene.state, scene.constraints, parameters),
      .faces = compute->find_external_contact_faces(scene.state, scene.constraints, parameters),
  };
}

// A container makes a row for every face a cell end is near: more rows than
// for the nearest face alone, and at least two at a corner.
void test_container_faces(const ContactReference& reference) {
  require(reference.faces.size() > reference.nearest.size(), "every face adds container rows");
  bool corner = false;
  for (std::size_t slot = 0; slot < reference.faces.cell_count(); ++slot) {
    std::size_t container_rows = 0;
    for (const auto index : reference.faces.incident_contact_indices(static_cast<
                                                                     microsimulator::Slot>(slot))) {
      const auto& contact = reference.faces.contacts()[index];
      if (contact.constraint_kind == microsimulator::ExternalConstraintKind::box &&
          contact.location == microsimulator::RodContactLocation::positive &&
          contact.constraint_id == reference.faces.contacts()[index].constraint_id) {
        ++container_rows;
      }
    }
    corner = corner || container_rows >= 2;
  }
  require(corner, "a cell in a corner meets two faces");
}

// A drifting colony in a channel past a block, with a velocity field that
// shears and turns.
struct DriftScene {
  microsimulator::WorldState state;
  std::optional<microsimulator::SignalGrid> grid;
};

DriftScene drift_scene() {
  auto spec = box(30, 40, 3);
  spec.obstacles.assign(spec.site_count(), 0);
  for (std::uint32_t x = 12; x < 16; ++x) {
    for (std::uint32_t y = 18; y < 22; ++y) {
      for (std::uint32_t z = 0; z < spec.shape.z; ++z) {
        spec.obstacles[site(spec, x, y, z)] = 1;
      }
    }
  }
  microsimulator::SignalGridVelocityField field;
  field.x_faces.resize(spec.x_face_count());
  field.y_faces.resize(spec.y_face_count());
  field.z_faces.assign(spec.z_face_count(), 0.0F);
  for (std::uint32_t x = 0; x <= spec.shape.x; ++x) {
    for (std::uint32_t y = 0; y < spec.shape.y; ++y) {
      for (std::uint32_t z = 0; z < spec.shape.z; ++z) {
        const auto boundary = x == 0 || x == spec.shape.x;
        field.x_faces[(static_cast<std::size_t>(x) * spec.shape.y + y) * spec.shape.z + z] =
            boundary ? 0.0F
                     : 0.3F * std::sin(0.2F * static_cast<float>(y)) *
                           (1.0F + 0.1F * static_cast<float>(z));
      }
    }
  }
  for (std::uint32_t x = 0; x < spec.shape.x; ++x) {
    for (std::uint32_t y = 0; y <= spec.shape.y; ++y) {
      for (std::uint32_t z = 0; z < spec.shape.z; ++z) {
        field.y_faces[(static_cast<std::size_t>(x) * (spec.shape.y + 1) + y) * spec.shape.z + z] =
            2.0F + 0.05F * static_cast<float>(x) * static_cast<float>(x);
      }
    }
  }
  // The field vanishes on every face of a solid site.
  for (std::uint32_t x = 0; x < spec.shape.x; ++x) {
    for (std::uint32_t y = 0; y < spec.shape.y; ++y) {
      for (std::uint32_t z = 0; z < spec.shape.z; ++z) {
        if (spec.obstacles[site(spec, x, y, z)] == 0) {
          continue;
        }
        const auto yz = static_cast<std::size_t>(spec.shape.z);
        field.x_faces[(static_cast<std::size_t>(x) * spec.shape.y + y) * yz + z] = 0.0F;
        field.x_faces[(static_cast<std::size_t>(x + 1) * spec.shape.y + y) * yz + z] = 0.0F;
        field.y_faces[(static_cast<std::size_t>(x) * (spec.shape.y + 1) + y) * yz + z] = 0.0F;
        field.y_faces[(static_cast<std::size_t>(x) * (spec.shape.y + 1) + y + 1) * yz + z] = 0.0F;
      }
    }
  }
  // Fluid enters and leaves through the y faces.
  spec.y_lower = {.kind = microsimulator::GridBoundaryKind::fixed, .values = {0.0F}};
  spec.y_upper = {.kind = microsimulator::GridBoundaryKind::fixed, .values = {0.0F}};
  spec.velocity_field = std::move(field);
  DriftScene scene{microsimulator::WorldState(300), {}};
  scene.grid.emplace(spec);
  std::uint32_t state = 99;
  const auto next = [&state] {
    state = state * 1664525U + 1013904223U;
    return static_cast<float>(state >> 8) / static_cast<float>(1U << 24);
  };
  for (std::size_t cell = 0; cell < 250; ++cell) {
    Vec3 position{spec.origin.x + 29.0F * next(), spec.origin.y + 39.0F * next(), 2.0F * next()};
    if (position.x > -3.5F && position.x < 1.5F && position.y > -2.5F && position.y < 2.5F) {
      position.x -= 6.0F;
    }
    const auto angle = 6.2831853F * next();
    scene.state.add_cell({.position = position,
                          .direction = {std::cos(angle), std::sin(angle), 0.0F},
                          .length = 1.0F + 3.0F * next(),
                          .radius = 0.5F,
                          .fixed = cell % 11 == 0});
  }
  return scene;
}

struct DriftReference {
  std::vector<std::array<float, 6>> rates;
  std::vector<microsimulator::CellSnapshot> moved;
};

DriftReference run_drift(BackendKind backend, std::uint32_t device_index) {
  auto scene = drift_scene();
  auto compute = microsimulator::make_compute_backend(backend, device_index);
  DriftReference result;
  result.rates = compute->flow_drift_rates(scene.state, *scene.grid, 0.05F);
  compute->advance_flow_drift(scene.state, *scene.grid, 0.4F, {});
  result.moved = scene.state.cells();
  return result;
}

bool same_drift(const DriftReference& left, const DriftReference& right) {
  if (left.rates != right.rates || left.moved.size() != right.moved.size()) {
    return false;
  }
  for (std::size_t index = 0; index < left.moved.size(); ++index) {
    const auto& a = left.moved[index];
    const auto& b = right.moved[index];
    if (a.position.x != b.position.x || a.position.y != b.position.y ||
        a.position.z != b.position.z || a.direction.x != b.direction.x ||
        a.direction.y != b.direction.y || a.direction.z != b.direction.z) {
      return false;
    }
  }
  return true;
}

void require_close_drift(const DriftReference& actual, const DriftReference& expected,
                         const std::string& label) {
  require_near(flatten(actual.rates), flatten(expected.rates), backend_tolerance,
               label + " drift rates");
  require(actual.moved.size() == expected.moved.size(), label + ": drifted cells differ");
  std::vector<float> a;
  std::vector<float> b;
  for (std::size_t index = 0; index < actual.moved.size(); ++index) {
    for (const auto* cell : {&actual.moved[index], &expected.moved[index]}) {
      auto& values = cell == &actual.moved[index] ? a : b;
      values.insert(values.end(), {cell->position.x, cell->position.y, cell->position.z,
                                   cell->direction.x, cell->direction.y, cell->direction.z});
    }
  }
  require_near(a, b, backend_tolerance, label + " drifted geometry");
}

// Free cells move downstream and fixed cells stay where they are.
void test_drift_moves_free_cells(const DriftReference& reference) {
  const auto initial = drift_scene().state.cells();
  for (std::size_t index = 0; index < initial.size(); ++index) {
    const auto& before = initial[index];
    const auto& after = reference.moved[index];
    if (before.fixed) {
      require(before.position.x == after.position.x && before.position.y == after.position.y,
              "a fixed cell does not drift");
    } else {
      require(after.position.y > before.position.y, "a free cell drifts downstream");
    }
  }
}

}  // namespace

int main() {
  test_conservation_and_truncation();
  test_wall_separates_support();
  test_solid_support_is_rejected();
  test_closure_formula();

  microsimulator::set_cpu_thread_count(1);
  const auto reference = run(BackendKind::cpu, 0);
  const auto contacts = run_contacts(BackendKind::cpu, 0);
  const auto drift = run_drift(BackendKind::cpu, 0);
  test_container_faces(contacts);
  test_drift_moves_free_cells(drift);
  // The CPU gives the same bits on any number of threads.
  for (const std::size_t threads : {2U, 5U, 8U}) {
    microsimulator::set_cpu_thread_count(threads);
    const auto label = "cpu with " + std::to_string(threads) + " threads";
    require_same(reference, run(BackendKind::cpu, 0), label);
    const auto other = run_contacts(BackendKind::cpu, 0);
    require(same_contacts(contacts.nearest, other.nearest), label + ": nearest-face contacts differ");
    require(same_contacts(contacts.faces, other.faces), label + ": every-face contacts differ");
    require(same_drift(drift, run_drift(BackendKind::cpu, 0)), label + ": flow drift differs");
  }
  microsimulator::set_cpu_thread_count(0);
  // A device repeats its own results exactly and agrees with the CPU within
  // the backend tolerance.
  microsimulator::test::for_each_backend_device([&](BackendKind backend, std::uint32_t index) {
    const auto label =
        "backend " + std::to_string(static_cast<int>(backend)) + " device " + std::to_string(index);
    const auto deposit = run(backend, index);
    require_same(deposit, run(backend, index), label + " on a second run");
    require_close(deposit, reference, label);
    const auto device_contacts = run_contacts(backend, index);
    const auto repeated = run_contacts(backend, index);
    require(same_contacts(device_contacts.nearest, repeated.nearest) &&
                same_contacts(device_contacts.faces, repeated.faces),
            label + ": contacts differ on a second run");
    require_close_contacts(device_contacts.nearest, contacts.nearest, label + " nearest-face");
    require_close_contacts(device_contacts.faces, contacts.faces, label + " every-face");
    const auto device_drift = run_drift(backend, index);
    require(same_drift(device_drift, run_drift(backend, index)),
            label + ": flow drift differs on a second run");
    require_close_drift(device_drift, drift, label);
  });
  return 0;
}
