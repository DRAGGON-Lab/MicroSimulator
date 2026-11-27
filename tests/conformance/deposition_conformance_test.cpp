#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend_devices.hpp"
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
    std::fprintf(stderr, "deposition conformance: %s\n", message.c_str());
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

// Identical results, as one backend gives on every run and the CPU gives on
// any number of threads.
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

std::vector<float> flatten(const std::vector<std::array<float, 2>>& pairs) {
  std::vector<float> values;
  values.reserve(2 * pairs.size());
  for (const auto& pair : pairs) {
    values.push_back(pair[0]);
    values.push_back(pair[1]);
  }
  return values;
}

void require_close(const Reference& actual, const Reference& expected, const std::string& label) {
  constexpr double tolerance = 1e-5;
  require_near(actual.deposit.density, expected.deposit.density, tolerance, label + " density");
  require_near(actual.deposit.channels, expected.deposit.channels, tolerance,
               label + " channels");
  require_near(actual.coupling.mobility, expected.coupling.mobility, tolerance,
               label + " mobility");
  require_near(flatten(actual.coupling.body_force), flatten(expected.coupling.body_force), tolerance,
               label + " body force");
  require_near(actual.drag, expected.drag, tolerance, label + " Brinkman drag");
}

}  // namespace

int main() {
  test_conservation_and_truncation();
  test_wall_separates_support();
  test_solid_support_is_rejected();
  test_closure_formula();

  microsimulator::set_cpu_thread_count(1);
  const auto reference = run(BackendKind::cpu, 0);
  for (const std::size_t threads : {2U, 5U, 8U}) {
    microsimulator::set_cpu_thread_count(threads);
    require_same(reference, run(BackendKind::cpu, 0),
                 "cpu with " + std::to_string(threads) + " threads");
  }
  microsimulator::set_cpu_thread_count(0);
  microsimulator::test::for_each_backend_device([&](BackendKind backend, std::uint32_t index) {
    const auto label =
        "backend " + std::to_string(static_cast<int>(backend)) + " device " + std::to_string(index);
    const auto first = run(backend, index);
    require_same(first, run(backend, index), label + " on a second run");
    require_close(first, reference, label);
  });
  return 0;
}
