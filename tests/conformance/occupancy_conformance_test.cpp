#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "cm/backend.hpp"
#include "cm/occupancy.hpp"

namespace {
void near(double actual, double expected, double tolerance = 2e-6) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
    throw std::runtime_error("occupancy conformance mismatch");
  }
}

void check(cm::BackendKind backend, std::uint32_t device) {
  cm::OccupancySolver solver(backend, device);
  const std::vector<std::array<float, 3>> centers{{0, 0, 0}, {1, 0, 0}, {2, 0, 0}};
  auto epsilon = solver.geometric_porosity(centers, {1, 1, 1}, {}, 8, {0, 1, 0});
  near(epsilon[0], 1);
  near(epsilon[1], 0);
  near(epsilon[2], 1);
  const cm::OccupancyCapsule cell{{0, 0, 0}, {2, 1, 0}, 0.5F, 0.4F};
  const auto single = solver.geometric_porosity(centers, {1, 1, 1}, {cell});
  const auto overlap = solver.geometric_porosity(centers, {1, 1, 1}, {cell, cell});

  if (single != overlap || single[0] >= 1) {
    throw std::runtime_error("capsule union mismatch");
  }

  auto remapped = solver.remap_amounts({2, 3, 1}, {1, 1, 1}, {0, 0.25F, 0.75F}, {{0, 1}, {1, 2}});
  near(remapped[0], 0);
  near(remapped[1], 3.5);
  near(remapped[2], 2.5);
  bool rejected = false;

  try {
    static_cast<void>(solver.remap_amounts({1, 0, 0}, {1, 0, 1}, {0, 0, 1}, {{0, 1}, {1, 2}}));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }

  if (!rejected) {
    throw std::runtime_error("closed component was accepted");
  }

  auto weights = solver.exchange_weights({0.5F, 0.5F}, {0.25F, 0.75F});
  near(weights[0], 0.25);
  near(weights[1], 0.75);
  const auto face = solver.porosity_face(0, 1, 0.25F, 0.75F, 1, 1, 1);
  near(face.conductance, 0.375);
  auto amount = std::vector<float>{4, 0};

  for (int step = 0; step < 1000; ++step) {
    const auto result = solver.backward_euler(amount, {0.5F, 1.5F}, {face}, 1);
    near(result.balance.residual(), 0, 2e-5);
    amount = result.amount;
  }

  near(amount[0], 1, 2e-4);
  near(amount[1], 3, 2e-4);
  near(double(amount[0]) + amount[1], 4, 2e-4);
  const auto open =
      solver.backward_euler({0.5F, 3}, {0.25F, 0.75F}, {{0, 1, 0.1F, 0.05F}}, 0.2F, {0.75F, 2.25F},
                            {0.3F, 0.7F}, {{0, 5, 0.2F, -0.1F}, {1, 0, 0, 0.1F}});
  near(open.balance.residual(), 0, 2e-5);

  if (!(open.balance.source > 0 && open.balance.reaction < 0 && open.balance.boundary > 0)) {
    throw std::runtime_error("occupancy ledger signs differ");
  }

  std::cout << "occupancy backend " << static_cast<int>(backend) << " device " << device
            << " passed\n";
}
}  // namespace

int main() {
  std::size_t tested = 0;

  for (const auto backend : {cm::BackendKind::metal, cm::BackendKind::cuda}) {
    for (std::uint32_t device = 0; device < cm::backend_device_count(backend); ++device) {
      check(backend, device);
      ++tested;
    }
  }

  if (!tested) {
    std::cout << "No Metal or CUDA device available\n";

    return 77;
  }
}
