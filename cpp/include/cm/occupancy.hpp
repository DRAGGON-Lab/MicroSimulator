#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "cm/types.hpp"

namespace cm {

struct OccupancyCapsule {
  std::array<float, 3> center;
  std::array<float, 3> direction;
  float length;
  float radius;
};

struct OccupancyFace {
  std::uint32_t first;
  std::uint32_t second;
  float conductance;
  float volume_flux{0};
};

struct OccupancyReservoir {
  std::uint32_t site;
  float concentration;
  float conductance{0};
  float volume_flux{0};
};

struct OccupancyBalance {
  double before{0}, after{0}, source{0}, reaction{0}, boundary{0};
  [[nodiscard]] double residual() const noexcept {
    return after - before - source - reaction - boundary;
  }
};

struct OccupancyStep {
  std::vector<float> amount;
  OccupancyBalance balance;
  std::uint32_t iterations{0};
  double relative_residual{0};
};

namespace detail {
class OccupancyDevice;
}

// Standalone native counterpart of ADR 0025's reference. This object does not
// change Simulation, controller staging, flow interpretation, or checkpoints.
// Every operation returns a candidate; caller-owned inputs are never modified.
class OccupancySolver {
 public:
  explicit OccupancySolver(BackendKind backend, std::uint32_t device_index = 0,
                           float epsilon_cutoff = 1e-8F);
  ~OccupancySolver();
  OccupancySolver(const OccupancySolver&) = delete;
  OccupancySolver& operator=(const OccupancySolver&) = delete;

  [[nodiscard]] std::vector<float> geometric_porosity(
      const std::vector<std::array<float, 3>>& centers, std::array<float, 3> spacing,
      const std::vector<OccupancyCapsule>& cells, std::uint32_t subdivisions = 8,
      const std::vector<std::uint32_t>& walls = {});
  [[nodiscard]] std::vector<float> accessible_volumes(const std::vector<float>& porosity,
                                                      float voxel_volume);
  [[nodiscard]] std::vector<float> concentration(const std::vector<float>& amount,
                                                 const std::vector<float>& volume);
  [[nodiscard]] OccupancyFace porosity_face(std::uint32_t first, std::uint32_t second,
                                            float epsilon_first, float epsilon_second,
                                            float diffusion, float area, float distance,
                                            float intrinsic_velocity = 0);
  [[nodiscard]] std::vector<float> remap_amounts(
      const std::vector<float>& amount, const std::vector<float>& old_volume,
      const std::vector<float>& new_volume,
      const std::vector<std::pair<std::uint32_t, std::uint32_t>>& neighbors);
  [[nodiscard]] std::vector<float> exchange_weights(const std::vector<float>& kernel,
                                                    const std::vector<float>& volume);
  [[nodiscard]] OccupancyStep backward_euler(
      const std::vector<float>& amount, const std::vector<float>& volume,
      const std::vector<OccupancyFace>& faces, float dt, const std::vector<float>& source = {},
      const std::vector<float>& loss = {}, const std::vector<OccupancyReservoir>& reservoirs = {},
      std::uint32_t max_iterations = 20000, float relative_tolerance = 1e-7F);

 private:
  std::unique_ptr<detail::OccupancyDevice> device_;
  float cutoff_;
};

}  // namespace cm
