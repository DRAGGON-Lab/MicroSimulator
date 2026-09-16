#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace cm {

class WorldState;

enum class RateOp : std::uint8_t {
  constant = 0,
  species = 1,
  position_x = 2,
  position_y = 3,
  position_z = 4,
  cell_length = 5,
  cell_radius = 6,
  growth_rate = 7,
  cell_type = 8,
  cell_volume = 9,
  cell_surface_area = 10,
  add = 11,
  subtract = 12,
  multiply = 13,
  divide = 14,
  power = 15,
  minimum = 16,
  maximum = 17,
  negate = 18,
  exponential = 19,
  logarithm = 20,
  less = 21,
  less_equal = 22,
  greater = 23,
  greater_equal = 24,
  equal = 25,
  select = 26,
  signal = 27,
  cell_volume_change_rate = 28,
};

struct RateInstruction {
  RateOp operation{RateOp::constant};
  std::uint32_t first{0};
  std::uint32_t second{0};
  std::uint32_t third{0};
  float value{0.0F};
};

class SpeciesRatePlan {
 public:
  SpeciesRatePlan() = default;
  SpeciesRatePlan(std::size_t species_count, std::vector<RateInstruction> instructions,
                  std::vector<std::uint32_t> outputs);

  [[nodiscard]] static SpeciesRatePlan zero(std::size_t species_count);

  [[nodiscard]] std::size_t species_count() const noexcept;
  [[nodiscard]] std::span<const RateInstruction> instructions() const& noexcept;
  [[nodiscard]] std::span<const RateInstruction> instructions() && = delete;
  [[nodiscard]] std::span<const std::uint32_t> outputs() const& noexcept;
  [[nodiscard]] std::span<const std::uint32_t> outputs() && = delete;
  void validate() const;

 private:
  std::size_t species_count_{0};
  std::vector<RateInstruction> instructions_;
  std::vector<std::uint32_t> outputs_;
};

[[nodiscard]] float effective_cell_volume(float length, float radius) noexcept;
[[nodiscard]] float effective_cell_surface_area(float length, float radius) noexcept;

void advance_species_cpu(WorldState& state, const SpeciesRatePlan& plan,
                         std::span<const float> previous_lengths, float dt);

}  // namespace cm
