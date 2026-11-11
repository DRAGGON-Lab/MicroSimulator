#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace cm {

class WorldState;

enum class RateOp : std::uint8_t {
#define CM_RATE_OP(name, code) name = code,
#include "cm/rate_ops.def"
#undef CM_RATE_OP
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
