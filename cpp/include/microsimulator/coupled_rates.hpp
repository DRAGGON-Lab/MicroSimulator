#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "microsimulator/species.hpp"

namespace microsimulator {

class SignalGrid;
class WorldState;
struct SignalSolveReport;

class CoupledRatePlan {
 public:
  CoupledRatePlan() = default;
  CoupledRatePlan(std::size_t species_count, std::size_t signal_count,
                  std::vector<RateInstruction> instructions,
                  std::vector<std::uint32_t> species_outputs,
                  std::vector<std::uint32_t> signal_outputs);

  [[nodiscard]] std::size_t species_count() const noexcept;
  [[nodiscard]] std::size_t signal_count() const noexcept;
  [[nodiscard]] std::span<const RateInstruction> instructions() const& noexcept;
  [[nodiscard]] std::span<const RateInstruction> instructions() && = delete;
  [[nodiscard]] std::span<const std::uint32_t> species_outputs() const& noexcept;
  [[nodiscard]] std::span<const std::uint32_t> species_outputs() && = delete;
  [[nodiscard]] std::span<const std::uint32_t> signal_outputs() const& noexcept;
  [[nodiscard]] std::span<const std::uint32_t> signal_outputs() && = delete;
  void validate() const;

 private:
  std::size_t species_count_{0};
  std::size_t signal_count_{0};
  std::vector<RateInstruction> instructions_;
  std::vector<std::uint32_t> species_outputs_;
  std::vector<std::uint32_t> signal_outputs_;
};

void validate_coupled_step(const WorldState& state, const SignalGrid& grid,
                           const CoupledRatePlan& plan, std::span<const float> previous_lengths,
                           float dt);
[[nodiscard]] SignalSolveReport advance_coupled_cpu(WorldState& state, SignalGrid& grid,
                                                    const CoupledRatePlan& plan,
                                                    std::span<const float> previous_lengths,
                                                    float dt);

}  // namespace microsimulator
