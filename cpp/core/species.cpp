#include "cm/species.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "rate_operations.hpp"

namespace cm {
void BiochemicalVolumeView::validate(std::size_t count) const {
  if (previous.empty() && current.empty()) {
    return;
  }

  if (previous.size() != count || current.size() != count) {
    throw std::invalid_argument("biochemical volume count disagrees with cells");
  }

  for (auto values : {previous, current}) {
    for (float v : values) {
      if (!std::isfinite(v) || v <= 0) {
        throw std::invalid_argument("biochemical volumes must be finite and positive");
      }
    }
  }
}

namespace {

std::uint32_t checked_index(std::size_t index) {
  if (index > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error("species rate plan exceeds the uint32 index space");
  }

  return static_cast<std::uint32_t>(index);
}

void validate_input(std::uint32_t input, std::size_t instruction_index, const char* name) {
  if (input >= instruction_index) {
    throw std::invalid_argument(std::string("species rate ") + name +
                                " input must refer to an earlier instruction");
  }
}

void validate_instruction(const RateInstruction& instruction, std::size_t index,
                          std::size_t species_count) {
  if (detail::is_cell_property_operation(instruction.operation)) {
    return;
  }

  if (detail::is_unary_operation(instruction.operation)) {
    validate_input(instruction.first, index, "unary");

    return;
  }

  if (detail::is_binary_operation(instruction.operation)) {
    validate_input(instruction.first, index, "first");
    validate_input(instruction.second, index, "second");

    return;
  }

  switch (instruction.operation) {
    case RateOp::constant:
      if (!std::isfinite(instruction.value)) {
        throw std::invalid_argument("species rate constants must be finite");
      }
      break;
    case RateOp::species:
      if (instruction.first >= species_count) {
        throw std::invalid_argument("species rate instruction uses an unknown species");
      }
      break;
    case RateOp::signal:
      throw std::invalid_argument("standalone species rate plans cannot sample signals");
    case RateOp::select:
      validate_input(instruction.first, index, "condition");
      validate_input(instruction.second, index, "true");
      validate_input(instruction.third, index, "false");
      break;
    default:
      throw std::invalid_argument("species rate plan uses an unknown operation");
  }
}

}  // namespace

SpeciesRatePlan::SpeciesRatePlan(std::size_t species_count,
                                 std::vector<RateInstruction> instructions,
                                 std::vector<std::uint32_t> outputs)
    : species_count_(species_count),
      instructions_(std::move(instructions)),
      outputs_(std::move(outputs)) {
  validate();
}

SpeciesRatePlan SpeciesRatePlan::zero(std::size_t species_count) {
  if (species_count == 0) {
    return {};
  }

  std::vector<RateInstruction> instructions{{.operation = RateOp::constant}};

  return SpeciesRatePlan(species_count, std::move(instructions),
                         std::vector<std::uint32_t>(species_count, 0));
}

std::size_t SpeciesRatePlan::species_count() const noexcept {
  return species_count_;
}

std::span<const RateInstruction> SpeciesRatePlan::instructions() const& noexcept {
  return instructions_;
}

std::span<const std::uint32_t> SpeciesRatePlan::outputs() const& noexcept {
  return outputs_;
}

void SpeciesRatePlan::validate() const {
  checked_index(species_count_);
  checked_index(instructions_.size());

  if (outputs_.size() != species_count_) {
    throw std::invalid_argument("species rate output count must match the species count");
  }

  if (species_count_ != 0 && instructions_.empty()) {
    throw std::invalid_argument("a nonempty species rate plan needs instructions");
  }

  for (std::size_t index = 0; index < instructions_.size(); ++index) {
    const auto& instruction = instructions_[index];

    validate_instruction(instruction, index, species_count_);
  }

  for (const auto output : outputs_) {
    if (output >= instructions_.size()) {
      throw std::invalid_argument("species rate output uses an unknown instruction");
    }
  }
}

float effective_cell_volume(float length, float radius) noexcept {
  // Conserved biomass volume for the endpoint-preserving division rule.
  // This is a biochemical measure, not the geometric capsule volume.
  constexpr float pi = 3.14159265358979323846F;

  return pi * radius * radius * (length + 2.0F * radius);
}

float effective_cell_surface_area(float length, float radius) noexcept {
  constexpr float pi = 3.14159265358979323846F;

  return 2.0F * pi * radius * (length + 2.0F * radius);
}

}  // namespace cm
