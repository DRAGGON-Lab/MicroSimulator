#include "cm/coupled_rates.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace cm {
namespace {

void checked_index(std::size_t index, const char* name) {
  if (index > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error(std::string("coupled rate ") + name +
                              " exceeds the uint32 index space");
  }
}

void validate_input(std::uint32_t input, std::size_t instruction_index, const char* name) {
  if (input >= instruction_index) {
    throw std::invalid_argument(std::string("coupled rate ") + name +
                                " input must refer to an earlier instruction");
  }
}

}  // namespace

CoupledRatePlan::CoupledRatePlan(std::size_t species_count, std::size_t signal_count,
                                 std::vector<RateInstruction> instructions,
                                 std::vector<std::uint32_t> species_outputs,
                                 std::vector<std::uint32_t> signal_outputs)
    : species_count_(species_count),
      signal_count_(signal_count),
      instructions_(std::move(instructions)),
      species_outputs_(std::move(species_outputs)),
      signal_outputs_(std::move(signal_outputs)) {
  validate();
}

std::size_t CoupledRatePlan::species_count() const noexcept { return species_count_; }

std::size_t CoupledRatePlan::signal_count() const noexcept { return signal_count_; }

std::span<const RateInstruction> CoupledRatePlan::instructions() const& noexcept {
  return instructions_;
}

std::span<const std::uint32_t> CoupledRatePlan::species_outputs() const& noexcept {
  return species_outputs_;
}

std::span<const std::uint32_t> CoupledRatePlan::signal_outputs() const& noexcept {
  return signal_outputs_;
}

void CoupledRatePlan::validate() const {
  checked_index(species_count_, "species count");
  checked_index(signal_count_, "signal count");
  checked_index(instructions_.size(), "instruction count");
  if (signal_count_ == 0) {
    throw std::invalid_argument("a coupled rate plan needs at least one signal");
  }
  if (species_outputs_.size() != species_count_) {
    throw std::invalid_argument("coupled species output count must match the species count");
  }
  if (signal_outputs_.size() != signal_count_) {
    throw std::invalid_argument("coupled signal output count must match the signal count");
  }
  if (instructions_.empty()) {
    throw std::invalid_argument("a coupled rate plan needs instructions");
  }

  for (std::size_t index = 0; index < instructions_.size(); ++index) {
    const auto& instruction = instructions_[index];
    switch (instruction.operation) {
      case RateOp::constant:
        if (!std::isfinite(instruction.value)) {
          throw std::invalid_argument("coupled rate constants must be finite");
        }
        break;
      case RateOp::species:
        if (instruction.first >= species_count_) {
          throw std::invalid_argument("coupled rate instruction uses an unknown species");
        }
        break;
      case RateOp::signal:
        if (instruction.first >= signal_count_) {
          throw std::invalid_argument("coupled rate instruction uses an unknown signal");
        }
        break;
      case RateOp::position_x:
      case RateOp::position_y:
      case RateOp::position_z:
      case RateOp::cell_length:
      case RateOp::cell_radius:
      case RateOp::growth_rate:
      case RateOp::cell_type:
      case RateOp::cell_volume:
      case RateOp::cell_volume_change_rate:
      case RateOp::cell_surface_area:
        break;
      case RateOp::negate:
      case RateOp::exponential:
      case RateOp::logarithm:
        validate_input(instruction.first, index, "unary");
        break;
      case RateOp::add:
      case RateOp::subtract:
      case RateOp::multiply:
      case RateOp::divide:
      case RateOp::power:
      case RateOp::minimum:
      case RateOp::maximum:
      case RateOp::less:
      case RateOp::less_equal:
      case RateOp::greater:
      case RateOp::greater_equal:
      case RateOp::equal:
        validate_input(instruction.first, index, "first");
        validate_input(instruction.second, index, "second");
        break;
      case RateOp::select:
        validate_input(instruction.first, index, "condition");
        validate_input(instruction.second, index, "true");
        validate_input(instruction.third, index, "false");
        break;
      default:
        throw std::invalid_argument("coupled rate plan uses an unknown operation");
    }
  }

  for (const auto output : species_outputs_) {
    if (output >= instructions_.size()) {
      throw std::invalid_argument("coupled species output uses an unknown instruction");
    }
  }
  for (const auto output : signal_outputs_) {
    if (output >= instructions_.size()) {
      throw std::invalid_argument("coupled signal output uses an unknown instruction");
    }
  }
}

}  // namespace cm
