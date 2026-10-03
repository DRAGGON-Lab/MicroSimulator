#pragma once

#include "cm/species.hpp"

namespace cm::detail {

inline bool is_cell_property_operation(RateOp operation) {
  switch (operation) {
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
      return true;
    default:
      return false;
  }
}

inline bool is_unary_operation(RateOp operation) {
  switch (operation) {
    case RateOp::negate:
    case RateOp::exponential:
    case RateOp::logarithm:
      return true;
    default:
      return false;
  }
}

inline bool is_binary_operation(RateOp operation) {
  switch (operation) {
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
      return true;
    default:
      return false;
  }
}

}  // namespace cm::detail
