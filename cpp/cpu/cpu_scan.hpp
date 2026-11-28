#pragma once

#include <cstdint>
#include <numeric>
#include <vector>

namespace microsimulator::detail {

// The inclusive prefix sum of `values`. Integer sums are exact, so the
// result is the same however the sum is grouped.
[[nodiscard]] inline std::vector<std::uint32_t> inclusive_scan(std::vector<std::uint32_t> values) {
  std::inclusive_scan(values.begin(), values.end(), values.begin());
  return values;
}

}  // namespace microsimulator::detail
