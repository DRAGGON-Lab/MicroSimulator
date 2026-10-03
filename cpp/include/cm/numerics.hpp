#pragma once
#include <cstdint>

namespace cm {
struct LinearSolveParameters {
  double relative_tolerance{1.0e-5};
  double absolute_tolerance{1.0e-7};
  std::uint32_t max_iterations{5000};
  std::uint64_t memory_limit_bytes{1024ULL * 1024 * 1024};
  void validate() const;
};

}  // namespace cm
