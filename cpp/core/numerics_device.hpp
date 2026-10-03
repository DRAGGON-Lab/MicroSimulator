#pragma once

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <vector>

#include "cm/numerics.hpp"
#include "cm/types.hpp"

namespace cm::detail {

// Native vector operations shared by fluid, transport, and growth. Reduction partials
// cross the device boundary during a solve; all Krylov vectors stay resident.
enum class NumericsKernel { fill, copy, axpy, dot, apply, jacobi, ilu, growth, conservative_apply };
inline constexpr const char* numerics_kernel_names[] = {
    "numerics_fill",   "numerics_copy", "numerics_axpy",   "numerics_dot",      "numerics_apply",
    "numerics_jacobi", "numerics_ilu",  "nutrient_growth", "conservative_apply"};

struct NumericsParameters {
  std::uint32_t count{0};
  float scalar{0};
};

struct NumericsBuffer {
  virtual ~NumericsBuffer() = default;
};

using NumericsBufferPtr = std::shared_ptr<NumericsBuffer>;

class NumericsDevice {
 public:
  virtual ~NumericsDevice() = default;
  virtual NumericsBufferPtr allocate(std::size_t bytes, const void* data = nullptr) = 0;
  virtual void read(const NumericsBufferPtr&, void*, std::size_t bytes) = 0;
  virtual void dispatch(NumericsKernel, const NumericsParameters&,
                        std::initializer_list<NumericsBufferPtr>) = 0;

  template <class T>
  NumericsBufferPtr upload(const std::vector<T>& values) {
    return allocate(values.size() * sizeof(T), values.empty() ? nullptr : values.data());
  }

  template <class T>
  std::vector<T> download(const NumericsBufferPtr& buffer, std::size_t count) {
    std::vector<T> result(count);

    if (count) {
      read(buffer, result.data(), count * sizeof(T));
    }

    return result;
  }
};

std::unique_ptr<NumericsDevice> make_metal_numerics_device(std::uint32_t);
std::unique_ptr<NumericsDevice> make_cuda_numerics_device(std::uint32_t);
std::unique_ptr<NumericsDevice> make_numerics_device(BackendKind, std::uint32_t);
using NumericsLinearRows = std::vector<std::vector<std::pair<std::uint32_t, double>>>;

struct NumericsLinearResult {
  std::vector<double> solution;
  std::uint32_t iterations{0};
  double relative_residual{0}, absolute_residual{0};
};

NumericsLinearResult solve_numerics_linear(NumericsDevice*, NumericsLinearRows rows,
                                           std::vector<double> rhs, const LinearSolveParameters&);

}  // namespace cm::detail
