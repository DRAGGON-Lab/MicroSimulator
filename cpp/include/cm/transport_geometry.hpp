#pragma once
#include "cm/fluid_geometry.hpp"

namespace cm {
namespace detail {
class NumericsDevice;
}

struct GeometricFluxReport {
  std::uint32_t projection_iterations{0};
  double maximum_volume_residual{0};
};

// Immutable conservative space-time mesh. No species or biology is required.
class TransportGeometry {
 public:
  TransportGeometry(const FluidGeometry& before, const FluidGeometry& after,
                    const MacVelocityField&, const std::vector<FlowPort>&, double dt,
                    BackendKind backend = BackendKind::cpu, std::uint32_t device_index = 0,
                    const LinearSolveParameters& parameters = {});
  const GeometricFluxReport& report() const;

 private:
  friend class SoluteTransportSolver;
  struct Impl;
  std::shared_ptr<const Impl> impl_;
  explicit TransportGeometry(std::shared_ptr<const Impl>);
  static TransportGeometry prepare(const FluidGeometry&, const FluidGeometry&,
                                   const MacVelocityField&, const std::vector<FlowPort>&, double,
                                   const LinearSolveParameters&, detail::NumericsDevice*);
};
}  // namespace cm
