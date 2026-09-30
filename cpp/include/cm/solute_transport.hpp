#pragma once

#include "cm/transport_geometry.hpp"

namespace cm {
struct Solute {
  std::string name;
  double diffusion{0};  // model length squared / model time
  std::string amount_unit{"model"};
};
enum class ChemicalBoundaryKind : std::uint8_t { reservoir_contact, advective, outflow };

struct ChemicalBoundary {
  std::string port;
  std::vector<double> concentrations;  // amount / model volume
  ChemicalBoundaryKind kind{ChemicalBoundaryKind::reservoir_contact};
  bool allow_backflow{true};
};

struct SurfaceTransferLaw {
  CellId body_id{0};
  std::uint32_t solute{0};
  double uptake_velocity{0};  // model length / model time
  double secretion_rate{0};   // total amount / model time, distributed by area
};

struct ChemicalTransfer {
  std::string port;
  CellId body_id{0};
  // Reservoir: positive into fluid. Cell: positive into cell.
  std::vector<double> amounts;
};

struct SoluteTransportReport {
  std::uint32_t projection_iterations{0}, transport_iterations{0};
  double maximum_volume_residual{0};
  std::vector<double> mass_balance_error;
};

struct SurfaceEnvironment {
  CellId body_id{0};
  double area{0};
  std::vector<double> concentrations;
};

struct SoluteTransportResult {
  // Fragment-major, solute-minor. Amounts are authoritative.
  std::vector<double> amounts, concentrations;
  std::vector<ChemicalTransfer> reservoirs, cells;
  SoluteTransportReport report;
  std::vector<SurfaceEnvironment> surfaces;
};

class SoluteTransportSolver {
 public:
  explicit SoluteTransportSolver(BackendKind backend = BackendKind::cpu,
                                 std::uint32_t device_index = 0);
  ~SoluteTransportSolver();
  SoluteTransportSolver(const SoluteTransportSolver&) = delete;
  SoluteTransportSolver& operator=(const SoluteTransportSolver&) = delete;
  [[nodiscard]] SoluteTransportResult step(const FluidGeometry& before, const FluidGeometry& after,
                                           const MacVelocityField& velocity,
                                           const std::vector<FlowPort>& ports,
                                           const std::vector<Solute>& solutes,
                                           const std::vector<ChemicalBoundary>& reservoirs,
                                           const std::vector<double>& amounts, double dt,
                                           const std::vector<SurfaceTransferLaw>& exchange = {},
                                           const LinearSolveParameters& parameters = {});

  [[nodiscard]] SoluteTransportResult propose(const TransportGeometry&, const std::vector<Solute>&,
                                              const std::vector<ChemicalBoundary>&,
                                              const std::vector<double>& amounts,
                                              const std::vector<SurfaceTransferLaw>& exchange = {},
                                              const LinearSolveParameters& parameters = {});

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace cm
