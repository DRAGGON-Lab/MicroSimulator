#pragma once
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "cm/grid.hpp"
#include "cm/numerics.hpp"
#include "cm/types.hpp"

namespace cm {

// Coordinates use model lengths; time uses model time. Hydraulic boundary
// values, viscosity, density, pressures and flow rates use the SI units named
// below. A fluid grid has no dependency on the number of chemical signals.
struct FluidGridSpec {
  GridShape shape;
  Vec3 origin;
  float spacing{1.0F};
  double length_unit_m{1.0e-6};
  double time_unit_s{1.0};
  std::vector<std::uint8_t> obstacles;
  void validate() const;
  [[nodiscard]] std::size_t site_count() const;

  [[nodiscard]] bool solid_site(std::size_t i) const noexcept {
    return !obstacles.empty() && obstacles[i] != 0;
  }
};

struct FluidProperties {
  double viscosity_pa_s{1.0e-3};
  double density_kg_m3{1000.0};
  void validate() const;
};

enum class FlowPortKind : std::uint8_t { pressure, flow_rate };

struct FlowPort {
  std::string name;
  FlowAxis axis{FlowAxis::y};
  bool upper{false};
  FlowPortKind kind{FlowPortKind::pressure};
  // Pressure in Pa or flow in m^3/s. Flow is positive OUT OF the domain.
  double value{0.0};
  // Boundary-adjacent site indices, in x-major/z-minor order. Empty selects
  // every non-wall site on this boundary. Each port must belong to one fluid
  // component. Different ports may not own the same face.
  std::vector<std::uint32_t> sites;
};

struct FlowPortResult {
  std::string name;
  double pressure_pa{0};
  double flow_rate_m3_s{0};
  double area_m2{0};
};

struct CapsuleBody {
  CellId id{1};
  std::array<double, 3> position{0, 0, 0};
  // Unit quaternion (w,x,y,z); the local capsule axis is +x.
  std::array<double, 4> orientation{1, 0, 0, 0};
  double length{0}, radius{1};  // model lengths, cylindrical length excludes caps
  double length_rate{0};        // model length/model time, prescribed elongation
  bool fixed{false};
  std::array<double, 3> force_n{0, 0, 0}, torque_nm{0, 0, 0};
  void validate() const;
  [[nodiscard]] double geometric_volume() const;
};

struct FluidBodyResult {
  CellId id{0};
  std::array<double, 3> velocity{0, 0, 0};          // model length/model time
  std::array<double, 3> angular_velocity{0, 0, 0};  // radians/model time
  std::array<double, 3> hydrodynamic_force_n{0, 0, 0}, hydrodynamic_torque_nm{0, 0, 0};
  double no_slip_rms_m_s{0};
  double volume_change_rate_m3_s{0};
  std::uint32_t marker_count{0};
};

struct FluidSolveReport {
  std::uint32_t iterations{0};
  double relative_residual{0};
  double absolute_residual{0};
  double divergence_rms_per_s{0};
  double continuity_rms_per_s{0};
  double source_volume_rate_m3_s{0};
  double max_speed_m_s{0};
  double reynolds_number{0};
  double viscous_relaxation_time_s{0};
  double net_flow_rate_m3_s{0};
  std::uint64_t estimated_memory_bytes{0};
};

struct FluidFlowResult {
  // Velocity in model length/model time, for compatibility with transport.
  MacVelocityField field;
  std::vector<double> pressure_pa;
  std::vector<FlowPortResult> ports;
  std::vector<FluidBodyResult> bodies;
  FluidSolveReport report;
};

struct FluidBodyStepParameters {
  double minimum_gap_m{1e-8};
  double maximum_displacement_fraction{0.25};
  std::uint32_t max_halvings{16};
  std::uint32_t max_contact_iterations{64};
  void validate() const;
};

struct FluidContactResult {
  CellId first_id{0}, second_id{0};  // second_id zero denotes a device wall
  std::array<double, 3> normal{0, 0, 0}, point_on_first{0, 0, 0};
  // normal points from first to second; force on first is -normal_force_n*normal.
  double initial_gap_m{0};
  double normal_force_n{0};
};

struct FluidBodyStep {
  double accepted_dt{0};  // model time; may be smaller than maximum_dt
  std::vector<CapsuleBody> bodies;
  FluidFlowResult flow;
  std::vector<FluidContactResult> contacts;
  std::uint32_t halvings{0}, contact_iterations{0};
};

// Standalone physical device-flow and rigid-body solve. Simulation mode and
// state changes are managed by the caller.
class StokesFlowSolver {
 public:
  explicit StokesFlowSolver(BackendKind backend = BackendKind::cpu, std::uint32_t device_index = 0);
  ~StokesFlowSolver();
  StokesFlowSolver(const StokesFlowSolver&) = delete;
  StokesFlowSolver& operator=(const StokesFlowSolver&) = delete;
  [[nodiscard]] FluidFlowResult solve(
      const FluidGridSpec& grid, const FluidProperties& fluid, const std::vector<FlowPort>& ports,
      const LinearSolveParameters& parameters = LinearSolveParameters{});
  [[nodiscard]] FluidFlowResult solve_bodies(
      const FluidGridSpec& grid, const FluidProperties& fluid, const std::vector<FlowPort>& ports,
      const std::vector<CapsuleBody>& bodies,
      const LinearSolveParameters& parameters = LinearSolveParameters{});
  // Propose one bounded substep. The caller commits the returned bodies and
  // advances its clock by accepted_dt only after coupled transport succeeds.
  [[nodiscard]] FluidBodyStep propose_body_step(
      const FluidGridSpec& grid, const FluidProperties& fluid, const std::vector<FlowPort>& ports,
      const std::vector<CapsuleBody>& bodies, double maximum_dt,
      const LinearSolveParameters& solve_parameters = LinearSolveParameters{},
      const FluidBodyStepParameters& step_parameters = FluidBodyStepParameters{});

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace cm
