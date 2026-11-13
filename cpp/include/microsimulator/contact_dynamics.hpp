#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "microsimulator/types.hpp"

namespace microsimulator {

// How a cell's elongation responds to the axial compressive load its contacts
// put on it.
enum class GrowthLoadLaw : std::uint8_t {
  // Elongation ignores load.
  none,
  // Elongation is scaled by exp(-sensitivity * load).
  exponential,
  // Elongation is unscaled up to `free_load`, falls linearly, and stops at
  // `stall_load`.
  linear,
};

struct GrowthLoadResponse {
  GrowthLoadLaw law{GrowthLoadLaw::none};
  float sensitivity{0.0F};
  float free_load{0.0F};
  float stall_load{1.0F};
};

void validate_growth_load_response(const GrowthLoadResponse& response);

// The factor in [0, 1] that scales a cell's positive elongation under `load`.
[[nodiscard]] float growth_load_factor(const GrowthLoadResponse& response, float load);

struct ContactDynamicsParameters {
  // Drag per unit capsule length. A cell of full length L resists translation
  // with drag * L and rotation with the matching rod inertias.
  float drag{1.0F};
  // Contact force per unit overlap. Contacts are unilateral springs solved
  // implicitly, so a resting contact under force f overlaps by f / stiffness.
  float contact_stiffness{1.0e4F};
  // Force per unit overlap of a contact with a constraint. Walls are rigid
  // next to cells, so this is the stiffer of the two.
  float wall_stiffness{1.0e6F};
  // Overlap, beyond what the contact force accounts for, accepted at the end
  // of a substep.
  float overlap_tolerance{1.0e-3F};
  // Pairs and walls closer than this enter the solve as candidate rows.
  float candidate_margin{0.1F};
  // Existing overlap that a pair free to move recovers per substep at most.
  // A pair that cannot move is pressed apart with the full contact force of
  // its overlap.
  float max_overlap_recovery{0.05F};
  // Elongation of one cell end per substep at most, in cell radii.
  float max_growth_fraction{0.25F};
  std::uint32_t max_iterations{20000};
  std::uint32_t max_refinements{10};
  std::uint32_t max_step_halvings{12};
  // Carries the cells with the signal grid's velocity field inside the solve,
  // so that contacts resist the flow pressing cells together.
  bool flow_drift{false};
  GrowthLoadResponse growth;
};

void validate_contact_dynamics_parameters(const ContactDynamicsParameters& parameters);

// One unilateral contact. The gap rate of the row is
// first_jacobian . (velocity, angular velocity) of the first cell plus the
// same product for the second, minus each cell's length lever times its
// elongation rate. The row yields by `compliance` per unit of its multiplier.
// A row against a constraint has no second cell.
struct ContactRow {
  Slot first_slot{invalid_slot};
  Slot second_slot{invalid_slot};
  std::array<float, 6> first_jacobian{};
  std::array<float, 6> second_jacobian{};
  float first_length_lever{0.0F};
  float second_length_lever{0.0F};
  float offset{0.0F};
  float compliance{0.0F};
};

// Mobility, nominal elongation, and imposed motion of one cell. A fixed cell
// has zero mobility. `drift` is the velocity then angular velocity the cell
// has without contact, which contact forces add to.
struct ContactBody {
  Vec3 axis{1.0F, 0.0F, 0.0F};
  float translational_mobility{0.0F};
  float transverse_mobility{0.0F};
  float axial_mobility{0.0F};
  float length_rate{0.0F};
  std::array<float, 6> drift{};
};

// The convex problem of one substep. With multipliers g >= 0, the predicted
// gap of every row is
//   offset + dt * J^T (drift + M^-1 J g) - dt * L^T (length_rate * factor(L g)) + compliance * g
// and the solution has gap >= 0 with gap_i * g_i = 0.
struct ContactSystem {
  std::span<const ContactRow> rows;
  std::span<const ContactBody> bodies;
  float dt{0.0F};
  GrowthLoadResponse growth;
};

enum class ContactSolveStatus : std::uint8_t {
  converged,
  iteration_limit,
  breakdown,
};

struct ContactSolveReport {
  ContactSolveStatus status{ContactSolveStatus::converged};
  std::uint32_t iterations{0};
  // Largest violation of the complementarity conditions, as a length.
  float residual{0.0F};
};

struct ContactSolveParameters {
  float tolerance{2.5e-4F};
  std::uint32_t max_iterations{20000};
};

struct ContactSolveResult {
  std::vector<float> multipliers;
  ContactSolveReport report;
};

// What a set of multipliers does to the cells.
struct ContactResponse {
  // Velocity then angular velocity of every cell.
  std::vector<std::array<float, 6>> velocities;
  std::vector<float> loads;
  std::vector<float> length_rates;
};

void validate_contact_system(const ContactSystem& system);

[[nodiscard]] ContactResponse evaluate_contact_response(const ContactSystem& system,
                                                        std::span<const float> multipliers);

[[nodiscard]] std::vector<float> evaluate_contact_gaps(const ContactSystem& system,
                                                       std::span<const float> multipliers);

// The largest diagonal entry of the linear part of the system, which bounds
// the first solver step.
[[nodiscard]] float contact_system_diagonal_bound(const ContactSystem& system);

[[nodiscard]] ContactSolveResult solve_contact_system_cpu(
    const ContactSystem& system, std::span<const float> initial_multipliers,
    const ContactSolveParameters& parameters = ContactSolveParameters{});

struct ContactDynamicsReport {
  std::uint32_t substeps{0};
  std::uint32_t halvings{0};
  std::uint32_t refinements{0};
  std::uint64_t iterations{0};
  std::size_t max_rows{0};
  // Deepest overlap between cells or with a constraint after the step.
  float max_overlap{0.0F};
};

struct ContactDynamicsResult {
  ContactDynamicsReport report;
  // Axial compressive load on each cell, by slot, over the last substep.
  std::vector<float> axial_loads;
  // Elongation applied to each cell, by slot, over the whole step.
  std::vector<float> length_increments;
};

}  // namespace microsimulator
