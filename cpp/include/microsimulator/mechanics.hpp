#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "microsimulator/constraints.hpp"
#include "microsimulator/contact_graph.hpp"
#include "microsimulator/types.hpp"
#include "microsimulator/world_state.hpp"

namespace microsimulator {

struct CellCorrection {
  Vec3 translation{};
  Vec3 rotation{};
  float length{0.0F};
};

struct MechanicsParameters {
  float mu_a{1.0F};
  float gamma{10.0F};
  float residual_rms_tolerance{5.0e-3F};
  std::uint32_t max_iterations{0};
};

enum class SolverStatus : std::uint8_t {
  converged,
  iteration_limit,
  breakdown,
};

enum class SolverBreakdown : std::uint8_t {
  none,
  non_finite_residual,
  non_finite_curvature,
  non_positive_curvature,
};

struct SolverReport {
  SolverStatus status{SolverStatus::converged};
  SolverBreakdown breakdown{SolverBreakdown::none};
  std::uint32_t iterations{0};
  float initial_residual_rms{0.0F};
  float final_residual_rms{0.0F};
};

struct MechanicsSolveResult {
  std::vector<CellCorrection> corrections;
  SolverReport report;
};

struct MechanicsIntegrationParameters {
  float max_rotation_radians{0.0872664626F};
  bool require_convergence{true};
};

// Growth that responds to compression. A cell's elongation over a step is
// scaled by a factor that is one while its deepest overlap with a neighbor is
// at most `free_overlap` cell radii and falls linearly to zero at
// `stall_overlap` radii. Contact relaxation leaves a small overlap between
// growing neighbors that are free to move, which the free band ignores; a cell
// that relaxation cannot separate from its neighbors stops elongating.
struct GrowthLimitParameters {
  float free_overlap{0.1F};
  float stall_overlap{0.4F};
  ContactParameters contacts;
};

void validate_mechanics_parameters(const MechanicsParameters& parameters);
void validate_mechanics_integration_parameters(const MechanicsIntegrationParameters& parameters);
void validate_growth_limit_parameters(const GrowthLimitParameters& parameters);

// The factor by which compression scales each cell's elongation, one per slot.
[[nodiscard]] std::vector<float> growth_limit_factors(const WorldState& state,
                                                      const ContactGraph& contacts,
                                                      const GrowthLimitParameters& parameters);

// The axis-angle rotation of a unit direction by a rotation vector, with the
// angle capped at ``max_rotation``.
[[nodiscard]] Vec3 rotate_axis_angle(Vec3 direction, Vec3 rotation, float max_rotation);

void integrate_mechanics_result(
    WorldState& state, const MechanicsSolveResult& result,
    const MechanicsIntegrationParameters& parameters = MechanicsIntegrationParameters{},
    std::span<const float> desired_length_increments = {});

[[nodiscard]] std::vector<CellCorrection> apply_mechanics_operator_cpu(
    const WorldState& state, const ContactGraph& contacts,
    const ExternalContactGraph& external_contacts, std::span<const CellCorrection> input,
    const MechanicsParameters& parameters = MechanicsParameters{});

[[nodiscard]] std::vector<CellCorrection> apply_mechanics_operator_cpu(
    const WorldState& state, const ContactGraph& contacts, std::span<const CellCorrection> input,
    const MechanicsParameters& parameters = MechanicsParameters{});

[[nodiscard]] std::vector<CellCorrection> build_mechanics_rhs_cpu(
    const WorldState& state, const ContactGraph& contacts,
    const ExternalContactGraph& external_contacts,
    const MechanicsParameters& parameters = MechanicsParameters{});

[[nodiscard]] std::vector<CellCorrection> build_mechanics_rhs_cpu(
    const WorldState& state, const ContactGraph& contacts,
    const MechanicsParameters& parameters = MechanicsParameters{});

[[nodiscard]] MechanicsSolveResult solve_cell_mechanics_cpu(
    const WorldState& state, const ContactGraph& contacts,
    const ExternalContactGraph& external_contacts,
    const MechanicsParameters& parameters = MechanicsParameters{});

[[nodiscard]] MechanicsSolveResult solve_cell_mechanics_cpu(
    const WorldState& state, const ContactGraph& contacts,
    const MechanicsParameters& parameters = MechanicsParameters{});

}  // namespace microsimulator
