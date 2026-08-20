#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "cm/types.hpp"

namespace cm {

enum class GridBoundaryKind : std::uint8_t {
  no_flux,
  periodic,
  fixed,
};

enum class SignalIntegrationKind : std::uint8_t {
  forward_euler,
  crank_nicolson,
};

struct SignalSolveParameters {
  std::uint32_t max_iterations{10'000};
  float absolute_tolerance{1.0e-7F};
  float relative_tolerance{1.0e-5F};

  void validate() const;
};

struct SignalSolveReport {
  bool converged{true};
  std::uint32_t iterations{0};
  float residual_rms{0.0F};
};

struct SignalSolveResult {
  std::vector<float> levels;
  SignalSolveReport report;
};

struct GridBoundary {
  GridBoundaryKind kind{GridBoundaryKind::no_flux};
  std::vector<float> values;

  void validate(std::size_t signal_count) const;
};

struct GridShape {
  std::uint32_t x{1};
  std::uint32_t y{1};
  std::uint32_t z{1};
};

struct SignalGridAffineReaction {
  std::vector<float> source_rates;
  std::vector<float> loss_rates;

  void validate(std::size_t level_count) const;
};

struct SignalGridVelocityField {
  std::vector<float> x_faces;
  std::vector<float> y_faces;
  std::vector<float> z_faces;
};

struct SignalGridSpec {
  std::uint32_t signal_count{0};
  GridShape shape;
  Vec3 origin;
  Vec3 spacing{1.0F, 1.0F, 1.0F};
  std::vector<float> diffusion;
  std::vector<Vec3> advection;
  std::optional<SignalGridAffineReaction> reaction;
  std::vector<std::uint8_t> obstacles;
  std::optional<SignalGridVelocityField> velocity_field;
  SignalIntegrationKind integration{SignalIntegrationKind::forward_euler};
  SignalSolveParameters solver;
  GridBoundary x_lower;
  GridBoundary x_upper;
  GridBoundary y_lower;
  GridBoundary y_upper;
  GridBoundary z_lower;
  GridBoundary z_upper;

  [[nodiscard]] std::size_t site_count() const;
  [[nodiscard]] std::size_t level_count() const;
  [[nodiscard]] float voxel_volume() const noexcept;
  [[nodiscard]] bool has_obstacles() const noexcept;
  [[nodiscard]] bool solid_site(std::size_t site) const noexcept;
  [[nodiscard]] std::size_t x_face_count() const;
  [[nodiscard]] std::size_t y_face_count() const;
  [[nodiscard]] std::size_t z_face_count() const;
  void validate_lattice() const;
  void validate() const;
};

struct SignalGridCheckpoint {
  SignalGridSpec spec;
  std::vector<float> levels;

  void validate() const;
};

struct SignalGridStencil {
  std::array<std::uint32_t, 8> sites{};
  std::array<float, 8> weights{};
  std::uint32_t count{0};
  // Every site of the stencil is solid, so it carries no fluid to interpolate
  // and every weight is zero. Concentration there is undefined and sampling it
  // is a model error; velocity there is zero by the field's own validation.
  bool entirely_solid{false};
};

// Whether a sample position outside the lattice of site centers is an error or
// is drawn in to the nearest in-lattice point.
enum class GridSampleBound : std::uint8_t { inside, clamped };

[[nodiscard]] SignalGridStencil signal_grid_stencil(
    const SignalGridSpec& spec, Vec3 position, GridSampleBound bound = GridSampleBound::inside);

class SignalGrid {
 public:
  explicit SignalGrid(const SignalGridSpec& spec, std::vector<float> levels = {});
  explicit SignalGrid(const SignalGridCheckpoint& checkpoint);

  [[nodiscard]] const SignalGridSpec& spec() const noexcept;
  [[nodiscard]] std::span<const float> levels() const& noexcept;
  [[nodiscard]] std::span<const float> levels() && = delete;
  [[nodiscard]] std::vector<float> sample(Vec3 position) const;
  [[nodiscard]] Vec3 sample_velocity(Vec3 position,
                                     GridSampleBound bound = GridSampleBound::inside) const;
  [[nodiscard]] SignalGridCheckpoint checkpoint() const;
  void set_levels(std::span<const float> levels);
  void replace_levels(std::vector<float> levels);
  void set_velocity_field(std::optional<SignalGridVelocityField> field);
  void set_reaction(std::optional<SignalGridAffineReaction> reaction);
  void validate_step(float dt) const;
  void validate() const;

 private:
  SignalGridSpec spec_;
  std::vector<float> levels_;
};

[[nodiscard]] SignalSolveReport advance_signal_grid_cpu(SignalGrid& grid, float dt);
[[nodiscard]] std::vector<float> signal_grid_forward_euler_candidate(const SignalGrid& grid,
                                                                     float dt);
[[nodiscard]] std::vector<float> signal_grid_transport_rates(const SignalGrid& grid,
                                                             std::span<const float> levels);
[[nodiscard]] SignalSolveResult signal_grid_crank_nicolson_candidate(
    const SignalGrid& grid, float dt, std::span<const float> source_rates = {});

}  // namespace cm
