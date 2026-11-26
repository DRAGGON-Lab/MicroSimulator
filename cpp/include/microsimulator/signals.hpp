#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "microsimulator/types.hpp"

namespace microsimulator {

enum class GridBoundaryKind : std::uint8_t {
  no_flux,
  periodic,
  fixed,
};

// The six faces of the lattice's bounding box, one per grid boundary.
enum class GridFace : std::uint8_t {
  x_lower,
  x_upper,
  y_lower,
  y_upper,
  z_lower,
  z_upper,
};

enum class SignalIntegrationKind : std::uint8_t {
  forward_euler,
  crank_nicolson,
  backward_euler,
};

// How a face's advective flux takes its concentration. `upwind` takes the
// upwind site's: first order, and never outside the range of its neighbours.
// `van_leer` adds a limited correction from the downwind site and the site
// two upwind: second order where the field is smooth, upwind at a front, and
// upwind wherever the site two upwind is a wall, a reservoir, or off the grid.
enum class SignalAdvectionKind : std::uint8_t {
  upwind,
  van_leer,
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

// A rectangle of boundary sites on one face of the lattice. The two entries
// index the face's in-plane axes in x, y, z order, so on an x face they are y
// and z, on a y face x and z, and on a z face x and y. Sites from `begin` up
// to but not including `end` belong to the rectangle.
struct GridFacePatch {
  std::array<std::uint32_t, 2> begin{};
  std::array<std::uint32_t, 2> end{};

  [[nodiscard]] bool contains(std::uint32_t first, std::uint32_t second) const noexcept {
    return first >= begin[0] && first < end[0] && second >= begin[1] && second < end[1];
  }
};

// Part of a fixed boundary that holds its own reservoir concentrations.
struct GridBoundaryPatch {
  GridFacePatch region;
  std::vector<float> values;
};

// One face of the lattice's bounding box. A fixed boundary holds one reservoir
// concentration per signal in `values`. Each of its `patches` replaces those
// concentrations over a rectangle of the face, so that several inlets on one
// face can carry different media; where patches overlap, the later one holds.
struct GridBoundary {
  GridBoundaryKind kind{GridBoundaryKind::no_flux};
  std::vector<float> values;
  std::vector<GridBoundaryPatch> patches;

  // The reservoir concentrations at one boundary site of a fixed boundary,
  // given the site's two in-plane indices.
  [[nodiscard]] std::span<const float> values_at(std::uint32_t first,
                                                 std::uint32_t second) const noexcept;
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
  SignalAdvectionKind advection_scheme{SignalAdvectionKind::upwind};
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
  [[nodiscard]] const GridBoundary& boundary(GridFace face) const;
  [[nodiscard]] GridBoundary& boundary(GridFace face);
  // The site counts of a face along its two in-plane axes, in x, y, z order.
  [[nodiscard]] std::array<std::uint32_t, 2> face_extent(GridFace face) const noexcept;
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

// Largest diagonal entry of the operator an implicit step of `implicit_dt`
// solves, I - implicit_dt * L, where L is the grid's transport and loss
// operator. It is at least one, and grows with diffusion over the squared
// spacing, outflow, and loss.
[[nodiscard]] float signal_implicit_operator_scale(const SignalGridSpec& spec, float implicit_dt);

// What transport derives from a grid's specification and reuses from step to
// step. Transport visits the fluid sites alone, so a step costs what the
// fluid costs however much of the lattice is obstacle.
struct SignalGridLayout {
  // The sites that are not obstacles, in ascending order.
  std::vector<std::uint32_t> fluid_sites;
  // The diagonal of the transport and reaction operator at each fluid site,
  // signal by signal in the order of fluid_sites.
  std::vector<float> diagonal;
  // The least of the diagonal and zero.
  float least_diagonal = 0.0F;
  // The greatest rate at which a velocity field empties any one site.
  double greatest_outflow = 0.0;

  // The scale of the implicit operator I - implicit_dt * A, as
  // signal_implicit_operator_scale gives it.
  [[nodiscard]] float implicit_operator_scale(float implicit_dt) const noexcept;
};

namespace detail {
struct SignalGridCache;
}

class SignalGrid {
 public:
  explicit SignalGrid(const SignalGridSpec& spec, std::vector<float> levels = {});
  explicit SignalGrid(const SignalGridCheckpoint& checkpoint);

  [[nodiscard]] const SignalGridSpec& spec() const noexcept;
  // A number that names the grid's specification: it differs between two
  // grids and changes whenever this grid's specification does, and never when
  // only its levels do. A backend that keeps device copies of the
  // specification's arrays rewrites them when the number it last saw differs.
  [[nodiscard]] std::uint64_t revision() const noexcept;
  // The layout of the present specification. It is derived when first asked
  // for after the specification changes and held until it changes again.
  [[nodiscard]] const SignalGridLayout& layout() const;
  // The layout together with the working arrays the CPU transport reuses.
  [[nodiscard]] detail::SignalGridCache& transport_cache() const;
  [[nodiscard]] std::span<const float> levels() const& noexcept;
  [[nodiscard]] std::span<const float> levels() && = delete;
  [[nodiscard]] std::vector<float> sample(Vec3 position) const;
  [[nodiscard]] Vec3 sample_velocity(Vec3 position,
                                     GridSampleBound bound = GridSampleBound::inside) const;
  [[nodiscard]] SignalGridCheckpoint checkpoint() const;
  void set_levels(std::span<const float> levels);
  void replace_levels(std::vector<float> levels);
  // Takes the levels a transport step produced: settles them with
  // settle_step_levels, then validates and holds them.
  void accept_step(std::vector<float> levels);
  void set_velocity_field(std::optional<SignalGridVelocityField> field);
  void set_reaction(std::optional<SignalGridAffineReaction> reaction);
  // Replaces the per-signal values of a fixed boundary, such as the
  // composition of the media entering at an inlet. The boundary kind does not
  // change, so only a boundary configured as fixed has values to replace.
  void set_boundary_values(GridFace face, std::span<const float> values);
  // Replaces the per-signal values of one patch of a fixed boundary, such as
  // the composition of the media entering at one of several inlets on a face.
  void set_boundary_patch_values(GridFace face, std::size_t patch, std::span<const float> values);
  void validate_step(float dt) const;
  void validate() const;

 private:
  // Holds what is derived from the grid. A copy of a grid derives its own.
  struct CacheSlot {
    std::shared_ptr<detail::SignalGridCache> held;
    CacheSlot() noexcept = default;
    CacheSlot(const CacheSlot& /*other*/) noexcept {
    }
    CacheSlot(CacheSlot&&) noexcept = default;
    CacheSlot& operator=(const CacheSlot& /*other*/) noexcept {
      held.reset();
      return *this;
    }
    CacheSlot& operator=(CacheSlot&&) noexcept = default;
    ~CacheSlot() = default;
  };

  SignalGridSpec spec_;
  std::vector<float> levels_;
  std::uint64_t revision_;
  mutable CacheSlot cache_;
};

// Checks the levels of a grid against its specification, which is taken as
// valid: one finite, non-negative level per signal and site, and zero at every
// obstacle site. A step checks its levels with this; nothing in a step changes
// the specification, which is validated where it is set.
void validate_signal_levels(const SignalGridSpec& spec, std::span<const float> levels);

// What to report when an implicit signal solve stops short of its tolerance.
// `solve` names the solve, such as "Implicit signal solve". Under the van
// Leer scheme the report adds the remedy: the limited flux makes the system
// nonlinear, and its iteration settles only while a step carries fluid across
// a few sites.
[[nodiscard]] std::string signal_solve_failure(std::string_view solve, const SignalGridSpec& spec,
                                               std::uint32_t iterations);

// Settles the levels a transport step produced before they are validated.
// Under the van Leer scheme a level may come out below zero by rounding
// alone, where upwind transport cannot; those within rounding of zero for the
// field's largest level become zero. Every other level is left as it is.
void settle_step_levels(const SignalGridSpec& spec, std::vector<float>& levels);

[[nodiscard]] SignalSolveReport advance_signal_grid_cpu(SignalGrid& grid, float dt);
[[nodiscard]] std::vector<float> signal_grid_forward_euler_candidate(const SignalGrid& grid,
                                                                     float dt);
[[nodiscard]] std::vector<float> signal_grid_transport_rates(const SignalGrid& grid,
                                                             std::span<const float> levels);
[[nodiscard]] SignalSolveResult signal_grid_crank_nicolson_candidate(
    const SignalGrid& grid, float dt, std::span<const float> source_rates = {});

}  // namespace microsimulator
