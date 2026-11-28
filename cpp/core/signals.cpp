#include "microsimulator/signals.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#include "core/parallel.hpp"
#include "core/shared_constants.hpp"
#include "core/signal_jacobi.hpp"

namespace microsimulator {

namespace detail {

// A grid's layout and the working arrays of the CPU transport. Each array
// holds one value per level and stays zero at obstacle sites, which the
// transport never writes.
struct SignalGridCache {
  // The specification revision the layout describes; zero before the first.
  std::uint64_t revision = 0;
  SignalGridLayout layout;
  std::vector<float> rates;
  std::vector<float> right_hand_side;
  std::vector<float> next;
  std::vector<float> residual;
};

}  // namespace detail

namespace {

// Grid values handed to one thread at a time.
constexpr std::size_t level_grain = 16384;
// Fluid sites handed to one thread at a time.
constexpr std::size_t fluid_grain = 2048;

[[nodiscard]] bool all_finite(std::span<const float> values) {
  return detail::parallel_reduce(
      values.size(), level_grain, true,
      [&](std::size_t begin, std::size_t end) {
        return std::all_of(values.begin() + static_cast<std::ptrdiff_t>(begin),
                           values.begin() + static_cast<std::ptrdiff_t>(end),
                           [](float value) { return std::isfinite(value); });
      },
      [](bool left, bool right) { return left && right; });
}

// Calls body(signal, site, x, y, z) for every fluid site of every signal.
template <typename Body>
void for_each_fluid(const SignalGridSpec& spec, std::span<const std::uint32_t> fluid,
                    const Body& body) {
  const auto plane = static_cast<std::size_t>(spec.shape.y) * spec.shape.z;
  detail::parallel_for(
      spec.signal_count * fluid.size(), fluid_grain, [&](std::size_t begin, std::size_t end) {
        for (std::size_t index = begin; index < end; ++index) {
          const std::size_t site = fluid[index % fluid.size()];
          body(index / fluid.size(), site, static_cast<std::uint32_t>(site / plane),
               static_cast<std::uint32_t>((site % plane) / spec.shape.z),
               static_cast<std::uint32_t>(site % spec.shape.z));
        }
      });
}

std::size_t checked_multiply(std::size_t left, std::size_t right, const char* name) {
  if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
    throw std::overflow_error(std::string("signal grid ") + name + " exceeds address space");
  }
  return left * right;
}

bool finite(Vec3 value) {
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

void validate_periodic_pair(const GridBoundary& lower, const GridBoundary& upper,
                            const char* axis) {
  const auto lower_periodic = lower.kind == GridBoundaryKind::periodic;
  const auto upper_periodic = upper.kind == GridBoundaryKind::periodic;
  if (lower_periodic != upper_periodic) {
    throw std::invalid_argument(std::string("signal grid periodic ") + axis +
                                " boundaries must be paired");
  }
}

struct AxisWeights {
  std::array<std::uint32_t, 2> indices{};
  std::array<float, 2> weights{1.0F, 0.0F};
  std::size_t count{1};
};

AxisWeights interpolation_axis(float position, float origin, float spacing, std::uint32_t dimension,
                               const char* axis, GridSampleBound bound) {
  if (!std::isfinite(position)) {
    throw std::invalid_argument("signal sample position must be finite");
  }
  if (dimension == 1) {
    return {};
  }

  auto coordinate =
      (static_cast<double>(position) - static_cast<double>(origin)) / static_cast<double>(spacing);
  const auto upper_bound = static_cast<double>(dimension - 1);
  if (bound == GridSampleBound::clamped) {
    // Clamping happens in lattice coordinates, the same space the bound is
    // tested in, so a clamped position always lands inside the lattice.
    coordinate = std::clamp(coordinate, 0.0, upper_bound);
  } else if (coordinate < 0.0 || coordinate > upper_bound) {
    throw std::out_of_range(std::string("signal sample is outside the ") + axis + " grid bound");
  }
  const auto lower = static_cast<std::uint32_t>(std::floor(coordinate));
  if (lower == dimension - 1) {
    return {.indices = {lower, lower}, .weights = {1.0F, 0.0F}, .count = 1};
  }
  const auto fraction = static_cast<float>(coordinate - static_cast<double>(lower));
  return {
      .indices = {lower, lower + 1},
      .weights = {1.0F - fraction, fraction},
      .count = 2,
  };
}

std::size_t flat_site(const GridShape& shape, std::uint32_t x, std::uint32_t y, std::uint32_t z) {
  return (static_cast<std::size_t>(x) * shape.y * shape.z) +
         (static_cast<std::size_t>(y) * shape.z) + z;
}

std::size_t x_face_index(const GridShape& shape, std::uint32_t fx, std::uint32_t y,
                         std::uint32_t z) {
  return (static_cast<std::size_t>(fx) * shape.y * shape.z) +
         (static_cast<std::size_t>(y) * shape.z) + z;
}

std::size_t y_face_index(const GridShape& shape, std::uint32_t x, std::uint32_t fy,
                         std::uint32_t z) {
  return (static_cast<std::size_t>(x) * (shape.y + 1) * shape.z) +
         (static_cast<std::size_t>(fy) * shape.z) + z;
}

std::size_t z_face_index(const GridShape& shape, std::uint32_t x, std::uint32_t y,
                         std::uint32_t fz) {
  return (static_cast<std::size_t>(x) * shape.y * (shape.z + 1)) +
         (static_cast<std::size_t>(y) * (shape.z + 1)) + fz;
}

struct FaceVelocities {
  float lower[3];
  float upper[3];
};

FaceVelocities face_velocities(const SignalGridSpec& spec, std::size_t signal, std::uint32_t x,
                               std::uint32_t y, std::uint32_t z) {
  FaceVelocities result{};
  if (spec.velocity_field.has_value()) {
    const auto& field = *spec.velocity_field;
    result.lower[0] = field.x_faces[x_face_index(spec.shape, x, y, z)];
    result.upper[0] = field.x_faces[x_face_index(spec.shape, x + 1, y, z)];
    result.lower[1] = field.y_faces[y_face_index(spec.shape, x, y, z)];
    result.upper[1] = field.y_faces[y_face_index(spec.shape, x, y + 1, z)];
    result.lower[2] = field.z_faces[z_face_index(spec.shape, x, y, z)];
    result.upper[2] = field.z_faces[z_face_index(spec.shape, x, y, z + 1)];
    return result;
  }
  const std::array<float, 3> velocity{spec.advection[signal].x, spec.advection[signal].y,
                                      spec.advection[signal].z};
  for (std::size_t axis = 0; axis < velocity.size(); ++axis) {
    result.lower[axis] = velocity[axis];
    result.upper[axis] = velocity[axis];
  }
  return result;
}

// The exterior stencil value beyond a boundary site whose two in-plane indices
// are `first` and `second`.
float boundary_value(const GridBoundary& boundary, std::size_t signal, float current,
                     float periodic, std::uint32_t first, std::uint32_t second) {
  switch (boundary.kind) {
    case GridBoundaryKind::no_flux:
      return current;
    case GridBoundaryKind::periodic:
      return periodic;
    case GridBoundaryKind::fixed:
      return boundary.values_at(first, second)[signal];
  }
  throw std::logic_error("unknown signal grid boundary kind");
}

bool solid_at(const SignalGridSpec& spec, std::uint32_t x, std::uint32_t y, std::uint32_t z) {
  return spec.has_obstacles() && spec.obstacles[flat_site(spec.shape, x, y, z)] != 0;
}

// What the van Leer scheme adds to the upwind site's concentration at a face:
// half the limited slope, 0.5 * psi(r) * (downwind - upwind) with
// r = (upwind - far) / (downwind - upwind) and psi(r) = (r + |r|) / (1 + |r|).
// It is zero at an extremum, where the two differences disagree in sign.
float van_leer_increment(float upwind, float downwind, float far) {
  const auto behind = upwind - far;
  const auto ahead = downwind - upwind;
  return behind * ahead > 0.0F ? behind * ahead / (behind + ahead) : 0.0F;
}

struct FaceClosure {
  bool lower[3];
  bool upper[3];
};

FaceClosure face_closure(const SignalGridSpec& spec, std::uint32_t x, std::uint32_t y,
                         std::uint32_t z) {
  const std::array<const GridBoundary*, 3> lower_boundaries{&spec.x_lower, &spec.y_lower,
                                                            &spec.z_lower};
  const std::array<const GridBoundary*, 3> upper_boundaries{&spec.x_upper, &spec.y_upper,
                                                            &spec.z_upper};
  const std::array<std::uint32_t, 3> dimensions{spec.shape.x, spec.shape.y, spec.shape.z};
  const std::array<std::uint32_t, 3> coordinates{x, y, z};
  FaceClosure closure{};
  for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
    auto lower_neighbor = coordinates;
    auto upper_neighbor = coordinates;
    if (coordinates[axis] == 0) {
      switch (lower_boundaries[axis]->kind) {
        case GridBoundaryKind::no_flux:
          closure.lower[axis] = true;
          break;
        case GridBoundaryKind::periodic:
          lower_neighbor[axis] = dimensions[axis] - 1;
          closure.lower[axis] =
              solid_at(spec, lower_neighbor[0], lower_neighbor[1], lower_neighbor[2]);
          break;
        case GridBoundaryKind::fixed:
          break;
      }
    } else {
      lower_neighbor[axis] = coordinates[axis] - 1;
      closure.lower[axis] = solid_at(spec, lower_neighbor[0], lower_neighbor[1], lower_neighbor[2]);
    }
    if (coordinates[axis] + 1 == dimensions[axis]) {
      switch (upper_boundaries[axis]->kind) {
        case GridBoundaryKind::no_flux:
          closure.upper[axis] = true;
          break;
        case GridBoundaryKind::periodic:
          upper_neighbor[axis] = 0;
          closure.upper[axis] =
              solid_at(spec, upper_neighbor[0], upper_neighbor[1], upper_neighbor[2]);
          break;
        case GridBoundaryKind::fixed:
          break;
      }
    } else {
      upper_neighbor[axis] = coordinates[axis] + 1;
      closure.upper[axis] = solid_at(spec, upper_neighbor[0], upper_neighbor[1], upper_neighbor[2]);
    }
  }
  return closure;
}

float signal_operator_diagonal(const SignalGridSpec& spec, std::size_t signal, std::uint32_t x,
                               std::uint32_t y, std::uint32_t z) {
  if (solid_at(spec, x, y, z)) {
    return 0.0F;
  }
  const std::array<std::uint32_t, 3> dimensions{spec.shape.x, spec.shape.y, spec.shape.z};
  const std::array<float, 3> spacing{spec.spacing.x, spec.spacing.y, spec.spacing.z};
  const auto faces = face_velocities(spec, signal, x, y, z);
  const auto closure = face_closure(spec, x, y, z);
  float diagonal = 0.0F;
  for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
    if (dimensions[axis] == 1) {
      continue;
    }
    const auto inverse_spacing = 1.0F / spacing[axis];
    const auto diffusion_scale = spec.diffusion[signal] * inverse_spacing * inverse_spacing;
    diagonal -= 2.0F * diffusion_scale;
    if (closure.lower[axis]) {
      diagonal += diffusion_scale;
    }
    if (closure.upper[axis]) {
      diagonal += diffusion_scale;
    }
    if (!closure.upper[axis] && faces.upper[axis] > 0.0F) {
      diagonal -= faces.upper[axis] * inverse_spacing;
    }
    if (!closure.lower[axis] && faces.lower[axis] < 0.0F) {
      diagonal += faces.lower[axis] * inverse_spacing;
    }
  }
  if (spec.reaction.has_value()) {
    const auto site = flat_site(spec.shape, x, y, z);
    diagonal -= spec.reaction->loss_rates[(signal * spec.site_count()) + site];
  }
  return diagonal;
}

// Transport rates of a grid whose invariants the caller has already validated.
void transport_rates_into(const SignalGrid& grid, std::span<const float> levels,
                          std::vector<float>& rates);

// Transport plus affine reaction rates of an already validated grid.
void operator_rates_into(const SignalGrid& grid, std::span<const float> levels,
                         std::vector<float>& rates) {
  transport_rates_into(grid, levels, rates);
  const auto& reaction = grid.spec().reaction;
  if (!reaction.has_value()) {
    return;
  }
  const auto sites = grid.spec().site_count();
  for_each_fluid(
      grid.spec(), grid.layout().fluid_sites,
      [&](std::size_t signal, std::size_t site, std::uint32_t, std::uint32_t, std::uint32_t) {
        const auto index = (signal * sites) + site;
        rates[index] +=
            reaction->source_rates[index] - (reaction->loss_rates[index] * levels[index]);
      });
}

double max_reaction_loss(const SignalGridSpec& spec, std::size_t signal) {
  if (!spec.reaction.has_value()) {
    return 0.0;
  }
  const auto sites = spec.site_count();
  const auto begin =
      spec.reaction->loss_rates.begin() + static_cast<std::ptrdiff_t>(signal * sites);
  return static_cast<double>(*std::max_element(begin, begin + static_cast<std::ptrdiff_t>(sites)));
}

// The root mean square, over every level of the grid, of values that are zero
// at obstacle sites.
float rms(const SignalGrid& grid, std::span<const float> values) {
  if (values.empty()) {
    return 0.0F;
  }
  const std::span<const std::uint32_t> fluid = grid.layout().fluid_sites;
  const auto sites = grid.spec().site_count();
  const auto sum = detail::parallel_reduce(
      grid.spec().signal_count * fluid.size(), fluid_grain, 0.0,
      [&](std::size_t begin, std::size_t end) {
        double piece = 0.0;
        for (std::size_t index = begin; index < end; ++index) {
          const auto value = static_cast<double>(
              values[((index / fluid.size()) * sites) + fluid[index % fluid.size()]]);
          piece += value * value;
        }
        return piece;
      },
      [](double left, double right) { return left + right; });
  return static_cast<float>(std::sqrt(sum / static_cast<double>(values.size())));
}

}  // namespace

SignalGridStencil signal_grid_stencil(const SignalGridSpec& spec, Vec3 position,
                                      GridSampleBound bound) {
  spec.validate_lattice();
  const auto x =
      interpolation_axis(position.x, spec.origin.x, spec.spacing.x, spec.shape.x, "x", bound);
  const auto y =
      interpolation_axis(position.y, spec.origin.y, spec.spacing.y, spec.shape.y, "y", bound);
  const auto z =
      interpolation_axis(position.z, spec.origin.z, spec.spacing.z, spec.shape.z, "z", bound);
  SignalGridStencil result;
  std::array<std::array<std::uint32_t, 3>, 8> coordinates{};
  for (std::size_t xi = 0; xi < x.count; ++xi) {
    for (std::size_t yi = 0; yi < y.count; ++yi) {
      for (std::size_t zi = 0; zi < z.count; ++zi) {
        coordinates[result.count] = {x.indices[xi], y.indices[yi], z.indices[zi]};
        result.sites[result.count] = static_cast<std::uint32_t>(
            flat_site(spec.shape, x.indices[xi], y.indices[yi], z.indices[zi]));
        result.weights[result.count] = x.weights[xi] * y.weights[yi] * z.weights[zi];
        ++result.count;
      }
    }
  }
  if (spec.has_obstacles()) {
    // Only the face-connected fluid component of the strongest interpolation
    // weight may exchange material with this cell. Corner contact is not flow.
    std::size_t seed = result.count;
    for (std::size_t i = 0; i < result.count; ++i) {
      if (spec.solid_site(result.sites[i])) result.weights[i] = 0.0F;
      if (result.weights[i] > 0.0F &&
          (seed == result.count || result.weights[i] > result.weights[seed]))
        seed = i;
    }
    if (seed == result.count) {
      result.entirely_solid = true;
      return result;
    }
    std::array<bool, 8> connected{};
    connected[seed] = true;
    // The component grows by at least one site per pass until it is whole.
    for (bool grew = true; grew;) {
      grew = false;
      for (std::size_t i = 0; i < result.count; ++i) {
        if (result.weights[i] <= 0.0F || connected[i]) continue;
        for (std::size_t j = 0; j < result.count; ++j) {
          unsigned distance = 0;
          for (std::size_t axis = 0; axis < 3; ++axis) {
            distance += static_cast<unsigned>(std::abs(static_cast<int>(coordinates[i][axis]) -
                                                       static_cast<int>(coordinates[j][axis])));
          }
          if (connected[j] && distance == 1) {
            connected[i] = true;
            grew = true;
            break;
          }
        }
      }
    }
    float total = 0.0F;
    for (std::size_t i = 0; i < result.count; ++i) {
      if (!connected[i]) result.weights[i] = 0.0F;
      total += result.weights[i];
    }
    for (std::size_t i = 0; i < result.count; ++i) result.weights[i] /= total;
  }
  return result;
}

float signal_implicit_operator_scale(const SignalGridSpec& spec, float implicit_dt) {
  auto scale = 1.0F;
  for (std::size_t signal = 0; signal < spec.signal_count; ++signal) {
    for (std::uint32_t x = 0; x < spec.shape.x; ++x) {
      for (std::uint32_t y = 0; y < spec.shape.y; ++y) {
        for (std::uint32_t z = 0; z < spec.shape.z; ++z) {
          scale = std::max(scale,
                           1.0F - (implicit_dt * signal_operator_diagonal(spec, signal, x, y, z)));
        }
      }
    }
  }
  return scale;
}

std::span<const float> GridBoundary::values_at(std::uint32_t first,
                                               std::uint32_t second) const noexcept {
  for (auto patch = patches.rbegin(); patch != patches.rend(); ++patch) {
    if (patch->region.contains(first, second)) {
      return patch->values;
    }
  }
  return values;
}

void GridBoundary::validate(std::size_t signal_count) const {
  const auto validate_values = [&](const std::vector<float>& candidate) {
    if (candidate.size() != signal_count) {
      throw std::invalid_argument("fixed signal grid boundary values must match signal count");
    }
    for (const auto value : candidate) {
      if (!std::isfinite(value) || value < 0.0F) {
        throw std::invalid_argument(
            "fixed signal grid boundary values must be finite and non-negative");
      }
    }
  };
  switch (kind) {
    case GridBoundaryKind::no_flux:
    case GridBoundaryKind::periodic:
      if (!values.empty() || !patches.empty()) {
        throw std::invalid_argument("non-fixed signal grid boundaries cannot have values");
      }
      return;
    case GridBoundaryKind::fixed:
      validate_values(values);
      for (const auto& patch : patches) {
        if (patch.region.end[0] <= patch.region.begin[0] ||
            patch.region.end[1] <= patch.region.begin[1]) {
          throw std::invalid_argument("signal grid boundary patches must hold at least one site");
        }
        validate_values(patch.values);
      }
      return;
  }
  throw std::invalid_argument("unknown signal grid boundary kind");
}

void SignalSolveParameters::validate() const {
  if (max_iterations == 0) {
    throw std::invalid_argument("signal solver iteration limit must be positive");
  }
  if (!std::isfinite(absolute_tolerance) || absolute_tolerance < 0.0F ||
      !std::isfinite(relative_tolerance) || relative_tolerance < 0.0F ||
      (absolute_tolerance == 0.0F && relative_tolerance == 0.0F)) {
    throw std::invalid_argument("signal solver tolerances must be finite and non-negative");
  }
}

void SignalGridAffineReaction::validate(std::size_t level_count) const {
  if (source_rates.size() != level_count || loss_rates.size() != level_count) {
    throw std::invalid_argument(
        "signal grid affine reaction arrays must match the grid level count");
  }
  for (const auto value : source_rates) {
    if (!std::isfinite(value) || value < 0.0F) {
      throw std::invalid_argument(
          "signal grid affine source rates must be finite and non-negative");
    }
  }
  for (const auto value : loss_rates) {
    if (!std::isfinite(value) || value < 0.0F) {
      throw std::invalid_argument("signal grid affine loss rates must be finite and non-negative");
    }
  }
}

const GridBoundary& SignalGridSpec::boundary(GridFace face) const {
  switch (face) {
    case GridFace::x_lower:
      return x_lower;
    case GridFace::x_upper:
      return x_upper;
    case GridFace::y_lower:
      return y_lower;
    case GridFace::y_upper:
      return y_upper;
    case GridFace::z_lower:
      return z_lower;
    case GridFace::z_upper:
      return z_upper;
  }
  throw std::invalid_argument("unknown signal grid face");
}

GridBoundary& SignalGridSpec::boundary(GridFace face) {
  return const_cast<GridBoundary&>(std::as_const(*this).boundary(face));
}

std::array<std::uint32_t, 2> SignalGridSpec::face_extent(GridFace face) const noexcept {
  switch (face) {
    case GridFace::x_lower:
    case GridFace::x_upper:
      return {shape.y, shape.z};
    case GridFace::y_lower:
    case GridFace::y_upper:
      return {shape.x, shape.z};
    case GridFace::z_lower:
    case GridFace::z_upper:
      break;
  }
  return {shape.x, shape.y};
}

std::size_t SignalGridSpec::site_count() const {
  const auto xy = checked_multiply(shape.x, shape.y, "site count");
  return checked_multiply(xy, shape.z, "site count");
}

std::size_t SignalGridSpec::level_count() const {
  return checked_multiply(signal_count, site_count(), "level count");
}

float SignalGridSpec::voxel_volume() const noexcept {
  return spacing.x * spacing.y * spacing.z;
}

std::size_t SignalGridSpec::x_face_count() const {
  return checked_multiply(static_cast<std::size_t>(shape.x) + 1,
                          checked_multiply(shape.y, shape.z, "face plane"), "x face count");
}

std::size_t SignalGridSpec::y_face_count() const {
  return checked_multiply(static_cast<std::size_t>(shape.y) + 1,
                          checked_multiply(shape.x, shape.z, "face plane"), "y face count");
}

std::size_t SignalGridSpec::z_face_count() const {
  return checked_multiply(static_cast<std::size_t>(shape.z) + 1,
                          checked_multiply(shape.x, shape.y, "face plane"), "z face count");
}

bool SignalGridSpec::has_obstacles() const noexcept {
  return !obstacles.empty();
}

bool SignalGridSpec::solid_site(std::size_t site) const noexcept {
  return !obstacles.empty() && obstacles[site] != 0;
}

void SignalGridSpec::validate_lattice() const {
  if (signal_count == 0) {
    throw std::invalid_argument("signal grid must contain at least one signal");
  }
  if (shape.x == 0 || shape.y == 0 || shape.z == 0) {
    throw std::invalid_argument("signal grid dimensions must be positive");
  }
  if (!finite(origin)) {
    throw std::invalid_argument("signal grid origin must be finite");
  }
  if (!finite(spacing) || spacing.x <= 0.0F || spacing.y <= 0.0F || spacing.z <= 0.0F) {
    throw std::invalid_argument("signal grid spacing must be finite and positive");
  }
  if (!obstacles.empty() && obstacles.size() != site_count()) {
    throw std::invalid_argument("signal grid obstacle mask must cover every site");
  }
  if (velocity_field.has_value()) {
    const auto& field = *velocity_field;
    if (field.x_faces.size() != x_face_count() || field.y_faces.size() != y_face_count() ||
        field.z_faces.size() != z_face_count()) {
      throw std::invalid_argument("signal grid velocity field must cover every lattice face");
    }
  }
}

void SignalGridSpec::validate() const {
  validate_lattice();
  if (diffusion.size() != signal_count || advection.size() != signal_count) {
    throw std::invalid_argument("signal grid transport arrays must match signal count");
  }
  for (const auto value : diffusion) {
    if (!std::isfinite(value) || value < 0.0F) {
      throw std::invalid_argument("signal diffusion must be finite and non-negative");
    }
  }
  for (const auto velocity : advection) {
    if (!finite(velocity)) {
      throw std::invalid_argument("signal advection must be finite");
    }
  }
  if (reaction.has_value()) {
    reaction->validate(level_count());
  }
  if (!obstacles.empty()) {
    for (const auto value : obstacles) {
      if (value > 1) {
        throw std::invalid_argument("signal grid obstacle mask values must be 0 or 1");
      }
    }
    if (reaction.has_value()) {
      const auto sites = site_count();
      for (std::size_t signal = 0; signal < signal_count; ++signal) {
        for (std::size_t site = 0; site < sites; ++site) {
          if (obstacles[site] != 0 && (reaction->source_rates[(signal * sites) + site] != 0.0F ||
                                       reaction->loss_rates[(signal * sites) + site] != 0.0F)) {
            throw std::invalid_argument(
                "signal grid affine reaction must be zero at obstacle sites");
          }
        }
      }
    }
  }
  if (velocity_field.has_value()) {
    const auto& field = *velocity_field;
    for (const auto* faces : {&field.x_faces, &field.y_faces, &field.z_faces}) {
      for (const auto value : *faces) {
        if (!std::isfinite(value)) {
          throw std::invalid_argument("signal grid velocity field values must be finite");
        }
      }
    }
    for (const auto velocity : advection) {
      if (velocity.x != 0.0F || velocity.y != 0.0F || velocity.z != 0.0F) {
        throw std::invalid_argument(
            "signal grid velocity field requires zero constant advection vectors");
      }
    }
    for (std::uint32_t x = 0; x < shape.x; ++x) {
      for (std::uint32_t y = 0; y < shape.y; ++y) {
        for (std::uint32_t z = 0; z < shape.z; ++z) {
          const auto closure = face_closure(*this, x, y, z);
          const auto faces = face_velocities(*this, 0, x, y, z);
          const auto solid_here = solid_site(flat_site(shape, x, y, z));
          const std::array<std::uint32_t, 3> dimensions{shape.x, shape.y, shape.z};
          for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
            if (dimensions[axis] == 1) {
              continue;
            }
            if ((closure.lower[axis] || solid_here) && faces.lower[axis] != 0.0F) {
              throw std::invalid_argument(
                  "signal grid velocity field must be zero on closed faces");
            }
            if ((closure.upper[axis] || solid_here) && faces.upper[axis] != 0.0F) {
              throw std::invalid_argument(
                  "signal grid velocity field must be zero on closed faces");
            }
          }
        }
      }
    }
    if (x_lower.kind == GridBoundaryKind::periodic && shape.x > 1) {
      for (std::uint32_t y = 0; y < shape.y; ++y) {
        for (std::uint32_t z = 0; z < shape.z; ++z) {
          if (field.x_faces[x_face_index(shape, 0, y, z)] !=
              field.x_faces[x_face_index(shape, shape.x, y, z)]) {
            throw std::invalid_argument(
                "signal grid velocity field periodic faces must hold equal values");
          }
        }
      }
    }
    if (y_lower.kind == GridBoundaryKind::periodic && shape.y > 1) {
      for (std::uint32_t x = 0; x < shape.x; ++x) {
        for (std::uint32_t z = 0; z < shape.z; ++z) {
          if (field.y_faces[y_face_index(shape, x, 0, z)] !=
              field.y_faces[y_face_index(shape, x, shape.y, z)]) {
            throw std::invalid_argument(
                "signal grid velocity field periodic faces must hold equal values");
          }
        }
      }
    }
    if (z_lower.kind == GridBoundaryKind::periodic && shape.z > 1) {
      for (std::uint32_t x = 0; x < shape.x; ++x) {
        for (std::uint32_t y = 0; y < shape.y; ++y) {
          if (field.z_faces[z_face_index(shape, x, y, 0)] !=
              field.z_faces[z_face_index(shape, x, y, shape.z)]) {
            throw std::invalid_argument(
                "signal grid velocity field periodic faces must hold equal values");
          }
        }
      }
    }
  }
  switch (integration) {
    case SignalIntegrationKind::forward_euler:
    case SignalIntegrationKind::crank_nicolson:
    case SignalIntegrationKind::backward_euler:
      break;
    default:
      throw std::invalid_argument("unknown signal integration kind");
  }
  switch (advection_scheme) {
    case SignalAdvectionKind::upwind:
    case SignalAdvectionKind::van_leer:
      break;
    default:
      throw std::invalid_argument("unknown signal advection scheme");
  }
  solver.validate();
  for (const auto face : {GridFace::x_lower, GridFace::x_upper, GridFace::y_lower,
                          GridFace::y_upper, GridFace::z_lower, GridFace::z_upper}) {
    const auto& held = boundary(face);
    held.validate(signal_count);
    const auto extent = face_extent(face);
    for (const auto& patch : held.patches) {
      if (patch.region.end[0] > extent[0] || patch.region.end[1] > extent[1]) {
        throw std::invalid_argument("signal grid boundary patches must lie on their face");
      }
    }
  }
  validate_periodic_pair(x_lower, x_upper, "x");
  validate_periodic_pair(y_lower, y_upper, "y");
  validate_periodic_pair(z_lower, z_upper, "z");
  const auto levels = level_count();
  if (levels > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error("signal grid level count exceeds the uint32 index space");
  }
}

void validate_signal_levels(const SignalGridSpec& spec, std::span<const float> levels) {
  if (levels.size() != spec.level_count()) {
    throw std::invalid_argument("signal grid level count does not match its specification");
  }
  // What is wrong with a range of levels, if anything: 1 for a level that is
  // not finite or is negative, 2 for one that is not zero at an obstacle.
  const auto sites = spec.site_count();
  const auto obstacles = spec.has_obstacles();
  const auto fault = detail::parallel_reduce(
      levels.size(), level_grain, 0,
      [&](std::size_t begin, std::size_t end) {
        int found = 0;
        for (std::size_t index = begin; index < end; ++index) {
          const auto level = levels[index];
          if (!std::isfinite(level) || level < 0.0F) {
            return 1;
          }
          if (obstacles && level != 0.0F && spec.solid_site(index % sites)) {
            found = 2;
          }
        }
        return found;
      },
      [](int left, int right) { return left == 1 || right == 1 ? 1 : std::max(left, right); });
  if (fault == 1) {
    throw std::invalid_argument("signal grid levels must be finite and non-negative");
  }
  if (fault == 2) {
    throw std::invalid_argument("signal grid levels must be zero at obstacle sites");
  }
}

void SignalGridCheckpoint::validate() const {
  spec.validate();
  validate_signal_levels(spec, levels);
}

namespace {

// A number no grid specification has had before.
std::uint64_t next_revision() noexcept {
  static std::atomic<std::uint64_t> counter{0};
  return ++counter;
}

}  // namespace

SignalGrid::SignalGrid(const SignalGridSpec& spec, std::vector<float> levels)
    : spec_(spec), levels_(std::move(levels)), revision_(next_revision()) {
  spec_.validate();
  if (levels_.empty()) {
    levels_.resize(spec_.level_count(), 0.0F);
  }
  validate_signal_levels(spec_, levels_);
}

SignalGrid::SignalGrid(const SignalGridCheckpoint& checkpoint)
    : spec_(checkpoint.spec), levels_(checkpoint.levels), revision_(next_revision()) {
  validate();
}

const SignalGridSpec& SignalGrid::spec() const noexcept {
  return spec_;
}

std::uint64_t SignalGrid::revision() const noexcept {
  return revision_;
}

float SignalGridLayout::implicit_operator_scale(float implicit_dt) const noexcept {
  return std::max(1.0F, 1.0F - (implicit_dt * least_diagonal));
}

const SignalGridLayout& SignalGrid::layout() const {
  return transport_cache().layout;
}

detail::SignalGridCache& SignalGrid::transport_cache() const {
  if (cache_.held == nullptr) {
    cache_.held = std::make_shared<detail::SignalGridCache>();
  }
  auto& cache = *cache_.held;
  if (cache.revision == revision_) {
    return cache;
  }
  auto& layout = cache.layout;
  const auto sites = spec_.site_count();
  layout.fluid_sites.clear();
  if (spec_.has_obstacles()) {
    for (std::size_t site = 0; site < sites; ++site) {
      if (spec_.obstacles[site] == 0) {
        layout.fluid_sites.push_back(static_cast<std::uint32_t>(site));
      }
    }
  } else {
    layout.fluid_sites.resize(sites);
    std::iota(layout.fluid_sites.begin(), layout.fluid_sites.end(), std::uint32_t{0});
  }
  const std::span<const std::uint32_t> fluid = layout.fluid_sites;
  layout.diagonal.assign(spec_.signal_count * fluid.size(), 0.0F);
  const auto plane = static_cast<std::size_t>(spec_.shape.y) * spec_.shape.z;
  const auto coordinates = [&](std::size_t site) {
    return std::array<std::uint32_t, 3>{static_cast<std::uint32_t>(site / plane),
                                        static_cast<std::uint32_t>((site % plane) / spec_.shape.z),
                                        static_cast<std::uint32_t>(site % spec_.shape.z)};
  };
  layout.least_diagonal = detail::parallel_reduce(
      layout.diagonal.size(), fluid_grain, 0.0F,
      [&](std::size_t begin, std::size_t end) {
        auto least = 0.0F;
        for (std::size_t index = begin; index < end; ++index) {
          const auto at = coordinates(fluid[index % fluid.size()]);
          layout.diagonal[index] =
              signal_operator_diagonal(spec_, index / fluid.size(), at[0], at[1], at[2]);
          least = std::min(least, layout.diagonal[index]);
        }
        return least;
      },
      [](float left, float right) { return std::min(left, right); });
  layout.greatest_outflow = 0.0;
  if (spec_.velocity_field.has_value()) {
    const std::array<std::uint32_t, 3> dimensions{spec_.shape.x, spec_.shape.y, spec_.shape.z};
    const std::array<float, 3> spacing{spec_.spacing.x, spec_.spacing.y, spec_.spacing.z};
    layout.greatest_outflow = detail::parallel_reduce(
        fluid.size(), fluid_grain, 0.0,
        [&](std::size_t begin, std::size_t end) {
          double greatest = 0.0;
          for (std::size_t index = begin; index < end; ++index) {
            const auto at = coordinates(fluid[index]);
            const auto faces = face_velocities(spec_, 0, at[0], at[1], at[2]);
            double outflow = 0.0;
            for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
              if (dimensions[axis] == 1) {
                continue;
              }
              const auto inverse_spacing = 1.0 / static_cast<double>(spacing[axis]);
              outflow += (std::max(static_cast<double>(faces.upper[axis]), 0.0) -
                          std::min(static_cast<double>(faces.lower[axis]), 0.0)) *
                         inverse_spacing;
            }
            greatest = std::max(greatest, outflow);
          }
          return greatest;
        },
        [](double left, double right) { return std::max(left, right); });
  }
  cache.revision = revision_;
  return cache;
}

std::span<const float> SignalGrid::levels() const& noexcept {
  return levels_;
}

std::vector<float> SignalGrid::sample(Vec3 position) const {
  const auto stencil = signal_grid_stencil(spec_, position);
  if (stencil.entirely_solid) {
    throw std::invalid_argument("signal sample position is inside a grid obstacle");
  }
  const auto sites = spec_.site_count();
  std::vector<float> result(spec_.signal_count, 0.0F);
  for (std::size_t entry = 0; entry < stencil.count; ++entry) {
    for (std::size_t signal = 0; signal < spec_.signal_count; ++signal) {
      result[signal] += stencil.weights[entry] * levels_[(signal * sites) + stencil.sites[entry]];
    }
  }
  return result;
}

Vec3 SignalGrid::sample_velocity(Vec3 position, GridSampleBound bound) const {
  if (!spec_.velocity_field.has_value()) {
    throw std::logic_error("signal grid does not declare a velocity field");
  }
  const auto stencil = signal_grid_stencil(spec_, position, bound);
  // The field is zero on every face of a solid site, so a stencil with no
  // fluid in it samples zero: a cell that mechanics has pressed into a wall
  // does not drift.
  const auto& field = *spec_.velocity_field;
  Vec3 result{};
  for (std::size_t entry = 0; entry < stencil.count; ++entry) {
    const auto site = stencil.sites[entry];
    const auto weight = stencil.weights[entry];
    if (weight == 0.0F) {
      continue;
    }
    const auto z = site % spec_.shape.z;
    const auto y = (site / spec_.shape.z) % spec_.shape.y;
    const auto x = site / (static_cast<std::size_t>(spec_.shape.y) * spec_.shape.z);
    const auto fx = static_cast<std::uint32_t>(x);
    const auto fy = static_cast<std::uint32_t>(y);
    const auto fz = static_cast<std::uint32_t>(z);
    result.x += weight * 0.5F *
                (field.x_faces[x_face_index(spec_.shape, fx, fy, fz)] +
                 field.x_faces[x_face_index(spec_.shape, fx + 1, fy, fz)]);
    result.y += weight * 0.5F *
                (field.y_faces[y_face_index(spec_.shape, fx, fy, fz)] +
                 field.y_faces[y_face_index(spec_.shape, fx, fy + 1, fz)]);
    result.z += weight * 0.5F *
                (field.z_faces[z_face_index(spec_.shape, fx, fy, fz)] +
                 field.z_faces[z_face_index(spec_.shape, fx, fy, fz + 1)]);
  }
  return result;
}

SignalGridCheckpoint SignalGrid::checkpoint() const {
  validate();
  return {.spec = spec_, .levels = levels_};
}

void SignalGrid::set_levels(std::span<const float> levels) {
  replace_levels(std::vector<float>(levels.begin(), levels.end()));
}

std::string signal_solve_failure(std::string_view solve, const SignalGridSpec& spec,
                                 std::uint32_t iterations) {
  auto message =
      std::string(solve) + " did not converge after " + std::to_string(iterations) + " iterations";
  if (spec.advection_scheme == SignalAdvectionKind::van_leer) {
    message +=
        "; under the van Leer scheme, shorten the step until flow crosses only a few "
        "sites in one";
  }
  return message;
}

void settle_step_levels(const SignalGridSpec& spec, std::vector<float>& levels) {
  if (spec.advection_scheme != SignalAdvectionKind::van_leer) {
    return;
  }
  const auto largest = detail::parallel_reduce(
      levels.size(), level_grain, 0.0F,
      [&](std::size_t begin, std::size_t end) {
        auto value = 0.0F;
        for (std::size_t index = begin; index < end; ++index) {
          value = std::max(value, levels[index]);
        }
        return value;
      },
      [](float left, float right) { return std::max(left, right); });
  const auto rounding = -16.0F * shared::float_epsilon * largest;
  detail::parallel_for(levels.size(), level_grain, [&](std::size_t begin, std::size_t end) {
    for (std::size_t index = begin; index < end; ++index) {
      if (levels[index] < 0.0F && levels[index] >= rounding) {
        levels[index] = 0.0F;
      }
    }
  });
}

void SignalGrid::accept_step(std::vector<float> levels) {
  settle_step_levels(spec_, levels);
  replace_levels(std::move(levels));
}

void SignalGrid::replace_levels(std::vector<float> levels) {
  validate_signal_levels(spec_, levels);
  levels_ = std::move(levels);
}

void SignalGrid::set_velocity_field(std::optional<SignalGridVelocityField> field) {
  // The field is validated in place and put back if it is refused, so that
  // neither outcome copies the specification.
  std::swap(spec_.velocity_field, field);
  try {
    spec_.validate();
  } catch (...) {
    std::swap(spec_.velocity_field, field);
    throw;
  }
  revision_ = next_revision();
}

void SignalGrid::set_boundary_values(GridFace face, std::span<const float> values) {
  auto& boundary = spec_.boundary(face);
  if (boundary.kind != GridBoundaryKind::fixed) {
    throw std::logic_error("only a fixed signal grid boundary has values to set");
  }
  GridBoundary candidate{.kind = GridBoundaryKind::fixed,
                         .values = std::vector<float>(values.begin(), values.end()),
                         .patches = boundary.patches};
  candidate.validate(spec_.signal_count);
  boundary = std::move(candidate);
  revision_ = next_revision();
}

void SignalGrid::set_boundary_patch_values(GridFace face, std::size_t patch,
                                           std::span<const float> values) {
  auto& boundary = spec_.boundary(face);
  if (boundary.kind != GridBoundaryKind::fixed) {
    throw std::logic_error("only a fixed signal grid boundary has values to set");
  }
  if (patch >= boundary.patches.size()) {
    throw std::out_of_range("signal grid boundary has no such patch");
  }
  auto candidate = boundary;
  candidate.patches[patch].values.assign(values.begin(), values.end());
  candidate.validate(spec_.signal_count);
  boundary = std::move(candidate);
  revision_ = next_revision();
}

void SignalGrid::set_reaction(std::optional<SignalGridAffineReaction> reaction) {
  std::swap(spec_.reaction, reaction);
  try {
    spec_.validate();
  } catch (...) {
    std::swap(spec_.reaction, reaction);
    throw;
  }
  revision_ = next_revision();
}

void SignalGrid::validate_step(float dt) const {
  if (!std::isfinite(dt) || dt < 0.0F) {
    throw std::invalid_argument("time step must be finite and non-negative");
  }
  if (spec_.integration != SignalIntegrationKind::forward_euler) {
    return;
  }
  const std::array<std::uint32_t, 3> dimensions{spec_.shape.x, spec_.shape.y, spec_.shape.z};
  const std::array<float, 3> spacing{spec_.spacing.x, spec_.spacing.y, spec_.spacing.z};
  const auto maximum_outflow = layout().greatest_outflow;
  for (std::size_t signal = 0; signal < spec_.signal_count; ++signal) {
    const std::array<float, 3> velocity{spec_.advection[signal].x, spec_.advection[signal].y,
                                        spec_.advection[signal].z};
    double inverse_square_sum = 0.0;
    double courant_sum = maximum_outflow;
    for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
      if (dimensions[axis] == 1) {
        continue;
      }
      const auto inverse_spacing = 1.0 / static_cast<double>(spacing[axis]);
      inverse_square_sum += inverse_spacing * inverse_spacing;
      courant_sum += std::abs(static_cast<double>(velocity[axis])) * inverse_spacing;
    }
    // A limited face flux carries up to twice the upwind flux, so the
    // explicit step that keeps levels within the range of their neighbours
    // is half as long in its advective part.
    if (spec_.advection_scheme == SignalAdvectionKind::van_leer) {
      courant_sum *= 2.0;
    }
    const auto factor = static_cast<double>(dt) *
                        ((2.0 * static_cast<double>(spec_.diffusion[signal]) * inverse_square_sum) +
                         courant_sum + max_reaction_loss(spec_, signal));
    if (!std::isfinite(factor) || factor > 1.0) {
      throw std::invalid_argument("signal grid time step violates the explicit stability bound");
    }
  }
}

void SignalGrid::validate() const {
  spec_.validate();
  validate_signal_levels(spec_, levels_);
}

std::vector<float> signal_grid_forward_euler_candidate(const SignalGrid& grid, float dt) {
  grid.validate_step(dt);
  if (dt == 0.0F) {
    return std::vector<float>(grid.levels().begin(), grid.levels().end());
  }
  const auto levels = grid.levels();
  auto& rates = grid.transport_cache().rates;
  operator_rates_into(grid, levels, rates);
  std::vector<float> updated(levels.begin(), levels.end());
  const auto upwind = grid.spec().advection_scheme == SignalAdvectionKind::upwind;
  const auto sites = grid.spec().site_count();
  for_each_fluid(
      grid.spec(), grid.layout().fluid_sites,
      [&](std::size_t signal, std::size_t site, std::uint32_t, std::uint32_t, std::uint32_t) {
        const auto index = (signal * sites) + site;
        const auto candidate = levels[index] + (dt * rates[index]);
        // Under the van Leer scheme a level may fall below zero by
        // rounding; SignalGrid::accept_step judges those.
        if (!std::isfinite(candidate) || (candidate < 0.0F && upwind)) {
          throw std::runtime_error(
              "signal grid update produced a non-finite or negative concentration");
        }
        updated[index] = candidate;
      });
  return updated;
}

std::vector<float> signal_grid_transport_rates(const SignalGrid& grid,
                                               std::span<const float> levels) {
  std::vector<float> rates;
  transport_rates_into(grid, levels, rates);
  return rates;
}

namespace {

void transport_rates_into(const SignalGrid& grid, std::span<const float> levels,
                          std::vector<float>& rates) {
  const auto& spec = grid.spec();
  if (levels.size() != spec.level_count()) {
    throw std::invalid_argument("signal transport level count does not match the grid");
  }
  const auto sites = spec.site_count();
  // Only fluid sites are written, so the rates at obstacle sites are zero as
  // long as the array began so.
  if (rates.size() != levels.size()) {
    rates.assign(levels.size(), 0.0F);
  }
  std::atomic<bool> finite_levels{true};

  const auto level = [&](std::size_t signal, std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    return levels[(signal * sites) + flat_site(spec.shape, x, y, z)];
  };

  const std::array<bool, 3> periodic_axis{
      spec.x_lower.kind == GridBoundaryKind::periodic,
      spec.y_lower.kind == GridBoundaryKind::periodic,
      spec.z_lower.kind == GridBoundaryKind::periodic,
  };
  for_each_fluid(
      spec, grid.layout().fluid_sites,
      [&](std::size_t signal, std::size_t fluid_site, std::uint32_t x, std::uint32_t y,
          std::uint32_t z) {
        const auto diffusion = spec.diffusion[signal];
        const std::array<float, 3> spacing{spec.spacing.x, spec.spacing.y, spec.spacing.z};
        const auto current = levels[(signal * sites) + fluid_site];
        if (!std::isfinite(current)) {
          finite_levels.store(false, std::memory_order_relaxed);
        }
        const auto faces = face_velocities(spec, signal, x, y, z);
        const auto closure = face_closure(spec, x, y, z);
        std::array<float, 3> lower{};
        std::array<float, 3> upper{};
        lower[0] = x == 0 ? boundary_value(spec.x_lower, signal, current,
                                           level(signal, spec.shape.x - 1, y, z), y, z)
                          : level(signal, x - 1, y, z);
        upper[0] = x + 1 == spec.shape.x
                       ? boundary_value(spec.x_upper, signal, current, level(signal, 0, y, z), y, z)
                       : level(signal, x + 1, y, z);
        lower[1] = y == 0 ? boundary_value(spec.y_lower, signal, current,
                                           level(signal, x, spec.shape.y - 1, z), x, z)
                          : level(signal, x, y - 1, z);
        upper[1] = y + 1 == spec.shape.y
                       ? boundary_value(spec.y_upper, signal, current, level(signal, x, 0, z), x, z)
                       : level(signal, x, y + 1, z);
        lower[2] = z == 0 ? boundary_value(spec.z_lower, signal, current,
                                           level(signal, x, y, spec.shape.z - 1), x, y)
                          : level(signal, x, y, z - 1);
        upper[2] = z + 1 == spec.shape.z
                       ? boundary_value(spec.z_upper, signal, current, level(signal, x, y, 0), x, y)
                       : level(signal, x, y, z + 1);

        float rate = 0.0F;
        const std::array<std::uint32_t, 3> dimensions{spec.shape.x, spec.shape.y, spec.shape.z};
        for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
          if (dimensions[axis] == 1) {
            continue;
          }
          if (closure.lower[axis]) {
            lower[axis] = current;
          }
          if (closure.upper[axis]) {
            upper[axis] = current;
          }
          const auto inverse_spacing = 1.0F / spacing[axis];
          rate += diffusion * (lower[axis] - (2.0F * current) + upper[axis]) * inverse_spacing *
                  inverse_spacing;
          auto lower_flux = faces.lower[axis] >= 0.0F ? faces.lower[axis] * lower[axis]
                                                      : faces.lower[axis] * current;
          auto upper_flux = faces.upper[axis] >= 0.0F ? faces.upper[axis] * current
                                                      : faces.upper[axis] * upper[axis];
          if (closure.lower[axis]) {
            lower_flux = 0.0F;
          }
          if (closure.upper[axis]) {
            upper_flux = 0.0F;
          }
          rate -= (upper_flux - lower_flux) * inverse_spacing;
          if (spec.advection_scheme != SignalAdvectionKind::van_leer) {
            continue;
          }
          // The limited correction to each face's flux. A face takes it only
          // when its upwind site, its downwind site and the site two upwind
          // are all fluid sites of the grid joined by open faces; otherwise
          // the face keeps the upwind flux. The same three sites decide a
          // face's correction from either side of it, so fluxes stay
          // conservative.
          const std::array<std::uint32_t, 3> coordinates{x, y, z};
          const auto periodic = periodic_axis[axis] && dimensions[axis] >= 3;
          const auto at = [&](std::uint32_t along) {
            auto site = coordinates;
            site[axis] = along;
            return site;
          };
          // The level `offset` sites along the axis, or nothing where that
          // site is off the grid or solid.
          const auto far = [&](int offset) -> std::optional<float> {
            const auto count = dimensions[axis];
            const auto along = static_cast<std::int64_t>(coordinates[axis]) + offset;
            if (!periodic && (along < 0 || along >= static_cast<std::int64_t>(count))) {
              return std::nullopt;
            }
            const auto site = at(static_cast<std::uint32_t>((along + count) % count));
            if (solid_at(spec, site[0], site[1], site[2])) {
              return std::nullopt;
            }
            return level(signal, site[0], site[1], site[2]);
          };
          const auto lower_site = !closure.lower[axis] && (coordinates[axis] > 0 || periodic);
          const auto upper_site =
              !closure.upper[axis] && (coordinates[axis] + 1 < dimensions[axis] || periodic);
          float lower_correction = 0.0F;
          float upper_correction = 0.0F;
          if (lower_site) {
            if (faces.lower[axis] >= 0.0F) {
              if (const auto beyond = far(-2)) {
                lower_correction =
                    faces.lower[axis] * van_leer_increment(lower[axis], current, *beyond);
              }
            } else if (upper_site) {
              lower_correction =
                  faces.lower[axis] * van_leer_increment(current, lower[axis], upper[axis]);
            }
          }
          if (upper_site) {
            if (faces.upper[axis] >= 0.0F) {
              if (lower_site) {
                upper_correction =
                    faces.upper[axis] * van_leer_increment(current, upper[axis], lower[axis]);
              }
            } else if (const auto beyond = far(2)) {
              upper_correction =
                  faces.upper[axis] * van_leer_increment(upper[axis], current, *beyond);
            }
          }
          rate -= (upper_correction - lower_correction) * inverse_spacing;
        }
        rates[(signal * sites) + fluid_site] = rate;
      });
  if (!finite_levels.load()) {
    throw std::invalid_argument("signal transport levels must be finite");
  }
}

}  // namespace

SignalSolveResult signal_grid_crank_nicolson_candidate(const SignalGrid& grid, float dt,
                                                       std::span<const float> source_rates) {
  if (!std::isfinite(dt) || dt < 0.0F) {
    throw std::invalid_argument("time step must be finite and non-negative");
  }
  const auto& spec = grid.spec();
  if (!source_rates.empty() && source_rates.size() != spec.level_count()) {
    throw std::invalid_argument("signal source rate count does not match the grid");
  }
  if (!all_finite(source_rates)) {
    throw std::invalid_argument("signal source rates must be finite");
  }
  const auto old = grid.levels();
  if (dt == 0.0F) {
    return {.levels = std::vector<float>(old.begin(), old.end()), .report = {}};
  }

  auto& cache = grid.transport_cache();
  const auto& layout = cache.layout;
  const std::span<const std::uint32_t> fluid = layout.fluid_sites;
  const auto sites = spec.site_count();
  auto& rates = cache.rates;
  auto& right_hand_side = cache.right_hand_side;
  auto& next = cache.next;
  auto& residual = cache.residual;
  for (auto* array : {&right_hand_side, &next, &residual}) {
    if (array->size() != old.size()) {
      array->assign(old.size(), 0.0F);
    }
  }
  // Calls body(level index, diagonal index) for every fluid level.
  const auto for_each_level = [&](const auto& body) {
    detail::parallel_for(
        spec.signal_count * fluid.size(), fluid_grain, [&](std::size_t begin, std::size_t end) {
          for (std::size_t index = begin; index < end; ++index) {
            body(((index / fluid.size()) * sites) + fluid[index % fluid.size()], index);
          }
        });
  };
  operator_rates_into(grid, old, rates);
  const bool backward = spec.integration == SignalIntegrationKind::backward_euler;
  const auto half_dt = (backward ? 1.0F : 0.5F) * dt;
  const auto explicit_dt = backward ? 0.0F : half_dt;
  for_each_level([&](std::size_t index, std::size_t) {
    const auto source = source_rates.empty() ? 0.0F : source_rates[index];
    right_hand_side[index] = old[index] + (explicit_dt * rates[index]) + (dt * source);
  });
  const std::span<const float> diagonal = layout.diagonal;
  const auto operator_scale = layout.implicit_operator_scale(half_dt);
  std::vector<float> current(old.begin(), old.end());
  const auto relaxation =
      spec.advection_scheme == SignalAdvectionKind::van_leer ? shared::van_leer_relaxation : 1.0F;
  // Leaves the operator rates of the current iterate in `rates`, which the
  // next sweep reuses.
  const auto residual_rms = [&] {
    operator_rates_into(grid, current, rates);
    for_each_level([&](std::size_t index, std::size_t) {
      residual[index] = right_hand_side[index] - current[index] + (half_dt * rates[index]);
    });
    return rms(grid, residual);
  };
  const detail::SignalJacobiOperations operations{
      .initial_residual_rms = residual_rms,
      .right_hand_side_rms = [&] { return rms(grid, right_hand_side); },
      .operator_scale = operator_scale,
      .begin_sweeps = [] {},
      .sweep =
          [&] {
            for_each_level([&](std::size_t index, std::size_t entry) {
              const auto remainder = rates[index] - (diagonal[entry] * current[index]);
              const auto update = (right_hand_side[index] + (half_dt * remainder)) /
                                  (1.0F - (half_dt * diagonal[entry]));
              next[index] = current[index] + (relaxation * (update - current[index]));
            });
            current.swap(next);
            return detail::SignalSweepResult{.residual_rms = residual_rms()};
          },
      .sweep_batch = {},
      .batch = 1,
  };
  const auto report = detail::solve_signal_jacobi(operations, spec.solver);
  return {.levels = std::move(current), .report = report};
}

SignalSolveReport advance_signal_grid_cpu(SignalGrid& grid, float dt) {
  if (grid.spec().integration == SignalIntegrationKind::forward_euler) {
    grid.accept_step(signal_grid_forward_euler_candidate(grid, dt));
    return {};
  }
  auto result = signal_grid_crank_nicolson_candidate(grid, dt);
  if (!result.report.converged) {
    throw std::runtime_error(
        signal_solve_failure("Implicit signal solve", grid.spec(), result.report.iterations));
  }
  grid.accept_step(std::move(result.levels));
  return result.report;
}

}  // namespace microsimulator
