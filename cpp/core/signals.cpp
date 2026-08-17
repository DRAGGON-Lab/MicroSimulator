#include "cm/signals.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace cm {
namespace {

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
                               const char* axis) {
  if (!std::isfinite(position)) {
    throw std::invalid_argument("signal sample position must be finite");
  }
  if (dimension == 1) {
    return {};
  }

  const auto coordinate =
      (static_cast<double>(position) - static_cast<double>(origin)) / static_cast<double>(spacing);
  const auto upper_bound = static_cast<double>(dimension - 1);
  if (coordinate < 0.0 || coordinate > upper_bound) {
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

float boundary_value(const GridBoundary& boundary, std::size_t signal, float current,
                     float periodic) {
  switch (boundary.kind) {
    case GridBoundaryKind::no_flux:
      return current;
    case GridBoundaryKind::periodic:
      return periodic;
    case GridBoundaryKind::fixed:
      return boundary.values[signal];
  }
  throw std::logic_error("unknown signal grid boundary kind");
}

bool solid_at(const SignalGridSpec& spec, std::uint32_t x, std::uint32_t y, std::uint32_t z) {
  return spec.has_obstacles() && spec.obstacles[flat_site(spec.shape, x, y, z)] != 0;
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
  const std::array<float, 3> velocity{spec.advection[signal].x, spec.advection[signal].y,
                                      spec.advection[signal].z};
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
    if (velocity[axis] >= 0.0F) {
      if (!closure.upper[axis]) {
        diagonal -= velocity[axis] * inverse_spacing;
      }
    } else if (!closure.lower[axis]) {
      diagonal += velocity[axis] * inverse_spacing;
    }
  }
  if (spec.reaction.has_value()) {
    const auto site = flat_site(spec.shape, x, y, z);
    diagonal -= spec.reaction->loss_rates[(signal * spec.site_count()) + site];
  }
  return diagonal;
}

std::vector<float> signal_grid_operator_rates(const SignalGrid& grid,
                                              std::span<const float> levels) {
  auto rates = signal_grid_transport_rates(grid, levels);
  const auto& reaction = grid.spec().reaction;
  if (!reaction.has_value()) {
    return rates;
  }
  for (std::size_t index = 0; index < rates.size(); ++index) {
    rates[index] += reaction->source_rates[index] - (reaction->loss_rates[index] * levels[index]);
  }
  return rates;
}

double max_reaction_loss(const SignalGridSpec& spec, std::size_t signal) {
  if (!spec.reaction.has_value()) {
    return 0.0;
  }
  const auto sites = spec.site_count();
  const auto begin = spec.reaction->loss_rates.begin() +
                     static_cast<std::ptrdiff_t>(signal * sites);
  return static_cast<double>(
      *std::max_element(begin, begin + static_cast<std::ptrdiff_t>(sites)));
}

float rms(std::span<const float> values) {
  if (values.empty()) {
    return 0.0F;
  }
  double sum = 0.0;
  for (const auto value : values) {
    sum += static_cast<double>(value) * static_cast<double>(value);
  }
  return static_cast<float>(std::sqrt(sum / static_cast<double>(values.size())));
}

}  // namespace

SignalGridStencil signal_grid_stencil(const SignalGridSpec& spec, Vec3 position) {
  spec.validate();
  const auto x = interpolation_axis(position.x, spec.origin.x, spec.spacing.x, spec.shape.x, "x");
  const auto y = interpolation_axis(position.y, spec.origin.y, spec.spacing.y, spec.shape.y, "y");
  const auto z = interpolation_axis(position.z, spec.origin.z, spec.spacing.z, spec.shape.z, "z");
  SignalGridStencil result;
  for (std::size_t xi = 0; xi < x.count; ++xi) {
    for (std::size_t yi = 0; yi < y.count; ++yi) {
      for (std::size_t zi = 0; zi < z.count; ++zi) {
        result.sites[result.count] = static_cast<std::uint32_t>(
            flat_site(spec.shape, x.indices[xi], y.indices[yi], z.indices[zi]));
        result.weights[result.count] = x.weights[xi] * y.weights[yi] * z.weights[zi];
        ++result.count;
      }
    }
  }
  if (spec.has_obstacles()) {
    float fluid_weight = 0.0F;
    bool dropped = false;
    for (std::size_t entry = 0; entry < result.count; ++entry) {
      if (spec.solid_site(result.sites[entry]) && result.weights[entry] != 0.0F) {
        result.weights[entry] = 0.0F;
        dropped = true;
      } else {
        fluid_weight += result.weights[entry];
      }
    }
    if (dropped) {
      if (fluid_weight <= 0.0F) {
        throw std::invalid_argument("signal sample position is inside a grid obstacle");
      }
      for (std::size_t entry = 0; entry < result.count; ++entry) {
        result.weights[entry] /= fluid_weight;
      }
    }
  }
  return result;
}

void GridBoundary::validate(std::size_t signal_count) const {
  switch (kind) {
    case GridBoundaryKind::no_flux:
    case GridBoundaryKind::periodic:
      if (!values.empty()) {
        throw std::invalid_argument("non-fixed signal grid boundaries cannot have values");
      }
      return;
    case GridBoundaryKind::fixed:
      if (values.size() != signal_count) {
        throw std::invalid_argument("fixed signal grid boundary values must match signal count");
      }
      for (const auto value : values) {
        if (!std::isfinite(value) || value < 0.0F) {
          throw std::invalid_argument(
              "fixed signal grid boundary values must be finite and non-negative");
        }
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

std::size_t SignalGridSpec::site_count() const {
  const auto xy = checked_multiply(shape.x, shape.y, "site count");
  return checked_multiply(xy, shape.z, "site count");
}

std::size_t SignalGridSpec::level_count() const {
  return checked_multiply(signal_count, site_count(), "level count");
}

float SignalGridSpec::voxel_volume() const noexcept { return spacing.x * spacing.y * spacing.z; }

bool SignalGridSpec::has_obstacles() const noexcept { return !obstacles.empty(); }

bool SignalGridSpec::solid_site(std::size_t site) const noexcept {
  return !obstacles.empty() && obstacles[site] != 0;
}

void SignalGridSpec::validate() const {
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
    if (obstacles.size() != site_count()) {
      throw std::invalid_argument("signal grid obstacle mask must cover every site");
    }
    for (const auto value : obstacles) {
      if (value > 1) {
        throw std::invalid_argument("signal grid obstacle mask values must be 0 or 1");
      }
    }
    if (reaction.has_value()) {
      const auto sites = site_count();
      for (std::size_t signal = 0; signal < signal_count; ++signal) {
        for (std::size_t site = 0; site < sites; ++site) {
          if (obstacles[site] != 0 &&
              (reaction->source_rates[(signal * sites) + site] != 0.0F ||
               reaction->loss_rates[(signal * sites) + site] != 0.0F)) {
            throw std::invalid_argument(
                "signal grid affine reaction must be zero at obstacle sites");
          }
        }
      }
    }
  }
  switch (integration) {
    case SignalIntegrationKind::forward_euler:
    case SignalIntegrationKind::crank_nicolson:
      break;
    default:
      throw std::invalid_argument("unknown signal integration kind");
  }
  solver.validate();
  for (const auto* boundary : {&x_lower, &x_upper, &y_lower, &y_upper, &z_lower, &z_upper}) {
    boundary->validate(signal_count);
  }
  validate_periodic_pair(x_lower, x_upper, "x");
  validate_periodic_pair(y_lower, y_upper, "y");
  validate_periodic_pair(z_lower, z_upper, "z");
  const auto levels = level_count();
  if (levels > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error("signal grid level count exceeds the uint32 index space");
  }
}

void SignalGridCheckpoint::validate() const {
  spec.validate();
  if (levels.size() != spec.level_count()) {
    throw std::invalid_argument("signal grid level count does not match its specification");
  }
  for (const auto level : levels) {
    if (!std::isfinite(level) || level < 0.0F) {
      throw std::invalid_argument("signal grid levels must be finite and non-negative");
    }
  }
  if (spec.has_obstacles()) {
    const auto sites = spec.site_count();
    for (std::size_t index = 0; index < levels.size(); ++index) {
      if (spec.solid_site(index % sites) && levels[index] != 0.0F) {
        throw std::invalid_argument("signal grid levels must be zero at obstacle sites");
      }
    }
  }
}

SignalGrid::SignalGrid(const SignalGridSpec& spec, std::vector<float> levels)
    : spec_(spec), levels_(std::move(levels)) {
  spec_.validate();
  if (levels_.empty()) {
    levels_.resize(spec_.level_count(), 0.0F);
  }
  validate();
}

SignalGrid::SignalGrid(const SignalGridCheckpoint& checkpoint)
    : spec_(checkpoint.spec), levels_(checkpoint.levels) {
  validate();
}

const SignalGridSpec& SignalGrid::spec() const noexcept { return spec_; }

std::span<const float> SignalGrid::levels() const& noexcept { return levels_; }

std::vector<float> SignalGrid::sample(Vec3 position) const {
  const auto stencil = signal_grid_stencil(spec_, position);
  const auto sites = spec_.site_count();
  std::vector<float> result(spec_.signal_count, 0.0F);
  for (std::size_t entry = 0; entry < stencil.count; ++entry) {
    for (std::size_t signal = 0; signal < spec_.signal_count; ++signal) {
      result[signal] += stencil.weights[entry] * levels_[(signal * sites) + stencil.sites[entry]];
    }
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

void SignalGrid::replace_levels(std::vector<float> levels) {
  SignalGridCheckpoint candidate{.spec = spec_, .levels = levels};
  candidate.validate();
  levels_ = std::move(levels);
}

void SignalGrid::validate_step(float dt) const {
  if (!std::isfinite(dt) || dt < 0.0F) {
    throw std::invalid_argument("time step must be finite and non-negative");
  }
  if (spec_.integration == SignalIntegrationKind::crank_nicolson) {
    return;
  }
  const std::array<std::uint32_t, 3> dimensions{spec_.shape.x, spec_.shape.y, spec_.shape.z};
  const std::array<float, 3> spacing{spec_.spacing.x, spec_.spacing.y, spec_.spacing.z};
  for (std::size_t signal = 0; signal < spec_.signal_count; ++signal) {
    const std::array<float, 3> velocity{spec_.advection[signal].x, spec_.advection[signal].y,
                                        spec_.advection[signal].z};
    double inverse_square_sum = 0.0;
    double courant_sum = 0.0;
    for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
      if (dimensions[axis] == 1) {
        continue;
      }
      const auto inverse_spacing = 1.0 / static_cast<double>(spacing[axis]);
      inverse_square_sum += inverse_spacing * inverse_spacing;
      courant_sum += std::abs(static_cast<double>(velocity[axis])) * inverse_spacing;
    }
    const auto factor =
        static_cast<double>(dt) *
        ((2.0 * static_cast<double>(spec_.diffusion[signal]) * inverse_square_sum) + courant_sum +
         max_reaction_loss(spec_, signal));
    if (!std::isfinite(factor) || factor > 1.0) {
      throw std::invalid_argument("signal grid time step violates the explicit stability bound");
    }
  }
}

void SignalGrid::validate() const {
  SignalGridCheckpoint{.spec = spec_, .levels = levels_}.validate();
}

std::vector<float> signal_grid_forward_euler_candidate(const SignalGrid& grid, float dt) {
  grid.validate();
  grid.validate_step(dt);
  if (dt == 0.0F) {
    return std::vector<float>(grid.levels().begin(), grid.levels().end());
  }
  const auto levels = grid.levels();
  const auto rates = signal_grid_operator_rates(grid, levels);
  std::vector<float> updated(levels.begin(), levels.end());
  for (std::size_t index = 0; index < updated.size(); ++index) {
    const auto candidate = levels[index] + (dt * rates[index]);
    if (!std::isfinite(candidate) || candidate < 0.0F) {
      throw std::runtime_error(
          "signal grid update produced a non-finite or negative concentration");
    }
    updated[index] = candidate;
  }
  return updated;
}

std::vector<float> signal_grid_transport_rates(const SignalGrid& grid,
                                               std::span<const float> levels) {
  grid.validate();
  const auto& spec = grid.spec();
  if (levels.size() != spec.level_count()) {
    throw std::invalid_argument("signal transport level count does not match the grid");
  }
  if (!std::ranges::all_of(levels, [](float value) { return std::isfinite(value); })) {
    throw std::invalid_argument("signal transport levels must be finite");
  }
  const auto sites = spec.site_count();
  std::vector<float> rates(levels.size(), 0.0F);

  const auto level = [&](std::size_t signal, std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    return levels[(signal * sites) + flat_site(spec.shape, x, y, z)];
  };

  for (std::size_t signal = 0; signal < spec.signal_count; ++signal) {
    const auto diffusion = spec.diffusion[signal];
    const std::array<float, 3> velocity{spec.advection[signal].x, spec.advection[signal].y,
                                        spec.advection[signal].z};
    const std::array<float, 3> spacing{spec.spacing.x, spec.spacing.y, spec.spacing.z};
    for (std::uint32_t x = 0; x < spec.shape.x; ++x) {
      for (std::uint32_t y = 0; y < spec.shape.y; ++y) {
        for (std::uint32_t z = 0; z < spec.shape.z; ++z) {
          if (solid_at(spec, x, y, z)) {
            rates[(signal * sites) + flat_site(spec.shape, x, y, z)] = 0.0F;
            continue;
          }
          const auto current = level(signal, x, y, z);
          const auto closure = face_closure(spec, x, y, z);
          std::array<float, 3> lower{};
          std::array<float, 3> upper{};
          lower[0] = x == 0 ? boundary_value(spec.x_lower, signal, current,
                                             level(signal, spec.shape.x - 1, y, z))
                            : level(signal, x - 1, y, z);
          upper[0] = x + 1 == spec.shape.x
                         ? boundary_value(spec.x_upper, signal, current, level(signal, 0, y, z))
                         : level(signal, x + 1, y, z);
          lower[1] = y == 0 ? boundary_value(spec.y_lower, signal, current,
                                             level(signal, x, spec.shape.y - 1, z))
                            : level(signal, x, y - 1, z);
          upper[1] = y + 1 == spec.shape.y
                         ? boundary_value(spec.y_upper, signal, current, level(signal, x, 0, z))
                         : level(signal, x, y + 1, z);
          lower[2] = z == 0 ? boundary_value(spec.z_lower, signal, current,
                                             level(signal, x, y, spec.shape.z - 1))
                            : level(signal, x, y, z - 1);
          upper[2] = z + 1 == spec.shape.z
                         ? boundary_value(spec.z_upper, signal, current, level(signal, x, y, 0))
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
            auto lower_flux =
                velocity[axis] >= 0.0F ? velocity[axis] * lower[axis] : velocity[axis] * current;
            auto upper_flux =
                velocity[axis] >= 0.0F ? velocity[axis] * current : velocity[axis] * upper[axis];
            if (closure.lower[axis]) {
              lower_flux = 0.0F;
            }
            if (closure.upper[axis]) {
              upper_flux = 0.0F;
            }
            rate -= (upper_flux - lower_flux) * inverse_spacing;
          }
          rates[(signal * sites) + flat_site(spec.shape, x, y, z)] = rate;
        }
      }
    }
  }
  return rates;
}

SignalSolveResult signal_grid_crank_nicolson_candidate(const SignalGrid& grid, float dt,
                                                       std::span<const float> source_rates) {
  grid.validate();
  if (!std::isfinite(dt) || dt < 0.0F) {
    throw std::invalid_argument("time step must be finite and non-negative");
  }
  const auto& spec = grid.spec();
  if (!source_rates.empty() && source_rates.size() != spec.level_count()) {
    throw std::invalid_argument("signal source rate count does not match the grid");
  }
  if (!std::ranges::all_of(source_rates, [](float value) { return std::isfinite(value); })) {
    throw std::invalid_argument("signal source rates must be finite");
  }
  const auto old = grid.levels();
  if (dt == 0.0F) {
    return {.levels = std::vector<float>(old.begin(), old.end()), .report = {}};
  }

  const auto old_rates = signal_grid_operator_rates(grid, old);
  const auto half_dt = 0.5F * dt;
  std::vector<float> right_hand_side(old.size());
  for (std::size_t index = 0; index < old.size(); ++index) {
    const auto source = source_rates.empty() ? 0.0F : source_rates[index];
    right_hand_side[index] = old[index] + (half_dt * old_rates[index]) + (dt * source);
  }
  const auto threshold =
      spec.solver.absolute_tolerance + (spec.solver.relative_tolerance * rms(right_hand_side));
  std::vector<float> current(old.begin(), old.end());
  std::vector<float> residual(old.size());
  auto residual_rms = std::numeric_limits<float>::infinity();
  std::uint32_t iterations = 0;

  for (; iterations <= spec.solver.max_iterations; ++iterations) {
    const auto rates = signal_grid_operator_rates(grid, current);
    for (std::size_t index = 0; index < current.size(); ++index) {
      residual[index] = right_hand_side[index] - current[index] + (half_dt * rates[index]);
    }
    residual_rms = rms(residual);
    if (!std::isfinite(residual_rms)) {
      break;
    }
    if (residual_rms <= threshold) {
      return {
          .levels = std::move(current),
          .report = {.converged = true, .iterations = iterations, .residual_rms = residual_rms},
      };
    }
    if (iterations == spec.solver.max_iterations) {
      break;
    }

    const auto sites = spec.site_count();
    std::vector<float> next(current.size());
    for (std::size_t signal = 0; signal < spec.signal_count; ++signal) {
      for (std::uint32_t x = 0; x < spec.shape.x; ++x) {
        for (std::uint32_t y = 0; y < spec.shape.y; ++y) {
          for (std::uint32_t z = 0; z < spec.shape.z; ++z) {
            const auto index = (signal * sites) + flat_site(spec.shape, x, y, z);
            const auto diagonal = signal_operator_diagonal(spec, signal, x, y, z);
            const auto remainder = rates[index] - (diagonal * current[index]);
            next[index] =
                (right_hand_side[index] + (half_dt * remainder)) / (1.0F - (half_dt * diagonal));
          }
        }
      }
    }
    current = std::move(next);
  }
  return {
      .levels = std::move(current),
      .report = {.converged = false, .iterations = iterations, .residual_rms = residual_rms},
  };
}

SignalSolveReport advance_signal_grid_cpu(SignalGrid& grid, float dt) {
  if (grid.spec().integration == SignalIntegrationKind::forward_euler) {
    grid.replace_levels(signal_grid_forward_euler_candidate(grid, dt));
    return {};
  }
  auto result = signal_grid_crank_nicolson_candidate(grid, dt);
  if (!result.report.converged) {
    throw std::runtime_error("Crank-Nicolson signal solve did not converge after " +
                             std::to_string(result.report.iterations) + " iterations");
  }
  grid.replace_levels(std::move(result.levels));
  return result.report;
}

}  // namespace cm
