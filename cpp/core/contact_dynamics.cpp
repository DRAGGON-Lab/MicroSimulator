#include "microsimulator/contact_dynamics.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "core/contact_stepper.hpp"
#include "core/shared_constants.hpp"
#include "microsimulator/mechanics.hpp"

namespace microsimulator {
namespace {

// The device kernels select the growth load law by these codes.
static_assert(static_cast<std::uint32_t>(GrowthLoadLaw::none) == shared::growth_load_law_none);
static_assert(static_cast<std::uint32_t>(GrowthLoadLaw::exponential) ==
              shared::growth_load_law_exponential);
static_assert(static_cast<std::uint32_t>(GrowthLoadLaw::linear) == shared::growth_load_law_linear);

// Cylinder lengths below this have no axis to elongate along.
constexpr float minimum_lever_length = 1.0e-6F;
// The solve converges to this fraction of the overlap tolerance, which leaves
// the rest for the error of moving the cells along a linearized gap.
constexpr float solve_tolerance_fraction = 0.25F;
// Fraction of the load scale of the growth response that an accepted overlap
// may add to a contact force.
constexpr float load_resolution_fraction = 0.1F;
// A rotation this large in one substep is outside the linearization the
// substep's rows rest on.
constexpr float maximum_substep_rotation = 0.5F;

[[nodiscard]] bool finite(const Vec3& value) {
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

struct Geometry {
  std::vector<Vec3> positions;
  std::vector<Vec3> directions;
  std::vector<float> lengths;

  [[nodiscard]] std::size_t size() const noexcept {
    return lengths.size();
  }
};

[[nodiscard]] Geometry capture(const WorldState& state) {
  const auto view = state.geometry_state();
  Geometry geometry;
  geometry.positions.reserve(view.size());
  geometry.directions.reserve(view.size());
  geometry.lengths.assign(view.lengths.begin(), view.lengths.end());
  for (std::size_t slot = 0; slot < view.size(); ++slot) {
    geometry.positions.push_back(
        {view.position_x[slot], view.position_y[slot], view.position_z[slot]});
    geometry.directions.push_back(
        {view.direction_x[slot], view.direction_y[slot], view.direction_z[slot]});
  }
  return geometry;
}

void restore(WorldState& state, const Geometry& geometry) {
  for (std::size_t slot = 0; slot < geometry.size(); ++slot) {
    state.set_cell_geometry(static_cast<Slot>(slot), geometry.positions[slot],
                            geometry.directions[slot], geometry.lengths[slot]);
  }
}

// A contact reduced to what a row needs. `partner` is the second cell's slot
// for a pair and the constraint for a wall.
struct Touch {
  Slot first_slot{invalid_slot};
  Slot second_slot{invalid_slot};
  std::uint64_t partner{0};
  Vec3 point{};
  Vec3 normal{};
  float separation{0.0F};
};

using TouchKey = std::tuple<Slot, bool, std::uint64_t>;

[[nodiscard]] TouchKey key_of(const Touch& touch) {
  return {touch.first_slot, touch.second_slot == invalid_slot, touch.partner};
}

[[nodiscard]] std::vector<Touch> find_touches(const WorldState& state,
                                              const ConstraintSet& constraints,
                                              ComputeBackend& backend,
                                              ContactParameters contact_parameters,
                                              ConstraintContactParameters constraint_parameters,
                                              float margin) {
  contact_parameters.activation_margin = margin;
  constraint_parameters.activation_margin = margin;
  std::vector<Touch> touches;
  const auto pairs = backend.find_cell_contacts(state, contact_parameters);
  touches.reserve(pairs.size());
  for (const auto& contact : pairs.contacts()) {
    touches.push_back({
        .first_slot = contact.first_slot,
        .second_slot = contact.second_slot,
        .partner = contact.second_slot,
        .point = contact.point_on_first,
        .normal = contact.normal,
        .separation = contact.signed_separation,
    });
  }
  if (!constraints.empty()) {
    const auto walls = find_external_contact_faces(state, constraints, constraint_parameters);
    for (const auto& contact : walls.contacts()) {
      touches.push_back({
          .first_slot = contact.cell_slot,
          .partner = contact.constraint_id,
          .point = contact.point_on_cell,
          .normal = contact.normal,
          .separation = contact.signed_separation,
      });
    }
  }
  return touches;
}

// The fraction of a cell's elongation that closes a contact whose centerline
// point is `centerline_point` and whose normal points out of the cell. The
// cylinder stretches uniformly about its center and the caps ride on its ends.
[[nodiscard]] float length_lever(const Geometry& geometry, Slot slot, Vec3 centerline_point,
                                 Vec3 outward_normal) {
  const auto length = geometry.lengths[slot];
  if (length <= minimum_lever_length) {
    return 0.0F;
  }
  const auto& axis = geometry.directions[slot];
  const auto coordinate = dot(axis, centerline_point - geometry.positions[slot]) / length;
  return std::max(0.0F, std::clamp(coordinate, -0.5F, 0.5F) * dot(axis, outward_normal));
}

// The row of a contact at the given geometry. A contact deeper than the
// recovery limit asks for only that much separation, and is stiffened in
// proportion so that a pair unable to separate still meets the full contact
// force of its overlap.
[[nodiscard]] ContactRow make_row(const Geometry& geometry, std::span<const float> radii,
                                  const Touch& touch, float compliance, float recovery) {
  ContactRow row;
  row.first_slot = touch.first_slot;
  row.second_slot = touch.second_slot;
  row.offset = std::max(touch.separation, -recovery);
  row.compliance =
      touch.separation < -recovery ? compliance * recovery / -touch.separation : compliance;
  const auto& normal = touch.normal;
  const auto first_arm = cross(touch.point - geometry.positions[touch.first_slot], normal);
  row.first_jacobian = {-normal.x, -normal.y, -normal.z, -first_arm.x, -first_arm.y, -first_arm.z};
  const auto first_centerline = touch.point - normal * radii[touch.first_slot];
  row.first_length_lever = length_lever(geometry, touch.first_slot, first_centerline, normal);
  if (touch.second_slot != invalid_slot) {
    const auto second_arm = cross(touch.point - geometry.positions[touch.second_slot], normal);
    row.second_jacobian = {normal.x, normal.y, normal.z, second_arm.x, second_arm.y, second_arm.z};
    const auto second_centerline =
        touch.point + normal * (touch.separation + radii[touch.second_slot]);
    row.second_length_lever =
        length_lever(geometry, touch.second_slot, second_centerline, normal * -1.0F);
  }
  return row;
}

// Mobilities come from the geometry the state holds. Elongation rates come
// from `rate_lengths`, the lengths at the start of the whole step, so that the
// elongation of an unloaded cell does not depend on how the step is divided.
[[nodiscard]] std::vector<ContactBody> make_bodies(const WorldState& state,
                                                   std::span<const float> rate_lengths,
                                                   float drag) {
  const auto geometry = state.geometry_state();
  const auto attributes = state.cell_attributes();
  std::vector<ContactBody> bodies(geometry.size());
  for (std::size_t slot = 0; slot < bodies.size(); ++slot) {
    auto& body = bodies[slot];
    body.axis = {geometry.direction_x[slot], geometry.direction_y[slot],
                 geometry.direction_z[slot]};
    body.length_rate = attributes.growth_rates[slot] * rate_lengths[slot];
    if (attributes.fixed[slot] != 0) {
      continue;
    }
    const auto radius = geometry.radii[slot];
    const auto total_length = geometry.lengths[slot] + 2.0F * radius;
    const auto translational = drag * total_length;
    const auto axial = 0.5F * translational * radius * radius;
    const auto transverse =
        translational * ((total_length * total_length) + 3.0F * radius * radius) / 12.0F;
    body.translational_mobility = 1.0F / translational;
    body.axial_mobility = 1.0F / axial;
    body.transverse_mobility = 1.0F / transverse;
  }
  return bodies;
}

[[nodiscard]] float predicted_rate(const ContactRow& row, const ContactResponse& response) {
  const auto product = [](const std::array<float, 6>& left, const std::array<float, 6>& right) {
    auto result = 0.0F;
    for (std::size_t index = 0; index < left.size(); ++index) {
      result += left[index] * right[index];
    }
    return result;
  };
  auto rate = product(row.first_jacobian, response.velocities[row.first_slot]) -
              row.first_length_lever * response.length_rates[row.first_slot];
  if (row.second_slot != invalid_slot) {
    rate += product(row.second_jacobian, response.velocities[row.second_slot]) -
            row.second_length_lever * response.length_rates[row.second_slot];
  }
  return rate;
}

// The geometry after moving every cell for `dt` under a response, or nothing
// when the move is not a valid geometry.
[[nodiscard]] std::optional<Geometry> moved(const Geometry& base, const ContactResponse& response,
                                            float dt) {
  auto result = base;
  for (std::size_t slot = 0; slot < base.size(); ++slot) {
    const auto& velocity = response.velocities[slot];
    const Vec3 rotation{velocity[3] * dt, velocity[4] * dt, velocity[5] * dt};
    if (!finite(rotation) || norm(rotation) > maximum_substep_rotation) {
      return std::nullopt;
    }
    result.positions[slot] =
        base.positions[slot] + Vec3{velocity[0], velocity[1], velocity[2]} * dt;
    result.directions[slot] =
        rotate_axis_angle(base.directions[slot], rotation, maximum_substep_rotation);
    result.lengths[slot] = base.lengths[slot] + response.length_rates[slot] * dt;
    if (!finite(result.positions[slot]) || !finite(result.directions[slot]) ||
        !std::isfinite(result.lengths[slot]) || result.lengths[slot] < 0.0F) {
      return std::nullopt;
    }
  }
  return result;
}

// The overlap accepted at the end of a substep. An overlap of e that a jammed
// pair cannot recover stays as a contact force of e * stiffness. Where growth
// responds to load, the tolerance is therefore held to the overlap that a
// small fraction of the load scale of that response presses into a contact.
[[nodiscard]] float effective_overlap_tolerance(const ContactDynamicsParameters& parameters) {
  auto load_scale = 0.0F;
  if (parameters.growth.law == GrowthLoadLaw::linear) {
    load_scale = parameters.growth.stall_load - parameters.growth.free_load;
  } else if (parameters.growth.law == GrowthLoadLaw::exponential &&
             parameters.growth.sensitivity > 0.0F) {
    load_scale = 1.0F / parameters.growth.sensitivity;
  }
  if (load_scale <= 0.0F) {
    return parameters.overlap_tolerance;
  }
  return std::min(parameters.overlap_tolerance,
                  load_resolution_fraction * load_scale / parameters.contact_stiffness);
}

struct Substep {
  Geometry geometry;
  std::vector<float> loads;
  std::uint32_t refinements{0};
  std::size_t rows{0};
  float max_overlap{0.0F};
};

// One substep from the geometry the state holds. On failure the state holds
// that geometry again.
[[nodiscard]] std::optional<Substep> advance_substep(
    WorldState& state, const Geometry& base, std::span<const float> rate_lengths,
    const ConstraintSet& constraints, ComputeBackend& backend, float dt,
    const ContactDynamicsParameters& parameters, const ContactParameters& contact_parameters,
    const ConstraintContactParameters& constraint_parameters, std::uint64_t& iterations) {
  const auto radii = state.geometry_state().radii;
  const auto bodies = make_bodies(state, rate_lengths, parameters.drag);
  const auto candidates = find_touches(state, constraints, backend, contact_parameters,
                                       constraint_parameters, parameters.candidate_margin);
  std::vector<ContactRow> rows;
  std::vector<TouchKey> keys;
  rows.reserve(candidates.size());
  keys.reserve(candidates.size());
  auto starting_overlap = 0.0F;
  const auto compliance = 1.0F / parameters.contact_stiffness;
  for (const auto& touch : candidates) {
    starting_overlap = std::max(starting_overlap, -touch.separation);
    rows.push_back(make_row(base, radii, touch, compliance, parameters.max_overlap_recovery));
    keys.push_back(key_of(touch));
  }
  // Overlap deeper than one substep recovers is accepted for now.
  const auto tolerance = effective_overlap_tolerance(parameters);
  const auto slack = tolerance + std::max(0.0F, starting_overlap - parameters.max_overlap_recovery);
  const ContactSolveParameters solve_parameters{
      .tolerance = solve_tolerance_fraction * tolerance,
      .max_iterations = parameters.max_iterations,
  };

  std::vector<float> multipliers(rows.size(), 0.0F);
  for (std::uint32_t refinement = 0; refinement <= parameters.max_refinements; ++refinement) {
    const ContactSystem system{
        .rows = rows,
        .bodies = bodies,
        .dt = dt,
        .growth = parameters.growth,
    };
    auto solve = backend.solve_contact_forces(system, multipliers, solve_parameters);
    iterations += solve.report.iterations;
    if (solve.report.status != ContactSolveStatus::converged) {
      break;
    }
    multipliers = std::move(solve.multipliers);
    const auto response = evaluate_contact_response(system, multipliers);
    const auto trial = moved(base, response, dt);
    if (!trial.has_value()) {
      break;
    }
    restore(state, *trial);

    const auto found =
        find_touches(state, constraints, backend, contact_parameters, constraint_parameters, 0.0F);
    std::map<TouchKey, float> forces;
    for (std::size_t index = 0; index < rows.size(); ++index) {
      forces[keys[index]] += multipliers[index];
    }
    std::vector<Touch> overlapping;
    auto max_overlap = 0.0F;
    for (const auto& touch : found) {
      max_overlap = std::max(max_overlap, -touch.separation);
      const auto force = forces.find(key_of(touch));
      const auto allowed = slack + (force == forces.end() ? 0.0F : force->second * compliance);
      if (touch.separation < -allowed) {
        overlapping.push_back(touch);
      }
    }
    if (overlapping.empty()) {
      return Substep{
          .geometry = *trial,
          .loads = response.loads,
          .refinements = refinement,
          .rows = rows.size(),
          .max_overlap = max_overlap,
      };
    }

    // Each new row is linearized where it was found. Its offset is set so
    // that the multipliers that produced the move predict the gap that the
    // move actually left.
    for (const auto& touch : overlapping) {
      auto row = make_row(*trial, radii, touch, compliance, parameters.max_overlap_recovery);
      row.offset -= dt * predicted_rate(row, response);
      rows.push_back(row);
      keys.push_back(key_of(touch));
      multipliers.push_back(0.0F);
    }
  }
  restore(state, base);
  return std::nullopt;
}

}  // namespace

void validate_growth_load_response(const GrowthLoadResponse& response) {
  if (response.law > GrowthLoadLaw::linear) {
    throw std::invalid_argument("growth load law is invalid");
  }
  if (!std::isfinite(response.sensitivity) || response.sensitivity < 0.0F) {
    throw std::invalid_argument("growth load sensitivity must be finite and non-negative");
  }
  if (!std::isfinite(response.free_load) || !std::isfinite(response.stall_load) ||
      response.free_load < 0.0F || response.stall_load <= response.free_load) {
    throw std::invalid_argument(
        "growth stall load must be finite and exceed a finite, non-negative free load");
  }
}

float growth_load_factor(const GrowthLoadResponse& response, float load) {
  switch (response.law) {
    case GrowthLoadLaw::none:
      return 1.0F;
    case GrowthLoadLaw::exponential:
      return std::exp(-response.sensitivity * std::max(0.0F, load));
    case GrowthLoadLaw::linear:
      return std::clamp(
          1.0F - (load - response.free_load) / (response.stall_load - response.free_load), 0.0F,
          1.0F);
  }
  throw std::invalid_argument("growth load law is invalid");
}

void validate_contact_dynamics_parameters(const ContactDynamicsParameters& parameters) {
  const auto positive = [](float value) { return std::isfinite(value) && value > 0.0F; };
  if (!positive(parameters.drag)) {
    throw std::invalid_argument("contact dynamics drag must be finite and positive");
  }
  if (!positive(parameters.contact_stiffness)) {
    throw std::invalid_argument("contact stiffness must be finite and positive");
  }
  if (!positive(parameters.overlap_tolerance)) {
    throw std::invalid_argument("contact overlap tolerance must be finite and positive");
  }
  if (!std::isfinite(parameters.candidate_margin) || parameters.candidate_margin < 0.0F) {
    throw std::invalid_argument("contact candidate margin must be finite and non-negative");
  }
  if (!positive(parameters.max_overlap_recovery)) {
    throw std::invalid_argument("contact overlap recovery must be finite and positive");
  }
  if (!positive(parameters.max_growth_fraction)) {
    throw std::invalid_argument("contact growth fraction must be finite and positive");
  }
  if (parameters.max_iterations == 0) {
    throw std::invalid_argument("contact dynamics iteration limit must be positive");
  }
  validate_growth_load_response(parameters.growth);
}

void validate_contact_system(const ContactSystem& system) {
  if (!std::isfinite(system.dt) || system.dt <= 0.0F) {
    throw std::invalid_argument("contact system time step must be finite and positive");
  }
  validate_growth_load_response(system.growth);
  for (const auto& body : system.bodies) {
    if (!finite(body.axis) || !std::isfinite(body.translational_mobility) ||
        !std::isfinite(body.transverse_mobility) || !std::isfinite(body.axial_mobility) ||
        !std::isfinite(body.length_rate) || body.translational_mobility < 0.0F ||
        body.transverse_mobility < 0.0F || body.axial_mobility < 0.0F) {
      throw std::invalid_argument("contact body is invalid");
    }
  }
  for (const auto& row : system.rows) {
    if (row.first_slot >= system.bodies.size() ||
        (row.second_slot != invalid_slot && row.second_slot >= system.bodies.size())) {
      throw std::invalid_argument("contact row slot is out of range");
    }
    auto valid = std::isfinite(row.offset) && std::isfinite(row.compliance) &&
                 row.compliance >= 0.0F && std::isfinite(row.first_length_lever) &&
                 std::isfinite(row.second_length_lever) && row.first_length_lever >= 0.0F &&
                 row.second_length_lever >= 0.0F;
    for (std::size_t index = 0; index < row.first_jacobian.size(); ++index) {
      valid = valid && std::isfinite(row.first_jacobian[index]) &&
              std::isfinite(row.second_jacobian[index]);
    }
    if (!valid) {
      throw std::invalid_argument("contact row is invalid");
    }
  }
}

namespace detail {

ContactDynamicsResult advance_contact_dynamics(
    WorldState& state, const ConstraintSet& constraints, ComputeBackend& backend, float dt,
    const ContactDynamicsParameters& parameters, const ContactParameters& contact_parameters,
    const ConstraintContactParameters& constraint_parameters) {
  if (!std::isfinite(dt) || dt < 0.0F) {
    throw std::invalid_argument("time step must be finite and non-negative");
  }
  validate_contact_dynamics_parameters(parameters);
  validate_contact_parameters(contact_parameters);
  validate_constraint_contact_parameters(constraint_parameters);
  state.validate();

  ContactDynamicsResult result;
  result.axial_loads.assign(state.size(), 0.0F);
  result.length_increments.assign(state.size(), 0.0F);
  if (state.empty() || dt == 0.0F) {
    return result;
  }

  const auto initial = capture(state);
  auto base = initial;
  try {
    // The first substep is as long as the growth bound allows.
    const auto view = state.geometry_state();
    const auto rates = state.cell_attributes().growth_rates;
    auto growth = 0.0F;
    for (std::size_t slot = 0; slot < view.size(); ++slot) {
      growth = std::max(growth,
                        std::abs(rates[slot]) * view.lengths[slot] * 0.5F * dt / view.radii[slot]);
    }
    const auto pieces = std::max(1.0F, std::ceil(growth / parameters.max_growth_fraction));
    if (!std::isfinite(pieces)) {
      throw std::runtime_error("contact dynamics growth per step is not finite");
    }
    const auto longest = dt / pieces;
    const auto shortest = std::ldexp(longest, -static_cast<int>(parameters.max_step_halvings));

    auto remaining = static_cast<double>(dt);
    auto length = longest;
    while (remaining > 0.0) {
      // A final piece within rounding of the remainder takes all of it.
      if (static_cast<double>(length) >= remaining * (1.0 - 1.0e-6)) {
        length = static_cast<float>(remaining);
      }
      auto substep =
          advance_substep(state, base, initial.lengths, constraints, backend, length, parameters,
                          contact_parameters, constraint_parameters, result.report.iterations);
      if (!substep.has_value()) {
        length *= 0.5F;
        ++result.report.halvings;
        if (length < shortest) {
          throw std::runtime_error(
              "contact dynamics did not resolve the contacts of a substep of " +
              std::to_string(length * 2.0F));
        }
        continue;
      }
      remaining -= static_cast<double>(length);
      base = std::move(substep->geometry);
      result.axial_loads = std::move(substep->loads);
      ++result.report.substeps;
      result.report.refinements += substep->refinements;
      result.report.max_rows = std::max(result.report.max_rows, substep->rows);
      result.report.max_overlap = substep->max_overlap;
      length = std::min(longest, length * 2.0F);
    }
  } catch (...) {
    restore(state, initial);
    throw;
  }
  for (std::size_t slot = 0; slot < initial.size(); ++slot) {
    result.length_increments[slot] = base.lengths[slot] - initial.lengths[slot];
  }
  return result;
}

}  // namespace detail
}  // namespace microsimulator
