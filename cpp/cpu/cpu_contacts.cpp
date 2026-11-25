#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include "core/parallel.hpp"
#include "core/shared_constants.hpp"
#include "microsimulator/contact_graph.hpp"

namespace microsimulator {
namespace {

using shared::inverse_sqrt_two;

struct Capsule {
  CellId id;
  Slot slot;
  Vec3 center;
  Vec3 axis;
  float length;
  float radius;
};

struct PointPair {
  Vec3 first;
  Vec3 second;
};

// The one or two contact points of a capsule pair.
struct PointPairs {
  std::array<PointPair, 2> pairs{};
  std::size_t count{0};
};

// Candidate pairs handed to one thread at a time.
constexpr std::size_t pair_grain = 512;

Capsule capsule_at(const CellGeometryView& state, std::size_t index) {
  return {
      .id = state.ids[index],
      .slot = static_cast<Slot>(index),
      .center = {state.position_x[index], state.position_y[index], state.position_z[index]},
      .axis = {state.direction_x[index], state.direction_y[index], state.direction_z[index]},
      .length = state.lengths[index],
      .radius = state.radii[index],
  };
}

PointPair closest_points(const Capsule& first, const Capsule& second, float epsilon) {
  const auto first_half = first.length * 0.5F;
  const auto second_half = second.length * 0.5F;
  const auto first_start = first.center - (first.axis * first_half);
  const auto first_end = first.center + (first.axis * first_half);
  const auto second_start = second.center - (second.axis * second_half);
  const auto second_end = second.center + (second.axis * second_half);
  const auto first_delta = first_end - first_start;
  const auto second_delta = second_end - second_start;
  const auto between_starts = first_start - second_start;
  const auto first_length_squared = dot(first_delta, first_delta);
  const auto second_length_squared = dot(second_delta, second_delta);
  const auto second_projection = dot(second_delta, between_starts);

  float first_parameter = 0.0F;
  float second_parameter = 0.0F;
  if (first_length_squared <= epsilon * epsilon && second_length_squared <= epsilon * epsilon) {
    return {.first = first_start, .second = second_start};
  }
  if (first_length_squared <= epsilon * epsilon) {
    second_parameter = std::clamp(second_projection / second_length_squared, 0.0F, 1.0F);
  } else {
    const auto first_projection = dot(first_delta, between_starts);
    if (second_length_squared <= epsilon * epsilon) {
      first_parameter = std::clamp(-first_projection / first_length_squared, 0.0F, 1.0F);
    } else {
      const auto cross_projection = dot(first_delta, second_delta);
      const auto denominator =
          (first_length_squared * second_length_squared) - (cross_projection * cross_projection);
      const auto parallel_tolerance =
          std::numeric_limits<float>::epsilon() * first_length_squared * second_length_squared;
      if (denominator > parallel_tolerance) {
        first_parameter = std::clamp(
            ((cross_projection * second_projection) - (first_projection * second_length_squared)) /
                denominator,
            0.0F, 1.0F);
      }
      second_parameter =
          (cross_projection * first_parameter + second_projection) / second_length_squared;
      if (second_parameter < 0.0F) {
        second_parameter = 0.0F;
        first_parameter = std::clamp(-first_projection / first_length_squared, 0.0F, 1.0F);
      } else if (second_parameter > 1.0F) {
        second_parameter = 1.0F;
        first_parameter =
            std::clamp((cross_projection - first_projection) / first_length_squared, 0.0F, 1.0F);
      }
    }
  }

  return {
      .first = first_start + (first_delta * first_parameter),
      .second = second_start + (second_delta * second_parameter),
  };
}

PointPairs contact_points(const Capsule& first, const Capsule& second,
                          const ContactParameters& parameters) {
  const auto axis_dot = std::clamp(dot(first.axis, second.axis), -1.0F, 1.0F);
  const auto sine = std::sqrt(std::max(0.0F, 1.0F - (axis_dot * axis_dot)));
  if (sine > parameters.parallel_sine_threshold || first.length <= parameters.degeneracy_epsilon ||
      second.length <= parameters.degeneracy_epsilon) {
    return {.pairs = {closest_points(first, second, parameters.degeneracy_epsilon)}, .count = 1};
  }

  const auto first_half = first.length * 0.5F;
  const auto second_half = second.length * 0.5F;
  const auto center_coordinate = dot(second.center - first.center, first.axis);
  const auto projected_second_half = second_half * std::abs(axis_dot);
  const auto overlap_begin = std::max(-first_half, center_coordinate - projected_second_half);
  const auto overlap_end = std::min(first_half, center_coordinate + projected_second_half);
  if (overlap_end - overlap_begin <= parameters.degeneracy_epsilon) {
    return {.pairs = {closest_points(first, second, parameters.degeneracy_epsilon)}, .count = 1};
  }

  PointPairs result;
  for (const auto first_parameter : {overlap_begin, overlap_end}) {
    const auto point_on_first = first.center + (first.axis * first_parameter);
    const auto second_parameter =
        std::clamp(dot(point_on_first - second.center, second.axis), -second_half, second_half);
    result.pairs[result.count++] = {
        .first = point_on_first,
        .second = second.center + (second.axis * second_parameter),
    };
  }
  return result;
}

Vec3 deterministic_normal(const Capsule& first, const Capsule& second, const PointPair& points,
                          float epsilon) {
  const auto point_delta = points.second - points.first;
  if (norm(point_delta) > epsilon) {
    return normalized(point_delta);
  }

  const auto axes_cross = cross(first.axis, second.axis);
  if (norm(axes_cross) > epsilon) {
    return normalized(axes_cross);
  }

  const auto center_delta = second.center - first.center;
  const auto transverse_center_delta = center_delta - (first.axis * dot(center_delta, first.axis));
  if (norm(transverse_center_delta) > epsilon) {
    return normalized(transverse_center_delta);
  }

  const std::array basis{Vec3{1.0F, 0.0F, 0.0F}, Vec3{0.0F, 1.0F, 0.0F}, Vec3{0.0F, 0.0F, 1.0F}};
  const auto least_aligned =
      std::ranges::min_element(basis, [&first](const Vec3& left, const Vec3& right) {
        return std::abs(dot(first.axis, left)) < std::abs(dot(first.axis, right));
      });
  return normalized(cross(first.axis, *least_aligned));
}

ContactGraph contacts_for_candidates(const WorldState& state, const ContactParameters& parameters,
                                     std::span<const ContactCandidate> candidates) {
  const auto geometry = state.geometry_state();
  std::vector<std::vector<CellContact>> pieces((candidates.size() + pair_grain - 1) / pair_grain);
  detail::parallel_for(candidates.size(), pair_grain, [&](std::size_t begin, std::size_t end) {
    auto& piece = pieces[begin / pair_grain];
    for (std::size_t index = begin; index < end; ++index) {
      const auto first = capsule_at(geometry, candidates[index].first_slot);
      const auto second = capsule_at(geometry, candidates[index].second_slot);

      const auto points = contact_points(first, second, parameters);
      const auto weight = points.count == 2 ? inverse_sqrt_two : 1.0F;
      for (std::size_t ordinal = 0; ordinal < points.count; ++ordinal) {
        const auto& pair = points.pairs[ordinal];
        const auto point_delta = pair.second - pair.first;
        const auto separation = norm(point_delta) - (first.radius + second.radius);
        if (separation >= parameters.activation_margin) {
          continue;
        }
        const auto normal =
            deterministic_normal(first, second, pair, parameters.degeneracy_epsilon);
        piece.push_back({
            .first_id = first.id,
            .second_id = second.id,
            .first_slot = first.slot,
            .second_slot = second.slot,
            .ordinal = static_cast<std::uint8_t>(ordinal),
            .point_on_first = pair.first + (normal * first.radius),
            .normal = normal,
            .signed_separation = separation,
            .weight = weight,
        });
      }
    }
  });
  auto contacts = detail::concatenate(pieces);
  // Candidates sorted by identifier yield contacts already in order.
  const auto key = [](const CellContact& contact) {
    return std::tuple{contact.first_id, contact.second_id, contact.ordinal};
  };
  if (!std::ranges::is_sorted(contacts, {}, key)) {
    std::ranges::sort(contacts, {}, key);
  }
  return ContactGraph(geometry.size(), std::move(contacts));
}

}  // namespace

ContactGraph find_cell_contacts_cpu(const WorldState& state, const ContactParameters& parameters) {
  const auto candidates = find_cell_contact_candidates(state, parameters);
  return contacts_for_candidates(state, parameters, candidates);
}

ContactGraph find_cell_contacts_cpu_exhaustive(const WorldState& state,
                                               const ContactParameters& parameters) {
  validate_contact_parameters(parameters);
  const auto geometry = state.geometry_state();
  std::vector<ContactCandidate> candidates;
  if (geometry.size() > 1 &&
      geometry.size() - 1 > std::numeric_limits<std::size_t>::max() / geometry.size()) {
    throw std::overflow_error("exhaustive contact candidate count overflow");
  }
  const auto pair_count =
      geometry.size() < 2 ? std::size_t{0} : geometry.size() * (geometry.size() - 1) / 2;
  candidates.reserve(pair_count);
  for (std::size_t first = 0; first < geometry.size(); ++first) {
    for (std::size_t second = first + 1; second < geometry.size(); ++second) {
      candidates.push_back(
          geometry.ids[first] < geometry.ids[second]
              ? ContactCandidate{static_cast<Slot>(first), static_cast<Slot>(second)}
              : ContactCandidate{static_cast<Slot>(second), static_cast<Slot>(first)});
    }
  }
  return contacts_for_candidates(state, parameters, candidates);
}

}  // namespace microsimulator
