#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include "cm/contact_graph.hpp"

namespace cm {
namespace {

constexpr float inverse_sqrt_two = 0.7071067811865475F;

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

std::vector<PointPair> contact_points(const Capsule& first, const Capsule& second,
                                      const ContactParameters& parameters) {
  const auto axis_dot = std::clamp(dot(first.axis, second.axis), -1.0F, 1.0F);
  const auto sine = std::sqrt(std::max(0.0F, 1.0F - (axis_dot * axis_dot)));
  if (sine > parameters.parallel_sine_threshold || first.length <= parameters.degeneracy_epsilon ||
      second.length <= parameters.degeneracy_epsilon) {
    return {closest_points(first, second, parameters.degeneracy_epsilon)};
  }

  const auto first_half = first.length * 0.5F;
  const auto second_half = second.length * 0.5F;
  const auto center_coordinate = dot(second.center - first.center, first.axis);
  const auto projected_second_half = second_half * std::abs(axis_dot);
  const auto overlap_begin = std::max(-first_half, center_coordinate - projected_second_half);
  const auto overlap_end = std::min(first_half, center_coordinate + projected_second_half);
  if (overlap_end - overlap_begin <= parameters.degeneracy_epsilon) {
    return {closest_points(first, second, parameters.degeneracy_epsilon)};
  }

  std::vector<PointPair> result;
  result.reserve(2);
  for (const auto first_parameter : {overlap_begin, overlap_end}) {
    const auto point_on_first = first.center + (first.axis * first_parameter);
    const auto second_parameter =
        std::clamp(dot(point_on_first - second.center, second.axis), -second_half, second_half);
    result.push_back({
        .first = point_on_first,
        .second = second.center + (second.axis * second_parameter),
    });
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
  std::vector<CellContact> contacts;

  for (const auto& candidate : candidates) {
    auto first = capsule_at(geometry, candidate.first_slot);
    auto second = capsule_at(geometry, candidate.second_slot);

    const auto points = contact_points(first, second, parameters);
    const auto weight = points.size() == 2 ? inverse_sqrt_two : 1.0F;
    for (std::size_t ordinal = 0; ordinal < points.size(); ++ordinal) {
      const auto point_delta = points[ordinal].second - points[ordinal].first;
      const auto separation = norm(point_delta) - (first.radius + second.radius);
      if (separation >= parameters.activation_margin) {
        continue;
      }
      const auto normal =
          deterministic_normal(first, second, points[ordinal], parameters.degeneracy_epsilon);
      contacts.push_back({
          .first_id = first.id,
          .second_id = second.id,
          .first_slot = first.slot,
          .second_slot = second.slot,
          .ordinal = static_cast<std::uint8_t>(ordinal),
          .point_on_first = points[ordinal].first + (normal * first.radius),
          .normal = normal,
          .signed_separation = separation,
          .weight = weight,
      });
    }
  }

  std::ranges::sort(contacts, {}, [](const CellContact& contact) {
    return std::tuple{contact.first_id, contact.second_id, contact.ordinal};
  });
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

}  // namespace cm
