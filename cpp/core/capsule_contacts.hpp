#pragma once

#include <limits>

#include "capsule_bodies.hpp"

namespace cm::detail {
inline constexpr std::uint32_t fluid_wall_body = std::numeric_limits<std::uint32_t>::max();

struct CapsuleContact {
  std::uint32_t first, second;
  BodyVector normal, first_offset, second_offset;
  double gap;
  double required_speed{0};
};

inline BodyVector body_cross(const BodyVector& a, const BodyVector& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

inline std::array<double, 6> contact_velocity_row(const CapsuleContact& contact, bool second,
                                                  double h) {
  const auto& offset = second ? contact.second_offset : contact.first_offset;
  const auto torque = body_cross(offset, contact.normal);
  const double sign = second ? 1 : -1;

  return {sign * contact.normal[0], sign * contact.normal[1], sign * contact.normal[2],
          sign * torque[0] / h,     sign * torque[1] / h,     sign * torque[2] / h};
}

inline double contact_growth_speed(const CapsuleBody& body, const BodyVector& offset,
                                   const BodyVector& normal) {
  const auto axis = body_rotate(body.orientation, {1, 0, 0});
  const double axial = body_dot(axis, offset);
  const double fraction =
      body.length == 0 ? std::copysign(0.5, axial) : std::clamp(axial / body.length, -0.5, 0.5);

  return body_dot(axis, normal) * fraction * body.length_rate;
}

inline double contact_separation_speed(const CapsuleContact& contact, const FluidFlowResult& flow) {
  double result = 0;

  for (bool second : {false, true}) {
    const auto index = second ? contact.second : contact.first;

    if (index == fluid_wall_body) {
      continue;
    }

    const auto coefficients = contact_velocity_row(contact, second, 1);
    const auto& motion = flow.bodies[index];

    for (std::size_t a = 0; a < 3; ++a) {
      result +=
          coefficients[a] * motion.velocity[a] + coefficients[3 + a] * motion.angular_velocity[a];
    }
  }

  return result;
}

inline std::pair<BodyVector, BodyVector> body_closest_box(const BodyVector& start,
                                                          const BodyVector& end,
                                                          const BodyVector& lo,
                                                          const BodyVector& hi) {
  BodyVector delta{};
  std::vector<double> cuts{0, 1};

  for (std::size_t a = 0; a < 3; ++a) {
    delta[a] = end[a] - start[a];

    if (delta[a] != 0) {
      for (double bound : {lo[a], hi[a]}) {
        const double t = (bound - start[a]) / delta[a];

        if (t > 0 && t < 1) {
          cuts.push_back(t);
        }
      }
    }
  }

  std::sort(cuts.begin(), cuts.end());
  double best = std::numeric_limits<double>::infinity();
  std::pair<BodyVector, BodyVector> points;
  auto evaluate = [&](double t) {
    BodyVector p{}, q{};
    double distance = 0;

    for (std::size_t a = 0; a < 3; ++a) {
      p[a] = start[a] + t * delta[a];
      q[a] = std::clamp(p[a], lo[a], hi[a]);
      distance += (p[a] - q[a]) * (p[a] - q[a]);
    }

    if (distance < best) {
      best = distance;
      points = {p, q};
    }
  };
  evaluate(0);
  evaluate(1);

  for (std::size_t i = 1; i < cuts.size(); ++i) {
    const double t = (cuts[i - 1] + cuts[i]) / 2;
    double aa = 0, bb = 0;

    for (std::size_t a = 0; a < 3; ++a) {
      const double x = start[a] + t * delta[a];

      if (x >= lo[a] && x <= hi[a]) {
        continue;
      }

      const double bound = x < lo[a] ? lo[a] : hi[a];
      aa += delta[a] * delta[a];
      bb += delta[a] * (start[a] - bound);
    }

    evaluate(cuts[i]);

    if (aa > 0) {
      evaluate(std::clamp(-bb / aa, cuts[i - 1], cuts[i]));
    }
  }

  return points;
}

inline void append_capsule_contact(std::vector<CapsuleContact>& contacts, const FluidGridSpec& grid,
                                   const std::vector<CapsuleBody>& bodies, double minimum_gap,
                                   double dt, double margin, std::uint32_t a, std::uint32_t b,
                                   const BodyVector& pa, const BodyVector& pb,
                                   double second_radius) {
  BodyVector normal{}, first_offset{}, second_offset{};
  double distance = 0;

  for (std::size_t d = 0; d < 3; ++d) {
    normal[d] = pb[d] - pa[d];
    distance += normal[d] * normal[d];
  }

  distance = std::sqrt(distance);
  const double gap = distance - bodies[a].radius - second_radius;

  if (gap > minimum_gap + margin) {
    return;
  }

  if (distance == 0 || gap < -1e-10 * grid.spacing) {
    throw std::invalid_argument("fluid contact geometry overlaps");
  }

  for (std::size_t d = 0; d < 3; ++d) {
    normal[d] /= distance;
    first_offset[d] = pa[d] - bodies[a].position[d] + bodies[a].radius * normal[d];

    if (b != fluid_wall_body) {
      second_offset[d] = pb[d] - bodies[b].position[d] - second_radius * normal[d];
    }
  }

  CapsuleContact contact{a, b, normal, first_offset, second_offset, gap};
  contact.required_speed =
      (minimum_gap - gap) / dt + contact_growth_speed(bodies[a], first_offset, normal);

  if (b != fluid_wall_body) {
    contact.required_speed -= contact_growth_speed(bodies[b], second_offset, normal);
  }

  for (const auto& old : contacts) {
    if (old.first == a && old.second == b) {
      double difference = 0;

      for (std::size_t d = 0; d < 3; ++d) {
        difference += std::pow(old.first_offset[d] - first_offset[d], 2);
      }

      if (difference < 1e-20 * grid.spacing * grid.spacing &&
          body_dot(old.normal, normal) > 1 - 1e-10) {
        return;
      }
    }
  }

  contacts.push_back(contact);
}

inline std::vector<CapsuleContact> capsule_contacts(const FluidGridSpec& grid,
                                                    const std::vector<CapsuleBody>& bodies,
                                                    double minimum_gap, double dt, double margin) {
  std::vector<CapsuleContact> contacts;
  auto append = [&](std::uint32_t a, std::uint32_t b, const BodyVector& pa, const BodyVector& pb,
                    double second_radius) {
    append_capsule_contact(contacts, grid, bodies, minimum_gap, dt, margin, a, b, pa, pb,
                           second_radius);
  };
  const BodyVector origin{grid.origin.x, grid.origin.y, grid.origin.z};
  const std::array<std::uint32_t, 3> dimensions{grid.shape.x, grid.shape.y, grid.shape.z};

  for (std::uint32_t a = 0; a < bodies.size(); ++a) {
    const auto [a0, a1] = body_endpoints(bodies[a]);

    for (std::uint32_t b = a + 1; b < bodies.size(); ++b) {
      const auto [b0, b1] = body_endpoints(bodies[b]);
      const auto [pa, pb] = body_closest_segments(a0, a1, b0, b1);
      append(a, b, pa, pb, bodies[b].radius);

      // Endpoints supply both members of a parallel-rod contact manifold.
      for (const auto& endpoint : {a0, a1}) {
        auto [p, q] = body_closest_segments(endpoint, endpoint, b0, b1);
        append(a, b, p, q, bodies[b].radius);
      }

      for (const auto& endpoint : {b0, b1}) {
        auto [p, q] = body_closest_segments(a0, a1, endpoint, endpoint);
        append(a, b, p, q, bodies[b].radius);
      }
    }

    for (std::size_t axis = 0; axis < 3; ++axis) {
      for (bool upper : {false, true}) {
        const double boundary =
            origin[axis] + (upper ? double(dimensions[axis]) - 0.5 : -0.5) * grid.spacing;

        for (const auto& endpoint : {a0, a1}) {
          auto wall = endpoint;
          wall[axis] = boundary;
          append(a, fluid_wall_body, endpoint, wall, 0);
        }
      }
    }

    for (std::size_t s = 0; s < grid.obstacles.size(); ++s) {
      if (grid.obstacles[s]) {
        auto index = s;
        const auto z = index % dimensions[2];
        index /= dimensions[2];
        const auto y = index % dimensions[1], x = index / dimensions[1];
        const std::array<std::size_t, 3> coordinate{x, y, z};
        BodyVector lo{}, hi{};

        for (std::size_t d = 0; d < 3; ++d) {
          lo[d] = origin[d] + (double(coordinate[d]) - 0.5) * grid.spacing;
          hi[d] = lo[d] + grid.spacing;
        }

        const auto [p, q] = body_closest_box(a0, a1, lo, hi);
        append(a, fluid_wall_body, p, q, 0);
      }
    }
  }

  return contacts;
}
}  // namespace cm::detail
