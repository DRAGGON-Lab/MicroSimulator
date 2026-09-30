#pragma once

#include <cstdint>
#include <limits>

#include "capsule_contacts.hpp"

namespace cm::detail {
struct ClipPlane {
  BodyVector normal;
  double offset;
  std::uint64_t tag;
};

struct ConvexPolygon {
  std::vector<BodyVector> vertices;
  std::uint64_t tag;
};

struct ConvexPolyhedron {
  std::vector<ConvexPolygon> faces;
};

struct PolyhedronMeasure {
  double volume{0};
  BodyVector centroid{};
};

inline BodyVector body_difference(const BodyVector& a, const BodyVector& b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

inline double polygon_epsilon(const std::vector<BodyVector>& polygon) {
  double scale = 0;

  for (const auto& p : polygon) {
    for (double x : p) {
      scale = std::max(scale, std::abs(x));
    }
  }

  return 256 * std::numeric_limits<double>::epsilon() * scale;
}

inline void clean_polygon(std::vector<BodyVector>& polygon) {
  const double epsilon = polygon_epsilon(polygon);
  std::vector<BodyVector> clean;

  for (const auto& p : polygon) {
    if (!clean.empty()) {
      const auto delta = body_difference(p, clean.back());

      if (body_dot(delta, delta) <= epsilon * epsilon) {
        continue;
      }
    }

    clean.push_back(p);
  }

  if (clean.size() > 1) {
    const auto delta = body_difference(clean.front(), clean.back());

    if (body_dot(delta, delta) <= epsilon * epsilon) {
      clean.pop_back();
    }
  }

  BodyVector oriented_area{};

  for (std::size_t i = 2; i < clean.size(); ++i) {
    const auto cross =
        body_cross(body_difference(clean[i - 1], clean[0]), body_difference(clean[i], clean[0]));

    for (unsigned d = 0; d < 3; ++d) {
      oriented_area[d] += cross[d];
    }
  }

  bool changed = true;

  while (changed && clean.size() > 2) {
    changed = false;

    for (std::size_t i = 0; i < clean.size(); ++i) {
      const auto a = body_difference(clean[i], clean[(i + clean.size() - 1) % clean.size()]);
      const auto b = body_difference(clean[(i + 1) % clean.size()], clean[i]);
      const auto cross = body_cross(a, b);

      // All input and clipped faces are convex. Intersections of nearly
      // parallel planes can produce two roundoff-separated copies of a corner
      // with a reversed turn. Retaining that spurious edge creates an inward
      // halfspace when matching faces, removing a macroscopic face area.
      if (body_dot(cross, oriented_area) < 0 ||
          body_dot(cross, cross) <= epsilon * epsilon * (body_dot(a, a) + body_dot(b, b))) {
        clean.erase(clean.begin() + static_cast<std::ptrdiff_t>(i));
        changed = true;
        break;
      }
    }
  }

  polygon = std::move(clean);
}

inline double plane_distance(const ClipPlane& plane, const BodyVector& p) {
  const double distance = body_dot(plane.normal, p) - plane.offset;
  double scale = std::abs(plane.offset);

  for (unsigned d = 0; d < 3; ++d) {
    scale += std::abs(plane.normal[d] * p[d]);
  }

  return std::abs(distance) <= 64 * std::numeric_limits<double>::epsilon() * scale ? 0 : distance;
}

inline BodyVector body_normalized(BodyVector a) {
  const double norm = std::sqrt(body_dot(a, a));

  if (!(norm > 0) || !std::isfinite(norm)) {
    throw std::invalid_argument("degenerate fluid geometry normal");
  }

  for (double& x : a) {
    x /= norm;
  }

  return a;
}

inline double polygon_area(const std::vector<BodyVector>& polygon) {
  if (polygon.size() < 3) {
    return 0;
  }

  BodyVector area{};

  for (std::size_t i = 2; i < polygon.size(); ++i) {
    const auto cross = body_cross(body_difference(polygon[i - 1], polygon[0]),
                                  body_difference(polygon[i], polygon[0]));

    for (std::size_t d = 0; d < 3; ++d) {
      area[d] += cross[d] / 2;
    }
  }

  return std::sqrt(body_dot(area, area));
}

inline PolyhedronMeasure measure_polyhedron(const ConvexPolyhedron& poly) {
  PolyhedronMeasure result;
  BodyVector reference{};
  std::size_t count = 0;

  for (const auto& face : poly.faces) {
    for (const auto& p : face.vertices) {
      for (std::size_t d = 0; d < 3; ++d) {
        reference[d] += p[d];
      }

      ++count;
    }
  }

  if (!count) {
    return result;
  }

  for (double& x : reference) {
    x /= double(count);
  }

  for (const auto& face : poly.faces) {
    for (std::size_t i = 2; i < face.vertices.size(); ++i) {
      const auto& a = face.vertices[0];
      const auto& b = face.vertices[i - 1];
      const auto& c = face.vertices[i];
      const double v =
          body_dot(body_difference(a, reference),
                   body_cross(body_difference(b, reference), body_difference(c, reference))) /
          6;
      result.volume += v;

      for (std::size_t d = 0; d < 3; ++d) {
        result.centroid[d] += v * (reference[d] + a[d] + b[d] + c[d]) / 4;
      }
    }
  }

  if (result.volume > 0) {
    for (double& x : result.centroid) {
      x /= result.volume;
    }
  }

  return result;
}

inline std::vector<BodyVector> clip_polygon(const std::vector<BodyVector>& polygon,
                                            const ClipPlane& plane,
                                            std::vector<BodyVector>* intersections = nullptr) {
  std::vector<BodyVector> result;

  if (polygon.empty()) {
    return result;
  }

  auto previous = polygon.back();
  double dp = plane_distance(plane, previous);

  for (const auto& current : polygon) {
    const double dc = plane_distance(plane, current);

    if ((dp <= 0) != (dc <= 0)) {
      const double t = dp / (dp - dc);
      BodyVector point{};

      for (std::size_t d = 0; d < 3; ++d) {
        point[d] = previous[d] + t * (current[d] - previous[d]);
      }

      result.push_back(point);

      if (intersections) {
        intersections->push_back(point);
      }
    }

    if (dc <= 0) {
      result.push_back(current);
    }

    previous = current;
    dp = dc;
  }

  clean_polygon(result);

  return result;
}

inline ConvexPolyhedron clip_polyhedron(const ConvexPolyhedron& poly, const ClipPlane& plane) {
  ConvexPolyhedron result;
  std::vector<BodyVector> intersections;

  for (const auto& face : poly.faces) {
    auto clipped = clip_polygon(face.vertices, plane, &intersections);
    const double epsilon = polygon_epsilon(clipped);

    if (polygon_area(clipped) > epsilon * epsilon) {
      result.faces.push_back({std::move(clipped), face.tag});
    }
  }

  std::vector<BodyVector> cap;
  const double epsilon = polygon_epsilon(intersections);

  for (const auto& point : intersections) {
    bool duplicate = false;

    for (const auto& previous : cap) {
      const auto delta = body_difference(point, previous);

      if (body_dot(delta, delta) <= epsilon * epsilon) {
        duplicate = true;
        break;
      }
    }

    if (!duplicate) {
      cap.push_back(point);
    }
  }

  if (cap.size() >= 3) {
    BodyVector center{};

    for (const auto& p : cap) {
      for (std::size_t d = 0; d < 3; ++d) {
        center[d] += p[d] / double(cap.size());
      }
    }

    const auto normal = body_normalized(plane.normal);
    const BodyVector axis = std::abs(normal[0]) < 0.9 ? BodyVector{1, 0, 0} : BodyVector{0, 1, 0};
    const auto u = body_normalized(body_cross(normal, axis));
    const auto v = body_cross(normal, u);
    std::sort(cap.begin(), cap.end(), [&](const auto& a, const auto& b) {
      const auto da = body_difference(a, center), db = body_difference(b, center);

      return std::atan2(body_dot(da, v), body_dot(da, u)) <
             std::atan2(body_dot(db, v), body_dot(db, u));
    });
    clean_polygon(cap);

    if (polygon_area(cap) > epsilon * epsilon) {
      result.faces.push_back({std::move(cap), plane.tag});
    }
  }

  return result;
}

inline ConvexPolyhedron box_polyhedron(const BodyVector& lo, const BodyVector& hi,
                                       std::array<std::uint64_t, 6> tags = {}) {
  std::array<BodyVector, 8> p;

  for (unsigned i = 0; i < 8; ++i) {
    p[i] = {(i & 4) ? hi[0] : lo[0], (i & 2) ? hi[1] : lo[1], (i & 1) ? hi[2] : lo[2]};
  }

  constexpr std::array<std::array<unsigned, 4>, 6> indices{
      {{0, 1, 3, 2}, {4, 6, 7, 5}, {0, 4, 5, 1}, {2, 3, 7, 6}, {0, 2, 6, 4}, {1, 5, 7, 3}}};
  ConvexPolyhedron poly;

  for (std::size_t f = 0; f < 6; ++f) {
    ConvexPolygon face;
    face.tag = tags[f];

    for (auto i : indices[f]) {
      face.vertices.push_back(p[i]);
    }

    poly.faces.push_back(std::move(face));
  }

  return poly;
}

struct CapsulePolyhedron {
  ConvexPolyhedron poly;
  std::vector<ClipPlane> planes;
  double surface_error_bound{0};
};

inline void add_capsule_face(CapsulePolyhedron& result, const CapsuleBody& body,
                             std::vector<BodyVector> vertices, BodyVector inside_reference) {
  const double epsilon = polygon_epsilon(vertices);

  if (polygon_area(vertices) <= epsilon * epsilon) {
    return;
  }

  auto normal = body_normalized(body_cross(body_difference(vertices[1], vertices[0]),
                                           body_difference(vertices[2], vertices[0])));

  if (body_dot(normal, body_difference(vertices[0], inside_reference)) < 0) {
    std::reverse(vertices.begin(), vertices.end());

    for (double& x : normal) {
      x = -x;
    }
  }

  const double offset = body_dot(normal, body_difference(vertices[0], inside_reference));
  result.surface_error_bound = std::max(result.surface_error_bound, body.radius - offset);

  for (auto& p : vertices) {
    p = body_rotate(body.orientation, p);

    for (std::size_t d = 0; d < 3; ++d) {
      p[d] += body.position[d];
    }
  }

  normal = body_rotate(body.orientation, normal);
  const auto tag = std::uint64_t{1} << 63 | result.planes.size();
  result.planes.push_back({normal, body_dot(normal, vertices[0]), tag});
  result.poly.faces.push_back({std::move(vertices), tag});
}

// A convex capsule approximation with exactly the analytical cross-sectional
// area and capsule volume. Transverse and axial corrections tend to one under
// angular refinement. Their maximum surface error is reported explicitly.
inline CapsulePolyhedron capsule_polyhedron(const CapsuleBody& body, unsigned resolution) {
  if (resolution < 8 || resolution % 4 != 0 || resolution > 1024) {
    throw std::invalid_argument(
        "capsule surface resolution must be a multiple of four in [8,1024]");
  }

  const unsigned latitude = resolution / 4;
  const double pi = std::numbers::pi;
  const double area = 0.5 * resolution * std::sin(2 * pi / resolution);
  const double transverse = std::sqrt(pi / area);
  double sphere_volume = 0;

  for (unsigned i = 0; i < latitude; ++i) {
    const double a = double(i) * pi / (2 * latitude), b = double(i + 1) * pi / (2 * latitude);
    const double r0 = std::sin(a), r1 = std::sin(b), dx = std::cos(a) - std::cos(b);
    sphere_volume += 2 * dx * area * (r0 * r0 + r0 * r1 + r1 * r1) / 3;
  }

  const double axial = (4 * pi / 3) / (sphere_volume * transverse * transverse);
  CapsulePolyhedron result;
  result.surface_error_bound = body.radius * (std::max(transverse, axial) - 1);
  auto ring_point = [&](int sign, unsigned latitude_index, unsigned longitude, bool offset_length) {
    const double theta = double(latitude_index) * pi / (2 * latitude);
    const double phi = double(longitude % resolution) * 2 * pi / resolution;

    return BodyVector{
        sign * (axial * body.radius * std::cos(theta) + (offset_length ? body.length / 2 : 0)),
        transverse * body.radius * std::sin(theta) * std::cos(phi),
        transverse * body.radius * std::sin(theta) * std::sin(phi)};
  };
  auto add_face = [&](std::vector<BodyVector> vertices, BodyVector inside_reference) {
    add_capsule_face(result, body, std::move(vertices), inside_reference);
  };

  for (int sign : {-1, 1}) {
    for (unsigned i = 0; i < latitude; ++i) {
      for (unsigned j = 0; j < resolution; ++j) {
        std::vector<BodyVector> face;
        face.push_back(ring_point(sign, i, j, true));

        if (i != 0) {
          face.push_back(ring_point(sign, i, j + 1, true));
        }

        face.push_back(ring_point(sign, i + 1, j + 1, true));
        face.push_back(ring_point(sign, i + 1, j, true));
        add_face(std::move(face), {sign * body.length / 2, 0, 0});
      }
    }
  }

  if (body.length > 0) {
    for (unsigned j = 0; j < resolution; ++j) {
      add_face({ring_point(-1, latitude, j, true), ring_point(-1, latitude, j + 1, true),
                ring_point(1, latitude, j + 1, true), ring_point(1, latitude, j, true)},
               {0, 0, 0});
    }
  }

  return result;
}
}  // namespace cm::detail
