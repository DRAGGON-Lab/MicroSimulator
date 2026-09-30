#include "cm/fluid_geometry.hpp"

#include <map>
#include <numeric>
#include <set>

#include "convex_polyhedron.hpp"
#include "flow_system.hpp"

namespace cm {
namespace {
using namespace detail;
constexpr std::uint64_t body_tag = std::uint64_t{1} << 63;

struct Bounds {
  BodyVector lo, hi;
};

Bounds bounds(const ConvexPolyhedron& p) {
  Bounds b{{INFINITY, INFINITY, INFINITY}, {-INFINITY, -INFINITY, -INFINITY}};

  for (const auto& f : p.faces) {
    for (const auto& v : f.vertices) {
      for (unsigned d = 0; d < 3; ++d) {
        b.lo[d] = std::min(b.lo[d], v[d]);
        b.hi[d] = std::max(b.hi[d], v[d]);
      }
    }
  }

  return b;
}

bool overlap(const Bounds& a, const Bounds& b, double tolerance = 0) {
  for (unsigned d = 0; d < 3; ++d) {
    if (a.lo[d] > b.hi[d] + tolerance || a.hi[d] < b.lo[d] - tolerance) {
      return false;
    }
  }

  return true;
}

BodyVector normal(const ConvexPolygon& p) {
  BodyVector area{};

  for (std::size_t i = 2; i < p.vertices.size(); ++i) {
    const auto c = body_cross(body_difference(p.vertices[i - 1], p.vertices[0]),
                              body_difference(p.vertices[i], p.vertices[0]));

    for (unsigned d = 0; d < 3; ++d) {
      area[d] += c[d];
    }
  }

  return body_normalized(area);
}

std::pair<double, BodyVector> polygon_measure(const std::vector<BodyVector>& p) {
  double area = 0;
  BodyVector centroid{};

  for (std::size_t i = 2; i < p.size(); ++i) {
    const auto c = body_cross(body_difference(p[i - 1], p[0]), body_difference(p[i], p[0]));
    const double a = 0.5 * std::sqrt(body_dot(c, c));
    area += a;

    for (unsigned d = 0; d < 3; ++d) {
      centroid[d] += a * (p[0][d] + p[i - 1][d] + p[i][d]) / 3;
    }
  }

  if (area > 0) {
    for (double& v : centroid) {
      v /= area;
    }
  }

  return {area, centroid};
}

std::vector<BodyVector> intersect_polygon(std::vector<BodyVector> p, const ConvexPolygon& q) {
  const auto n = normal(q);
  const double epsilon = polygon_epsilon(q.vertices);

  for (std::size_t i = 0; i < q.vertices.size() && !p.empty(); ++i) {
    const auto& a = q.vertices[i];
    const auto& b = q.vertices[(i + 1) % q.vertices.size()];
    const auto edge = body_difference(b, a);

    if (body_dot(edge, edge) <= epsilon * epsilon) {
      continue;
    }

    const auto outward = body_normalized(body_cross(edge, n));
    p = clip_polygon(p, {outward, body_dot(outward, a), 0});
  }

  return p;
}

std::uint64_t poly_bytes(const ConvexPolyhedron& p) {
  std::uint64_t bytes = sizeof(p);

  for (const auto& f : p.faces) {
    bytes += sizeof(f) + f.vertices.size() * sizeof(BodyVector);
  }

  return bytes;
}
}  // namespace

struct FluidGeometry::Impl {
  FluidGridSpec grid;
  std::vector<CapsuleBody> bodies;
  std::vector<ConvexPolyhedron> pieces;
  std::vector<std::uint32_t> piece_fragment;
  std::vector<FluidFragment> fragments;
  std::vector<FluidFace> faces;
  FluidGeometryReport report;
  std::uint32_t surface_resolution{0};

  struct FaceRef {
    std::uint32_t fragment;
    const ConvexPolygon* polygon;
    BodyVector normal, centroid, unmatched_moment;
    Bounds bound;
    double area, unmatched;
  };

  std::vector<CapsuleBody> local_body_coordinates(const BodyVector& origin) const {
    auto local_bodies = bodies;
    std::set<CellId> ids;

    for (auto& body : local_bodies) {
      body.validate();

      if (!ids.insert(body.id).second) {
        throw std::invalid_argument("duplicate fluid geometry body id");
      }

      for (unsigned d = 0; d < 3; ++d) {
        body.position[d] -= origin[d];
      }
    }

    return local_bodies;
  }

  void build_surfaces(const std::vector<CapsuleBody>& local_bodies,
                      const FluidGeometryParameters& p, std::vector<CapsulePolyhedron>& surfaces,
                      std::vector<Bounds>& surface_bounds,
                      std::map<std::uint64_t, std::size_t>& tag_owner, const auto& charge) {
    const double h = grid.spacing;
    std::uint64_t next_tag = body_tag;

    for (std::size_t i = 0; i < local_bodies.size(); ++i) {
      charge(std::uint64_t(p.surface_resolution) * p.surface_resolution * 512);
      auto surface = capsule_polyhedron(local_bodies[i], p.surface_resolution);

      if (surface.surface_error_bound > p.maximum_surface_error_fraction * h) {
        throw std::invalid_argument(
            "culture capsule surface error exceeds configured limit; refine surface_resolution");
      }

      report.maximum_surface_error =
          std::max(report.maximum_surface_error, surface.surface_error_bound);

      for (std::size_t f = 0; f < surface.planes.size(); ++f) {
        surface.planes[f].tag = next_tag;
        surface.poly.faces[f].tag = next_tag;
        tag_owner[next_tag++] = i;
      }

      surface_bounds.push_back(bounds(surface.poly));
      surfaces.push_back(std::move(surface));
    }
  }

  void validate_surface_gaps(const std::vector<CapsulePolyhedron>& surfaces) const {
    // The surface approximation must not close a physical gap. Reject uncertain
    // gaps so the caller can refine; never bridge fluid fragments by a cutoff.
    const auto contacts = capsule_contacts(grid, bodies, 2 * report.maximum_surface_error, 1, 0);

    for (const auto& c : contacts) {
      const double error =
          surfaces[c.first].surface_error_bound +
          (c.second == fluid_wall_body ? 0 : surfaces[c.second].surface_error_bound);

      if (c.gap <= error) {
        throw std::invalid_argument(
            "fluid geometry gap is smaller than surface error; refine surface_resolution");
      }
    }
  }

  static bool outside_surface(const ConvexPolyhedron& poly, const CapsulePolyhedron& surface) {
    bool outside = false;

    for (const auto& plane : surface.planes) {
      double minimum = INFINITY;

      for (const auto& face : poly.faces) {
        for (const auto& point : face.vertices) {
          minimum = std::min(minimum, body_dot(plane.normal, point) - plane.offset);
        }
      }

      if (minimum >= 0) {
        outside = true;
        break;
      }
    }

    return outside;
  }

  static void subtract_surface(std::vector<ConvexPolyhedron>& fluid,
                               const CapsulePolyhedron& surface,
                               std::vector<ConvexPolyhedron>& next, const auto& charge) {
    for (auto& poly : fluid) {
      if (outside_surface(poly, surface)) {
        next.push_back(std::move(poly));
        continue;
      }

      for (const auto& plane : surface.planes) {
        if (poly.faces.empty()) {
          break;
        }

        ClipPlane reverse = plane;

        for (double& x : reverse.normal) {
          x = -x;
        }

        reverse.offset = -reverse.offset;
        auto cut = clip_polyhedron(poly, reverse);

        if (measure_polyhedron(cut).volume > 0) {
          charge(poly_bytes(cut) * 3);
          next.push_back(std::move(cut));
        }

        poly = clip_polyhedron(poly, plane);
      }
    }
  }

  static std::vector<ConvexPolyhedron> voxel_fluid(const Bounds& box,
                                                   const std::array<std::uint64_t, 6>& tags,
                                                   const std::vector<CapsulePolyhedron>& surfaces,
                                                   const std::vector<Bounds>& surface_bounds,
                                                   const auto& charge) {
    std::vector<ConvexPolyhedron> fluid{box_polyhedron(box.lo, box.hi, tags)};

    for (std::size_t body = 0; body < surfaces.size() && !fluid.empty(); ++body) {
      if (!overlap(box, surface_bounds[body])) {
        continue;
      }

      const auto& surface = surfaces[body];
      std::vector<ConvexPolyhedron> next;

      subtract_surface(fluid, surface, next, charge);

      fluid = std::move(next);
    }

    return fluid;
  }

  std::size_t build_fragments(const FlowGridLayout& layout,
                              const std::vector<CapsulePolyhedron>& surfaces,
                              const std::vector<Bounds>& surface_bounds, const auto& charge) {
    const auto& spec = grid;
    const double h = grid.spacing;
    std::size_t fluid_sites = 0;

    for (std::size_t site = 0; site < grid.site_count(); ++site) {
      if (spec.solid_site(site)) {
        continue;
      }

      ++fluid_sites;
      const auto xyz = layout.site_coordinates(site);
      Bounds box;
      std::array<std::uint64_t, 6> tags;

      for (unsigned d = 0; d < 3; ++d) {
        box.lo[d] = (double(xyz[d]) - 0.5) * h;
        box.hi[d] = box.lo[d] + h;
        auto upper = xyz;
        ++upper[d];
        tags[2 * d] = layout.face_index(d, xyz[0], xyz[1], xyz[2]);
        tags[2 * d + 1] = layout.face_index(d, upper[0], upper[1], upper[2]);
      }

      auto fluid = voxel_fluid(box, tags, surfaces, surface_bounds, charge);

      for (auto& poly : fluid) {
        const auto m = measure_polyhedron(poly);

        if (!(m.volume > 0) || !std::isfinite(m.volume)) {
          throw std::runtime_error("invalid extracellular fragment volume");
        }

        charge(poly_bytes(poly) * 3 + 256);

        if (fragments.size() >= fluid_boundary) {
          throw std::length_error("too many culture fragments");
        }

        fragments.push_back({static_cast<std::uint32_t>(site), 0, m.volume, m.centroid});
        report.fluid_volume += m.volume;
        pieces.push_back(std::move(poly));
      }
    }

    return fluid_sites;
  }

  void validate_volume(std::size_t fluid_sites) {
    const double h = grid.spacing;
    report.expected_fluid_volume = double(fluid_sites) * h * h * h;

    for (const auto& body : bodies) {
      report.expected_fluid_volume -= body.geometric_volume();
    }

    report.volume_error = report.fluid_volume - report.expected_fluid_volume;

    if (std::abs(report.volume_error) > 1e-10 * double(fluid_sites) * h * h * h) {
      throw std::runtime_error("extracellular geometry failed volume conservation");
    }
  }

  std::map<std::uint64_t, std::vector<FaceRef>> group_piece_faces(double area_tolerance) const {
    std::map<std::uint64_t, std::vector<FaceRef>> groups;

    for (std::uint32_t i = 0; i < pieces.size(); ++i) {
      for (const auto& face : pieces[i].faces) {
        const auto [area, center] = polygon_measure(face.vertices);

        if (area <= area_tolerance) {
          continue;
        }

        BodyVector moment;

        for (unsigned d = 0; d < 3; ++d) {
          moment[d] = area * center[d];
        }

        groups[face.tag].push_back(
            {i, &face, normal(face), center, moment, bounds({{face}}), area, area});
      }
    }

    return groups;
  }

  void match_internal_faces(std::uint64_t tag, std::vector<FaceRef>& group, double h,
                            double area_tolerance, const auto& append_face) const {
    for (std::size_t i = 0; i < group.size(); ++i) {
      for (std::size_t j = i + 1; j < group.size(); ++j) {
        auto& a = group[i];
        auto& b = group[j];

        if ((tag >= body_tag && fragments[a.fragment].site != fragments[b.fragment].site) ||
            body_dot(a.normal, b.normal) > -0.999999999 || !overlap(a.bound, b.bound, 1e-12 * h)) {
          continue;
        }

        const auto intersection = intersect_polygon(a.polygon->vertices, *b.polygon);
        const auto [area, center] = polygon_measure(intersection);

        if (area <= area_tolerance) {
          continue;
        }

        a.unmatched -= area;
        b.unmatched -= area;

        for (unsigned d = 0; d < 3; ++d) {
          a.unmatched_moment[d] -= area * center[d];
          b.unmatched_moment[d] -= area * center[d];
        }

        append_face(a.fragment, b.fragment, area, center, a.normal, tag);
      }
    }
  }

  CellId coincident_surface_body(const FaceRef& a, const BodyVector& center,
                                 const std::vector<CapsulePolyhedron>& surfaces, double h) const {
    CellId body = 0;

    // A membrane can coincide with a grid face. Identify its surface
    // plane explicitly instead of labelling it as an impermeable wall.
    for (std::size_t k = 0; k < surfaces.size(); ++k) {
      if (!std::all_of(surfaces[k].planes.begin(), surfaces[k].planes.end(),
                       [&](const auto& plane) {
                         return plane_distance(plane, center) <= 1e-10 * h;
                       })) {
        continue;
      }

      for (const auto& plane : surfaces[k].planes) {
        if (body_dot(a.normal, plane.normal) < -1 + 1e-12 &&
            std::abs(body_dot(plane.normal, center) - plane.offset) < 1e-10 * h) {
          body = bodies[k].id;
        }
      }
    }

    return body;
  }

  CellId unmatched_face_body(const FaceRef& a, std::uint64_t tag, const BodyVector& center,
                             const FlowGridLayout& layout,
                             const std::vector<CapsulePolyhedron>& surfaces,
                             const std::map<std::uint64_t, std::size_t>& tag_owner,
                             double h) const {
    const auto& spec = grid;
    CellId body = 0;

    if (tag >= body_tag) {
      const auto owner = tag_owner.at(tag);
      body = bodies[owner].id;
      const auto& planes = surfaces[owner].planes;
      const auto plane = std::find_if(planes.begin(), planes.end(), [&](const auto& q) {
        return q.tag == tag;
      });

      if (body_dot(a.normal, plane->normal) > 0) {
        throw std::runtime_error(
            "unmatched internal extracellular partition: area=" + std::to_string(a.unmatched) +
            ", fragment=" + std::to_string(a.fragment) + ", tag=" + std::to_string(tag));
      }
    } else {
      const auto [axis, xyz] = layout.face_coordinates(tag);
      const auto other = layout.adjacent_site(axis, xyz, a.normal[axis] > 0 ? 1 : -1);

      if (other && !spec.solid_site(*other)) {
        body = coincident_surface_body(a, center, surfaces, h);

        if (!body) {
          throw std::runtime_error(
              "unmatched open extracellular grid face: area=" + std::to_string(a.unmatched) +
              ", site=" + std::to_string(fragments[a.fragment].site) +
              ", face=" + std::to_string(tag));
        }
      }
    }

    return body;
  }

  void append_boundary_faces(std::uint64_t tag, const std::vector<FaceRef>& group,
                             const FlowGridLayout& layout,
                             const std::vector<CapsulePolyhedron>& surfaces,
                             const std::map<std::uint64_t, std::size_t>& tag_owner, double h,
                             const auto& append_face) const {
    for (const auto& a : group) {
      if (a.unmatched < -1e-9 * h * h) {
        throw std::runtime_error("overlapping extracellular faces");
      }

      if (a.unmatched <= 1e-10 * h * h) {
        continue;
      }

      BodyVector center;

      for (unsigned d = 0; d < 3; ++d) {
        center[d] = a.unmatched_moment[d] / a.unmatched;
      }

      const auto body = unmatched_face_body(a, tag, center, layout, surfaces, tag_owner, h);

      append_face(a.fragment, fluid_boundary, a.unmatched, center, a.normal, tag, body);
    }
  }

  void assign_components(const BodyVector& origin, const auto& find_root) {
    std::map<std::uint32_t, std::uint32_t> components;

    for (std::uint32_t i = 0; i < fragments.size(); ++i) {
      const auto r = find_root(i);
      auto [it, inserted] =
          components.try_emplace(r, static_cast<std::uint32_t>(components.size()));
      fragments[i].component = it->second;

      for (unsigned d = 0; d < 3; ++d) {
        fragments[i].centroid[d] += origin[d];
      }
    }

    report.component_count = static_cast<std::uint32_t>(components.size());

    for (auto& face : faces) {
      for (unsigned d = 0; d < 3; ++d) {
        face.centroid[d] += origin[d];
      }
    }
  }

  void merge_fragments(std::vector<std::uint32_t>& root, const auto& find_root) {
    // Merge only pieces connected by a positive-area face within one voxel.
    // The convex decomposition remains available for exact overlap integrals.
    std::iota(root.begin(), root.end(), 0);

    for (const auto& face : faces) {
      if (face.second != fluid_boundary &&
          fragments[face.first].site == fragments[face.second].site) {
        root[find_root(face.first)] = find_root(face.second);
      }
    }

    std::map<std::uint32_t, std::uint32_t> merged_ids;
    std::vector<FluidFragment> merged;
    piece_fragment.resize(fragments.size());

    for (std::uint32_t i = 0; i < fragments.size(); ++i) {
      auto [it, inserted] =
          merged_ids.try_emplace(find_root(i), static_cast<std::uint32_t>(merged.size()));

      if (inserted) {
        merged.push_back({fragments[i].site, fragments[i].component});
      }

      const auto j = it->second;
      piece_fragment[i] = j;
      merged[j].volume += fragments[i].volume;

      for (unsigned d = 0; d < 3; ++d) {
        merged[j].centroid[d] += fragments[i].volume * fragments[i].centroid[d];
      }
    }

    for (auto& f : merged) {
      for (double& x : f.centroid) {
        x /= f.volume;
      }
    }

    for (auto& face : faces) {
      face.first = piece_fragment[face.first];

      if (face.second != fluid_boundary) {
        face.second = piece_fragment[face.second];
      }
    }

    std::erase_if(faces, [](const auto& face) {
      return face.first == face.second;
    });
    fragments = std::move(merged);
  }

  void validate_overlap_grid(const Impl& b) const {
    const auto& a = *this;

    if (a.grid.shape.x != b.grid.shape.x || a.grid.shape.y != b.grid.shape.y ||
        a.grid.shape.z != b.grid.shape.z || a.grid.origin.x != b.grid.origin.x ||
        a.grid.origin.y != b.grid.origin.y || a.grid.origin.z != b.grid.origin.z ||
        a.grid.spacing != b.grid.spacing || a.grid.obstacles != b.grid.obstacles) {
      throw std::invalid_argument("culture overlap requires the same device grid");
    }
  }

  bool identical_surface(const Impl& b) const {
    const auto& a = *this;
    bool identical =
        a.bodies.size() == b.bodies.size() && a.surface_resolution == b.surface_resolution;

    for (std::size_t i = 0; i < a.bodies.size() && identical; ++i) {
      identical = a.bodies[i].position == b.bodies[i].position &&
                  a.bodies[i].orientation == b.bodies[i].orientation &&
                  a.bodies[i].length == b.bodies[i].length &&
                  a.bodies[i].radius == b.bodies[i].radius;
    }

    return identical;
  }

  Impl(const FluidGridSpec& g, const std::vector<CapsuleBody>& b, const FluidGeometryParameters& p)
      : grid(g), bodies(b), surface_resolution(p.surface_resolution) {
    grid.validate();
    p.validate();
    const double h = grid.spacing, area_tolerance = 1e-14 * h * h;
    // Work in coordinates relative to the grid origin. This avoids subtracting
    // large absolute coordinates in clipping and volume integration.
    const BodyVector origin{grid.origin.x, grid.origin.y, grid.origin.z};
    const auto local_bodies = local_body_coordinates(origin);

    std::vector<CapsulePolyhedron> surfaces;
    std::vector<Bounds> surface_bounds;
    std::map<std::uint64_t, std::size_t> tag_owner;
    report.estimated_memory_bytes = grid.site_count() * 256;
    auto charge = [&](std::uint64_t bytes) {
      if (bytes > p.memory_limit_bytes ||
          report.estimated_memory_bytes > p.memory_limit_bytes - bytes) {
        throw std::length_error("fluid geometry exceeds memory_limit_bytes");
      }

      report.estimated_memory_bytes += bytes;
    };
    charge(0);

    build_surfaces(local_bodies, p, surfaces, surface_bounds, tag_owner, charge);

    validate_surface_gaps(surfaces);

    const auto& spec = grid;
    const FlowGridLayout layout(spec, FlowAxis::x);

    if (layout.total_face_count() >= fluid_boundary) {
      throw std::length_error("fluid geometry exceeds uint32 face indexing");
    }

    const auto fluid_sites = build_fragments(layout, surfaces, surface_bounds, charge);

    validate_volume(fluid_sites);

    auto groups = group_piece_faces(area_tolerance);

    std::vector<std::uint32_t> root(fragments.size());
    std::iota(root.begin(), root.end(), 0);
    auto find_root = [&](std::uint32_t i) {
      while (i != root[i]) {
        root[i] = root[root[i]];
        i = root[i];
      }

      return i;
    };
    auto append_face = [&](std::uint32_t a, std::uint32_t b, double area, BodyVector center,
                           BodyVector n, std::uint64_t tag, CellId body = 0) {
      FluidFace face{a, b, area, center, n, body};

      if (tag < body_tag) {
        face.grid_face = static_cast<std::uint32_t>(tag);
        face.axis = static_cast<FlowAxis>(layout.face_coordinates(tag).first);
      }

      charge(sizeof(FluidFace) * 2);
      faces.push_back(face);

      if (b != fluid_boundary) {
        root[find_root(a)] = find_root(b);
      }
    };

    for (auto& [tag, group] : groups) {
      match_internal_faces(tag, group, h, area_tolerance, append_face);

      append_boundary_faces(tag, group, layout, surfaces, tag_owner, h, append_face);
    }

    assign_components(origin, find_root);

    merge_fragments(root, find_root);
  }
};

void FluidGeometryParameters::validate() const {
  if (surface_resolution < 8 || surface_resolution % 4 || surface_resolution > 1024 ||
      !std::isfinite(maximum_surface_error_fraction) || maximum_surface_error_fraction <= 0 ||
      maximum_surface_error_fraction > 0.25 || memory_limit_bytes == 0) {
    throw std::invalid_argument("invalid fluid geometry parameters");
  }
}

FluidGeometry::FluidGeometry(const FluidGridSpec& g, const std::vector<CapsuleBody>& b,
                             const FluidGeometryParameters& p)
    : impl_(std::make_shared<Impl>(g, b, p)) {}

const FluidGridSpec& FluidGeometry::grid() const noexcept {
  return impl_->grid;
}

const std::vector<CapsuleBody>& FluidGeometry::bodies() const noexcept {
  return impl_->bodies;
}

const std::vector<FluidFragment>& FluidGeometry::fragments() const noexcept {
  return impl_->fragments;
}

const std::vector<FluidFace>& FluidGeometry::faces() const noexcept {
  return impl_->faces;
}

const FluidGeometryReport& FluidGeometry::report() const noexcept {
  return impl_->report;
}

std::vector<FluidOverlap> FluidGeometry::overlaps(const FluidGeometry& other) const {
  const auto& a = *impl_;
  const auto& b = *other.impl_;

  a.validate_overlap_grid(b);

  const bool identical = a.identical_surface(b);

  std::vector<FluidOverlap> result;

  if (identical) {
    for (std::uint32_t i = 0; i < a.fragments.size(); ++i) {
      result.push_back({i, i, a.fragments[i].volume});
    }

    return result;
  }

  std::vector<std::vector<std::uint32_t>> by_site(a.grid.site_count());
  std::vector<Bounds> b_bounds;
  std::vector<std::vector<ClipPlane>> b_planes;

  for (std::uint32_t i = 0; i < b.pieces.size(); ++i) {
    by_site[b.fragments[b.piece_fragment[i]].site].push_back(i);
    b_bounds.push_back(bounds(b.pieces[i]));
    std::vector<ClipPlane> planes;

    for (const auto& face : b.pieces[i].faces) {
      const auto n = normal(face);
      planes.push_back({n, body_dot(n, face.vertices[0]), face.tag});
    }

    b_planes.push_back(std::move(planes));
  }

  std::map<std::pair<std::uint32_t, std::uint32_t>, double> volumes;

  for (std::uint32_t i = 0; i < a.pieces.size(); ++i) {
    const auto box = bounds(a.pieces[i]);

    for (auto j : by_site[a.fragments[a.piece_fragment[i]].site]) {
      if (!overlap(box, b_bounds[j])) {
        continue;
      }

      auto intersection = a.pieces[i];

      for (const auto& plane : b_planes[j]) {
        intersection = clip_polyhedron(intersection, plane);

        if (intersection.faces.empty()) {
          break;
        }
      }

      const double v = measure_polyhedron(intersection).volume;

      if (v > 0) {
        volumes[{a.piece_fragment[i], b.piece_fragment[j]}] += v;
      }
    }
  }

  for (const auto& [pair, v] : volumes) {
    result.push_back({pair.first, pair.second, v});
  }

  return result;
}
}  // namespace cm
