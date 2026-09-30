#include "cm/fluid_geometry.hpp"

#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include "core/convex_polyhedron.hpp"

namespace {
void check(bool value, const char* message) {
  if (!value) {
    throw std::runtime_error(message);
  }
}

void polyhedra() {
  using namespace cm::detail;
  const auto box = box_polyhedron({-1, -1, -1}, {1, 1, 1});
  check(std::abs(measure_polyhedron(box).volume - 8) < 1e-13, "box volume");
  const auto half = clip_polyhedron(box, {{1, 0, 0}, 0, 1});
  const auto measure = measure_polyhedron(half);
  check(std::abs(measure.volume - 4) < 1e-13 && std::abs(measure.centroid[0] + 0.5) < 1e-13,
        "half-box volume and centroid");
  cm::CapsuleBody body;
  body.position = {12.3, -4.6, 1.4};
  body.radius = 1.7;
  body.length = 3.9;
  body.orientation = {std::cos(0.31), 0, std::sin(0.31), 0};
  double previous_error = 1;

  for (unsigned resolution : {8, 16, 32, 64}) {
    auto surface = capsule_polyhedron(body, resolution);
    const auto m = measure_polyhedron(surface.poly);
    check(std::abs(m.volume / body.geometric_volume() - 1) < 1e-13, "capsule exact volume");

    for (unsigned d = 0; d < 3; ++d) {
      check(std::abs(m.centroid[d] - body.position[d]) < 1e-12, "capsule centroid");
    }

    check(surface.surface_error_bound < previous_error / 3, "capsule surface refinement");
    previous_error = surface.surface_error_bound;

    // Every surface vertex must be inside every halfspace: the volume-preserving
    // corrections must retain convexity, including the cylinder/cap junctions.
    for (const auto& plane : surface.planes) {
      for (const auto& face : surface.poly.faces) {
        for (const auto& point : face.vertices) {
          check(body_dot(plane.normal, point) - plane.offset < 1e-12, "capsule convexity");
        }
      }
    }

    double clipped_volume = 0;

    for (int slab = -8; slab < 32; ++slab) {
      auto clipped = clip_polyhedron(surface.poly, {{1, 0, 0}, 0.5 * (slab + 1), 1});
      clipped = clip_polyhedron(clipped, {{-1, 0, 0}, -0.5 * slab, 2});
      clipped_volume += measure_polyhedron(clipped).volume;
    }

    check(std::abs(clipped_volume / body.geometric_volume() - 1) < 1e-13,
          "cut-cell capsule volume partition");
  }
}

void extracellular() {
  cm::FluidGridSpec grid;
  grid.shape = {8, 8, 8};
  cm::FluidGeometry empty(grid, {});
  check(empty.fragments().size() == 512 && empty.report().fluid_volume == 512 &&
            empty.report().component_count == 1,
        "empty fluid mesh");
  cm::CapsuleBody body;
  body.position = {3.31, 3.67, 3.48};
  body.radius = 1.2;
  body.length = 1.3;
  body.orientation = {std::cos(0.31), 0, std::sin(0.31), 0};
  cm::FluidGeometry geometry(grid, {body});
  check(std::abs(geometry.report().fluid_volume - (512 - body.geometric_volume())) < 1e-9,
        "extracellular volume excludes exact capsule volume");
  check(geometry.report().component_count == 1, "capsule exterior connected");
  std::vector<std::array<double, 3>> closure(geometry.fragments().size());
  double membrane_area = 0;

  for (const auto& face : geometry.faces()) {
    for (unsigned d = 0; d < 3; ++d) {
      closure[face.first][d] += face.area * face.normal[d];

      if (face.second != cm::fluid_boundary) {
        closure[face.second][d] -= face.area * face.normal[d];
      }
    }

    if (face.body_id) {
      check(face.body_id == body.id, "membrane body id");
      membrane_area += face.area;
    }
  }

  for (const auto& sum : closure) {
    for (double x : sum) {
      check(std::abs(x) < 1e-9, "extracellular fragment face closure");
    }
  }

  const auto surface = cm::detail::capsule_polyhedron(body, 32);
  double analytic_mesh_area = 0;

  for (const auto& face : surface.poly.faces) {
    analytic_mesh_area += cm::detail::polygon_area(face.vertices);
  }

  check(std::abs(membrane_area / analytic_mesh_area - 1) < 1e-9, "membrane area partition");
  cm::FluidGeometryParameters refined_parameters;
  refined_parameters.surface_resolution = 64;
  const cm::FluidGeometry refined(grid, {body}, refined_parameters);
  double overlap_volume = 0;

  for (const auto& overlap : geometry.overlaps(refined)) {
    overlap_volume += overlap.volume;
  }

  // Equal analytic body volumes do not make two surface approximations identical.
  // Their common fluid region excludes the union of both polyhedra.
  check(overlap_volume < geometry.report().fluid_volume - 1e-4 &&
            overlap_volume > geometry.report().fluid_volume - 0.1,
        "surface refinement computes geometric overlap instead of an identity remap");
  std::cout << "extracellular fragments " << geometry.fragments().size() << " faces "
            << geometry.faces().size() << " estimated bytes "
            << geometry.report().estimated_memory_bytes << '\n';
  grid.obstacles.assign(512, 0);

  for (unsigned x = 0; x < 8; ++x) {
    for (unsigned z = 0; z < 8; ++z) {
      grid.obstacles[(x * 8 + 4) * 8 + z] = 1;
    }
  }

  cm::FluidGeometry split(grid, {});
  check(split.report().component_count == 2 && split.report().fluid_volume == 448,
        "wall separates extracellular components");
  cm::FluidGridSpec divided_grid;
  divided_grid.shape = {18, 12, 12};
  cm::CapsuleBody first, second;
  first.position = second.position = {8.5, 5.5, 5.5};
  first.radius = second.radius = 2;
  second.id = 2;
  const double fraction = double(0.4F), cap = 8.0 / 3;
  first.length = fraction * (8 + cap) - cap;
  second.length = (1 - fraction) * (8 + cap) - cap;
  const double separation = (first.length + second.length) / 2 + 4 + 0.10001;
  first.position[0] -= (1 - fraction) * separation;
  second.position[0] += fraction * separation;
  const cm::FluidGeometry daughters(divided_grid, {first, second});
  check(std::abs(daughters.report().volume_error) < 1e-9, "daughter extracellular geometry volume");
}
}  // namespace

int main() {
  polyhedra();
  extracellular();
  std::cout << "media geometry passed\n";
}
