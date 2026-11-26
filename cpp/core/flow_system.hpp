#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/parallel.hpp"
#include "microsimulator/flow.hpp"

namespace microsimulator::detail {

// Sites and faces handed to one thread at a time.
inline constexpr std::size_t flow_grain = 4096;

class FlowGridLayout {
 public:
  explicit FlowGridLayout(const SignalGridSpec& spec)
      : dimensions_{spec.shape.x, spec.shape.y, spec.shape.z},
        spacing_{spec.spacing.x, spec.spacing.y, spec.spacing.z} {
    face_counts_[0] = spec.x_face_count();
    face_counts_[1] = spec.y_face_count();
    face_counts_[2] = spec.z_face_count();
    face_offsets_[1] = face_counts_[0];
    face_offsets_[2] = face_counts_[0] + face_counts_[1];
    total_face_count_ = face_offsets_[2] + face_counts_[2];
  }

  [[nodiscard]] const std::array<std::uint32_t, 3>& dimensions() const noexcept {
    return dimensions_;
  }

  [[nodiscard]] const std::array<float, 3>& spacing() const noexcept {
    return spacing_;
  }

  [[nodiscard]] std::size_t site_count() const noexcept {
    return static_cast<std::size_t>(dimensions_[0]) * dimensions_[1] * dimensions_[2];
  }

  [[nodiscard]] std::size_t face_count(std::size_t component) const noexcept {
    return face_counts_[component];
  }

  [[nodiscard]] const std::array<std::size_t, 3>& face_counts() const noexcept {
    return face_counts_;
  }

  [[nodiscard]] const std::array<std::size_t, 3>& face_offsets() const noexcept {
    return face_offsets_;
  }

  [[nodiscard]] std::size_t total_face_count() const noexcept {
    return total_face_count_;
  }

  [[nodiscard]] std::size_t site_index(std::uint32_t x, std::uint32_t y,
                                       std::uint32_t z) const noexcept {
    return (static_cast<std::size_t>(x) * dimensions_[1] + y) * dimensions_[2] + z;
  }

  [[nodiscard]] std::array<std::uint32_t, 3> site_coordinates(std::size_t index) const noexcept {
    const auto z = static_cast<std::uint32_t>(index % dimensions_[2]);
    index /= dimensions_[2];
    const auto y = static_cast<std::uint32_t>(index % dimensions_[1]);
    return {static_cast<std::uint32_t>(index / dimensions_[1]), y, z};
  }

  [[nodiscard]] std::array<std::uint32_t, 3> face_dimensions(std::size_t component) const noexcept {
    auto result = dimensions_;
    ++result[component];
    return result;
  }

  [[nodiscard]] std::size_t local_face_index(std::size_t component, std::uint32_t x,
                                             std::uint32_t y, std::uint32_t z) const noexcept {
    if (component == 0) {
      return (static_cast<std::size_t>(x) * dimensions_[1] + y) * dimensions_[2] + z;
    }
    if (component == 1) {
      return (static_cast<std::size_t>(x) * (dimensions_[1] + 1) + y) * dimensions_[2] + z;
    }
    return (static_cast<std::size_t>(x) * dimensions_[1] + y) * (dimensions_[2] + 1) + z;
  }

  [[nodiscard]] std::size_t face_index(std::size_t component, std::uint32_t x, std::uint32_t y,
                                       std::uint32_t z) const noexcept {
    return face_offsets_[component] + local_face_index(component, x, y, z);
  }

  [[nodiscard]] std::pair<std::size_t, std::array<std::uint32_t, 3>> face_coordinates(
      std::size_t index) const noexcept {
    std::size_t component = 0;
    while (component < 2 && index >= face_offsets_[component] + face_counts_[component]) {
      ++component;
    }
    auto local = index - face_offsets_[component];
    const auto face_dims = face_dimensions(component);
    const auto z = static_cast<std::uint32_t>(local % face_dims[2]);
    local /= face_dims[2];
    const auto y = static_cast<std::uint32_t>(local % face_dims[1]);
    return {component, {static_cast<std::uint32_t>(local / face_dims[1]), y, z}};
  }

  [[nodiscard]] std::optional<std::size_t> adjacent_site(std::size_t component,
                                                         const std::array<std::uint32_t, 3>& face,
                                                         int side) const noexcept {
    auto site = face;
    if (side < 0) {
      if (face[component] == 0) {
        return std::nullopt;
      }
      --site[component];
    } else if (face[component] >= dimensions_[component]) {
      return std::nullopt;
    }
    return site_index(site[0], site[1], site[2]);
  }

  [[nodiscard]] std::optional<std::size_t> neighbor_site(std::size_t index, std::size_t axis,
                                                         int offset) const noexcept {
    auto site = site_coordinates(index);
    if (offset < 0) {
      if (site[axis] == 0) {
        return std::nullopt;
      }
      --site[axis];
    } else {
      if (site[axis] + 1 >= dimensions_[axis]) {
        return std::nullopt;
      }
      ++site[axis];
    }
    return site_index(site[0], site[1], site[2]);
  }

  [[nodiscard]] std::optional<std::size_t> neighbor_face(std::size_t index, std::size_t axis,
                                                         int offset) const noexcept {
    const auto [component, original] = face_coordinates(index);
    auto face = original;
    const auto face_dims = face_dimensions(component);
    if (offset < 0) {
      if (face[axis] == 0) {
        return std::nullopt;
      }
      --face[axis];
    } else {
      if (face[axis] + 1 >= face_dims[axis]) {
        return std::nullopt;
      }
      ++face[axis];
    }
    return face_index(component, face[0], face[1], face[2]);
  }

 private:
  std::array<std::uint32_t, 3> dimensions_{};
  std::array<float, 3> spacing_{};
  std::array<std::size_t, 3> face_counts_{};
  std::array<std::size_t, 3> face_offsets_{};
  std::size_t total_face_count_{0};
};

[[nodiscard]] inline float harmonic_mean(float first, float second) noexcept {
  const auto sum = first + second;
  return sum > 0.0F ? 2.0F * first * second / sum : 0.0F;
}

// The pressure held just outside each open exterior face of a grid, indexed as
// the grid's faces are. A face that is not open is closed to flow.
struct FlowBoundary {
  explicit FlowBoundary(std::size_t face_count) : open(face_count, 0), pressure(face_count, 0.0F) {
  }

  std::vector<std::uint8_t> open;
  std::vector<float> pressure;
};

// Opens both ends of one axis: the lower at pressure one and the upper at zero.
[[nodiscard]] inline FlowBoundary axis_flow_boundary(const FlowGridLayout& layout,
                                                     std::size_t axis) {
  FlowBoundary boundary(layout.total_face_count());
  const auto begin = layout.face_offsets()[axis];
  for (std::size_t index = begin; index < begin + layout.face_count(axis); ++index) {
    const auto face = layout.face_coordinates(index).second;
    if (face[axis] == 0) {
      boundary.open[index] = 1;
      boundary.pressure[index] = 1.0F;
    } else if (face[axis] == layout.dimensions()[axis]) {
      boundary.open[index] = 1;
    }
  }
  return boundary;
}

// Where a port's face lies: the axis it is normal to, whether it is that
// axis's upper face, and its two in-plane axes in x, y, z order.
struct PortFaceGeometry {
  std::size_t component{0};
  bool upper{false};
  std::array<std::size_t, 2> in_plane{};
};

[[nodiscard]] inline PortFaceGeometry port_face_geometry(GridFace face) {
  const auto index = static_cast<std::size_t>(face);
  const auto component = index / 2;
  if (component > 2) {
    throw std::invalid_argument("unknown flow port face");
  }
  constexpr std::array<std::array<std::size_t, 2>, 3> in_plane{{{1, 2}, {0, 2}, {0, 1}}};
  return {.component = component, .upper = index % 2 == 1, .in_plane = in_plane[component]};
}

// Calls body(face index, adjacent site index) for every exterior face of a
// grid that lies in a port's patch.
template <typename Body>
void for_each_port_face(const FlowGridLayout& layout, const FlowPort& port, const Body& body) {
  const auto geometry = port_face_geometry(port.face);
  std::array<std::uint32_t, 3> face{};
  face[geometry.component] = geometry.upper ? layout.dimensions()[geometry.component] : 0;
  for (auto first = port.patch.begin[0]; first < port.patch.end[0]; ++first) {
    for (auto second = port.patch.begin[1]; second < port.patch.end[1]; ++second) {
      face[geometry.in_plane[0]] = first;
      face[geometry.in_plane[1]] = second;
      const auto site = layout.adjacent_site(geometry.component, face, geometry.upper ? -1 : 1);
      body(layout.face_index(geometry.component, face[0], face[1], face[2]), *site);
    }
  }
}

// Holds each port's patch at that port's pressure.
[[nodiscard]] inline FlowBoundary port_flow_boundary(const FlowGridLayout& layout,
                                                     std::span<const FlowPort> ports,
                                                     std::span<const float> pressures) {
  if (ports.size() != pressures.size()) {
    throw std::logic_error("flow needs one pressure per port");
  }
  FlowBoundary boundary(layout.total_face_count());
  for (std::size_t port = 0; port < ports.size(); ++port) {
    for_each_port_face(layout, ports[port], [&](std::size_t face, std::size_t) {
      boundary.open[face] = 1;
      boundary.pressure[face] = pressures[port];
    });
  }
  return boundary;
}

// One pressure unknown per x/y column. The 2D solver returns depth-integrated
// flux q; lifting q to voxel faces is a conservative transport reconstruction.
class ShallowFlowReduction {
 public:
  ShallowFlowReduction(const SignalGridSpec& spec, std::span<const float> mobility)
      : original_(spec), original_layout_(spec), grid_(spec), reduced_layout_(spec) {
    spec.validate();
    if (!mobility.empty() && mobility.size() != spec.site_count()) {
      throw std::invalid_argument("flow mobility must hold one value per grid site");
    }
    grid_.shape.z = 1;
    grid_.velocity_field.reset();
    grid_.reaction.reset();
    grid_.z_lower = {};
    grid_.z_upper = {};
    // The grid of columns keeps each side's boundary kind, which says where a
    // port may open. Reservoir patches index the original grid's depth and
    // play no part in the flow.
    for (auto* boundary : {&grid_.x_lower, &grid_.x_upper, &grid_.y_lower, &grid_.y_upper}) {
      boundary->patches.clear();
    }
    const auto columns = grid_.site_count();
    grid_.obstacles.assign(columns, 1);
    heights_.assign(columns, 0.0F);
    conductance_.assign(columns, 0.0F);
    std::uint32_t common_bottom = spec.shape.z;
    float max_height = 0.0F;
    for (std::uint32_t x = 0; x < spec.shape.x; ++x) {
      for (std::uint32_t y = 0; y < spec.shape.y; ++y) {
        const auto column = static_cast<std::size_t>(x) * spec.shape.y + y;
        bool started = false, ended = false;
        float value = 1.0F;
        for (std::uint32_t z = 0; z < spec.shape.z; ++z) {
          const auto site = original_layout_.site_index(x, y, z);
          if (!mobility.empty() && (!std::isfinite(mobility[site]) || mobility[site] < 0.0F)) {
            throw std::invalid_argument("flow mobility must be finite and non-negative");
          }
          if (spec.solid_site(site)) {
            if (started) ended = true;
            continue;
          }
          if (ended)
            throw std::invalid_argument(
                "shallow flow needs contiguous fluid columns; use resolved flow");
          const float next = mobility.empty() ? 1.0F : mobility[site];
          if (!started) {
            if (common_bottom == spec.shape.z) common_bottom = z;
            if (z != common_bottom)
              throw std::invalid_argument(
                  "shallow flow needs a common planar floor; use resolved flow");
            value = next;
          } else if (std::abs(next - value) > 2.0e-6F * std::max(std::abs(value), std::abs(next))) {
            throw std::invalid_argument(
                "shallow-flow mobility must be constant through each depth column");
          }
          started = true;
          heights_[column] += spec.spacing.z;
        }
        if (started) {
          grid_.obstacles[column] = 0;
          conductance_[column] = heights_[column] * value;
          max_height = std::max(max_height, heights_[column]);
        }
      }
    }
    bottom_ = common_bottom;
    if (mobility.empty() && max_height > 0.0F) {
      for (std::size_t i = 0; i < columns; ++i) {
        const auto relative_height = heights_[i] / max_height;
        conductance_[i] *= relative_height * relative_height;
      }
    }
    for (std::size_t i = 0; i < columns; ++i)
      if (conductance_[i] == 0.0F) grid_.obstacles[i] = 1;
    reduced_layout_ = FlowGridLayout(grid_);
  }

  // The grid of columns: the original's x and y with a single z layer, solid
  // wherever a column carries no flow.
  const SignalGridSpec& grid() const {
    return grid_;
  }
  const FlowGridLayout& original_layout() const {
    return original_layout_;
  }
  const FlowGridLayout& reduced_layout() const {
    return reduced_layout_;
  }
  const std::vector<float>& conductance() const {
    return conductance_;
  }

  // A port of the original grid as a port of the grid of columns. A port must
  // open an x or y face and span every fluid site of each column it reaches.
  FlowPort reduced_port(const FlowPort& port) const {
    const auto geometry = port_face_geometry(port.face);
    if (geometry.component == 2) {
      throw std::invalid_argument("a shallow-flow port cannot open a z face; use resolved flow");
    }
    std::array<std::uint32_t, 3> site{};
    if (geometry.upper) {
      site[geometry.component] = original_layout_.dimensions()[geometry.component] - 1;
    }
    for (auto along = port.patch.begin[0]; along < port.patch.end[0]; ++along) {
      site[geometry.in_plane[0]] = along;
      for (std::uint32_t z = bottom_; z < original_.shape.z; ++z) {
        if (original_.solid_site(original_layout_.site_index(site[0], site[1], z))) break;
        if (z < port.patch.begin[1] || z >= port.patch.end[1]) {
          throw std::invalid_argument(
              "a shallow-flow port must span the full depth of its columns; use resolved flow");
        }
      }
    }
    auto reduced = port;
    reduced.patch.begin[1] = 0;
    reduced.patch.end[1] = 1;
    return reduced;
  }

  std::vector<FlowPort> reduced_ports(std::span<const FlowPort> ports) const {
    std::vector<FlowPort> reduced;
    reduced.reserve(ports.size());
    for (const auto& port : ports) {
      reduced.push_back(reduced_port(port));
    }
    return reduced;
  }

  // The faces of the original grid through which a port carries flow: those
  // of its patch that open onto a column with conductance.
  std::vector<std::size_t> open_port_faces(const FlowPort& port) const {
    std::vector<std::size_t> faces;
    for_each_port_face(original_layout_, port, [&](std::size_t face, std::size_t site) {
      if (!original_.solid_site(site) && conductance_[site / original_.shape.z] > 0.0F) {
        faces.push_back(face);
      }
    });
    return faces;
  }

  // Integrate the imposed pressure-gradient-equivalent body force across each
  // half cell in series. Only open faces carry this part of the flux.
  std::vector<float> body_flux(std::span<const std::array<float, 2>> force,
                               const FlowBoundary& boundary) const {
    if (force.size() != conductance_.size()) {
      throw std::invalid_argument("shallow flow needs one in-plane body force per column");
    }
    for (const auto& f : force) {
      if (!std::isfinite(f[0]) || !std::isfinite(f[1])) {
        throw std::invalid_argument("shallow-flow body force must be finite");
      }
    }
    std::vector<float> flux(reduced_layout_.total_face_count());
    for (std::size_t i = 0; i < flux.size(); ++i) {
      const auto [axis, face] = reduced_layout_.face_coordinates(i);
      if (axis == 2) continue;
      const auto lower = reduced_layout_.adjacent_site(axis, face, -1);
      const auto upper = reduced_layout_.adjacent_site(axis, face, 1);
      if (lower && upper) {
        flux[i] = harmonic_mean(conductance_[*lower], conductance_[*upper]) *
                  (0.5F * force[*lower][axis] + 0.5F * force[*upper][axis]);
      } else if (boundary.open[i]) {
        const auto site = lower ? *lower : *upper;
        flux[i] = conductance_[site] * force[site][axis];
      }
    }
    return flux;
  }

  std::vector<float> lift(std::span<const float> flux) const {
    if (flux.size() != reduced_layout_.total_face_count())
      throw std::logic_error("shallow flux size mismatch");
    std::vector<float> velocity(original_layout_.total_face_count(), 0.0F);
    for (std::size_t index = 0; index < velocity.size(); ++index) {
      const auto [component, face] = original_layout_.face_coordinates(index);
      if (component == 2) continue;
      const auto lower = original_layout_.adjacent_site(component, face, -1);
      const auto upper = original_layout_.adjacent_site(component, face, 1);
      if ((lower && original_.solid_site(*lower)) || (upper && original_.solid_site(*upper)))
        continue;
      // An exterior face has one column; a closed one carries no flux.
      const auto low_column = lower ? *lower / original_.shape.z : *upper / original_.shape.z;
      const auto high_column = upper ? *upper / original_.shape.z : low_column;
      const auto height = std::min(heights_[low_column], heights_[high_column]);
      if (height > 0.0F)
        velocity[index] = flux[reduced_layout_.face_index(component, face[0], face[1], 0)] / height;
    }
    // Integrate horizontal divergence upward, spreading the small column
    // pressure residual uniformly over depth so both floor and roof stay closed.
    for (std::uint32_t x = 0; x < original_.shape.x; ++x) {
      for (std::uint32_t y = 0; y < original_.shape.y; ++y) {
        std::vector<double> horizontal;
        double sum = 0.0;
        for (std::uint32_t z = bottom_; z < original_.shape.z; ++z) {
          if (original_.solid_site(original_layout_.site_index(x, y, z))) break;
          const double dx = (velocity[original_layout_.face_index(0, x + 1, y, z)] -
                             velocity[original_layout_.face_index(0, x, y, z)]) /
                            original_.spacing.x;
          const double dy = (velocity[original_layout_.face_index(1, x, y + 1, z)] -
                             velocity[original_layout_.face_index(1, x, y, z)]) /
                            original_.spacing.y;
          horizontal.push_back(dx + dy);
          sum += dx + dy;
        }
        if (horizontal.empty()) continue;
        const double mean = sum / static_cast<double>(horizontal.size());
        double vertical = 0.0;
        for (std::size_t k = 0; k + 1 < horizontal.size(); ++k) {
          vertical -= original_.spacing.z * (horizontal[k] - mean);
          velocity[original_layout_.face_index(
              2, x, y, bottom_ + static_cast<std::uint32_t>(k) + 1)] = static_cast<float>(vertical);
        }
      }
    }
    return velocity;
  }

  // The faces of the original grid's lower `axis` face that carry flow.
  std::vector<std::uint8_t> open_inlet_faces(std::size_t axis) const {
    std::vector<std::uint8_t> result(original_layout_.total_face_count(), 0);
    for (std::size_t index = 0; index < result.size(); ++index) {
      const auto [component, face] = original_layout_.face_coordinates(index);
      if (component != axis || face[component] != 0) continue;
      const auto site = original_layout_.adjacent_site(component, face, 1);
      if (site && !original_.solid_site(*site) && conductance_[*site / original_.shape.z] > 0.0F)
        result[index] = 1;
    }
    return result;
  }

 private:
  const SignalGridSpec& original_;
  FlowGridLayout original_layout_;
  SignalGridSpec grid_;
  FlowGridLayout reduced_layout_;
  std::vector<float> heights_, conductance_;
  std::uint32_t bottom_{0};
};

// The pressure equation of depth-averaged flow: conductance-weighted
// differences between neighboring sites, with the pressure of a FlowBoundary
// held half a spacing outside each open exterior face.
// A depth-averaged pressure system over its active sites alone: those whose
// pressure the system determines. A device that is mostly wall has few of
// them, and a solve over this form costs what they cost.
struct ActiveDepthSystem {
  // Marks a neighbor that is off the grid or carries no conductance.
  static constexpr std::uint32_t no_neighbor = std::numeric_limits<std::uint32_t>::max();

  // The active sites, ascending. Every other array is in their order.
  std::vector<std::uint32_t> sites;
  // Six per active site, lower then upper along x, y and z: the neighbor's
  // place among the active sites, and the harmonic mean of the two
  // mobilities, which the operator scales by the inverse square spacing.
  std::vector<std::uint32_t> neighbors;
  std::vector<float> conductance;
  std::vector<float> diagonal;
  std::vector<float> right_hand_side;
  std::vector<float> boundary_conductance;

  [[nodiscard]] std::size_t size() const noexcept {
    return sites.size();
  }
};

class DepthAveragedFlowSystem {
 public:
  DepthAveragedFlowSystem(const SignalGridSpec& spec, std::span<const float> mobility,
                          const FlowBoundary& boundary, std::span<const float> body_flux = {})
      : spec_(spec),
        layout_(spec),
        mobility_(layout_.site_count(), 1.0F),
        diagonal_(layout_.site_count()),
        right_hand_side_(layout_.site_count()),
        boundary_conductance_(layout_.site_count()),
        face_conductance_(layout_.total_face_count()),
        face_pressure_(layout_.total_face_count()) {
    if (boundary.open.size() != layout_.total_face_count() ||
        boundary.pressure.size() != layout_.total_face_count()) {
      throw std::logic_error("flow boundary does not match the grid's faces");
    }
    if (!body_flux.empty() && body_flux.size() != layout_.total_face_count()) {
      throw std::logic_error("body flux does not match the grid's faces");
    }
    if (!mobility.empty()) {
      if (mobility.size() != layout_.site_count()) {
        throw std::invalid_argument("flow mobility must hold one value per grid site");
      }
      std::copy(mobility.begin(), mobility.end(), mobility_.begin());
    }
    for (std::size_t site = 0; site < mobility_.size(); ++site) {
      if (!std::isfinite(mobility_[site]) || mobility_[site] < 0.0F) {
        throw std::invalid_argument("flow mobility values must be finite and non-negative");
      }
      if (spec_.solid_site(site)) {
        mobility_[site] = 0.0F;
      }
    }

    bool open = false;
    for (std::size_t site = 0; site < layout_.site_count(); ++site) {
      const auto value = mobility_[site];
      if (value == 0.0F) {
        continue;
      }
      const auto coordinates = layout_.site_coordinates(site);
      for (std::size_t component = 0; component < 3; ++component) {
        const auto spacing = layout_.spacing()[component];
        const auto inverse_square = 1.0F / (spacing * spacing);
        if (!body_flux.empty()) {
          auto upper = coordinates;
          ++upper[component];
          const auto lo =
              layout_.face_index(component, coordinates[0], coordinates[1], coordinates[2]);
          const auto hi = layout_.face_index(component, upper[0], upper[1], upper[2]);
          right_hand_side_[site] += (body_flux[lo] - body_flux[hi]) / spacing;
        }
        for (const auto offset : {-1, 1}) {
          const auto neighbor = layout_.neighbor_site(site, component, offset);
          if (neighbor.has_value()) {
            diagonal_[site] += harmonic_mean(value, mobility_[*neighbor]) * inverse_square;
            continue;
          }
          auto face_coordinates = coordinates;
          if (offset > 0) {
            ++face_coordinates[component];
          }
          const auto face = layout_.face_index(component, face_coordinates[0], face_coordinates[1],
                                               face_coordinates[2]);
          if (boundary.open[face] == 0) {
            continue;
          }
          // The boundary pressure sits half a spacing from the site's center.
          const auto conductance = 2.0F * value * inverse_square;
          diagonal_[site] += conductance;
          boundary_conductance_[site] += conductance;
          right_hand_side_[site] += conductance * boundary.pressure[face];
          // Positive velocity runs along the axis: into the grid through a
          // lower face and out of it through an upper one.
          face_conductance_[face] = (offset < 0 ? 2.0F : -2.0F) * value / spacing;
          face_pressure_[face] = boundary.pressure[face];
          open = true;
        }
      }
    }
    if (!open) {
      throw std::invalid_argument("the flow boundary opens onto no fluid");
    }

    std::vector<std::uint32_t> place(layout_.site_count(), ActiveDepthSystem::no_neighbor);
    for (std::size_t site = 0; site < layout_.site_count(); ++site) {
      if (diagonal_[site] != 0.0F) {
        place[site] = static_cast<std::uint32_t>(active_.sites.size());
        active_.sites.push_back(static_cast<std::uint32_t>(site));
      }
    }
    const auto count = active_.sites.size();
    active_.neighbors.assign(6 * count, ActiveDepthSystem::no_neighbor);
    active_.conductance.assign(6 * count, 0.0F);
    active_.diagonal.resize(count);
    active_.right_hand_side.resize(count);
    active_.boundary_conductance.resize(count);
    parallel_for(count, flow_grain, [&](std::size_t begin, std::size_t end) {
      for (std::size_t index = begin; index < end; ++index) {
        const std::size_t site = active_.sites[index];
        active_.diagonal[index] = diagonal_[site];
        active_.right_hand_side[index] = right_hand_side_[site];
        active_.boundary_conductance[index] = boundary_conductance_[site];
        for (std::size_t component = 0; component < 3; ++component) {
          for (const auto offset : {-1, 1}) {
            const auto neighbor = layout_.neighbor_site(site, component, offset);
            if (!neighbor.has_value() || place[*neighbor] == ActiveDepthSystem::no_neighbor) {
              continue;
            }
            const auto slot = (6 * index) + (2 * component) + (offset < 0 ? 0 : 1);
            active_.neighbors[slot] = place[*neighbor];
            active_.conductance[slot] = harmonic_mean(mobility_[site], mobility_[*neighbor]);
          }
        }
      }
    });
  }

  [[nodiscard]] const FlowGridLayout& layout() const noexcept {
    return layout_;
  }
  // The system over its active sites, which is the form the solvers iterate.
  [[nodiscard]] const ActiveDepthSystem& active() const noexcept {
    return active_;
  }
  [[nodiscard]] const std::vector<float>& mobility() const noexcept {
    return mobility_;
  }
  [[nodiscard]] const std::vector<float>& diagonal() const noexcept {
    return diagonal_;
  }
  [[nodiscard]] const std::vector<float>& right_hand_side() const noexcept {
    return right_hand_side_;
  }
  // Per site, the summed conductance to the boundary pressures beyond its
  // open exterior faces.
  [[nodiscard]] const std::vector<float>& boundary_conductance() const noexcept {
    return boundary_conductance_;
  }
  // Per face, what turns the pressure difference across an open exterior face
  // into its velocity, velocity = conductance * (boundary pressure - site
  // pressure), and the boundary pressure itself. Both are zero elsewhere.
  [[nodiscard]] const std::vector<float>& face_conductance() const noexcept {
    return face_conductance_;
  }
  [[nodiscard]] const std::vector<float>& face_pressure() const noexcept {
    return face_pressure_;
  }

  // The operator on a vector over the active sites.
  void apply(std::span<const double> input, std::vector<double>& output) const {
    if (input.size() != active_.size()) {
      throw std::invalid_argument("depth-averaged flow vector has the wrong size");
    }
    output.resize(input.size());
    std::array<double, 3> inverse_square{};
    for (std::size_t component = 0; component < 3; ++component) {
      inverse_square[component] =
          1.0 / (static_cast<double>(layout_.spacing()[component]) * layout_.spacing()[component]);
    }
    parallel_for(input.size(), flow_grain, [&](std::size_t begin, std::size_t end) {
      for (std::size_t index = begin; index < end; ++index) {
        double result = static_cast<double>(active_.boundary_conductance[index]) * input[index];
        for (std::size_t slot = 0; slot < 6; ++slot) {
          const auto neighbor = active_.neighbors[(6 * index) + slot];
          if (neighbor == ActiveDepthSystem::no_neighbor) {
            continue;
          }
          const auto conductance = static_cast<double>(active_.conductance[(6 * index) + slot]) *
                                   inverse_square[slot / 2];
          result += conductance * (input[index] - input[neighbor]);
        }
        output[index] = result;
      }
    });
  }

  // The pressure at every site of the grid, from the pressures of the active
  // sites; it is zero elsewhere.
  [[nodiscard]] std::vector<double> scatter(std::span<const double> pressure) const {
    if (pressure.size() != active_.size()) {
      throw std::invalid_argument("depth-averaged pressure vector has the wrong size");
    }
    std::vector<double> result(layout_.site_count(), 0.0);
    for (std::size_t index = 0; index < pressure.size(); ++index) {
      result[active_.sites[index]] = pressure[index];
    }
    return result;
  }

  // The velocity on every face of the grid, from the pressure at every site.
  [[nodiscard]] std::vector<float> velocity(std::span<const double> pressure) const {
    if (pressure.size() != layout_.site_count()) {
      throw std::invalid_argument("depth-averaged pressure vector has the wrong size");
    }
    std::vector<float> result(layout_.total_face_count(), 0.0F);
    parallel_for(result.size(), flow_grain, [&](std::size_t begin, std::size_t end) {
      for (std::size_t face_index = begin; face_index < end; ++face_index) {
        const auto [component, face] = layout_.face_coordinates(face_index);
        const auto lower = layout_.adjacent_site(component, face, -1);
        const auto upper = layout_.adjacent_site(component, face, 1);
        const auto spacing = static_cast<double>(layout_.spacing()[component]);
        double value = 0.0;
        if (lower.has_value() && upper.has_value()) {
          const auto face_mobility = harmonic_mean(mobility_[*lower], mobility_[*upper]);
          value =
              -static_cast<double>(face_mobility) * (pressure[*upper] - pressure[*lower]) / spacing;
        } else {
          const auto site = upper.has_value() ? *upper : *lower;
          value = static_cast<double>(face_conductance_[face_index]) *
                  (static_cast<double>(face_pressure_[face_index]) - pressure[site]);
        }
        result[face_index] = static_cast<float>(value);
      }
    });
    return result;
  }

 private:
  const SignalGridSpec& spec_;
  FlowGridLayout layout_;
  ActiveDepthSystem active_;
  std::vector<float> mobility_;
  std::vector<float> diagonal_;
  std::vector<float> right_hand_side_;
  std::vector<float> boundary_conductance_;
  std::vector<float> face_conductance_;
  std::vector<float> face_pressure_;
};

// The Stokes-Brinkman block system on a staggered grid: one velocity per face
// and one pressure per site. An exterior face is an unknown where the system's
// FlowBoundary opens it onto fluid, with the boundary's pressure held one
// spacing beyond the adjacent site's center; every other exterior face is a
// wall.
class ResolvedFlowSystem {
 public:
  // Flow along one axis, from its lower face to its upper. The system solves
  // for the pressure correction about the known linear inlet-to-outlet
  // profile. This algebraic shift avoids subtracting nearly equal O(1)
  // pressures to recover a small viscous forcing in binary32.
  ResolvedFlowSystem(const SignalGridSpec& spec, std::span<const float> drag, FlowAxis axis)
      : ResolvedFlowSystem(
            checked(spec, axis), drag,
            axis_flow_boundary(FlowGridLayout(spec), static_cast<std::size_t>(axis))) {
    flow_axis_ = static_cast<std::size_t>(axis);
    const auto component = *flow_axis_;
    const auto force = 1.0F / ((static_cast<float>(layout_.dimensions()[component]) + 1.0F) *
                               layout_.spacing()[component]);
    bool open_inlet = false;
    std::fill(force_.begin(), force_.end(), 0.0F);
    const auto begin = layout_.face_offsets()[component];
    for (std::size_t index = begin; index < begin + layout_.face_count(component); ++index) {
      if (active_[index] == 0) {
        continue;
      }
      force_[index] = force;
      open_inlet = open_inlet || layout_.face_coordinates(index).second[component] == 0;
    }
    if (!open_inlet) {
      throw std::invalid_argument("the resolved-flow inlet boundary is entirely blocked");
    }
  }

  // Flow driven by the pressures a FlowBoundary holds beyond the exterior
  // faces it opens. `known_pressure`, when given, holds one pressure per site
  // that is close to the answer; the system then solves for the correction to
  // it, which converges in fewer iterations and, as for the single-axis
  // system, keeps the differences of nearly equal pressures out of binary32.
  ResolvedFlowSystem(const SignalGridSpec& spec, std::span<const float> drag,
                     const FlowBoundary& boundary, std::span<const double> known_pressure = {})
      : spec_(spec),
        layout_(spec),
        fluid_(layout_.site_count(), 1),
        drag_(layout_.site_count()),
        active_(layout_.total_face_count()),
        exists_(layout_.total_face_count()),
        face_drag_(layout_.total_face_count()),
        diagonal_(layout_.total_face_count()),
        force_(layout_.total_face_count()) {
    if (boundary.open.size() != layout_.total_face_count() ||
        boundary.pressure.size() != layout_.total_face_count()) {
      throw std::logic_error("flow boundary does not match the grid's faces");
    }
    if (!known_pressure.empty() && known_pressure.size() != layout_.site_count()) {
      throw std::logic_error("a known pressure must hold one value per grid site");
    }
    if (!drag.empty()) {
      if (drag.size() != layout_.site_count()) {
        throw std::invalid_argument("resolved-flow drag must hold one value per grid site");
      }
      std::copy(drag.begin(), drag.end(), drag_.begin());
    }
    for (std::size_t site = 0; site < layout_.site_count(); ++site) {
      fluid_[site] = spec_.solid_site(site) ? 0 : 1;
      if (!std::isfinite(drag_[site]) || drag_[site] < 0.0F) {
        throw std::invalid_argument("resolved-flow drag values must be finite and non-negative");
      }
      if (fluid_[site] == 0) {
        drag_[site] = 0.0F;
      }
    }

    for (std::size_t index = 0; index < layout_.total_face_count(); ++index) {
      const auto [component, face] = layout_.face_coordinates(index);
      const auto lower = layout_.adjacent_site(component, face, -1);
      const auto upper = layout_.adjacent_site(component, face, 1);
      const auto lower_fluid = lower.has_value() && fluid_[*lower] != 0;
      const auto upper_fluid = upper.has_value() && fluid_[*upper] != 0;
      exists_[index] = lower_fluid || upper_fluid ? 1 : 0;
      const auto exterior = !lower.has_value() || !upper.has_value();
      const auto active = exterior ? boundary.open[index] != 0 && (lower_fluid || upper_fluid)
                                   : lower_fluid && upper_fluid;
      active_[index] = active ? 1 : 0;
      if (!active) {
        continue;
      }
      float sum = 0.0F;
      float count = 0.0F;
      if (lower_fluid) {
        sum += drag_[*lower];
        count += 1.0F;
      }
      if (upper_fluid) {
        sum += drag_[*upper];
        count += 1.0F;
      }
      face_drag_[index] = sum / count;
      diagonal_[index] = face_drag_[index];
      for (std::size_t axis_index = 0; axis_index < 3; ++axis_index) {
        if (layout_.dimensions()[axis_index] > 1) {
          diagonal_[index] +=
              2.0F / (layout_.spacing()[axis_index] * layout_.spacing()[axis_index]);
        }
      }
      // The system's pressure gradient takes a missing side as zero. The
      // forcing is what remains: the gradient of the known pressure, with the
      // boundary's pressure standing on the missing side of an exterior face.
      const auto known = [&](const std::optional<std::size_t>& site) {
        if (!site.has_value()) {
          return static_cast<double>(boundary.pressure[index]);
        }
        return known_pressure.empty() ? 0.0 : known_pressure[*site];
      };
      force_[index] = static_cast<float>(-(known(upper) - known(lower)) /
                                         static_cast<double>(layout_.spacing()[component]));
    }
  }

  [[nodiscard]] const FlowGridLayout& layout() const noexcept {
    return layout_;
  }
  [[nodiscard]] const std::vector<std::uint8_t>& fluid() const noexcept {
    return fluid_;
  }
  [[nodiscard]] const std::vector<std::uint8_t>& active() const noexcept {
    return active_;
  }
  [[nodiscard]] const std::vector<std::uint8_t>& exists() const noexcept {
    return exists_;
  }
  [[nodiscard]] const std::vector<float>& face_drag() const noexcept {
    return face_drag_;
  }
  [[nodiscard]] const std::vector<float>& diagonal() const noexcept {
    return diagonal_;
  }
  [[nodiscard]] const std::vector<float>& force() const noexcept {
    return force_;
  }

  void apply_momentum(std::span<const double> input, std::vector<double>& output) const {
    if (input.size() != layout_.total_face_count()) {
      throw std::invalid_argument("resolved-flow face vector has the wrong size");
    }
    output.assign(input.size(), 0.0);
    parallel_for(input.size(), flow_grain, [&](std::size_t begin, std::size_t end) {
      for (std::size_t index = begin; index < end; ++index) {
        if (active_[index] == 0) {
          continue;
        }
        const auto component = layout_.face_coordinates(index).first;
        auto result = static_cast<double>(face_drag_[index]) * input[index];
        for (std::size_t axis = 0; axis < 3; ++axis) {
          if (layout_.dimensions()[axis] == 1) {
            continue;
          }
          const auto inverse_square =
              1.0 / (static_cast<double>(layout_.spacing()[axis]) * layout_.spacing()[axis]);
          for (const auto offset : {-1, 1}) {
            const auto neighbor_index = layout_.neighbor_face(index, axis, offset);
            auto neighbor = 0.0;
            if (axis == component) {
              neighbor = neighbor_index.has_value() ? input[*neighbor_index] : input[index];
            } else if (neighbor_index.has_value() && exists_[*neighbor_index] != 0) {
              neighbor = input[*neighbor_index];
            } else {
              neighbor = -input[index];
            }
            result -= (neighbor - input[index]) * inverse_square;
          }
        }
        output[index] = result;
      }
    });
  }

  [[nodiscard]] std::vector<double> gradient(std::span<const double> pressure) const {
    if (pressure.size() != layout_.site_count()) {
      throw std::invalid_argument("resolved-flow pressure vector has the wrong size");
    }
    std::vector<double> result(layout_.total_face_count(), 0.0);
    parallel_for(result.size(), flow_grain, [&](std::size_t begin, std::size_t end) {
      for (std::size_t index = begin; index < end; ++index) {
        if (active_[index] == 0) {
          continue;
        }
        const auto [component, face] = layout_.face_coordinates(index);
        const auto lower = layout_.adjacent_site(component, face, -1);
        const auto upper = layout_.adjacent_site(component, face, 1);
        const auto lower_value = lower.has_value() && fluid_[*lower] != 0 ? pressure[*lower] : 0.0;
        const auto upper_value = upper.has_value() && fluid_[*upper] != 0 ? pressure[*upper] : 0.0;
        result[index] = (upper_value - lower_value) / layout_.spacing()[component];
      }
    });
    return result;
  }

  [[nodiscard]] std::vector<double> divergence(std::span<const double> velocity) const {
    if (velocity.size() != layout_.total_face_count()) {
      throw std::invalid_argument("resolved-flow velocity vector has the wrong size");
    }
    std::vector<double> result(layout_.site_count(), 0.0);
    parallel_for(result.size(), flow_grain, [&](std::size_t begin, std::size_t end) {
      for (std::size_t site = begin; site < end; ++site) {
        if (fluid_[site] == 0) {
          continue;
        }
        const auto coordinates = layout_.site_coordinates(site);
        for (std::size_t component = 0; component < 3; ++component) {
          auto upper = coordinates;
          ++upper[component];
          const auto upper_face = layout_.face_index(component, upper[0], upper[1], upper[2]);
          const auto lower_face =
              layout_.face_index(component, coordinates[0], coordinates[1], coordinates[2]);
          result[site] +=
              (velocity[upper_face] - velocity[lower_face]) / layout_.spacing()[component];
        }
      }
    });
    return result;
  }

  [[nodiscard]] std::vector<float> pressure_diagonal() const {
    std::vector<float> result(layout_.site_count(), 0.0F);
    for (std::size_t site = 0; site < result.size(); ++site) {
      if (fluid_[site] != 0) {
        const auto coordinates = layout_.site_coordinates(site);
        for (std::size_t axis = 0; axis < 3; ++axis) {
          auto upper = coordinates;
          ++upper[axis];
          for (const auto face :
               {layout_.face_index(axis, coordinates[0], coordinates[1], coordinates[2]),
                layout_.face_index(axis, upper[0], upper[1], upper[2])}) {
            if (active_[face] && diagonal_[face] > 0.0F) {
              result[site] +=
                  1.0F / (diagonal_[face] * layout_.spacing()[axis] * layout_.spacing()[axis]);
            }
          }
        }
      }
    }
    return result;
  }

  // The open faces of the lower end of the flow axis. Only a system built for
  // one axis has them.
  [[nodiscard]] std::vector<std::uint8_t> open_inlet_faces() const {
    if (!flow_axis_.has_value()) {
      throw std::logic_error("a port flow system has no inlet axis");
    }
    std::vector<std::uint8_t> result(layout_.total_face_count(), 0);
    const auto component = *flow_axis_;
    for (std::size_t index = layout_.face_offsets()[component];
         index < layout_.face_offsets()[component] + layout_.face_counts()[component]; ++index) {
      const auto [face_component, face] = layout_.face_coordinates(index);
      if (face_component == component && face[component] == 0 && active_[index] != 0) {
        result[index] = 1;
      }
    }
    return result;
  }

  // The fewest voxels across any gap between walls. A system built for one
  // axis measures across the other two. A port flow system measures along
  // every axis and passes over a run of fluid that ends at an open face,
  // which is a way through and not a gap.
  [[nodiscard]] std::uint32_t minimum_gap_voxels() const {
    std::uint32_t shortest = 0;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      if ((flow_axis_.has_value() && axis == *flow_axis_) || layout_.dimensions()[axis] <= 1) {
        continue;
      }
      const auto first_axis = (axis + 1) % 3;
      const auto second_axis = (axis + 2) % 3;
      for (std::uint32_t first = 0; first < layout_.dimensions()[first_axis]; ++first) {
        for (std::uint32_t second = 0; second < layout_.dimensions()[second_axis]; ++second) {
          std::array<std::uint32_t, 3> coordinates{};
          coordinates[first_axis] = first;
          coordinates[second_axis] = second;
          const auto open_face = [&](std::uint32_t along) {
            auto face = coordinates;
            face[axis] = along;
            return active_[layout_.face_index(axis, face[0], face[1], face[2])] != 0;
          };
          std::uint32_t run = 0;
          bool through = false;
          const auto close = [&] {
            if (run != 0 && !through) {
              shortest = shortest == 0 ? run : std::min(shortest, run);
            }
            run = 0;
            through = false;
          };
          for (std::uint32_t along = 0; along < layout_.dimensions()[axis]; ++along) {
            coordinates[axis] = along;
            if (fluid_[layout_.site_index(coordinates[0], coordinates[1], coordinates[2])] != 0) {
              through = through || (along == 0 && open_face(0));
              ++run;
            } else {
              close();
            }
          }
          through = through || (run != 0 && open_face(layout_.dimensions()[axis]));
          close();
        }
      }
    }
    return shortest;
  }

 private:
  const SignalGridSpec& spec_;
  FlowGridLayout layout_;
  std::vector<std::uint8_t> fluid_;
  std::vector<float> drag_;
  std::vector<std::uint8_t> active_;
  std::vector<std::uint8_t> exists_;
  std::vector<float> face_drag_;
  std::vector<float> diagonal_;
  std::vector<float> force_;
  std::optional<std::size_t> flow_axis_;

  [[nodiscard]] static const SignalGridSpec& checked(const SignalGridSpec& spec, FlowAxis axis) {
    validate_flow_grid(spec, axis);
    return spec;
  }
};

// The three face arrays of a velocity field from one array of every face,
// checked against the grid they are for.
[[nodiscard]] inline SignalGridVelocityField make_velocity_field(const SignalGridSpec& spec,
                                                                 const FlowGridLayout& layout,
                                                                 std::span<const float> values) {
  if (values.size() != layout.total_face_count()) {
    throw std::invalid_argument("flow velocity does not match the grid's faces");
  }
  SignalGridVelocityField field;
  const auto offsets = layout.face_offsets();
  const auto counts = layout.face_counts();
  field.x_faces.assign(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(counts[0]));
  field.y_faces.assign(values.begin() + static_cast<std::ptrdiff_t>(offsets[1]),
                       values.begin() + static_cast<std::ptrdiff_t>(offsets[1] + counts[1]));
  field.z_faces.assign(values.begin() + static_cast<std::ptrdiff_t>(offsets[2]), values.end());
  auto candidate = spec;
  candidate.velocity_field = field;
  for (auto& advection : candidate.advection) {
    advection = {};
  }
  candidate.validate();
  return field;
}

// The faces through which one port carries flow, with what turns their summed
// velocities into the volume entering the grid per unit time.
struct PortFaces {
  std::vector<std::size_t> faces;
  // The area of one face, negative on an upper face, where positive velocity
  // leaves the grid.
  double signed_area{0.0};

  [[nodiscard]] double open_area() const noexcept {
    return std::abs(signed_area) * static_cast<double>(faces.size());
  }

  [[nodiscard]] double flux(std::span<const double> velocity) const {
    double sum = 0.0;
    for (const auto face : faces) {
      sum += velocity[face];
    }
    return sum * signed_area;
  }
};

[[nodiscard]] inline PortFaces make_port_faces(const FlowGridLayout& layout, const FlowPort& port,
                                               std::vector<std::size_t> open_faces) {
  const auto geometry = port_face_geometry(port.face);
  const auto area = static_cast<double>(layout.spacing()[geometry.in_plane[0]]) *
                    layout.spacing()[geometry.in_plane[1]];
  return {.faces = std::move(open_faces), .signed_area = geometry.upper ? -area : area};
}

struct PortFlowSolution {
  std::vector<float> velocity;
  std::vector<FlowPortState> ports;
  std::uint32_t solves{0};
  float max_speed{0.0F};
};

// The field that carries each speed port's stated mean speed and holds each
// pressure port at its stated pressure. `solve(pressures)` returns the field,
// over every face of the grid, that holds each port at the given pressure.
// Port fluxes are linear in those pressures, so one solve for each speed port
// held alone at pressure one gives the pressures that deliver the stated
// speeds, and the field is the same combination of the solved fields.
template <typename Solve>
[[nodiscard]] PortFlowSolution superpose_port_flows(std::span<const FlowPort> ports,
                                                    std::span<const PortFaces> faces,
                                                    Solve&& solve) {
  if (ports.size() != faces.size()) {
    throw std::logic_error("flow needs the faces of every port");
  }
  std::vector<std::size_t> driven;
  std::vector<float> held(ports.size(), 0.0F);
  bool holds_pressure = false;
  for (std::size_t port = 0; port < ports.size(); ++port) {
    if (ports[port].kind == FlowPortKind::speed) {
      driven.push_back(port);
    } else {
      held[port] = ports[port].value;
      holds_pressure = holds_pressure || ports[port].value != 0.0F;
    }
  }

  PortFlowSolution solution;
  std::vector<double> velocity;
  const auto add = [&](std::span<const float> field, double weight) {
    if (velocity.empty()) {
      velocity.assign(field.size(), 0.0);
    }
    if (field.size() != velocity.size()) {
      throw std::logic_error("port flow solves disagree on the grid's faces");
    }
    for (std::size_t face = 0; face < field.size(); ++face) {
      velocity[face] += weight * static_cast<double>(field[face]);
    }
  };
  if (holds_pressure) {
    const std::vector<float> field = solve(std::span<const float>(held));
    ++solution.solves;
    add(field, 1.0);
  }

  std::vector<float> pressures = held;
  if (!driven.empty()) {
    // response[m][j] is the flux through speed port m when speed port j alone
    // is held at pressure one.
    const auto count = driven.size();
    std::vector<std::vector<float>> fields;
    std::vector<std::vector<double>> response(count, std::vector<double>(count + 1, 0.0));
    for (std::size_t j = 0; j < count; ++j) {
      std::vector<float> unit(ports.size(), 0.0F);
      unit[driven[j]] = 1.0F;
      fields.push_back(solve(std::span<const float>(unit)));
      ++solution.solves;
      const std::vector<double> field(fields.back().begin(), fields.back().end());
      for (std::size_t m = 0; m < count; ++m) {
        response[m][j] = faces[driven[m]].flux(field);
      }
    }
    for (std::size_t m = 0; m < count; ++m) {
      const auto& port = faces[driven[m]];
      const auto base = velocity.empty() ? 0.0 : port.flux(velocity);
      response[m][count] = static_cast<double>(ports[driven[m]].value) * port.open_area() - base;
    }
    // Gaussian elimination with partial pivoting on the augmented matrix.
    for (std::size_t column = 0; column < count; ++column) {
      auto pivot = column;
      for (std::size_t row = column + 1; row < count; ++row) {
        if (std::abs(response[row][column]) > std::abs(response[pivot][column])) {
          pivot = row;
        }
      }
      if (response[pivot][column] == 0.0) {
        throw std::runtime_error("a speed port carries no flow: no pressure port is reachable");
      }
      std::swap(response[column], response[pivot]);
      for (std::size_t row = 0; row < count; ++row) {
        if (row == column) {
          continue;
        }
        const auto factor = response[row][column] / response[column][column];
        for (std::size_t entry = column; entry <= count; ++entry) {
          response[row][entry] -= factor * response[column][entry];
        }
      }
    }
    for (std::size_t j = 0; j < count; ++j) {
      const auto pressure = response[j][count] / response[j][j];
      pressures[driven[j]] = static_cast<float>(pressure);
      add(fields[j], pressure);
    }
  }

  solution.velocity.resize(velocity.size());
  for (std::size_t face = 0; face < velocity.size(); ++face) {
    solution.velocity[face] = static_cast<float>(velocity[face]);
    if (!std::isfinite(solution.velocity[face])) {
      throw std::runtime_error("flow solve produced a non-finite velocity");
    }
    solution.max_speed = std::max(solution.max_speed, std::abs(solution.velocity[face]));
  }
  for (std::size_t port = 0; port < ports.size(); ++port) {
    const auto flux = faces[port].flux(velocity);
    solution.ports.push_back({
        .pressure = pressures[port],
        .flux = static_cast<float>(flux),
        .mean_speed = static_cast<float>(flux / faces[port].open_area()),
    });
  }
  return solution;
}

// The reduction of a shallow flow along one axis, after the checks that it
// states a solvable flow before and after columns without conductance close.
[[nodiscard]] inline ShallowFlowReduction reduce_shallow_flow(const SignalGridSpec& spec,
                                                              std::span<const float> mobility,
                                                              FlowAxis axis) {
  validate_flow_grid(spec, axis);
  if (axis == FlowAxis::z) {
    throw std::invalid_argument(
        "shallow flow is depth-integrated along z; flow axis must be x or y");
  }
  ShallowFlowReduction reduction(spec, mobility);
  validate_flow_grid(reduction.grid(), axis);
  return reduction;
}

// A shallow flow driven through ports: the reduction to columns, the ports on
// the grid of columns, and the faces each port carries flow through.
class ShallowPortFlow {
 public:
  ShallowPortFlow(const SignalGridSpec& spec, std::span<const FlowPort> ports,
                  std::span<const float> mobility)
      : ports_(ports.begin(), ports.end()), reduction_(checked(spec, ports), mobility) {
    reduced_ports_ = reduction_.reduced_ports(ports_);
    validate_flow_ports(reduction_.grid(), reduced_ports_);
    for (const auto& port : ports_) {
      faces_.push_back(
          make_port_faces(reduction_.original_layout(), port, reduction_.open_port_faces(port)));
    }
  }

  [[nodiscard]] const ShallowFlowReduction& reduction() const noexcept {
    return reduction_;
  }

  // The combined field over the original grid's faces. `velocity(system)`
  // solves one pressure system and returns its velocity on every face of the
  // grid of columns.
  template <typename Velocity>
  [[nodiscard]] PortFlowSolution solve(Velocity&& velocity) const {
    return superpose_port_flows(ports_, faces_, [&](std::span<const float> pressures) {
      const DepthAveragedFlowSystem system(
          reduction_.grid(), reduction_.conductance(),
          port_flow_boundary(reduction_.reduced_layout(), reduced_ports_, pressures));
      return reduction_.lift(velocity(system));
    });
  }

 private:
  [[nodiscard]] static const SignalGridSpec& checked(const SignalGridSpec& spec,
                                                     std::span<const FlowPort> ports) {
    validate_flow_ports(spec, ports);
    return spec;
  }

  std::vector<FlowPort> ports_;
  ShallowFlowReduction reduction_;
  std::vector<FlowPort> reduced_ports_;
  std::vector<PortFaces> faces_;
};

// The pressures of one depth-averaged pressure system, solved on the host in
// double precision.
[[nodiscard]] std::vector<double> solve_depth_averaged_pressure(
    const DepthAveragedFlowSystem& system, float tolerance, std::uint32_t max_iterations);

// The depth-averaged pressure that holds each port at the given pressure, at
// every site of the grid. It is empty for a geometry that shallow flow does
// not describe: columns that are not contiguous above one floor, or a port in
// a z face or short of its columns' depth.
[[nodiscard]] inline std::vector<double> shallow_port_pressure(const SignalGridSpec& spec,
                                                               std::span<const FlowPort> ports,
                                                               std::span<const float> pressures) {
  try {
    const ShallowFlowReduction reduction(spec, {});
    const auto reduced_ports = reduction.reduced_ports(ports);
    const DepthAveragedFlowSystem system(
        reduction.grid(), reduction.conductance(),
        port_flow_boundary(reduction.reduced_layout(), reduced_ports, pressures));
    const auto columns = solve_depth_averaged_pressure(system, 1.0e-6F, 50'000);
    std::vector<double> lifted(spec.site_count());
    for (std::size_t site = 0; site < lifted.size(); ++site) {
      lifted[site] = columns[site / spec.shape.z];
    }
    return lifted;
  } catch (const std::invalid_argument&) {
    return {};
  }
}

// How far one resolved block solve iterates.
struct ResolvedSolveControls {
  float relative_tolerance{1.0e-6F};
  std::uint32_t max_outer_iterations{500};
  float inner_relative_tolerance{1.0e-6F};
  std::uint32_t max_inner_iterations{50'000};
};

// One solved resolved system: its velocity on every face, as solved and
// before any scaling, and how the solve went.
struct ResolvedSystemSolve {
  std::vector<float> velocity;
  std::uint32_t outer_iterations{0};
  std::uint64_t inner_iterations{0};
  float relative_residual{0.0F};
  float momentum_relative_residual{0.0F};
  double divergence_rms{0.0};
};

struct ScaledVelocity {
  SignalGridVelocityField field;
  float solved_mean{0.0F};
  float max_speed{0.0F};
  float factor{1.0F};
};

[[nodiscard]] inline ScaledVelocity scale_velocity(const SignalGridSpec& spec,
                                                   const FlowGridLayout& layout,
                                                   std::span<const float> values,
                                                   std::span<const std::uint8_t> open_inlet,
                                                   float requested_mean,
                                                   std::span<const float> driven = {}) {
  if ((!driven.empty() && driven.size() != values.size()) ||
      values.size() != layout.total_face_count() || open_inlet.size() != values.size()) {
    throw std::invalid_argument("flow velocity scaling arrays have inconsistent sizes");
  }
  double inlet_sum = 0.0, driven_inlet_sum = 0.0;
  std::size_t inlet_count = 0;
  float peak = 0.0F;
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (!std::isfinite(values[index])) {
      throw std::runtime_error("flow solve produced a non-finite velocity");
    }
    peak = std::max(peak, std::abs(values[index]));
    if (open_inlet[index] != 0) {
      inlet_sum += values[index];
      if (!driven.empty()) driven_inlet_sum += driven[index];
      ++inlet_count;
    }
  }
  if (inlet_count == 0) {
    throw std::logic_error("flow solve has no open inlet faces");
  }
  const auto solved_mean = static_cast<float>(inlet_sum / static_cast<double>(inlet_count));
  if (peak == 0.0F || solved_mean <= 1.0e-9F * peak) {
    throw std::runtime_error("the device carries no through-flow: the outlet is unreachable");
  }
  const auto driven_mean = static_cast<float>(driven_inlet_sum / static_cast<double>(inlet_count));
  const auto factor = (requested_mean - driven_mean) / solved_mean;
  std::vector<float> scaled(values.size());
  float scaled_peak = 0.0F;
  for (std::size_t index = 0; index < values.size(); ++index) {
    scaled[index] = values[index] * factor + (driven.empty() ? 0.0F : driven[index]);
    if (!std::isfinite(scaled[index]))
      throw std::runtime_error("non-finite combined flow velocity");
    scaled_peak = std::max(scaled_peak, std::abs(scaled[index]));
  }

  auto field = make_velocity_field(spec, layout, scaled);
  return {.field = std::move(field),
          .solved_mean = solved_mean,
          .max_speed = scaled_peak,
          .factor = factor};
}

// For a moving population the flow is affine in the pressure drop. Project
// the body-driven flux at zero boundary pressure, then adjust only the
// pressure-driven solution to meet the imposed mean inlet speed.
template <typename SolveSystem>
DepthAveragedFlowResult solve_shallow_axis_flow(const SignalGridSpec& spec,
                                                std::span<const float> mobility,
                                                const DepthAveragedFlowParameters& parameters,
                                                SolveSystem&& solve_system) {
  parameters.validate();
  const auto reduction = reduce_shallow_flow(spec, mobility, parameters.axis);
  const auto axis = static_cast<std::size_t>(parameters.axis);
  auto boundary = axis_flow_boundary(reduction.reduced_layout(), axis);
  DepthAveragedFlowReport report;
  const DepthAveragedFlowSystem system(reduction.grid(), reduction.conductance(), boundary);
  const auto pressure_velocity = reduction.lift(solve_system(system, report));
  std::vector<float> driven;
  if (!parameters.body_force.empty()) {
    auto flux = reduction.body_flux(parameters.body_force, boundary);
    std::fill(boundary.pressure.begin(), boundary.pressure.end(), 0.0F);
    const DepthAveragedFlowSystem forced(reduction.grid(), reduction.conductance(), boundary, flux);
    const auto correction = solve_system(forced, report);
    for (std::size_t i = 0; i < flux.size(); ++i) flux[i] += correction[i];
    driven = reduction.lift(flux);
  }
  auto scaled =
      scale_velocity(spec, reduction.original_layout(), pressure_velocity,
                     reduction.open_inlet_faces(axis), parameters.mean_inlet_speed, driven);
  report.mean_inlet_speed = parameters.mean_inlet_speed;
  report.max_speed = scaled.max_speed;
  return {.field = std::move(scaled.field), .report = report};
}

// Resolved flow along one axis, scaled to the stated mean inlet speed.
// `solve_system(system, controls)` solves one block system on a backend.
template <typename SolveSystem>
[[nodiscard]] ResolvedFlowResult solve_resolved_axis_flow(const SignalGridSpec& spec,
                                                          std::span<const float> drag,
                                                          const ResolvedFlowParameters& parameters,
                                                          SolveSystem&& solve_system) {
  parameters.validate();
  const ResolvedFlowSystem system(spec, drag, parameters.axis);
  const ResolvedSystemSolve solved =
      solve_system(system, ResolvedSolveControls{
                               .relative_tolerance = parameters.relative_tolerance,
                               .max_outer_iterations = parameters.max_outer_iterations,
                               .inner_relative_tolerance = parameters.inner_relative_tolerance,
                               .max_inner_iterations = parameters.max_inner_iterations,
                           });
  const auto scaled = scale_velocity(spec, system.layout(), solved.velocity,
                                     system.open_inlet_faces(), parameters.mean_inlet_speed);
  return {
      .field = scaled.field,
      .report = {.outer_iterations = solved.outer_iterations,
                 .inner_iterations = solved.inner_iterations,
                 .relative_residual = solved.relative_residual,
                 .momentum_relative_residual = solved.momentum_relative_residual,
                 .divergence_rms =
                     static_cast<float>(solved.divergence_rms * std::abs(scaled.factor)),
                 .mean_inlet_speed = parameters.mean_inlet_speed,
                 .max_speed = scaled.max_speed,
                 .min_gap_voxels = system.minimum_gap_voxels()},
  };
}

// Resolved flow driven through ports, as the combination of one block solve
// for each speed port. `solve_system(system, controls)` solves one block
// system on a backend.
template <typename SolveSystem>
[[nodiscard]] ResolvedPortFlowResult solve_resolved_port_flow(
    const SignalGridSpec& spec, std::span<const FlowPort> ports, std::span<const float> drag,
    const ResolvedPortFlowParameters& parameters, SolveSystem&& solve_system) {
  parameters.validate();
  validate_flow_ports(spec, ports);
  const FlowGridLayout layout(spec);
  std::vector<PortFaces> faces;
  for (const auto& port : ports) {
    std::vector<std::size_t> open;
    for_each_port_face(layout, port, [&](std::size_t face, std::size_t site) {
      if (!spec.solid_site(site)) {
        open.push_back(face);
      }
    });
    faces.push_back(make_port_faces(layout, port, std::move(open)));
  }
  const ResolvedSolveControls controls{
      .relative_tolerance = parameters.relative_tolerance,
      .max_outer_iterations = parameters.max_outer_iterations,
      .inner_relative_tolerance = parameters.inner_relative_tolerance,
      .max_inner_iterations = parameters.max_inner_iterations,
  };
  ResolvedPortFlowReport report;
  const auto solution = superpose_port_flows(ports, faces, [&](std::span<const float> pressures) {
    // Where the geometry allows, solve for the correction to the shallow
    // flow's pressure, which already carries the drop along each channel.
    const auto known = shallow_port_pressure(spec, ports, pressures);
    const ResolvedFlowSystem system(spec, drag, port_flow_boundary(layout, ports, pressures),
                                    known);
    ResolvedSystemSolve solved = solve_system(system, controls);
    report.outer_iterations += solved.outer_iterations;
    report.inner_iterations += solved.inner_iterations;
    report.relative_residual = std::max(report.relative_residual, solved.relative_residual);
    report.momentum_relative_residual =
        std::max(report.momentum_relative_residual, solved.momentum_relative_residual);
    return std::move(solved.velocity);
  });

  // The divergence of the combined field and the gaps of the geometry do not
  // depend on the pressures the ports hold.
  const std::vector<float> at_rest(ports.size(), 0.0F);
  const ResolvedFlowSystem system(spec, drag, port_flow_boundary(layout, ports, at_rest));
  const std::vector<double> velocity(solution.velocity.begin(), solution.velocity.end());
  const auto divergence = system.divergence(velocity);
  double square_sum = 0.0;
  std::size_t fluid_count = 0;
  for (std::size_t site = 0; site < divergence.size(); ++site) {
    if (system.fluid()[site] != 0) {
      square_sum += divergence[site] * divergence[site];
      ++fluid_count;
    }
  }
  report.solves = solution.solves;
  report.divergence_rms =
      fluid_count == 0
          ? 0.0F
          : static_cast<float>(std::sqrt(square_sum / static_cast<double>(fluid_count)));
  report.max_speed = solution.max_speed;
  report.min_gap_voxels = system.minimum_gap_voxels();
  return {
      .field = make_velocity_field(spec, layout, solution.velocity),
      .report = report,
      .ports = solution.ports,
  };
}

}  // namespace microsimulator::detail
