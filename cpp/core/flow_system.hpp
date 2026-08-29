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

#include "cm/flow.hpp"

namespace cm::detail {

class FlowGridLayout {
 public:
  FlowGridLayout(const SignalGridSpec& spec, FlowAxis flow_axis)
      : dimensions_{spec.shape.x, spec.shape.y, spec.shape.z},
        spacing_{spec.spacing.x, spec.spacing.y, spec.spacing.z},
        flow_axis_(static_cast<std::size_t>(flow_axis)) {
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

  [[nodiscard]] const std::array<float, 3>& spacing() const noexcept { return spacing_; }

  [[nodiscard]] std::size_t flow_axis() const noexcept { return flow_axis_; }

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

  [[nodiscard]] std::size_t total_face_count() const noexcept { return total_face_count_; }

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
  std::size_t flow_axis_{0};
  std::array<std::size_t, 3> face_counts_{};
  std::array<std::size_t, 3> face_offsets_{};
  std::size_t total_face_count_{0};
};

[[nodiscard]] inline float harmonic_mean(float first, float second) noexcept {
  const auto sum = first + second;
  return sum > 0.0F ? 2.0F * first * second / sum : 0.0F;
}

class DepthAveragedFlowSystem {
 public:
  DepthAveragedFlowSystem(const SignalGridSpec& spec, std::span<const float> mobility,
                          FlowAxis axis)
      : spec_(spec),
        layout_(spec, axis),
        mobility_(layout_.site_count(), 1.0F),
        diagonal_(layout_.site_count()),
        right_hand_side_(layout_.site_count()) {
    validate_flow_grid(spec_, axis);
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

    bool open_inlet = false;
    for (std::size_t site = 0; site < layout_.site_count(); ++site) {
      const auto value = mobility_[site];
      if (value == 0.0F) {
        continue;
      }
      const auto coordinates = layout_.site_coordinates(site);
      for (std::size_t component = 0; component < 3; ++component) {
        const auto inverse_square =
            1.0F / (layout_.spacing()[component] * layout_.spacing()[component]);
        for (const auto offset : {-1, 1}) {
          const auto neighbor = layout_.neighbor_site(site, component, offset);
          if (neighbor.has_value()) {
            diagonal_[site] += harmonic_mean(value, mobility_[*neighbor]) * inverse_square;
          }
        }
      }
      if (coordinates[layout_.flow_axis()] == 0) {
        const auto boundary =
            2.0F * value /
            (layout_.spacing()[layout_.flow_axis()] * layout_.spacing()[layout_.flow_axis()]);
        diagonal_[site] += boundary;
        right_hand_side_[site] = boundary;
        open_inlet = true;
      }
      if (coordinates[layout_.flow_axis()] + 1 == layout_.dimensions()[layout_.flow_axis()]) {
        diagonal_[site] +=
            2.0F * value /
            (layout_.spacing()[layout_.flow_axis()] * layout_.spacing()[layout_.flow_axis()]);
      }
    }
    if (!open_inlet) {
      throw std::invalid_argument("the flow inlet boundary is entirely blocked");
    }
  }

  [[nodiscard]] const FlowGridLayout& layout() const noexcept { return layout_; }
  [[nodiscard]] const std::vector<float>& mobility() const noexcept { return mobility_; }
  [[nodiscard]] const std::vector<float>& diagonal() const noexcept { return diagonal_; }
  [[nodiscard]] const std::vector<float>& right_hand_side() const noexcept {
    return right_hand_side_;
  }

  void apply(std::span<const double> input, std::vector<double>& output) const {
    if (input.size() != layout_.site_count()) {
      throw std::invalid_argument("depth-averaged flow vector has the wrong size");
    }
    output.assign(input.size(), 0.0);
    for (std::size_t site = 0; site < input.size(); ++site) {
      if (diagonal_[site] == 0.0F) {
        continue;
      }
      auto result = static_cast<double>(diagonal_[site]) * input[site];
      for (std::size_t component = 0; component < 3; ++component) {
        const auto inverse_square = 1.0 / (static_cast<double>(layout_.spacing()[component]) *
                                           layout_.spacing()[component]);
        for (const auto offset : {-1, 1}) {
          const auto neighbor = layout_.neighbor_site(site, component, offset);
          if (neighbor.has_value()) {
            const auto conductance =
                static_cast<double>(harmonic_mean(mobility_[site], mobility_[*neighbor])) *
                inverse_square;
            result -= conductance * input[*neighbor];
          }
        }
      }
      output[site] = result;
    }
  }

  [[nodiscard]] std::vector<float> velocity(std::span<const double> pressure) const {
    if (pressure.size() != layout_.site_count()) {
      throw std::invalid_argument("depth-averaged pressure vector has the wrong size");
    }
    std::vector<float> result(layout_.total_face_count(), 0.0F);
    for (std::size_t face_index = 0; face_index < result.size(); ++face_index) {
      const auto [component, face] = layout_.face_coordinates(face_index);
      const auto lower = layout_.adjacent_site(component, face, -1);
      const auto upper = layout_.adjacent_site(component, face, 1);
      const auto spacing = static_cast<double>(layout_.spacing()[component]);
      double value = 0.0;
      if (lower.has_value() && upper.has_value()) {
        const auto face_mobility = harmonic_mean(mobility_[*lower], mobility_[*upper]);
        value =
            -static_cast<double>(face_mobility) * (pressure[*upper] - pressure[*lower]) / spacing;
      } else if (component == layout_.flow_axis() && upper.has_value()) {
        value = 2.0 * static_cast<double>(mobility_[*upper]) * (1.0 - pressure[*upper]) / spacing;
      } else if (component == layout_.flow_axis() && lower.has_value()) {
        value = 2.0 * static_cast<double>(mobility_[*lower]) * pressure[*lower] / spacing;
      }
      result[face_index] = static_cast<float>(value);
    }
    return result;
  }

  [[nodiscard]] std::vector<std::uint8_t> open_inlet_faces() const {
    std::vector<std::uint8_t> result(layout_.total_face_count(), 0);
    const auto component = layout_.flow_axis();
    const auto face_dims = layout_.face_dimensions(component);
    for (std::uint32_t x = 0; x < face_dims[0]; ++x) {
      for (std::uint32_t y = 0; y < face_dims[1]; ++y) {
        for (std::uint32_t z = 0; z < face_dims[2]; ++z) {
          std::array<std::uint32_t, 3> face{x, y, z};
          if (face[component] != 0) {
            continue;
          }
          const auto upper = layout_.adjacent_site(component, face, 1);
          if (upper.has_value() && mobility_[*upper] > 0.0F) {
            result[layout_.face_index(component, x, y, z)] = 1;
          }
        }
      }
    }
    return result;
  }

 private:
  const SignalGridSpec& spec_;
  FlowGridLayout layout_;
  std::vector<float> mobility_;
  std::vector<float> diagonal_;
  std::vector<float> right_hand_side_;
};

class ResolvedFlowSystem {
 public:
  ResolvedFlowSystem(const SignalGridSpec& spec, std::span<const float> drag, FlowAxis axis)
      : spec_(spec),
        layout_(spec, axis),
        fluid_(layout_.site_count(), 1),
        drag_(layout_.site_count()),
        active_(layout_.total_face_count()),
        exists_(layout_.total_face_count()),
        face_drag_(layout_.total_face_count()),
        diagonal_(layout_.total_face_count()),
        force_(layout_.total_face_count()) {
    validate_flow_grid(spec_, axis);
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

    bool open_inlet = false;
    for (std::size_t index = 0; index < layout_.total_face_count(); ++index) {
      const auto [component, face] = layout_.face_coordinates(index);
      const auto lower = layout_.adjacent_site(component, face, -1);
      const auto upper = layout_.adjacent_site(component, face, 1);
      const auto lower_fluid = lower.has_value() && fluid_[*lower] != 0;
      const auto upper_fluid = upper.has_value() && fluid_[*upper] != 0;
      exists_[index] = lower_fluid || upper_fluid ? 1 : 0;
      auto active = lower_fluid && upper_fluid;
      if (component == layout_.flow_axis() &&
          (face[component] == 0 || face[component] == layout_.dimensions()[component])) {
        active = lower_fluid || upper_fluid;
      }
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
      if (component == layout_.flow_axis() && face[component] == 0) {
        force_[index] = 1.0F / layout_.spacing()[component];
        open_inlet = true;
      }
    }
    if (!open_inlet) {
      throw std::invalid_argument("the resolved-flow inlet boundary is entirely blocked");
    }
  }

  [[nodiscard]] const FlowGridLayout& layout() const noexcept { return layout_; }
  [[nodiscard]] const std::vector<std::uint8_t>& fluid() const noexcept { return fluid_; }
  [[nodiscard]] const std::vector<std::uint8_t>& active() const noexcept { return active_; }
  [[nodiscard]] const std::vector<std::uint8_t>& exists() const noexcept { return exists_; }
  [[nodiscard]] const std::vector<float>& face_drag() const noexcept { return face_drag_; }
  [[nodiscard]] const std::vector<float>& diagonal() const noexcept { return diagonal_; }
  [[nodiscard]] const std::vector<float>& force() const noexcept { return force_; }

  void apply_momentum(std::span<const double> input, std::vector<double>& output) const {
    if (input.size() != layout_.total_face_count()) {
      throw std::invalid_argument("resolved-flow face vector has the wrong size");
    }
    output.assign(input.size(), 0.0);
    for (std::size_t index = 0; index < input.size(); ++index) {
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
  }

  [[nodiscard]] std::vector<double> gradient(std::span<const double> pressure) const {
    if (pressure.size() != layout_.site_count()) {
      throw std::invalid_argument("resolved-flow pressure vector has the wrong size");
    }
    std::vector<double> result(layout_.total_face_count(), 0.0);
    for (std::size_t index = 0; index < result.size(); ++index) {
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
    return result;
  }

  [[nodiscard]] std::vector<double> divergence(std::span<const double> velocity) const {
    if (velocity.size() != layout_.total_face_count()) {
      throw std::invalid_argument("resolved-flow velocity vector has the wrong size");
    }
    std::vector<double> result(layout_.site_count(), 0.0);
    for (std::size_t site = 0; site < result.size(); ++site) {
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
    return result;
  }

  [[nodiscard]] std::vector<float> pressure_diagonal() const {
    std::vector<float> result(layout_.site_count(), 0.0F);
    for (std::size_t site = 0; site < result.size(); ++site) {
      if (fluid_[site] != 0) {
        result[site] = 1.0F;
      }
    }
    return result;
  }

  [[nodiscard]] std::vector<std::uint8_t> open_inlet_faces() const {
    std::vector<std::uint8_t> result(layout_.total_face_count(), 0);
    const auto component = layout_.flow_axis();
    for (std::size_t index = layout_.face_offsets()[component];
         index < layout_.face_offsets()[component] + layout_.face_counts()[component]; ++index) {
      const auto [face_component, face] = layout_.face_coordinates(index);
      if (face_component == component && face[component] == 0 && active_[index] != 0) {
        result[index] = 1;
      }
    }
    return result;
  }

  [[nodiscard]] std::uint32_t minimum_gap_voxels() const {
    std::uint32_t shortest = 0;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      if (axis == layout_.flow_axis() || layout_.dimensions()[axis] <= 1) {
        continue;
      }
      const auto first_axis = (axis + 1) % 3;
      const auto second_axis = (axis + 2) % 3;
      for (std::uint32_t first = 0; first < layout_.dimensions()[first_axis]; ++first) {
        for (std::uint32_t second = 0; second < layout_.dimensions()[second_axis]; ++second) {
          std::uint32_t run = 0;
          for (std::uint32_t along = 0; along < layout_.dimensions()[axis]; ++along) {
            std::array<std::uint32_t, 3> coordinates{};
            coordinates[axis] = along;
            coordinates[first_axis] = first;
            coordinates[second_axis] = second;
            if (fluid_[layout_.site_index(coordinates[0], coordinates[1], coordinates[2])] != 0) {
              ++run;
            } else if (run != 0) {
              shortest = shortest == 0 ? run : std::min(shortest, run);
              run = 0;
            }
          }
          if (run != 0) {
            shortest = shortest == 0 ? run : std::min(shortest, run);
          }
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
                                                   float requested_mean) {
  if (values.size() != layout.total_face_count() || open_inlet.size() != values.size()) {
    throw std::invalid_argument("flow velocity scaling arrays have inconsistent sizes");
  }
  double inlet_sum = 0.0;
  std::size_t inlet_count = 0;
  float peak = 0.0F;
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (!std::isfinite(values[index])) {
      throw std::runtime_error("flow solve produced a non-finite velocity");
    }
    peak = std::max(peak, std::abs(values[index]));
    if (open_inlet[index] != 0) {
      inlet_sum += values[index];
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
  const auto factor = requested_mean / solved_mean;
  std::vector<float> scaled(values.size());
  float scaled_peak = 0.0F;
  for (std::size_t index = 0; index < values.size(); ++index) {
    scaled[index] = values[index] * factor;
    scaled_peak = std::max(scaled_peak, std::abs(scaled[index]));
  }

  SignalGridVelocityField field;
  const auto offsets = layout.face_offsets();
  const auto counts = layout.face_counts();
  field.x_faces.assign(scaled.begin(), scaled.begin() + static_cast<std::ptrdiff_t>(counts[0]));
  field.y_faces.assign(scaled.begin() + static_cast<std::ptrdiff_t>(offsets[1]),
                       scaled.begin() + static_cast<std::ptrdiff_t>(offsets[1] + counts[1]));
  field.z_faces.assign(scaled.begin() + static_cast<std::ptrdiff_t>(offsets[2]), scaled.end());
  auto candidate = spec;
  candidate.velocity_field = field;
  for (auto& advection : candidate.advection) {
    advection = {};
  }
  candidate.validate();
  return {.field = std::move(field),
          .solved_mean = solved_mean,
          .max_speed = scaled_peak,
          .factor = factor};
}

}  // namespace cm::detail
