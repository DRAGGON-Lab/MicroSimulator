#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "microsimulator/channels.hpp"
#include "microsimulator/constraints.hpp"
#include "microsimulator/signals.hpp"
#include "microsimulator/types.hpp"

namespace microsimulator {

class Simulation;

inline constexpr std::string_view scene_format = "microsimulator-scene";
inline constexpr std::uint32_t scene_format_version = 3;
inline constexpr std::size_t max_scene_bytes = std::size_t{1} << 30U;
// Presentation resource budget per channel group, independent of how many
// channels a simulation may hold.
inline constexpr std::uint32_t max_scene_channels = 4096;

// Raised when a scene frame cannot be encoded, decoded, or validated.
class SceneError : public std::invalid_argument {
 public:
  using std::invalid_argument::invalid_argument;
};

using SceneVector = std::array<float, 3>;

struct SceneBackend {
  BackendKind kind{BackendKind::cpu};
  std::string name;
  std::string device;
  std::uint32_t device_index{0};
  bool native{false};

  [[nodiscard]] bool operator==(const SceneBackend&) const = default;
};

struct SceneCell {
  CellId id{invalid_cell_id};
  std::optional<CellId> parent_id;
  Slot slot{0};
  SceneVector position{};
  SceneVector direction{1.0F, 0.0F, 0.0F};
  float length{0.0F};
  float radius{0.0F};
  float growth_rate{0.0F};
  std::int32_t cell_type{0};
  bool fixed{false};
  std::vector<float> species;

  [[nodiscard]] bool operator==(const SceneCell&) const = default;
};

struct SceneGridBoundary {
  GridBoundaryKind kind{GridBoundaryKind::no_flux};
  std::vector<float> values;

  [[nodiscard]] bool operator==(const SceneGridBoundary&) const = default;
};

// Signal levels are channel-major: the level of signal s at site (x, y, z) is
// at index ((s * X + x) * Y + y) * Z + z for a grid of shape (X, Y, Z).
struct SceneSignalGrid {
  std::uint32_t signal_count{1};
  std::array<std::uint32_t, 3> shape{1, 1, 1};
  SceneVector origin{};
  SceneVector spacing{1.0F, 1.0F, 1.0F};
  SceneGridBoundary x_lower;
  SceneGridBoundary x_upper;
  SceneGridBoundary y_lower;
  SceneGridBoundary y_upper;
  SceneGridBoundary z_lower;
  SceneGridBoundary z_upper;
  std::vector<float> levels;

  [[nodiscard]] bool operator==(const SceneSignalGrid&) const = default;
};

struct ScenePlaneConstraint {
  ConstraintId id{invalid_constraint_id};
  SceneVector point{};
  SceneVector inward_normal{0.0F, 1.0F, 0.0F};
  float coefficient{1.0F};

  [[nodiscard]] bool operator==(const ScenePlaneConstraint&) const = default;
};

struct SceneSphereConstraint {
  ConstraintId id{invalid_constraint_id};
  SceneVector center{};
  float radius{1.0F};
  float coefficient{1.0F};
  ConstraintRegion allowed_region{ConstraintRegion::outside};

  [[nodiscard]] bool operator==(const SceneSphereConstraint&) const = default;
};

struct SceneBoxConstraint {
  ConstraintId id{invalid_constraint_id};
  SceneVector center{};
  SceneVector half_extents{1.0F, 1.0F, 1.0F};
  float coefficient{1.0F};
  ConstraintRegion allowed_region{ConstraintRegion::outside};

  [[nodiscard]] bool operator==(const SceneBoxConstraint&) const = default;
};

struct SceneCylinderConstraint {
  ConstraintId id{invalid_constraint_id};
  SceneVector center{};
  float radius{1.0F};
  float half_height{1.0F};
  float coefficient{1.0F};
  ConstraintRegion allowed_region{ConstraintRegion::outside};

  [[nodiscard]] bool operator==(const SceneCylinderConstraint&) const = default;
};

struct SceneConstraints {
  std::vector<ScenePlaneConstraint> planes;
  std::vector<SceneSphereConstraint> spheres;
  std::vector<SceneBoxConstraint> boxes;
  std::vector<SceneCylinderConstraint> cylinders;

  [[nodiscard]] bool operator==(const SceneConstraints&) const = default;
};

// A complete presentation snapshot of one simulation instant. It is a
// projection for viewers and analysis, not a checkpoint: a simulation cannot
// be resumed from it.
struct SceneFrame {
  double time{0.0};
  SceneBackend backend;
  std::uint32_t species_count{0};
  std::vector<SceneCell> cells;
  SceneConstraints constraints;
  std::optional<SceneSignalGrid> signal_grid;
  ChannelMetadata channel_metadata;

  // Checks the channel budget, then replaces the channel metadata with its
  // resolved form. The budget check comes first so an oversized channel count
  // is rejected before any per-channel label is allocated. Throws SceneError
  // for the budget and ChannelMetadataError for labels that do not match.
  void resolve_channels();

  // Throws SceneError unless the frame is internally consistent: finite
  // values, compact slot order, unique identifiers, normalized directions,
  // positive sizes, and array lengths that match the declared counts.
  void validate() const;

  [[nodiscard]] bool operator==(const SceneFrame&) const = default;
};

// Captures a validated frame from the current simulation state. Throws
// ChannelMetadataError when the labels do not match the simulation's channels.
[[nodiscard]] SceneFrame capture_scene(const Simulation& simulation,
                                       const ChannelMetadata& channel_metadata = {});

// Encodes a validated frame as readable JSON text. Cell and constraint
// identifiers are decimal strings because they exceed the integer range of
// JSON consumers that read numbers as float64. The frame carries a SHA-256
// digest of its RFC 8785 canonical JSON, which detects accidental corruption
// and is not an authenticity signature.
[[nodiscard]] std::string encode_scene(const SceneFrame& frame);

// Decodes scene text. Rejects text that is not strict JSON, objects with
// unknown or missing keys, any other format version, a digest mismatch, and
// frames that fail validation.
[[nodiscard]] SceneFrame parse_scene(std::string_view text);

// Atomically replaces the file at the path with the encoded frame.
void save_scene(const std::filesystem::path& path, const SceneFrame& frame);

// Reads and decodes a scene of at most max_scene_bytes.
[[nodiscard]] SceneFrame load_scene(const std::filesystem::path& path);

}  // namespace microsimulator
