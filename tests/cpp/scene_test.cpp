#include "microsimulator/scene.hpp"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "microsimulator/atomic_file.hpp"
#include "microsimulator/json.hpp"
#include "microsimulator/sha256.hpp"
#include "microsimulator/simulation.hpp"

namespace {

using microsimulator::ChannelMetadata;
using microsimulator::ChannelMetadataError;
using microsimulator::JsonValue;
using microsimulator::SceneError;
using microsimulator::SceneFrame;

template <typename Exception, typename Function>
void assert_throws(Function&& function, std::string_view fragment) {
  bool rejected = false;
  try {
    function();
  } catch (const Exception& error) {
    rejected = std::string_view(error.what()).find(fragment) != std::string_view::npos;
  }
  assert(rejected);
}

void assert_invalid(const SceneFrame& frame, std::string_view fragment) {
  assert_throws<SceneError>([&] { frame.validate(); }, fragment);
  assert_throws<SceneError>([&] { static_cast<void>(microsimulator::encode_scene(frame)); },
                            fragment);
}

void assert_parse_rejected(std::string_view text, std::string_view fragment) {
  assert_throws<SceneError>([&] { static_cast<void>(microsimulator::parse_scene(text)); },
                            fragment);
}

[[nodiscard]] std::string replaced(std::string text, std::string_view from, std::string_view to) {
  const auto position = text.find(from);
  assert(position != std::string::npos);
  text.replace(position, from.size(), to);
  return text;
}

[[nodiscard]] JsonValue with_member(const JsonValue& object, std::string_view key,
                                    JsonValue value) {
  JsonValue::Object members;
  for (const auto& member : object.as_object()) {
    if (member.key != key) {
      members.push_back(member);
    }
  }
  members.push_back({std::string(key), std::move(value)});
  return JsonValue::object(std::move(members));
}

// Re-encodes a document whose frame has been edited, with a matching digest.
[[nodiscard]] std::string resigned(const JsonValue& root, JsonValue frame) {
  auto integrity = with_member(
      *root.find("integrity"), "frame",
      JsonValue::string(microsimulator::sha256_hex(microsimulator::canonical_json(frame))));
  return microsimulator::pretty_json(
      with_member(with_member(root, "integrity", std::move(integrity)), "frame", std::move(frame)));
}

[[nodiscard]] microsimulator::Simulation make_simulation() {
  microsimulator::Simulation simulation(microsimulator::BackendKind::cpu, 8, 2);
  microsimulator::SignalGridSpec spec;
  spec.signal_count = 2;
  spec.shape = {2, 2, 1};
  spec.spacing = {1.0F, 1.0F, 1.0F};
  spec.diffusion = {0.0F, 0.0F};
  spec.advection = {{0.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.0F}};
  spec.x_lower = {.kind = microsimulator::GridBoundaryKind::fixed, .values = {1.0F, 0.5F}};
  simulation.configure_signal_grid(spec, {0.0F, 0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.6F, 0.7F});

  microsimulator::CellInit cell;
  cell.position = {0.5F, 0.5F, 0.0F};
  cell.length = 4.5F;
  cell.radius = 0.4F;
  cell.growth_rate = 0.2F;
  cell.cell_type = -3;
  cell.species = {3.0F, 1.5F};
  const auto parent = simulation.add_cell(cell);
  simulation.add_plane_constraint({.point = {0.0F, -3.0F, 0.0F}, .coefficient = 1.25F});
  simulation.add_sphere_constraint({.center = {1.0F, 2.0F, 3.0F},
                                    .radius = 8.0F,
                                    .allowed_region = microsimulator::ConstraintRegion::inside});
  simulation.add_box_constraint({.center = {9.0F, 9.0F, 9.0F}, .half_extents = {1.0F, 2.0F, 3.0F}});
  simulation.add_cylinder_constraint({.center = {-9.0F, 0.0F, 0.0F}, .radius = 1.5F});
  simulation.step(0.125F);
  static_cast<void>(simulation.divide_equal(parent));
  return simulation;
}

void test_capture_and_round_trip() {
  const auto simulation = make_simulation();
  ChannelMetadata labels;
  labels.species = ChannelMetadata::Labels{std::string("<b>α 🧪</b>"), std::nullopt};
  const auto frame = microsimulator::capture_scene(simulation, labels);
  assert(frame.species_count == 2);
  assert(frame.cells.size() == 2);
  assert(frame.cells[0].parent_id == microsimulator::CellId{1});
  assert(frame.cells[0].slot == 0 && frame.cells[1].slot == 1);
  assert(frame.constraints.planes.size() == 1 && frame.constraints.cylinders.size() == 1);
  assert(frame.signal_grid.has_value() && frame.signal_grid->levels.size() == 8);
  assert(frame.signal_grid->x_lower.kind == microsimulator::GridBoundaryKind::fixed);
  assert(frame.channel_metadata == labels.resolved(2, 2));
  assert(frame.backend.kind == microsimulator::BackendKind::cpu && !frame.backend.name.empty());

  const auto text = microsimulator::encode_scene(frame);
  assert(text == microsimulator::encode_scene(frame));
  assert(text.find("\"id\": \"2\"") != std::string::npos);
  const auto decoded = microsimulator::parse_scene(text);
  assert(decoded == frame);
  assert(microsimulator::encode_scene(decoded) == text);

  // The digest covers the canonical frame, so reformatting the document or
  // rewriting a number in an equivalent form leaves it valid.
  const auto root = microsimulator::parse_json(text);
  assert(microsimulator::parse_scene(microsimulator::canonical_json(root)) == frame);

  // Identifiers span the full unsigned 64-bit range.
  auto wide = frame;
  wide.cells[1].id = std::numeric_limits<std::uint64_t>::max();
  wide.cells[1].parent_id = std::numeric_limits<std::uint64_t>::max() - 1;
  const auto wide_text = microsimulator::encode_scene(wide);
  assert(wide_text.find("\"18446744073709551615\"") != std::string::npos);
  assert(microsimulator::parse_scene(wide_text) == wide);

  assert_throws<ChannelMetadataError>(
      [&] {
        ChannelMetadata wrong;
        wrong.species = ChannelMetadata::Labels{std::nullopt};
        static_cast<void>(microsimulator::capture_scene(simulation, wrong));
      },
      "channel_metadata.species: expected 2 labels, got 1");

  const auto directory = std::filesystem::temp_directory_path() / "microsimulator-scene-test";
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  const auto path = directory / "frame.scene.json";
  microsimulator::save_scene(path, frame);
  microsimulator::save_scene(path, frame);
  std::size_t entries = 0;
  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    static_cast<void>(entry);
    ++entries;
  }
  assert(entries == 1);
  assert(microsimulator::load_scene(path) == frame);
  assert_throws<SceneError>(
      [&] { static_cast<void>(microsimulator::load_scene(directory / "absent.json")); },
      "could not read scene");
  std::filesystem::remove_all(directory);
}

void test_document_rejections() {
  const auto frame = microsimulator::capture_scene(make_simulation());
  const auto text = microsimulator::encode_scene(frame);
  assert_parse_rejected("", "scene is empty");
  assert_parse_rejected("{", "not valid JSON");
  assert_parse_rejected("[]", "$: expected an object");
  assert_parse_rejected(replaced(text, "\"version\": 3", "\"version\": 2"),
                        "$.version: unsupported scene version 2");
  assert_parse_rejected(replaced(text, "microsimulator-scene", "other"),
                        "$.format: not a MicroSimulator scene");
  assert_parse_rejected(replaced(text, "\"cell_type\": -3", "\"cell_type\": 3"),
                        "$.integrity.frame: frame digest does not match");
  assert_parse_rejected(replaced(text, "{\n", "{\n  \"extra\": 1,\n"), "$: unknown keys ['extra']");
  assert_parse_rejected(replaced(text, "\"frame\": {", "\"frame\": {\"time\": 0, "),
                        "duplicate key \"time\"");

  const auto root = microsimulator::parse_json(text);
  const auto& payload = *root.find("frame");
  const auto& cells = payload.find("cells")->as_array();

  auto colored = cells;
  colored[0] = with_member(cells[0], "color", JsonValue::string("red"));
  assert_parse_rejected(
      resigned(root, with_member(payload, "cells", JsonValue::array(std::move(colored)))),
      "$.frame.cells[0]: unknown keys ['color']");

  for (const std::string_view id : {"01", "0", "-1", "1.0", "", "18446744073709551616"}) {
    auto renamed = cells;
    renamed[0] = with_member(cells[0], "id", JsonValue::string(std::string(id)));
    assert_parse_rejected(
        resigned(root, with_member(payload, "cells", JsonValue::array(std::move(renamed)))),
        "$.frame.cells[0].id: ");
  }
  auto numeric = cells;
  numeric[0] = with_member(cells[0], "id", JsonValue::integer(std::uint64_t{1}));
  assert_parse_rejected(
      resigned(root, with_member(payload, "cells", JsonValue::array(std::move(numeric)))),
      "$.frame.cells[0].id: expected a string");

  // An oversized channel count is rejected by its declared value alone,
  // before the labels or per-cell arrays it would imply are examined.
  assert_parse_rejected(
      resigned(root,
               with_member(payload, "species_count", JsonValue::integer(std::uint64_t{4097}))),
      "$.frame.species_count: exceeds scene presentation channel budget of 4096 per group");
  const auto& grid = *payload.find("signal_grid");
  assert_parse_rejected(
      resigned(root, with_member(payload, "signal_grid",
                                 with_member(grid, "signal_count",
                                             JsonValue::integer(std::uint64_t{4097})))),
      "$.frame.signal_grid.signal_count: exceeds scene presentation channel budget");
  assert_parse_rejected(
      resigned(root, with_member(
                         payload, "channel_metadata",
                         microsimulator::parse_json(R"({"species":["a"],"signals":[null,null]})"))),
      "channel_metadata.species: expected 2 labels, got 1");
  assert_parse_rejected(
      resigned(root, with_member(payload, "signal_grid",
                                 with_member(grid, "levels", microsimulator::parse_json("[1.0]")))),
      "$.frame.signal_grid.levels: expected 8 values");
}

void test_frame_validation() {
  const auto frame = microsimulator::capture_scene(make_simulation());
  frame.validate();

  auto bad = frame;
  bad.cells[0].direction = {2.0F, 0.0F, 0.0F};
  assert_invalid(bad, "$.frame.cells[0].direction: must be normalized");
  bad = frame;
  bad.cells[1].slot = 5;
  assert_invalid(bad, "$.frame.cells[1].slot: cells must be compact and ordered by slot");
  bad = frame;
  bad.cells[1].id = bad.cells[0].id;
  assert_invalid(bad, "$.frame.cells[1].id: duplicate cell identifier");
  bad = frame;
  bad.cells[0].parent_id = bad.cells[0].id;
  assert_invalid(bad, "$.frame.cells[0].parent_id: must precede the child identifier");
  bad = frame;
  bad.cells[0].radius = 0.0F;
  assert_invalid(bad, "$.frame.cells[0].radius: must be positive");
  bad = frame;
  bad.cells[0].length = std::numeric_limits<float>::infinity();
  assert_invalid(bad, "$.frame.cells[0].length: number must be finite");
  bad = frame;
  bad.cells[0].species.pop_back();
  assert_invalid(bad, "$.frame.cells[0].species: expected 2 values");
  bad = frame;
  bad.time = -1.0;
  assert_invalid(bad, "$.frame.time: must be non-negative");
  bad = frame;
  bad.backend.device.clear();
  assert_invalid(bad, "$.frame.backend: name and device must not be empty");
  bad = frame;
  bad.constraints.spheres[0].id = bad.constraints.planes[0].id;
  assert_invalid(bad, "$.frame.constraints.spheres[0].id: duplicate constraint identifier");
  bad = frame;
  bad.constraints.planes[0].inward_normal = {0.0F, 2.0F, 0.0F};
  assert_invalid(bad, "$.frame.constraints.planes[0].inward_normal: must be normalized");
  bad = frame;
  bad.constraints.boxes[0].half_extents[1] = 0.0F;
  assert_invalid(bad, "$.frame.constraints.boxes[0].half_extents: values must be positive");
  bad = frame;
  bad.constraints.cylinders[0].half_height = -1.0F;
  assert_invalid(bad, "$.frame.constraints.cylinders[0].half_height: must be positive");
  bad = frame;
  bad.signal_grid->levels = {1.0F};
  assert_invalid(bad, "$.frame.signal_grid.levels: expected 8 values");
  bad = frame;
  bad.signal_grid->spacing[2] = 0.0F;
  assert_invalid(bad, "$.frame.signal_grid.spacing[2]: must be positive");
  bad = frame;
  bad.signal_grid->x_lower.values.pop_back();
  assert_invalid(bad, "$.frame.signal_grid.boundaries.x_lower.values: expected 2 values for fixed");
  bad = frame;
  bad.signal_grid->y_upper.values = {1.0F};
  assert_invalid(bad,
                 "$.frame.signal_grid.boundaries.y_upper.values: expected 0 values for no_flux");
  bad = frame;
  bad.channel_metadata.signals = ChannelMetadata::Labels{std::nullopt};
  assert_invalid(bad, "channel_metadata.signals: expected 2 labels, got 1");

  // Resolving channels bounds the declared counts before expanding labels.
  SceneFrame oversized;
  oversized.backend = frame.backend;
  oversized.species_count = microsimulator::max_scene_channels + 1;
  assert_throws<SceneError>([&] { oversized.resolve_channels(); },
                            "$.frame.species_count: exceeds scene presentation channel budget");
  assert(!oversized.channel_metadata.species.has_value());
  oversized.species_count = microsimulator::max_scene_channels;
  oversized.resolve_channels();
  assert(oversized.channel_metadata.species->size() == microsimulator::max_scene_channels);
  assert(oversized.channel_metadata.signals->empty());
}

void test_shared_fixture(const std::filesystem::path& path) {
  // A document shared with the TypeScript viewer's tests: both implementations
  // must compute the same canonical digest for it.
  const auto frame = microsimulator::load_scene(path);
  assert(frame.species_count == 2);
  assert(frame.channel_metadata.species ==
         ChannelMetadata::Labels({std::string("<b>α 🧪</b>"), std::string("<b>α 🧪</b>")}));
  assert(frame.channel_metadata.signals ==
         ChannelMetadata::Labels({std::nullopt, std::string(" ")}));
  assert(microsimulator::parse_scene(microsimulator::encode_scene(frame)) == frame);
}

}  // namespace

int main(int argument_count, char** arguments) {
  assert(argument_count == 2);
  test_capture_and_round_trip();
  test_document_rejections();
  test_frame_validation();
  test_shared_fixture(arguments[1]);
  return 0;
}
