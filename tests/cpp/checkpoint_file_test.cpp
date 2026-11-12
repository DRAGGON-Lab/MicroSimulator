#include "microsimulator/checkpoint_file.hpp"

#include <cassert>
#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>

#include "microsimulator/atomic_file.hpp"
#include "microsimulator/sha256.hpp"
#include "microsimulator/simulation.hpp"

namespace {

using microsimulator::CheckpointDocument;
using microsimulator::CheckpointError;
using microsimulator::JsonValue;

template <typename Function>
void assert_rejected(Function&& function, std::string_view fragment) {
  bool rejected = false;
  try {
    function();
  } catch (const CheckpointError& error) {
    rejected = std::string_view(error.what()).find(fragment) != std::string_view::npos;
  }
  assert(rejected);
}

void assert_parse_rejected(std::string_view text, std::string_view fragment) {
  assert_rejected([&] { static_cast<void>(microsimulator::parse_checkpoint(text)); }, fragment);
}

[[nodiscard]] std::string replaced(std::string text, std::string_view from, std::string_view to) {
  const auto position = text.find(from);
  assert(position != std::string::npos);
  text.replace(position, from.size(), to);
  return text;
}

// Returns a copy of an object with one member added or replaced.
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

// Re-encodes a document whose simulation payload has been edited, with a
// digest that matches the edit, so decoding reaches the state checks.
[[nodiscard]] std::string resigned(const JsonValue& root, JsonValue simulation) {
  auto integrity = with_member(
      *root.find("integrity"), "simulation",
      JsonValue::string(microsimulator::sha256_hex(microsimulator::canonical_json(simulation))));
  return microsimulator::pretty_json(with_member(
      with_member(root, "integrity", std::move(integrity)), "simulation", std::move(simulation)));
}

[[nodiscard]] microsimulator::Simulation make_simulation() {
  using enum microsimulator::RateOp;
  microsimulator::Simulation simulation(microsimulator::BackendKind::cpu, 8, 2);
  simulation.set_species_rate_plan(
      microsimulator::SpeciesRatePlan(2,
                                      {
                                          {.operation = species, .first = 0},
                                          {.operation = constant, .value = 0.125F},
                                          {.operation = add, .first = 0, .second = 1},
                                          {.operation = negate, .first = 0},
                                      },
                                      {2, 3}));

  microsimulator::SignalGridSpec spec;
  spec.signal_count = 2;
  spec.shape = {3, 2, 2};
  spec.spacing = {1.0F, 1.0F, 1.0F};
  spec.diffusion = {0.05F, 0.02F};
  spec.advection = {{0.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.0F}};
  spec.x_lower = {.kind = microsimulator::GridBoundaryKind::fixed, .values = {1.0F, 0.5F}};
  spec.reaction = microsimulator::SignalGridAffineReaction{
      .source_rates = std::vector<float>(24, 0.01F),
      .loss_rates = std::vector<float>(24, 0.02F),
  };
  std::vector<float> levels(24);
  for (std::size_t index = 0; index < levels.size(); ++index) {
    levels[index] = 0.1F * static_cast<float>(index);
  }
  simulation.configure_signal_grid(spec, levels);

  microsimulator::CellInit first;
  first.position = {1.25F, 0.5F, 0.5F};
  first.direction = {0.0F, 1.0F, 0.0F};
  first.length = 4.5F;
  first.radius = 0.4F;
  first.growth_rate = 0.2F;
  first.cell_type = 7;
  first.species = {3.0F, 1.5F};
  const auto first_id = simulation.add_cell(first);
  microsimulator::CellInit second;
  second.position = {0.5F, 0.25F, 0.75F};
  second.length = 2.75F;
  second.radius = 0.3F;
  second.growth_rate = 0.05F;
  second.cell_type = -2;
  second.fixed = true;
  second.species = {0.1F, 4.0F};
  simulation.add_cell(second);

  simulation.add_plane_constraint({.point = {0.0F, -3.0F, 0.0F}, .coefficient = 1.25F});
  simulation.add_sphere_constraint({.center = {1.0F, 2.0F, 3.0F},
                                    .radius = 8.0F,
                                    .coefficient = 0.75F,
                                    .allowed_region = microsimulator::ConstraintRegion::inside});
  simulation.add_box_constraint({.center = {9.0F, 9.0F, 9.0F}, .half_extents = {1.0F, 2.0F, 3.0F}});
  simulation.add_cylinder_constraint({.center = {-9.0F, 0.0F, 0.0F}, .radius = 1.5F});

  simulation.step(0.125F);
  static_cast<void>(simulation.divide_equal(first_id));
  simulation.step(0.03125F);
  return simulation;
}

void assert_states_equal(const microsimulator::SimulationCheckpoint& actual,
                         const microsimulator::SimulationCheckpoint& expected) {
  // The encoded form is a complete, deterministic function of the state, so
  // equal text means every field survived exactly.
  CheckpointDocument left;
  left.simulation = actual;
  CheckpointDocument right;
  right.simulation = expected;
  assert(microsimulator::encode_checkpoint(left) == microsimulator::encode_checkpoint(right));
}

[[nodiscard]] std::size_t entry_count(const std::filesystem::path& directory) {
  std::size_t count = 0;
  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    static_cast<void>(entry);
    ++count;
  }
  return count;
}

}  // namespace

int main() {
  auto original = make_simulation();

  CheckpointDocument document;
  document.simulation = original.checkpoint();
  document.source_backend = original.backend_info();
  document.controller = microsimulator::parse_json(
      R"({"completed_steps":18446744073709551615,"ratio":2.0,
          "nested":{"labels":["a",null,true],"x":-0.1}})");
  document.provenance = microsimulator::parse_json(R"({"model":{"seed":7,"name":"é🧪"}})");
  document.channel_metadata.species =
      microsimulator::ChannelMetadata::Labels{std::string("<b>α 🧪</b>"), std::nullopt};

  const auto text = microsimulator::encode_checkpoint(document);
  assert(text == microsimulator::encode_checkpoint(document));
  const auto decoded = microsimulator::parse_checkpoint(text);
  assert_states_equal(decoded.simulation, document.simulation);
  assert(decoded.controller == document.controller);
  assert(decoded.controller.find("completed_steps")->as_integer_token() == "18446744073709551615");
  assert(decoded.controller.find("ratio")->is_number());
  assert(decoded.provenance == document.provenance);
  assert(decoded.source_backend.kind == microsimulator::BackendKind::cpu);
  assert(decoded.source_backend.name == document.source_backend.name);
  assert(decoded.channel_metadata == document.channel_metadata.resolved(2, 2));
  assert(microsimulator::encode_checkpoint(decoded) == text);

  // A restored simulation continues exactly as the original does.
  microsimulator::Simulation restored(microsimulator::BackendKind::cpu, decoded.simulation);
  original.step(0.0625F);
  restored.step(0.0625F);
  assert_states_equal(restored.checkpoint(), original.checkpoint());

  // Files are replaced atomically and leave no temporary sibling behind.
  const auto directory =
      std::filesystem::temp_directory_path() / "microsimulator-checkpoint-file-test";
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  const auto path = directory / "run.json";
  microsimulator::save_checkpoint(path, document);
  microsimulator::save_checkpoint(path, document);
  assert(entry_count(directory) == 1);
  assert(microsimulator::read_file_prefix(path, text.size() + 1) == text);
  assert(microsimulator::encode_checkpoint(microsimulator::load_checkpoint(path)) == text);
  assert_rejected(
      [&] { static_cast<void>(microsimulator::load_checkpoint(directory / "absent.json")); },
      "could not read checkpoint");
  assert_rejected(
      [&] { microsimulator::save_checkpoint(directory / "missing" / "run.json", document); },
      "could not write checkpoint");
  microsimulator::write_file_atomically(path, "");
  assert_rejected([&] { static_cast<void>(microsimulator::load_checkpoint(path)); }, "is empty");
  std::filesystem::remove_all(directory);

  // Corruption and schema violations.
  assert_parse_rejected("", "checkpoint is empty");
  assert_parse_rejected("{", "not valid JSON");
  assert_parse_rejected("[]", "$: expected an object");
  assert_parse_rejected("{}", "$: missing keys ['version']");
  assert_parse_rejected(replaced(text, "\"version\": 10", "\"version\": 9"),
                        "$.version: unsupported checkpoint version 9");
  assert_parse_rejected(replaced(text, "\"version\": 10", "\"version\": 11"),
                        "$.version: unsupported checkpoint version 11");
  assert_parse_rejected(replaced(text, "{\n", "{\n  \"extra\": 1,\n"), "$: unknown keys ['extra']");
  assert_parse_rejected(replaced(text, "microsimulator-checkpoint", "other-format"),
                        "$.format: not a MicroSimulator checkpoint");
  assert_parse_rejected(replaced(text, "\"next_id\": 5", "\"next_id\": 6"),
                        "$.integrity.simulation: state digest does not match");
  assert_parse_rejected(replaced(text, "\"ratio\": 2.0", "\"ratio\": 2.5"),
                        "$.integrity.controller: controller digest does not match");
  assert_parse_rejected(replaced(text, "<b>α 🧪</b>", "renamed"),
                        "$.integrity.channel_metadata: channel metadata digest does not match");
  assert_parse_rejected(replaced(text, "\"sha256\"", "\"md5\""), "unsupported integrity algorithm");
  assert_parse_rejected(replaced(text, "\"kind\": \"cpu\"", "\"kind\": \"abacus\""),
                        "$.source_backend.kind: unknown backend kind 'abacus'");
  assert_parse_rejected(replaced(text, "\"provenance\": {", "\"provenance\": {\"model\": 1, "),
                        "duplicate key \"model\"");
  assert_parse_rejected(replaced(text, "\"ratio\": 2.0", "\"ratio\": NaN"), "not valid JSON");

  // Edits with a matching digest reach the typed decoder and state validation.
  const auto root = microsimulator::parse_json(text);
  const auto& state = *root.find("simulation");
  const auto& world = *state.find("world");
  const auto& cells = world.find("cells")->as_array();

  auto bad_slot_cells = cells;
  bad_slot_cells[0] = with_member(cells[0], "slot", JsonValue::integer(std::uint64_t{2}));
  assert_parse_rejected(
      resigned(root, with_member(
                         state, "world",
                         with_member(world, "cells", JsonValue::array(std::move(bad_slot_cells))))),
      "checkpoint state is invalid");

  auto bad_type_cells = cells;
  bad_type_cells[1] = with_member(cells[1], "cell_type", JsonValue::integer_token("2147483648"));
  assert_parse_rejected(
      resigned(root, with_member(
                         state, "world",
                         with_member(world, "cells", JsonValue::array(std::move(bad_type_cells))))),
      "$.simulation.world.cells[1].cell_type: integer is outside [-2147483648, 2147483647]");

  auto bad_float_cells = cells;
  bad_float_cells[0] = with_member(cells[0], "length", JsonValue::number(1.0e39));
  assert_parse_rejected(
      resigned(root, with_member(state, "world",
                                 with_member(world, "cells",
                                             JsonValue::array(std::move(bad_float_cells))))),
      "$.simulation.world.cells[0].length: number is outside the finite float32 range");

  auto unknown_member_cells = cells;
  unknown_member_cells[0] = with_member(cells[0], "color", JsonValue::string("red"));
  assert_parse_rejected(
      resigned(root, with_member(state, "world",
                                 with_member(world, "cells",
                                             JsonValue::array(std::move(unknown_member_cells))))),
      "$.simulation.world.cells[0]: unknown keys ['color']");

  assert_parse_rejected(resigned(root, with_member(state, "time", JsonValue::string("now"))),
                        "$.simulation.time: expected a number");

  // Encoding refuses state and labels that could not be loaded again.
  auto invalid = document;
  invalid.simulation.world.cells[0].slot = 5;
  assert_rejected([&] { static_cast<void>(microsimulator::encode_checkpoint(invalid)); },
                  "checkpoint state is invalid");
  auto mislabeled = document;
  mislabeled.channel_metadata.species = microsimulator::ChannelMetadata::Labels{std::nullopt};
  assert_rejected([&] { static_cast<void>(microsimulator::encode_checkpoint(mislabeled)); },
                  "channel_metadata.species: expected 2 labels, got 1");
  auto deep = document;
  deep.controller = JsonValue::array({});
  for (int depth = 1; depth < 256; ++depth) {
    deep.controller = JsonValue::array({std::move(deep.controller)});
  }
  assert_rejected([&] { static_cast<void>(microsimulator::encode_checkpoint(deep)); },
                  "checkpoint cannot be encoded: nesting is too deep");
  auto bad_provenance = document;
  bad_provenance.provenance = JsonValue::array({});
  assert_rejected([&] { static_cast<void>(microsimulator::encode_checkpoint(bad_provenance)); },
                  "provenance must be a JSON object");
  return 0;
}
