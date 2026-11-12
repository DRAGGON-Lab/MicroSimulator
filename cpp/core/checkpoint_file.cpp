#include "microsimulator/checkpoint_file.hpp"

#include <array>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "core/json_reader.hpp"
#include "core/producer.hpp"
#include "microsimulator/atomic_file.hpp"
#include "microsimulator/names.hpp"
#include "microsimulator/sha256.hpp"

namespace microsimulator {
namespace {

using detail::array_json;
using detail::boundary_names;
using detail::decode_array;
using detail::expect_array;
using detail::expect_boolean;
using detail::expect_float;
using detail::expect_floats;
using detail::expect_int32;
using detail::expect_number;
using detail::expect_object;
using detail::expect_string;
using detail::expect_unsigned;
using detail::expect_vec3;
using detail::fail;
using detail::floats_to_json;
using detail::JsonPath;
using detail::name_json;
using detail::uint32_max;
using detail::vec3_to_json;

constexpr std::uint64_t uint64_max = std::numeric_limits<std::uint64_t>::max();

[[nodiscard]] JsonValue unsigned_json(std::uint64_t value) {
  return JsonValue::integer(value);
}

[[nodiscard]] JsonValue cell_json(const CellSnapshot& cell) {
  return JsonValue::object({
      {"id", unsigned_json(cell.id)},
      {"slot", unsigned_json(cell.slot)},
      {"position", vec3_to_json(cell.position)},
      {"direction", vec3_to_json(cell.direction)},
      {"length", JsonValue::number(cell.length)},
      {"radius", JsonValue::number(cell.radius)},
      {"growth_rate", JsonValue::number(cell.growth_rate)},
      {"cell_type", JsonValue::integer(std::int64_t{cell.cell_type})},
      {"fixed", JsonValue::boolean(cell.fixed)},
      {"species", floats_to_json(cell.species)},
  });
}

[[nodiscard]] JsonValue instruction_json(const RateInstruction& instruction) {
  return JsonValue::object({
      {"operation", name_json(rate_op_name(instruction.operation))},
      {"first", unsigned_json(instruction.first)},
      {"second", unsigned_json(instruction.second)},
      {"third", unsigned_json(instruction.third)},
      {"value", JsonValue::number(instruction.value)},
  });
}

[[nodiscard]] JsonValue boundary_json(const GridBoundary& boundary) {
  return JsonValue::object({
      {"kind", name_json(grid_boundary_kind_name(boundary.kind))},
      {"values", floats_to_json(boundary.values)},
  });
}

[[nodiscard]] JsonValue signal_grid_json(const std::optional<SignalGridCheckpoint>& grid) {
  if (!grid.has_value()) {
    return JsonValue{};
  }
  const auto& spec = grid->spec;
  JsonValue reaction;
  if (spec.reaction.has_value()) {
    reaction = JsonValue::object({
        {"source_rates", floats_to_json(spec.reaction->source_rates)},
        {"loss_rates", floats_to_json(spec.reaction->loss_rates)},
    });
  }
  JsonValue velocity_field;
  if (spec.velocity_field.has_value()) {
    velocity_field = JsonValue::object({
        {"x_faces", floats_to_json(spec.velocity_field->x_faces)},
        {"y_faces", floats_to_json(spec.velocity_field->y_faces)},
        {"z_faces", floats_to_json(spec.velocity_field->z_faces)},
    });
  }
  return JsonValue::object({
      {"spec",
       JsonValue::object({
           {"signal_count", unsigned_json(spec.signal_count)},
           {"shape", JsonValue::array({unsigned_json(spec.shape.x), unsigned_json(spec.shape.y),
                                       unsigned_json(spec.shape.z)})},
           {"origin", vec3_to_json(spec.origin)},
           {"spacing", vec3_to_json(spec.spacing)},
           {"diffusion", floats_to_json(spec.diffusion)},
           {"advection", array_json(spec.advection, vec3_to_json)},
           {"reaction", std::move(reaction)},
           {"obstacles",
            array_json(spec.obstacles, [](std::uint8_t solid) { return unsigned_json(solid); })},
           {"velocity_field", std::move(velocity_field)},
           {"integration", name_json(signal_integration_kind_name(spec.integration))},
           {"solver", JsonValue::object({
                          {"max_iterations", unsigned_json(spec.solver.max_iterations)},
                          {"absolute_tolerance", JsonValue::number(spec.solver.absolute_tolerance)},
                          {"relative_tolerance", JsonValue::number(spec.solver.relative_tolerance)},
                      })},
           {"boundaries", JsonValue::object({
                              {"x_lower", boundary_json(spec.x_lower)},
                              {"x_upper", boundary_json(spec.x_upper)},
                              {"y_lower", boundary_json(spec.y_lower)},
                              {"y_upper", boundary_json(spec.y_upper)},
                              {"z_lower", boundary_json(spec.z_lower)},
                              {"z_upper", boundary_json(spec.z_upper)},
                          })},
       })},
      {"levels", floats_to_json(grid->levels)},
  });
}

[[nodiscard]] JsonValue simulation_json(const SimulationCheckpoint& checkpoint) {
  const auto& constraints = checkpoint.constraints;
  const auto& plan = checkpoint.species_rate_plan;
  JsonValue coupled;
  if (checkpoint.coupled_rate_plan.has_value()) {
    const auto& coupled_plan = *checkpoint.coupled_rate_plan;
    coupled = JsonValue::object({
        {"species_count", unsigned_json(coupled_plan.species_count())},
        {"signal_count", unsigned_json(coupled_plan.signal_count())},
        {"instructions", array_json(coupled_plan.instructions(), instruction_json)},
        {"species_outputs", array_json(coupled_plan.species_outputs(), unsigned_json)},
        {"signal_outputs", array_json(coupled_plan.signal_outputs(), unsigned_json)},
    });
  }
  return JsonValue::object({
      {"time", JsonValue::number(checkpoint.time)},
      {"world", JsonValue::object({
                    {"species_count", unsigned_json(checkpoint.world.species_count)},
                    {"next_id", unsigned_json(checkpoint.world.next_id)},
                    {"cells", array_json(checkpoint.world.cells, cell_json)},
                    {"lineage", array_json(checkpoint.world.lineage,
                                           [](const LineageEntry& entry) {
                                             return JsonValue::object({
                                                 {"child", unsigned_json(entry.child)},
                                                 {"parent", unsigned_json(entry.parent)},
                                             });
                                           })},
                })},
      {"constraints",
       JsonValue::object({
           {"next_id", unsigned_json(constraints.next_id)},
           {"planes", array_json(constraints.planes,
                                 [](const PlaneConstraint& plane) {
                                   return JsonValue::object({
                                       {"id", unsigned_json(plane.id)},
                                       {"point", vec3_to_json(plane.point)},
                                       {"inward_normal", vec3_to_json(plane.inward_normal)},
                                       {"coefficient", JsonValue::number(plane.coefficient)},
                                   });
                                 })},
           {"spheres", array_json(constraints.spheres,
                                  [](const SphereConstraint& sphere) {
                                    return JsonValue::object({
                                        {"id", unsigned_json(sphere.id)},
                                        {"center", vec3_to_json(sphere.center)},
                                        {"radius", JsonValue::number(sphere.radius)},
                                        {"coefficient", JsonValue::number(sphere.coefficient)},
                                        {"allowed_region",
                                         name_json(constraint_region_name(sphere.allowed_region))},
                                    });
                                  })},
           {"boxes", array_json(constraints.boxes,
                                [](const BoxConstraint& box) {
                                  return JsonValue::object({
                                      {"id", unsigned_json(box.id)},
                                      {"center", vec3_to_json(box.center)},
                                      {"half_extents", vec3_to_json(box.half_extents)},
                                      {"coefficient", JsonValue::number(box.coefficient)},
                                      {"allowed_region",
                                       name_json(constraint_region_name(box.allowed_region))},
                                  });
                                })},
           {"cylinders", array_json(constraints.cylinders,
                                    [](const CylinderConstraint& cylinder) {
                                      return JsonValue::object({
                                          {"id", unsigned_json(cylinder.id)},
                                          {"center", vec3_to_json(cylinder.center)},
                                          {"radius", JsonValue::number(cylinder.radius)},
                                          {"half_height", JsonValue::number(cylinder.half_height)},
                                          {"coefficient", JsonValue::number(cylinder.coefficient)},
                                          {"allowed_region", name_json(constraint_region_name(
                                                                 cylinder.allowed_region))},
                                      });
                                    })},
       })},
      {"species_rate_plan", JsonValue::object({
                                {"species_count", unsigned_json(plan.species_count())},
                                {"instructions", array_json(plan.instructions(), instruction_json)},
                                {"outputs", array_json(plan.outputs(), unsigned_json)},
                            })},
      {"signal_grid", signal_grid_json(checkpoint.signal_grid)},
      {"coupled_rate_plan", std::move(coupled)},
  });
}

[[nodiscard]] std::uint32_t decode_uint32(const JsonValue& value, const JsonPath& path) {
  return static_cast<std::uint32_t>(expect_unsigned(value, path, 0, uint32_max));
}

[[nodiscard]] std::uint64_t decode_id(const JsonValue& value, const JsonPath& path) {
  return expect_unsigned(value, path, 1, uint64_max);
}

[[nodiscard]] ConstraintRegion decode_region(const JsonValue& value, const JsonPath& path) {
  const auto& name = expect_string(value, path);
  const auto region = parse_constraint_region(name);
  if (!region.has_value()) {
    fail(path, "unknown constraint region '" + name + "'");
  }
  return *region;
}

[[nodiscard]] CellSnapshot decode_cell(const JsonValue& value, const JsonPath& path) {
  expect_object(value, path,
                {"id", "slot", "position", "direction", "length", "radius", "growth_rate",
                 "cell_type", "fixed", "species"});
  CellSnapshot cell;
  cell.id = decode_id(*value.find("id"), path["id"]);
  cell.slot = decode_uint32(*value.find("slot"), path["slot"]);
  cell.position = expect_vec3(*value.find("position"), path["position"]);
  cell.direction = expect_vec3(*value.find("direction"), path["direction"]);
  cell.length = expect_float(*value.find("length"), path["length"]);
  cell.radius = expect_float(*value.find("radius"), path["radius"]);
  cell.growth_rate = expect_float(*value.find("growth_rate"), path["growth_rate"]);
  cell.cell_type = expect_int32(*value.find("cell_type"), path["cell_type"]);
  cell.fixed = expect_boolean(*value.find("fixed"), path["fixed"]);
  cell.species = expect_floats(*value.find("species"), path["species"]);
  return cell;
}

[[nodiscard]] RateInstruction decode_instruction(const JsonValue& value, const JsonPath& path) {
  expect_object(value, path, {"operation", "first", "second", "third", "value"});
  const auto& name = expect_string(*value.find("operation"), path["operation"]);
  const auto operation = parse_rate_op(name);
  if (!operation.has_value()) {
    fail(path["operation"], "unknown rate operation '" + name + "'");
  }
  return {
      .operation = *operation,
      .first = decode_uint32(*value.find("first"), path["first"]),
      .second = decode_uint32(*value.find("second"), path["second"]),
      .third = decode_uint32(*value.find("third"), path["third"]),
      .value = expect_float(*value.find("value"), path["value"]),
  };
}

[[nodiscard]] GridBoundary decode_boundary(const JsonValue& value, const JsonPath& path) {
  expect_object(value, path, {"kind", "values"});
  const auto& name = expect_string(*value.find("kind"), path["kind"]);
  const auto kind = parse_grid_boundary_kind(name);
  if (!kind.has_value()) {
    fail(path["kind"], "unknown grid boundary kind '" + name + "'");
  }
  return {.kind = *kind, .values = expect_floats(*value.find("values"), path["values"])};
}

[[nodiscard]] std::optional<SignalGridCheckpoint> decode_signal_grid(const JsonValue& value,
                                                                     const JsonPath& path) {
  if (value.is_null()) {
    return std::nullopt;
  }
  expect_object(value, path, {"spec", "levels"});
  const auto& spec_value = *value.find("spec");
  const auto spec_path = path["spec"];
  expect_object(spec_value, spec_path,
                {"signal_count", "shape", "origin", "spacing", "diffusion", "advection", "reaction",
                 "obstacles", "velocity_field", "integration", "solver", "boundaries"});

  SignalGridCheckpoint grid;
  auto& spec = grid.spec;
  spec.signal_count = static_cast<std::uint32_t>(
      expect_unsigned(*spec_value.find("signal_count"), spec_path["signal_count"], 1, uint32_max));
  const auto shape_path = spec_path["shape"];
  const auto& shape = expect_array(*spec_value.find("shape"), shape_path);
  if (shape.size() != 3) {
    fail(shape_path, "expected exactly three dimensions");
  }
  spec.shape.x =
      static_cast<std::uint32_t>(expect_unsigned(shape[0], shape_path[0], 1, uint32_max));
  spec.shape.y =
      static_cast<std::uint32_t>(expect_unsigned(shape[1], shape_path[1], 1, uint32_max));
  spec.shape.z =
      static_cast<std::uint32_t>(expect_unsigned(shape[2], shape_path[2], 1, uint32_max));
  spec.origin = expect_vec3(*spec_value.find("origin"), spec_path["origin"]);
  spec.spacing = expect_vec3(*spec_value.find("spacing"), spec_path["spacing"]);
  spec.diffusion = expect_floats(*spec_value.find("diffusion"), spec_path["diffusion"]);
  spec.advection = decode_array(*spec_value.find("advection"), spec_path["advection"], expect_vec3);

  const auto& reaction = *spec_value.find("reaction");
  if (!reaction.is_null()) {
    const auto reaction_path = spec_path["reaction"];
    expect_object(reaction, reaction_path, {"source_rates", "loss_rates"});
    spec.reaction = SignalGridAffineReaction{
        .source_rates =
            expect_floats(*reaction.find("source_rates"), reaction_path["source_rates"]),
        .loss_rates = expect_floats(*reaction.find("loss_rates"), reaction_path["loss_rates"]),
    };
  }
  spec.obstacles =
      decode_array(*spec_value.find("obstacles"), spec_path["obstacles"],
                   [](const JsonValue& item, const JsonPath& item_path) {
                     return static_cast<std::uint8_t>(expect_unsigned(item, item_path, 0, 1));
                   });
  const auto& field = *spec_value.find("velocity_field");
  if (!field.is_null()) {
    const auto field_path = spec_path["velocity_field"];
    expect_object(field, field_path, {"x_faces", "y_faces", "z_faces"});
    spec.velocity_field = SignalGridVelocityField{
        .x_faces = expect_floats(*field.find("x_faces"), field_path["x_faces"]),
        .y_faces = expect_floats(*field.find("y_faces"), field_path["y_faces"]),
        .z_faces = expect_floats(*field.find("z_faces"), field_path["z_faces"]),
    };
  }

  const auto integration_path = spec_path["integration"];
  const auto& integration_name = expect_string(*spec_value.find("integration"), integration_path);
  const auto integration = parse_signal_integration_kind(integration_name);
  if (!integration.has_value()) {
    fail(integration_path, "unknown integration '" + integration_name + "'");
  }
  spec.integration = *integration;

  const auto& solver = *spec_value.find("solver");
  const auto solver_path = spec_path["solver"];
  expect_object(solver, solver_path,
                {"max_iterations", "absolute_tolerance", "relative_tolerance"});
  spec.solver.max_iterations = static_cast<std::uint32_t>(expect_unsigned(
      *solver.find("max_iterations"), solver_path["max_iterations"], 1, uint32_max));
  spec.solver.absolute_tolerance =
      expect_float(*solver.find("absolute_tolerance"), solver_path["absolute_tolerance"]);
  spec.solver.relative_tolerance =
      expect_float(*solver.find("relative_tolerance"), solver_path["relative_tolerance"]);

  const auto& boundaries = *spec_value.find("boundaries");
  const auto boundaries_path = spec_path["boundaries"];
  expect_object(boundaries, boundaries_path,
                {"x_lower", "x_upper", "y_lower", "y_upper", "z_lower", "z_upper"});
  const std::array<GridBoundary*, 6> targets{&spec.x_lower, &spec.x_upper, &spec.y_lower,
                                             &spec.y_upper, &spec.z_lower, &spec.z_upper};
  for (std::size_t index = 0; index < targets.size(); ++index) {
    const auto name = boundary_names[index];
    *targets[index] = decode_boundary(*boundaries.find(name), boundaries_path[name]);
  }

  grid.levels = expect_floats(*value.find("levels"), path["levels"]);
  return grid;
}

[[nodiscard]] SimulationCheckpoint decode_simulation(const JsonValue& value, const JsonPath& path) {
  expect_object(
      value, path,
      {"time", "world", "constraints", "species_rate_plan", "signal_grid", "coupled_rate_plan"});
  SimulationCheckpoint checkpoint;
  checkpoint.time = expect_number(*value.find("time"), path["time"]);

  const auto& world = *value.find("world");
  const auto world_path = path["world"];
  expect_object(world, world_path, {"species_count", "next_id", "cells", "lineage"});
  checkpoint.world.species_count = static_cast<std::size_t>(
      expect_unsigned(*world.find("species_count"), world_path["species_count"], 0,
                      std::numeric_limits<std::size_t>::max()));
  checkpoint.world.next_id = decode_id(*world.find("next_id"), world_path["next_id"]);
  checkpoint.world.cells = decode_array(*world.find("cells"), world_path["cells"], decode_cell);
  checkpoint.world.lineage =
      decode_array(*world.find("lineage"), world_path["lineage"],
                   [](const JsonValue& item, const JsonPath& item_path) {
                     expect_object(item, item_path, {"child", "parent"});
                     return LineageEntry{
                         .child = decode_id(*item.find("child"), item_path["child"]),
                         .parent = decode_id(*item.find("parent"), item_path["parent"]),
                     };
                   });

  const auto& constraints = *value.find("constraints");
  const auto constraints_path = path["constraints"];
  expect_object(constraints, constraints_path,
                {"next_id", "planes", "spheres", "boxes", "cylinders"});
  checkpoint.constraints.next_id =
      decode_id(*constraints.find("next_id"), constraints_path["next_id"]);
  checkpoint.constraints.planes = decode_array(
      *constraints.find("planes"), constraints_path["planes"],
      [](const JsonValue& item, const JsonPath& item_path) {
        expect_object(item, item_path, {"id", "point", "inward_normal", "coefficient"});
        return PlaneConstraint{
            .id = decode_id(*item.find("id"), item_path["id"]),
            .point = expect_vec3(*item.find("point"), item_path["point"]),
            .inward_normal = expect_vec3(*item.find("inward_normal"), item_path["inward_normal"]),
            .coefficient = expect_float(*item.find("coefficient"), item_path["coefficient"]),
        };
      });
  checkpoint.constraints.spheres = decode_array(
      *constraints.find("spheres"), constraints_path["spheres"],
      [](const JsonValue& item, const JsonPath& item_path) {
        expect_object(item, item_path, {"id", "center", "radius", "coefficient", "allowed_region"});
        return SphereConstraint{
            .id = decode_id(*item.find("id"), item_path["id"]),
            .center = expect_vec3(*item.find("center"), item_path["center"]),
            .radius = expect_float(*item.find("radius"), item_path["radius"]),
            .coefficient = expect_float(*item.find("coefficient"), item_path["coefficient"]),
            .allowed_region =
                decode_region(*item.find("allowed_region"), item_path["allowed_region"]),
        };
      });
  checkpoint.constraints.boxes = decode_array(
      *constraints.find("boxes"), constraints_path["boxes"],
      [](const JsonValue& item, const JsonPath& item_path) {
        expect_object(item, item_path,
                      {"id", "center", "half_extents", "coefficient", "allowed_region"});
        return BoxConstraint{
            .id = decode_id(*item.find("id"), item_path["id"]),
            .center = expect_vec3(*item.find("center"), item_path["center"]),
            .half_extents = expect_vec3(*item.find("half_extents"), item_path["half_extents"]),
            .coefficient = expect_float(*item.find("coefficient"), item_path["coefficient"]),
            .allowed_region =
                decode_region(*item.find("allowed_region"), item_path["allowed_region"]),
        };
      });
  checkpoint.constraints.cylinders = decode_array(
      *constraints.find("cylinders"), constraints_path["cylinders"],
      [](const JsonValue& item, const JsonPath& item_path) {
        expect_object(item, item_path,
                      {"id", "center", "radius", "half_height", "coefficient", "allowed_region"});
        return CylinderConstraint{
            .id = decode_id(*item.find("id"), item_path["id"]),
            .center = expect_vec3(*item.find("center"), item_path["center"]),
            .radius = expect_float(*item.find("radius"), item_path["radius"]),
            .half_height = expect_float(*item.find("half_height"), item_path["half_height"]),
            .coefficient = expect_float(*item.find("coefficient"), item_path["coefficient"]),
            .allowed_region =
                decode_region(*item.find("allowed_region"), item_path["allowed_region"]),
        };
      });

  const auto& plan = *value.find("species_rate_plan");
  const auto plan_path = path["species_rate_plan"];
  expect_object(plan, plan_path, {"species_count", "instructions", "outputs"});
  try {
    checkpoint.species_rate_plan = SpeciesRatePlan(
        static_cast<std::size_t>(expect_unsigned(*plan.find("species_count"),
                                                 plan_path["species_count"], 0,
                                                 std::numeric_limits<std::size_t>::max())),
        decode_array(*plan.find("instructions"), plan_path["instructions"], decode_instruction),
        decode_array(*plan.find("outputs"), plan_path["outputs"], decode_uint32));
  } catch (const JsonError&) {
    throw;
  } catch (const std::exception& error) {
    fail(plan_path, std::string("invalid species rate plan: ") + error.what());
  }

  checkpoint.signal_grid = decode_signal_grid(*value.find("signal_grid"), path["signal_grid"]);

  const auto& coupled = *value.find("coupled_rate_plan");
  if (!coupled.is_null()) {
    const auto coupled_path = path["coupled_rate_plan"];
    expect_object(
        coupled, coupled_path,
        {"species_count", "signal_count", "instructions", "species_outputs", "signal_outputs"});
    try {
      checkpoint.coupled_rate_plan = CoupledRatePlan(
          static_cast<std::size_t>(expect_unsigned(*coupled.find("species_count"),
                                                   coupled_path["species_count"], 0,
                                                   std::numeric_limits<std::size_t>::max())),
          static_cast<std::size_t>(expect_unsigned(*coupled.find("signal_count"),
                                                   coupled_path["signal_count"], 1,
                                                   std::numeric_limits<std::size_t>::max())),
          decode_array(*coupled.find("instructions"), coupled_path["instructions"],
                       decode_instruction),
          decode_array(*coupled.find("species_outputs"), coupled_path["species_outputs"],
                       decode_uint32),
          decode_array(*coupled.find("signal_outputs"), coupled_path["signal_outputs"],
                       decode_uint32));
    } catch (const JsonError&) {
      throw;
    } catch (const std::exception& error) {
      fail(coupled_path, std::string("invalid coupled rate plan: ") + error.what());
    }
  }
  return checkpoint;
}

void validate_state(const SimulationCheckpoint& checkpoint) {
  try {
    checkpoint.validate();
  } catch (const std::exception& error) {
    throw CheckpointError(std::string("checkpoint state is invalid: ") + error.what());
  }
}

void verify_digest(const JsonValue& integrity, std::string_view member, const JsonValue& payload,
                   std::string_view message) {
  const JsonPath root;
  const auto integrity_path = root["integrity"];
  const auto path = integrity_path[member];
  const auto& expected = expect_string(*integrity.find(member), path);
  if (!digests_equal(sha256_hex(canonical_json(payload)), expected)) {
    fail(path, message);
  }
}

[[nodiscard]] CheckpointDocument decode_document(const JsonValue& root) {
  const JsonPath path;
  if (!root.is_object()) {
    fail(path, "expected an object");
  }
  const auto* version = root.find("version");
  if (version == nullptr) {
    fail(path, "missing keys ['version']");
  }
  const auto found_version = expect_unsigned(*version, path["version"], 0, uint32_max);
  if (found_version != checkpoint_format_version) {
    fail(path["version"], "unsupported checkpoint version " + std::to_string(found_version));
  }
  expect_object(root, path,
                {"format", "version", "producer", "source_backend", "provenance", "integrity",
                 "simulation", "controller", "channel_metadata"});
  if (expect_string(*root.find("format"), path["format"]) != checkpoint_format) {
    fail(path["format"], "not a MicroSimulator checkpoint");
  }
  const auto& producer = *root.find("producer");
  const auto producer_path = path["producer"];
  expect_object(producer, producer_path, {"name", "version"});
  static_cast<void>(expect_string(*producer.find("name"), producer_path["name"]));
  static_cast<void>(expect_string(*producer.find("version"), producer_path["version"]));

  CheckpointDocument document;
  const auto& backend = *root.find("source_backend");
  const auto backend_path = path["source_backend"];
  expect_object(backend, backend_path, {"kind", "name", "device", "device_index", "native"});
  const auto& kind_name = expect_string(*backend.find("kind"), backend_path["kind"]);
  const auto kind = parse_backend_kind(kind_name);
  if (!kind.has_value()) {
    fail(backend_path["kind"], "unknown backend kind '" + kind_name + "'");
  }
  document.source_backend = BackendInfo{
      .kind = *kind,
      .name = expect_string(*backend.find("name"), backend_path["name"]),
      .device = expect_string(*backend.find("device"), backend_path["device"]),
      .device_index = decode_uint32(*backend.find("device_index"), backend_path["device_index"]),
      .native = expect_boolean(*backend.find("native"), backend_path["native"]),
  };

  document.provenance = *root.find("provenance");
  if (!document.provenance.is_object()) {
    fail(path["provenance"], "expected an object");
  }

  const auto& integrity = *root.find("integrity");
  const auto integrity_path = path["integrity"];
  expect_object(integrity, integrity_path,
                {"algorithm", "simulation", "controller", "channel_metadata"});
  if (expect_string(*integrity.find("algorithm"), integrity_path["algorithm"]) != "sha256") {
    fail(integrity_path["algorithm"], "unsupported integrity algorithm");
  }
  verify_digest(integrity, "simulation", *root.find("simulation"), "state digest does not match");
  verify_digest(integrity, "controller", *root.find("controller"),
                "controller digest does not match");
  verify_digest(integrity, "channel_metadata", *root.find("channel_metadata"),
                "channel metadata digest does not match");

  document.controller = *root.find("controller");
  document.simulation = decode_simulation(*root.find("simulation"), path["simulation"]);
  validate_state(document.simulation);
  document.channel_metadata = channel_metadata_from_json(
      *root.find("channel_metadata"), document.simulation.world.species_count,
      document.simulation.signal_grid.has_value()
          ? document.simulation.signal_grid->spec.signal_count
          : 0);
  return document;
}

}  // namespace

std::string encode_checkpoint(const CheckpointDocument& document) {
  validate_state(document.simulation);
  if (!document.provenance.is_object()) {
    throw CheckpointError("checkpoint provenance must be a JSON object");
  }
  const auto& simulation = document.simulation;
  JsonValue labels;
  try {
    labels = channel_metadata_to_json(
        document.channel_metadata, simulation.world.species_count,
        simulation.signal_grid.has_value() ? simulation.signal_grid->spec.signal_count : 0);
  } catch (const ChannelMetadataError& error) {
    throw CheckpointError(error.what());
  }
  auto state = simulation_json(simulation);
  auto integrity = JsonValue::object({
      {"algorithm", JsonValue::string("sha256")},
      {"simulation", JsonValue::string(sha256_hex(canonical_json(state)))},
      {"controller", JsonValue::string(sha256_hex(canonical_json(document.controller)))},
      {"channel_metadata", JsonValue::string(sha256_hex(canonical_json(labels)))},
  });
  const auto& backend = document.source_backend;
  auto root = JsonValue::object({
      {"format", name_json(checkpoint_format)},
      {"version", unsigned_json(checkpoint_format_version)},
      {"producer", detail::producer_json()},
      {"source_backend", JsonValue::object({
                             {"kind", name_json(backend_kind_name(backend.kind))},
                             {"name", JsonValue::string(backend.name)},
                             {"device", JsonValue::string(backend.device)},
                             {"device_index", unsigned_json(backend.device_index)},
                             {"native", JsonValue::boolean(backend.native)},
                         })},
      {"provenance", document.provenance},
      {"integrity", std::move(integrity)},
      {"simulation", std::move(state)},
      {"controller", document.controller},
      {"channel_metadata", std::move(labels)},
  });
  try {
    return pretty_json(root);
  } catch (const JsonError& error) {
    throw CheckpointError(std::string("checkpoint cannot be encoded: ") + error.what());
  }
}

CheckpointDocument parse_checkpoint(std::string_view text) {
  if (text.empty()) {
    throw CheckpointError("checkpoint is empty");
  }
  if (text.size() > max_checkpoint_bytes) {
    throw CheckpointError("checkpoint exceeds the " + std::to_string(max_checkpoint_bytes) +
                          "-byte limit");
  }
  JsonValue root;
  try {
    root = parse_json(text);
  } catch (const JsonError& error) {
    throw CheckpointError(std::string("checkpoint is not valid JSON: ") + error.what());
  }
  try {
    return decode_document(root);
  } catch (const CheckpointError&) {
    throw;
  } catch (const JsonError& error) {
    throw CheckpointError(error.what());
  } catch (const ChannelMetadataError& error) {
    throw CheckpointError(error.what());
  }
}

void save_checkpoint(const std::filesystem::path& path, const CheckpointDocument& document) {
  const auto encoded = encode_checkpoint(document);
  if (encoded.size() > max_checkpoint_bytes) {
    throw CheckpointError("checkpoint exceeds the " + std::to_string(max_checkpoint_bytes) +
                          "-byte limit");
  }
  try {
    write_file_atomically(path, encoded);
  } catch (const FileError& error) {
    throw CheckpointError("could not write checkpoint " + path_text(path) + ": " + error.what());
  }
}

CheckpointDocument load_checkpoint(const std::filesystem::path& path) {
  std::string encoded;
  try {
    encoded = read_file_prefix(path, max_checkpoint_bytes + 1);
  } catch (const FileError& error) {
    throw CheckpointError("could not read checkpoint " + path_text(path) + ": " + error.what());
  }
  return parse_checkpoint(encoded);
}

}  // namespace microsimulator
