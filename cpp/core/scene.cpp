#include "microsimulator/scene.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "core/json_reader.hpp"
#include "core/producer.hpp"
#include "microsimulator/atomic_file.hpp"
#include "microsimulator/names.hpp"
#include "microsimulator/sha256.hpp"
#include "microsimulator/simulation.hpp"

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
using detail::floats_to_json;
using detail::JsonPath;
using detail::name_json;
using detail::uint32_max;

[[noreturn]] void scene_fail(const JsonPath& path, std::string_view message) {
  throw SceneError(path.str() + ": " + std::string(message));
}

void check_channel_budget(std::uint64_t count, const JsonPath& path) {
  if (count > max_scene_channels) {
    scene_fail(path, "exceeds scene presentation channel budget of " +
                         std::to_string(max_scene_channels) + " per group");
  }
}

[[nodiscard]] std::array<const SceneGridBoundary*, 6> boundaries_of(const SceneSignalGrid& grid) {
  return {&grid.x_lower, &grid.x_upper, &grid.y_lower, &grid.y_upper, &grid.z_lower, &grid.z_upper};
}

// The number of levels a grid declares, or an empty optional when the product
// does not fit in 64 bits.
[[nodiscard]] std::optional<std::uint64_t> level_count(const SceneSignalGrid& grid) noexcept {
  std::uint64_t count = grid.signal_count;
  for (const auto dimension : grid.shape) {
    if (dimension != 0 && count > std::numeric_limits<std::uint64_t>::max() / dimension) {
      return std::nullopt;
    }
    count *= dimension;
  }
  return count;
}

void check_finite(float value, const JsonPath& path) {
  if (!std::isfinite(value)) {
    scene_fail(path, "number must be finite");
  }
}

void check_vector(const SceneVector& vector, const JsonPath& path) {
  for (std::size_t component = 0; component < vector.size(); ++component) {
    check_finite(vector[component], path[component]);
  }
}

void check_positive(float value, const JsonPath& path) {
  check_finite(value, path);
  if (value <= 0.0F) {
    scene_fail(path, "must be positive");
  }
}

void check_normalized(const SceneVector& vector, const JsonPath& path) {
  double squared = 0.0;
  for (const float component : vector) {
    squared += static_cast<double>(component) * static_cast<double>(component);
  }
  if (std::abs(std::sqrt(squared) - 1.0) > 1.0e-5) {
    scene_fail(path, "must be normalized");
  }
}

void check_boundary(const SceneGridBoundary& boundary, std::uint32_t signal_count,
                    const JsonPath& path) {
  const std::size_t expected = boundary.kind == GridBoundaryKind::fixed ? signal_count : 0;
  const auto values_path = path["values"];
  if (boundary.values.size() != expected) {
    scene_fail(values_path, "expected " + std::to_string(expected) + " values for " +
                                std::string(grid_boundary_kind_name(boundary.kind)) + " boundary");
  }
  for (std::size_t index = 0; index < boundary.values.size(); ++index) {
    check_finite(boundary.values[index], values_path[index]);
  }
}

[[nodiscard]] SceneVector to_scene_vector(const Vec3& value) noexcept {
  return {value.x, value.y, value.z};
}

[[nodiscard]] SceneGridBoundary to_scene_boundary(const GridBoundary& boundary) {
  return {.kind = boundary.kind, .values = boundary.values};
}

[[nodiscard]] JsonValue id_json(std::uint64_t id) { return JsonValue::string(std::to_string(id)); }

[[nodiscard]] JsonValue vector_json(const SceneVector& value) {
  return JsonValue::array(
      {JsonValue::number(value[0]), JsonValue::number(value[1]), JsonValue::number(value[2])});
}

[[nodiscard]] JsonValue boundary_json(const SceneGridBoundary& boundary) {
  return JsonValue::object({
      {"kind", name_json(grid_boundary_kind_name(boundary.kind))},
      {"values", floats_to_json(boundary.values)},
  });
}

[[nodiscard]] JsonValue frame_json(const SceneFrame& frame) {
  JsonValue grid;
  if (frame.signal_grid.has_value()) {
    const auto& value = *frame.signal_grid;
    grid = JsonValue::object({
        {"signal_count", JsonValue::integer(std::uint64_t{value.signal_count})},
        {"shape", array_json(value.shape,
                             [](std::uint32_t dimension) {
                               return JsonValue::integer(std::uint64_t{dimension});
                             })},
        {"origin", vector_json(value.origin)},
        {"spacing", vector_json(value.spacing)},
        {"boundaries", JsonValue::object({
                           {"x_lower", boundary_json(value.x_lower)},
                           {"x_upper", boundary_json(value.x_upper)},
                           {"y_lower", boundary_json(value.y_lower)},
                           {"y_upper", boundary_json(value.y_upper)},
                           {"z_lower", boundary_json(value.z_lower)},
                           {"z_upper", boundary_json(value.z_upper)},
                       })},
        {"levels", floats_to_json(value.levels)},
    });
  }
  return JsonValue::object({
      {"time", JsonValue::number(frame.time)},
      {"backend",
       JsonValue::object({
           {"kind", name_json(backend_kind_name(frame.backend.kind))},
           {"name", JsonValue::string(frame.backend.name)},
           {"device", JsonValue::string(frame.backend.device)},
           {"device_index", JsonValue::integer(std::uint64_t{frame.backend.device_index})},
           {"native", JsonValue::boolean(frame.backend.native)},
       })},
      {"species_count", JsonValue::integer(std::uint64_t{frame.species_count})},
      {"cells", array_json(frame.cells,
                           [](const SceneCell& cell) {
                             return JsonValue::object({
                                 {"id", id_json(cell.id)},
                                 {"parent_id", cell.parent_id.has_value() ? id_json(*cell.parent_id)
                                                                          : JsonValue{}},
                                 {"slot", JsonValue::integer(std::uint64_t{cell.slot})},
                                 {"position", vector_json(cell.position)},
                                 {"direction", vector_json(cell.direction)},
                                 {"length", JsonValue::number(cell.length)},
                                 {"radius", JsonValue::number(cell.radius)},
                                 {"growth_rate", JsonValue::number(cell.growth_rate)},
                                 {"cell_type", JsonValue::integer(std::int64_t{cell.cell_type})},
                                 {"fixed", JsonValue::boolean(cell.fixed)},
                                 {"species", floats_to_json(cell.species)},
                             });
                           })},
      {"constraints",
       JsonValue::object({
           {"planes", array_json(frame.constraints.planes,
                                 [](const ScenePlaneConstraint& plane) {
                                   return JsonValue::object({
                                       {"id", id_json(plane.id)},
                                       {"point", vector_json(plane.point)},
                                       {"inward_normal", vector_json(plane.inward_normal)},
                                       {"coefficient", JsonValue::number(plane.coefficient)},
                                   });
                                 })},
           {"spheres", array_json(frame.constraints.spheres,
                                  [](const SceneSphereConstraint& sphere) {
                                    return JsonValue::object({
                                        {"id", id_json(sphere.id)},
                                        {"center", vector_json(sphere.center)},
                                        {"radius", JsonValue::number(sphere.radius)},
                                        {"coefficient", JsonValue::number(sphere.coefficient)},
                                        {"allowed_region",
                                         name_json(constraint_region_name(sphere.allowed_region))},
                                    });
                                  })},
           {"boxes", array_json(frame.constraints.boxes,
                                [](const SceneBoxConstraint& box) {
                                  return JsonValue::object({
                                      {"id", id_json(box.id)},
                                      {"center", vector_json(box.center)},
                                      {"half_extents", vector_json(box.half_extents)},
                                      {"coefficient", JsonValue::number(box.coefficient)},
                                      {"allowed_region",
                                       name_json(constraint_region_name(box.allowed_region))},
                                  });
                                })},
           {"cylinders", array_json(frame.constraints.cylinders,
                                    [](const SceneCylinderConstraint& cylinder) {
                                      return JsonValue::object({
                                          {"id", id_json(cylinder.id)},
                                          {"center", vector_json(cylinder.center)},
                                          {"radius", JsonValue::number(cylinder.radius)},
                                          {"half_height", JsonValue::number(cylinder.half_height)},
                                          {"coefficient", JsonValue::number(cylinder.coefficient)},
                                          {"allowed_region", name_json(constraint_region_name(
                                                                 cylinder.allowed_region))},
                                      });
                                    })},
       })},
      {"signal_grid", std::move(grid)},
      {"channel_metadata",
       channel_metadata_to_json(
           frame.channel_metadata, frame.species_count,
           frame.signal_grid.has_value() ? frame.signal_grid->signal_count : 0)},
  });
}

[[nodiscard]] std::uint64_t decode_id(const JsonValue& value, const JsonPath& path) {
  const std::string_view text = expect_string(value, path);
  bool canonical = !text.empty() && text.front() != '0';
  for (const char digit : text) {
    canonical = canonical && digit >= '0' && digit <= '9';
  }
  if (!canonical) {
    detail::fail(path, "expected a canonical positive decimal uint64 string");
  }
  std::uint64_t id = 0;
  if (!detail::parse_uint64(text, id)) {
    detail::fail(path, "identifier is outside the positive uint64 range");
  }
  return id;
}

[[nodiscard]] SceneVector decode_vector(const JsonValue& value, const JsonPath& path) {
  const auto& items = expect_array(value, path);
  if (items.size() != 3) {
    detail::fail(path, "expected exactly three values");
  }
  return {expect_float(items[0], path[0]), expect_float(items[1], path[1]),
          expect_float(items[2], path[2])};
}

[[nodiscard]] ConstraintRegion decode_region(const JsonValue& value, const JsonPath& path) {
  const auto& name = expect_string(value, path);
  const auto region = parse_constraint_region(name);
  if (!region.has_value()) {
    detail::fail(path, "unknown region kind '" + name + "'");
  }
  return *region;
}

[[nodiscard]] std::uint32_t decode_channel_count(const JsonValue& value, const JsonPath& path,
                                                 std::uint64_t minimum) {
  const auto count = expect_unsigned(value, path, minimum, uint32_max);
  check_channel_budget(count, path);
  return static_cast<std::uint32_t>(count);
}

[[nodiscard]] SceneGridBoundary decode_boundary(const JsonValue& value, const JsonPath& path) {
  expect_object(value, path, {"kind", "values"});
  const auto& name = expect_string(*value.find("kind"), path["kind"]);
  const auto kind = parse_grid_boundary_kind(name);
  if (!kind.has_value()) {
    detail::fail(path["kind"], "unknown boundary kind '" + name + "'");
  }
  return {.kind = *kind, .values = expect_floats(*value.find("values"), path["values"])};
}

[[nodiscard]] std::optional<SceneSignalGrid> decode_signal_grid(const JsonValue& value,
                                                                const JsonPath& path) {
  if (value.is_null()) {
    return std::nullopt;
  }
  expect_object(value, path,
                {"signal_count", "shape", "origin", "spacing", "boundaries", "levels"});
  SceneSignalGrid grid;
  grid.signal_count = decode_channel_count(*value.find("signal_count"), path["signal_count"], 1);
  const auto shape_path = path["shape"];
  const auto& shape = expect_array(*value.find("shape"), shape_path);
  if (shape.size() != 3) {
    detail::fail(shape_path, "expected exactly three dimensions");
  }
  for (std::size_t index = 0; index < 3; ++index) {
    grid.shape[index] =
        static_cast<std::uint32_t>(expect_unsigned(shape[index], shape_path[index], 1, uint32_max));
  }
  grid.origin = decode_vector(*value.find("origin"), path["origin"]);
  grid.spacing = decode_vector(*value.find("spacing"), path["spacing"]);
  const auto& boundaries = *value.find("boundaries");
  const auto boundaries_path = path["boundaries"];
  expect_object(boundaries, boundaries_path,
                {"x_lower", "x_upper", "y_lower", "y_upper", "z_lower", "z_upper"});
  const std::array<SceneGridBoundary*, 6> targets{&grid.x_lower, &grid.x_upper, &grid.y_lower,
                                                  &grid.y_upper, &grid.z_lower, &grid.z_upper};
  for (std::size_t index = 0; index < targets.size(); ++index) {
    const auto name = boundary_names[index];
    *targets[index] = decode_boundary(*boundaries.find(name), boundaries_path[name]);
  }
  grid.levels = expect_floats(*value.find("levels"), path["levels"]);
  return grid;
}

[[nodiscard]] SceneFrame decode_frame(const JsonValue& value, const JsonPath& path) {
  expect_object(value, path,
                {"time", "backend", "species_count", "cells", "constraints", "signal_grid",
                 "channel_metadata"});
  SceneFrame frame;
  // Channel counts are bounded before any per-channel array is decoded.
  frame.species_count =
      decode_channel_count(*value.find("species_count"), path["species_count"], 0);
  frame.signal_grid = decode_signal_grid(*value.find("signal_grid"), path["signal_grid"]);
  frame.channel_metadata = channel_metadata_from_json(
      *value.find("channel_metadata"), frame.species_count,
      frame.signal_grid.has_value() ? frame.signal_grid->signal_count : 0);
  frame.time = expect_number(*value.find("time"), path["time"]);

  const auto& backend = *value.find("backend");
  const auto backend_path = path["backend"];
  expect_object(backend, backend_path, {"kind", "name", "device", "device_index", "native"});
  const auto& kind_name = expect_string(*backend.find("kind"), backend_path["kind"]);
  const auto kind = parse_backend_kind(kind_name);
  if (!kind.has_value()) {
    detail::fail(backend_path["kind"], "unknown backend kind '" + kind_name + "'");
  }
  frame.backend = {
      .kind = *kind,
      .name = expect_string(*backend.find("name"), backend_path["name"]),
      .device = expect_string(*backend.find("device"), backend_path["device"]),
      .device_index = static_cast<std::uint32_t>(expect_unsigned(
          *backend.find("device_index"), backend_path["device_index"], 0, uint32_max)),
      .native = expect_boolean(*backend.find("native"), backend_path["native"]),
  };

  frame.cells = decode_array(
      *value.find("cells"), path["cells"], [](const JsonValue& item, const JsonPath& item_path) {
        expect_object(item, item_path,
                      {"id", "parent_id", "slot", "position", "direction", "length", "radius",
                       "growth_rate", "cell_type", "fixed", "species"});
        SceneCell cell;
        cell.id = decode_id(*item.find("id"), item_path["id"]);
        const auto& parent = *item.find("parent_id");
        if (!parent.is_null()) {
          cell.parent_id = decode_id(parent, item_path["parent_id"]);
        }
        cell.slot = static_cast<Slot>(
            expect_unsigned(*item.find("slot"), item_path["slot"], 0, uint32_max - 1));
        cell.position = decode_vector(*item.find("position"), item_path["position"]);
        cell.direction = decode_vector(*item.find("direction"), item_path["direction"]);
        cell.length = expect_float(*item.find("length"), item_path["length"]);
        cell.radius = expect_float(*item.find("radius"), item_path["radius"]);
        cell.growth_rate = expect_float(*item.find("growth_rate"), item_path["growth_rate"]);
        cell.cell_type = expect_int32(*item.find("cell_type"), item_path["cell_type"]);
        cell.fixed = expect_boolean(*item.find("fixed"), item_path["fixed"]);
        cell.species = expect_floats(*item.find("species"), item_path["species"]);
        return cell;
      });

  const auto& constraints = *value.find("constraints");
  const auto constraints_path = path["constraints"];
  expect_object(constraints, constraints_path, {"planes", "spheres", "boxes", "cylinders"});
  frame.constraints.planes = decode_array(
      *constraints.find("planes"), constraints_path["planes"],
      [](const JsonValue& item, const JsonPath& item_path) {
        expect_object(item, item_path, {"id", "point", "inward_normal", "coefficient"});
        return ScenePlaneConstraint{
            .id = decode_id(*item.find("id"), item_path["id"]),
            .point = decode_vector(*item.find("point"), item_path["point"]),
            .inward_normal = decode_vector(*item.find("inward_normal"), item_path["inward_normal"]),
            .coefficient = expect_float(*item.find("coefficient"), item_path["coefficient"]),
        };
      });
  frame.constraints.spheres = decode_array(
      *constraints.find("spheres"), constraints_path["spheres"],
      [](const JsonValue& item, const JsonPath& item_path) {
        expect_object(item, item_path, {"id", "center", "radius", "coefficient", "allowed_region"});
        return SceneSphereConstraint{
            .id = decode_id(*item.find("id"), item_path["id"]),
            .center = decode_vector(*item.find("center"), item_path["center"]),
            .radius = expect_float(*item.find("radius"), item_path["radius"]),
            .coefficient = expect_float(*item.find("coefficient"), item_path["coefficient"]),
            .allowed_region =
                decode_region(*item.find("allowed_region"), item_path["allowed_region"]),
        };
      });
  frame.constraints.boxes = decode_array(
      *constraints.find("boxes"), constraints_path["boxes"],
      [](const JsonValue& item, const JsonPath& item_path) {
        expect_object(item, item_path,
                      {"id", "center", "half_extents", "coefficient", "allowed_region"});
        return SceneBoxConstraint{
            .id = decode_id(*item.find("id"), item_path["id"]),
            .center = decode_vector(*item.find("center"), item_path["center"]),
            .half_extents = decode_vector(*item.find("half_extents"), item_path["half_extents"]),
            .coefficient = expect_float(*item.find("coefficient"), item_path["coefficient"]),
            .allowed_region =
                decode_region(*item.find("allowed_region"), item_path["allowed_region"]),
        };
      });
  frame.constraints.cylinders = decode_array(
      *constraints.find("cylinders"), constraints_path["cylinders"],
      [](const JsonValue& item, const JsonPath& item_path) {
        expect_object(item, item_path,
                      {"id", "center", "radius", "half_height", "coefficient", "allowed_region"});
        return SceneCylinderConstraint{
            .id = decode_id(*item.find("id"), item_path["id"]),
            .center = decode_vector(*item.find("center"), item_path["center"]),
            .radius = expect_float(*item.find("radius"), item_path["radius"]),
            .half_height = expect_float(*item.find("half_height"), item_path["half_height"]),
            .coefficient = expect_float(*item.find("coefficient"), item_path["coefficient"]),
            .allowed_region =
                decode_region(*item.find("allowed_region"), item_path["allowed_region"]),
        };
      });
  return frame;
}

[[nodiscard]] SceneFrame decode_document(const JsonValue& root) {
  const JsonPath path;
  expect_object(root, path, {"format", "version", "producer", "integrity", "frame"});
  if (expect_string(*root.find("format"), path["format"]) != scene_format) {
    detail::fail(path["format"], "not a MicroSimulator scene");
  }
  const auto version = expect_unsigned(*root.find("version"), path["version"], 0, uint32_max);
  if (version != scene_format_version) {
    detail::fail(path["version"], "unsupported scene version " + std::to_string(version));
  }
  const auto& producer = *root.find("producer");
  const auto producer_path = path["producer"];
  expect_object(producer, producer_path, {"name", "version"});
  static_cast<void>(expect_string(*producer.find("name"), producer_path["name"]));
  static_cast<void>(expect_string(*producer.find("version"), producer_path["version"]));

  const auto& integrity = *root.find("integrity");
  const auto integrity_path = path["integrity"];
  expect_object(integrity, integrity_path, {"algorithm", "frame"});
  if (expect_string(*integrity.find("algorithm"), integrity_path["algorithm"]) != "sha256") {
    detail::fail(integrity_path["algorithm"], "unsupported integrity algorithm");
  }
  const auto& expected = expect_string(*integrity.find("frame"), integrity_path["frame"]);
  if (!digests_equal(sha256_hex(canonical_json(*root.find("frame"))), expected)) {
    detail::fail(integrity_path["frame"], "frame digest does not match");
  }
  auto frame = decode_frame(*root.find("frame"), path["frame"]);
  frame.validate();
  return frame;
}

}  // namespace

void SceneFrame::resolve_channels() {
  const JsonPath root;
  const auto frame_path = root["frame"];
  check_channel_budget(species_count, frame_path["species_count"]);
  if (signal_grid.has_value()) {
    const auto grid_path = frame_path["signal_grid"];
    check_channel_budget(signal_grid->signal_count, grid_path["signal_count"]);
  }
  channel_metadata = channel_metadata.resolved(
      species_count, signal_grid.has_value() ? signal_grid->signal_count : 0);
}

void SceneFrame::validate() const {
  const JsonPath root;
  const auto path = root["frame"];
  check_channel_budget(species_count, path["species_count"]);
  const auto grid_path = path["signal_grid"];
  if (signal_grid.has_value()) {
    if (signal_grid->signal_count == 0) {
      scene_fail(grid_path["signal_count"], "integer is outside [1, 4294967295]");
    }
    check_channel_budget(signal_grid->signal_count, grid_path["signal_count"]);
  }
  try {
    static_cast<void>(channel_metadata.resolved(
        species_count, signal_grid.has_value() ? signal_grid->signal_count : 0));
  } catch (const ChannelMetadataError& error) {
    throw SceneError(error.what());
  }
  if (!std::isfinite(time)) {
    scene_fail(path["time"], "number must be finite");
  }
  if (time < 0.0) {
    scene_fail(path["time"], "must be non-negative");
  }
  if (backend.name.empty() || backend.device.empty()) {
    scene_fail(path["backend"], "name and device must not be empty");
  }

  const auto cells_path = path["cells"];
  std::unordered_set<CellId> cell_ids;
  cell_ids.reserve(cells.size());
  for (std::size_t index = 0; index < cells.size(); ++index) {
    const auto& cell = cells[index];
    const auto cell_path = cells_path[index];
    if (cell.slot != index) {
      scene_fail(cell_path["slot"], "cells must be compact and ordered by slot");
    }
    if (cell.id == invalid_cell_id) {
      scene_fail(cell_path["id"], "integer is outside [1, 18446744073709551615]");
    }
    if (!cell_ids.insert(cell.id).second) {
      scene_fail(cell_path["id"], "duplicate cell identifier");
    }
    if (cell.parent_id.has_value()) {
      if (*cell.parent_id == invalid_cell_id) {
        scene_fail(cell_path["parent_id"], "integer is outside [1, 18446744073709551615]");
      }
      if (*cell.parent_id >= cell.id) {
        scene_fail(cell_path["parent_id"], "must precede the child identifier");
      }
    }
    check_vector(cell.position, cell_path["position"]);
    check_vector(cell.direction, cell_path["direction"]);
    check_normalized(cell.direction, cell_path["direction"]);
    check_finite(cell.length, cell_path["length"]);
    check_finite(cell.radius, cell_path["radius"]);
    if (cell.length < 0.0F) {
      scene_fail(cell_path["length"], "must be non-negative");
    }
    if (cell.radius <= 0.0F) {
      scene_fail(cell_path["radius"], "must be positive");
    }
    check_finite(cell.growth_rate, cell_path["growth_rate"]);
    const auto species_path = cell_path["species"];
    if (cell.species.size() != species_count) {
      scene_fail(species_path, "expected " + std::to_string(species_count) + " values");
    }
    for (std::size_t species = 0; species < cell.species.size(); ++species) {
      check_finite(cell.species[species], species_path[species]);
    }
  }

  const auto constraints_path = path["constraints"];
  std::unordered_set<ConstraintId> constraint_ids;
  const auto check_id = [&](ConstraintId id, const JsonPath& id_path) {
    if (id == invalid_constraint_id) {
      scene_fail(id_path, "integer is outside [1, 18446744073709551615]");
    }
    if (!constraint_ids.insert(id).second) {
      scene_fail(id_path, "duplicate constraint identifier");
    }
  };
  const auto planes_path = constraints_path["planes"];
  for (std::size_t index = 0; index < constraints.planes.size(); ++index) {
    const auto& plane = constraints.planes[index];
    const auto item_path = planes_path[index];
    check_id(plane.id, item_path["id"]);
    check_vector(plane.point, item_path["point"]);
    check_vector(plane.inward_normal, item_path["inward_normal"]);
    check_normalized(plane.inward_normal, item_path["inward_normal"]);
    check_positive(plane.coefficient, item_path["coefficient"]);
  }
  const auto spheres_path = constraints_path["spheres"];
  for (std::size_t index = 0; index < constraints.spheres.size(); ++index) {
    const auto& sphere = constraints.spheres[index];
    const auto item_path = spheres_path[index];
    check_id(sphere.id, item_path["id"]);
    check_vector(sphere.center, item_path["center"]);
    check_positive(sphere.radius, item_path["radius"]);
    check_positive(sphere.coefficient, item_path["coefficient"]);
  }
  const auto boxes_path = constraints_path["boxes"];
  for (std::size_t index = 0; index < constraints.boxes.size(); ++index) {
    const auto& box = constraints.boxes[index];
    const auto item_path = boxes_path[index];
    check_id(box.id, item_path["id"]);
    check_vector(box.center, item_path["center"]);
    check_vector(box.half_extents, item_path["half_extents"]);
    for (const float extent : box.half_extents) {
      if (extent <= 0.0F) {
        scene_fail(item_path["half_extents"], "values must be positive");
      }
    }
    check_positive(box.coefficient, item_path["coefficient"]);
  }
  const auto cylinders_path = constraints_path["cylinders"];
  for (std::size_t index = 0; index < constraints.cylinders.size(); ++index) {
    const auto& cylinder = constraints.cylinders[index];
    const auto item_path = cylinders_path[index];
    check_id(cylinder.id, item_path["id"]);
    check_vector(cylinder.center, item_path["center"]);
    check_positive(cylinder.radius, item_path["radius"]);
    check_positive(cylinder.half_height, item_path["half_height"]);
    check_positive(cylinder.coefficient, item_path["coefficient"]);
  }

  if (!signal_grid.has_value()) {
    return;
  }
  const auto& grid = *signal_grid;
  const auto shape_path = grid_path["shape"];
  for (std::size_t index = 0; index < grid.shape.size(); ++index) {
    if (grid.shape[index] == 0) {
      scene_fail(shape_path[index], "integer is outside [1, 4294967295]");
    }
  }
  check_vector(grid.origin, grid_path["origin"]);
  const auto spacing_path = grid_path["spacing"];
  for (std::size_t component = 0; component < grid.spacing.size(); ++component) {
    check_positive(grid.spacing[component], spacing_path[component]);
  }
  const auto boundaries_path = grid_path["boundaries"];
  const auto boundaries = boundaries_of(grid);
  for (std::size_t index = 0; index < boundaries.size(); ++index) {
    check_boundary(*boundaries[index], grid.signal_count, boundaries_path[boundary_names[index]]);
  }
  const auto levels_path = grid_path["levels"];
  const auto expected = level_count(grid);
  if (!expected.has_value() || grid.levels.size() != *expected) {
    scene_fail(levels_path, expected.has_value()
                                ? "expected " + std::to_string(*expected) + " values"
                                : std::string("declared level count is too large"));
  }
  for (std::size_t index = 0; index < grid.levels.size(); ++index) {
    check_finite(grid.levels[index], levels_path[index]);
  }
}

SceneFrame capture_scene(const Simulation& simulation, const ChannelMetadata& channel_metadata) {
  // Reject an oversized channel count before copying state or expanding labels.
  const JsonPath root;
  const auto frame_path = root["frame"];
  const auto grid_path = frame_path["signal_grid"];
  check_channel_budget(simulation.species_count(), frame_path["species_count"]);
  check_channel_budget(simulation.signal_count(), grid_path["signal_count"]);

  auto checkpoint = simulation.checkpoint();
  const auto backend = simulation.backend_info();
  std::unordered_map<CellId, CellId> parents;
  parents.reserve(checkpoint.world.lineage.size());
  for (const auto& entry : checkpoint.world.lineage) {
    parents.emplace(entry.child, entry.parent);
  }

  SceneFrame frame;
  frame.time = checkpoint.time;
  frame.backend = {
      .kind = backend.kind,
      .name = backend.name,
      .device = backend.device,
      .device_index = backend.device_index,
      .native = backend.native,
  };
  frame.species_count = static_cast<std::uint32_t>(checkpoint.world.species_count);
  frame.cells.reserve(checkpoint.world.cells.size());
  for (auto& cell : checkpoint.world.cells) {
    const auto parent = parents.find(cell.id);
    frame.cells.push_back({
        .id = cell.id,
        .parent_id = parent == parents.end() ? std::nullopt : std::optional<CellId>(parent->second),
        .slot = cell.slot,
        .position = to_scene_vector(cell.position),
        .direction = to_scene_vector(cell.direction),
        .length = cell.length,
        .radius = cell.radius,
        .growth_rate = cell.growth_rate,
        .cell_type = cell.cell_type,
        .fixed = cell.fixed,
        .species = std::move(cell.species),
    });
  }
  for (const auto& plane : checkpoint.constraints.planes) {
    frame.constraints.planes.push_back({
        .id = plane.id,
        .point = to_scene_vector(plane.point),
        .inward_normal = to_scene_vector(plane.inward_normal),
        .coefficient = plane.coefficient,
    });
  }
  for (const auto& sphere : checkpoint.constraints.spheres) {
    frame.constraints.spheres.push_back({
        .id = sphere.id,
        .center = to_scene_vector(sphere.center),
        .radius = sphere.radius,
        .coefficient = sphere.coefficient,
        .allowed_region = sphere.allowed_region,
    });
  }
  for (const auto& box : checkpoint.constraints.boxes) {
    frame.constraints.boxes.push_back({
        .id = box.id,
        .center = to_scene_vector(box.center),
        .half_extents = to_scene_vector(box.half_extents),
        .coefficient = box.coefficient,
        .allowed_region = box.allowed_region,
    });
  }
  for (const auto& cylinder : checkpoint.constraints.cylinders) {
    frame.constraints.cylinders.push_back({
        .id = cylinder.id,
        .center = to_scene_vector(cylinder.center),
        .radius = cylinder.radius,
        .half_height = cylinder.half_height,
        .coefficient = cylinder.coefficient,
        .allowed_region = cylinder.allowed_region,
    });
  }
  if (checkpoint.signal_grid.has_value()) {
    const auto& spec = checkpoint.signal_grid->spec;
    frame.signal_grid = SceneSignalGrid{
        .signal_count = spec.signal_count,
        .shape = {spec.shape.x, spec.shape.y, spec.shape.z},
        .origin = to_scene_vector(spec.origin),
        .spacing = to_scene_vector(spec.spacing),
        .x_lower = to_scene_boundary(spec.x_lower),
        .x_upper = to_scene_boundary(spec.x_upper),
        .y_lower = to_scene_boundary(spec.y_lower),
        .y_upper = to_scene_boundary(spec.y_upper),
        .z_lower = to_scene_boundary(spec.z_lower),
        .z_upper = to_scene_boundary(spec.z_upper),
        .levels = std::move(checkpoint.signal_grid->levels),
    };
  }
  frame.channel_metadata =
      channel_metadata.resolved(simulation.species_count(), simulation.signal_count());
  frame.validate();
  return frame;
}

std::string encode_scene(const SceneFrame& frame) {
  frame.validate();
  auto payload = frame_json(frame);
  auto digest = sha256_hex(canonical_json(payload));
  return pretty_json(JsonValue::object({
      {"format", name_json(scene_format)},
      {"version", JsonValue::integer(std::uint64_t{scene_format_version})},
      {"producer", detail::producer_json()},
      {"integrity", JsonValue::object({
                        {"algorithm", JsonValue::string("sha256")},
                        {"frame", JsonValue::string(std::move(digest))},
                    })},
      {"frame", std::move(payload)},
  }));
}

SceneFrame parse_scene(std::string_view text) {
  if (text.empty()) {
    throw SceneError("scene is empty");
  }
  if (text.size() > max_scene_bytes) {
    throw SceneError("scene exceeds the " + std::to_string(max_scene_bytes) + "-byte limit");
  }
  JsonValue root;
  try {
    root = parse_json(text);
  } catch (const JsonError& error) {
    throw SceneError(std::string("scene is not valid JSON: ") + error.what());
  }
  try {
    return decode_document(root);
  } catch (const SceneError&) {
    throw;
  } catch (const JsonError& error) {
    throw SceneError(error.what());
  } catch (const ChannelMetadataError& error) {
    throw SceneError(error.what());
  }
}

void save_scene(const std::filesystem::path& path, const SceneFrame& frame) {
  const auto encoded = encode_scene(frame);
  if (encoded.size() > max_scene_bytes) {
    throw SceneError("scene exceeds the " + std::to_string(max_scene_bytes) + "-byte limit");
  }
  try {
    write_file_atomically(path, encoded);
  } catch (const FileError& error) {
    throw SceneError("could not write scene " + path_text(path) + ": " + error.what());
  }
}

SceneFrame load_scene(const std::filesystem::path& path) {
  std::string encoded;
  try {
    encoded = read_file_prefix(path, max_scene_bytes + 1);
  } catch (const FileError& error) {
    throw SceneError("could not read scene " + path_text(path) + ": " + error.what());
  }
  return parse_scene(encoded);
}

}  // namespace microsimulator
