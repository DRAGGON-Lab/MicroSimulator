#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "microsimulator/channels.hpp"
#include "microsimulator/checkpoint_file.hpp"
#include "microsimulator/json.hpp"
#include "microsimulator/names.hpp"
#include "microsimulator/scene.hpp"
#include "microsimulator/simulation.hpp"
#include "native_call.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using microsimulator::ChannelMetadata;
using microsimulator::ChannelMetadataError;
using microsimulator::CheckpointDocument;
using microsimulator::CheckpointError;
using microsimulator::JsonError;
using microsimulator::JsonValue;
using microsimulator::SceneBackend;
using microsimulator::SceneBoxConstraint;
using microsimulator::SceneCell;
using microsimulator::SceneConstraints;
using microsimulator::SceneCylinderConstraint;
using microsimulator::SceneError;
using microsimulator::SceneFrame;
using microsimulator::SceneGridBoundary;
using microsimulator::ScenePlaneConstraint;
using microsimulator::SceneSignalGrid;
using microsimulator::SceneSphereConstraint;
using microsimulator::SceneVector;
using microsimulator::Simulation;

constexpr std::size_t max_json_depth = 256;

// Converts Python data made of None, bool, int, float, str, list, tuple, and
// dict with string keys. An int of any magnitude keeps its exact value.
[[nodiscard]] JsonValue json_from_python(nb::handle value, std::size_t depth = 0) {
  if (depth > max_json_depth) {
    throw JsonError("nesting is too deep");
  }
  if (value.is_none()) {
    return JsonValue{};
  }
  if (PyBool_Check(value.ptr())) {
    return JsonValue::boolean(value.ptr() == Py_True);
  }
  if (PyLong_Check(value.ptr())) {
    const auto digits = nb::steal<nb::object>(PyNumber_ToBase(value.ptr(), 10));
    if (!digits.is_valid()) {
      PyErr_Clear();
      throw JsonError("integer cannot be written in decimal");
    }
    return JsonValue::integer_token(nb::cast<std::string>(digits));
  }
  if (PyFloat_Check(value.ptr())) {
    return JsonValue::number(PyFloat_AsDouble(value.ptr()));
  }
  if (PyUnicode_Check(value.ptr())) {
    Py_ssize_t size = 0;
    const char* text = PyUnicode_AsUTF8AndSize(value.ptr(), &size);
    if (text == nullptr) {
      PyErr_Clear();
      throw JsonError("string is not valid Unicode text");
    }
    return JsonValue::string(std::string(text, static_cast<std::size_t>(size)));
  }
  if (PyList_Check(value.ptr()) || PyTuple_Check(value.ptr())) {
    JsonValue::Array items;
    items.reserve(nb::len(value));
    for (const auto item : value) {
      items.push_back(json_from_python(item, depth + 1));
    }
    return JsonValue::array(std::move(items));
  }
  if (PyDict_Check(value.ptr())) {
    JsonValue::Object members;
    for (const auto [key, item] : nb::borrow<nb::dict>(value)) {
      if (!PyUnicode_Check(key.ptr())) {
        throw JsonError("object keys must be strings");
      }
      members.push_back(
          {json_from_python(key, depth + 1).as_string(), json_from_python(item, depth + 1)});
    }
    return JsonValue::object(std::move(members));
  }
  throw JsonError("value of type " +
                  nb::cast<std::string>(nb::handle(value).type().attr("__name__")) +
                  " is not JSON data");
}

[[nodiscard]] nb::object json_to_python(const JsonValue& value) {
  if (value.is_null()) {
    return nb::none();
  }
  if (value.is_boolean()) {
    return nb::bool_(value.as_boolean());
  }
  if (value.is_integer()) {
    return nb::steal<nb::object>(PyLong_FromString(value.as_integer_token().c_str(), nullptr, 10));
  }
  if (value.is_number()) {
    return nb::float_(value.as_number());
  }
  if (value.is_string()) {
    return nb::str(value.as_string().data(), value.as_string().size());
  }
  if (value.is_array()) {
    nb::list items;
    for (const auto& item : value.as_array()) {
      items.append(json_to_python(item));
    }
    return items;
  }
  nb::dict members;
  for (const auto& member : value.as_object()) {
    members[nb::str(member.key.data(), member.key.size())] = json_to_python(member.value);
  }
  return members;
}

[[nodiscard]] JsonValue checkpoint_json(nb::handle value, std::string_view role) {
  try {
    return json_from_python(value);
  } catch (const JsonError& error) {
    throw CheckpointError("checkpoint " + std::string(role) +
                          " is not finite JSON data: " + error.what());
  }
}

template <typename Value>
[[nodiscard]] nb::tuple to_tuple(const std::vector<Value>& values) {
  nb::list items;
  for (const auto& value : values) {
    items.append(nb::cast(value));
  }
  return nb::tuple(items);
}

[[nodiscard]] nb::tuple to_tuple(const SceneVector& value) {
  return nb::make_tuple(value[0], value[1], value[2]);
}

[[nodiscard]] nb::object labels_to_python(const std::optional<ChannelMetadata::Labels>& labels) {
  if (!labels.has_value()) {
    return nb::none();
  }
  nb::list items;
  for (const auto& label : *labels) {
    if (label.has_value()) {
      items.append(nb::str(label->data(), label->size()));
    } else {
      items.append(nb::none());
    }
  }
  return nb::tuple(items);
}

[[nodiscard]] std::optional<ChannelMetadata::Labels> labels_from_python(nb::handle value,
                                                                        std::string_view kind) {
  const auto path = "channel_metadata." + std::string(kind);
  if (value.is_none()) {
    return std::nullopt;
  }
  if (!PyTuple_Check(value.ptr())) {
    throw ChannelMetadataError(path + ": expected a tuple or None");
  }
  ChannelMetadata::Labels labels;
  std::size_t index = 0;
  for (const auto item : value) {
    const auto item_path = path + "[" + std::to_string(index) + "]";
    if (item.is_none()) {
      labels.emplace_back();
    } else if (PyUnicode_Check(item.ptr())) {
      Py_ssize_t size = 0;
      const char* text = PyUnicode_AsUTF8AndSize(item.ptr(), &size);
      if (text == nullptr) {
        PyErr_Clear();
        throw ChannelMetadataError(item_path + ": invalid Unicode scalar value");
      }
      labels.emplace_back(std::string(text, static_cast<std::size_t>(size)));
    } else {
      throw ChannelMetadataError(item_path + ": expected a string or None");
    }
    ++index;
  }
  return labels;
}

[[nodiscard]] microsimulator::BackendKind backend_kind(const std::string& name) {
  const auto kind = microsimulator::parse_backend_kind(name);
  if (!kind.has_value()) {
    throw SceneError("unknown backend kind '" + name + "'");
  }
  return *kind;
}

[[nodiscard]] microsimulator::GridBoundaryKind boundary_kind(const std::string& name) {
  const auto kind = microsimulator::parse_grid_boundary_kind(name);
  if (!kind.has_value()) {
    throw SceneError("unknown boundary kind '" + name + "'");
  }
  return *kind;
}

[[nodiscard]] microsimulator::ConstraintRegion region_kind(const std::string& name) {
  const auto region = microsimulator::parse_constraint_region(name);
  if (!region.has_value()) {
    throw SceneError("unknown region kind '" + name + "'");
  }
  return *region;
}

[[nodiscard]] std::string region_name(microsimulator::ConstraintRegion region) {
  return std::string(microsimulator::constraint_region_name(region));
}

// Gives a bound value type the behavior of an immutable record: equality and
// hashing by value, a repr listing its fields, copying and pickling through
// its constructor, and replace(**changes), which builds a new instance from
// the current fields with the named ones substituted. The fields are listed
// in constructor argument order.
template <typename Class>
void def_record(nb::class_<Class>& bound, std::vector<const char*> fields) {
  bound.def("__eq__", [](const Class& self, nb::handle other) {
    return nb::isinstance<Class>(other) && self == nb::cast<const Class&>(other);
  });
  bound.def("__repr__", [fields](nb::handle self) {
    auto text = nb::cast<std::string>(self.type().attr("__name__")) + "(";
    for (std::size_t index = 0; index < fields.size(); ++index) {
      text += index == 0 ? "" : ", ";
      text += fields[index];
      text += "=";
      text += nb::repr(self.attr(fields[index])).c_str();
    }
    return text + ")";
  });
  bound.def("replace", [fields](nb::handle self, const nb::kwargs& changes) {
    nb::dict arguments;
    for (const char* field : fields) {
      arguments[field] = self.attr(field);
    }
    for (const auto [name, value] : changes) {
      if (!arguments.contains(name)) {
        throw nb::type_error(
            ("replace() got an unexpected field " + nb::cast<std::string>(nb::repr(name))).c_str());
      }
      arguments[name] = value;
    }
    return self.type()(**arguments);
  });
  bound.def("__reduce__", [fields](nb::handle self) {
    nb::list values;
    for (const char* field : fields) {
      values.append(self.attr(field));
    }
    return nb::make_tuple(self.type(), nb::tuple(values));
  });
  bound.def("__hash__", [fields](nb::handle self) {
    nb::list values;
    for (const char* field : fields) {
      values.append(self.attr(field));
    }
    const auto hash = PyObject_Hash(nb::tuple(values).ptr());
    if (hash == -1 && PyErr_Occurred() != nullptr) {
      throw nb::python_error();
    }
    return hash;
  });
}

void bind_channels(nb::module_& module) {
  auto metadata = nb::class_<ChannelMetadata>(module, "ChannelMetadata");
  metadata
      .def(
          "__init__",
          [](ChannelMetadata* self, nb::handle species, nb::handle signals) {
            new (self) ChannelMetadata{
                .species = labels_from_python(species, "species"),
                .signals = labels_from_python(signals, "signals"),
            };
          },
          "species"_a = nb::none(), "signals"_a = nb::none())
      .def_prop_ro("species",
                   [](const ChannelMetadata& self) { return labels_to_python(self.species); })
      .def_prop_ro("signals",
                   [](const ChannelMetadata& self) { return labels_to_python(self.signals); })
      .def("resolved", &ChannelMetadata::resolved, "species_count"_a, "signal_count"_a)
      .def(
          "to_json",
          [](const ChannelMetadata& self, std::size_t species_count, std::size_t signal_count) {
            return json_to_python(
                microsimulator::channel_metadata_to_json(self, species_count, signal_count));
          },
          "species_count"_a, "signal_count"_a)
      .def_static(
          "from_json",
          [](nb::handle value, std::size_t species_count, std::size_t signal_count) {
            JsonValue document;
            try {
              document = json_from_python(value);
            } catch (const JsonError& error) {
              throw ChannelMetadataError(std::string("channel_metadata: ") + error.what());
            }
            return microsimulator::channel_metadata_from_json(document, species_count,
                                                              signal_count);
          },
          "value"_a, "species_count"_a, "signal_count"_a);
  def_record(metadata, {"species", "signals"});
}

void bind_checkpoint_files(nb::module_& module) {
  module.attr("CHECKPOINT_FORMAT") = std::string(microsimulator::checkpoint_format);
  module.attr("CHECKPOINT_VERSION") = microsimulator::checkpoint_format_version;
  module.attr("MAX_CHECKPOINT_BYTES") = microsimulator::max_checkpoint_bytes;

  nb::class_<CheckpointDocument>(module, "_CheckpointDocument")
      .def_prop_ro(
          "simulation",
          [](const CheckpointDocument& self) -> const microsimulator::SimulationCheckpoint& {
            return self.simulation;
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro("controller",
                   [](const CheckpointDocument& self) { return json_to_python(self.controller); })
      .def_prop_ro("provenance",
                   [](const CheckpointDocument& self) { return json_to_python(self.provenance); })
      .def_prop_ro(
          "source_backend",
          [](const CheckpointDocument& self) -> const microsimulator::BackendInfo& {
            return self.source_backend;
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro("channel_metadata", [](const CheckpointDocument& self) -> ChannelMetadata {
        return self.channel_metadata;
      });

  module.def(
      "_save_checkpoint",
      [](Simulation& simulation, const std::filesystem::path& path, nb::handle provenance,
         nb::handle controller, const ChannelMetadata& channel_metadata) {
        microsimulator::python::BusyObjects::require_idle(&simulation);
        CheckpointDocument document;
        document.simulation = simulation.checkpoint();
        document.source_backend = simulation.backend_info();
        document.provenance = checkpoint_json(provenance, "provenance");
        document.controller = checkpoint_json(controller, "controller");
        document.channel_metadata = channel_metadata;
        const nb::gil_scoped_release release;
        microsimulator::save_checkpoint(path, document);
      },
      "simulation"_a, "path"_a, "provenance"_a, "controller"_a.none(), "channel_metadata"_a);
  module.def(
      "_load_checkpoint",
      [](const std::filesystem::path& path) {
        const nb::gil_scoped_release release;
        return microsimulator::load_checkpoint(path);
      },
      "path"_a);
}

void bind_scenes(nb::module_& module) {
  module.attr("SCENE_FORMAT") = std::string(microsimulator::scene_format);
  module.attr("SCENE_VERSION") = microsimulator::scene_format_version;
  module.attr("MAX_SCENE_BYTES") = microsimulator::max_scene_bytes;
  module.attr("MAX_SCENE_CHANNELS") = microsimulator::max_scene_channels;

  auto backend = nb::class_<SceneBackend>(module, "SceneBackend");
  backend
      .def(
          "__init__",
          [](SceneBackend* self, const std::string& kind, std::string name, std::string device,
             std::uint32_t device_index, bool native) {
            new (self) SceneBackend{
                .kind = backend_kind(kind),
                .name = std::move(name),
                .device = std::move(device),
                .device_index = device_index,
                .native = native,
            };
          },
          "kind"_a, "name"_a, "device"_a, "device_index"_a, "native"_a)
      .def_prop_ro("kind",
                   [](const SceneBackend& self) {
                     return std::string(microsimulator::backend_kind_name(self.kind));
                   })
      .def_ro("name", &SceneBackend::name)
      .def_ro("device", &SceneBackend::device)
      .def_ro("device_index", &SceneBackend::device_index)
      .def_ro("native", &SceneBackend::native);
  def_record(backend, {"kind", "name", "device", "device_index", "native"});

  auto cell = nb::class_<SceneCell>(module, "SceneCell");
  cell.def(
          "__init__",
          [](SceneCell* self, microsimulator::CellId id,
             std::optional<microsimulator::CellId> parent_id, microsimulator::Slot slot,
             SceneVector position, SceneVector direction, float length, float radius,
             float growth_rate, std::int32_t cell_type, bool fixed, std::vector<float> species) {
            new (self) SceneCell{
                .id = id,
                .parent_id = parent_id,
                .slot = slot,
                .position = position,
                .direction = direction,
                .length = length,
                .radius = radius,
                .growth_rate = growth_rate,
                .cell_type = cell_type,
                .fixed = fixed,
                .species = std::move(species),
            };
          },
          "id"_a, "parent_id"_a.none(), "slot"_a, "position"_a, "direction"_a, "length"_a,
          "radius"_a, "growth_rate"_a, "cell_type"_a, "fixed"_a, "species"_a)
      .def_ro("id", &SceneCell::id)
      .def_ro("parent_id", &SceneCell::parent_id)
      .def_ro("slot", &SceneCell::slot)
      .def_prop_ro("position", [](const SceneCell& self) { return to_tuple(self.position); })
      .def_prop_ro("direction", [](const SceneCell& self) { return to_tuple(self.direction); })
      .def_ro("length", &SceneCell::length)
      .def_ro("radius", &SceneCell::radius)
      .def_ro("growth_rate", &SceneCell::growth_rate)
      .def_ro("cell_type", &SceneCell::cell_type)
      .def_ro("fixed", &SceneCell::fixed)
      .def_prop_ro("species", [](const SceneCell& self) { return to_tuple(self.species); });
  def_record(cell, {"id", "parent_id", "slot", "position", "direction", "length", "radius",
                    "growth_rate", "cell_type", "fixed", "species"});

  auto boundary = nb::class_<SceneGridBoundary>(module, "SceneGridBoundary");
  boundary
      .def(
          "__init__",
          [](SceneGridBoundary* self, const std::string& kind, std::vector<float> values) {
            new (self) SceneGridBoundary{.kind = boundary_kind(kind), .values = std::move(values)};
          },
          "kind"_a, "values"_a)
      .def_prop_ro("kind",
                   [](const SceneGridBoundary& self) {
                     return std::string(microsimulator::grid_boundary_kind_name(self.kind));
                   })
      .def_prop_ro("values", [](const SceneGridBoundary& self) { return to_tuple(self.values); });
  def_record(boundary, {"kind", "values"});

  auto grid = nb::class_<SceneSignalGrid>(module, "SceneSignalGrid");
  grid.def(
          "__init__",
          [](SceneSignalGrid* self, std::uint32_t signal_count, std::array<std::uint32_t, 3> shape,
             SceneVector origin, SceneVector spacing, SceneGridBoundary x_lower,
             SceneGridBoundary x_upper, SceneGridBoundary y_lower, SceneGridBoundary y_upper,
             SceneGridBoundary z_lower, SceneGridBoundary z_upper, std::vector<float> levels) {
            new (self) SceneSignalGrid{
                .signal_count = signal_count,
                .shape = shape,
                .origin = origin,
                .spacing = spacing,
                .x_lower = std::move(x_lower),
                .x_upper = std::move(x_upper),
                .y_lower = std::move(y_lower),
                .y_upper = std::move(y_upper),
                .z_lower = std::move(z_lower),
                .z_upper = std::move(z_upper),
                .levels = std::move(levels),
            };
          },
          "signal_count"_a, "shape"_a, "origin"_a, "spacing"_a, "x_lower"_a, "x_upper"_a,
          "y_lower"_a, "y_upper"_a, "z_lower"_a, "z_upper"_a, "levels"_a)
      .def_ro("signal_count", &SceneSignalGrid::signal_count)
      .def_prop_ro("shape",
                   [](const SceneSignalGrid& self) {
                     return nb::make_tuple(self.shape[0], self.shape[1], self.shape[2]);
                   })
      .def_prop_ro("origin", [](const SceneSignalGrid& self) { return to_tuple(self.origin); })
      .def_prop_ro("spacing", [](const SceneSignalGrid& self) { return to_tuple(self.spacing); })
      .def_ro("x_lower", &SceneSignalGrid::x_lower)
      .def_ro("x_upper", &SceneSignalGrid::x_upper)
      .def_ro("y_lower", &SceneSignalGrid::y_lower)
      .def_ro("y_upper", &SceneSignalGrid::y_upper)
      .def_ro("z_lower", &SceneSignalGrid::z_lower)
      .def_ro("z_upper", &SceneSignalGrid::z_upper)
      .def_prop_ro("levels", [](const SceneSignalGrid& self) { return to_tuple(self.levels); });
  def_record(grid, {"signal_count", "shape", "origin", "spacing", "x_lower", "x_upper", "y_lower",
                    "y_upper", "z_lower", "z_upper", "levels"});

  auto plane = nb::class_<ScenePlaneConstraint>(module, "ScenePlaneConstraint");
  plane
      .def(
          "__init__",
          [](ScenePlaneConstraint* self, microsimulator::ConstraintId id, SceneVector point,
             SceneVector inward_normal, float coefficient) {
            new (self) ScenePlaneConstraint{
                .id = id,
                .point = point,
                .inward_normal = inward_normal,
                .coefficient = coefficient,
            };
          },
          "id"_a, "point"_a, "inward_normal"_a, "coefficient"_a)
      .def_ro("id", &ScenePlaneConstraint::id)
      .def_prop_ro("point", [](const ScenePlaneConstraint& self) { return to_tuple(self.point); })
      .def_prop_ro("inward_normal",
                   [](const ScenePlaneConstraint& self) { return to_tuple(self.inward_normal); })
      .def_ro("coefficient", &ScenePlaneConstraint::coefficient);
  def_record(plane, {"id", "point", "inward_normal", "coefficient"});

  auto sphere = nb::class_<SceneSphereConstraint>(module, "SceneSphereConstraint");
  sphere
      .def(
          "__init__",
          [](SceneSphereConstraint* self, microsimulator::ConstraintId id, SceneVector center,
             float radius, float coefficient, const std::string& allowed_region) {
            new (self) SceneSphereConstraint{
                .id = id,
                .center = center,
                .radius = radius,
                .coefficient = coefficient,
                .allowed_region = region_kind(allowed_region),
            };
          },
          "id"_a, "center"_a, "radius"_a, "coefficient"_a, "allowed_region"_a)
      .def_ro("id", &SceneSphereConstraint::id)
      .def_prop_ro("center",
                   [](const SceneSphereConstraint& self) { return to_tuple(self.center); })
      .def_ro("radius", &SceneSphereConstraint::radius)
      .def_ro("coefficient", &SceneSphereConstraint::coefficient)
      .def_prop_ro("allowed_region", [](const SceneSphereConstraint& self) {
        return region_name(self.allowed_region);
      });
  def_record(sphere, {"id", "center", "radius", "coefficient", "allowed_region"});

  auto box = nb::class_<SceneBoxConstraint>(module, "SceneBoxConstraint");
  box.def(
         "__init__",
         [](SceneBoxConstraint* self, microsimulator::ConstraintId id, SceneVector center,
            SceneVector half_extents, float coefficient, const std::string& allowed_region) {
           new (self) SceneBoxConstraint{
               .id = id,
               .center = center,
               .half_extents = half_extents,
               .coefficient = coefficient,
               .allowed_region = region_kind(allowed_region),
           };
         },
         "id"_a, "center"_a, "half_extents"_a, "coefficient"_a, "allowed_region"_a)
      .def_ro("id", &SceneBoxConstraint::id)
      .def_prop_ro("center", [](const SceneBoxConstraint& self) { return to_tuple(self.center); })
      .def_prop_ro("half_extents",
                   [](const SceneBoxConstraint& self) { return to_tuple(self.half_extents); })
      .def_ro("coefficient", &SceneBoxConstraint::coefficient)
      .def_prop_ro("allowed_region",
                   [](const SceneBoxConstraint& self) { return region_name(self.allowed_region); });
  def_record(box, {"id", "center", "half_extents", "coefficient", "allowed_region"});

  auto cylinder = nb::class_<SceneCylinderConstraint>(module, "SceneCylinderConstraint");
  cylinder
      .def(
          "__init__",
          [](SceneCylinderConstraint* self, microsimulator::ConstraintId id, SceneVector center,
             float radius, float half_height, float coefficient,
             const std::string& allowed_region) {
            new (self) SceneCylinderConstraint{
                .id = id,
                .center = center,
                .radius = radius,
                .half_height = half_height,
                .coefficient = coefficient,
                .allowed_region = region_kind(allowed_region),
            };
          },
          "id"_a, "center"_a, "radius"_a, "half_height"_a, "coefficient"_a, "allowed_region"_a)
      .def_ro("id", &SceneCylinderConstraint::id)
      .def_prop_ro("center",
                   [](const SceneCylinderConstraint& self) { return to_tuple(self.center); })
      .def_ro("radius", &SceneCylinderConstraint::radius)
      .def_ro("half_height", &SceneCylinderConstraint::half_height)
      .def_ro("coefficient", &SceneCylinderConstraint::coefficient)
      .def_prop_ro("allowed_region", [](const SceneCylinderConstraint& self) {
        return region_name(self.allowed_region);
      });
  def_record(cylinder, {"id", "center", "radius", "half_height", "coefficient", "allowed_region"});

  auto constraints = nb::class_<SceneConstraints>(module, "SceneConstraints");
  constraints
      .def(
          "__init__",
          [](SceneConstraints* self, std::vector<ScenePlaneConstraint> planes,
             std::vector<SceneSphereConstraint> spheres, std::vector<SceneBoxConstraint> boxes,
             std::vector<SceneCylinderConstraint> cylinders) {
            new (self) SceneConstraints{
                .planes = std::move(planes),
                .spheres = std::move(spheres),
                .boxes = std::move(boxes),
                .cylinders = std::move(cylinders),
            };
          },
          "planes"_a, "spheres"_a, "boxes"_a, "cylinders"_a)
      .def_prop_ro("planes", [](const SceneConstraints& self) { return to_tuple(self.planes); })
      .def_prop_ro("spheres", [](const SceneConstraints& self) { return to_tuple(self.spheres); })
      .def_prop_ro("boxes", [](const SceneConstraints& self) { return to_tuple(self.boxes); })
      .def_prop_ro("cylinders",
                   [](const SceneConstraints& self) { return to_tuple(self.cylinders); });
  def_record(constraints, {"planes", "spheres", "boxes", "cylinders"});

  auto frame = nb::class_<SceneFrame>(module, "SceneFrame");
  frame
      .def(
          "__init__",
          [](SceneFrame* self, double time, SceneBackend backend, std::uint32_t species_count,
             std::vector<SceneCell> cells, SceneConstraints constraints,
             std::optional<SceneSignalGrid> signal_grid, ChannelMetadata channel_metadata) {
            SceneFrame value{
                .time = time,
                .backend = std::move(backend),
                .species_count = species_count,
                .cells = std::move(cells),
                .constraints = std::move(constraints),
                .signal_grid = std::move(signal_grid),
                .channel_metadata = std::move(channel_metadata),
            };
            value.resolve_channels();
            new (self) SceneFrame(std::move(value));
          },
          "time"_a, "backend"_a, "species_count"_a, "cells"_a, "constraints"_a,
          "signal_grid"_a.none(), "channel_metadata"_a = ChannelMetadata{})
      .def_ro("time", &SceneFrame::time)
      .def_ro("backend", &SceneFrame::backend)
      .def_ro("species_count", &SceneFrame::species_count)
      .def_prop_ro("cells", [](const SceneFrame& self) { return to_tuple(self.cells); })
      .def_ro("constraints", &SceneFrame::constraints)
      .def_ro("signal_grid", &SceneFrame::signal_grid)
      .def_ro("channel_metadata", &SceneFrame::channel_metadata);
  def_record(frame, {"time", "backend", "species_count", "cells", "constraints", "signal_grid",
                     "channel_metadata"});

  module.def(
      "capture_scene",
      [](Simulation& simulation, const ChannelMetadata& channel_metadata) {
        const microsimulator::python::ReleasedCall call(&simulation);
        return microsimulator::capture_scene(simulation, channel_metadata);
      },
      "simulation"_a, nb::kw_only(), "channel_metadata"_a = ChannelMetadata{});
  module.def(
      "dumps_scene",
      [](const SceneFrame& frame) {
        const nb::gil_scoped_release release;
        return microsimulator::encode_scene(frame);
      },
      "frame"_a);
  module.def(
      "save_scene",
      [](const SceneFrame& frame, const std::filesystem::path& path) {
        const nb::gil_scoped_release release;
        microsimulator::save_scene(path, frame);
      },
      "frame"_a, "path"_a);
  module.def(
      "parse_scene",
      [](nb::handle source) {
        std::string text;
        if (PyUnicode_Check(source.ptr())) {
          Py_ssize_t size = 0;
          const char* data = PyUnicode_AsUTF8AndSize(source.ptr(), &size);
          if (data == nullptr) {
            PyErr_Clear();
            throw SceneError("scene is not valid JSON: text is not valid Unicode");
          }
          text.assign(data, static_cast<std::size_t>(size));
        } else {
          Py_buffer view;
          if (PyObject_GetBuffer(source.ptr(), &view, PyBUF_SIMPLE) != 0) {
            PyErr_Clear();
            throw nb::type_error("parse_scene() expects str or a bytes-like object");
          }
          text.assign(static_cast<const char*>(view.buf), static_cast<std::size_t>(view.len));
          PyBuffer_Release(&view);
        }
        const nb::gil_scoped_release release;
        return microsimulator::parse_scene(text);
      },
      "source"_a);
  module.def(
      "load_scene",
      [](const std::filesystem::path& path) {
        const nb::gil_scoped_release release;
        return microsimulator::load_scene(path);
      },
      "path"_a);
}

}  // namespace

void bind_serialization(nb::module_& module) {
  nb::exception<ChannelMetadataError>(module, "ChannelMetadataError", PyExc_ValueError);
  nb::exception<CheckpointError>(module, "CheckpointError", PyExc_ValueError);
  nb::exception<SceneError>(module, "SceneError", PyExc_ValueError);
  bind_channels(module);
  bind_checkpoint_files(module);
  bind_scenes(module);
}
