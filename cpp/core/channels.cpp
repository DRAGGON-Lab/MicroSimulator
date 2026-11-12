#include "microsimulator/channels.hpp"

#include <array>
#include <string_view>
#include <utility>

namespace microsimulator {
namespace {

struct Group {
  std::string_view name;
  const std::optional<ChannelMetadata::Labels>* labels;
  std::size_t count;
};

[[nodiscard]] std::string group_path(std::string_view name) {
  return "channel_metadata." + std::string(name);
}

[[nodiscard]] JsonValue labels_to_json(const ChannelMetadata::Labels& labels) {
  JsonValue::Array values;
  values.reserve(labels.size());
  for (const auto& label : labels) {
    values.push_back(label.has_value() ? JsonValue::string(*label) : JsonValue{});
  }
  return JsonValue::array(std::move(values));
}

[[nodiscard]] ChannelMetadata::Labels labels_from_json(const JsonValue& value,
                                                       std::string_view name) {
  if (!value.is_array()) {
    throw ChannelMetadataError(group_path(name) + ": expected an array");
  }
  ChannelMetadata::Labels labels;
  labels.reserve(value.as_array().size());
  std::size_t index = 0;
  for (const auto& item : value.as_array()) {
    if (item.is_null()) {
      labels.emplace_back();
    } else if (item.is_string()) {
      labels.emplace_back(item.as_string());
    } else {
      throw ChannelMetadataError(group_path(name) + "[" + std::to_string(index) +
                                 "]: expected a string or None");
    }
    ++index;
  }
  return labels;
}

}  // namespace

void ChannelMetadata::validate() const {
  for (const auto& [name, labels] :
       std::array<std::pair<std::string_view, const std::optional<Labels>*>, 2>{
           {{"species", &species}, {"signals", &signals}}}) {
    if (!labels->has_value()) {
      continue;
    }
    std::size_t index = 0;
    for (const auto& label : **labels) {
      if (label.has_value()) {
        try {
          static_cast<void>(JsonValue::string(*label));
        } catch (const JsonError&) {
          throw ChannelMetadataError(group_path(name) + "[" + std::to_string(index) +
                                     "]: invalid Unicode scalar value");
        }
      }
      ++index;
    }
  }
}

ChannelMetadata ChannelMetadata::resolved(std::size_t species_count,
                                          std::size_t signal_count) const {
  validate();
  ChannelMetadata result;
  const std::array<Group, 2> groups{
      {{"species", &species, species_count}, {"signals", &signals, signal_count}}};
  const std::array<std::optional<Labels>*, 2> outputs{&result.species, &result.signals};
  for (std::size_t index = 0; index < groups.size(); ++index) {
    const auto& group = groups[index];
    if (group.labels->has_value() && (*group.labels)->size() != group.count) {
      throw ChannelMetadataError(group_path(group.name) + ": expected " +
                                 std::to_string(group.count) + " labels, got " +
                                 std::to_string((*group.labels)->size()));
    }
    *outputs[index] = group.labels->has_value() ? **group.labels : Labels(group.count);
  }
  return result;
}

JsonValue channel_metadata_to_json(const ChannelMetadata& metadata, std::size_t species_count,
                                   std::size_t signal_count) {
  const auto labels = metadata.resolved(species_count, signal_count);
  return JsonValue::object({
      {"species", labels_to_json(*labels.species)},
      {"signals", labels_to_json(*labels.signals)},
  });
}

ChannelMetadata channel_metadata_from_json(const JsonValue& value, std::size_t species_count,
                                           std::size_t signal_count) {
  if (!value.is_object() || value.as_object().size() != 2 || value.find("species") == nullptr ||
      value.find("signals") == nullptr) {
    throw ChannelMetadataError("channel_metadata: expected exactly species and signals");
  }
  ChannelMetadata metadata;
  metadata.species = labels_from_json(*value.find("species"), "species");
  metadata.signals = labels_from_json(*value.find("signals"), "signals");
  return metadata.resolved(species_count, signal_count);
}

}  // namespace microsimulator
