#pragma once

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "microsimulator/json.hpp"

namespace microsimulator {

// Raised when labels cannot describe the declared numerical channels.
class ChannelMetadataError : public std::invalid_argument {
 public:
  using std::invalid_argument::invalid_argument;
};

// Optional presentation labels for species and signal channels. Numerical
// channel indices remain authoritative.
//
// An absent group leaves every channel of that kind unnamed without storing
// one entry per channel. A present group holds exactly one entry per channel,
// where an absent entry is an unnamed channel. Empty and whitespace-only
// labels are stored as written.
struct ChannelMetadata {
  using Labels = std::vector<std::optional<std::string>>;

  std::optional<Labels> species;
  std::optional<Labels> signals;

  // Throws ChannelMetadataError unless every label is well-formed UTF-8.
  void validate() const;

  // Checks present groups against the channel counts and expands absent
  // groups to unnamed entries.
  [[nodiscard]] ChannelMetadata resolved(std::size_t species_count, std::size_t signal_count) const;

  [[nodiscard]] bool operator==(const ChannelMetadata&) const = default;
};

// The closed {"species": [...], "signals": [...]} representation shared by
// checkpoints and scenes, with null for an unnamed channel.
[[nodiscard]] JsonValue channel_metadata_to_json(const ChannelMetadata& metadata,
                                                 std::size_t species_count,
                                                 std::size_t signal_count);
[[nodiscard]] ChannelMetadata channel_metadata_from_json(const JsonValue& value,
                                                         std::size_t species_count,
                                                         std::size_t signal_count);

}  // namespace microsimulator
