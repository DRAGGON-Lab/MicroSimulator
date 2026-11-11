#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

#include "microsimulator/channels.hpp"
#include "microsimulator/checkpoint.hpp"
#include "microsimulator/json.hpp"
#include "microsimulator/types.hpp"

namespace microsimulator {

inline constexpr std::string_view checkpoint_format = "microsimulator-checkpoint";
inline constexpr std::uint32_t checkpoint_format_version = 10;
inline constexpr std::size_t max_checkpoint_bytes = std::size_t{1} << 30U;

// Raised when a checkpoint document cannot be encoded, decoded, or validated.
class CheckpointError : public std::invalid_argument {
 public:
  using std::invalid_argument::invalid_argument;
};

// Everything a checkpoint file holds.
//
// The controller and provenance are opaque JSON owned by the caller: the
// controller is whatever a model needs to resume and is null for a bare
// simulation, and provenance is an object describing how the run was made.
// Neither is interpreted or executed.
struct CheckpointDocument {
  SimulationCheckpoint simulation;
  JsonValue controller;
  JsonValue provenance{JsonValue::object({})};
  BackendInfo source_backend;
  ChannelMetadata channel_metadata;
};

// Encodes a document as readable JSON text. The simulation state, the
// controller, and the channel metadata each carry a SHA-256 digest of their
// canonical JSON. A digest detects accidental corruption; it is not an
// authenticity signature.
[[nodiscard]] std::string encode_checkpoint(const CheckpointDocument& document);

// Decodes checkpoint text. Rejects text that is not strict JSON, objects with
// unknown or missing keys, values outside their numeric ranges, any other
// format version, a digest mismatch, and state that fails validation.
[[nodiscard]] CheckpointDocument parse_checkpoint(std::string_view text);

// Atomically replaces the file at the path with the encoded document.
void save_checkpoint(const std::filesystem::path& path, const CheckpointDocument& document);

// Reads and decodes a checkpoint of at most max_checkpoint_bytes.
[[nodiscard]] CheckpointDocument load_checkpoint(const std::filesystem::path& path);

}  // namespace microsimulator
