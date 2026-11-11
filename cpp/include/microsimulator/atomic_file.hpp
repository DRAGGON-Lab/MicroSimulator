#pragma once

#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

namespace microsimulator {

// Raised when a file cannot be read or written.
class FileError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Replaces the destination with the contents in one step. The bytes are
// written to a hidden sibling named ".<name>.<random>.tmp", flushed to
// storage, and renamed over the destination, so a reader sees either the
// previous file or the complete new one. The sibling is removed on failure.
void write_file_atomically(const std::filesystem::path& destination, std::string_view contents);

// Reads at most limit bytes from the start of the file.
[[nodiscard]] std::string read_file_prefix(const std::filesystem::path& source, std::size_t limit);

}  // namespace microsimulator
