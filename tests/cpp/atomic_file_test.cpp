#include "microsimulator/atomic_file.hpp"

#include <cassert>
#include <filesystem>
#include <string>

namespace {

std::size_t entry_count(const std::filesystem::path& directory) {
  std::size_t count = 0;
  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    static_cast<void>(entry);
    ++count;
  }
  return count;
}

}  // namespace

int main() {
  using microsimulator::FileError;
  using microsimulator::read_file_prefix;
  using microsimulator::write_file_atomically;

  const auto directory = std::filesystem::temp_directory_path() / "microsimulator-atomic-file-test";
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  const auto target = directory / "state.json";

  write_file_atomically(target, "first");
  assert(read_file_prefix(target, 1024) == "first");
  const std::string large(3'000'000, 'x');
  write_file_atomically(target, large);
  assert(read_file_prefix(target, large.size() + 1) == large);
  assert(read_file_prefix(target, 10) == std::string(10, 'x'));
  write_file_atomically(target, "");
  assert(read_file_prefix(target, 1024).empty());
  assert(entry_count(directory) == 1);

  bool rejected = false;
  try {
    write_file_atomically(directory / "missing" / "state.json", "data");
  } catch (const FileError&) {
    rejected = true;
  }
  assert(rejected);

  rejected = false;
  try {
    static_cast<void>(read_file_prefix(directory / "absent.json", 16));
  } catch (const FileError&) {
    rejected = true;
  }
  assert(rejected);
  assert(entry_count(directory) == 1);

  std::filesystem::remove_all(directory);
  return 0;
}
