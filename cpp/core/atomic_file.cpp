#include "microsimulator/atomic_file.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <random>
#include <string>
#include <system_error>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace microsimulator {
namespace {

#if defined(_WIN32)
[[nodiscard]] int open_exclusive(const std::filesystem::path& path) noexcept {
  return _wopen(path.c_str(), _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY | _O_NOINHERIT,
                _S_IREAD | _S_IWRITE);
}

[[nodiscard]] int open_for_reading(const std::filesystem::path& path) noexcept {
  return _wopen(path.c_str(), _O_RDONLY | _O_BINARY | _O_NOINHERIT);
}

[[nodiscard]] long long write_some(int descriptor, const char* data, std::size_t size) noexcept {
  constexpr std::size_t chunk = 1U << 30U;
  return _write(descriptor, data, static_cast<unsigned>(size < chunk ? size : chunk));
}

[[nodiscard]] long long read_some(int descriptor, char* data, std::size_t size) noexcept {
  constexpr std::size_t chunk = 1U << 30U;
  return _read(descriptor, data, static_cast<unsigned>(size < chunk ? size : chunk));
}

[[nodiscard]] bool flush_to_storage(int descriptor) noexcept {
  return _commit(descriptor) == 0;
}

void close_descriptor(int descriptor) noexcept {
  _close(descriptor);
}

[[nodiscard]] bool replace_file(const std::filesystem::path& source,
                                const std::filesystem::path& destination) noexcept {
  return MoveFileExW(source.c_str(), destination.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
}
#else
[[nodiscard]] int open_exclusive(const std::filesystem::path& path) noexcept {
  return ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
}

[[nodiscard]] int open_for_reading(const std::filesystem::path& path) noexcept {
  return ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
}

[[nodiscard]] long long write_some(int descriptor, const char* data, std::size_t size) noexcept {
  return ::write(descriptor, data, size);
}

[[nodiscard]] long long read_some(int descriptor, char* data, std::size_t size) noexcept {
  return ::read(descriptor, data, size);
}

[[nodiscard]] bool flush_to_storage(int descriptor) noexcept {
  return ::fsync(descriptor) == 0;
}

void close_descriptor(int descriptor) noexcept {
  ::close(descriptor);
}

[[nodiscard]] bool replace_file(const std::filesystem::path& source,
                                const std::filesystem::path& destination) noexcept {
  return ::rename(source.c_str(), destination.c_str()) == 0;
}
#endif

class Descriptor {
 public:
  explicit Descriptor(int value) noexcept : value_(value) {
  }
  Descriptor(const Descriptor&) = delete;
  Descriptor& operator=(const Descriptor&) = delete;
  ~Descriptor() {
    close();
  }

  [[nodiscard]] int get() const noexcept {
    return value_;
  }
  [[nodiscard]] bool valid() const noexcept {
    return value_ >= 0;
  }
  void close() noexcept {
    if (value_ >= 0) {
      close_descriptor(value_);
      value_ = -1;
    }
  }

 private:
  int value_;
};

[[nodiscard]] std::string random_suffix() {
  static constexpr std::string_view hex = "0123456789abcdef";
  std::random_device device;
  std::string suffix;
  for (int word = 0; word < 2; ++word) {
    auto bits = static_cast<std::uint32_t>(device());
    for (int digit = 0; digit < 8; ++digit) {
      suffix += hex[bits & 0x0FU];
      bits >>= 4U;
    }
  }
  return suffix;
}

[[nodiscard]] std::filesystem::path sibling(const std::filesystem::path& destination) {
  auto name = std::filesystem::path(".");
  name += destination.filename();
  name += ".";
  name += random_suffix();
  name += ".tmp";
  return destination.parent_path() / name;
}

void remove_quietly(const std::filesystem::path& path) noexcept {
  std::error_code ignored;
  std::filesystem::remove(path, ignored);
}

}  // namespace

std::string path_text(const std::filesystem::path& path) {
  const auto text = path.u8string();
  return {reinterpret_cast<const char*>(text.data()), text.size()};
}

void write_file_atomically(const std::filesystem::path& destination, std::string_view contents) {
  if (!destination.has_filename()) {
    throw FileError("destination has no file name");
  }
  std::filesystem::path temporary;
  int raw = -1;
  for (int attempt = 0; attempt < 16 && raw < 0; ++attempt) {
    temporary = sibling(destination);
    raw = open_exclusive(temporary);
    if (raw < 0 && errno != EEXIST) {
      break;
    }
  }
  if (raw < 0) {
    throw FileError("could not create a temporary file");
  }

  Descriptor descriptor(raw);
  std::size_t written = 0;
  while (written < contents.size()) {
    const auto count =
        write_some(descriptor.get(), contents.data() + written, contents.size() - written);
    if (count < 0 && errno == EINTR) {
      continue;
    }
    if (count <= 0) {
      descriptor.close();
      remove_quietly(temporary);
      throw FileError("could not write the temporary file");
    }
    written += static_cast<std::size_t>(count);
  }
  if (!flush_to_storage(descriptor.get())) {
    descriptor.close();
    remove_quietly(temporary);
    throw FileError("could not flush the temporary file");
  }
  descriptor.close();
  if (!replace_file(temporary, destination)) {
    remove_quietly(temporary);
    throw FileError("could not replace the destination");
  }
}

std::string read_file_prefix(const std::filesystem::path& source, std::size_t limit) {
  Descriptor descriptor(open_for_reading(source));
  if (!descriptor.valid()) {
    throw FileError("could not open the file");
  }
  std::string contents;
  std::array<char, 1U << 16U> buffer{};
  while (contents.size() < limit) {
    const auto wanted = std::min(buffer.size(), limit - contents.size());
    const auto count = read_some(descriptor.get(), buffer.data(), wanted);
    if (count < 0 && errno == EINTR) {
      continue;
    }
    if (count < 0) {
      throw FileError("could not read the file");
    }
    if (count == 0) {
      break;
    }
    contents.append(buffer.data(), static_cast<std::size_t>(count));
  }
  return contents;
}

}  // namespace microsimulator
