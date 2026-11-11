#pragma once

#include <string>
#include <string_view>

namespace microsimulator {

// The SHA-256 digest of the bytes as 64 lowercase hexadecimal digits.
[[nodiscard]] std::string sha256_hex(std::string_view bytes);

// Compares two digests in time that depends only on their lengths.
[[nodiscard]] bool digests_equal(std::string_view left, std::string_view right) noexcept;

}  // namespace microsimulator
