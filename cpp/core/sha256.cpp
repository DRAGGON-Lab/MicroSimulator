#include "microsimulator/sha256.hpp"

#include <picosha2/picosha2.h>

#include <cstddef>

namespace microsimulator {

std::string sha256_hex(std::string_view bytes) {
  return picosha2::hash256_hex_string(bytes.begin(), bytes.end());
}

bool digests_equal(std::string_view left, std::string_view right) noexcept {
  if (left.size() != right.size()) {
    return false;
  }
  unsigned difference = 0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    difference |= static_cast<unsigned>(static_cast<unsigned char>(left[index]) ^
                                        static_cast<unsigned char>(right[index]));
  }
  return difference == 0;
}

}  // namespace microsimulator
