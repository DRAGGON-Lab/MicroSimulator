#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "microsimulator/json.hpp"
#include "microsimulator/types.hpp"

// Typed, closed-schema access to a parsed JSON document. Every failure throws
// JsonError whose message starts with the JSONPath of the offending value,
// such as "$.simulation.world.cells[0].slot: expected an integer".
namespace microsimulator::detail {

// A JSONPath built from stack-allocated links, rendered only when a check
// fails. A child refers to its parent, which must outlive it.
class JsonPath {
 public:
  JsonPath() = default;

  [[nodiscard]] JsonPath operator[](std::string_view key) const noexcept {
    return JsonPath(this, key, 0, false);
  }
  [[nodiscard]] JsonPath operator[](std::size_t index) const noexcept {
    return JsonPath(this, {}, index, true);
  }

  [[nodiscard]] std::string str() const {
    if (parent_ == nullptr) {
      return "$";
    }
    auto text = parent_->str();
    if (indexed_) {
      text += '[';
      text += std::to_string(index_);
      text += ']';
    } else {
      text += '.';
      text += key_;
    }
    return text;
  }

 private:
  JsonPath(const JsonPath* parent, std::string_view key, std::size_t index, bool indexed) noexcept
      : parent_(parent), key_(key), index_(index), indexed_(indexed) {}

  const JsonPath* parent_{nullptr};
  std::string_view key_;
  std::size_t index_{0};
  bool indexed_{false};
};

inline constexpr std::uint64_t uint32_max = std::numeric_limits<std::uint32_t>::max();

// The six faces of a signal grid, in the member order documents use.
inline constexpr std::array<std::string_view, 6> boundary_names{
    "x_lower", "x_upper", "y_lower", "y_upper", "z_lower", "z_upper",
};

[[nodiscard]] inline JsonValue name_json(std::string_view name) {
  return JsonValue::string(std::string(name));
}

// Encodes each item of a range with the given function.
template <typename Items, typename Encode>
[[nodiscard]] JsonValue array_json(const Items& items, Encode&& encode) {
  JsonValue::Array values;
  values.reserve(items.size());
  for (const auto& item : items) {
    values.push_back(encode(item));
  }
  return JsonValue::array(std::move(values));
}

[[noreturn]] inline void fail(const JsonPath& path, std::string_view message) {
  throw JsonError(path.str() + ": " + std::string(message));
}

[[nodiscard]] inline const JsonValue::Array& expect_array(const JsonValue& value,
                                                          const JsonPath& path) {
  if (!value.is_array()) {
    fail(path, "expected an array");
  }
  return value.as_array();
}

[[nodiscard]] inline std::string quoted_keys(const std::vector<std::string_view>& keys) {
  std::string text = "[";
  for (std::size_t index = 0; index < keys.size(); ++index) {
    text += index == 0 ? "'" : ", '";
    text += keys[index];
    text += '\'';
  }
  return text + "]";
}

// Requires an object whose keys are exactly the given set.
inline const JsonValue& expect_object(const JsonValue& value, const JsonPath& path,
                                      std::initializer_list<std::string_view> keys) {
  if (!value.is_object()) {
    fail(path, "expected an object");
  }
  std::vector<std::string_view> missing;
  for (const auto key : keys) {
    if (value.find(key) == nullptr) {
      missing.push_back(key);
    }
  }
  if (!missing.empty()) {
    fail(path, "missing keys " + quoted_keys(missing));
  }
  if (value.as_object().size() != keys.size()) {
    std::vector<std::string_view> unknown;
    for (const auto& member : value.as_object()) {
      bool known = false;
      for (const auto key : keys) {
        known = known || key == member.key;
      }
      if (!known) {
        unknown.push_back(member.key);
      }
    }
    fail(path, "unknown keys " + quoted_keys(unknown));
  }
  return value;
}

[[nodiscard]] inline const std::string& expect_string(const JsonValue& value,
                                                      const JsonPath& path) {
  if (!value.is_string()) {
    fail(path, "expected a string");
  }
  return value.as_string();
}

[[nodiscard]] inline bool expect_boolean(const JsonValue& value, const JsonPath& path) {
  if (!value.is_boolean()) {
    fail(path, "expected a boolean");
  }
  return value.as_boolean();
}

// Parses canonical decimal digits into an unsigned 64-bit value, or reports
// that the magnitude does not fit.
[[nodiscard]] inline bool parse_uint64(std::string_view digits, std::uint64_t& out) noexcept {
  std::uint64_t value = 0;
  for (const char digit : digits) {
    const auto step = static_cast<std::uint64_t>(digit - '0');
    if (value > (std::numeric_limits<std::uint64_t>::max() - step) / 10U) {
      return false;
    }
    value = (value * 10U) + step;
  }
  out = value;
  return true;
}

[[nodiscard]] inline std::uint64_t expect_unsigned(const JsonValue& value, const JsonPath& path,
                                                   std::uint64_t minimum, std::uint64_t maximum) {
  if (!value.is_integer()) {
    fail(path, "expected an integer");
  }
  const std::string_view digits = value.as_integer_token();
  std::uint64_t result = 0;
  if (digits.front() == '-' || !parse_uint64(digits, result) || result < minimum ||
      result > maximum) {
    fail(path,
         "integer is outside [" + std::to_string(minimum) + ", " + std::to_string(maximum) + "]");
  }
  return result;
}

[[nodiscard]] inline std::int32_t expect_int32(const JsonValue& value, const JsonPath& path) {
  if (!value.is_integer()) {
    fail(path, "expected an integer");
  }
  std::string_view digits = value.as_integer_token();
  const bool negative = digits.front() == '-';
  if (negative) {
    digits.remove_prefix(1);
  }
  constexpr std::uint64_t positive_limit = 2147483647U;
  std::uint64_t magnitude = 0;
  if (!parse_uint64(digits, magnitude) || magnitude > positive_limit + (negative ? 1U : 0U)) {
    fail(path, "integer is outside [-2147483648, 2147483647]");
  }
  const auto wide = static_cast<std::int64_t>(magnitude);
  return static_cast<std::int32_t>(negative ? -wide : wide);
}

// Accepts an integer or a number and returns it as a finite float64.
[[nodiscard]] inline double expect_number(const JsonValue& value, const JsonPath& path) {
  if (value.is_number()) {
    return value.as_number();
  }
  if (!value.is_integer()) {
    fail(path, "expected a number");
  }
  // A 64-bit integer converts by accumulation, which is exact below 2^53 and
  // rounds above it.
  std::string_view digits = value.as_integer_token();
  const bool negative = digits.front() == '-';
  if (negative) {
    digits.remove_prefix(1);
  }
  double result = 0.0;
  for (const char digit : digits) {
    result = (result * 10.0) + static_cast<double>(digit - '0');
  }
  return negative ? -result : result;
}

// Accepts a number within the finite float32 range.
[[nodiscard]] inline float expect_float(const JsonValue& value, const JsonPath& path) {
  const double result = expect_number(value, path);
  if (std::abs(result) > static_cast<double>(std::numeric_limits<float>::max())) {
    fail(path, "number is outside the finite float32 range");
  }
  return static_cast<float>(result);
}

[[nodiscard]] inline std::vector<float> expect_floats(const JsonValue& value,
                                                      const JsonPath& path) {
  const auto& items = expect_array(value, path);
  std::vector<float> result;
  result.reserve(items.size());
  for (std::size_t index = 0; index < items.size(); ++index) {
    result.push_back(expect_float(items[index], path[index]));
  }
  return result;
}

[[nodiscard]] inline Vec3 expect_vec3(const JsonValue& value, const JsonPath& path) {
  const auto& items = expect_array(value, path);
  if (items.size() != 3) {
    fail(path, "expected exactly three values");
  }
  return {expect_float(items[0], path[0]), expect_float(items[1], path[1]),
          expect_float(items[2], path[2])};
}

// Decodes each item of an array with the given function, which receives the
// item and its path.
template <typename Decode>
[[nodiscard]] auto decode_array(const JsonValue& value, const JsonPath& path, Decode&& decode) {
  const auto& items = expect_array(value, path);
  std::vector<decltype(decode(items[0], path[0]))> result;
  result.reserve(items.size());
  for (std::size_t index = 0; index < items.size(); ++index) {
    result.push_back(decode(items[index], path[index]));
  }
  return result;
}

[[nodiscard]] inline JsonValue floats_to_json(const std::vector<float>& values) {
  JsonValue::Array items;
  items.reserve(values.size());
  for (const float value : values) {
    items.push_back(JsonValue::number(static_cast<double>(value)));
  }
  return JsonValue::array(std::move(items));
}

[[nodiscard]] inline JsonValue vec3_to_json(const Vec3& value) {
  return JsonValue::array({JsonValue::number(static_cast<double>(value.x)),
                           JsonValue::number(static_cast<double>(value.y)),
                           JsonValue::number(static_cast<double>(value.z))});
}

}  // namespace microsimulator::detail
