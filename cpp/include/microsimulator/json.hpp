#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace microsimulator {

// Raised when text is not strict JSON or a value cannot be represented as JSON.
class JsonError : public std::invalid_argument {
 public:
  using std::invalid_argument::invalid_argument;
};

struct JsonMember;

namespace detail {
class JsonBuilder;
}  // namespace detail

// An immutable-by-convention JSON document tree.
//
// Integers and other numbers are distinct kinds. An integer keeps its exact
// decimal digits at any magnitude within the finite float64 range, so 64-bit
// identifiers and larger integers in opaque payloads survive a round trip. Object members are
// unique and ordered by key in UTF-16 code-unit order, the RFC 8785 member order.
class JsonValue {
 public:
  using Array = std::vector<JsonValue>;
  using Object = std::vector<JsonMember>;

  JsonValue() = default;

  [[nodiscard]] static JsonValue boolean(bool value);
  [[nodiscard]] static JsonValue integer(std::int64_t value);
  [[nodiscard]] static JsonValue integer(std::uint64_t value);
  // Accepts an optional minus sign followed by decimal digits without a
  // leading zero. Negative zero is stored as zero. Throws JsonError for a
  // magnitude beyond the finite float64 range.
  [[nodiscard]] static JsonValue integer_token(std::string digits);
  // Throws JsonError for a non-finite value.
  [[nodiscard]] static JsonValue number(double value);
  // Throws JsonError unless the text is well-formed UTF-8.
  [[nodiscard]] static JsonValue string(std::string value);
  [[nodiscard]] static JsonValue array(Array values);
  // Orders the members and throws JsonError for a duplicate key.
  [[nodiscard]] static JsonValue object(Object members);

  [[nodiscard]] bool is_null() const noexcept;
  [[nodiscard]] bool is_boolean() const noexcept;
  [[nodiscard]] bool is_integer() const noexcept;
  [[nodiscard]] bool is_number() const noexcept;
  [[nodiscard]] bool is_string() const noexcept;
  [[nodiscard]] bool is_array() const noexcept;
  [[nodiscard]] bool is_object() const noexcept;

  // Each accessor throws std::logic_error when the value has another kind.
  [[nodiscard]] bool as_boolean() const;
  [[nodiscard]] const std::string& as_integer_token() const;
  [[nodiscard]] double as_number() const;
  [[nodiscard]] const std::string& as_string() const;
  [[nodiscard]] const Array& as_array() const;
  [[nodiscard]] const Object& as_object() const;

  // The member with this key, or nullptr. Throws std::logic_error unless the
  // value is an object.
  [[nodiscard]] const JsonValue* find(std::string_view key) const;

  [[nodiscard]] bool operator==(const JsonValue& other) const;

 private:
  struct Integer {
    std::string digits;
    [[nodiscard]] bool operator==(const Integer&) const = default;
  };
  using Storage = std::variant<std::monostate, bool, Integer, double, std::string, Array, Object>;

  friend class detail::JsonBuilder;

  explicit JsonValue(Storage storage);
  // Construction from parser output, whose text is already validated.
  [[nodiscard]] static JsonValue trusted_string(std::string value);
  [[nodiscard]] static JsonValue trusted_object(Object members);

  Storage storage_;
};

struct JsonMember {
  std::string key;
  JsonValue value;

  [[nodiscard]] bool operator==(const JsonMember&) const = default;
};

struct JsonParseLimits {
  std::size_t max_depth{256};
};

// Parses one strict JSON document. Rejects duplicate object keys, the
// non-finite literals NaN and Infinity, numbers outside the finite float64
// range, malformed UTF-8, unpaired surrogate escapes, nesting beyond the
// limit, and anything but whitespace after the document. A leading UTF-8
// byte order mark is skipped.
[[nodiscard]] JsonValue parse_json(std::string_view text, const JsonParseLimits& limits = {});

// The canonical encoding that integrity digests cover: RFC 8785 member order,
// string escaping, and ECMAScript number formatting, with no insignificant
// whitespace. An integer is written as its exact decimal digits, which equals
// the RFC 8785 form whenever the integer is exactly representable as float64.
[[nodiscard]] std::string canonical_json(const JsonValue& value);

// A human-readable encoding with two-space indentation and a trailing newline.
// A number that is not an integer always carries a fraction or an exponent,
// and a negative zero keeps its sign, so parsing the text restores each value
// exactly. Throws JsonError for nesting beyond the default parse limit, so
// every document this writes can be read back.
[[nodiscard]] std::string pretty_json(const JsonValue& value);

}  // namespace microsimulator
