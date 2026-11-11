#include "microsimulator/json.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <nlohmann/json.hpp>
#include <string>
#include <system_error>
#include <utility>

namespace microsimulator {
namespace {

// UTF-16 code-unit order differs from UTF-8 byte order only where a
// supplementary-plane character (lead byte 0xF0 and above, encoded in UTF-16
// as surrogates below U+E000) meets a character in U+E000..U+FFFF (lead bytes
// 0xEE and 0xEF). Ranking every supplementary lead byte just below 0xEE
// restores the UTF-16 order without transcoding; two supplementary lead bytes
// share a rank and keep their byte order.
[[nodiscard]] int utf16_rank(unsigned char byte) noexcept {
  return byte >= 0xF0U ? (0xED * 2) + 1 : byte * 2;
}

[[nodiscard]] bool key_less(std::string_view left, std::string_view right) noexcept {
  const auto common = std::min(left.size(), right.size());
  for (std::size_t index = 0; index < common; ++index) {
    const auto a = static_cast<unsigned char>(left[index]);
    const auto b = static_cast<unsigned char>(right[index]);
    if (a != b) {
      const auto rank_a = utf16_rank(a);
      const auto rank_b = utf16_rank(b);
      return rank_a != rank_b ? rank_a < rank_b : a < b;
    }
  }
  return left.size() < right.size();
}

// Text is validated by the JSON library's serializer, which rejects every
// ill-formed UTF-8 sequence.
[[nodiscard]] bool valid_utf8(std::string_view text) {
  try {
    static_cast<void>(nlohmann::json(std::string(text)).dump());
  } catch (const nlohmann::json::exception&) {
    return false;
  }
  return true;
}

[[nodiscard]] bool valid_integer_token(std::string_view digits) noexcept {
  if (!digits.empty() && digits.front() == '-') {
    digits.remove_prefix(1);
  }
  if (digits.empty() || (digits.size() > 1 && digits.front() == '0')) {
    return false;
  }
  return std::ranges::all_of(digits, [](char digit) { return digit >= '0' && digit <= '9'; });
}

void append_string(std::string& out, std::string_view text) {
  static constexpr std::string_view hex = "0123456789abcdef";
  out += '"';
  for (const char character : text) {
    const auto byte = static_cast<unsigned char>(character);
    switch (character) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (byte < 0x20U) {
          out += "\\u00";
          out += hex[byte >> 4U];
          out += hex[byte & 0x0FU];
        } else {
          out += character;
        }
    }
  }
  out += '"';
}

// Writes the ECMAScript Number::toString form of a finite value. With
// keep_kind, an integral value gains a ".0" suffix so it reads back as a
// number rather than an integer.
void append_number(std::string& out, double value, bool keep_kind) {
  if (value == 0.0) {
    // The canonical form has one zero; the readable form keeps the sign.
    out += !keep_kind ? "0" : std::signbit(value) ? "-0.0" : "0.0";
    return;
  }
  std::array<char, 40> buffer{};
  const auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value,
                                    std::chars_format::scientific);
  if (result.ec != std::errc{}) {
    throw JsonError("number cannot be formatted");
  }
  std::string_view text(buffer.data(), static_cast<std::size_t>(result.ptr - buffer.data()));
  if (text.front() == '-') {
    out += '-';
    text.remove_prefix(1);
  }
  const auto exponent_mark = text.find('e');
  std::string digits(1, text.front());
  if (exponent_mark > 1) {
    digits.append(text.substr(2, exponent_mark - 2));
  }
  auto exponent_text = text.substr(exponent_mark + 1);
  const bool negative_exponent = exponent_text.front() == '-';
  exponent_text.remove_prefix(1);
  int exponent = 0;
  std::from_chars(exponent_text.data(), exponent_text.data() + exponent_text.size(), exponent);
  if (negative_exponent) {
    exponent = -exponent;
  }

  const int digit_count = static_cast<int>(digits.size());
  const int point = exponent + 1;
  if (digit_count <= point && point <= 21) {
    out += digits;
    out.append(static_cast<std::size_t>(point - digit_count), '0');
    if (keep_kind) {
      out += ".0";
    }
  } else if (0 < point && point <= 21) {
    out.append(digits, 0, static_cast<std::size_t>(point));
    out += '.';
    out.append(digits, static_cast<std::size_t>(point));
  } else if (-6 < point && point <= 0) {
    out += "0.";
    out.append(static_cast<std::size_t>(-point), '0');
    out += digits;
  } else {
    out += digits.front();
    if (digit_count > 1) {
      out += '.';
      out.append(digits, 1);
    }
    out += 'e';
    out += exponent < 0 ? '-' : '+';
    out += std::to_string(exponent < 0 ? -exponent : exponent);
  }
}

[[nodiscard]] bool scalar(const JsonValue& value) noexcept {
  return !value.is_array() && !value.is_object();
}

void write_scalar(std::string& out, const JsonValue& value, bool keep_kind) {
  if (value.is_null()) {
    out += "null";
  } else if (value.is_boolean()) {
    out += value.as_boolean() ? "true" : "false";
  } else if (value.is_integer()) {
    out += value.as_integer_token();
  } else if (value.is_number()) {
    append_number(out, value.as_number(), keep_kind);
  } else {
    append_string(out, value.as_string());
  }
}

void write_canonical(std::string& out, const JsonValue& value) {
  if (value.is_array()) {
    out += '[';
    bool first = true;
    for (const auto& item : value.as_array()) {
      if (!first) {
        out += ',';
      }
      first = false;
      write_canonical(out, item);
    }
    out += ']';
  } else if (value.is_object()) {
    out += '{';
    bool first = true;
    for (const auto& member : value.as_object()) {
      if (!first) {
        out += ',';
      }
      first = false;
      append_string(out, member.key);
      out += ':';
      write_canonical(out, member.value);
    }
    out += '}';
  } else {
    write_scalar(out, value, false);
  }
}

void write_pretty(std::string& out, const JsonValue& value, std::size_t indent) {
  if (!scalar(value) && indent / 2 >= JsonParseLimits{}.max_depth) {
    throw JsonError("nesting is too deep");
  }
  if (value.is_array()) {
    const auto& items = value.as_array();
    if (std::ranges::all_of(items, scalar)) {
      out += '[';
      bool first = true;
      for (const auto& item : items) {
        if (!first) {
          out += ", ";
        }
        first = false;
        write_scalar(out, item, true);
      }
      out += ']';
      return;
    }
    out += "[\n";
    bool first = true;
    for (const auto& item : items) {
      if (!first) {
        out += ",\n";
      }
      first = false;
      out.append(indent + 2, ' ');
      write_pretty(out, item, indent + 2);
    }
    out += '\n';
    out.append(indent, ' ');
    out += ']';
  } else if (value.is_object()) {
    const auto& members = value.as_object();
    if (members.empty()) {
      out += "{}";
      return;
    }
    out += "{\n";
    bool first = true;
    for (const auto& member : members) {
      if (!first) {
        out += ",\n";
      }
      first = false;
      out.append(indent + 2, ' ');
      append_string(out, member.key);
      out += ": ";
      write_pretty(out, member.value, indent + 2);
    }
    out += '\n';
    out.append(indent, ' ');
    out += '}';
  } else {
    write_scalar(out, value, true);
  }
}

}  // namespace

JsonValue::JsonValue(Storage storage) : storage_(std::move(storage)) {}

JsonValue JsonValue::boolean(bool value) { return JsonValue(Storage(value)); }

JsonValue JsonValue::integer(std::int64_t value) {
  return JsonValue(Storage(Integer{std::to_string(value)}));
}

JsonValue JsonValue::integer(std::uint64_t value) {
  return JsonValue(Storage(Integer{std::to_string(value)}));
}

JsonValue JsonValue::integer_token(std::string digits) {
  if (!valid_integer_token(digits)) {
    throw JsonError("malformed JSON integer");
  }
  if (digits == "-0") {
    digits = "0";
  }
  // The parser reads an integer too long for 64 bits through float64, so a
  // magnitude beyond that range could be written but never read back.
  double magnitude = 0.0;
  for (const char digit : digits) {
    if (digit != '-') {
      magnitude = (magnitude * 10.0) + static_cast<double>(digit - '0');
    }
  }
  if (!std::isfinite(magnitude)) {
    throw JsonError("integer is outside the finite float64 range");
  }
  return JsonValue(Storage(Integer{std::move(digits)}));
}

JsonValue JsonValue::number(double value) {
  if (!std::isfinite(value)) {
    throw JsonError("non-finite JSON number");
  }
  return JsonValue(Storage(value));
}

JsonValue JsonValue::string(std::string value) {
  if (!valid_utf8(value)) {
    throw JsonError("string is not valid UTF-8");
  }
  return JsonValue(Storage(std::move(value)));
}

JsonValue JsonValue::array(Array values) { return JsonValue(Storage(std::move(values))); }

JsonValue JsonValue::object(Object members) {
  for (const auto& member : members) {
    if (!valid_utf8(member.key)) {
      throw JsonError("object key is not valid UTF-8");
    }
  }
  return trusted_object(std::move(members));
}

JsonValue JsonValue::trusted_string(std::string value) {
  return JsonValue(Storage(std::move(value)));
}

JsonValue JsonValue::trusted_object(Object members) {
  std::ranges::sort(members, key_less, &JsonMember::key);
  const auto duplicate = std::ranges::adjacent_find(members, {}, &JsonMember::key);
  if (duplicate != members.end()) {
    std::string quoted;
    append_string(quoted, duplicate->key);
    throw JsonError("duplicate key " + quoted);
  }
  return JsonValue(Storage(std::move(members)));
}

bool JsonValue::is_null() const noexcept {
  return std::holds_alternative<std::monostate>(storage_);
}
bool JsonValue::is_boolean() const noexcept { return std::holds_alternative<bool>(storage_); }
bool JsonValue::is_integer() const noexcept { return std::holds_alternative<Integer>(storage_); }
bool JsonValue::is_number() const noexcept { return std::holds_alternative<double>(storage_); }
bool JsonValue::is_string() const noexcept { return std::holds_alternative<std::string>(storage_); }
bool JsonValue::is_array() const noexcept { return std::holds_alternative<Array>(storage_); }
bool JsonValue::is_object() const noexcept { return std::holds_alternative<Object>(storage_); }

bool JsonValue::as_boolean() const {
  if (const auto* value = std::get_if<bool>(&storage_)) {
    return *value;
  }
  throw std::logic_error("JSON value is not a boolean");
}

const std::string& JsonValue::as_integer_token() const {
  if (const auto* value = std::get_if<Integer>(&storage_)) {
    return value->digits;
  }
  throw std::logic_error("JSON value is not an integer");
}

double JsonValue::as_number() const {
  if (const auto* value = std::get_if<double>(&storage_)) {
    return *value;
  }
  throw std::logic_error("JSON value is not a number");
}

const std::string& JsonValue::as_string() const {
  if (const auto* value = std::get_if<std::string>(&storage_)) {
    return *value;
  }
  throw std::logic_error("JSON value is not a string");
}

const JsonValue::Array& JsonValue::as_array() const {
  if (const auto* value = std::get_if<Array>(&storage_)) {
    return *value;
  }
  throw std::logic_error("JSON value is not an array");
}

const JsonValue::Object& JsonValue::as_object() const {
  if (const auto* value = std::get_if<Object>(&storage_)) {
    return *value;
  }
  throw std::logic_error("JSON value is not an object");
}

const JsonValue* JsonValue::find(std::string_view key) const {
  const auto& members = as_object();
  const auto found = std::ranges::lower_bound(members, key, key_less, &JsonMember::key);
  if (found == members.end() || found->key != key) {
    return nullptr;
  }
  return &found->value;
}

bool JsonValue::operator==(const JsonValue& other) const { return storage_ == other.storage_; }

namespace detail {

// Receives the JSON library's parse events and assembles a JsonValue tree.
// The library tokenizes and validates the text, including its UTF-8 and
// escapes; this class adds the nesting limit, the duplicate-key check, and
// exact storage of integers.
class JsonBuilder {
 public:
  using number_integer_t = nlohmann::json::number_integer_t;
  using number_unsigned_t = nlohmann::json::number_unsigned_t;
  using number_float_t = nlohmann::json::number_float_t;
  using string_t = nlohmann::json::string_t;
  using binary_t = nlohmann::json::binary_t;

  explicit JsonBuilder(const JsonParseLimits& limits) : limits_(limits) {}

  [[nodiscard]] JsonValue take() { return std::move(root_); }

  bool null() { return add(JsonValue{}); }
  bool boolean(bool value) { return add(JsonValue::boolean(value)); }
  bool number_integer(number_integer_t value) {
    return add(JsonValue::integer(std::int64_t{value}));
  }
  bool number_unsigned(number_unsigned_t value) {
    return add(JsonValue::integer(std::uint64_t{value}));
  }
  // The library reports an integer too long for 64 bits as a float together
  // with its source text, which is kept so the value stays exact.
  bool number_float(number_float_t value, const string_t& text) {
    if (text.find_first_not_of("-0123456789") == string_t::npos) {
      return add(JsonValue::integer_token(text));
    }
    return add(JsonValue::number(value));
  }
  bool string(string_t& value) { return add(JsonValue::trusted_string(std::move(value))); }
  bool binary(binary_t& /*value*/) { throw JsonError("invalid JSON: unexpected binary value"); }

  bool start_object(std::size_t /*size*/) { return open(true); }
  bool key(string_t& value) {
    frames_.back().key = std::move(value);
    return true;
  }
  bool end_object() {
    auto members = std::move(frames_.back().members);
    frames_.pop_back();
    return add(JsonValue::trusted_object(std::move(members)));
  }

  bool start_array(std::size_t /*size*/) { return open(false); }
  bool end_array() {
    auto items = std::move(frames_.back().items);
    frames_.pop_back();
    return add(JsonValue::array(std::move(items)));
  }

  bool parse_error(std::size_t /*position*/, const std::string& /*last_token*/,
                   const nlohmann::json::exception& error) {
    // The library quotes the offending input, which may be long or may not
    // be text at all. The message keeps a bounded, printable ASCII rendering.
    constexpr std::size_t max_detail = 240;
    std::string message = "invalid JSON: ";
    for (const char character : std::string_view(error.what()).substr(0, max_detail)) {
      message += character >= ' ' && character <= '~' ? character : '?';
    }
    throw JsonError(message);
  }

 private:
  struct Frame {
    bool object{false};
    std::string key;
    JsonValue::Array items;
    JsonValue::Object members;
  };

  bool open(bool object) {
    if (frames_.size() >= limits_.max_depth) {
      throw JsonError("invalid JSON: nesting is too deep");
    }
    frames_.push_back({.object = object, .key = {}, .items = {}, .members = {}});
    return true;
  }

  bool add(JsonValue value) {
    if (frames_.empty()) {
      root_ = std::move(value);
    } else if (frames_.back().object) {
      frames_.back().members.push_back({std::move(frames_.back().key), std::move(value)});
    } else {
      frames_.back().items.push_back(std::move(value));
    }
    return true;
  }

  JsonParseLimits limits_;
  std::vector<Frame> frames_;
  JsonValue root_;
};

}  // namespace detail

JsonValue parse_json(std::string_view text, const JsonParseLimits& limits) {
  detail::JsonBuilder builder(limits);
  nlohmann::json::sax_parse(text.begin(), text.end(), &builder);
  return builder.take();
}

std::string canonical_json(const JsonValue& value) {
  std::string out;
  write_canonical(out, value);
  return out;
}

std::string pretty_json(const JsonValue& value) {
  std::string out;
  write_pretty(out, value, 0);
  out += '\n';
  return out;
}

}  // namespace microsimulator
