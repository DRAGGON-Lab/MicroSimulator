#include "microsimulator/json.hpp"

#include <bit>
#include <cassert>
#include <cstdint>
#include <string>
#include <string_view>

namespace {

using microsimulator::JsonError;
using microsimulator::JsonValue;

template <typename Function>
void assert_rejected(Function&& function, std::string_view fragment = {}) {
  bool rejected = false;
  try {
    function();
  } catch (const JsonError& error) {
    rejected = std::string_view(error.what()).find(fragment) != std::string_view::npos;
  }
  assert(rejected);
}

void assert_parse_rejected(std::string_view text, std::string_view fragment = {}) {
  assert_rejected([&] { static_cast<void>(microsimulator::parse_json(text)); }, fragment);
}

std::string canonical_number(std::uint64_t bits) {
  return microsimulator::canonical_json(JsonValue::number(std::bit_cast<double>(bits)));
}

void test_canonical_numbers() {
  // The IEEE 754 samples of RFC 8785 appendix B.
  assert(canonical_number(0x0000000000000000U) == "0");
  assert(canonical_number(0x8000000000000000U) == "0");
  assert(canonical_number(0x0000000000000001U) == "5e-324");
  assert(canonical_number(0x8000000000000001U) == "-5e-324");
  assert(canonical_number(0x7fefffffffffffffU) == "1.7976931348623157e+308");
  assert(canonical_number(0xffefffffffffffffU) == "-1.7976931348623157e+308");
  assert(canonical_number(0x4340000000000000U) == "9007199254740992");
  assert(canonical_number(0xc340000000000000U) == "-9007199254740992");
  assert(canonical_number(0x4430000000000000U) == "295147905179352830000");
  assert(canonical_number(0x44b52d02c7e14af5U) == "9.999999999999997e+22");
  assert(canonical_number(0x44b52d02c7e14af6U) == "1e+23");
  assert(canonical_number(0x44b52d02c7e14af7U) == "1.0000000000000001e+23");
  assert(canonical_number(0x444b1ae4d6e2ef4eU) == "999999999999999700000");
  assert(canonical_number(0x444b1ae4d6e2ef4fU) == "999999999999999900000");
  assert(canonical_number(0x444b1ae4d6e2ef50U) == "1e+21");
  assert(canonical_number(0x3eb0c6f7a0b5ed8cU) == "9.999999999999997e-7");
  assert(canonical_number(0x3eb0c6f7a0b5ed8dU) == "0.000001");
  assert(canonical_number(0x41b3de4355555553U) == "333333333.3333332");
  assert(canonical_number(0x41b3de4355555554U) == "333333333.33333325");
  assert(canonical_number(0x41b3de4355555555U) == "333333333.3333333");
  assert(canonical_number(0x41b3de4355555556U) == "333333333.3333334");
  assert(canonical_number(0x41b3de4355555557U) == "333333333.33333343");
  assert(canonical_number(0xbecbf647612f3696U) == "-0.0000033333333333333333");
  assert(canonical_number(0x43143ff3c1cb0959U) == "1424953923781206.2");
}

void test_canonical_document() {
  // The serialization sample of RFC 8785 section 3.2.
  const auto document = microsimulator::parse_json(R"({
    "numbers": [333333333.33333329, 1E30, 4.50, 2e-3, 0.000000000000000000000000001],
    "string": "\u20ac$\u000F\u000aA'\u0042\u0022\u005c\\\"\/",
    "literals": [null, true, false]
  })");
  assert(microsimulator::canonical_json(document) ==
         "{\"literals\":[null,true,false],"
         "\"numbers\":[333333333.3333333,1e+30,4.5,0.002,1e-27],"
         "\"string\":\"\xE2\x82\xAC$\\u000f\\nA'B\\\"\\\\\\\\\\\"/\"}");

  // Members sort by UTF-16 code units, so a supplementary-plane key precedes
  // a key in U+E000..U+FFFF although its UTF-8 bytes and code point are larger.
  const auto sorted = microsimulator::parse_json(
      R"({"\u20ac":1,"\r":2,"\ufb33":3,"1":4,"\ud83d\ude00":5,"\u0080":6,"\u00f6":7})");
  assert(microsimulator::canonical_json(sorted) ==
         "{\"\\r\":2,\"1\":4,\"\xC2\x80\":6,\"\xC3\xB6\":7,\"\xE2\x82\xAC\":1,"
         "\"\xF0\x9F\x98\x80\":5,\"\xEF\xAC\xB3\":3}");
  assert(sorted.find("\xF0\x9F\x98\x80") != nullptr);
  assert(sorted.find("\xEF\xAC\xB3")->as_integer_token() == "3");
  assert(sorted.find("missing") == nullptr);
}

void test_number_kinds() {
  const auto document = microsimulator::parse_json(R"([18446744073709551615, -0, 2.0, 1e2, 0.1,
        123456789012345678901234567890, -7])");
  const auto& items = document.as_array();
  assert(items[0].is_integer() && items[0].as_integer_token() == "18446744073709551615");
  assert(items[1].is_integer() && items[1].as_integer_token() == "0");
  assert(items[2].is_number() && items[2].as_number() == 2.0);
  assert(items[3].is_number() && items[3].as_number() == 100.0);
  assert(items[4].is_number() && items[4].as_number() == 0.1);
  assert(items[5].as_integer_token() == "123456789012345678901234567890");
  assert(items[6].as_integer_token() == "-7");

  // The readable form keeps each number's kind and exact value.
  const auto text = microsimulator::pretty_json(document);
  assert(text ==
         "[18446744073709551615, 0, 2.0, 100.0, 0.1, 123456789012345678901234567890, "
         "-7]\n");
  assert(microsimulator::parse_json(text) == document);
  assert(microsimulator::canonical_json(document) ==
         "[18446744073709551615,0,2,100,0.1,123456789012345678901234567890,-7]");

  assert(JsonValue::integer(std::int64_t{-42}).as_integer_token() == "-42");
  assert(JsonValue::integer(std::uint64_t{42}).as_integer_token() == "42");
  assert_rejected([] { static_cast<void>(JsonValue::integer_token("01")); });
  assert_rejected([] { static_cast<void>(JsonValue::integer_token("1.0")); });
  assert_rejected([] { static_cast<void>(JsonValue::integer_token("")); });

  // An integer beyond the float64 range can be neither built nor parsed, so
  // every integer that is written reads back.
  const std::string huge = "1" + std::string(400, '0');
  assert_rejected([&] { static_cast<void>(JsonValue::integer_token(huge)); },
                  "finite float64 range");
  assert_parse_rejected(huge, "number overflow");
  const std::string wide = "-1" + std::string(300, '0');
  assert(microsimulator::parse_json(wide).as_integer_token() == wide);
}

void test_pretty_round_trip() {
  const auto document = microsimulator::parse_json(
      R"({"b":{"nested":[{"x":1},[]],"empty":{}},"a":[1.5,"two",null,true],"text":"line\nbreak"})");
  const auto text = microsimulator::pretty_json(document);
  assert(text ==
         "{\n"
         "  \"a\": [1.5, \"two\", null, true],\n"
         "  \"b\": {\n"
         "    \"empty\": {},\n"
         "    \"nested\": [\n"
         "      {\n"
         "        \"x\": 1\n"
         "      },\n"
         "      []\n"
         "    ]\n"
         "  },\n"
         "  \"text\": \"line\\nbreak\"\n"
         "}\n");
  assert(microsimulator::parse_json(text) == document);
}

void test_rejections() {
  assert_parse_rejected("", "empty input");
  assert_parse_rejected("   ");
  assert_parse_rejected(R"({"a":1,"a":2})", "duplicate key \"a\"");
  assert_parse_rejected(R"({"a":{"b":1,"b":2}})", "duplicate key");
  assert_parse_rejected("NaN", "invalid literal");
  assert_parse_rejected("[Infinity]", "invalid literal");
  assert_parse_rejected("[-Infinity]", "invalid number");
  assert_parse_rejected("1e400", "number overflow");
  assert_parse_rejected("-1e400", "number overflow");
  assert_parse_rejected("[1,]");
  assert_parse_rejected("{\"a\":1,}");
  assert_parse_rejected("{'a':1}");
  assert_parse_rejected("[1] 2", "expected end of input");
  assert_parse_rejected("[1] // comment");
  assert_parse_rejected("[1, /* comment */ 2]");
  assert_parse_rejected("01");
  assert_parse_rejected("1.");
  assert_parse_rejected(".5");
  assert_parse_rejected("1e");
  assert_parse_rejected("+1");
  assert_parse_rejected("-");
  assert_parse_rejected("tru");
  assert_parse_rejected("\"unterminated");
  assert_parse_rejected("\"tab\tinside\"", "control character");
  assert_parse_rejected(R"("\x41")", "backslash");
  assert_parse_rejected(R"("\ud83d")", "surrogate");
  assert_parse_rejected(R"("\ude00")", "surrogate");
  assert_parse_rejected(R"("\ud83d\u0041")", "surrogate");
  assert_parse_rejected("\"\xC0\xAF\"", "UTF-8");
  assert_parse_rejected("\"\xED\xA0\x80\"", "UTF-8");
  assert_parse_rejected("\"\xF4\x90\x80\x80\"", "UTF-8");
  assert_parse_rejected("\"\xE2\x82\"", "UTF-8");
  assert(microsimulator::parse_json("\xEF\xBB\xBF[]").as_array().empty());

  // A rejection message is short printable ASCII whatever the input held.
  try {
    static_cast<void>(microsimulator::parse_json("\xFF\"" + std::string(5000, '\x80')));
    assert(false);
  } catch (const JsonError& error) {
    const std::string_view message = error.what();
    assert(message.size() < 300);
    for (const char character : message) {
      assert(character >= ' ' && character <= '~');
    }
  }

  // Underflow rounds toward zero rather than failing.
  assert(microsimulator::parse_json("1e-400").as_number() == 0.0);

  std::string deep(300, '[');
  deep.append(300, ']');
  assert_parse_rejected(deep, "too deep");
  std::string allowed(200, '[');
  allowed.append(200, ']');
  static_cast<void>(microsimulator::parse_json(allowed));

  assert_rejected(
      [] {
        static_cast<void>(
            JsonValue::number(std::bit_cast<double>(std::uint64_t{0x7ff8000000000000U})));
      },
      "non-finite");
  assert_rejected([] { static_cast<void>(JsonValue::string("\xFF")); }, "UTF-8");
  assert_rejected(
      [] { static_cast<void>(JsonValue::object({{"k", JsonValue{}}, {"k", JsonValue{}}})); },
      "duplicate key");
}

}  // namespace

int main() {
  test_canonical_numbers();
  test_canonical_document();
  test_number_kinds();
  test_pretty_round_trip();
  test_rejections();
  return 0;
}
