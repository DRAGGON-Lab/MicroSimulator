#pragma once

#include <string>

#include "microsimulator/json.hpp"

namespace microsimulator::detail {

// The {"name", "version"} object identifying this library as the writer of a
// document. MICROSIMULATOR_VERSION is the project version set by the build.
[[nodiscard]] inline JsonValue producer_json() {
  return JsonValue::object({
      {"name", JsonValue::string("microsimulator")},
      {"version", JsonValue::string(MICROSIMULATOR_VERSION)},
  });
}

}  // namespace microsimulator::detail
