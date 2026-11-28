# INPUT names one Metal source, or several to concatenate in order. A Metal
# library is compiled from source at runtime with no include path, so a source
# that shares helpers with another receives them by concatenation here. Each
# input becomes its own string literal, which keeps every literal within the
# length C++ compilers must support, and the literals are joined once at
# startup.
if(NOT DEFINED INPUT OR NOT DEFINED OUTPUT OR NOT DEFINED SYMBOL)
  message(FATAL_ERROR "EmbedMetalSource.cmake requires INPUT, OUTPUT, and SYMBOL")
endif()
if(NOT SYMBOL MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
  message(FATAL_ERROR "EmbedMetalSource.cmake received an invalid C++ symbol")
endif()

get_filename_component(MICROSIMULATOR_OUTPUT_DIRECTORY "${OUTPUT}" DIRECTORY)
file(MAKE_DIRECTORY "${MICROSIMULATOR_OUTPUT_DIRECTORY}")
file(WRITE "${OUTPUT}" "#pragma once\n\n#include <string>\n\nnamespace microsimulator::metal {\ninline const char* const ${SYMBOL} = [] {\n  static const std::string joined = std::string()")
foreach(MICROSIMULATOR_METAL_INPUT IN LISTS INPUT)
  file(READ "${MICROSIMULATOR_METAL_INPUT}" MICROSIMULATOR_METAL_SOURCE)
  if(MICROSIMULATOR_METAL_SOURCE MATCHES "METAL_SOURCE\\(" OR MICROSIMULATOR_METAL_SOURCE MATCHES "\\)METAL_SOURCE")
    message(FATAL_ERROR "Metal source ${MICROSIMULATOR_METAL_INPUT} contains the embedding delimiter")
  endif()
  file(APPEND "${OUTPUT}" " +\n      R\"METAL_SOURCE(")
  file(APPEND "${OUTPUT}" "${MICROSIMULATOR_METAL_SOURCE}")
  file(APPEND "${OUTPUT}" ")METAL_SOURCE\"")
endforeach()
file(APPEND "${OUTPUT}" [=[;
  return joined.c_str();
}();
}  // namespace microsimulator::metal
]=])
