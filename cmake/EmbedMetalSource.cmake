# INPUT names one Metal source, or several to concatenate in order. A Metal
# library is compiled from source at runtime with no include path, so a source
# that shares helpers with another receives them by concatenation here.
if(NOT DEFINED INPUT OR NOT DEFINED OUTPUT OR NOT DEFINED SYMBOL)
  message(FATAL_ERROR "EmbedMetalSource.cmake requires INPUT, OUTPUT, and SYMBOL")
endif()
if(NOT SYMBOL MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
  message(FATAL_ERROR "EmbedMetalSource.cmake received an invalid C++ symbol")
endif()

get_filename_component(CM_OUTPUT_DIRECTORY "${OUTPUT}" DIRECTORY)
file(MAKE_DIRECTORY "${CM_OUTPUT_DIRECTORY}")
file(WRITE "${OUTPUT}" "#pragma once\n\nnamespace cm::metal {\ninline constexpr char ${SYMBOL}[] = R\"CM_METAL(")
foreach(CM_METAL_INPUT IN LISTS INPUT)
  file(READ "${CM_METAL_INPUT}" CM_METAL_SOURCE)
  if(CM_METAL_SOURCE MATCHES "CM_METAL\\(" OR CM_METAL_SOURCE MATCHES "\\)CM_METAL")
    message(FATAL_ERROR "Metal source ${CM_METAL_INPUT} contains the embedding delimiter")
  endif()
  file(APPEND "${OUTPUT}" "${CM_METAL_SOURCE}")
endforeach()
file(APPEND "${OUTPUT}" [=[)CM_METAL";
}  // namespace cm::metal
]=])
