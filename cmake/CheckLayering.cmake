# Checks that the engine in cpp/core reaches per-element compute only through
# the provider interfaces of microsimulator/backend.hpp.
#
# 1. No file under cpp/core includes a backend's own headers.
# 2. Host loops over cells, sites, faces, or contacts (detail::parallel_for and
#    detail::parallel_reduce) and calls to CPU kernels appear only in the
#    files listed below. Those files hold the compute that runs on the host
#    for every backend; any other file that gains such a loop fails this
#    check, so the loop becomes a provider method with a kernel on every
#    backend.
#
# Run as: cmake -DSOURCE_DIR=<repository root> -P cmake/CheckLayering.cmake
cmake_minimum_required(VERSION 3.25)

if(NOT DEFINED SOURCE_DIR)
  message(FATAL_ERROR "CheckLayering.cmake requires SOURCE_DIR")
endif()

# Each file follows a comment naming the host compute it holds.
set(MICROSIMULATOR_HOST_COMPUTE
  # Contact dynamics assembles rows and bodies, moves cells, finds wall
  # contacts, and evaluates the solved response on the host.
  "cpp/core/contact_dynamics.cpp"
  # Contact graphs are built on the host from each backend's contact list.
  "cpp/core/contact_graph.cpp"
  # Flow drift samples the velocity field for every cell on the host.
  "cpp/core/simulation.cpp"
  # The CPU backend's signal transport.
  "cpp/core/signals.cpp"
  # Flow-system assembly and the reduction of shallow flow, before a solve.
  "cpp/core/flow_system.hpp"
  # The thread pool itself.
  "cpp/core/parallel.hpp"
  "cpp/core/parallel.cpp"
)

file(GLOB_RECURSE MICROSIMULATOR_CORE_SOURCES
  "${SOURCE_DIR}/cpp/core/*.cpp" "${SOURCE_DIR}/cpp/core/*.hpp")
set(MICROSIMULATOR_LAYERING_FAILURES "")
foreach(path IN LISTS MICROSIMULATOR_CORE_SOURCES)
  file(RELATIVE_PATH relative "${SOURCE_DIR}" "${path}")
  file(READ "${path}" text)
  if(text MATCHES "#include \"(cpu|cuda|metal)/")
    list(APPEND MICROSIMULATOR_LAYERING_FAILURES
      "${relative} includes a backend header (${CMAKE_MATCH_0})")
  endif()
  if(relative IN_LIST MICROSIMULATOR_HOST_COMPUTE)
    continue()
  endif()
  if(text MATCHES "parallel_(for|reduce)\\(")
    list(APPEND MICROSIMULATOR_LAYERING_FAILURES
      "${relative} runs a host loop (${CMAKE_MATCH_0}) outside a provider")
  endif()
  if(text MATCHES "[A-Za-z_]+_cpu\\(")
    list(APPEND MICROSIMULATOR_LAYERING_FAILURES
      "${relative} calls a CPU kernel (${CMAKE_MATCH_0}) outside a provider")
  endif()
endforeach()

if(MICROSIMULATOR_LAYERING_FAILURES)
  list(JOIN MICROSIMULATOR_LAYERING_FAILURES "\n  " detail)
  message(FATAL_ERROR "engine layering violations:\n  ${detail}")
endif()
