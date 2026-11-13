cmake_minimum_required(VERSION 3.25)

project(
  MicroSimulator
  VERSION 0.1.0
  DESCRIPTION "Microfluidic simulation of microbial populations, flow, and solute transport"
  LANGUAGES CXX
)

option(MICROSIMULATOR_BUILD_PYTHON "Build the nanobind Python extension" ON)
option(MICROSIMULATOR_BUILD_TESTS "Build C++ tests" ON)
option(MICROSIMULATOR_ENABLE_CUDA "Build the native NVIDIA CUDA backend" OFF)

set(MICROSIMULATOR_METAL_DEFAULT OFF)
if(APPLE)
  set(MICROSIMULATOR_METAL_DEFAULT ON)
endif()
option(MICROSIMULATOR_ENABLE_METAL "Build the native Apple Metal backend" ${MICROSIMULATOR_METAL_DEFAULT})

add_library(microsimulator_core STATIC
  cpp/core/atomic_file.cpp
  cpp/core/channels.cpp
  cpp/core/checkpoint.cpp
  cpp/core/checkpoint_file.cpp
  cpp/core/contact_dynamics.cpp
  cpp/core/contact_graph.cpp
  cpp/core/constraints.cpp
  cpp/core/coupled_rates.cpp
  cpp/core/json.cpp
  cpp/core/mechanics_integration.cpp
  cpp/core/names.cpp
  cpp/core/occupancy.cpp
  cpp/core/scene.cpp
  cpp/core/sha256.cpp
  cpp/core/signals.cpp
  cpp/core/species.cpp
  cpp/core/world_state.cpp
  cpp/core/simulation.cpp
  cpp/cpu/cpu_backend.cpp
  cpp/cpu/cpu_contacts.cpp
  cpp/cpu/cpu_constraints.cpp
  cpp/cpu/cpu_contact_dynamics.cpp
  cpp/cpu/cpu_coupled.cpp
  cpp/cpu/cpu_flow.cpp
  cpp/cpu/cpu_mechanics.cpp
  cpp/cpu/cpu_species.cpp
)
add_library(microsimulator::core ALIAS microsimulator_core)

target_compile_features(microsimulator_core PUBLIC cxx_std_23)
target_include_directories(microsimulator_core PUBLIC cpp/include PRIVATE cpp)
# Vendored single-header libraries, used only inside core sources.
target_include_directories(microsimulator_core SYSTEM PRIVATE cpp/third_party)
set_target_properties(microsimulator_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
# Documents record the version of the library that wrote them. A Python package
# build supplies the package version; a plain CMake build uses the project's.
if(DEFINED SKBUILD_PROJECT_VERSION)
  set(MICROSIMULATOR_PRODUCER_VERSION "${SKBUILD_PROJECT_VERSION}")
else()
  set(MICROSIMULATOR_PRODUCER_VERSION "${PROJECT_VERSION}")
endif()
target_compile_definitions(
  microsimulator_core PRIVATE MICROSIMULATOR_VERSION="${MICROSIMULATOR_PRODUCER_VERSION}"
)

if(MICROSIMULATOR_ENABLE_METAL)
  if(NOT APPLE)
    message(FATAL_ERROR "MICROSIMULATOR_ENABLE_METAL requires an Apple platform")
  endif()

  enable_language(OBJCXX)
  find_library(MICROSIMULATOR_FOUNDATION_FRAMEWORK Foundation REQUIRED)
  find_library(MICROSIMULATOR_METAL_FRAMEWORK Metal REQUIRED)

  set(MICROSIMULATOR_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

  # Every shader source opens with the prelude, the shared definition lists it
  # gives meaning to, and the shared structure layouts.
  set(MICROSIMULATOR_METAL_SHARED_SOURCES
    cpp/metal/kernels/prelude.metal
    cpp/shared/constants.def
    cpp/include/microsimulator/rate_ops.def
    cpp/shared/device_types.h
  )

  # Embeds the shared sources followed by the given kernel sources, in order,
  # as the string microsimulator::metal::<name>_source in microsimulator/metal/<name>_source.hpp.
  function(microsimulator_embed_metal_source name)
    set(sources ${MICROSIMULATOR_METAL_SHARED_SOURCES} ${ARGN})
    list(TRANSFORM sources PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/" OUTPUT_VARIABLE inputs)
    set(output "${MICROSIMULATOR_GENERATED_DIR}/microsimulator/metal/${name}_source.hpp")
    add_custom_command(
      OUTPUT "${output}"
      COMMAND "${CMAKE_COMMAND}"
        "-DINPUT=${inputs}"
        "-DOUTPUT=${output}"
        "-DSYMBOL=${name}_source"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedMetalSource.cmake"
      DEPENDS ${sources} cmake/EmbedMetalSource.cmake
      VERBATIM
    )
    target_sources(microsimulator_core PRIVATE "${output}")
  endfunction()

  microsimulator_embed_metal_source(growth cpp/metal/kernels/growth.metal)
  microsimulator_embed_metal_source(species cpp/metal/kernels/species.metal)
  microsimulator_embed_metal_source(contacts cpp/metal/kernels/contacts.metal)
  microsimulator_embed_metal_source(mechanics cpp/metal/kernels/mechanics.metal)
  microsimulator_embed_metal_source(signals
    cpp/metal/kernels/grid_transport.metal cpp/metal/kernels/signals.metal)
  microsimulator_embed_metal_source(coupled_rates
    cpp/metal/kernels/grid_transport.metal cpp/metal/kernels/coupled_rates.metal)
  microsimulator_embed_metal_source(flow cpp/metal/kernels/flow_grid.metal cpp/metal/kernels/flow.metal)
  microsimulator_embed_metal_source(depth_flow
    cpp/metal/kernels/flow_grid.metal cpp/metal/kernels/depth_flow.metal)

  set(MICROSIMULATOR_METAL_OCCUPANCY_HEADER "${MICROSIMULATOR_GENERATED_DIR}/microsimulator/metal/occupancy_source.hpp")
  add_custom_command(
    OUTPUT "${MICROSIMULATOR_METAL_OCCUPANCY_HEADER}"
    COMMAND "${CMAKE_COMMAND}"
      "-DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/cpp/metal/kernels/occupancy.metal"
      "-DOUTPUT=${MICROSIMULATOR_METAL_OCCUPANCY_HEADER}"
      "-DSYMBOL=occupancy_source"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedMetalSource.cmake"
    DEPENDS cpp/metal/kernels/occupancy.metal cmake/EmbedMetalSource.cmake
    VERBATIM
  )

  target_sources(microsimulator_core PRIVATE
    cpp/metal/metal_backend.mm
    cpp/metal/metal_flow.mm
    cpp/metal/metal_occupancy.mm
    "${MICROSIMULATOR_METAL_OCCUPANCY_HEADER}"
  )
  target_include_directories(microsimulator_core PRIVATE "${MICROSIMULATOR_GENERATED_DIR}")
  target_compile_definitions(microsimulator_core PRIVATE MICROSIMULATOR_HAS_METAL=1)
  target_compile_options(microsimulator_core PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
  target_link_libraries(microsimulator_core PRIVATE
    "${MICROSIMULATOR_FOUNDATION_FRAMEWORK}"
    "${MICROSIMULATOR_METAL_FRAMEWORK}"
  )
  set_target_properties(microsimulator_core PROPERTIES OBJCXX_STANDARD 23 OBJCXX_STANDARD_REQUIRED ON)
endif()

if(MICROSIMULATOR_ENABLE_CUDA)
  if(APPLE)
    message(FATAL_ERROR "MICROSIMULATOR_ENABLE_CUDA requires a platform supported by the CUDA Toolkit")
  endif()

  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  target_sources(microsimulator_core PRIVATE
    cpp/cuda/cuda_backend.cu
    cpp/cuda/cuda_flow.cu
    cpp/cuda/cuda_occupancy.cu
    cpp/cuda/kernels/occupancy.cu
    cpp/cuda/kernels/contacts.cu
    cpp/cuda/kernels/coupled_rates.cu
    cpp/cuda/kernels/growth.cu
    cpp/cuda/kernels/flow.cu
    cpp/cuda/kernels/mechanics.cu
    cpp/cuda/kernels/signals.cu
    cpp/cuda/kernels/species.cu
  )
  target_compile_definitions(microsimulator_core PRIVATE MICROSIMULATOR_HAS_CUDA=1)
  set_source_files_properties(cpp/cuda/kernels/occupancy.cu PROPERTIES COMPILE_OPTIONS "--fmad=false")
  target_link_libraries(microsimulator_core PRIVATE CUDA::cudart)
  set_target_properties(microsimulator_core PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
  )
endif()

if(MSVC)
  target_compile_options(microsimulator_core PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:/W4;/permissive->"
  )
else()
  target_compile_options(microsimulator_core PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wconversion>"
    "$<$<COMPILE_LANGUAGE:OBJCXX>:-Wall;-Wextra;-Wpedantic;-Wconversion>"
  )
endif()

if(MICROSIMULATOR_BUILD_PYTHON)
  find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(nanobind CONFIG REQUIRED)

  nanobind_add_module(
    _core
    cpp/python/bindings.cpp
    cpp/python/occupancy_bindings.cpp
    cpp/python/serialization_bindings.cpp
  )
  target_link_libraries(_core PRIVATE microsimulator::core)
  install(TARGETS _core LIBRARY DESTINATION microsimulator)
endif()

if(MICROSIMULATOR_BUILD_TESTS)
  include(CTest)
  add_executable(microsimulator_occupancy_conformance_test tests/conformance/occupancy_conformance_test.cpp)
  target_link_libraries(microsimulator_occupancy_conformance_test PRIVATE microsimulator::core)
  add_test(NAME occupancy_conformance COMMAND microsimulator_occupancy_conformance_test)
  set_tests_properties(occupancy_conformance PROPERTIES LABELS conformance SKIP_RETURN_CODE 77)
  add_executable(microsimulator_world_state_test tests/cpp/world_state_test.cpp)
  target_link_libraries(microsimulator_world_state_test PRIVATE microsimulator::core)
  add_test(NAME world_state COMMAND microsimulator_world_state_test)

  add_executable(microsimulator_contact_geometry_test tests/cpp/contact_geometry_test.cpp)
  target_link_libraries(microsimulator_contact_geometry_test PRIVATE microsimulator::core)
  add_test(NAME contact_geometry COMMAND microsimulator_contact_geometry_test)

  add_executable(microsimulator_constraint_geometry_test tests/cpp/constraint_geometry_test.cpp)
  target_link_libraries(microsimulator_constraint_geometry_test PRIVATE microsimulator::core)
  add_test(NAME constraint_geometry COMMAND microsimulator_constraint_geometry_test)

  add_executable(microsimulator_mechanics_operator_test tests/cpp/mechanics_operator_test.cpp)
  target_link_libraries(microsimulator_mechanics_operator_test PRIVATE microsimulator::core)
  add_test(NAME mechanics_operator COMMAND microsimulator_mechanics_operator_test)

  add_executable(microsimulator_contact_dynamics_test tests/cpp/contact_dynamics_test.cpp)
  target_link_libraries(microsimulator_contact_dynamics_test PRIVATE microsimulator::core)
  add_test(NAME contact_dynamics COMMAND microsimulator_contact_dynamics_test)

  add_executable(microsimulator_mechanics_integration_test tests/cpp/mechanics_integration_test.cpp)
  target_link_libraries(microsimulator_mechanics_integration_test PRIVATE microsimulator::core)
  add_test(NAME mechanics_integration COMMAND microsimulator_mechanics_integration_test)

  add_executable(microsimulator_species_test tests/cpp/species_test.cpp)
  target_link_libraries(microsimulator_species_test PRIVATE microsimulator::core)
  add_test(NAME species COMMAND microsimulator_species_test)

  add_executable(microsimulator_signal_grid_test tests/cpp/signal_grid_test.cpp)
  target_link_libraries(microsimulator_signal_grid_test PRIVATE microsimulator::core)
  add_test(NAME signal_grid COMMAND microsimulator_signal_grid_test)

  add_executable(microsimulator_coupled_rates_test tests/cpp/coupled_rates_test.cpp)
  target_link_libraries(microsimulator_coupled_rates_test PRIVATE microsimulator::core)
  add_test(NAME coupled_rates COMMAND microsimulator_coupled_rates_test)

  add_executable(microsimulator_flow_test tests/cpp/flow_test.cpp)
  target_link_libraries(microsimulator_flow_test PRIVATE microsimulator::core)
  add_test(NAME flow COMMAND microsimulator_flow_test)

  add_executable(microsimulator_json_test tests/cpp/json_test.cpp)
  target_link_libraries(microsimulator_json_test PRIVATE microsimulator::core)
  add_test(NAME json COMMAND microsimulator_json_test)

  add_executable(microsimulator_sha256_test tests/cpp/sha256_test.cpp)
  target_link_libraries(microsimulator_sha256_test PRIVATE microsimulator::core)
  add_test(NAME sha256 COMMAND microsimulator_sha256_test)

  add_executable(microsimulator_atomic_file_test tests/cpp/atomic_file_test.cpp)
  target_link_libraries(microsimulator_atomic_file_test PRIVATE microsimulator::core)
  add_test(NAME atomic_file COMMAND microsimulator_atomic_file_test)

  add_executable(microsimulator_checkpoint_file_test tests/cpp/checkpoint_file_test.cpp)
  target_link_libraries(microsimulator_checkpoint_file_test PRIVATE microsimulator::core)
  add_test(NAME checkpoint_file COMMAND microsimulator_checkpoint_file_test)

  add_executable(microsimulator_scene_test tests/cpp/scene_test.cpp)
  target_link_libraries(microsimulator_scene_test PRIVATE microsimulator::core)
  add_test(
    NAME scene
    COMMAND microsimulator_scene_test
            ${CMAKE_CURRENT_SOURCE_DIR}/viewer/tests/fixtures/channels-v3.scene.json
  )

  add_executable(microsimulator_checkpoint_test tests/cpp/checkpoint_test.cpp)
  target_link_libraries(microsimulator_checkpoint_test PRIVATE microsimulator::core)
  target_include_directories(microsimulator_checkpoint_test PRIVATE tests/conformance)
  add_test(NAME checkpoint COMMAND microsimulator_checkpoint_test)
  set_tests_properties(checkpoint PROPERTIES LABELS conformance)

  add_executable(microsimulator_growth_conformance_test tests/conformance/growth_conformance_test.cpp)
  target_link_libraries(microsimulator_growth_conformance_test PRIVATE microsimulator::core)
  add_test(NAME growth_conformance COMMAND microsimulator_growth_conformance_test)
  set_tests_properties(growth_conformance PROPERTIES LABELS conformance)

  add_executable(microsimulator_backend_contract_conformance_test
    tests/conformance/backend_contract_conformance_test.cpp
  )
  target_link_libraries(microsimulator_backend_contract_conformance_test PRIVATE microsimulator::core)
  add_test(NAME backend_contract_conformance COMMAND microsimulator_backend_contract_conformance_test)
  set_tests_properties(backend_contract_conformance PROPERTIES LABELS conformance)

  add_executable(microsimulator_species_conformance_test
    tests/conformance/species_conformance_test.cpp
  )
  target_link_libraries(microsimulator_species_conformance_test PRIVATE microsimulator::core)
  add_test(NAME species_conformance COMMAND microsimulator_species_conformance_test)
  set_tests_properties(species_conformance PROPERTIES LABELS conformance)

  add_executable(microsimulator_signal_grid_conformance_test
    tests/conformance/signal_grid_conformance_test.cpp
  )
  target_link_libraries(microsimulator_signal_grid_conformance_test PRIVATE microsimulator::core)
  add_test(NAME signal_grid_conformance COMMAND microsimulator_signal_grid_conformance_test)
  set_tests_properties(signal_grid_conformance PROPERTIES LABELS conformance)

  add_executable(microsimulator_coupled_rates_conformance_test
    tests/conformance/coupled_rates_conformance_test.cpp
  )
  target_link_libraries(microsimulator_coupled_rates_conformance_test PRIVATE microsimulator::core)
  add_test(NAME coupled_rates_conformance COMMAND microsimulator_coupled_rates_conformance_test)
  set_tests_properties(coupled_rates_conformance PROPERTIES LABELS conformance)

  add_executable(microsimulator_flow_conformance_test
    tests/conformance/flow_conformance_test.cpp
  )
  target_link_libraries(microsimulator_flow_conformance_test PRIVATE microsimulator::core)
  target_include_directories(microsimulator_flow_conformance_test PRIVATE tests/conformance)
  add_test(NAME flow_conformance COMMAND microsimulator_flow_conformance_test)
  set_tests_properties(flow_conformance PROPERTIES LABELS conformance)
  add_test(NAME flow_resolved_conformance COMMAND microsimulator_flow_conformance_test resolved)
  set_tests_properties(flow_resolved_conformance PROPERTIES LABELS conformance)

  add_executable(microsimulator_lifecycle_conformance_test
    tests/conformance/lifecycle_conformance_test.cpp
  )
  target_link_libraries(microsimulator_lifecycle_conformance_test PRIVATE microsimulator::core)
  add_test(NAME lifecycle_conformance COMMAND microsimulator_lifecycle_conformance_test)
  set_tests_properties(lifecycle_conformance PROPERTIES LABELS conformance)

  add_executable(microsimulator_trajectory_conformance_test
    tests/conformance/trajectory_conformance_test.cpp
  )
  target_link_libraries(microsimulator_trajectory_conformance_test PRIVATE microsimulator::core)
  add_test(NAME trajectory_conformance COMMAND microsimulator_trajectory_conformance_test)
  set_tests_properties(trajectory_conformance PROPERTIES LABELS conformance)

  add_executable(microsimulator_contact_conformance_test
    tests/conformance/contact_conformance_test.cpp
  )
  target_link_libraries(microsimulator_contact_conformance_test PRIVATE microsimulator::core)
  add_test(NAME contact_conformance COMMAND microsimulator_contact_conformance_test)
  set_tests_properties(contact_conformance PROPERTIES LABELS conformance)

  add_executable(microsimulator_constraint_conformance_test
    tests/conformance/constraint_conformance_test.cpp
  )
  target_link_libraries(microsimulator_constraint_conformance_test PRIVATE microsimulator::core)
  add_test(NAME constraint_conformance COMMAND microsimulator_constraint_conformance_test)
  set_tests_properties(constraint_conformance PROPERTIES LABELS conformance)

  add_executable(microsimulator_mechanics_conformance_test
    tests/conformance/mechanics_conformance_test.cpp
  )
  target_link_libraries(microsimulator_mechanics_conformance_test PRIVATE microsimulator::core)
  add_test(NAME mechanics_conformance COMMAND microsimulator_mechanics_conformance_test)
  set_tests_properties(mechanics_conformance PROPERTIES LABELS conformance)

  add_executable(microsimulator_constraint_mechanics_conformance_test
    tests/conformance/constraint_mechanics_conformance_test.cpp
  )
  target_link_libraries(microsimulator_constraint_mechanics_conformance_test PRIVATE microsimulator::core)
  add_test(NAME constraint_mechanics_conformance COMMAND microsimulator_constraint_mechanics_conformance_test)
  set_tests_properties(constraint_mechanics_conformance PROPERTIES LABELS conformance)

  if(MICROSIMULATOR_ENABLE_CUDA)
    add_executable(microsimulator_cuda_runtime_gate_test
      tests/conformance/cuda_runtime_gate_test.cpp
    )
    target_link_libraries(microsimulator_cuda_runtime_gate_test PRIVATE microsimulator::core)
    add_test(NAME cuda_runtime_gate COMMAND microsimulator_cuda_runtime_gate_test)
    set_property(TEST cuda_runtime_gate PROPERTY LABELS conformance cuda)
  endif()

  if(MICROSIMULATOR_ENABLE_METAL)
    add_executable(microsimulator_metal_runtime_gate_test
      tests/conformance/metal_runtime_gate_test.cpp
    )
    target_link_libraries(microsimulator_metal_runtime_gate_test PRIVATE microsimulator::core)
    add_test(NAME metal_runtime_gate COMMAND microsimulator_metal_runtime_gate_test)
    set_property(TEST metal_runtime_gate PROPERTY LABELS conformance metal)
  endif()

  # Assertions are the test checks; keep them active in every build type.
  get_property(microsimulator_test_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
  foreach(microsimulator_test_target IN LISTS microsimulator_test_targets)
    if(microsimulator_test_target MATCHES "^microsimulator_.*_test$")
      if(MSVC)
        target_compile_options(${microsimulator_test_target} PRIVATE /UNDEBUG)
      else()
        target_compile_options(${microsimulator_test_target} PRIVATE -UNDEBUG)
      endif()
    endif()
  endforeach()
endif()
