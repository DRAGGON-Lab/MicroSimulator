cmake_minimum_required(VERSION 3.25)

project(
  MicroSimulator
  VERSION 0.1.0
  DESCRIPTION "Microfluidic simulation of microbial populations, flow, and solute transport"
  LANGUAGES CXX
)

option(CM_BUILD_PYTHON "Build the nanobind Python extension" ON)
option(CM_BUILD_TESTS "Build C++ tests" ON)
option(CM_ENABLE_CUDA "Build the native NVIDIA CUDA backend" OFF)

set(CM_METAL_DEFAULT OFF)
if(APPLE)
  set(CM_METAL_DEFAULT ON)
endif()
option(CM_ENABLE_METAL "Build the native Apple Metal backend" ${CM_METAL_DEFAULT})

add_library(cm_core STATIC
  cpp/core/checkpoint.cpp
  cpp/core/contact_graph.cpp
  cpp/core/constraints.cpp
  cpp/core/coupled_rates.cpp
  cpp/core/mechanics_integration.cpp
  cpp/core/occupancy.cpp
  cpp/core/signals.cpp
  cpp/core/species.cpp
  cpp/core/world_state.cpp
  cpp/core/simulation.cpp
  cpp/cpu/cpu_backend.cpp
  cpp/cpu/cpu_contacts.cpp
  cpp/cpu/cpu_constraints.cpp
  cpp/cpu/cpu_coupled.cpp
  cpp/cpu/cpu_flow.cpp
  cpp/cpu/cpu_mechanics.cpp
  cpp/cpu/cpu_species.cpp
)
add_library(cm::core ALIAS cm_core)

target_compile_features(cm_core PUBLIC cxx_std_23)
target_include_directories(cm_core PUBLIC cpp/include PRIVATE cpp)
set_target_properties(cm_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(CM_ENABLE_METAL)
  if(NOT APPLE)
    message(FATAL_ERROR "CM_ENABLE_METAL requires an Apple platform")
  endif()

  enable_language(OBJCXX)
  find_library(CM_FOUNDATION_FRAMEWORK Foundation REQUIRED)
  find_library(CM_METAL_FRAMEWORK Metal REQUIRED)

  set(CM_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

  # Every shader source opens with the prelude, the shared definition lists it
  # gives meaning to, and the shared structure layouts.
  set(CM_METAL_SHARED_SOURCES
    cpp/metal/kernels/prelude.metal
    cpp/shared/constants.def
    cpp/include/cm/rate_ops.def
    cpp/shared/device_types.h
  )

  # Embeds the shared sources followed by the given kernel sources, in order,
  # as the string cm::metal::<name>_source in cm/metal/<name>_source.hpp.
  function(cm_embed_metal_source name)
    set(sources ${CM_METAL_SHARED_SOURCES} ${ARGN})
    list(TRANSFORM sources PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/" OUTPUT_VARIABLE inputs)
    set(output "${CM_GENERATED_DIR}/cm/metal/${name}_source.hpp")
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
    target_sources(cm_core PRIVATE "${output}")
  endfunction()

  cm_embed_metal_source(growth cpp/metal/kernels/growth.metal)
  cm_embed_metal_source(species cpp/metal/kernels/species.metal)
  cm_embed_metal_source(contacts cpp/metal/kernels/contacts.metal)
  cm_embed_metal_source(mechanics cpp/metal/kernels/mechanics.metal)
  cm_embed_metal_source(signals
    cpp/metal/kernels/grid_transport.metal cpp/metal/kernels/signals.metal)
  cm_embed_metal_source(coupled_rates
    cpp/metal/kernels/grid_transport.metal cpp/metal/kernels/coupled_rates.metal)
  cm_embed_metal_source(flow cpp/metal/kernels/flow_grid.metal cpp/metal/kernels/flow.metal)
  cm_embed_metal_source(depth_flow
    cpp/metal/kernels/flow_grid.metal cpp/metal/kernels/depth_flow.metal)

  set(CM_METAL_OCCUPANCY_HEADER "${CM_GENERATED_DIR}/cm/metal/occupancy_source.hpp")
  add_custom_command(
    OUTPUT "${CM_METAL_OCCUPANCY_HEADER}"
    COMMAND "${CMAKE_COMMAND}"
      "-DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/cpp/metal/kernels/occupancy.metal"
      "-DOUTPUT=${CM_METAL_OCCUPANCY_HEADER}"
      "-DSYMBOL=occupancy_source"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedMetalSource.cmake"
    DEPENDS cpp/metal/kernels/occupancy.metal cmake/EmbedMetalSource.cmake
    VERBATIM
  )

  target_sources(cm_core PRIVATE
    cpp/metal/metal_backend.mm
    cpp/metal/metal_flow.mm
    cpp/metal/metal_occupancy.mm
    "${CM_METAL_OCCUPANCY_HEADER}"
  )
  target_include_directories(cm_core PRIVATE "${CM_GENERATED_DIR}")
  target_compile_definitions(cm_core PRIVATE CM_HAS_METAL=1)
  target_compile_options(cm_core PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
  target_link_libraries(cm_core PRIVATE
    "${CM_FOUNDATION_FRAMEWORK}"
    "${CM_METAL_FRAMEWORK}"
  )
  set_target_properties(cm_core PROPERTIES OBJCXX_STANDARD 23 OBJCXX_STANDARD_REQUIRED ON)
endif()

if(CM_ENABLE_CUDA)
  if(APPLE)
    message(FATAL_ERROR "CM_ENABLE_CUDA requires a platform supported by the CUDA Toolkit")
  endif()

  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  target_sources(cm_core PRIVATE
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
  target_compile_definitions(cm_core PRIVATE CM_HAS_CUDA=1)
  set_source_files_properties(cpp/cuda/kernels/occupancy.cu PROPERTIES COMPILE_OPTIONS "--fmad=false")
  target_link_libraries(cm_core PRIVATE CUDA::cudart)
  set_target_properties(cm_core PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
  )
endif()

if(MSVC)
  target_compile_options(cm_core PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:/W4;/permissive->"
  )
else()
  target_compile_options(cm_core PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wconversion>"
    "$<$<COMPILE_LANGUAGE:OBJCXX>:-Wall;-Wextra;-Wpedantic;-Wconversion>"
  )
endif()

if(CM_BUILD_PYTHON)
  find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(nanobind CONFIG REQUIRED)

  nanobind_add_module(_core cpp/python/bindings.cpp cpp/python/occupancy_bindings.cpp)
  target_link_libraries(_core PRIVATE cm::core)
  install(TARGETS _core LIBRARY DESTINATION microsimulator)
endif()

if(CM_BUILD_TESTS)
  include(CTest)
  add_executable(cm_occupancy_conformance_test tests/conformance/occupancy_conformance_test.cpp)
  target_link_libraries(cm_occupancy_conformance_test PRIVATE cm::core)
  add_test(NAME occupancy_conformance COMMAND cm_occupancy_conformance_test)
  set_tests_properties(occupancy_conformance PROPERTIES LABELS conformance SKIP_RETURN_CODE 77)
  add_executable(cm_world_state_test tests/cpp/world_state_test.cpp)
  target_link_libraries(cm_world_state_test PRIVATE cm::core)
  add_test(NAME world_state COMMAND cm_world_state_test)

  add_executable(cm_contact_geometry_test tests/cpp/contact_geometry_test.cpp)
  target_link_libraries(cm_contact_geometry_test PRIVATE cm::core)
  add_test(NAME contact_geometry COMMAND cm_contact_geometry_test)

  add_executable(cm_constraint_geometry_test tests/cpp/constraint_geometry_test.cpp)
  target_link_libraries(cm_constraint_geometry_test PRIVATE cm::core)
  add_test(NAME constraint_geometry COMMAND cm_constraint_geometry_test)

  add_executable(cm_mechanics_operator_test tests/cpp/mechanics_operator_test.cpp)
  target_link_libraries(cm_mechanics_operator_test PRIVATE cm::core)
  add_test(NAME mechanics_operator COMMAND cm_mechanics_operator_test)

  add_executable(cm_mechanics_integration_test tests/cpp/mechanics_integration_test.cpp)
  target_link_libraries(cm_mechanics_integration_test PRIVATE cm::core)
  add_test(NAME mechanics_integration COMMAND cm_mechanics_integration_test)

  add_executable(cm_species_test tests/cpp/species_test.cpp)
  target_link_libraries(cm_species_test PRIVATE cm::core)
  add_test(NAME species COMMAND cm_species_test)

  add_executable(cm_signal_grid_test tests/cpp/signal_grid_test.cpp)
  target_link_libraries(cm_signal_grid_test PRIVATE cm::core)
  add_test(NAME signal_grid COMMAND cm_signal_grid_test)

  add_executable(cm_coupled_rates_test tests/cpp/coupled_rates_test.cpp)
  target_link_libraries(cm_coupled_rates_test PRIVATE cm::core)
  add_test(NAME coupled_rates COMMAND cm_coupled_rates_test)

  add_executable(cm_flow_test tests/cpp/flow_test.cpp)
  target_link_libraries(cm_flow_test PRIVATE cm::core)
  add_test(NAME flow COMMAND cm_flow_test)

  add_executable(cm_checkpoint_test tests/cpp/checkpoint_test.cpp)
  target_link_libraries(cm_checkpoint_test PRIVATE cm::core)
  target_include_directories(cm_checkpoint_test PRIVATE tests/conformance)
  add_test(NAME checkpoint COMMAND cm_checkpoint_test)
  set_tests_properties(checkpoint PROPERTIES LABELS conformance)

  add_executable(cm_growth_conformance_test tests/conformance/growth_conformance_test.cpp)
  target_link_libraries(cm_growth_conformance_test PRIVATE cm::core)
  add_test(NAME growth_conformance COMMAND cm_growth_conformance_test)
  set_tests_properties(growth_conformance PROPERTIES LABELS conformance)

  add_executable(cm_backend_contract_conformance_test
    tests/conformance/backend_contract_conformance_test.cpp
  )
  target_link_libraries(cm_backend_contract_conformance_test PRIVATE cm::core)
  add_test(NAME backend_contract_conformance COMMAND cm_backend_contract_conformance_test)
  set_tests_properties(backend_contract_conformance PROPERTIES LABELS conformance)

  add_executable(cm_species_conformance_test
    tests/conformance/species_conformance_test.cpp
  )
  target_link_libraries(cm_species_conformance_test PRIVATE cm::core)
  add_test(NAME species_conformance COMMAND cm_species_conformance_test)
  set_tests_properties(species_conformance PROPERTIES LABELS conformance)

  add_executable(cm_signal_grid_conformance_test
    tests/conformance/signal_grid_conformance_test.cpp
  )
  target_link_libraries(cm_signal_grid_conformance_test PRIVATE cm::core)
  add_test(NAME signal_grid_conformance COMMAND cm_signal_grid_conformance_test)
  set_tests_properties(signal_grid_conformance PROPERTIES LABELS conformance)

  add_executable(cm_coupled_rates_conformance_test
    tests/conformance/coupled_rates_conformance_test.cpp
  )
  target_link_libraries(cm_coupled_rates_conformance_test PRIVATE cm::core)
  add_test(NAME coupled_rates_conformance COMMAND cm_coupled_rates_conformance_test)
  set_tests_properties(coupled_rates_conformance PROPERTIES LABELS conformance)

  add_executable(cm_flow_conformance_test
    tests/conformance/flow_conformance_test.cpp
  )
  target_link_libraries(cm_flow_conformance_test PRIVATE cm::core)
  target_include_directories(cm_flow_conformance_test PRIVATE tests/conformance)
  add_test(NAME flow_conformance COMMAND cm_flow_conformance_test)
  set_tests_properties(flow_conformance PROPERTIES LABELS conformance)
  add_test(NAME flow_resolved_conformance COMMAND cm_flow_conformance_test resolved)
  set_tests_properties(flow_resolved_conformance PROPERTIES LABELS conformance)

  add_executable(cm_lifecycle_conformance_test
    tests/conformance/lifecycle_conformance_test.cpp
  )
  target_link_libraries(cm_lifecycle_conformance_test PRIVATE cm::core)
  add_test(NAME lifecycle_conformance COMMAND cm_lifecycle_conformance_test)
  set_tests_properties(lifecycle_conformance PROPERTIES LABELS conformance)

  add_executable(cm_trajectory_conformance_test
    tests/conformance/trajectory_conformance_test.cpp
  )
  target_link_libraries(cm_trajectory_conformance_test PRIVATE cm::core)
  add_test(NAME trajectory_conformance COMMAND cm_trajectory_conformance_test)
  set_tests_properties(trajectory_conformance PROPERTIES LABELS conformance)

  add_executable(cm_contact_conformance_test
    tests/conformance/contact_conformance_test.cpp
  )
  target_link_libraries(cm_contact_conformance_test PRIVATE cm::core)
  add_test(NAME contact_conformance COMMAND cm_contact_conformance_test)
  set_tests_properties(contact_conformance PROPERTIES LABELS conformance)

  add_executable(cm_constraint_conformance_test
    tests/conformance/constraint_conformance_test.cpp
  )
  target_link_libraries(cm_constraint_conformance_test PRIVATE cm::core)
  add_test(NAME constraint_conformance COMMAND cm_constraint_conformance_test)
  set_tests_properties(constraint_conformance PROPERTIES LABELS conformance)

  add_executable(cm_mechanics_conformance_test
    tests/conformance/mechanics_conformance_test.cpp
  )
  target_link_libraries(cm_mechanics_conformance_test PRIVATE cm::core)
  add_test(NAME mechanics_conformance COMMAND cm_mechanics_conformance_test)
  set_tests_properties(mechanics_conformance PROPERTIES LABELS conformance)

  add_executable(cm_constraint_mechanics_conformance_test
    tests/conformance/constraint_mechanics_conformance_test.cpp
  )
  target_link_libraries(cm_constraint_mechanics_conformance_test PRIVATE cm::core)
  add_test(NAME constraint_mechanics_conformance COMMAND cm_constraint_mechanics_conformance_test)
  set_tests_properties(constraint_mechanics_conformance PROPERTIES LABELS conformance)

  if(CM_ENABLE_CUDA)
    add_executable(cm_cuda_runtime_gate_test
      tests/conformance/cuda_runtime_gate_test.cpp
    )
    target_link_libraries(cm_cuda_runtime_gate_test PRIVATE cm::core)
    add_test(NAME cuda_runtime_gate COMMAND cm_cuda_runtime_gate_test)
    set_property(TEST cuda_runtime_gate PROPERTY LABELS conformance cuda)
  endif()

  if(CM_ENABLE_METAL)
    add_executable(cm_metal_runtime_gate_test
      tests/conformance/metal_runtime_gate_test.cpp
    )
    target_link_libraries(cm_metal_runtime_gate_test PRIVATE cm::core)
    add_test(NAME metal_runtime_gate COMMAND cm_metal_runtime_gate_test)
    set_property(TEST metal_runtime_gate PROPERTY LABELS conformance metal)
  endif()

  # Assertions are the test checks; keep them active in every build type.
  get_property(cm_test_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
  foreach(cm_test_target IN LISTS cm_test_targets)
    if(cm_test_target MATCHES "^cm_.*_test$")
      if(MSVC)
        target_compile_options(${cm_test_target} PRIVATE /UNDEBUG)
      else()
        target_compile_options(${cm_test_target} PRIVATE -UNDEBUG)
      endif()
    endif()
  endforeach()
endif()
