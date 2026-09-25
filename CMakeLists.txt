cmake_minimum_required(VERSION 3.24)
project(tropical_cuda LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(CUDAToolkit REQUIRED)

include(FetchContent)
FetchContent_Declare(
  timers
  GIT_REPOSITORY https://github.com/fsossai/timers.git
  GIT_TAG fade26e404d74f09a4247c25e73afaa6129e6f3e
)
FetchContent_MakeAvailable(timers)

add_executable(tropical_sssp
  src/main.cpp
  src/graph.cpp
  src/graph_binary.cpp
  src/tropical_common.cu
  src/tropical_exact.cu
  src/tropical_apx.cu
  src/cusparse_solver.cu
)
set_target_properties(tropical_sssp PROPERTIES OUTPUT_NAME sssp)
target_compile_features(tropical_sssp PRIVATE cxx_std_17 cuda_std_17)
target_include_directories(tropical_sssp PRIVATE ${timers_SOURCE_DIR}/include src)
target_link_libraries(tropical_sssp PRIVATE CUDA::cudart CUDA::cusparse CUDA::nvrtc)

add_executable(graph_convert
  src/graph_convert.cpp
  src/graph.cpp
  src/graph_binary.cpp
)
target_compile_features(graph_convert PRIVATE cxx_std_17)
target_include_directories(graph_convert PRIVATE src)

option(TROPICAL_ENABLE_CUGRAPH "Build the cuGraph backend" ON)
if(TROPICAL_ENABLE_CUGRAPH)
  include(cmake/cuGraph.cmake)
  find_package(cugraph CONFIG REQUIRED)
endif()

if(TARGET cugraph::cugraph_c)
  target_sources(tropical_sssp PRIVATE src/cugraph_solver.cpp)
  if(TROPICAL_CUGRAPH_LIB_DIRS)
    target_link_directories(tropical_sssp PRIVATE ${TROPICAL_CUGRAPH_LIB_DIRS})
    target_link_options(tropical_sssp PRIVATE
      "-Wl,-rpath-link,${TROPICAL_CUGRAPH_LIB_RPATH}"
      "-Wl,--disable-new-dtags")
  endif()
  target_link_libraries(tropical_sssp PRIVATE cugraph::cugraph_c CUDA::cusolver)
else()
  target_sources(tropical_sssp PRIVATE src/cugraph_unavailable.cpp)
  message(STATUS "cuGraph backend disabled; set TROPICAL_ENABLE_CUGRAPH=ON to enable it")
endif()

include(CTest)
if(BUILD_TESTING)
  add_test(NAME default_source COMMAND tropical_sssp ${CMAKE_CURRENT_SOURCE_DIR}/tests/first_source.txt)
  set_tests_properties(default_source PROPERTIES PASS_REGULAR_EXPRESSION "source *: *7([^0-9]|$)")
  add_test(NAME explicit_source COMMAND tropical_sssp ${CMAKE_CURRENT_SOURCE_DIR}/tests/first_source.txt --source 0)
  set_tests_properties(explicit_source PROPERTIES PASS_REGULAR_EXPRESSION "source *: *0([^0-9]|$)")
  add_test(NAME cli_usage COMMAND ${CMAKE_COMMAND}
    -Dprogram=$<TARGET_FILE:tropical_sssp>
    -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_usage_test.cmake)
endif()
