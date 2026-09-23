cmake_minimum_required(VERSION 3.24)
project(tropical_cuda LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
)
target_compile_features(tropical_sssp PRIVATE cxx_std_17)
target_include_directories(tropical_sssp PRIVATE ${timers_SOURCE_DIR}/include src)

option(TROPICAL_ENABLE_CUGRAPH "Enable the cuGraph backend when available" ON)
if(TROPICAL_ENABLE_CUGRAPH)
  find_package(cugraph CONFIG QUIET)
endif()
if(TARGET cugraph::cugraph_c)
  find_package(CUDAToolkit REQUIRED)
  target_sources(tropical_sssp PRIVATE src/cugraph_solver.cpp)
  target_link_libraries(tropical_sssp PRIVATE cugraph::cugraph_c CUDA::cudart)
else()
  target_sources(tropical_sssp PRIVATE src/cugraph_unavailable.cpp)
  if(TROPICAL_ENABLE_CUGRAPH)
    message(STATUS "cuGraph C library not found; cugraph backend unavailable")
  else()
    message(STATUS "cuGraph backend disabled")
  endif()
endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(graph_test tests/graph_test.cpp src/graph.cpp)
  target_compile_features(graph_test PRIVATE cxx_std_17)
  target_include_directories(graph_test PRIVATE src)
  add_test(NAME graph_test COMMAND graph_test)
  add_test(NAME default_source COMMAND tropical_sssp ${CMAKE_CURRENT_SOURCE_DIR}/tests/first_source.txt)
  set_tests_properties(default_source PROPERTIES PASS_REGULAR_EXPRESSION "source *: *7([^0-9]|$)")
  add_test(NAME explicit_source COMMAND tropical_sssp ${CMAKE_CURRENT_SOURCE_DIR}/tests/first_source.txt --source 0)
  set_tests_properties(explicit_source PROPERTIES PASS_REGULAR_EXPRESSION "source *: *0([^0-9]|$)")
  if(TARGET cugraph::cugraph_c)
    add_executable(cugraph_test tests/cugraph_test.cpp src/graph.cpp src/cugraph_solver.cpp)
    target_compile_features(cugraph_test PRIVATE cxx_std_17)
    target_include_directories(cugraph_test PRIVATE ${timers_SOURCE_DIR}/include src)
    target_link_libraries(cugraph_test PRIVATE cugraph::cugraph_c CUDA::cudart)
    add_test(NAME cugraph_test COMMAND cugraph_test)
  endif()
endif()
