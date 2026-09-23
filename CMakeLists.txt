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
  src/tropical_solver.cu
)
set_target_properties(tropical_sssp PROPERTIES OUTPUT_NAME sssp)
target_compile_features(tropical_sssp PRIVATE cxx_std_17 cuda_std_17)
target_include_directories(tropical_sssp PRIVATE ${timers_SOURCE_DIR}/include src)

option(TROPICAL_ENABLE_CUGRAPH "Build the cuGraph backend" ON)
if(TROPICAL_ENABLE_CUGRAPH)
  include(cmake/cuGraph.cmake)
  find_package(cugraph CONFIG REQUIRED)
endif()
if(TARGET cugraph::cugraph_c)
  set(TROPICAL_CUGRAPH_LINK_LIBS cugraph::cugraph_c CUDA::cudart CUDA::cusolver)
  find_library(TROPICAL_CUGRAPH_NVJITLINK NAMES libnvJitLink.so.13
    PATHS "${TROPICAL_CUGRAPH_SITE_PACKAGES}/nvidia/cu13/lib"
    NO_DEFAULT_PATH)
  if(TROPICAL_CUGRAPH_NVJITLINK)
    list(APPEND TROPICAL_CUGRAPH_LINK_LIBS "${TROPICAL_CUGRAPH_NVJITLINK}")
  endif()
  target_sources(tropical_sssp PRIVATE src/cugraph_solver.cpp)
  if(TROPICAL_CUGRAPH_LIB_DIRS)
    target_link_directories(tropical_sssp PRIVATE ${TROPICAL_CUGRAPH_LIB_DIRS})
    target_link_options(tropical_sssp PRIVATE
      "-Wl,-rpath-link,${TROPICAL_CUGRAPH_LIB_RPATH}"
      "-Wl,--disable-new-dtags")
  endif()
  target_link_libraries(tropical_sssp PRIVATE ${TROPICAL_CUGRAPH_LINK_LIBS})
else()
  target_sources(tropical_sssp PRIVATE src/cugraph_unavailable.cpp)
  message(STATUS "cuGraph backend disabled; set TROPICAL_ENABLE_CUGRAPH=ON to enable it")
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
  add_test(NAME cli_usage COMMAND ${CMAKE_COMMAND}
    -Dprogram=$<TARGET_FILE:tropical_sssp>
    -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_usage_test.cmake)
  if(TARGET cugraph::cugraph_c)
    add_executable(cugraph_test tests/cugraph_test.cpp src/graph.cpp src/cugraph_solver.cpp)
    target_compile_features(cugraph_test PRIVATE cxx_std_17)
    target_include_directories(cugraph_test PRIVATE ${timers_SOURCE_DIR}/include src)
    if(TROPICAL_CUGRAPH_LIB_DIRS)
      target_link_directories(cugraph_test PRIVATE ${TROPICAL_CUGRAPH_LIB_DIRS})
      target_link_options(cugraph_test PRIVATE
        "-Wl,-rpath-link,${TROPICAL_CUGRAPH_LIB_RPATH}"
        "-Wl,--disable-new-dtags")
    endif()
    target_link_libraries(cugraph_test PRIVATE ${TROPICAL_CUGRAPH_LINK_LIBS})
    add_test(NAME cugraph_test COMMAND cugraph_test)
  endif()
endif()
