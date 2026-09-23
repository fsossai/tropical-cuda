# Locate RAPIDS CMake packages installed by pip. By default, the active
# Python environment supplies the site-packages directory. Set
# TROPICAL_CUGRAPH_SITE_PACKAGES to override that discovery.
set(TROPICAL_CUGRAPH_SITE_PACKAGES "" CACHE PATH
    "Python site-packages directory containing RAPIDS packages")

set(_cugraph_site_packages "${TROPICAL_CUGRAPH_SITE_PACKAGES}")
set(_cugraph_python_result 0)
if(NOT _cugraph_site_packages)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c
              "import sysconfig; print(sysconfig.get_path('purelib'))"
      OUTPUT_VARIABLE _cugraph_site_packages
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _cugraph_python_result)
  endif()
endif()

if(NOT _cugraph_python_result EQUAL 0 OR
   NOT EXISTS "${_cugraph_site_packages}/libcugraph")
  return()
endif()

# Set a package config directory only when the caller has not supplied one.
function(cugraph_locate_package package directory)
  set(_cugraph_config
      "${_cugraph_site_packages}/${directory}/lib64/cmake/${package}")
  if(NOT ${package}_DIR AND EXISTS "${_cugraph_config}")
    set(${package}_DIR "${_cugraph_config}" CACHE PATH
        "CMake package directory for ${package}")
  endif()
endfunction()

cugraph_locate_package(cugraph libcugraph)
cugraph_locate_package(cuco libcugraph)
cugraph_locate_package(raft libraft)
cugraph_locate_package(rmm librmm)
cugraph_locate_package(cuvs libcuvs)
cugraph_locate_package(cudf libcudf)
cugraph_locate_package(kvikio libkvikio)
cugraph_locate_package(ucxx libucxx)
cugraph_locate_package(rapids_logger rapids_logger)
cugraph_locate_package(nvtx3 librmm)
cugraph_locate_package(bs_thread_pool libkvikio)

# Keep RAPIDS shared libraries visible to the linker and executable without
# exposing the pip environment's CUDA libraries to CMake's CUDA toolkit.
set(TROPICAL_CUGRAPH_LIB_DIRS)
foreach(directory IN ITEMS libcugraph libraft librmm libcuvs libcudf libkvikio libucxx rapids_logger)
  if(EXISTS "${_cugraph_site_packages}/${directory}/lib64")
    list(APPEND TROPICAL_CUGRAPH_LIB_DIRS
         "${_cugraph_site_packages}/${directory}/lib64")
  endif()
endforeach()
string(JOIN ":" TROPICAL_CUGRAPH_LIB_RPATH ${TROPICAL_CUGRAPH_LIB_DIRS})
