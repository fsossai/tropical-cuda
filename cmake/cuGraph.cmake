# Locates the RAPIDS cuGraph pip-wheel packages so that find_package(cugraph)
# can resolve them. The pip wheels place their CMake configs under
# <pkg>/lib64/cmake/<pkg>, which find_package does not discover via
# CMAKE_PREFIX_PATH, so this sets the individual <pkg>_DIR variables instead.
# It only sets a variable when the config file actually exists, so a machine
# without cuGraph keeps the normal find_package(... QUIET) fallback behavior.

# Override with -DTROPICAL_CUGRAPH_SITE_PACKAGES=<dir> for another install.
set(TROPICAL_CUGRAPH_SITE_PACKAGES
    "/tank/federico/.local/venvs/rapids/lib/python3.12/site-packages"
    CACHE PATH "Directory holding the RAPIDS cuGraph pip-wheel packages")

if(NOT TROPICAL_CUGRAPH_SITE_PACKAGES)
  return()
endif()

# Set <pkg>_DIR to the pip-wheel config dir when the package is installed.
# <dir> is the on-disk package folder, which may differ from <pkg> (e.g. the
# "raft" package lives in the "libraft" folder).
function(cugraph_locate_package pkg dir)
  if(NOT ${pkg}_DIR
     AND EXISTS "${TROPICAL_CUGRAPH_SITE_PACKAGES}/${dir}/lib64/cmake/${pkg}/")
    set(${pkg}_DIR "${TROPICAL_CUGRAPH_SITE_PACKAGES}/${dir}/lib64/cmake/${pkg}"
        CACHE PATH "" FORCE)
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

# nvtx3 and bs_thread_pool ship inside librmm and libkvikio respectively.
if(NOT nvtx3_DIR
   AND EXISTS "${TROPICAL_CUGRAPH_SITE_PACKAGES}/librmm/lib64/cmake/nvtx3/")
  set(nvtx3_DIR "${TROPICAL_CUGRAPH_SITE_PACKAGES}/librmm/lib64/cmake/nvtx3"
      CACHE PATH "" FORCE)
endif()
if(NOT bs_thread_pool_DIR
   AND EXISTS "${TROPICAL_CUGRAPH_SITE_PACKAGES}/libkvikio/lib64/cmake/bs_thread_pool/")
  set(bs_thread_pool_DIR
      "${TROPICAL_CUGRAPH_SITE_PACKAGES}/libkvikio/lib64/cmake/bs_thread_pool"
      CACHE PATH "" FORCE)
endif()

unset(cugraph_locate_package)

# Library directories of the located packages, needed both to find the
# transitive shared libraries at link time and to run the resulting binaries.
set(TROPICAL_CUGRAPH_LIB_DIRS)
foreach(dir IN ITEMS libcugraph libraft librmm libcuvs libcudf libkvikio libucxx rapids_logger)
  if(EXISTS "${TROPICAL_CUGRAPH_SITE_PACKAGES}/${dir}/lib64")
    list(APPEND TROPICAL_CUGRAPH_LIB_DIRS
         "${TROPICAL_CUGRAPH_SITE_PACKAGES}/${dir}/lib64")
  endif()
endforeach()
# NVIDIA CUDA runtime and NCCL libraries shipped with the pip wheels.
foreach(dir IN ITEMS "nvidia/cu13/lib" "nvidia/nccl/lib")
  if(EXISTS "${TROPICAL_CUGRAPH_SITE_PACKAGES}/${dir}")
    list(APPEND TROPICAL_CUGRAPH_LIB_DIRS
         "${TROPICAL_CUGRAPH_SITE_PACKAGES}/${dir}")
  endif()
endforeach()
string(JOIN ":" TROPICAL_CUGRAPH_LIB_RPATH ${TROPICAL_CUGRAPH_LIB_DIRS})
set(CMAKE_BUILD_RPATH "${TROPICAL_CUGRAPH_LIB_RPATH}")
