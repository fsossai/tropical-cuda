#pragma once

#include <cuda_runtime_api.h>
#include <cusparse.h>

#include <stdexcept>
#include <string>

#define CHECK_CUDA(call)                                                                           \
  do {                                                                                             \
    const cudaError_t cuda_status = (call);                                                        \
    if (cuda_status != cudaSuccess) {                                                              \
      throw std::runtime_error(std::string("CUDA error in ") + __func__ + " at line " +            \
                               std::to_string(__LINE__) +                                          \
                               " (" #call "): " + cudaGetErrorString(cuda_status));                \
    }                                                                                              \
  } while (false)

#define CHECK_CUSPARSE(call)                                                                       \
  do {                                                                                             \
    const cusparseStatus_t cusparse_status = (call);                                               \
    if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {                                              \
      throw std::runtime_error(std::string("cuSPARSE error in ") + __func__ + " at line " +        \
                               std::to_string(__LINE__) +                                          \
                               " (" #call "): " + std::to_string(static_cast<int>(cusparse_status))); \
    }                                                                                              \
  } while (false)
