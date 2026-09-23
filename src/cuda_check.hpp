#pragma once

#include <cuda_runtime_api.h>

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
