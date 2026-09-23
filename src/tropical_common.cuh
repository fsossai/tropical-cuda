#pragma once

#include "cuda_check.hpp"
#include "graph.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <stddef.h>

// Own a device allocation for one typed array.
template <typename T> class DeviceBuffer {
public:
  explicit DeviceBuffer(size_t count) {
    void* allocation = nullptr;
    CHECK_CUDA(cudaMalloc(&allocation, std::max<size_t>(count, 1) * sizeof(T)));
    data_ = static_cast<T*>(allocation);
  }

  ~DeviceBuffer() { cudaFree(data_); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  T* data() const { return data_; }

private:
  T* data_ = nullptr;
};

// Convert outgoing CSR to incoming CSR with cuSPARSE CSR-to-CSC conversion.
CsrGraph transpose_csr_with_cusparse(const CsrGraph& graph);
