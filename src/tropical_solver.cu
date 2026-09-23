#include "cuda_check.hpp"
#include "tropical_solver.cuh"
#include <timers/ScopedTimer.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <math.h>
#include <stddef.h>
#include <stdexcept>
#include <stdint.h>
#include <vector>

namespace {

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

// Own a cuSPARSE handle for the duration of one sparse-matrix operation.
class CusparseHandle {
public:
  CusparseHandle() { CHECK_CUSPARSE(cusparseCreate(&handle_)); }

  ~CusparseHandle() { cusparseDestroy(handle_); }

  CusparseHandle(const CusparseHandle&) = delete;
  CusparseHandle& operator=(const CusparseHandle&) = delete;

  cusparseHandle_t get() const { return handle_; }

private:
  cusparseHandle_t handle_ = nullptr;
};

// Placeholder for one min-plus relaxation step over outgoing CSR rows.
__global__ void tropical_spmv(const uint32_t* __restrict__ row_offsets,
                              const uint32_t* __restrict__ column_indices,
                              const float* __restrict__ weights, const float* __restrict__ input,
                              float* __restrict__ output, uint32_t vertex_count,
                              int* __restrict__ changed) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const int warp_idx = tid / 32;
  const int lane = tid % 32;
  const int grid_stride = (blockDim.x * gridDim.x) / 32;
  const int warp_stride = 32;
  bool warp_changed = false;
  __shared__ bool block_changed;

  if (threadIdx.x == 0) {
    block_changed = false;
  }
  __syncthreads();

  for (int i = warp_idx; i < vertex_count; i += grid_stride) {
    const uint32_t row_begin = row_offsets[i] + lane;
    const uint32_t row_end = row_offsets[i + 1];
    float distance = INFINITY;
    for (uint32_t col_offset = row_begin; col_offset < row_end; col_offset += warp_stride) {
      const uint32_t j = column_indices[col_offset];
      distance = fminf(distance, input[j] + weights[col_offset]);
    }
    // warp-level min-reduction
    for (int delta = warpSize / 2; delta >= 1; delta /= 2) {
      distance = fmin(distance, __shfl_down_sync(0xffffffffu, distance, delta));
    }
    if (lane == 0) {
      if (distance < input[i]) {
        output[i] = distance;
        warp_changed = true;
      } else {
        output[i] = input[i];
      }
    }
  }
  __syncthreads();
  if (__popc(__ballot_sync(0xffffffffu, warp_changed))) {
    if (lane == 0) {
      block_changed = true;
    }
  }
  __syncthreads();
  if (block_changed) {
    if (threadIdx.x == 0) {
      atomicExch(changed, 1);
    }
  }
}

} // namespace

CsrGraph transpose_csr_with_cusparse(const CsrGraph& graph) {
  if (graph.vertex_count == 0 ||
      graph.row_offsets.size() != static_cast<size_t>(graph.vertex_count) + 1 ||
      graph.column_indices.size() != graph.weights.size() ||
      graph.vertex_count > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
      graph.column_indices.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("invalid CSR graph for transposition");
  }

  const int vertex_count = static_cast<int>(graph.vertex_count);
  const int edge_count = static_cast<int>(graph.column_indices.size());
  const auto offset_bytes = graph.row_offsets.size() * sizeof(uint32_t);
  const auto edge_bytes = graph.column_indices.size() * sizeof(uint32_t);
  const auto weight_bytes = graph.weights.size() * sizeof(float);

  DeviceBuffer<uint32_t> device_offsets(graph.row_offsets.size());
  DeviceBuffer<uint32_t> device_columns(graph.column_indices.size());
  DeviceBuffer<float> device_weights(graph.weights.size());
  DeviceBuffer<uint32_t> device_transpose_offsets(graph.row_offsets.size());
  DeviceBuffer<uint32_t> device_transpose_columns(graph.column_indices.size());
  DeviceBuffer<float> device_transpose_weights(graph.weights.size());

  CHECK_CUDA(cudaMemcpy(device_offsets.data(), graph.row_offsets.data(), offset_bytes,
                        cudaMemcpyHostToDevice));
  if (edge_count != 0) {
    CHECK_CUDA(cudaMemcpy(device_columns.data(), graph.column_indices.data(), edge_bytes,
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_weights.data(), graph.weights.data(), weight_bytes,
                          cudaMemcpyHostToDevice));
  }

  CusparseHandle handle;
  const auto* csr_offsets = reinterpret_cast<const int*>(device_offsets.data());
  const auto* csr_columns = reinterpret_cast<const int*>(device_columns.data());
  auto* csc_offsets = reinterpret_cast<int*>(device_transpose_offsets.data());
  auto* csc_rows = reinterpret_cast<int*>(device_transpose_columns.data());
  size_t workspace_bytes = 0;
  CHECK_CUSPARSE(cusparseCsr2cscEx2_bufferSize(
      handle.get(), vertex_count, vertex_count, edge_count, device_weights.data(), csr_offsets,
      csr_columns, device_transpose_weights.data(), csc_offsets, csc_rows, CUDA_R_32F,
      CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &workspace_bytes));
  DeviceBuffer<char> workspace(workspace_bytes);
  CHECK_CUSPARSE(cusparseCsr2cscEx2(
      handle.get(), vertex_count, vertex_count, edge_count, device_weights.data(), csr_offsets,
      csr_columns, device_transpose_weights.data(), csc_offsets, csc_rows, CUDA_R_32F,
      CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, workspace.data()));

  CsrGraph transpose;
  transpose.vertex_count = graph.vertex_count;
  transpose.row_offsets.resize(graph.row_offsets.size());
  transpose.column_indices.resize(graph.column_indices.size());
  transpose.weights.resize(graph.weights.size());
  CHECK_CUDA(cudaMemcpy(transpose.row_offsets.data(), device_transpose_offsets.data(), offset_bytes,
                        cudaMemcpyDeviceToHost));
  if (edge_count != 0) {
    CHECK_CUDA(cudaMemcpy(transpose.column_indices.data(), device_transpose_columns.data(),
                          edge_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(transpose.weights.data(), device_transpose_weights.data(), weight_bytes,
                          cudaMemcpyDeviceToHost));
  }
  return transpose;
}

std::vector<float> run_tropical_exact_sssp(const CsrGraph& graph, uint32_t source,
                                           uint32_t repetitions,
                                           std::optional<uint32_t> max_iterations) {
  Stopwatch sw_memcpy("kernel.memcpy", /*stats=*/false);

  if (graph.vertex_count == 0 || source >= graph.vertex_count ||
      graph.row_offsets.size() != static_cast<size_t>(graph.vertex_count) + 1 ||
      graph.column_indices.size() != graph.weights.size() ||
      (max_iterations && *max_iterations == 0)) {
    throw std::invalid_argument("invalid tropical SSSP inputs");
  }

  const auto vertex_bytes = static_cast<size_t>(graph.vertex_count) * sizeof(float);
  const auto offset_bytes = graph.row_offsets.size() * sizeof(uint32_t);
  const auto edge_bytes = graph.column_indices.size() * sizeof(uint32_t);
  const auto weight_bytes = graph.weights.size() * sizeof(float);

  DeviceBuffer<uint32_t> device_offsets(graph.row_offsets.size());
  DeviceBuffer<uint32_t> device_columns(graph.column_indices.size());
  DeviceBuffer<float> device_weights(graph.weights.size());
  DeviceBuffer<float> device_a(graph.vertex_count);
  DeviceBuffer<float> device_b(graph.vertex_count);
  DeviceBuffer<int> device_changed(1);

  sw_memcpy.start();
  CHECK_CUDA(cudaMemcpy(device_offsets.data(), graph.row_offsets.data(), offset_bytes,
                        cudaMemcpyHostToDevice));
  if (!graph.column_indices.empty()) {
    CHECK_CUDA(cudaMemcpy(device_columns.data(), graph.column_indices.data(), edge_bytes,
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_weights.data(), graph.weights.data(), weight_bytes,
                          cudaMemcpyHostToDevice));
  }
  sw_memcpy.stop();

  std::vector<float> initial(graph.vertex_count, std::numeric_limits<float>::infinity());
  initial[source] = 0.0f;
  std::vector<float> result(graph.vertex_count);

  const uint32_t iterations = max_iterations.value_or(graph.vertex_count - 1);
  constexpr uint32_t threads_per_block = 256;
  const uint32_t blocks = (graph.vertex_count + threads_per_block - 1) / threads_per_block;

  float* current = device_a.data();
  float* next = device_b.data();
  current = device_a.data();
  next = device_b.data();

  Stopwatch sw_kernel("kernel", false);
  for (uint32_t repetition = 0; repetition < repetitions; ++repetition) {
    ScopedTimer st1(sw_kernel);
    ScopedTimer st2("kernel.rep");
    sw_memcpy.start();
    CHECK_CUDA(cudaMemcpy(current, initial.data(), vertex_bytes, cudaMemcpyHostToDevice));
    sw_memcpy.stop();
    int changed = 1;

    for (uint32_t iteration = 0; changed && (iteration < iterations); ++iteration) {
      ScopedTimer st("kernel.rep.it");
      CHECK_CUDA(cudaMemsetAsync(device_changed.data(), 0, sizeof(int)));
      tropical_spmv<<<blocks, threads_per_block>>>(device_offsets.data(), device_columns.data(),
                                                   device_weights.data(), current, next,
                                                   graph.vertex_count, device_changed.data());
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize());
      sw_memcpy.start();
      CHECK_CUDA(cudaMemcpy(&changed, device_changed.data(), sizeof(int), cudaMemcpyDeviceToHost));
      sw_memcpy.stop();
      std::swap(current, next);
    }
    sw_memcpy.start();
    CHECK_CUDA(cudaMemcpy(result.data(), current, vertex_bytes, cudaMemcpyDeviceToHost));
    sw_memcpy.stop();
  }

  return result;
}
