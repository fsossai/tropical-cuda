#include "cuda_check.hpp"
#include "tropical_solver.cuh"
#include <timers/ScopedTimer.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
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

// Placeholder for one min-plus relaxation step over outgoing CSR rows.
__global__ void tropical_spmv(const uint32_t* row_offsets, const uint32_t* column_indices,
                              const float* weights, const float* current, float* next,
                              uint32_t vertex_count) {
  // TODO: Relax outgoing edges into next using tropical addition and multiplication.
}

} // namespace

std::vector<float> run_tropical_exact_sssp(const CsrGraph& graph, uint32_t source,
                                           uint32_t repetitions,
                                           std::optional<uint32_t> max_iterations) {
  Stopwatch sw_memcpy("total.kernel.memcpy", /*stats=*/false);

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

  std::vector<float> result(graph.vertex_count, std::numeric_limits<float>::infinity());
  result[source] = 0.0f;

  const uint32_t iterations = max_iterations.value_or(graph.vertex_count - 1);
  constexpr uint32_t threads_per_block = 256;
  const uint32_t blocks = (graph.vertex_count + threads_per_block - 1) / threads_per_block;

  float* current = device_a.data();
  float* next = device_b.data();
  current = device_a.data();
  next = device_b.data();
  sw_memcpy.start();
  CHECK_CUDA(cudaMemcpy(current, result.data(), vertex_bytes, cudaMemcpyHostToDevice));
  sw_memcpy.stop();

  for (uint32_t repetition = 0; repetition < repetitions; ++repetition) {
    for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
      tropical_spmv<<<blocks, threads_per_block>>>(device_offsets.data(), device_columns.data(),
                                                   device_weights.data(), current, next,
                                                   graph.vertex_count);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize());
      std::swap(current, next);
    }
    sw_memcpy.start();
    CHECK_CUDA(cudaMemcpy(result.data(), current, vertex_bytes, cudaMemcpyDeviceToHost));
    sw_memcpy.stop();

    throw std::logic_error("tropical SpMV solver is not implemented yet");
  }

  return result;
}