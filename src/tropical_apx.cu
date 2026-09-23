#include "tropical_apx.cuh"
#include "tropical_common.cuh"

#include <timers/ScopedTimer.hpp>

#include <cuda_runtime.h>

#include <limits>
#include <math.h>
#include <memory>
#include <stdexcept>

namespace {

// Own cuSPARSE resources for repeated CSR matrix-vector products.
class CusparseSpmv {
public:
  CusparseSpmv(const CsrGraph& graph, uint32_t* offsets, uint32_t* columns, float* weights,
               float* input, float* output) {
    CHECK_CUSPARSE(cusparseCreate(&handle_));
    CHECK_CUSPARSE(cusparseCreateCsr(&matrix_, graph.vertex_count, graph.vertex_count,
                                     graph.edge_count, reinterpret_cast<int*>(offsets),
                                     reinterpret_cast<int*>(columns), weights, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&input_, graph.vertex_count, input, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&output_, graph.vertex_count, output, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_,
                                           matrix_, input_, &beta_, output_, CUDA_R_32F,
                                           CUSPARSE_SPMV_ALG_DEFAULT, &workspace_bytes_));
    workspace_ = std::make_unique<DeviceBuffer<char>>(workspace_bytes_);
  }

  ~CusparseSpmv() {
    cusparseDestroyDnVec(output_);
    cusparseDestroyDnVec(input_);
    cusparseDestroySpMat(matrix_);
    cusparseDestroy(handle_);
  }

  CusparseSpmv(const CusparseSpmv&) = delete;
  CusparseSpmv& operator=(const CusparseSpmv&) = delete;

  // Multiply the configured CSR matrix by input and store the result in output.
  void multiply(float* input, float* output) {
    CHECK_CUSPARSE(cusparseDnVecSetValues(input_, input));
    CHECK_CUSPARSE(cusparseDnVecSetValues(output_, output));
    CHECK_CUSPARSE(cusparseSpMV(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_, matrix_, input_,
                                &beta_, output_, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                                workspace_->data()));
  }

private:
  cusparseHandle_t handle_ = nullptr;
  cusparseSpMatDescr_t matrix_ = nullptr;
  cusparseDnVecDescr_t input_ = nullptr;
  cusparseDnVecDescr_t output_ = nullptr;
  float alpha_ = 1.0f;
  float beta_ = 1.0f;
  size_t workspace_bytes_ = 0;
  std::unique_ptr<DeviceBuffer<char>> workspace_;
};

// Transform a tropical value into the exponential domain.
__device__ float encode_value(float x, float beta) { return expf(-beta * x); }

// Transform an exponential-domain value back into the tropical domain.
__device__ float decode_value(float x, float beta) { return -logf(x) / beta; }

// Transform values between tropical and exponential domains in place.
template <bool encode>
__global__ void transform_domain(float beta, float* __restrict__ values, uint32_t num_values) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const int grid_stride = blockDim.x * gridDim.x;
  // TODO: vectorize
  for (int i = tid; i < num_values; i += grid_stride) {
    if constexpr (encode) {
      values[i] = encode_value(values[i], beta);
    } else {
      values[i] = decode_value(values[i], beta);
    }
  }
}

} // namespace

// Run the approximate backend until convergence or its requested iteration cap.
std::vector<float> run_tropical_apx_sssp(const CsrGraph& graph, uint32_t source,
                                         uint32_t repetitions,
                                         std::optional<uint32_t> max_iterations) {
  Stopwatch sw_memcpy("kernel.memcpy", /*stats=*/false);
  Stopwatch sw_encode("kernel.encode", /*stats=*/false);
  Stopwatch sw_decode("kernel.decode", /*stats=*/false);

  if (graph.vertex_count == 0 || source >= graph.vertex_count || repetitions == 0 ||
      graph.row_offsets.size() != static_cast<size_t>(graph.vertex_count) + 1 ||
      graph.edge_count != graph.column_indices.size() ||
      graph.column_indices.size() != graph.weights.size() ||
      (max_iterations && *max_iterations == 0)) {
    throw std::invalid_argument("invalid tropical approximate SSSP inputs");
  }

  const auto vertex_bytes = static_cast<size_t>(graph.vertex_count) * sizeof(float);
  const auto offset_bytes = graph.row_offsets.size() * sizeof(uint32_t);
  const auto edge_bytes = graph.edge_count * sizeof(uint32_t);
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

  std::vector<float> initial(graph.vertex_count, std::numeric_limits<float>::infinity());
  initial[source] = 0.0f;
  std::vector<float> result(graph.vertex_count);
  const uint32_t iterations = max_iterations.value_or(graph.vertex_count - 1);
  constexpr uint32_t threads_per_block = 256;
  const uint32_t v_blocks = (graph.vertex_count + threads_per_block - 1) / threads_per_block;
  const uint32_t e_blocks = (graph.edge_count + threads_per_block - 1) / threads_per_block;
  float* current = device_a.data();
  float* next = device_b.data();

  sw_encode.start();
  transform_domain</*encode=*/true><<<e_blocks, threads_per_block>>>(
      /*beta=*/1.0f, device_weights.data(), graph.edge_count);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  sw_encode.stop();

  CusparseSpmv spmv(graph, device_offsets.data(), device_columns.data(), device_weights.data(),
                    current, next);

  Stopwatch sw_kernel("kernel", false);
  for (uint32_t repetition = 0; repetition < repetitions; ++repetition) {
    ScopedTimer st_kernel(sw_kernel);
    ScopedTimer st_repetition("kernel.rep");
    sw_memcpy.start();
    CHECK_CUDA(cudaMemcpy(current, initial.data(), vertex_bytes, cudaMemcpyHostToDevice));
    sw_memcpy.stop();

    sw_encode.start();
    transform_domain</*encode=*/true><<<v_blocks, threads_per_block>>>(
        /*beta=*/1.0f, current, graph.vertex_count);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    sw_encode.stop();

    for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
      CHECK_CUDA(cudaMemcpy(next, current, vertex_bytes, cudaMemcpyDeviceToDevice));
      spmv.multiply(current, next);
      std::swap(current, next);
    }
    sw_decode.start();
    sw_kernel.start();
    transform_domain</*encode=*/false><<<v_blocks, threads_per_block>>>(
        /*beta=*/1.0f, current, graph.vertex_count);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    sw_kernel.stop();
    sw_decode.stop();

    sw_memcpy.start();
    CHECK_CUDA(cudaMemcpy(result.data(), current, vertex_bytes, cudaMemcpyDeviceToHost));
    sw_memcpy.stop();
  }

  return result;
}
