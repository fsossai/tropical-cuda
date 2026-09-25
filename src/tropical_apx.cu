#include "tropical_apx.cuh"
#include "tropical_common.cuh"

#include <timers/ScopedTimer.hpp>

#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <math.h>
#include <memory>
#include <stdexcept>
#include <string>

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
  float beta_ = 0.0f;
  size_t workspace_bytes_ = 0;
  std::unique_ptr<DeviceBuffer<char>> workspace_;
};

// Replace each edge weight w with exp(-beta * w) in place.
__global__ void encode_weights(float beta, float* __restrict__ weights, uint32_t edge_count) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const int grid_stride = blockDim.x * gridDim.x;

  for (int i = tid; i < edge_count; i += grid_stride) {
    weights[i] = expf(-beta * weights[i]);
  }
}

// Return the distance of a pending vertex, or infinity for any other vertex.
struct PendingDistance {
  __host__ __device__ float operator()(const thrust::tuple<float, uint8_t>& vertex) const {
    return thrust::get<1>(vertex) ? thrust::get<0>(vertex) : INFINITY;
  }
};

// Encode the pending vertices that fall inside the current distance window and clear their
// pending flag; every other vertex is encoded as zero so that it does not contribute.
__global__ void encode_window(float beta, float shift, float window,
                              const float* __restrict__ distances, uint8_t* __restrict__ pending,
                              float* __restrict__ encoded, uint32_t vertex_count) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const int grid_stride = blockDim.x * gridDim.x;

  for (int i = tid; i < vertex_count; i += grid_stride) {
    // Distances are encoded relative to the shift, which is the smallest pending distance, rather
    // than relative to zero. Without the shift, a vertex at distance d becomes exp(-beta * d),
    // which rounds to zero once d is large, and far away vertices would silently look
    // unreachable. With the shift, every encoded value lies between exp(-beta * window) and 1, no
    // matter how far the window is from the source.
    //
    // Vertices outside the window are encoded as zero instead of being squeezed in. Their current
    // distance is already stored and will not be forgotten, because the decoded result is only
    // ever combined with the stored distance through an exact minimum. Pending vertices above the
    // window keep their pending flag, so they get their turn in a later step once the window
    // reaches them. Clearing the flag only for vertices we actually encode is what guarantees that
    // no improvement is dropped because it did not fit in floating point.
    const float offset = distances[i] - shift;
    if (pending[i] && offset <= window) {
      encoded[i] = expf(-beta * offset);
      pending[i] = 0;
    } else {
      encoded[i] = 0.0f;
    }
  }
}

// Decode the SpMV result and keep it only where it improves the stored distance.
__global__ void decode_and_relax(float beta, float shift, const float* __restrict__ products,
                                 float* __restrict__ distances, uint8_t* __restrict__ pending,
                                 uint32_t vertex_count) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const int grid_stride = blockDim.x * gridDim.x;

  for (int i = tid; i < vertex_count; i += grid_stride) {
    // A product of exactly zero means that no vertex in the window has an edge into this one, so
    // there is nothing to decode. Taking its logarithm would only produce an infinity.
    const float product = products[i];
    if (product == 0.0f) {
      continue;
    }

    // Undo the shift that was applied while encoding. The sum inside the logarithm is a soft
    // minimum over the edges coming from the window, so the candidate can be smaller than the
    // true shortest distance by at most log(k) / beta, where k is the number of such edges.
    const float candidate = shift - logf(product) / beta;

    // The soft minimum is only used inside one step. Across steps, results are combined with an
    // exact minimum, so the error does not pile up over all walks through the graph the way it
    // does when the transformed values are summed step after step without decoding.
    //
    // Because the soft minimum can underestimate, a cycle whose edges are lighter than
    // log(k) / beta can keep lowering its own distances a little on every visit. Keeping beta
    // above log(max in-degree) / (smallest weight) avoids this; otherwise the iteration cap is
    // what eventually stops the loop.
    if (candidate < distances[i]) {
      distances[i] = candidate;
      pending[i] = 1;
    }
  }
}

} // namespace

// Run the approximate backend as a windowed Bellman-Ford that uses cuSPARSE SpMV for each step.
std::vector<float> run_tropical_apx_sssp(const CsrGraph& graph, uint32_t source,
                                         uint32_t repetitions,
                                         std::optional<uint32_t> max_iterations, float beta) {
  Stopwatch sw_memcpy("kernel.memcpy", /*stats=*/false);
  Stopwatch sw_encode("kernel.encode", /*stats=*/false);

  if (graph.vertex_count == 0 || source >= graph.vertex_count || repetitions == 0 ||
      graph.row_offsets.size() != static_cast<size_t>(graph.vertex_count) + 1 ||
      graph.edge_count != graph.column_indices.size() ||
      graph.column_indices.size() != graph.weights.size() ||
      (max_iterations && *max_iterations == 0) || !std::isfinite(beta) || beta <= 0.0f) {
    throw std::invalid_argument("invalid tropical approximate SSSP inputs");
  }

  // Every quantity that goes through the SpMV must stay a normal float, meaning at least FLT_MIN,
  // which is about exp(-87.3). Below that, floats lose precision gradually and then become zero,
  // and a zero reads as "unreachable". The smallest term the SpMV sees is an encoded distance at
  // the far edge of the window multiplied by the encoded weight of the heaviest edge, so the
  // window and the heaviest edge have to share this budget: beta * (window + max_weight) must not
  // exceed 87.3. A larger beta gives a more accurate soft minimum but a narrower window, and so
  // more steps; beta now trades accuracy for parallelism instead of accuracy for underflow.
  //
  // Encoded distances are at most 1 and encoded weights are at most 1, so a row of the SpMV sums
  // at most k values no larger than 1. That cannot overflow, which is why the window only extends
  // above the shift and never below it.
  const float exponent_budget = -logf(std::numeric_limits<float>::min());
  const float max_weight =
      graph.weights.empty() ? 0.0f : *std::max_element(graph.weights.begin(), graph.weights.end());
  const float window = exponent_budget / beta - max_weight;
  if (!(window >= 0.0f)) {
    throw std::invalid_argument("--beta is too large for the heaviest edge weight; use at most " +
                                std::to_string(exponent_budget / max_weight));
  }

  const auto vertex_bytes = static_cast<size_t>(graph.vertex_count) * sizeof(float);
  const auto offset_bytes = graph.row_offsets.size() * sizeof(uint32_t);
  const auto edge_bytes = graph.edge_count * sizeof(uint32_t);
  const auto weight_bytes = graph.weights.size() * sizeof(float);
  DeviceBuffer<uint32_t> device_offsets(graph.row_offsets.size());
  DeviceBuffer<uint32_t> device_columns(graph.column_indices.size());
  DeviceBuffer<float> device_weights(graph.weights.size());
  DeviceBuffer<float> device_distances(graph.vertex_count);
  DeviceBuffer<uint8_t> device_pending(graph.vertex_count);
  DeviceBuffer<float> device_encoded(graph.vertex_count);
  DeviceBuffer<float> device_products(graph.vertex_count);

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
  std::vector<uint8_t> initial_pending(graph.vertex_count, 0);
  initial_pending[source] = 1;
  std::vector<float> result(graph.vertex_count);
  const uint32_t iterations = max_iterations.value_or(graph.vertex_count - 1);
  constexpr uint32_t threads_per_block = 256;
  const uint32_t v_blocks = (graph.vertex_count + threads_per_block - 1) / threads_per_block;
  const uint32_t e_blocks = (graph.edge_count + threads_per_block - 1) / threads_per_block;
  uint32_t iterations_performed = 0;

  sw_encode.start();
  encode_weights<<<std::max(e_blocks, 1u), threads_per_block>>>(beta, device_weights.data(),
                                                                graph.edge_count);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  sw_encode.stop();

  CusparseSpmv spmv(graph, device_offsets.data(), device_columns.data(), device_weights.data(),
                    device_encoded.data(), device_products.data());
  const auto distances_begin = thrust::device_pointer_cast(device_distances.data());
  const auto pending_begin = thrust::device_pointer_cast(device_pending.data());
  const auto vertices_begin =
      thrust::make_zip_iterator(thrust::make_tuple(distances_begin, pending_begin));
  const auto vertices_end = vertices_begin + graph.vertex_count;

  Stopwatch sw_kernel("kernel", false);
  for (uint32_t repetition = 0; repetition < repetitions; ++repetition) {
    ScopedTimer st_kernel(sw_kernel);
    ScopedTimer st_repetition("kernel.rep");
    sw_memcpy.start();
    CHECK_CUDA(
        cudaMemcpy(device_distances.data(), initial.data(), vertex_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_pending.data(), initial_pending.data(), graph.vertex_count,
                          cudaMemcpyHostToDevice));
    sw_memcpy.stop();

    for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
      // The shift is chosen fresh at every step as the smallest pending distance. Since all
      // weights are non-negative, nothing reachable from the pending vertices can end up closer
      // than that, so starting the window there wastes none of the floating-point range.
      const float shift = thrust::transform_reduce(
          thrust::device, vertices_begin, vertices_end, PendingDistance{},
          std::numeric_limits<float>::infinity(), thrust::minimum<float>{});
      if (std::isinf(shift)) {
        break;
      }

      encode_window<<<v_blocks, threads_per_block>>>(beta, shift, window, device_distances.data(),
                                                     device_pending.data(), device_encoded.data(),
                                                     graph.vertex_count);
      CHECK_CUDA(cudaGetLastError());
      spmv.multiply(device_encoded.data(), device_products.data());
      decode_and_relax<<<v_blocks, threads_per_block>>>(beta, shift, device_products.data(),
                                                        device_distances.data(),
                                                        device_pending.data(), graph.vertex_count);
      CHECK_CUDA(cudaGetLastError());
      ++iterations_performed;
    }

    sw_memcpy.start();
    CHECK_CUDA(
        cudaMemcpy(result.data(), device_distances.data(), vertex_bytes, cudaMemcpyDeviceToHost));
    sw_memcpy.stop();
  }

  std::cout << "iterations: " << iterations_performed << '\n';
  return result;
}
