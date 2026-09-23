#include "tropical_common.cuh"

#include <limits>
#include <stdexcept>

namespace {

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

} // namespace

// Transpose CSR rows on the GPU to provide incoming edges to tropical solvers.
CsrGraph transpose_csr_with_cusparse(const CsrGraph& graph) {
  if (graph.vertex_count == 0 ||
      graph.row_offsets.size() != static_cast<size_t>(graph.vertex_count) + 1 ||
      graph.edge_count != graph.column_indices.size() ||
      graph.column_indices.size() != graph.weights.size() ||
      graph.vertex_count > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
      graph.column_indices.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("invalid CSR graph for transposition");
  }

  const int vertex_count = static_cast<int>(graph.vertex_count);
  const int edge_count = static_cast<int>(graph.edge_count);
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
  transpose.edge_count = graph.edge_count;
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
