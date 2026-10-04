#include "minplus_spmmop.cuh"
#include "tropical_common.cuh"

#include "deferred_timer.hpp"
#include <timers/ScopedTimer.hpp>

#include <cuda_runtime.h>
#include <nvrtc.h>

#include <thrust/device_ptr.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>

#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string.h>
#include <string>
#include <utility>
#include <vector>

#define CHECK_NVRTC(call)                                                                          \
  do {                                                                                             \
    const nvrtcResult nvrtc_status = (call);                                                       \
    if (nvrtc_status != NVRTC_SUCCESS) {                                                           \
      throw std::runtime_error(std::string("NVRTC error in ") + __func__ + " at line " +           \
                               std::to_string(__LINE__) +                                          \
                               " (" #call "): " + nvrtcGetErrorString(nvrtc_status));              \
    }                                                                                              \
  } while (false)

namespace {

// SpMMOp seeds some partial reductions with 0.0f, so min-plus cannot be expressed directly.
// Distances are therefore stored as bits(+inf) - bits(d): a decreasing map sending infinity to
// 0.0f and 0 to +inf, which turns min into max with 0.0f as its identity. Weights stay raw, and
// mul_op decodes, adds, and re-encodes. The epilogue folds in the previous distance, which plays
// the role of the zero-weight self loop. Each operator must be a separate LTO-IR module.
constexpr const char* add_operation =
    "__device__ float add_op(float a, float b) { return fmaxf(a, b); }";
constexpr const char* mul_operation =
    "__device__ float mul_op(float weight, float encoded) {\n"
    "  const float distance = __int_as_float(0x7f800000 - __float_as_int(encoded));\n"
    "  return __int_as_float(0x7f800000 - __float_as_int(distance + weight));\n"
    "}";
constexpr const char* epilogue_operation =
    "__device__ float epilogue(float product, float previous) { return fmaxf(product, previous); }";

// Map a non-negative distance to or from the max-reduction encoding; the map is an involution.
float flip_encoding(float value) {
  constexpr uint32_t infinity_bits = 0x7f800000u;
  uint32_t bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  bits = infinity_bits - bits;
  memcpy(&value, &bits, sizeof(bits));
  return value;
}

// Own an NVRTC program handle for the duration of one compilation.
class NvrtcProgram {
public:
  NvrtcProgram(const char* source, const char* name) {
    CHECK_NVRTC(nvrtcCreateProgram(&program_, source, name, 0, nullptr, nullptr));
  }

  ~NvrtcProgram() { nvrtcDestroyProgram(&program_); }

  NvrtcProgram(const NvrtcProgram&) = delete;
  NvrtcProgram& operator=(const NvrtcProgram&) = delete;

  nvrtcProgram get() const { return program_; }

private:
  nvrtcProgram program_ = nullptr;
};

// Compile one device function to LTO-IR for the current device's architecture.
std::vector<char> compile_lto_ir(const char* source, const char* name) {
  int device = 0;
  int major = 0;
  int minor = 0;
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  CHECK_CUDA(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
  const std::string architecture =
      "--gpu-architecture=compute_" + std::to_string(major) + std::to_string(minor);
  const char* options[] = {architecture.c_str(), "-dlto", "--relocatable-device-code=true"};

  NvrtcProgram program(source, name);
  const nvrtcResult status = nvrtcCompileProgram(program.get(), 3, options);
  if (status != NVRTC_SUCCESS) {
    size_t log_size = 0;
    nvrtcGetProgramLogSize(program.get(), &log_size);
    std::string log(log_size, '\0');
    nvrtcGetProgramLog(program.get(), log.data());
    throw std::runtime_error(std::string("NVRTC failed to compile ") + name + ": " + log);
  }

  size_t lto_ir_size = 0;
  CHECK_NVRTC(nvrtcGetLTOIRSize(program.get(), &lto_ir_size));
  std::vector<char> lto_ir(lto_ir_size);
  CHECK_NVRTC(nvrtcGetLTOIR(program.get(), lto_ir.data()));
  return lto_ir;
}

// Own cuSPARSE SpMMOp plans that compute output = min(W (min,+) input, output) for both
// directions of a ping-pong pair of distance buffers.
class CusparseSpmmOp {
public:
  CusparseSpmmOp(const CsrGraph& graph, uint32_t* offsets, uint32_t* columns, float* weights,
                 float* buffer_a, float* buffer_b) {
    const auto add_ir = compile_lto_ir(add_operation, "add_op.cu");
    const auto mul_ir = compile_lto_ir(mul_operation, "mul_op.cu");
    const auto epilogue_ir = compile_lto_ir(epilogue_operation, "epilogue.cu");
    const int64_t n = graph.vertex_count;

    CHECK_CUSPARSE(cusparseCreate(&handle_));
    CHECK_CUSPARSE(cusparseCreateCsr(&matrix_, n, n, graph.edge_count,
                                     reinterpret_cast<int*>(offsets),
                                     reinterpret_cast<int*>(columns), weights, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(
        cusparseCreateDnMat(&dense_a_, n, 1, n, buffer_a, CUDA_R_32F, CUSPARSE_ORDER_COL));
    CHECK_CUSPARSE(
        cusparseCreateDnMat(&dense_b_, n, 1, n, buffer_b, CUDA_R_32F, CUSPARSE_ORDER_COL));

    create_plan(&plan_a_to_b_, dense_a_, dense_b_, add_ir, mul_ir, epilogue_ir, workspace_a_to_b_);
    create_plan(&plan_b_to_a_, dense_b_, dense_a_, add_ir, mul_ir, epilogue_ir, workspace_b_to_a_);
  }

  ~CusparseSpmmOp() {
    if (plan_b_to_a_) {
      cusparseSpMMOp_destroyPlan(plan_b_to_a_);
    }
    if (plan_a_to_b_) {
      cusparseSpMMOp_destroyPlan(plan_a_to_b_);
    }
    cusparseDestroyDnMat(dense_b_);
    cusparseDestroyDnMat(dense_a_);
    cusparseDestroySpMat(matrix_);
    cusparseDestroy(handle_);
  }

  CusparseSpmmOp(const CusparseSpmmOp&) = delete;
  CusparseSpmmOp& operator=(const CusparseSpmmOp&) = delete;

  // Relax buffer_b from buffer_a when forward is true, otherwise buffer_a from buffer_b.
  void multiply(bool forward) {
    if (forward) {
      CHECK_CUSPARSE(cusparseSpMMOp(plan_a_to_b_, workspace_a_to_b_->data()));
    } else {
      CHECK_CUSPARSE(cusparseSpMMOp(plan_b_to_a_, workspace_b_to_a_->data()));
    }
  }

private:
  void create_plan(cusparseSpMMOpPlan_t* plan, cusparseDnMatDescr_t input,
                   cusparseDnMatDescr_t output, const std::vector<char>& add_ir,
                   const std::vector<char>& mul_ir, const std::vector<char>& epilogue_ir,
                   std::unique_ptr<DeviceBuffer<char>>& workspace) {
    size_t workspace_bytes = 0;
    CHECK_CUSPARSE(cusparseSpMMOp_createPlan(
        handle_, plan, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, matrix_,
        input, output, CUDA_R_32F, CUSPARSE_SPMM_OP_ALG_DEFAULT, add_ir.data(), add_ir.size(),
        mul_ir.data(), mul_ir.size(), epilogue_ir.data(), epilogue_ir.size(), &workspace_bytes));
    workspace = std::make_unique<DeviceBuffer<char>>(workspace_bytes);
  }

  cusparseHandle_t handle_ = nullptr;
  cusparseSpMatDescr_t matrix_ = nullptr;
  cusparseDnMatDescr_t dense_a_ = nullptr;
  cusparseDnMatDescr_t dense_b_ = nullptr;
  cusparseSpMMOpPlan_t plan_a_to_b_ = nullptr;
  cusparseSpMMOpPlan_t plan_b_to_a_ = nullptr;
  std::unique_ptr<DeviceBuffer<char>> workspace_a_to_b_;
  std::unique_ptr<DeviceBuffer<char>> workspace_b_to_a_;
};

} // namespace

// Iterate cuSPARSE min-plus relaxations over incoming CSR rows until distances stop changing.
std::vector<float> run_minplus_spmmop_sssp(const CsrGraph& graph, uint32_t source,
                                           uint32_t repetitions,
                                           std::optional<uint32_t> max_iterations) {
  DeferredTimer teardown("teardown");
  if (graph.vertex_count == 0 || source >= graph.vertex_count || repetitions == 0 ||
      graph.row_offsets.size() != static_cast<size_t>(graph.vertex_count) + 1 ||
      graph.edge_count != graph.column_indices.size() ||
      graph.column_indices.size() != graph.weights.size() ||
      (max_iterations && *max_iterations == 0)) {
    throw std::invalid_argument("invalid cuSPARSE SSSP inputs");
  }

  const auto vertex_bytes = static_cast<size_t>(graph.vertex_count) * sizeof(float);
  const auto offset_bytes = graph.row_offsets.size() * sizeof(uint32_t);
  const auto edge_bytes = graph.column_indices.size() * sizeof(uint32_t);
  const auto weight_bytes = graph.weights.size() * sizeof(float);

  TIMER_START("setup");
  DeviceBuffer<uint32_t> device_offsets(graph.row_offsets.size());
  DeviceBuffer<uint32_t> device_columns(graph.column_indices.size());
  DeviceBuffer<float> device_weights(graph.weights.size());
  DeviceBuffer<float> device_a(graph.vertex_count);
  DeviceBuffer<float> device_b(graph.vertex_count);

  CHECK_CUDA(cudaMemcpy(device_offsets.data(), graph.row_offsets.data(), offset_bytes,
                        cudaMemcpyHostToDevice));
  if (!graph.column_indices.empty()) {
    CHECK_CUDA(cudaMemcpy(device_columns.data(), graph.column_indices.data(), edge_bytes,
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_weights.data(), graph.weights.data(), weight_bytes,
                          cudaMemcpyHostToDevice));
  }
  CusparseSpmmOp spmm(graph, device_offsets.data(), device_columns.data(), device_weights.data(),
                      device_a.data(), device_b.data());
  CHECK_CUDA(cudaDeviceSynchronize());
  TIMER_STOP();

  std::vector<float> initial(graph.vertex_count,
                             flip_encoding(std::numeric_limits<float>::infinity()));
  initial[source] = flip_encoding(0.0f);
  std::vector<float> result(graph.vertex_count);
  const uint32_t iterations = max_iterations.value_or(graph.vertex_count - 1);
  uint32_t iterations_performed = 0;
  float* current = device_a.data();

  for (uint32_t repetition = 0; repetition < repetitions; ++repetition) {
    ScopedTimer st_kernel("kernel");
    iterations_performed = 0;
    CHECK_CUDA(cudaMemcpy(device_a.data(), initial.data(), vertex_bytes, cudaMemcpyHostToDevice));

    bool forward = true;
    current = device_a.data();
    float* next = device_b.data();
    bool changed = true;

    for (uint32_t iteration = 0; changed && iteration < iterations; ++iteration) {
      // The epilogue reads the output buffer as the previous distance, so it must start as a copy.
      CHECK_CUDA(cudaMemcpyAsync(next, current, vertex_bytes, cudaMemcpyDeviceToDevice));
      spmm.multiply(forward);
      changed = !thrust::equal(thrust::device, thrust::device_pointer_cast(current),
                               thrust::device_pointer_cast(current) + graph.vertex_count,
                               thrust::device_pointer_cast(next));
      std::swap(current, next);
      forward = !forward;
      ++iterations_performed;
    }
  }

  TIMER_START("download");
  CHECK_CUDA(cudaMemcpy(result.data(), current, vertex_bytes, cudaMemcpyDeviceToHost));
  for (float& distance : result) {
    distance = flip_encoding(distance);
  }
  TIMER_STOP();
  teardown.start();

  std::cout << "iterations: " << iterations_performed << '\n';
  return result;
}
