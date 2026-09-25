#include "cugraph_solver.hpp"
#include "cuda_check.hpp"
#include "deferred_timer.hpp"

#include <memory>
#include <timers/ScopedTimer.hpp>

#include <cugraph_c/array.h>
#include <cugraph_c/graph.h>
#include <cugraph_c/resource_handle.h>
#include <cugraph_c/traversal_algorithms.h>

#include <limits>
#include <stddef.h>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

#define CHECK_CUGRAPH(code, error)                                                                 \
  do {                                                                                             \
    const cugraph_error_code_t cugraph_status = (code);                                            \
    std::unique_ptr<cugraph_error_t, decltype(&cugraph_error_free)> cugraph_error(                 \
        (error), &cugraph_error_free);                                                             \
    if (cugraph_status != CUGRAPH_SUCCESS) {                                                       \
      std::string message = std::string("cuGraph error in ") + __func__ + " at line " +            \
                            std::to_string(__LINE__) + " (code " +                                 \
                            std::to_string(static_cast<int>(cugraph_status)) + ")";                \
      if (cugraph_error) {                                                                         \
        message += ": ";                                                                           \
        message += cugraph_error_message(cugraph_error.get());                                     \
      }                                                                                            \
      throw std::runtime_error(message);                                                           \
    }                                                                                              \
  } while (false)

namespace {

using Handle = std::unique_ptr<cugraph_resource_handle_t, decltype(&cugraph_free_resource_handle)>;
using DeviceArray = std::unique_ptr<cugraph_type_erased_device_array_t,
                                    decltype(&cugraph_type_erased_device_array_free)>;
using DeviceView = std::unique_ptr<cugraph_type_erased_device_array_view_t,
                                   decltype(&cugraph_type_erased_device_array_view_free)>;
using Graph = std::unique_ptr<cugraph_graph_t, decltype(&cugraph_graph_free)>;
using Paths = std::unique_ptr<cugraph_paths_result_t, decltype(&cugraph_paths_result_free)>;

// Own a cuGraph device array initialized from host data.
class InputArray {
public:
  InputArray(const cugraph_resource_handle_t* handle, const void* data, size_t size,
             cugraph_data_type_id_t type)
      : array_(nullptr, &cugraph_type_erased_device_array_free),
        view_(nullptr, &cugraph_type_erased_device_array_view_free) {
    cugraph_type_erased_device_array_t* raw_array = nullptr;
    cugraph_error_t* error = nullptr;
    const auto code =
        cugraph_type_erased_device_array_create(handle, size, type, &raw_array, &error);
    array_.reset(raw_array);
    CHECK_CUGRAPH(code, error);

    view_.reset(cugraph_type_erased_device_array_view(array_.get()));
    if (!view_) {
      throw std::runtime_error("cuGraph device array view creation failed");
    }

    error = nullptr;
    const auto copy_code = cugraph_type_erased_device_array_view_copy_from_host(
        handle, view_.get(), reinterpret_cast<const byte_t*>(data), &error);
    CHECK_CUGRAPH(copy_code, error);
  }

  const cugraph_type_erased_device_array_view_t* view() const { return view_.get(); }

private:
  DeviceArray array_;
  DeviceView view_;
};

} // namespace

bool cugraph_available() { return true; }

// Build a cuGraph CSR graph and compute SSSP on the GPU for the requested runs.
std::vector<float> run_cugraph_sssp(CsrGraphView graph, uint32_t source, uint32_t repetitions) {
  DeferredTimer teardown("teardown");
  if (repetitions == 0) {
    throw std::runtime_error("cuGraph repetitions must be positive");
  }

  TIMER_START("setup");
  Handle handle(cugraph_create_resource_handle(nullptr), &cugraph_free_resource_handle);
  if (!handle) {
    throw std::runtime_error("cuGraph resource handle creation failed");
  }

  if (graph.vertex_count == 0 || source >= graph.vertex_count || !graph.row_offsets ||
      !graph.column_indices || !graph.weights ||
      graph.vertex_count > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
      graph.edge_count > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
    throw std::runtime_error("cuGraph requires 32-bit vertex IDs and CSR offsets");
  }

  InputArray device_offsets(handle.get(), reinterpret_cast<const int32_t*>(graph.row_offsets),
                            static_cast<size_t>(graph.vertex_count) + 1, INT32);
  InputArray device_indices(handle.get(), reinterpret_cast<const int32_t*>(graph.column_indices),
                            graph.edge_count, INT32);
  InputArray device_weights(handle.get(), graph.weights, graph.edge_count, FLOAT32);

  const cugraph_graph_properties_t properties{FALSE, TRUE};
  cugraph_graph_t* raw_graph = nullptr;
  cugraph_error_t* error = nullptr;
  const auto graph_code = cugraph_graph_create_sg_from_csr(
      handle.get(), &properties, device_offsets.view(), device_indices.view(),
      device_weights.view(), nullptr, nullptr, FALSE, FALSE, FALSE, FALSE, &raw_graph, &error);
  Graph device_graph(raw_graph, &cugraph_graph_free);
  CHECK_CUGRAPH(graph_code, error);
  CHECK_CUDA(cudaDeviceSynchronize());
  TIMER_STOP();

  Paths paths(nullptr, &cugraph_paths_result_free);
  for (uint32_t run = 0; run < repetitions; ++run) {
    paths.reset();
    cugraph_paths_result_t* raw_paths = nullptr;
    {
      ScopedTimer st_kernel("kernel");
      error = nullptr;
      const auto code =
          cugraph_sssp(handle.get(), device_graph.get(), source, std::numeric_limits<float>::max(),
                       FALSE, FALSE, &raw_paths, &error);
      CHECK_CUGRAPH(code, error);
      CHECK_CUDA(cudaDeviceSynchronize());
    }
    paths.reset(raw_paths);
  }

  TIMER_START("download");
  DeviceView vertices(cugraph_paths_result_get_vertices(paths.get()),
                      &cugraph_type_erased_device_array_view_free);
  DeviceView distances(cugraph_paths_result_get_distances(paths.get()),
                       &cugraph_type_erased_device_array_view_free);
  if (!vertices || !distances) {
    throw std::runtime_error("cuGraph SSSP returned incomplete results");
  }

  const auto count = cugraph_type_erased_device_array_view_size(vertices.get());
  if (count != cugraph_type_erased_device_array_view_size(distances.get()) ||
      cugraph_type_erased_device_array_view_type(vertices.get()) != INT32 ||
      cugraph_type_erased_device_array_view_type(distances.get()) != FLOAT32) {
    throw std::runtime_error("cuGraph SSSP returned unexpected result types");
  }

  std::vector<int32_t> vertex_ids(count);
  std::vector<float> values(count);
  error = nullptr;
  const auto vertices_code = cugraph_type_erased_device_array_view_copy_to_host(
      handle.get(), reinterpret_cast<byte_t*>(vertex_ids.data()), vertices.get(), &error);
  CHECK_CUGRAPH(vertices_code, error);
  error = nullptr;
  const auto distances_code = cugraph_type_erased_device_array_view_copy_to_host(
      handle.get(), reinterpret_cast<byte_t*>(values.data()), distances.get(), &error);
  CHECK_CUGRAPH(distances_code, error);
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> result(graph.vertex_count, std::numeric_limits<float>::infinity());
  for (size_t i = 0; i < count; ++i) {
    const auto vertex = vertex_ids[i];
    if (vertex < 0 || static_cast<uint64_t>(vertex) >= graph.vertex_count) {
      throw std::runtime_error("cuGraph SSSP returned an out-of-range vertex");
    }
    result[vertex] = values[i] == std::numeric_limits<float>::max()
                         ? std::numeric_limits<float>::infinity()
                         : values[i];
  }
  TIMER_STOP();
  teardown.start();

  return result;
}
