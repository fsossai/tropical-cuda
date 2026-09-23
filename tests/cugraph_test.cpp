#include "cugraph_solver.hpp"
#include "graph.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

int main() {
  const std::vector<Edge> edges = {{0, 1, 5.0f}, {0, 1, 9.0f},  {0, 2, 2.0f}, {2, 1, 1.0f},
                                   {1, 3, 4.0f}, {2, 3, 10.0f}, {4, 5, 1.0f}};
  const auto graph = build_csr(edges, 0);
  const auto distances = run_cugraph_sssp(graph, 0, 2);
  const std::vector<float> expected = {0.0f, 3.0f, 2.0f, 7.0f};
  if (distances.size() != 6) {
    throw std::runtime_error("cuGraph returned the wrong vertex count");
  }
  for (std::size_t vertex = 0; vertex < expected.size(); ++vertex) {
    if (std::fabs(distances[vertex] - expected[vertex]) > 0.00001f) {
      throw std::runtime_error("cuGraph returned an incorrect distance");
    }
  }
  if (!std::isinf(distances[4]) || !std::isinf(distances[5])) {
    throw std::runtime_error("cuGraph did not mark the unreachable vertex");
  }
}
