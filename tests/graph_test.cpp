#include "graph.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace {
void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}
} // namespace

int main() {
  const std::string weighted = "# header\n0 2 1.5\n  ...  \n1\t2\t0.25\n2 3 4\n";
  const auto edges = parse_edges(weighted, WeightMode::file);
  require(edges.size() == 3, "wrong number of edges");
  const auto graph = build_csr(edges, 0);
  require(graph.vertex_count == 4, "wrong vertex count");
  require(graph.row_offsets == std::vector<std::uint32_t>({0, 1, 2, 3, 3}),
          "wrong CSR row offsets");
  require(graph.column_indices == std::vector<std::uint32_t>({2, 2, 3}),
          "wrong CSR column indices");
  require(std::fabs(graph.weights[1] - 0.25f) < 0.00001f, "wrong edge weight");

  const auto unit_edges = parse_edges("0 1\n", WeightMode::unit);
  require(unit_edges[0].weight == 1.0f, "unit edge weight was not assigned");
  try {
    parse_edges("0 1 -1\n", WeightMode::file);
    throw std::runtime_error("negative weight was accepted");
  } catch (const std::runtime_error& error) {
    require(std::string(error.what()).find("out-of-range edge") != std::string::npos,
            "wrong negative-weight error");
  }
  return 0;
}
