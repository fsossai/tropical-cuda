#pragma once

#include <stdint.h>
#include <string>
#include <vector>

enum class WeightMode { unit, file };

struct Edge {
  uint32_t source;
  uint32_t destination;
  float weight;
};

struct CsrGraph {
  uint32_t vertex_count = 0;
  uint32_t edge_count = 0;
  // Row u contains outgoing edges u -> v.
  std::vector<uint32_t> row_offsets;
  std::vector<uint32_t> column_indices;
  std::vector<float> weights;
};

// Reference CSR arrays without taking ownership of their storage.
struct CsrGraphView {
  uint32_t vertex_count = 0;
  uint32_t edge_count = 0;
  const uint32_t* row_offsets = nullptr;
  const uint32_t* column_indices = nullptr;
  const float* weights = nullptr;
};

// Create a non-owning view of an in-memory CSR graph.
inline CsrGraphView make_csr_graph_view(const CsrGraph& graph) {
  return {graph.vertex_count, graph.edge_count, graph.row_offsets.data(),
          graph.column_indices.data(), graph.weights.data()};
}

std::vector<Edge> parse_edges(const std::string& contents, WeightMode weight_mode);
CsrGraph build_csr(const std::vector<Edge>& edges, uint32_t source);
