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
  // Row u contains outgoing edges u -> v.
  std::vector<uint32_t> row_offsets;
  std::vector<uint32_t> column_indices;
  std::vector<float> weights;
};

std::vector<Edge> parse_edges(const std::string& contents, WeightMode weight_mode);
CsrGraph build_csr(const std::vector<Edge>& edges, uint32_t source);
