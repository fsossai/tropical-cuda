#pragma once

#include <cstdint>
#include <string>
#include <vector>

enum class WeightMode { unit, file };

struct Edge {
  std::uint32_t source;
  std::uint32_t destination;
  float weight;
};

struct CsrGraph {
  std::uint32_t vertex_count = 0;
  // Row u contains outgoing edges u -> v.
  std::vector<std::uint32_t> row_offsets;
  std::vector<std::uint32_t> column_indices;
  std::vector<float> weights;
};

std::vector<Edge> parse_edges(const std::string& contents, WeightMode weight_mode);
CsrGraph build_csr(const std::vector<Edge>& edges, std::uint32_t source);
