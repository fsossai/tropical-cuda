#include "graph.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

std::vector<Edge> parse_edges(const std::string& contents, WeightMode weight_mode) {
  std::vector<Edge> edges;
  std::istringstream input(contents);
  std::string line;
  std::size_t line_number = 0;

  while (std::getline(input, line)) {
    ++line_number;
    const auto first = line.find_first_not_of(" \t\r");

    if (first == std::string::npos || line[first] == '#') {
      continue;
    }

    const auto last = line.find_last_not_of(" \t\r");
    if (line.substr(first, last - first + 1) == "...") {
      continue;
    }

    std::istringstream row(line);
    std::uint64_t source = 0;
    std::uint64_t destination = 0;
    double weight = 1.0;

    if (!(row >> source >> destination)) {
      throw std::runtime_error("invalid edge at line " + std::to_string(line_number));
    }

    if (weight_mode == WeightMode::file && !(row >> weight)) {
      throw std::runtime_error("missing weight at line " + std::to_string(line_number));
    }

    std::string extra;
    if (row >> extra) {
      throw std::runtime_error("unexpected field at line " + std::to_string(line_number));
    }

    if (source > std::numeric_limits<std::uint32_t>::max() ||
        destination > std::numeric_limits<std::uint32_t>::max() || !std::isfinite(weight) ||
        weight < 0 || weight > std::numeric_limits<float>::max()) {
      throw std::runtime_error("out-of-range edge at line " + std::to_string(line_number));
    }
    edges.push_back({static_cast<std::uint32_t>(source), static_cast<std::uint32_t>(destination),
                     static_cast<float>(weight)});
  }
  return edges;
}

CsrGraph build_csr(const std::vector<Edge>& edges, std::uint32_t source) {
  if (edges.empty()) {
    throw std::runtime_error("graph contains no edges");
  }

  std::uint64_t vertex_count = 0;
  for (const auto& edge : edges) {
    vertex_count = std::max(vertex_count, static_cast<std::uint64_t>(edge.source) + 1);
    vertex_count = std::max(vertex_count, static_cast<std::uint64_t>(edge.destination) + 1);
  }

  if (vertex_count > std::numeric_limits<std::uint32_t>::max() ||
      edges.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("graph exceeds 32-bit CSR limits");
  }

  if (source >= vertex_count) {
    throw std::runtime_error("source vertex is outside the graph");
  }

  CsrGraph graph;
  graph.vertex_count = static_cast<std::uint32_t>(vertex_count);
  graph.row_offsets.assign(vertex_count + 1, 0);
  graph.column_indices.resize(edges.size());
  graph.weights.resize(edges.size());

  for (const auto& edge : edges) {
    ++graph.row_offsets[edge.source + 1];
  }

  for (std::size_t i = 1; i < graph.row_offsets.size(); ++i) {
    graph.row_offsets[i] += graph.row_offsets[i - 1];
  }

  auto positions = graph.row_offsets;
  for (const auto& edge : edges) {
    const auto position = positions[edge.source]++;
    graph.column_indices[position] = edge.destination;
    graph.weights[position] = edge.weight;
  }
  return graph;
}
