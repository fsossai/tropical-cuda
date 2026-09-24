#pragma once

#include "graph.hpp"

#include <stdint.h>
#include <stddef.h>
#include <string>

// Pair a binary CSR graph with the input graph's default SSSP source.
struct BinaryCsrGraph {
  CsrGraph graph;
  uint32_t default_source = 0;
};

// Own a read-only memory mapping of a compact CSR binary graph.
class MappedCsrGraph {
public:
  explicit MappedCsrGraph(const std::string& path);
  ~MappedCsrGraph();

  MappedCsrGraph(const MappedCsrGraph&) = delete;
  MappedCsrGraph& operator=(const MappedCsrGraph&) = delete;
  MappedCsrGraph(MappedCsrGraph&& other) noexcept;
  MappedCsrGraph& operator=(MappedCsrGraph&& other) noexcept;

  CsrGraphView view() const { return view_; }
  uint32_t default_source() const { return default_source_; }

private:
  int file_descriptor_ = -1;
  void* mapping_ = nullptr;
  size_t mapping_size_ = 0;
  CsrGraphView view_;
  uint32_t default_source_ = 0;
};

// Read a compact CSR graph whose arrays are directly consumable by cuGraph.
BinaryCsrGraph read_csr_binary(const std::string& path);

// Write a compact CSR graph whose arrays are directly consumable by cuGraph.
void write_csr_binary(const std::string& path, const CsrGraph& graph, uint32_t default_source);
