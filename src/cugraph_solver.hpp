#pragma once

#include "graph.hpp"

#include <stdint.h>
#include <vector>

bool cugraph_available();
std::vector<float> run_cugraph_sssp(CsrGraphView graph, uint32_t source, uint32_t repetitions);

inline std::vector<float> run_cugraph_sssp(const CsrGraph& graph, uint32_t source,
                                           uint32_t repetitions) {
  return run_cugraph_sssp(make_csr_graph_view(graph), source, repetitions);
}
