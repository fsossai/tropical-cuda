#pragma once

#include "graph.hpp"

#include <cstdint>
#include <vector>

bool cugraph_available();
std::vector<float> run_cugraph_sssp(const CsrGraph& graph, std::uint32_t source,
                                    std::uint32_t repetitions);
