#pragma once

#include "graph.hpp"

#include <optional>
#include <stdint.h>
#include <vector>

// Compute exact SSSP using hand-written tropical SpMV.
std::vector<float> run_minplus_spmv_sssp(const CsrGraph& graph, uint32_t source,
                                         uint32_t repetitions,
                                         std::optional<uint32_t> max_iterations);
