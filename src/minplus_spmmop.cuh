#pragma once

#include "graph.hpp"

#include <optional>
#include <stdint.h>
#include <vector>

// Compute exact SSSP using cuSPARSE SpMMOp with JIT-linked min-plus operators.
std::vector<float> run_minplus_spmmop_sssp(const CsrGraph& graph, uint32_t source,
                                           uint32_t repetitions,
                                           std::optional<uint32_t> max_iterations);
