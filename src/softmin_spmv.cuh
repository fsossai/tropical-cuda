#pragma once

#include "graph.hpp"

#include <optional>
#include <stdint.h>
#include <vector>

// Compute an approximate SSSP solution using the accelerated backend.
std::vector<float> run_softmin_spmv_sssp(const CsrGraph& graph, uint32_t source,
                                         uint32_t repetitions,
                                         std::optional<uint32_t> max_iterations, float beta);
