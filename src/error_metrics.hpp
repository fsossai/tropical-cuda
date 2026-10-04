#pragma once

#include "graph.hpp"

#include <stdint.h>
#include <vector>

// Compute exact distances on the CPU with Dijkstra's algorithm over outgoing CSR rows.
std::vector<float> dijkstra_sssp(CsrGraphView graph, uint32_t source);

// Print how far distances deviate from reference distances, one metric per line.
void print_error_metrics(const std::vector<float>& reference, const std::vector<float>& distances,
                         bool integer_weights);
