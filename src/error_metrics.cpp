#include "error_metrics.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <stddef.h>
#include <stdexcept>
#include <utility>

namespace {

// Summarize relative errors by their mean and 99th percentile; reorders the input.
struct ErrorSummary {
  double mean = 0.0;
  double p99 = 0.0;
};

ErrorSummary summarize(std::vector<double>& errors) {
  ErrorSummary summary;

  if (errors.empty()) {
    return summary;
  }

  for (const double error : errors) {
    summary.mean += error;
  }

  summary.mean /= static_cast<double>(errors.size());
  const size_t rank = static_cast<size_t>(0.99 * static_cast<double>(errors.size() - 1));
  std::nth_element(errors.begin(), errors.begin() + rank, errors.end());
  summary.p99 = errors[rank];
  return summary;
}

} // namespace

// Run Dijkstra in double precision so the reference does not inherit float rounding.
std::vector<float> dijkstra_sssp(CsrGraphView graph, uint32_t source) {
  if (source >= graph.vertex_count) {
    throw std::invalid_argument("source vertex is outside the graph");
  }

  constexpr double infinity = std::numeric_limits<double>::infinity();
  std::vector<double> distances(graph.vertex_count, infinity);
  using Entry = std::pair<double, uint32_t>;
  std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> queue;
  distances[source] = 0.0;
  queue.push({0.0, source});

  while (!queue.empty()) {
    const auto [distance, vertex] = queue.top();
    queue.pop();

    if (distance > distances[vertex]) {
      continue;
    }

    for (uint32_t edge = graph.row_offsets[vertex]; edge < graph.row_offsets[vertex + 1]; ++edge) {
      const uint32_t neighbor = graph.column_indices[edge];
      const double candidate = distance + graph.weights[edge];

      if (candidate < distances[neighbor]) {
        distances[neighbor] = candidate;
        queue.push({candidate, neighbor});
      }
    }
  }

  return {distances.begin(), distances.end()};
}

// Compare against the reference over reachable vertices other than the source.
void print_error_metrics(const std::vector<float>& reference, const std::vector<float>& distances) {
  if (reference.size() != distances.size()) {
    throw std::invalid_argument("reference and distances differ in length");
  }

  size_t lost = 0;
  size_t spurious = 0;
  size_t overestimates = 0;
  size_t exact_count = 0;
  std::vector<double> relative_errors;
  relative_errors.reserve(reference.size());

  for (size_t vertex = 0; vertex < reference.size(); ++vertex) {
    const bool reachable = std::isfinite(reference[vertex]);
    const bool reported = std::isfinite(distances[vertex]);

    if (reachable && !reported) {
      ++lost;
    } else if (!reachable && reported) {
      ++spurious;
    }

    if (!reachable || !reported || reference[vertex] == 0.0f) {
      continue;
    }

    const double exact = reference[vertex];
    const double relative = (exact - distances[vertex]) / exact;
    relative_errors.push_back(relative);
    overestimates += relative < -1e-6 ? 1 : 0;
    exact_count += distances[vertex] == reference[vertex] ? 1 : 0;
  }

  const double exact_share =
      relative_errors.empty()
          ? 0.0
          : 100.0 * static_cast<double>(exact_count) / static_cast<double>(relative_errors.size());
  const ErrorSummary summary = summarize(relative_errors);
  std::cout << "lost: " << lost << '\n'
            << "spurious: " << spurious << '\n'
            << "overestimates: " << overestimates << '\n'
            << std::fixed << std::setprecision(4) << "rel_error_mean: " << summary.mean * 100.0
            << " %\n"
            << "rel_error_p99: " << summary.p99 * 100.0 << " %\n"
            << "exact: " << exact_share << " %\n"
            << std::defaultfloat;
}
