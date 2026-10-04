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
void print_error_metrics(const std::vector<float>& reference, const std::vector<float>& distances,
                         bool integer_weights) {
  if (reference.size() != distances.size()) {
    throw std::invalid_argument("reference and distances differ in length");
  }

  size_t lost = 0;
  size_t spurious = 0;
  size_t overestimates = 0;
  size_t rounded_exact = 0;
  std::vector<double> relative_errors;
  std::vector<double> rounded_errors;
  relative_errors.reserve(reference.size());
  rounded_errors.reserve(reference.size());

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
    // The approximation only underestimates, so rounding up recovers any error below one unit.
    // The small offset keeps float noise just above an integer from rounding up a full unit.
    const double rounded = std::ceil(distances[vertex] - 1e-3);
    relative_errors.push_back(relative);
    rounded_errors.push_back((exact - rounded) / exact);
    overestimates += relative < -1e-6 ? 1 : 0;
    rounded_exact += rounded == exact ? 1 : 0;
  }

  const size_t compared = relative_errors.size();
  const ErrorSummary raw = summarize(relative_errors);
  std::cout << "lost: " << lost << '\n'
            << "spurious: " << spurious << '\n'
            << "overestimates: " << overestimates << '\n'
            << std::fixed << std::setprecision(4) << "rel_error_mean: " << raw.mean * 100.0
            << " %\n"
            << "rel_error_p99: " << raw.p99 * 100.0 << " %\n";

  // Rounding only recovers exact distances when every distance is an integer.
  if (integer_weights && compared != 0) {
    const ErrorSummary rounded = summarize(rounded_errors);
    std::cout << "rounded_up_exact: "
              << 100.0 * static_cast<double>(rounded_exact) / static_cast<double>(compared)
              << " %\n"
              << "rounded_up_rel_error_mean: " << rounded.mean * 100.0 << " %\n"
              << "rounded_up_rel_error_p99: " << rounded.p99 * 100.0 << " %\n";
  }

  std::cout << std::defaultfloat;
}
