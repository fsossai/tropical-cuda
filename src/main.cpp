#include "cugraph_solver.hpp"
#include "graph.hpp"
#include "tropical_apx.cuh"
#include "tropical_common.cuh"
#include "tropical_exact.cuh"

#include <timers/ScopedTimer.hpp>

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
  std::string graph_path;
  std::optional<uint32_t> source;
  std::string algorithm = "tropical_exact";
  WeightMode weights = WeightMode::unit;
  std::optional<std::string> output_path;
  uint32_t repetitions = 1;
  std::optional<uint32_t> max_iterations;
  std::optional<float> beta;
};

void print_usage(const char* program) {
  std::cout << "Usage: " << program << " <graph.txt> [options]\n"
            << "\n"
            << "Options:\n"
            << "  --source NODE          Source vertex; defaults to the first edge source\n"
            << "  --algorithm NAME       tropical_exact (default), tropical_apx, cugraph\n"
            << "  --weights MODE         unit (default) or weighted\n"
            << "  --repetitions N        Number of SSSP runs (default: 1)\n"
            << "  --max-iterations N     Maximum Bellman-Ford iterations\n"
            << "  --beta VALUE           Positive tropical_apx approximation parameter\n"
            << "  --output PATH          Write distances to PATH\n"
            << "  --help                 Show this message\n";
}

uint32_t parse_uint32(std::string_view value, std::string_view option) {
  uint32_t result = 0;
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
  if (error != std::errc{} || end != value.data() + value.size()) {
    throw std::runtime_error("invalid value for " + std::string(option));
  }
  return result;
}

// Parse a finite, strictly positive floating-point command-line value.
float parse_positive_float(std::string_view value, std::string_view option) {
  const std::string text(value);
  char* end = nullptr;
  errno = 0;
  const float result = std::strtof(text.c_str(), &end);
  if (errno == ERANGE || end != text.c_str() + text.size() || !std::isfinite(result) ||
      result <= 0.0f) {
    throw std::runtime_error("invalid value for " + std::string(option));
  }

  return result;
}

Options parse_options(int argc, char** argv) {
  if (argc == 1) {
    print_usage(argv[0]);
    std::exit(0);
  }

  Options options;
  options.graph_path = argv[1];

  for (int index = 2; index < argc; ++index) {
    const std::string_view argument = argv[index];
    const auto require_value = [&]() -> std::string_view {
      if (++index == argc) {
        throw std::runtime_error("missing value for " + std::string(argument));
      }
      return argv[index];
    };

    if (argument == "--help") {
      print_usage(argv[0]);
      std::exit(0);
    }
    if (argument == "--source") {
      options.source = parse_uint32(require_value(), argument);
    } else if (argument == "--algorithm") {
      options.algorithm = require_value();
      if (options.algorithm != "cugraph" && options.algorithm != "tropical_exact" &&
          options.algorithm != "tropical_apx") {
        throw std::runtime_error("unknown algorithm: " + options.algorithm);
      }
    } else if (argument == "--weights") {
      const auto mode = require_value();
      if (mode == "unit") {
        options.weights = WeightMode::unit;
      } else if (mode == "weighted") {
        options.weights = WeightMode::file;
      } else {
        throw std::runtime_error("unknown weight mode: " + std::string(mode));
      }
    } else if (argument == "--repetitions") {
      options.repetitions = parse_uint32(require_value(), argument);
      if (options.repetitions == 0) {
        throw std::runtime_error("--repetitions must be positive");
      }
    } else if (argument == "--max-iterations") {
      options.max_iterations = parse_uint32(require_value(), argument);
    } else if (argument == "--beta") {
      options.beta = parse_positive_float(require_value(), argument);
    } else if (argument == "--output") {
      options.output_path = std::string(require_value());
    } else {
      throw std::runtime_error("unknown option: " + std::string(argument));
    }
  }

  return options;
}

std::string read_file(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("could not open graph file: " + path);
  }

  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void write_distances(const std::string& path, const std::vector<float>& distances) {
  ScopedTimer st("output");
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("could not create output file: " + path);
  }

  output << std::setprecision(std::numeric_limits<float>::max_digits10);
  for (uint32_t vertex = 0; vertex < distances.size(); ++vertex) {
    output << vertex << ' ';
    if (std::isinf(distances[vertex])) {
      output << "inf";
    } else {
      output << distances[vertex];
    }
    output << '\n';
  }
}

} // namespace

int main(int argc, char** argv) {
  ScopedTimer st_total("total");

  try {
    Options options;
    try {
      options = parse_options(argc, argv);
    } catch (const std::exception& error) {
      std::cerr << "error: " << error.what() << '\n';
      print_usage(argv[0]);
      return 1;
    }

    if (options.algorithm == "cugraph" && options.max_iterations) {
      throw std::runtime_error("--max-iterations is unsupported by cugraph");
    }
    if (options.algorithm != "tropical_apx" && options.beta) {
      throw std::runtime_error("--beta is only supported by tropical_apx");
    }

    TIMER_START("file_io");
    const auto contents = read_file(options.graph_path);
    TIMER_STOP();

    TIMER_START("parsing");
    auto edges = parse_edges(contents, options.weights);
    if (edges.empty()) {
      throw std::runtime_error("graph contains no edges");
    }
    const auto source = options.source.value_or(edges.front().source);
    TIMER_STOP();

    TIMER_START("csr_build");
    auto graph = build_csr(edges, source);
    TIMER_STOP();

    std::cout << "graph   : " << options.graph_path << '\n'
              << "vertices: " << graph.vertex_count << '\n'
              << "edges   : " << graph.edge_count << '\n'
              << "source  : " << source << '\n'
              << "algorithm: " << options.algorithm << '\n';

    if (options.algorithm == "tropical_exact" || options.algorithm == "tropical_apx") {
      Stopwatch sw_transpose("transpose", /*stats=*/false);
      ScopedTimer st_transpose(sw_transpose);
      graph = transpose_csr_with_cusparse(graph);
    }

    std::vector<float> distances;
    {
      ScopedTimer st("end_to_end");
      if (options.algorithm == "cugraph") {
        distances = run_cugraph_sssp(graph, source, options.repetitions);
      } else if (options.algorithm == "tropical_exact") {
        distances =
            run_tropical_exact_sssp(graph, source, options.repetitions, options.max_iterations);
      } else {
        distances =
            run_tropical_apx_sssp(graph, source, options.repetitions, options.max_iterations,
                                  options.beta.value_or(1.0f));
      }
    }

    if (options.output_path) {
      write_distances(*options.output_path, distances);
      std::cout << "output  : " << *options.output_path << '\n';
    }

    const auto reachable = static_cast<uint32_t>(std::count_if(
        distances.begin(), distances.end(), [](float distance) { return !std::isinf(distance); }));
    std::cout << "reachable: " << reachable << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
