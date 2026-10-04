#include "cugraph_solver.hpp"
#include "cusparse_solver.cuh"
#include "graph.hpp"
#include "graph_binary.hpp"
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
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct Options {
  std::string graph_path;
  std::optional<uint32_t> source;
  std::string solver = "tropical_exact";
  WeightMode weights = WeightMode::unit;
  std::optional<std::string> output_path;
  uint32_t repetitions = 1;
  std::optional<uint32_t> max_iterations;
  std::optional<float> beta;
};

void print_usage(const char* program) {
  std::cout << "Usage: " << program << " <graph.txt|graph.csrbin> [options]\n"
            << "\n"
            << "Options:\n"
            << "  --source NODE          Source vertex; defaults to the first edge source\n"
            << "  --solver NAME          tropical_exact (default), tropical_apx, cusparse,\n"
            << "                         cugraph\n"
            << "  --weights MODE         unit (default) or weighted\n"
            << "  --runs N               Number of SSSP runs (default: 1)\n"
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
    } else if (argument == "--solver") {
      options.solver = require_value();
      if (options.solver != "cugraph" && options.solver != "tropical_exact" &&
          options.solver != "tropical_apx" && options.solver != "cusparse") {
        throw std::runtime_error("unknown solver: " + options.solver);
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
    } else if (argument == "--runs") {
      options.repetitions = parse_uint32(require_value(), argument);
      if (options.repetitions == 0) {
        throw std::runtime_error("--runs must be positive");
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

// Report whether a path names the compact CSR binary graph format.
bool is_csr_binary_path(const std::string& path) {
  constexpr std::string_view extension = ".csrbin";
  return path.size() >= extension.size() &&
         std::string_view(path).substr(path.size() - extension.size()) == extension;
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

    if (options.solver == "cugraph" && options.max_iterations) {
      throw std::runtime_error("--max-iterations is unsupported by cugraph");
    }
    if (options.solver != "tropical_apx" && options.beta) {
      throw std::runtime_error("--beta is only supported by tropical_apx");
    }

    // Create the CUDA context up front so that no solver's timings include its one-time cost.
    TIMER_START("cuda_init");
    CHECK_CUDA(cudaFree(nullptr));
    TIMER_STOP();

    CsrGraph graph;
    CsrGraphView graph_view;
    std::unique_ptr<MappedCsrGraph> mapped_graph;
    uint32_t source = 0;
    if (is_csr_binary_path(options.graph_path)) {
      TIMER_START("file_io");
      if (options.solver == "cugraph") {
        mapped_graph = std::make_unique<MappedCsrGraph>(options.graph_path);
        graph_view = mapped_graph->view();
        source = options.source.value_or(mapped_graph->default_source());
      } else {
        auto binary_graph = read_csr_binary(options.graph_path);
        graph = std::move(binary_graph.graph);
        graph_view = make_csr_graph_view(graph);
        source = options.source.value_or(binary_graph.default_source);
      }
      TIMER_STOP();
    } else {
      TIMER_START("file_io");
      const auto contents = read_file(options.graph_path);
      TIMER_STOP();

      TIMER_START("parsing");
      auto edges = parse_edges(contents, options.weights);
      if (edges.empty()) {
        throw std::runtime_error("graph contains no edges");
      }
      source = options.source.value_or(edges.front().source);
      TIMER_STOP();

      TIMER_START("csr_build");
      graph = build_csr(edges, source);
      TIMER_STOP();
      graph_view = make_csr_graph_view(graph);
    }

    std::cout << "graph   : " << options.graph_path << '\n'
              << "vertices: " << graph_view.vertex_count << '\n'
              << "edges   : " << graph_view.edge_count << '\n'
              << "source  : " << source << '\n'
              << "solver  : " << options.solver << '\n';

    if (options.solver == "tropical_exact" || options.solver == "tropical_apx" ||
        options.solver == "cusparse") {
      Stopwatch sw_transpose("transpose", /*stats=*/false);
      ScopedTimer st_transpose(sw_transpose);
      graph = transpose_csr_with_cusparse(graph);
      graph_view = make_csr_graph_view(graph);
    }

    std::vector<float> distances;
    {
      ScopedTimer st("end_to_end");
      if (options.solver == "cugraph") {
        distances = run_cugraph_sssp(graph_view, source, options.repetitions);
      } else if (options.solver == "tropical_exact") {
        distances =
            run_tropical_exact_sssp(graph, source, options.repetitions, options.max_iterations);
      } else if (options.solver == "cusparse") {
        distances = run_cusparse_sssp(graph, source, options.repetitions, options.max_iterations);
      } else {
        distances =
            run_tropical_apx_sssp(graph, source, options.repetitions, options.max_iterations,
                                  options.beta.value_or(8.0f));
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
