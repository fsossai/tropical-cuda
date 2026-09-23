#include "cugraph_solver.hpp"
#include "graph.hpp"
#include "tropical_solver.cuh"

#include <timers/ScopedTimer.hpp>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <stddef.h>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <string_view>

namespace {

struct Options {
  std::string graph_path;
  std::optional<uint32_t> source;
  std::string algorithm = "tropical_exact";
  WeightMode weights = WeightMode::unit;
  std::optional<std::string> output_path;
  uint32_t repetitions = 1;
  std::optional<uint32_t> max_iterations;
};

// Show the command syntax and supported options.
void print_usage(const char* program) {
  std::cout << "Usage: " << program << " graph.txt [options]\n"
            << "  --source N             source vertex, default: first edge's source\n"
            << "  --algorithm NAME       tropical_exact (default), tropical_apx, cugraph\n"
            << "  --weights MODE         unit (default) or file\n"
            << "  --output PATH          write cuGraph distances as vertex and distance pairs\n"
            << "  --repetitions N        compute repetitions, default 1\n"
            << "  --max-iterations N     iteration cap for iterative solvers\n"
            << "  --help                 show this help\n"
            << "The cuGraph solver requires libcugraph; other solvers are not implemented yet.\n";
}

uint32_t parse_number(std::string_view value, const std::string& name, bool allow_zero) {
  uint32_t number = 0;
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), number);
  if (error != std::errc{} || end != value.data() + value.size() || (!allow_zero && number == 0)) {
    throw std::runtime_error("invalid value for " + name + ": " + std::string(value));
  }
  return number;
}

// Parse and validate the graph path and command-line options.
Options parse_options(int argc, char** argv) {
  if (argc < 2 || argv[1][0] == '-') {
    throw std::runtime_error("the first argument must be a graph .txt path");
  }
  Options options;
  options.graph_path = argv[1];

  if (options.graph_path.size() < 4 ||
      options.graph_path.substr(options.graph_path.size() - 4) != ".txt") {
    throw std::runtime_error("graph path must end in .txt");
  }

  for (int i = 2; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--help") {
      print_usage(argv[0]);
      std::exit(0);
    }

    if (flag != "--source" && flag != "--algorithm" && flag != "--weights" && flag != "--output" &&
        flag != "--repetitions" && flag != "--max-iterations") {
      throw std::runtime_error("unknown option: " + flag);
    }

    if (i + 1 >= argc) {
      throw std::runtime_error("missing value for " + flag);
    }

    const std::string value = argv[++i];
    if (flag == "--source") {
      options.source = parse_number(value, flag, true);
    } else if (flag == "--algorithm") {
      if (value != "tropical_exact" && value != "tropical_apx" && value != "cugraph") {
        throw std::runtime_error("unknown algorithm: " + value);
      }
      options.algorithm = value;
    } else if (flag == "--weights") {
      if (value != "unit" && value != "file") {
        throw std::runtime_error("weights must be unit or file");
      }
      options.weights = value == "unit" ? WeightMode::unit : WeightMode::file;
    } else if (flag == "--output") {
      options.output_path = value;
    } else if (flag == "--repetitions") {
      options.repetitions = parse_number(value, flag, false);
    } else if (flag == "--max-iterations") {
      options.max_iterations = parse_number(value, flag, false);
    }
  }
  return options;
}

std::string read_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot open graph: " + path);
  }
  std::string contents(std::istreambuf_iterator<char>{input}, {});
  if (!input.eof() && input.fail()) {
    throw std::runtime_error("failed while reading graph: " + path);
  }
  return contents;
}

// Write one distance per vertex to the requested output file.
void write_distances(const std::string& path, const std::vector<float>& distances) {
  ScopedTimer st("output");
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("cannot open output: " + path);
  }

  output << std::setprecision(std::numeric_limits<float>::max_digits10);
  for (size_t vertex = 0; vertex < distances.size(); ++vertex) {
    output << vertex << ' ';
    if (std::isinf(distances[vertex])) {
      output << "inf\n";
    } else {
      output << distances[vertex] << '\n';
    }
  }

  output.close();
  if (!output) {
    throw std::runtime_error("failed while writing output: " + path);
  }
}

} // namespace

// Run the selected backend and report its results and phase timings.
int main(int argc, char** argv) {
  ScopedTimer st_total("total");

  try {
    if (argc == 1 || (argc == 2 && std::string_view(argv[1]) == "--help")) {
      print_usage(argv[0]);
      return 0;
    }
    Options options;
    try {
      options = parse_options(argc, argv);
    } catch (const std::exception& error) {
      std::cerr << "error: " << error.what() << '\n';
      print_usage(argv[0]);
      return 1;
    }

    if (options.algorithm == "cugraph" && !cugraph_available()) {
      throw std::runtime_error(
          "cuGraph backend unavailable; install libcugraph and reconfigure CMake");
    }

    if (options.algorithm == "cugraph" && options.max_iterations) {
      throw std::runtime_error("--max-iterations is not supported by cuGraph SSSP");
    }

    TIMER_START("file_io");
    const auto contents = read_file(options.graph_path);
    TIMER_STOP();

    TIMER_START("parsing");
    const auto edges = parse_edges(contents, options.weights);
    if (edges.empty()) {
      throw std::runtime_error("graph contains no edges");
    }
    const uint32_t source = options.source.value_or(edges.front().source);
    TIMER_STOP();

    TIMER_START("csr_build");
    auto graph = build_csr(edges, source);
    TIMER_STOP();

    std::cout << "graph          : " << options.graph_path << '\n'
              << "vertices       : " << graph.vertex_count << '\n'
              << "edges          : " << graph.column_indices.size() << '\n'
              << "csr_rows       : source\n"
              << "csr_columns    : destination\n"
              << "source         : " << source << '\n'
              << "algorithm      : " << options.algorithm << '\n'
              << "weights        : " << (options.weights == WeightMode::unit ? "unit" : "file")
              << '\n'
              << "repetitions    : " << options.repetitions << '\n'
              << "max_iterations : "
              << (options.max_iterations ? std::to_string(*options.max_iterations) : "auto")
              << '\n';

    if (options.algorithm == "tropical_exact" || options.algorithm == "tropical_apx") {
      Stopwatch sw_transpose("transpose", /*stats=*/false);
      ScopedTimer st_transpose(sw_transpose);
      graph = transpose_csr_with_cusparse(graph);
    }

    std::vector<float> distances;
    TIMER_START("end_to_end");
    if (options.algorithm == "cugraph") {
      distances = run_cugraph_sssp(graph, source, options.repetitions);
    } else if (options.algorithm == "tropical_exact") {
      distances =
          run_tropical_exact_sssp(graph, source, options.repetitions, options.max_iterations);
    }
    TIMER_STOP();

    if (options.output_path) {
      write_distances(*options.output_path, distances);
      std::cout << "output         : " << *options.output_path << "\n";
    }

    const auto reachable = std::count_if(distances.begin(), distances.end(),
                                         [](float distance) { return std::isfinite(distance); });
    std::cout << "reachable      : " << reachable << '\n'
              << "unreachable    : " << distances.size() - reachable << '\n';

    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
