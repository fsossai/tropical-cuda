#include "cugraph_solver.hpp"
#include "graph.hpp"

#include <timers/ScopedTimer.hpp>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

Stopwatch sw_total("total", /*stats=*/false);
Stopwatch sw_file_io("total.file_io", /*stats=*/false);
Stopwatch sw_parsing("total.parsing", /*stats=*/false);
Stopwatch sw_csr_build("total.csr_build", /*stats=*/false);
Stopwatch sw_output("total.output", /*stats=*/false);
Stopwatch sw_reporting("total.reporting", /*stats=*/false);

struct Options {
  std::string graph_path;
  std::optional<std::uint32_t> source;
  std::string algorithm = "exact_spmv";
  WeightMode weights = WeightMode::unit;
  std::optional<std::string> output_path;
  std::uint32_t repetitions = 1;
  std::optional<std::uint32_t> max_iterations;
};

void print_usage(const char* program) {
  std::cout << "Usage: " << program << " graph.txt [options]\n"
            << "  --source N             source vertex, default: first edge's source\n"
            << "  --algorithm NAME       exact_spmv (default), approx_cusparse, gapbs, cugraph\n"
            << "  --weights MODE         unit (default) or file\n"
            << "  --output PATH          write cuGraph distances as vertex and distance pairs\n"
            << "  --repetitions N        compute repetitions, default 1\n"
            << "  --max-iterations N     iteration cap for iterative solvers\n"
            << "  --help                 show this help\n"
            << "The cuGraph solver requires libcugraph; other solvers are not implemented yet.\n";
}

std::uint32_t parse_number(std::string_view value, const std::string& name, bool allow_zero) {
  std::uint32_t number = 0;
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), number);
  if (error != std::errc{} || end != value.data() + value.size() || (!allow_zero && number == 0)) {
    throw std::runtime_error("invalid value for " + name + ": " + std::string(value));
  }
  return number;
}

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

    if (i + 1 >= argc) {
      throw std::runtime_error("missing value for " + flag);
    }

    const std::string value = argv[++i];
    if (flag == "--source") {
      options.source = parse_number(value, flag, true);
    } else if (flag == "--algorithm") {
      if (value != "exact_spmv" && value != "approx_cusparse" && value != "gapbs" &&
          value != "cugraph") {
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
    } else {
      throw std::runtime_error("unknown option: " + flag);
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

void write_distances(const std::string& path, const std::vector<float>& distances) {
  ScopedTimer st1(sw_output);
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("cannot open output: " + path);
  }

  output << std::setprecision(std::numeric_limits<float>::max_digits10);
  for (std::size_t vertex = 0; vertex < distances.size(); ++vertex) {
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

int main(int argc, char** argv) {
  ScopedTimer st1(sw_total);
  try {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
      print_usage(argv[0]);
      return 0;
    }
    const Options options = parse_options(argc, argv);
    if (options.algorithm == "cugraph" && !cugraph_available()) {
      throw std::runtime_error(
          "cuGraph backend unavailable; install libcugraph and reconfigure CMake");
    }

    if (options.algorithm == "cugraph" && options.max_iterations) {
      throw std::runtime_error("--max-iterations is not supported by cuGraph SSSP");
    }

    sw_file_io.start();
    const auto contents = read_file(options.graph_path);
    sw_file_io.stop();

    sw_parsing.start();
    const auto edges = parse_edges(contents, options.weights);
    if (edges.empty()) {
      throw std::runtime_error("graph contains no edges");
    }
    const std::uint32_t source = options.source.value_or(edges.front().source);
    sw_parsing.stop();

    sw_csr_build.start();
    const auto graph = build_csr(edges, source);
    sw_csr_build.stop();

    std::vector<float> distances;
    if (options.algorithm == "cugraph") {
      distances = run_cugraph_sssp(graph, source, options.repetitions);
      if (options.output_path) {
        write_distances(*options.output_path, distances);
      }
    }

    sw_reporting.start();
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

    if (options.output_path) {
      std::cout << "output         : " << *options.output_path;
      if (options.algorithm != "cugraph") {
        std::cout << " (reserved; no distances written)";
      }
      std::cout << '\n';
    }

    if (options.algorithm == "cugraph") {
      const auto reachable = std::count_if(distances.begin(), distances.end(),
                                           [](float distance) { return std::isfinite(distance); });
      std::cout << "reachable      : " << reachable << '\n'
                << "unreachable    : " << distances.size() - reachable << '\n'
                << "status         : solved\n";
    } else {
      std::cout << "status         : parsed_csr_only\n";
    }
    sw_reporting.stop();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
