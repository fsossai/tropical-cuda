#include "graph.hpp"
#include "graph_binary.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

// Print the graph converter command syntax.
void print_usage(const char* program) {
  std::cout << "Usage: " << program << " <input.txt> <output.csrbin> [--weights MODE]\n"
            << "  --weights MODE         unit (default) or weighted\n";
}

// Read an input text graph in one buffered operation.
std::string read_file(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("could not open graph file: " + path);
  }

  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 3 || argc > 5) {
      print_usage(argv[0]);
      return 1;
    }

    WeightMode weight_mode = WeightMode::unit;
    if (argc == 5) {
      if (std::string_view(argv[3]) != "--weights") {
        throw std::runtime_error("unknown option: " + std::string(argv[3]));
      }
      if (std::string_view(argv[4]) == "unit") {
        weight_mode = WeightMode::unit;
      } else if (std::string_view(argv[4]) == "weighted") {
        weight_mode = WeightMode::file;
      } else {
        throw std::runtime_error("unknown weight mode: " + std::string(argv[4]));
      }
    } else if (argc != 3) {
      throw std::runtime_error("missing value for --weights");
    }

    const auto edges = parse_edges(read_file(argv[1]), weight_mode);
    if (edges.empty()) {
      throw std::runtime_error("graph contains no edges");
    }
    const auto graph = build_csr(edges, edges.front().source);
    write_csr_binary(argv[2], graph, edges.front().source);
    std::cout << "vertices: " << graph.vertex_count << '\n'
              << "edges   : " << graph.edge_count << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
