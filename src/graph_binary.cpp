#include "graph_binary.hpp"

#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr std::array<char, 8> binary_magic = {'T', 'C', 'S', 'R', 'B', 'I', 'N', '\0'};
constexpr uint32_t binary_version = 2;

// Describe the fixed, cache-line-aligned prefix of a memory-mappable CSR file.
struct BinaryHeader {
  std::array<char, 8> magic;
  uint32_t version;
  uint32_t vertex_count;
  uint32_t edge_count;
  uint32_t default_source;
  std::array<uint32_t, 10> reserved;
};
static_assert(sizeof(BinaryHeader) == 64, "binary CSR header must remain cache-line aligned");

// Check a CSR graph before serializing it.
void validate_graph(const CsrGraph& graph, uint32_t default_source) {
  if (graph.vertex_count == 0 || default_source >= graph.vertex_count ||
      graph.row_offsets.size() != static_cast<size_t>(graph.vertex_count) + 1 ||
      graph.edge_count != graph.column_indices.size() || graph.edge_count != graph.weights.size() ||
      graph.row_offsets.front() != 0 || graph.row_offsets.back() != graph.edge_count) {
    throw std::invalid_argument("invalid CSR graph for binary serialization");
  }
}

// Return the exact byte count required for a binary CSR graph.
size_t binary_size(uint32_t vertex_count, uint32_t edge_count) {
  return sizeof(BinaryHeader) + (static_cast<size_t>(vertex_count) + 1) * sizeof(uint32_t) +
         static_cast<size_t>(edge_count) * (sizeof(uint32_t) + sizeof(float));
}

// Validate a mapped file and expose its directly addressable CSR arrays.
CsrGraphView parse_mapping(const void* mapping, size_t mapping_size, uint32_t* default_source) {
  if (mapping_size < sizeof(BinaryHeader)) {
    throw std::runtime_error("truncated binary CSR graph");
  }

  const auto* header = static_cast<const BinaryHeader*>(mapping);
  if (header->magic != binary_magic) {
    throw std::runtime_error("invalid binary CSR graph magic");
  }
  if (header->version != binary_version) {
    throw std::runtime_error("unsupported binary CSR graph version");
  }
  if (header->vertex_count == 0 || header->default_source >= header->vertex_count ||
      mapping_size < binary_size(header->vertex_count, header->edge_count)) {
    throw std::runtime_error("invalid binary CSR graph header");
  }

  const auto* bytes = static_cast<const char*>(mapping) + sizeof(BinaryHeader);
  const auto* row_offsets = reinterpret_cast<const uint32_t*>(bytes);
  const auto* column_indices = row_offsets + static_cast<size_t>(header->vertex_count) + 1;
  const auto* weights = reinterpret_cast<const float*>(column_indices + header->edge_count);
  *default_source = header->default_source;
  return {header->vertex_count, header->edge_count, row_offsets, column_indices, weights};
}

} // namespace

void write_csr_binary(const std::string& path, const CsrGraph& graph, uint32_t default_source) {
  validate_graph(graph, default_source);

  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("could not create binary CSR graph: " + path);
  }

  const BinaryHeader header{binary_magic, binary_version, graph.vertex_count, graph.edge_count,
                            default_source, {}};
  output.write(reinterpret_cast<const char*>(&header), sizeof(header));
  output.write(reinterpret_cast<const char*>(graph.row_offsets.data()),
               graph.row_offsets.size() * sizeof(uint32_t));
  output.write(reinterpret_cast<const char*>(graph.column_indices.data()),
               graph.column_indices.size() * sizeof(uint32_t));
  output.write(reinterpret_cast<const char*>(graph.weights.data()), graph.weights.size() * sizeof(float));
  if (!output) {
    throw std::runtime_error("could not write binary CSR graph: " + path);
  }
}

MappedCsrGraph::MappedCsrGraph(const std::string& path) {
  file_descriptor_ = open(path.c_str(), O_RDONLY);
  if (file_descriptor_ < 0) {
    throw std::runtime_error("could not open binary CSR graph: " + path + ": " + std::strerror(errno));
  }

  struct stat status {};
  if (fstat(file_descriptor_, &status) != 0 || status.st_size <= 0) {
    close(file_descriptor_);
    file_descriptor_ = -1;
    throw std::runtime_error("could not stat binary CSR graph: " + path);
  }
  mapping_size_ = static_cast<size_t>(status.st_size);
  mapping_ = mmap(nullptr, mapping_size_, PROT_READ, MAP_PRIVATE, file_descriptor_, 0);
  if (mapping_ == MAP_FAILED) {
    mapping_ = nullptr;
    close(file_descriptor_);
    file_descriptor_ = -1;
    throw std::runtime_error("could not map binary CSR graph: " + path + ": " + std::strerror(errno));
  }

  try {
    view_ = parse_mapping(mapping_, mapping_size_, &default_source_);
  } catch (...) {
    munmap(mapping_, mapping_size_);
    close(file_descriptor_);
    mapping_ = nullptr;
    file_descriptor_ = -1;
    throw;
  }
}

MappedCsrGraph::~MappedCsrGraph() {
  if (mapping_) {
    munmap(mapping_, mapping_size_);
  }
  if (file_descriptor_ >= 0) {
    close(file_descriptor_);
  }
}

MappedCsrGraph::MappedCsrGraph(MappedCsrGraph&& other) noexcept
    : file_descriptor_(other.file_descriptor_), mapping_(other.mapping_),
      mapping_size_(other.mapping_size_), view_(other.view_), default_source_(other.default_source_) {
  other.file_descriptor_ = -1;
  other.mapping_ = nullptr;
  other.mapping_size_ = 0;
  other.view_ = {};
}

MappedCsrGraph& MappedCsrGraph::operator=(MappedCsrGraph&& other) noexcept {
  if (this != &other) {
    if (mapping_) {
      munmap(mapping_, mapping_size_);
    }
    if (file_descriptor_ >= 0) {
      close(file_descriptor_);
    }
    file_descriptor_ = other.file_descriptor_;
    mapping_ = other.mapping_;
    mapping_size_ = other.mapping_size_;
    view_ = other.view_;
    default_source_ = other.default_source_;
    other.file_descriptor_ = -1;
    other.mapping_ = nullptr;
    other.mapping_size_ = 0;
    other.view_ = {};
  }

  return *this;
}

BinaryCsrGraph read_csr_binary(const std::string& path) {
  MappedCsrGraph mapped(path);
  const auto view = mapped.view();
  BinaryCsrGraph binary;
  binary.default_source = mapped.default_source();
  binary.graph.vertex_count = view.vertex_count;
  binary.graph.edge_count = view.edge_count;
  binary.graph.row_offsets.assign(view.row_offsets, view.row_offsets + view.vertex_count + 1);
  binary.graph.column_indices.assign(view.column_indices, view.column_indices + view.edge_count);
  binary.graph.weights.assign(view.weights, view.weights + view.edge_count);
  validate_graph(binary.graph, binary.default_source);
  return binary;
}
