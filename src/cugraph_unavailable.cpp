#include "cugraph_solver.hpp"

#include <stdexcept>

bool cugraph_available() { return false; }

std::vector<float> run_cugraph_sssp(const CsrGraph&, uint32_t, uint32_t) {
  throw std::runtime_error("cuGraph backend unavailable; install libcugraph and reconfigure CMake");
}
