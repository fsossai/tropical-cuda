# Project design

## Scope

- Graphs are directed. Support unweighted graphs with unit integer edge weights and weighted graphs with non-negative floating-point edge weights.
- Each invocation computes distances from one source vertex.
- The variants will include a manually implemented tropical sparse matrix-vector multiplication (SpMV) and an approximation that reuses optimized cuSPARSE kernels, such as SpMV.
- A correct GPU result will serve as the accuracy reference. A CPU-only implementation, potentially from GAPBS, is also of interest as a comparison.

## Evaluation

Compare the variants on accuracy and time to answer, excluding graph loading time. Accuracy should include both a graph-wide summary or average error and a maximum or bound-oriented error measure. The exact metric definitions and the treatment of unreachable vertices remain to be specified. Performance measurements should make the speed-accuracy trade-off visible across variants.

The initial milestone is a correct GPU SSSP implementation. Approximate variants, comparisons, and analysis follow from that baseline. GPU hardware and CUDA version requirements are not yet specified.
