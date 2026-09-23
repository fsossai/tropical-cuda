# Tropical Shortest Paths with CUDA

**Work in progress.** This project is an experiment in the performance and accuracy of GPU single-source shortest-path (SSSP) algorithms based on tropical algebra. It will compare several variants with different trade-offs, starting with a correct GPU algorithm before introducing approximations.

## Scope

- Graphs are directed. Support unweighted graphs with unit integer edge weights and weighted graphs with non-negative floating-point edge weights.
- Each invocation computes distances from one source vertex.
- The variants will include a manually implemented tropical sparse matrix-vector multiplication (SpMV) and an approximation that reuses optimized cuSPARSE kernels, such as SpMV.
- A correct GPU result will serve as the accuracy reference. A CPU-only implementation, potentially from GAPBS, is also of interest as a comparison.

## Evaluation

Compare the variants on accuracy and time to answer, excluding graph loading time. Accuracy should include both a graph-wide summary or average error and a maximum or bound-oriented error measure. The exact metric definitions and the treatment of unreachable vertices remain to be specified. Performance measurements should make the speed-accuracy trade-off visible across variants.

The initial milestone is a correct GPU SSSP implementation. Approximate variants, comparisons, and analysis follow from that baseline. GPU hardware and CUDA version requirements are not yet specified.

## Current command-line prototype

The current executable only parses a graph on the host and builds outgoing-edge CSR (one row per source vertex). It does not compute shortest paths yet. Build it with CMake and run it on a SNAP-style edge list:

```sh
cmake -S . -B build
cmake --build build
./build/tropical_sssp data/a.txt --algorithm exact_spmv --weights unit
```

The first argument must be a `.txt` path. Each non-comment line contains `source destination`, or `source destination weight` with `--weights file`. Vertex IDs are zero-based. Blank lines, `#` comments, and a standalone `...` placeholder are ignored. The vertex count is inferred from the largest ID in the edge list; header counts are informational. In particular, `data/a.txt` is an excerpt and its header does not describe only the edges present in that file.

`--source N` selects the source vertex; if omitted, it defaults to the source vertex of the first edge in the file. `--algorithm` accepts `exact_spmv`, `approx_cusparse`, `gapbs`, or `cugraph`. `--weights` accepts `unit` (default) or `file`. The CLI also accepts `--output PATH`, `--repetitions N`, and `--max-iterations N`; these are reserved for solver execution and currently have no computational effect. No output file is written yet.

The program reports timings for argument parsing, file I/O, edge parsing, CSR construction, reporting, and total elapsed time. These are host wall-clock timings measured with [fsossai/timers](https://github.com/fsossai/timers). Future time-to-answer measurements will exclude graph loading and preprocessing.

## TODO

- [ ] GPU: Exact tropical Bellman-Ford with hand-crafted SpMV.
- [ ] GPU: Approximate tropical Bellman-Ford using cuSPARSE.
- [ ] CPU: Multithreaded implementation based on GAPBS (Beamer et al.).
- [ ] GPU: Exact implementation using cuGraph.

## Graphs

[web-Google](https://snap.stanford.edu/data/web-Google.html) — web graph with varied topology.

[roadNet-CA](https://snap.stanford.edu/data/roadNet-CA.html) — road-like, sparse, relatively large diameter.

[soc-LiveJournal1](https://snap.stanford.edu/data/soc-LiveJournal1.html) — scale-free/social structure, high-degree hubs.

These [SNAP](https://snap.stanford.edu/data/index.html) datasets are candidates for the experimental study.

## Background

The tropical sparse-matrix approach is motivated by the following work:

```latex
@inproceedings{,
  title={Sparse matrix computations on manycore GPU's},
  author={Garland, Michael},
  booktitle={Proceedings of the 45th annual design automation conference},
  pages={2--6},
  year={2008}
}
```

GAPBS:
```latex
@article{beamer2015gap,
  title={The GAP benchmark suite},
  author={Beamer, Scott and Asanovi{\'c}, Krste and Patterson, David},
  journal={arXiv preprint arXiv:1508.03619},
  year={2015}
}
```
