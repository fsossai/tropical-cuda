# Tropical Shortest Paths with CUDA

Did you know that you can solve the single-source shortest-path problem with matrix multiplication?

Given a weighted directed graph and a source vertex $s$, single-source shortest paths (SSSP) asks for the least cost of reaching every vertex from $s$. Let $d_i$ be the best distance known so far for vertex $i$, and let $w_{ji}$ be the weight of an edge from $j$ to $i$. One Bellman–Ford relaxation updates each distance by keeping its old value or taking a route through an incoming neighbor:

$$
d'_i = \min\left(d_i, \min_{(j, i) \in E}\left(d_j + w_{ji}\right)\right).
$$

Now make a matrix $A$: $A_{ij}$ is $w_{ji}$ when the edge exists, $\infty$ when it does not, and $0$ on the diagonal. The same update is a matrix-vector product if multiplication means addition and addition means `min`:

$$
d' = A \otimes d, \qquad (A \otimes d)_i = \min_j\left(A_{ij} + d_j\right).
$$

This is tropical algebra, also called the min-plus semiring. Repeating the product propagates distances through the graph. The idea is inspired by Michael Garland's paper, *Sparse matrix computations on manycore GPUs*, and this project asks whether the connection is useful in practice.

## Questions

- How does a tropical matrix-based SSSP solver compare with cuGraph?
- Can optimized sparse-matrix libraries such as cuSPARSE solve SSSP without a custom tropical matrix multiply?

I evaluate these questions on graphs from the [SNAP dataset collection](https://snap.stanford.edu/data/).

## Quick start

Build the project:

```sh
make
```

Download the SNAP inputs and convert them to memory-mappable CSR binaries:

```sh
make inputs
```

Run the hand-written tropical solver:

```sh
./build/sssp data/web-Google.csrbin --algorithm tropical_exact --repetitions 1
```

Run cuGraph on the same input:

```sh
./build/sssp data/web-Google.csrbin --algorithm cugraph --repetitions 1
```

## cuGraph dependency

The `cugraph` backend requires a separate [RAPIDS libcugraph installation](https://docs.rapids.ai/api/cugraph/legacy/installation/getting_cugraph/). CMake enables it when it finds `cugraph::cugraph_c`.

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