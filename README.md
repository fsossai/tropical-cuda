# Tropical Shortest Paths with CUDA

Did you know that you can solve the single-source shortest path (SSSP) problem via matrix-vector multiplication?

This project implements an algorithm that I will refer to as "tropical" and compare it to cuGraph's implementation.
The tropical algorithm requires a variation of SpMV that cuSPARSE's standard SpMV does not provide (see below).

Here, I want to answer a few practical questions:

- **Q1**: How does a tropical SSSP solver compare with the cuGraph implementation?
- **Q2**: Can optimized sparse-matrix libraries such as cuSPARSE solve SSSP without a custom tropical matrix multiply?

I evaluate these questions on graphs from the [SNAP dataset collection](https://snap.stanford.edu/data/).

## Tropical SpMV

### The Exact Formulation

A matrix-vector product computes each output element by combining a matrix row with an input vector. In ordinary arithmetic and in a generic algebraic form, respectively, it is:

$$
\begin{aligned}
y_i &= \sum_j W_{ij} x_j
&\qquad\qquad
y_i &= \bigoplus_j \left(W_{ij} \otimes x_j\right).
\end{aligned}
$$

On the left, $+$ is addition and juxtaposition is multiplication. On the right, $\oplus$ is the addition operation and $\otimes$ is the multiplication operation, which a semiring is free to redefine.
For the min-plus, or [tropical](https://en.wikipedia.org/wiki/Tropical_semiring), semiring, they are:

$$
a \oplus b = \min(a, b)
\qquad\qquad
a \otimes b = a + b.
$$

Let $d_i$ be the best distance known so far for vertex $i$, and $w_{ij}$ be the cost of going from $j$ to $i$.
One [Bellman-Ford](https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm) relaxation is therefore a tropical matrix-vector product:

$$
d_i \leftarrow \bigoplus_j \left(w_{ij} \otimes d_j\right) = \min_j\left(w_{ij} + d_j\right).
$$

Here a missing edge has cost $\infty$ and $w_{ii} = 0$ for every vertex $i$.
Repeating the product propagates distances through the graph. The idea is inspired by Michael Garland's paper, *Sparse matrix computations on manycore GPUs*.

### An Approximate Formulation

Because a graph's adjacency matrix is sparse, this operation is an SpMV, but cuSPARSE's standard SpMV supports only ordinary arithmetic rather than the min-plus variant.
Its preview `cusparseSpMMOp` API accepts custom operators, which the `cusparse` backend uses to compute the exact min-plus product.
One alternative is to transform the weights and distances so that conventional arithmetic SpMV approximates the tropical product.
For a positive parameter $\beta$, define the forward transformation and its inverse as:

$$
T_\beta(x) = e^{-\beta x}
\qquad\qquad
T_\beta^{-1}(z) = -\frac{1}{\beta}\log z.
$$

Thanks to the property of exponentials, the ordinary sum of two scalars $x$ and $y$ is a product in the transformed space:

$$
T_\beta(x) T_\beta(y) = T_\beta(x+y)
$$

For the min operation, one can prove that ordinary addition in the transformed space becomes equivalent in the limit:

$$
\min(x, y) = \lim_{\beta \to \infty} T_\beta^{-1}\left(T_\beta(x) + T_\beta(y)\right).
$$

As a result, the expression is a soft-min, and it approaches the true minimum as $\beta$ grows.
Unfortunately, finite-precision arithmetic makes this transformation numerically fragile, as explained in [Floating-Point Considerations](docs/floating_point.md).

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
./build/sssp data/web-Google.csrbin --solver tropical_exact --runs 1
```

Run cuGraph on the same input:

```sh
./build/sssp data/web-Google.csrbin --solver cugraph --runs 1
```

## Metrics

Each run prints the time in seconds of the following steps:

- `setup`: one-time preparation on the GPU, such as allocating memory and uploading the graph.
- `kernel`: computing the distances, printed once per repetition.
- `download`: copying the distances back to the host.
- `end_to_end`: the whole solver call, which is roughly `setup`, plus every `kernel`, plus `download`.

None of them include loading the graph from disk, creating the CUDA context, or transposing the graph.

## cuGraph dependency

The `cugraph` backend requires a separate [RAPIDS libcugraph installation](https://docs.rapids.ai/api/cugraph/legacy/installation/getting_cugraph/). CMake enables it when it finds `cugraph::cugraph_c`.

## References

- Michael Garland. "Sparse matrix computations on manycore GPU's." *Proceedings of the 45th Annual Design Automation Conference (DAC)*, pp. 2–6, 2008.
- Andrew Davidson, Sean Baxter, Michael Garland, and John D. Owens. "Work-efficient parallel GPU methods for single-source shortest paths." *2014 IEEE 28th International Parallel and Distributed Processing Symposium (IPDPS)*, pp. 349–359, IEEE, 2014.
