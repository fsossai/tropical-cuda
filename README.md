# Tropical Shortest Paths with CUDA

Did you know that you can solve the single-source shortest-path problem with matrix multiplication?

For this project I want to implement an algorithm that I will refer to as "tropical" and compare it to cuGraph's implementation.
The tropical algorithm requires a variation of SpMV that cuSPARSE does not provide.

Here, I want answer a few practical questions:

- **Q1**: How does a tropical SSSP solver compare with the cuGraph implementation?
- **Q2**: Can optimized sparse-matrix libraries such as cuSPARSE solve SSSP without a custom tropical matrix multiply?

I evaluate these questions on graphs from the [SNAP dataset collection](https://snap.stanford.edu/data/).

## Background

Let $d_i$ be the best distance known so far for vertex $i$, and $w_{ij}$ be the cost of going from $j$ to $i$.
One relaxation of the [Bellman-Ford](https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm) algorithm for a vertex $i$ can be written as follows:

$$
c_i \leftarrow \min_{(i, j) \in E}\left(d_j + w_{ij}\right)
$$

$$
d_i \leftarrow \min\left(d_i, c_i\right).
$$

where a missing edge has cost $\infty$ and $w_{ii} = 0$ for every vertex $i$.
Now if we take interpret matrix multiplication where scalar multiplication and addition and replaced by min and addition, respectively, then each relaxation of the algorithm becomes:
$$
\vec{d} \leftarrow W \vec{d}
$$

This new semiring is called min-plus, also called the [tropical semiring](https://en.wikipedia.org/wiki/Tropical_semiring)
Repeating the product propagates distances through the graph.
The idea is inspired by Michael Garland's paper, *Sparse matrix computations on manycore GPUs*.
Because a graph's adjacency matrix is sparse, this operation is an SpMV, but cuSPARSE provides only ordinary arithmetic SpMV rather than the min-plus variant.
One alternative is to transform the weights and distances so that conventional arithmetic SpMV approximates the tropical product.
For a positive parameter $\beta$, define the forward transformation and its inverse as:

$$
T_\beta(x) = e^{-\beta x}
\qquad\qquad
T_\beta^{-1}(z) = -\frac{1}{\beta}\log z.
$$

For two scalars $x$ and $y$, their ordinary sum is a product in the transformed space:
$$
-\frac{1}{\beta}\log {e^{-\beta x} e^{-\beta y}} = -\frac{1}{\beta}\log e^{-\beta (x+y)} = x + y
$$

This implements the $+$ operation of min-plus multiplication exactly, for every positive $\beta$.
For the min operation, ordinary addition in the transformed space becomes equivalent in the limit:

$$
\min(x, y) = \lim_{\beta \to \infty} T_\beta^{-1}\left(T_\beta(x) + T_\beta(y)\right).
$$

The transformation maps smaller distances to larger values. Therefore, as $\beta$ grows, the sum $T_\beta(x) + T_\beta(y)$ is dominated by the term associated with $\min(x, y)$; applying the inverse transformation then recovers that minimum.

Unfortunately, finite-precision arithmetic makes this transformation numerically fragile.
As $\beta$ grows, the decoded result approaches the minimum, but large values can underflow the exponentials while small values give a less accurate approximation.

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

## References

The tropical sparse-matrix approach is motivated by the following work:

```latex
@inproceedings{,
  title={Sparse matrix computations on manycore GPU's},
  author={Garland, Michael},
  booktitle={Proceedings of the 45th annual design automation conference},
  pages={2--6},
  year={2008}
}
@inproceedings{davidson2014work,
  title={Work-efficient parallel GPU methods for single-source shortest paths},
  author={Davidson, Andrew and Baxter, Sean and Garland, Michael and Owens, John D},
  booktitle={2014 IEEE 28th International Parallel and Distributed Processing Symposium},
  pages={349--359},
  year={2014},
  organization={IEEE}
}
```
