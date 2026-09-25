# Tropical Shortest Paths with CUDA

Did you know that you can solve the Single-Source Shortest Path problem via matrix-vector multiplication?

For this project I want to implement an algorithm that I will refer to as "tropical" and compare it to cuGraph's implementation.
The tropical algorithm requires a variation of SpMV that cuSPARSE does not provide (see below).

Here, I want answer a few practical questions:

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

On the left, $+$ is addition and juxtaposition is multiplication. On the right, $\oplus$ is the addition operation and $\otimes$ is the multiplication operation, which can be defined differently.
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

Because a graph's adjacency matrix is sparse, this operation is an SpMV, but cuSPARSE provides only ordinary arithmetic SpMV rather than the min-plus variant.
One alternative is to transform the weights and distances so that conventional arithmetic SpMV approximates the tropical product.
For a positive parameter $\beta$, define the forward transformation and its inverse as:

$$
T_\beta(x) = e^{-\beta x}
\qquad\qquad
T_\beta^{-1}(z) = -\frac{1}{\beta}\log z.
$$

Thanks to the property of exponentials, the ordinary sum of two scalars $x$ and $y$ is a product in the transformed space:

$$
\quad T_\beta(x) T_\beta(y) = T_\beta(x+y) 
$$

For the min operation, one can prove that ordinary addition in the transformed space becomes equivalent in the limit:

$$
\min(x, y) = \lim_{\beta \to \infty} T_\beta^{-1}\left(T_\beta(x) + T_\beta(y)\right).
$$

**Proof.** Let $\beta > 0$, $m = \min(x, y)$ and $d = |x - y|$. Factoring out $e^{-\beta m}$,

$$
T_\beta^{-1}\left(e^{-\beta x} + e^{-\beta y}\right)
= -\frac{1}{\beta}\log\left(e^{-\beta m}\left(1 + e^{-\beta d}\right)\right)
= m - \frac{1}{\beta}\log\left(1 + e^{-\beta d}\right).
$$

Letting $\beta \to \infty$, the squeeze theorem gives the limit $m = \min(x, y)$. $\blacksquare$
As a result, the expression is a soft-min, and it approaches the true minimum as $\beta$ grows.

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
