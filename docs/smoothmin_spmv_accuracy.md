# Accuracy of `smoothmin_spmv`

`smoothmin_spmv` computes shortest-path distances with cuSPARSE's ordinary SpMV in the exponential domain, so its distances are approximate (see [Floating-Point Considerations](floating_point.md)).
All graphs are unweighted, so every exact distance is an integer number of hops.

## Summary

- From $\beta = 8$ up, every graph converges in as many iterations as `minplus_spmv`.
- On the 10 graphs, with $\beta=32$, other than the road networks, 99.6% to 100% of the distances are correct after rounding, with a mean error of at most 0.02%.
- On the road networks, only 8% to 24% of the distances are correct at $\beta = 32$, because the error accumulates over hundreds of hops, but the error stays small: 0.14% to 0.21% on average and at most 0.5% at the 99th percentile.

## Error Metric

With $d_v$ the exact distance of vertex $v$ and $\tilde d_v$ the one reported by `smoothmin_spmv`, the relative error is

$$
e_v = \frac{d_v - \tilde d_v}{d_v},
$$

computed over the vertices that both solvers reach, excluding the source. It is positive when $\tilde d_v$ underestimates $d_v$.
The tables report its mean and its 99th percentile (p99). A lost vertex is reachable but reported as unreachable.

The option `--error` prints these metrics after the run by comparing the result with a CPU Dijkstra reference:

```
lost: 0
spurious: 0
overestimates: 0
rel_error_mean: 0.5925 %
rel_error_p99: 1.1029 %
```

## Convergence

A small $\beta$ makes the smooth minimum underestimate by up to $\log(k)/\beta$ per hop, where $k$ is the number of edges combined.
When that exceeds the weight of a cycle, the cycle keeps lowering its own distances, like a negative cycle, and the solver never converges.
A capped run underestimates by far more than the distances themselves and loses the vertices whose improvement was never propagated, so its result is unusable.
The rule $\beta > \ln(\text{max in-degree})$ suggested by the bound is too pessimistic: most web and social graphs already converge at $\beta = 4$, and the road networks at $\beta = 2$.

## Choice of $\beta$

`smoothmin_spmv` uses $\beta = 32$ unless `--beta` is given, for 3 reasons:

- It converges on every graph
- It is much more accurate than $\beta = 8$
- It does not affect the run time

The limit is the window: the solver requires $\beta \le 87.3 / w_{\max}$, where $w_{\max}$ is the heaviest edge weight, so $\beta = 32$ only accepts graphs whose heaviest edge is at most about 2.7.
All benchmark graphs have unit weights; a weighted graph needs a smaller `--beta`.

## Overestimation

In exact arithmetic, a converged run cannot overestimate.
Each candidate distance is a smooth minimum over the edges entering a vertex, and a sum of positive terms is at least its largest term, so the candidate is never larger than the best single path through the current window.
Stored distances only decrease, and a vertex whose distance improves stays pending until it has propagated that improvement, so at convergence $\tilde d_i \le \tilde d_j + w_{ji}$ holds for every edge $j \to i$.
Following the true shortest path from the source then gives $\tilde d_v \le d_v$ for every vertex.

The guarantee does not cover capped runs, whose pending improvements were never propagated, and floating-point rounding could in principle push a distance a few units in the last place above the exact one.
No such case was observed.

## Rounding

When every weight is an integer, so is every exact distance, and the approximate distances can be turned into integers.

- Rounding to the nearest integer recovers $d_v$ only when the error is below 0.5.
- Rounding up recovers it whenever the error is below 1, and since $\tilde d_v \le d_v$, it can never overshoot. `smoothmin_spmv` therefore rounds its distances up as its last step whenever all weights are integers.
- Rounding up is free and keeps the guarantee of never overestimating
- Rounding to the nearest integer is worse than rounding up on every graph

Where rounding up is not enough, a larger $\beta$ is the simplest remedy. A more elaborate one would be the [Richardson extrapolation](https://en.wikipedia.org/wiki/Richardson_extrapolation), which combines 2 runs with different $\beta$ to cancel the leading error term.

## Known Issues

- The orkut input stores each undirected edge in one direction only, so its results do not reflect the real Orkut graph.
