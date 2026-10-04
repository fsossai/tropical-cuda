# Accuracy of `smoothmin_spmv`

`smoothmin_spmv` computes shortest-path distances with cuSPARSE's ordinary SpMV in the exponential domain, so its distances are approximate (see [Floating-Point Considerations](floating_point.md)).
This note measures how far they are from the exact ones, how the parameter $\beta$ affects them, and how to recover exact distances afterwards.
All graphs are unweighted, so every exact distance is an integer number of hops.

## Summary

- **$\beta$ must be at least 8.** At $\beta = 2$, 10 of the 13 graphs drift and never converge, and at $\beta = 4$ 5 still do. From $\beta = 8$ up, every graph converges in as many iterations as `minplus_spmv`.
- **The distances are never too long.** No converged run overestimated a distance, which matches the argument above, and the error shrinks as $1/\beta$.
- **At the default $\beta = 32$, short-path graphs are effectively exact.** On the 10 graphs other than the road networks, 99.6% to 100% of the distances are correct after rounding, with a mean error of at most 0.02%.
- **Long-path graphs stay approximate.** On the road networks, only 8% to 24% of the distances are correct at $\beta = 32$, because the error accumulates over hundreds of hops, but the error stays small: 0.14% to 0.21% on average and at most 0.5% at the 99th percentile.
- **Rounding up is the right way to recover integers.** At $\beta = 8$ it makes 7 of the 13 graphs fully exact, where rounding to the nearest integer makes none, and it keeps the guarantee of never overestimating.

## Measuring the Error

With $d_v$ the exact distance of vertex $v$ and $\tilde d_v$ the one reported by `smoothmin_spmv`, the relative error is

$$
e_v = \frac{d_v - \tilde d_v}{d_v},
$$

computed over the vertices that both solvers reach, excluding the source. It is positive when $\tilde d_v$ underestimates $d_v$.
The tables report its mean and its 99th percentile (p99). A lost vertex is reachable but reported as unreachable.

`sssp ... --error` prints these metrics after the run by comparing the result with a CPU Dijkstra reference:

```
lost: 0
spurious: 0
overestimates: 0
rel_error_mean: 0.5925 %
rel_error_p99: 1.1029 %
```

This example is roadNet-CA at $\beta = 8$.

## Method

Each graph was solved from its benchmark source with `minplus_spmv`, which gives the reference distances and its iteration count.
`smoothmin_spmv` was then run once per $\beta \in \{2, 4, 8, 16, 32\}$, capped at 3 times the exact iteration count plus 100.
A run that hits the cap has drifted instead of converging. The cap is generous: a converging run needs about as many iterations as `minplus_spmv`, at most 22 against 8 here, while a drifting run never stops on its own.
amazon0601 and the 4 web graphs were measured on the sources they had before the [source selection](source_selection.md) was refined.

## Convergence

| Graph | $\beta=2$ | $\beta=4$ | $\beta \ge 8$ |
|---|---|---|---|
| roadNet-CA, roadNet-PA, roadNet-TX | converged | converged | converged |
| amazon0601, email-EuAll, web-Google, orkut | capped | converged | converged |
| web-Stanford | capped, 930 vertices lost | converged | converged |
| wiki-Talk, soc-Pokec, soc-LiveJournal1 | capped | capped | converged |
| web-NotreDame | capped, 1582 vertices lost | capped, 628 vertices lost | converged |
| web-BerkStan | capped, 9257 vertices lost | capped, 8249 vertices lost | converged |

A small $\beta$ makes the soft minimum underestimate by up to $\log(k)/\beta$ per hop, where $k$ is the number of edges combined.
When that exceeds the weight of a cycle, the cycle keeps lowering its own distances, like a negative cycle, and the solver never converges.
A capped run underestimates by far more than the distances themselves and loses the vertices whose improvement was never propagated, so its result is unusable.
The rule $\beta > \ln(\text{max in-degree})$ suggested by the bound is too pessimistic: most web and social graphs already converge at $\beta = 4$, and the road networks at $\beta = 2$.

## Why $\beta = 32$ by default

`smoothmin_spmv` uses $\beta = 32$ unless `--beta` is given, for 3 reasons:

- **It converges on every graph.** All graphs converge from $\beta = 8$ up, so $\beta = 32$ is well clear of the drift seen at $\beta = 2$ and $4$.
- **It is 4 times more accurate than $\beta = 8$.** The error shrinks as $1/\beta$, and at $\beta = 32$ every short-path graph is exact or nearly so after rounding up.
- **It costs nothing.** A converged run takes as many iterations as `minplus_spmv` whatever $\beta$ is, so the larger value adds no work on these graphs.

The limit is the window: the solver requires $\beta \le 87.3 / w_{\max}$, where $w_{\max}$ is the heaviest edge weight, so $\beta = 32$ only accepts graphs whose heaviest edge is at most about 2.7.
All benchmark graphs have unit weights; a weighted graph needs a smaller `--beta`.

## Error of Converged Runs

Because every exact distance is an integer, `smoothmin_spmv` rounds its distances up as its last step (see [Recovering Exact Distances](#recovering-exact-distances)).
The table reports the share of vertices with the correct distance and the mean and 99th percentile of $e_v$ after that step.
It was measured by rounding to the nearest integer, so rounding up only improves on it.

| Graph | $\beta=8$: correct | $\beta=8$: mean / p99 | $\beta=32$: correct | $\beta=32$: mean / p99 |
|---|---|---|---|---|
| roadNet-CA | 0.36% | 0.73% / 1.36% | 10.9% | 0.21% / 0.46% |
| roadNet-PA | 1.73% | 0.76% / 1.42% | 24.0% | 0.17% / 0.50% |
| roadNet-TX | 0.09% | 0.59% / 1.19% | 8.08% | 0.14% / 0.31% |
| amazon0601 | 19.9% | 3.99% / 6.67% | 100% | 0% / 0% |
| email-EuAll | 95.8% | 0.85% / 20.0% | 100% | 0% / 0% |
| web-Google | 79.5% | 1.46% / 8.33% | 100% | 0% / 0% |
| web-Stanford | 82.5% | 1.52% / 14.3% | 99.9% | 0.01% / 0% |
| orkut | 85.6% | 2.86% / 20.0% | 100% | 0% / 0% |
| wiki-Talk | 81.2% | 4.69% / 25.0% | 100% | 0% / 0% |
| soc-Pokec | 90.1% | 1.54% / 16.7% | 100% | 0% / 0% |
| soc-LiveJournal1 | 84.9% | 1.70% / 12.5% | 100% | 0% / 0% |
| web-NotreDame | 99.5% | 0.03% / 0% | 100% | 0% / 0% |
| web-BerkStan | 55.1% | 2.62% / 10.0% | 99.6% | 0.02% / 0% |

- Every converged run takes as many iterations as `minplus_spmv`, and the error halves each time $\beta$ doubles.
- At $\beta = 32$, every short-path graph is exact or nearly so. On the road networks, the error accumulates along paths of hundreds of hops, so few distances are correct even at $\beta = 32$.
- At $\beta = 8$, a distance that rounds the wrong way is off by a whole unit, which is 10% to 25% of a distance of 4 to 10. This is why the 99th percentile is high on the short-path graphs even when most distances are correct.
- No converged run ever overestimated a distance or reached an unreachable vertex, even when compared without any tolerance.

## Never Overestimating

In exact arithmetic, a converged run cannot overestimate.
Each candidate distance is a soft minimum over the edges entering a vertex, and a sum of positive terms is at least its largest term, so the candidate is never larger than the best single path through the current window.
Stored distances only decrease, and a vertex whose distance improves stays pending until it has propagated that improvement, so at convergence $\tilde d_i \le \tilde d_j + w_{ji}$ holds for every edge $j \to i$.
Following the true shortest path from the source then gives $\tilde d_v \le d_v$ for every vertex.

The guarantee does not cover capped runs, whose pending improvements were never propagated, and floating-point rounding could in principle push a distance a few units in the last place above the exact one.
No such case was observed.

## Recovering Exact Distances

When every weight is an integer, so is every exact distance, and the approximate distances can be turned into integers.

- Rounding to the nearest integer recovers $d_v$ only when the error is below 0.5.
- Rounding up recovers it whenever the error is below 1, and since $\tilde d_v \le d_v$, it can never overshoot. `smoothmin_spmv` therefore rounds its distances up as its last step whenever all weights are integers.

Both were compared at $\beta = 8$. Here the 4 web graphs used their current sources and amazon0601 its previous one.

| Method | Graphs fully exact | Worst share exact | Worst p99 error | Overestimates |
|---|---|---|---|---|
| Round to nearest | 0 of 13 | 0.09% (roadNet-TX) | 25% | none |
| Round up | 7 of 13 | 1.40% (roadNet-TX) | 7.7% | none |

- Rounding up is free and keeps the guarantee of never overestimating. It is exact on the 7 graphs with short paths and little error, and reaches 89% to 95% on amazon0601, web-BerkStan and web-Stanford.
- Rounding to the nearest integer is worse than rounding up on every graph. Compared with no rounding, it usually lowers the mean error but raises the 99th percentile, because a distance that rounds the wrong way is off by a whole unit, which is 10% to 25% of a distance of 4 to 10.
- Neither rounding helps the road networks, where the error accumulates past 0.5, and mostly past 1, along paths of hundreds of hops.

Where rounding up is not enough, a larger $\beta$ is the simplest remedy. A more elaborate one is [Richardson extrapolation](https://en.wikipedia.org/wiki/Richardson_extrapolation), which combines 2 runs with different $\beta$ to cancel the leading error term.

## Open Issues

- The orkut input stores each undirected edge in one direction only, so its results do not reflect the real Orkut graph.
- All results are on unweighted graphs. On weighted graphs, heavy edges narrow the window and light cycles make drift more likely, which remains untested.
