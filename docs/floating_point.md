# Floating-Point Considerations

This note complements the approximate formulation in the [README](../README.md#an-approximate-formulation), which defines the transformation $T_\beta(x) = e^{-\beta x}$ and its inverse $T_\beta^{-1}(z) = -\frac{1}{\beta}\log z$.

**Approximation error.** For $k$ terms with minimum $m$, the transformed sum satisfies $e^{-\beta m} \le \sum_j e^{-\beta x_j} \le k\,e^{-\beta m}$, so decoding gives

$$
m - \frac{\log k}{\beta} \le T_\beta^{-1}\left(\sum_j T_\beta(x_j)\right) \le m.
$$

The soft-min therefore never overestimates the true minimum, and its error is at most $\log(k)/\beta$.
Because $T_\beta^{-1}$ followed by $T_\beta$ is the identity, chaining transformed products without decoding in between yields a soft-min over every walk combined so far.
In that case $k$ is the number of walks reaching a vertex, which can be very large.

**Underflow.** In single precision, $e^{-\beta x}$ drops below the smallest normal number when $\beta x > 87.3$ and loses precision as a subnormal, then rounds to zero when $\beta x > 103.3$.
A vertex whose transformed distance underflows to zero decodes as $\infty$ and looks unreachable.
Double precision moves these thresholds to about $708.4$ and $744.4$.

**The trade-off.** If $D$ is the largest finite distance, keeping every value in the normal range requires $\beta \le 87.3 / D$, which makes the error bound for the farthest vertex at least $D \log(k) / 87.3$.
Relative to $D$, the guaranteed error is roughly $\log(k) / 87.3$ no matter which $\beta$ is chosen, so increasing $\beta$ trades approximation error for underflow rather than removing it.
