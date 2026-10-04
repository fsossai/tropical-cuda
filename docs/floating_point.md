# Floating-Point Considerations

The soft-min never overestimates the true minimum. With $k$ terms, it underestimates it by at most $\log(k)/\beta$, so a larger $\beta$ is more accurate.

A larger $\beta$ also makes $e^{-\beta x}$ smaller. In single precision it loses precision once $\beta x > 87.3$ and becomes zero once $\beta x > 103.3$, and a zero decodes as $\infty$, so the vertex looks unreachable.
If distances are encoded once and decoded only at the end, every distance must fit in this range at the same time, and far vertices are lost as soon as $\beta$ grows.

`softmin_spmv` avoids this by decoding after every step and re-encoding relative to a shift $c$, the smallest distance still waiting to be propagated.
A vertex at distance $d$ is encoded as $e^{-\beta (d - c)}$, and the shift cancels when decoding, so it never changes the result.
Only vertices within a window above $c$ are encoded; the others count as zero for that step and wait for a later one.
The window is as wide as the range allows, $87.3/\beta - w_{\max}$, so values never underflow no matter how far the vertices are from the source.
A larger $\beta$ now only means a narrower window and more steps.

Results from different steps are combined with an exact minimum, so the error of the soft-min does not pile up across steps.
It can still pile up around a cycle: if the cycle's edges are lighter than $\log(k)/\beta$, its distances keep decreasing and the solver runs until the iteration cap.
With unit weights this happens when $\beta$ is below about $\log$ of the largest in-degree. On web-Google, $\beta = 2$ never converges, while $\beta = 8$ finds every reachable vertex with a maximum error of 1.02.

## Known Problems

The default $\beta = 2$ is below this threshold on most graphs. In the benchmark sweep, `softmin_spmv` drifted on 7 of the 12 graphs: 3 runs reached the iteration cap after 20 to 46 seconds, and 4 were stopped after a minute, where the exact solvers take milliseconds. On web-NotreDame it also missed 412 reachable vertices.
The sweep cannot pick a better $\beta$ yet, because its trial command passes no `--beta` and `sssp` rejects that option for the other solvers.
The sweep also records only timings, so the accuracy of `softmin_spmv` has been checked by hand on a few graphs but is not measured systematically.
