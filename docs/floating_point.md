# Floating-Point Considerations

The smooth min never overestimates the true minimum. With $k$ terms, it underestimates it by at most $\log(k)/\beta$, so a larger $\beta$ is more accurate.

A larger $\beta$ also makes $e^{-\beta x}$ smaller. In single precision it loses precision once $\beta x > 87.3$ and becomes zero once $\beta x > 103.3$, and a zero decodes as $\infty$, so the vertex looks unreachable.
If distances are encoded once and decoded only at the end, every distance must fit in this range at the same time, and far vertices are lost as soon as $\beta$ grows.

`smoothmin_spmv` avoids this by decoding after every step and re-encoding relative to a shift $c$, the smallest distance still waiting to be propagated.
A vertex at distance $d$ is encoded as $e^{-\beta (d - c)}$, and the shift cancels when decoding, so it never changes the result.
Only vertices within a window above $c$ are encoded; the others count as zero for that step and wait for a later one.
The window is as wide as the range allows, $87.3/\beta - w_{\max}$, so values never underflow no matter how far the vertices are from the source.
A larger $\beta$ now only means a narrower window and more steps.
