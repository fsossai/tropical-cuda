# Selecting the SSSP Source

Every benchmark graph is solved from one fixed source vertex, set as the `source` attribute of the graph in `yuclid.json`.
This note explains how those sources were chosen.

## Goal

The source decides how much work an SSSP solve does: how many vertices it reaches, and how many hops separate it from the farthest of them.
A source that reaches little, or reaches everything in a few hops, makes every solver finish quickly and says little about its performance.
The sources were therefore chosen to make the solve as long as possible, using the cuGraph solver as the yardstick.

## Method

1. **Find the largest component.** For a directed graph, this is the largest strongly connected component: every vertex in it reaches every other, so a source there reaches the bulk of the graph. For an undirected graph, it is the largest connected component.
2. **Pick candidates from it:** ten vertices drawn uniformly at random, with a fixed seed.
3. **Run cuGraph from each candidate** with `sssp --solver cugraph --runs 11`, and take the median kernel time of runs 2 to 11, discarding the first run, which includes warm-up.
4. **Keep the slowest candidate.** The timer has a resolution of 1 ms, so ties are broken by the larger depth, then the larger reach.

All graphs are unweighted, so a breadth-first search gives the exact hop distances.

## Selected Sources

Reach is the number of vertices with a finite distance, and depth is the largest hop distance from the source.

| Graph | Largest component (share of vertices) | Source | Reach | Depth | cuGraph kernel |
|---|---|---|---|---|---|
| amazon0601 | 395,234 (98.0%) | 57,330 | 402,793 | 35 | 36.0 ms |
| email_euall | 34,203 (12.9%) | 75,795 | 52,103 | 8 | 6.0 ms |
| roadnet_ca | 1,957,027 (99.3%) | 982,529 | 1,957,027 | 725 | 429.5 ms |
| roadnet_pa | 1,087,562 (99.7%) | 319,360 | 1,087,562 | 628 | 370.0 ms |
| roadnet_tx | 1,351,137 (97.0%) | 900,309 | 1,351,137 | 896 | 515.0 ms |
| livejournal | 3,828,682 (79.0%) | 3,487,746 | 4,400,347 | 15 | 67.0 ms |
| soc_pokec_relationships | 1,304,537 (79.9%) | 1,236,350 | 1,504,295 | 11 | 39.5 ms |
| orkut | n/a | 1 (skipped) | n/a | n/a | n/a |
| web_berkstan | 334,857 (48.9%) | 598,568 | 459,831 | 576 | 338.5 ms |
| web_google | 434,818 (47.4%) | 461,148 | 600,493 | 38 | 32.0 ms |
| web_notredame | 53,968 (16.6%) | 251,608 | 325,729 | 58 | 40.0 ms |
| web_stanford | 150,532 (53.4%) | 7,039 | 218,048 | 144 | 85.0 ms |
| wiki_talk | 111,881 (4.7%) | 2,161,385 | 2,354,316 | 7 | 41.5 ms |

The orkut file stores each undirected edge in one direction only, from the lower to the higher vertex ID, so it has no cycles and every strongly connected component is a single vertex.
The method does not apply, and its source is kept at the default, vertex 1, which is the only vertex that reaches the whole graph, until the conversion is fixed.
