# Tropical Shortest Paths with CUDA

The goal of this project is to implement and characterize an approximation algorithm to compute the single-source shortest paths in a graph based on tropical algebra.
The algorithm is based on \[garland2008sparse\].
We will conduct a study of its approximation ratio on graphs from the [SNAP](https://snap.stanford.edu/data/index.html) collection.

## Graphs

[web-Google](https://snap.stanford.edu/data/web-Google.html) — directed graph with very different topology.

[roadNet-CA](https://snap.stanford.edu/data/roadNet-CA.html) — road-like, sparse, relatively large diameter.

[soc-LiveJournal1](https://snap.stanford.edu/data/soc-LiveJournal1.html) — scale-free/social structure, high-degree hubs.

```latex
@inproceedings{,
  title={Sparse matrix computations on manycore GPU's},
  author={Garland, Michael},
  booktitle={Proceedings of the 45th annual design automation conference},
  pages={2--6},
  year={2008}
}
```