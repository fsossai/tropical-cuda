# Tropical Shortest Paths with CUDA

**Work in progress.**
This project is an experiment in the performance and accuracy of GPU single-source shortest-path (SSSP) algorithms based on tropical algebra.
It will compare several variants with different trade-offs, starting with a correct GPU algorithm before introducing approximations.

## TODO

- [ ] GPU: Exact tropical Bellman-Ford with hand-crafted SpMV.
- [ ] GPU: Approximate tropical Bellman-Ford using cuSPARSE.
- [ ] CPU: Multithreaded implementation based on GAPBS \[Beamer et al.\].
- [ ] GPU: Exact implementation using cuGraph.

## Graphs

[web-Google](https://snap.stanford.edu/data/web-Google.html) — web graph with varied topology.

[roadNet-CA](https://snap.stanford.edu/data/roadNet-CA.html) — road-like, sparse, relatively large diameter.

[soc-LiveJournal1](https://snap.stanford.edu/data/soc-LiveJournal1.html) — scale-free/social structure, high-degree hubs.

These [SNAP](https://snap.stanford.edu/data/index.html) datasets are candidates for the experimental study.

## Background

The tropical sparse-matrix approach is motivated by the following work:

```latex
@inproceedings{,
  title={Sparse matrix computations on manycore GPU's},
  author={Garland, Michael},
  booktitle={Proceedings of the 45th annual design automation conference},
  pages={2--6},
  year={2008}
}
```

GAPBS:
```latex
@article{beamer2015gap,
  title={The GAP benchmark suite},
  author={Beamer, Scott and Asanovi{\'c}, Krste and Patterson, David},
  journal={arXiv preprint arXiv:1508.03619},
  year={2015}
}
```
