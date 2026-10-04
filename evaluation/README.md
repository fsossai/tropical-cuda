# Evaluation

`*.yuclid.jsonl` are produced by [yuclid](https://pypi.org/project/yuclid/) through the `yuclid.json` at the repository root.

Install yuclid and run the two sweeps from the repository root:

```sh
pip install yuclid

# Kernel time of every solver on every graph, ten runs each.
yuclid run --repeat 10 --metric kernel

# Accuracy of softmin_spmv against a CPU Dijkstra reference.
yuclid run --select solver=softmin_spmv --metric rel_error_mean rel_error_p99 exact
```

Each sweep builds the project, prepares the inputs, and writes its results to `<run-id>.yuclid.jsonl` in the repository root.
The error metrics are deterministic, so one run per graph is enough.
