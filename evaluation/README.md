# Evaluation

`*.yuclid.jsonl` are produced by [yuclid](https://pypi.org/project/yuclid/) through the `yuclid.json` at the repository root.

Install yuclid and run the sweep from the repository root:

```sh
pip install yuclid
yuclid run --select solver=tropical_exact,cugraph,cusparse --repeat 10 --metric kernel
```

This builds the project, prepares the inputs, runs every graph with both solvers ten times, and writes the results to `<run-id>.yuclid.jsonl` in the repository root.
