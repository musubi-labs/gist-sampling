# Experiments

Install experiment dependencies first:

```bash
uv sync --extra extra
# or: pip install -e ".[extra]"
```

## T&S dataset benchmarks

```bash
uv run experiments/benchmark_experiments.py
```

Tip: high `--seed-jobs` can run faster, but memory usage scales with worker count.

## Synthetic cluster demo

```bash
uv run experiments/synthetic_demo.py
```

## Tabular limitation demo

```bash
uv run experiments/tabular_demo.py --seed 42 --max-rows 5000
```

