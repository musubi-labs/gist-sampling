# Experiments

Scripts that compare GIST with random and stratified sampling, and one that measures speed.
Install their dependencies first:

```bash
uv sync --group experiments
```

The results quoted in `blog_draft.md` were produced with version 0.1 of the package. Version 0.2
changed some defaults (see the [changelog](../CHANGELOG.md)), so rerunning the scripts will not
reproduce those numbers exactly.

## Trust & Safety dataset benchmarks

Trains a classifier on subsets of public content-moderation datasets chosen by each method.
Downloads the datasets and an embedding model on first use.

```bash
uv run experiments/benchmark_experiments.py
```

High `--seed-jobs` runs faster, but memory use grows with the number of workers.

## Synthetic cluster demo

Five topic clusters, one of which holds 60% of the points.

```bash
uv run experiments/synthetic_demo.py
```

## Tabular limitation demo

One-hot encoded tabular data, where distances carry little signal and GIST does not help.

```bash
uv run experiments/tabular_demo.py --seed 42 --max-rows 5000
```

## Speed benchmark

Times `fit` on synthetic data of increasing size. Needs no extra dependencies.

```bash
uv run experiments/speed_benchmark.py
```

## README figure

Redraws `docs/selection.svg`, the figure at the top of the main README. Needs no extra
dependencies.

```bash
uv run experiments/readme_figure.py
```
