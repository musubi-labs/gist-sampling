# gist-sampling

Pick a small subset of a dataset that still represents all of it and contains no near-duplicates.

![The same points twice, with 14 picks each. The random sample puts 11 picks in the largest group and misses 4 of 7 groups. GIST puts 7 in the largest group and reaches every group.](https://raw.githubusercontent.com/musubi-labs/gist-sampling/main/docs/selection.svg)

Fourteen picks from the same 1,034 points. The circles have one radius throughout: half the
smallest distance between two of GIST's picks. A random sample piles up where the data is dense;
GIST's picks keep their distance and reach every group.

`gist-sampling` implements GIST (Greedy Independent Set Thresholding) from
[Fahrbach et al., arXiv:2405.18754](https://arxiv.org/abs/2405.18754). Typical uses are choosing
which rows to label, building a compact training or evaluation set, and thinning a queue that one
kind of content has flooded. It works on anything with a meaningful distance: embeddings, dense
numeric features, or a distance matrix you computed yourself.

## Installation

```bash
pip install gist-sampling
```

Python 3.11 or later. To install from source instead:

```bash
pip install git+https://github.com/musubi-labs/gist-sampling
```

## Quick start

```python
import numpy as np
from gist_sampling import GISTSelector

rng = np.random.default_rng(0)
X = rng.normal(size=(2_000, 16))  # your features or embeddings, one row per item

selector = GISTSelector(n_samples=100).fit(X)

selector.selected_indices_  # positions of the selected rows
X_small = selector.transform(X)  # the selected rows themselves
```

For a one-off, `gist_sample` does the same in one call. It accepts DataFrames, uses their numeric
columns to measure distance, and returns the selected rows with every column intact:

```python
import pandas as pd
from gist_sampling import gist_sample

df = pd.DataFrame(X[:, :3], columns=["a", "b", "c"]).assign(note="free text")
sample = gist_sample(df, n_samples=50)
```

The selection is deterministic: the same data and parameters always give the same subset.

## How it works

GIST looks for at most `n_samples` points `S` that maximise

```text
f(S) = utility(S) + lambda_diversity * diversity(S)
```

- **Utility** is how well `S` represents the data: each point in the dataset contributes its
  similarity to the most similar selected point (the facility-location function). It ranges from
  0 to the number of points in the dataset.
- **Diversity** is the smallest distance between two selected points, as a fraction of the largest
  distance in the dataset. It ranges from 0 to 1, and is 0 as soon as the selection contains two
  copies of the same point.

A greedy pass that only maximises utility picks representative points but can pick two that
nearly coincide. GIST repeats the greedy pass under a rule that no two selected points may be
closer than a distance `d`, for a range of values of `d`, and returns the best subset it finds.
That subset is guaranteed to score within a factor `1/2 - epsilon` of the best possible one.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `n_samples` | required | Maximum number of points to select. |
| `metric` | `"euclidean"` | `"euclidean"`, `"cosine"`, `"manhattan"`, `"chebyshev"` or `"precomputed"`. |
| `lambda_diversity` | `1.0` | Weight of diversity, in points' worth of utility. `0` gives plain greedy selection. |
| `epsilon` | `0.1` | Spacing of the distance thresholds tried, between 0.0001 and 1. Smaller is slower and more thorough. |
| `similarity` | `"rbf"` | Turns a distance `d` into a similarity: `"rbf"` is `exp(-gamma * d**2)`, `"inverse"` is `1 / (1 + d)`. |
| `gamma` | `None` | Scale of the `"rbf"` similarity. `None` sets it from the median distance between points. |
| `n_neighbors` | `"auto"` | Similarities kept per point: an integer, `None` for all of them, or `"auto"`. |
| `columns` | `None` | DataFrame column or columns to measure distance on. `None` uses all numeric columns. |
| `n_jobs` | `-1` | Number of threads. `-1` uses all cores. |

After `fit`, the selector exposes `selected_indices_`, `objective_value_`, `utility_`,
`diversity_` (the smallest distance between two selected points, in the units of the metric),
`threshold_`, `diameter_`, `gamma_` and `n_neighbors_`.

### Choosing `lambda_diversity`

`lambda_diversity` is the amount of utility you would give up to go from a selection containing
duplicates to one that is as spread out as the data allows. Utility counts points, so with the
default of `1.0` GIST separates selected points as long as that costs less than one point's worth
of representation in total. On data of any real size this breaks ties in favour of spread without
changing which regions are covered. To push selected points further apart, raise it in proportion
to the size of the dataset: 1% of the number of points is a mild setting and 10% a strong one.
Compare `diversity_` and `utility_` between settings to see what a change bought and cost.

### Large datasets

Keeping every pairwise similarity takes memory quadratic in the number of points. Passing
`n_neighbors=k` keeps only each point's `k` nearest neighbours, which takes memory linear in the
number of points and changes the selection very little. The default `"auto"` keeps all pairs up
to 5,000 points and 100 neighbours beyond that. `n_neighbors_` reports what was used.

Approximate wall-clock time of `fit` with default settings and `metric="cosine"`, selecting 1% of
the points, on a 16-core laptop that was also running other work
(`experiments/speed_benchmark.py`):

| Points | 64 features | 384 features |
| ---: | ---: | ---: |
| 1,000 | 0.01 s | 0.03 s |
| 5,000 | 0.1 s | 0.7 s |
| 20,000 | 0.7 s | 2.3 s |
| 100,000 | 10 s | 31 s |

Beyond 5,000 points almost all of the time goes to finding nearest neighbours. Float32 input is
processed in float32, which halves memory use.

## Recipes

### Text, images and other embedded data

Embed the items, then select with cosine distance. `transform` subsets anything that lines up row
for row with the data you fitted, so you can get the original items back directly:

```python
embeddings = rng.normal(size=(2_000, 64)).astype(np.float32)  # from your encoder
texts = [f"post {i}" for i in range(2_000)]

selector = GISTSelector(n_samples=100, metric="cosine").fit(embeddings)
texts_to_label = selector.transform(texts)
```

### Keeping rare classes

GIST does not see labels. If you have them and some classes are rare, select within each class:

```python
labels = rng.integers(0, 3, size=2_000)

selected = np.concatenate(
    [
        np.flatnonzero(labels == c)[
            GISTSelector(n_samples=30, metric="cosine")
            .fit(embeddings[labels == c])
            .selected_indices_
        ]
        for c in np.unique(labels)
    ]
)
```

### Your own distance

Pass a square matrix of pairwise distances and `metric="precomputed"`:

```python
from scipy.spatial.distance import cdist

distances = cdist(X[:500], X[:500], metric="mahalanobis")
selector = GISTSelector(n_samples=20, metric="precomputed").fit(distances)
```

## Questions

**Why did I get fewer than `n_samples` points?** `n_samples` is an upper bound. GIST returns
fewer points when the extra ones would add less utility than they cost in diversity, which
happens when the data has fewer distinct regions than the budget, or many exact duplicates. In
that case the points left out are redundant with points already selected. Set
`lambda_diversity=0` to always fill the budget: that is plain greedy selection, which never
stops early.

**Why is the first call slow?** The numerical kernels are compiled on first use, which takes a
few seconds. The compiled code is cached on disk, so this happens once per installation.

**When does GIST not help?** When distance does not reflect similarity, as with sparse one-hot
encoded tabular data, or when the classes you care about do not form regions in feature space.
Compare against random and stratified sampling on your own task;
[`experiments/`](https://github.com/musubi-labs/gist-sampling/tree/main/experiments) has scripts that do so.

**What is the input allowed to contain?** Numbers only, without NaN or infinity. Cosine distance
additionally rejects rows that are entirely zero.

## Upgrading from 0.1

Version 0.2 is a rewrite and changes some parameters and defaults. See the
[changelog](https://github.com/musubi-labs/gist-sampling/blob/main/CHANGELOG.md) for what changed and how to update your code.

Defaults may still change before version 1.0. Every such change will be listed in the changelog.

## Development

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

## Reference

Matthew Fahrbach, Srikumar Ramalingam, Morteza Zadimoghaddam, Sara Ahmadian, Gui Citovsky and
Giulia DeSalvo. *GIST: Greedy Independent Set Thresholding for Max-Min Diversification with
Submodular Utility.* NeurIPS 2025. [arXiv:2405.18754](https://arxiv.org/abs/2405.18754).

This is an independent implementation of the published method, released under the MIT license.
