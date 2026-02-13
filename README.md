# GIST-Sampling [Experimental]

Diversity-aware DataFrame downsampling using the GIST (Greedy Independent Set Thresholding) algorithm.

## Installation

This project isn't published on PyPI. Install from a local checkout:

```bash
pip install -e .
```

If you use uv:

```bash
uv pip install -e .
```

## Quick start

```python
import pandas as pd
from gist_sampling import GISTSelector

df = pd.read_csv("data.csv")

selector = GISTSelector(
    n_samples=1000,
    epsilon=0.1,
    lambda_diversity=1.0,
)
df_sampled = selector.fit_transform(df)

print(f"Selected: {len(selector.selected_indices_)}")
print(f"Min diversity: {selector.diversity_:.4f}")
```

For one-line usage:

```python
from gist_sampling import gist_sample

sampled = gist_sample(data, n_samples=100)
```

## Data requirements

- Inputs must be a pandas DataFrame or NumPy array.
- NumPy arrays must be numeric and 1D or 2D.
- DataFrames use numeric columns by default; non-numeric columns are ignored.
- Missing values (NaN/Inf) are not allowed in the features used for selection.
  - If `columns` is provided, only those columns are validated for NaN/Inf.
  - If `columns` is not provided, all numeric columns are validated.

## API

### GISTSelector

Key parameters:
- `n_samples` (int): number of samples to select
- `metric` (str): "euclidean" | "cosine" | "manhattan" | "chebyshev"
- `columns` (list[str] | None): DataFrame columns used for distance computation
- `mode` (str): "auto" | "exact" | "approximate"
- `approximate_threshold` (int): threshold used by `mode="auto"` to switch to approximate mode
- `k_neighbors` (int): number of nearest neighbors used in approximate mode
- `distance_matrix` (np.ndarray | None): precomputed distance matrix (exact mode only)
- `similarity` (str): "rbf" | "inverse"
- `gamma` (float | None): RBF kernel parameter (optional)
- `lambda_diversity` (float): diversity weight (λ) in f(S) = g(S) + λ·div(S)
- `random_state` (int | None): seed for randomized diameter estimation
- `n_jobs` (int): max threads for internal parallelism (approximate + BLAS/OpenMP)

#### Mode behavior

- `mode="exact"`: full pairwise distances, O(n^2) memory/time behavior.
- `mode="approximate"`: sparse similarity + lazy distances, usually better scaling for large datasets.
- `mode="auto"`: exact below `approximate_threshold`, approximate above it.
- If `distance_matrix` is provided, use `mode="exact"`.

#### Reproducibility

- Set `random_state` for repeatable approximate-mode behavior.
- Exact mode is deterministic for fixed inputs and parameters.

### gist_sample

```python
from gist_sampling import gist_sample

sampled = gist_sample(data, n_samples=100)
```

`gist_sample` forwards most keyword arguments directly to `GISTSelector`.

## Performance notes

- Exact mode computes full pairwise distances and is O(n²) in time and memory.
- For large datasets, prefer approximate mode (`mode="approximate"` or `mode="auto"` with a sensible threshold).
- Use exact mode when you need full-fidelity objective evaluation from dense pairwise distances.
- GIST is most useful when distance is meaningful, e.g., dense numeric features or vector embeddings.

## Examples

Exact mode with a precomputed distance matrix:

```python
from scipy.spatial.distance import cdist
from gist_sampling import GISTSelector

dist = cdist(X, X, metric="euclidean")
selector = GISTSelector(
    n_samples=200,
    mode="exact",
    distance_matrix=dist,
)
X_sampled = selector.fit_transform(X)
```

Approximate mode for larger datasets:

```python
selector = GISTSelector(
    n_samples=200,
    mode="approximate",
    k_neighbors=200,
    random_state=42,
)
X_sampled = selector.fit_transform(X)
```

## Reference

Based on the GIST algorithm from [arXiv:2405.18754](https://arxiv.org/abs/2405.18754).
This is an independent experimental implementation of the published method.
