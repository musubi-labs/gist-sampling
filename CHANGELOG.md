# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-11-10

A rewrite of the package. The public interface is now `GISTSelector`, `gist_sample` and
`__version__`; everything else is private. Selections are deterministic and, with default
settings, do not depend on the units of the data.

### Upgrading from 0.1

| 0.1 | 0.2 |
| --- | --- |
| `mode="exact"` | `n_neighbors=None` |
| `mode="approximate", k_neighbors=k` | `n_neighbors=k` |
| `mode="auto", approximate_threshold=t` | `n_neighbors="auto"` (the switch is fixed at 5,000 points) |
| `GISTSelector(k, distance_matrix=D).fit(X)` | `GISTSelector(k, metric="precomputed").fit(D)` |
| `random_state=seed` | remove it; selection is deterministic |
| `utility="facility_location"` | remove it |
| `selector.mode_used_` | `selector.n_neighbors_` (`None` means all pairs) |
| `GISTSelector(k, "facility_location", "cosine")` | `GISTSelector(k, metric="cosine")`: all parameters after `n_samples` are keyword-only |
| `from gist_sampling.core import gist` and other submodules | no longer public; use `GISTSelector` |

Results will differ from 0.1 for three reasons, each described under "Changed" below: the default
`gamma`, the scaling of `lambda_diversity`, and the removal of early stopping.

### Changed

- **Default `gamma` no longer depends on the units of the data.** It is now `1 / m**2`, where `m`
  is the median distance between points, instead of `1 / m`. Two points at the median distance
  have similarity `1/e` whatever the scale of the features. For cosine distance on embeddings the
  two are close; for unnormalised Euclidean data the old default could make every similarity
  vanish. Pass `gamma` explicitly to reproduce 0.1.
- **`lambda_diversity` weighs diversity as a fraction of the diameter** of the data rather than
  as a raw distance, so its effect no longer changes when the features are rescaled. To reproduce
  the 0.1 objective, multiply the old value by `selector.diameter_`.
- **GIST always tries every distance threshold.** 0.1 stopped after three thresholds without
  improvement, which could miss better selections that the algorithm is meant to find, and so
  could return a full budget where the objective prefers fewer, better-separated points. If you
  relied on always receiving `n_samples` points, set `lambda_diversity=0`.
- Parameters are validated in `fit` rather than in the constructor, following scikit-learn, and
  `GISTSelector` now supports `get_params`, `set_params` and `clone`.
- `transform` accepts anything aligned row for row with the fitted data, including lists and
  Series, and raises if the number of rows differs.
- Float32 input is processed in float32 rather than converted to float64.
- pandas is no longer a dependency. DataFrames are still accepted.
- Python 3.11 or later is required.

### Added

- `metric="precomputed"`, which also works together with `n_neighbors`.
- Fitted attributes `utility_`, `threshold_`, `diameter_`, `gamma_`, `n_neighbors_` and
  `n_samples_in_`.
- Type hints are shipped (`py.typed`).
- `experiments/speed_benchmark.py`.

### Fixed

- Cosine distance with all-zero rows gave undefined results; it now raises an error.
- Data whose distances overflow or underflow floating point raises an error instead of returning
  an arbitrary selection.
- Fitting from several threads at once no longer crashes the process; concurrent fits take turns.
- The diameter used when only neighbour similarities are kept is estimated by farthest-first
  traversal instead of from random rows, which is both closer to the true diameter and
  deterministic.

### Performance

The whole selection now runs in compiled code, the distance thresholds are tried in parallel, and
no distance matrix is stored. Measured against 0.1 on the same machine and the same objective
(16 cores, clustered synthetic data, compilation excluded):

| Points | Features | `n_samples` | Similarities | 0.1 | 0.2 | Speed-up |
| ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 2,000 | 32 | 100 | all pairs | 0.24 s | 0.013 s | 18x |
| 5,000 | 64 | 200 | all pairs | 2.9 s | 0.09 s | 32x |
| 5,000 | 384 | 250 | all pairs | 15.0 s | 0.55 s | 27x |
| 5,000 | 64 | 200 | 100 nearest | 1.8 s | 0.22 s | 8x |
| 20,000 | 64 | 500 | 100 nearest | 6.6 s | 0.61 s | 11x |
| 20,000 | 384 | 500 | 100 nearest | 12.2 s | 1.33 s | 9x |

In every case the objective reached by 0.2 was equal to or higher than that of 0.1. Keeping all
pairwise similarities now takes 4 bytes per pair instead of 16.

## 0.1.0 - 2026-02-13

Initial experimental version. It was never tagged or published to PyPI.

[0.2.0]: https://github.com/musubi-labs/gist-sampling/releases/tag/v0.2.0
