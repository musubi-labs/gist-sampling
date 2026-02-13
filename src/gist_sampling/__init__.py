"""GIST-Sampling: Diversity-aware DataFrame downsampling using GIST algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gist_sampling._version import __version__
from gist_sampling.distance.metrics import MetricType
from gist_sampling.selectors.gist_selector import GISTSelector

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "__version__",
    "GISTSelector",
    "gist_sample",
]


def gist_sample(
    X: pd.DataFrame | np.ndarray,
    n_samples: int,
    *,
    metric: MetricType = "euclidean",
    epsilon: float = 0.1,
    lambda_diversity: float = 1.0,
    random_state: int | None = None,
    n_jobs: int = -1,
    **kwargs,
) -> pd.DataFrame | np.ndarray:
    """
    One-liner convenience function for GIST sampling.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Input data to sample from.
    n_samples : int
        Number of samples to select.
    metric : MetricType, default="euclidean"
        Distance metric. One of "euclidean", "cosine", "manhattan", "chebyshev".
    epsilon : float, default=0.1
        Approximation parameter controlling threshold granularity.
    lambda_diversity : float, default=1.0
        Diversity weight (λ) in f(S) = g(S) + λ·div(S).
    random_state : int | None, default=None
        Seed for randomized diameter estimation (approximate mode).
    n_jobs : int, default=-1
        Max threads for internal parallelism (k-NN + BLAS/OpenMP).
    **kwargs
        Additional arguments passed to GISTSelector.

    Returns
    -------
    pd.DataFrame or np.ndarray
        Sampled data with the same type as input.

    Examples
    --------
    >>> import numpy as np
    >>> from gist_sampling import gist_sample
    >>> X = np.random.randn(100, 2)
    >>> X_sampled = gist_sample(X, n_samples=10)
    >>> len(X_sampled)
    10
    """
    return GISTSelector(
        n_samples,
        metric=metric,
        epsilon=epsilon,
        lambda_diversity=lambda_diversity,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs,
    ).fit_transform(X)
