"""Distance computation utilities."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.spatial.distance import cdist

MetricType = Literal["euclidean", "cosine", "manhattan", "chebyshev"]

SUPPORTED_METRICS: list[MetricType] = ["euclidean", "cosine", "manhattan", "chebyshev"]

# Map user-friendly names to scipy cdist metric names
SCIPY_METRIC_MAP: dict[str, str] = {
    "euclidean": "euclidean",
    "cosine": "cosine",
    "manhattan": "cityblock",  # scipy uses "cityblock" for L1 distance
    "chebyshev": "chebyshev",
}

# Map user-friendly names to sklearn metric names
SKLEARN_METRIC_MAP: dict[str, str] = {
    "euclidean": "euclidean",
    "cosine": "cosine",
    "manhattan": "manhattan",
    "chebyshev": "chebyshev",
}


def compute_distance_matrix(
    X: np.ndarray,
    metric: MetricType = "euclidean",
) -> np.ndarray:
    """
    Compute pairwise distance matrix.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n, d).
    metric : MetricType
        Distance metric. One of 'euclidean', 'cosine', 'manhattan', 'chebyshev'.

    Returns
    -------
    np.ndarray
        Distance matrix of shape (n, n).
    """
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"Unsupported metric: {metric}. Supported: {SUPPORTED_METRICS}")

    scipy_metric = SCIPY_METRIC_MAP[metric]
    return cdist(X, X, metric=scipy_metric)


def compute_similarity_matrix(
    distance_matrix: np.ndarray,
    method: Literal["rbf", "inverse"] = "rbf",
    gamma: float | None = None,
) -> np.ndarray:
    """
    Convert distance matrix to similarity matrix.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Pairwise distance matrix of shape (n, n).
    method : Literal["rbf", "inverse"]
        Similarity method. One of 'rbf' (Gaussian RBF), 'inverse'.
    gamma : float, optional
        RBF kernel parameter. If None, uses 1 / median(distances).

    Returns
    -------
    np.ndarray
        Similarity matrix of shape (n, n).
    """
    if method == "rbf":
        if gamma is None:
            # Use median heuristic
            non_zero_dists = distance_matrix[distance_matrix > 0]
            if len(non_zero_dists) > 0:
                gamma = 1.0 / np.median(non_zero_dists)
            else:
                gamma = 1.0
        return np.exp(-gamma * distance_matrix**2)

    elif method == "inverse":
        # Avoid division by zero
        return 1.0 / (1.0 + distance_matrix)

    else:
        raise ValueError(f"Unsupported similarity method: {method}")
