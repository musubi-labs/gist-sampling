"""Sparse similarity computation using approximate nearest neighbors."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import sparse

from gist_sampling.distance.metrics import SCIPY_METRIC_MAP, SKLEARN_METRIC_MAP, MetricType


def compute_sparse_similarity(
    X: np.ndarray,
    k_neighbors: int = 100,
    metric: MetricType = "euclidean",
    similarity_method: Literal["rbf", "inverse"] = "rbf",
    gamma: float | None = None,
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
    n_jobs: int = -1,
    random_state: int | None = None,
) -> sparse.csr_matrix:
    """
    Compute sparse similarity matrix using k-nearest neighbors.

    This function computes similarities only for the k-nearest neighbors
    of each point, resulting in O(n × k) memory instead of O(n²).

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n, d).
    k_neighbors : int, default=100
        Number of nearest neighbors to compute.
    metric : MetricType, default="euclidean"
        Distance metric.
    similarity_method : {"rbf", "inverse"}, default="rbf"
        Method to convert distances to similarities.
    gamma : float, optional
        RBF kernel parameter. If None, uses 1 / median(knn distances).
    algorithm : {"auto", "ball_tree", "kd_tree", "brute"}, default="auto"
        Algorithm for nearest neighbor search.
    n_jobs : int, default=-1
        Number of parallel jobs for nearest neighbor search.
    random_state : int | None, default=None
        Seed for gamma estimation sampling.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse similarity matrix of shape (n, n).

    Notes
    -----
    For the RBF kernel, similarities beyond ~3σ are negligible (<0.01).
    With k=100-200, this covers the effective radius for typical datasets.
    """
    from sklearn.neighbors import NearestNeighbors

    n = X.shape[0]

    # Clamp k_neighbors to valid range
    k_neighbors = min(k_neighbors, n - 1)
    if k_neighbors <= 0:
        k_neighbors = min(100, n - 1)

    sklearn_metric = SKLEARN_METRIC_MAP.get(metric, metric)
    scipy_metric = SCIPY_METRIC_MAP.get(metric, metric)

    # Find k-nearest neighbors
    nn = NearestNeighbors(
        n_neighbors=k_neighbors + 1,  # +1 to include self
        metric=sklearn_metric,
        algorithm=algorithm,
        n_jobs=n_jobs,
    )
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    # Compute gamma if not provided
    # Use sampled pairwise distances for better estimation (not just k-NN)
    if gamma is None:
        rng = np.random.default_rng(random_state) if random_state is not None else None
        gamma = _estimate_gamma_sampled(X, metric=scipy_metric, rng=rng)

    # Convert distances to similarities
    if similarity_method == "rbf":
        similarities = np.exp(-gamma * distances**2)
    elif similarity_method == "inverse":
        similarities = 1.0 / (1.0 + distances)
    else:
        raise ValueError(f"Unsupported similarity method: {similarity_method}")

    # Build sparse matrix
    # Row indices: repeat each row index k_neighbors times
    row_indices = np.repeat(np.arange(n), k_neighbors + 1)
    col_indices = indices.ravel()
    data = similarities.ravel()

    similarity_matrix = sparse.csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(n, n),
    )

    # Make symmetric by taking element-wise max
    # This ensures similarity[i,j] = similarity[j,i]
    similarity_matrix = similarity_matrix.maximum(similarity_matrix.T)

    return similarity_matrix


def _paired_distances(X: np.ndarray, idx1: np.ndarray, idx2: np.ndarray, metric: str) -> np.ndarray:
    """Compute distances for paired indices in a single vectorized operation.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n, d).
    idx1, idx2 : np.ndarray
        Index arrays of equal length identifying the pairs.
    metric : str
        Scipy distance metric name.

    Returns
    -------
    np.ndarray
        Distance for each pair, shape (len(idx1),).
    """
    A = X[idx1]
    B = X[idx2]
    diff = A - B

    if metric == "euclidean":
        return np.sqrt(np.sum(diff * diff, axis=1))
    elif metric == "cityblock":
        return np.sum(np.abs(diff), axis=1)
    elif metric == "chebyshev":
        return np.max(np.abs(diff), axis=1)
    elif metric == "cosine":
        dot = np.sum(A * B, axis=1)
        norm_a = np.linalg.norm(A, axis=1)
        norm_b = np.linalg.norm(B, axis=1)
        denom = norm_a * norm_b
        denom = np.where(denom > 0, denom, 1.0)
        return 1.0 - dot / denom
    else:
        # Generic fallback: use cdist on stacked rows, extract diagonal
        from scipy.spatial.distance import cdist

        chunk = 256
        dists = np.empty(len(idx1))
        for start in range(0, len(idx1), chunk):
            end = min(start + chunk, len(idx1))
            dists[start:end] = np.diag(cdist(A[start:end], B[start:end], metric=metric))
        return dists


def _estimate_gamma_sampled(
    X: np.ndarray,
    metric: str = "euclidean",
    n_samples: int = 1000,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Estimate gamma by sampling random pairs from the data.

    This gives a better estimate than using only k-NN distances,
    which are biased towards smaller values.
    """
    from scipy.spatial.distance import cdist

    n = X.shape[0]

    if n <= n_samples:
        # Small enough to compute full distance matrix
        dist_matrix = cdist(X, X, metric=metric)
        non_zero = dist_matrix[dist_matrix > 0]
    else:
        # Sample random pairs and compute distances in one vectorized batch
        if rng is None:
            rng = np.random.default_rng()
        idx1 = rng.choice(n, size=n_samples, replace=True)
        idx2 = rng.choice(n, size=n_samples, replace=True)

        # Filter out self-pairs before computing distances
        mask = idx1 != idx2
        idx1 = idx1[mask]
        idx2 = idx2[mask]

        pair_dists = _paired_distances(X, idx1, idx2, metric)
        non_zero = pair_dists[pair_dists > 0]

    if len(non_zero) > 0:
        return 1.0 / np.median(non_zero)
    else:
        return 1.0
