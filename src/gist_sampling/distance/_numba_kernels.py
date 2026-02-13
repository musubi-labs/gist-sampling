"""Numba JIT-compiled distance computation kernels."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# Flag to track if Numba is available
_NUMBA_AVAILABLE = False

try:
    from numba import jit, prange

    _NUMBA_AVAILABLE = True
except ImportError:

    def prange(*args):  # noqa: D103
        """Dummy prange when Numba is not available."""
        return range(*args)

    def jit(*args, **kwargs):
        """Dummy decorator when Numba is not available."""

        def decorator(func):
            return func

        return decorator


def is_numba_available() -> bool:
    """Check if Numba is available."""
    return _NUMBA_AVAILABLE


if _NUMBA_AVAILABLE:

    @jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def _euclidean_distance_row(X: np.ndarray, idx: int) -> np.ndarray:
        """Compute Euclidean distances from point idx to all points."""
        n, d = X.shape
        result = np.empty(n, dtype=np.float64)
        point = X[idx]

        for i in prange(n):
            dist_sq = 0.0
            for j in range(d):
                diff = point[j] - X[i, j]
                dist_sq += diff * diff
            result[i] = np.sqrt(dist_sq)

        return result

    @jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def _manhattan_distance_row(X: np.ndarray, idx: int) -> np.ndarray:
        """Compute Manhattan distances from point idx to all points."""
        n, d = X.shape
        result = np.empty(n, dtype=np.float64)
        point = X[idx]

        for i in prange(n):
            dist = 0.0
            for j in range(d):
                dist += abs(point[j] - X[i, j])
            result[i] = dist

        return result

    @jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def _chebyshev_distance_row(X: np.ndarray, idx: int) -> np.ndarray:
        """Compute Chebyshev distances from point idx to all points."""
        n, d = X.shape
        result = np.empty(n, dtype=np.float64)
        point = X[idx]

        for i in prange(n):
            max_diff = 0.0
            for j in range(d):
                diff = abs(point[j] - X[i, j])
                if diff > max_diff:
                    max_diff = diff
            result[i] = max_diff

        return result

    @jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def _cosine_distance_row(X: np.ndarray, idx: int) -> np.ndarray:
        """Compute cosine distances from point idx to all points."""
        n, d = X.shape
        result = np.empty(n, dtype=np.float64)
        point = X[idx]

        # Compute norm of reference point
        norm_ref = 0.0
        for j in range(d):
            norm_ref += point[j] * point[j]
        norm_ref = np.sqrt(norm_ref)

        for i in prange(n):
            dot = 0.0
            norm_other = 0.0
            for j in range(d):
                dot += point[j] * X[i, j]
                norm_other += X[i, j] * X[i, j]
            norm_other = np.sqrt(norm_other)

            if norm_ref > 0 and norm_other > 0:
                result[i] = 1.0 - dot / (norm_ref * norm_other)
            else:
                result[i] = 0.0 if i == idx else 1.0

        return result

else:
    # Fallback implementations when Numba is not available.
    # Uses scipy cdist which handles memory internally, avoiding the
    # huge (n, d) temporaries that broadcasting (X - X[idx]) would create.
    from scipy.spatial.distance import cdist as _cdist

    def _euclidean_distance_row(X: np.ndarray, idx: int) -> np.ndarray:
        """Compute Euclidean distances from point idx to all points (scipy)."""
        return _cdist(X[idx : idx + 1], X, metric="euclidean")[0]

    def _manhattan_distance_row(X: np.ndarray, idx: int) -> np.ndarray:
        """Compute Manhattan distances from point idx to all points (scipy)."""
        return _cdist(X[idx : idx + 1], X, metric="cityblock")[0]

    def _chebyshev_distance_row(X: np.ndarray, idx: int) -> np.ndarray:
        """Compute Chebyshev distances from point idx to all points (scipy)."""
        return _cdist(X[idx : idx + 1], X, metric="chebyshev")[0]

    def _cosine_distance_row(X: np.ndarray, idx: int) -> np.ndarray:
        """Compute cosine distances from point idx to all points (scipy)."""
        return _cdist(X[idx : idx + 1], X, metric="cosine")[0]


# Kernel registry
_ROW_KERNELS: dict[str, Callable] = {
    "euclidean": _euclidean_distance_row,
    "cosine": _cosine_distance_row,
    "manhattan": _manhattan_distance_row,
    "chebyshev": _chebyshev_distance_row,
}


def get_distance_kernel(metric: str) -> Callable[[np.ndarray, int], np.ndarray]:
    """
    Get the distance row kernel for a metric.

    Parameters
    ----------
    metric : str
        Distance metric name.

    Returns
    -------
    Callable
        Function that computes distances from one point to all others.

    Raises
    ------
    ValueError
        If the metric is not supported.
    """
    if metric not in _ROW_KERNELS:
        raise ValueError(f"Unsupported metric: {metric}. Supported: {list(_ROW_KERNELS.keys())}")
    return _ROW_KERNELS[metric]
