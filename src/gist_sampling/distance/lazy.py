"""Lazy distance matrix for memory-efficient computation."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from scipy.spatial.distance import cdist

from gist_sampling.distance.metrics import SCIPY_METRIC_MAP, MetricType


class LazyDistanceMatrix:
    """
    Lazy distance matrix that computes rows on-demand.

    Instead of precomputing the full O(n²) distance matrix, this class
    computes distance rows only when accessed. It uses an LRU cache to
    avoid recomputing frequently accessed rows.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n, d).
    metric : str
        Distance metric. One of 'euclidean', 'cosine', 'manhattan', 'chebyshev'.
    cache_size : int, default=1024
        Maximum number of rows to cache in memory.
    use_numba : bool, default=True
        Whether to use Numba JIT compilation if available.

    Attributes
    ----------
    shape : tuple[int, int]
        Shape of the distance matrix (n, n).
    n : int
        Number of data points.
    """

    def __init__(
        self,
        X: np.ndarray,
        metric: MetricType = "euclidean",
        cache_size: int = 1024,
        use_numba: bool = True,
    ) -> None:
        self._X = np.ascontiguousarray(X, dtype=np.float64)
        self._metric = metric
        self._scipy_metric = SCIPY_METRIC_MAP.get(metric, metric)
        self._n = X.shape[0]
        self._cache_size = cache_size

        # Try to load Numba kernels
        self._numba_kernel = None
        if use_numba:
            try:
                from gist_sampling.distance._numba_kernels import get_distance_kernel

                self._numba_kernel = get_distance_kernel(metric)
            except ImportError:
                pass

        # OrderedDict-based LRU cache storing numpy arrays directly.
        # This avoids the costly tuple(row) / np.array(tuple) round-trip
        # that functools.lru_cache would require for hashability.
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the distance matrix."""
        return (self._n, self._n)

    @property
    def n(self) -> int:
        """Number of data points."""
        return self._n

    def _compute_row(self, idx: int) -> np.ndarray:
        """Compute distance row as a numpy array."""
        if self._numba_kernel is not None:
            row = self._numba_kernel(self._X, idx)
        else:
            row = cdist(self._X[idx : idx + 1], self._X, metric=self._scipy_metric)[0]
        row.flags.writeable = False
        return row

    def get_row(self, idx: int) -> np.ndarray:
        """
        Get a single row of the distance matrix.

        Parameters
        ----------
        idx : int
            Row index.

        Returns
        -------
        np.ndarray
            Distance row of shape (n,).
        """
        if idx < 0 or idx >= self._n:
            raise IndexError(f"Index {idx} out of bounds for matrix with {self._n} rows")

        if idx in self._cache:
            self._cache.move_to_end(idx)
            self._cache_hits += 1
            return self._cache[idx]

        row = self._compute_row(idx)
        self._cache[idx] = row
        self._cache_misses += 1

        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

        return row

    def __getitem__(self, key):
        """
        Get item(s) from the distance matrix.

        Supports:
        - lazy_dist[i] -> row i as ndarray
        - lazy_dist[i, j] -> single distance value
        - lazy_dist[i, :] -> row i as ndarray
        """
        if isinstance(key, (int, np.integer)):
            return self.get_row(int(key))
        elif isinstance(key, tuple) and len(key) == 2:
            i, j = key
            if isinstance(j, slice) and j == slice(None):
                return self.get_row(int(i))
            elif isinstance(i, (int, np.integer)) and isinstance(j, (int, np.integer)):
                return self.get_row(int(i))[int(j)]
            elif isinstance(i, (int, np.integer)) and isinstance(j, (list, np.ndarray)):
                return self.get_row(int(i))[np.asarray(j)]
        raise TypeError(f"Invalid index type: {type(key)}")

    def clear_cache(self) -> None:
        """Clear the row cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def cache_info(self):
        """Get cache statistics.

        Returns a SimpleNamespace mimicking functools.lru_cache CacheInfo
        with hits, misses, maxsize, and currsize attributes.
        """
        from types import SimpleNamespace

        return SimpleNamespace(
            hits=self._cache_hits,
            misses=self._cache_misses,
            maxsize=self._cache_size,
            currsize=len(self._cache),
        )

    def to_dense(self) -> np.ndarray:
        """
        Convert to dense distance matrix.

        Warning: This creates an O(n²) matrix. Only use for small datasets.

        Returns
        -------
        np.ndarray
            Dense distance matrix of shape (n, n).
        """
        return np.vstack([self.get_row(i) for i in range(self._n)])


def compute_lazy_distance_matrix(
    X: np.ndarray,
    metric: MetricType = "euclidean",
    cache_size: int = 1024,
    use_numba: bool = True,
) -> LazyDistanceMatrix:
    """
    Create a lazy distance matrix.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n, d).
    metric : MetricType
        Distance metric.
    cache_size : int, default=1024
        Maximum number of rows to cache.
    use_numba : bool, default=True
        Whether to use Numba JIT compilation if available.

    Returns
    -------
    LazyDistanceMatrix
        Lazy distance matrix instance.
    """
    return LazyDistanceMatrix(X, metric=metric, cache_size=cache_size, use_numba=use_numba)
