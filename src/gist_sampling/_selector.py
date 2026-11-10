"""Public interface: :class:`GISTSelector` and :func:`gist_sample`."""

from __future__ import annotations

import math
import threading
from collections.abc import Generator, Sequence
from contextlib import contextmanager, suppress
from numbers import Integral, Real
from typing import Any, Literal

import numba
import numpy as np
from sklearn.base import BaseEstimator
from threadpoolctl import threadpool_limits

from gist_sampling import _kernels as K
from gist_sampling import _similarity
from gist_sampling._gist import SimilarityMatrix, gist

Metric = Literal["euclidean", "cosine", "manhattan", "chebyshev", "precomputed"]
Similarity = Literal["rbf", "inverse"]

# Largest input for which ``n_neighbors="auto"`` keeps all pairwise
# similarities (a float32 matrix of this size takes 100 MB).
_AUTO_EXACT_LIMIT = 5_000
_AUTO_NEIGHBORS = 100

# Numba's default threading layer aborts the process when entered from two
# Python threads at once, so concurrent fits take turns.
_KERNEL_LOCK = threading.Lock()

_SCALE_ERROR = (
    "Distances between the rows of X overflow or underflow floating point. "
    "Rescale X so that typical distances are closer to 1."
)


class GISTSelector(BaseEstimator):
    """Select a diverse, representative subset of a dataset with GIST.

    GIST (Greedy Independent Set Thresholding, `arXiv:2405.18754
    <https://arxiv.org/abs/2405.18754>`_) picks at most ``n_samples`` points
    that maximise

    .. code-block:: text

        f(S) = g(S) + lambda_diversity * div(S)

    ``g`` is the facility-location utility: the sum, over every point in the
    data, of its similarity to the most similar selected point. It rewards
    subsets that represent the whole dataset, and ranges from 0 to the number
    of points. ``div(S)`` is the smallest distance between any two selected
    points, as a fraction of the largest distance in the data. It rewards
    subsets without near duplicates, and ranges from 0 to 1. When all
    similarities are kept, the result is guaranteed to score within a factor
    ``1/2 - epsilon`` of the best possible subset.

    Parameters
    ----------
    n_samples : int
        Maximum number of points to select. Fewer are returned when a smaller,
        more spread-out subset scores higher, which happens for large
        ``lambda_diversity`` or data with many duplicates.
    metric : {"euclidean", "cosine", "manhattan", "chebyshev", "precomputed"}, \
default="euclidean"
        Distance between points. With ``"precomputed"``, ``fit`` takes a square
        matrix of pairwise distances instead of features.
    lambda_diversity : float, default=1.0
        Weight of the diversity term: how many points' worth of representation
        a perfectly spread-out selection is worth. Because the utility counts
        points, the values that visibly change the selection grow with the
        size of the data. ``0`` reduces GIST to greedy facility location.
    epsilon : float, default=0.1
        Spacing of the distance thresholds that GIST tries, between 0.0001
        and 1. Smaller values try more thresholds: slower, with a tighter
        guarantee.
    similarity : {"rbf", "inverse"}, default="rbf"
        How a distance ``d`` becomes a similarity: ``exp(-gamma * d**2)`` or
        ``1 / (1 + d)``.
    gamma : float, optional
        Scale of the ``"rbf"`` similarity. By default ``1 / m**2`` where ``m``
        is the median distance between points, so that two points at the
        median distance have similarity ``1/e`` whatever the units of the data.
    n_neighbors : int, None or "auto", default="auto"
        How many similarities each point keeps. ``None`` keeps all of them,
        which is exact and takes memory quadratic in the number of points. An
        integer keeps, for each point, only its ``n_neighbors`` nearest
        neighbours, which scales to large data and changes the result very
        little. ``"auto"`` is exact up to 5,000 points and uses 100 neighbours
        beyond that.
    columns : str or sequence of str, optional
        DataFrame column or columns to compute distances from. By default all numeric
        columns are used and the others are ignored.
    n_jobs : int, default=-1
        Number of threads. ``-1`` uses all available cores.

    Attributes
    ----------
    selected_indices_ : ndarray of shape (n_selected,)
        Positions of the selected points, in the order they were selected.
    objective_value_ : float
        ``utility_ + lambda_diversity * diversity_ / diameter_``.
    utility_ : float
        Facility-location utility of the selection. With ``n_neighbors`` set
        it counts only the similarities that were kept.
    diversity_ : float
        Smallest distance between two selected points, in the units of the
        metric. When fewer than two points are selected it is ``diameter_``,
        following the paper.
    threshold_ : float
        Distance threshold of the winning run: every selected point is at
        least this far from the others. ``0.0`` means plain greedy won and
        ``nan`` that the two farthest points did.
    diameter_ : float
        Largest distance between two points. Exact when all similarities are
        kept. With ``n_neighbors`` set it is an estimate from below, usually
        within a few percent, so ``diversity_ / diameter_`` can slightly
        exceed 1.
    gamma_ : float or None
        The ``gamma`` that was used, or ``None`` for ``similarity="inverse"``.
    n_neighbors_ : int or None
        The ``n_neighbors`` that was used, ``None`` meaning all pairs.
    n_samples_in_ : int
        Number of points seen by ``fit``.

    Notes
    -----
    Selection is deterministic: the same data and parameters always give the
    same subset. With the default ``gamma`` it also does not depend on the
    units of the data: rescaling all features leaves the selection unchanged.

    The paper states the objective with the diversity in distance units,
    ``g(S) + lambda * div(S)``. That is the objective used here with
    ``lambda = lambda_diversity / diameter_``.

    Examples
    --------
    >>> import numpy as np
    >>> from gist_sampling import GISTSelector
    >>> rng = np.random.default_rng(0)
    >>> X = np.concatenate([rng.normal(0, 1, (50, 2)), rng.normal(10, 1, (50, 2))])
    >>> selector = GISTSelector(n_samples=10).fit(X)
    >>> selector.selected_indices_.shape
    (10,)
    >>> selector.transform(X).shape
    (10, 2)
    """

    def __init__(
        self,
        n_samples: int,
        *,
        metric: Metric = "euclidean",
        lambda_diversity: float = 1.0,
        epsilon: float = 0.1,
        similarity: Similarity = "rbf",
        gamma: float | None = None,
        n_neighbors: int | Literal["auto"] | None = "auto",
        columns: str | Sequence[str] | None = None,
        n_jobs: int = -1,
    ) -> None:
        self.n_samples = n_samples
        self.metric = metric
        self.lambda_diversity = lambda_diversity
        self.epsilon = epsilon
        self.similarity = similarity
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.columns = columns
        self.n_jobs = n_jobs

    def fit(self, X: Any, y: Any = None) -> GISTSelector:
        """Select a subset of the rows of ``X``.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n, n_features)
            The data, one row per point, free of NaN and infinity. A 1-D array
            is treated as a single feature. With ``metric="precomputed"``, a
            symmetric ``(n, n)`` matrix of pairwise distances.
        y : ignored

        Returns
        -------
        self
        """
        self._check_params()
        metric = K.METRICS[self.metric]
        kind = K.SIMILARITIES[self.similarity]
        points = _as_points(X, self.columns, metric)
        n_neighbors = self._resolve_n_neighbors(points.shape[0])

        with _thread_limit(self.n_jobs) as n_threads:
            gamma = None
            if kind == K.RBF:
                gamma = self.gamma
                if gamma is None:
                    squared_median = _similarity.median_distance(metric, points) ** 2
                    if not 0 < squared_median < math.inf:
                        raise ValueError(_SCALE_ERROR)
                    gamma = 1 / squared_median
            scale = 0.0 if gamma is None else float(gamma)

            sim: SimilarityMatrix
            if n_neighbors is None:
                sim, farthest_pair = _similarity.dense_similarity(metric, points, kind, scale)
            else:
                sim = _similarity.sparse_similarity(
                    metric, points, kind, scale, n_neighbors, n_threads
                )
                farthest_pair = _similarity.farthest_pair(metric, points)

            diameter = farthest_pair[2]
            if not math.isfinite(diameter) or (diameter == 0 and (points != points[0]).any()):
                raise ValueError(_SCALE_ERROR)
            weight = self.lambda_diversity / diameter if diameter > 0 else 0.0
            best = gist(
                metric,
                points,
                sim,
                self.n_samples,
                lambda_diversity=weight,
                epsilon=self.epsilon,
                farthest_pair=farthest_pair,
            )

        self.selected_indices_ = best.indices.astype(np.intp)
        self.objective_value_ = best.objective(weight)
        self.utility_ = best.utility
        self.diversity_ = best.diversity
        self.threshold_ = best.threshold
        self.diameter_ = diameter
        self.gamma_ = None if gamma is None else float(gamma)
        self.n_neighbors_ = n_neighbors
        self.n_samples_in_ = points.shape[0]
        return self

    def transform(self, X: Any) -> Any:
        """Return the selected rows of ``X``.

        Parameters
        ----------
        X : array-like, DataFrame, Series or list
            Data with one row per point passed to ``fit``. It need not be the
            object that was fitted: use this to subset labels or raw records
            that line up with the fitted features.

        Returns
        -------
        The selected rows, as the same type as ``X``.
        """
        if not hasattr(self, "selected_indices_"):
            raise RuntimeError("This GISTSelector is not fitted yet; call fit() first.")
        if len(X) != self.n_samples_in_:
            raise ValueError(
                f"X has {len(X)} rows but the selector was fitted on {self.n_samples_in_}."
            )
        if hasattr(X, "iloc"):
            return X.iloc[self.selected_indices_]
        if isinstance(X, (list, tuple)):
            return [X[i] for i in self.selected_indices_]
        return X[self.selected_indices_]

    def fit_transform(self, X: Any, y: Any = None) -> Any:
        """Select a subset of the rows of ``X`` and return it."""
        return self.fit(X).transform(X)

    def _check_params(self) -> None:
        if not _is_int(self.n_samples) or self.n_samples < 1:
            raise ValueError(f"n_samples must be a positive integer, got {self.n_samples!r}.")
        if self.metric not in K.METRICS:
            raise ValueError(f"metric must be one of {sorted(K.METRICS)}, got {self.metric!r}.")
        if not _is_finite(self.lambda_diversity) or self.lambda_diversity < 0:
            raise ValueError(
                f"lambda_diversity must be a non-negative number, got {self.lambda_diversity!r}."
            )
        if not _is_finite(self.epsilon) or not 1e-4 <= self.epsilon <= 1:
            raise ValueError(f"epsilon must be between 0.0001 and 1, got {self.epsilon!r}.")
        if self.similarity not in K.SIMILARITIES:
            raise ValueError(
                f"similarity must be one of {sorted(K.SIMILARITIES)}, got {self.similarity!r}."
            )
        if self.gamma is not None and (not _is_finite(self.gamma) or self.gamma <= 0):
            raise ValueError(f"gamma must be a positive number or None, got {self.gamma!r}.")
        if not (
            self.n_neighbors is None
            or (isinstance(self.n_neighbors, str) and self.n_neighbors == "auto")
            or (_is_int(self.n_neighbors) and self.n_neighbors >= 1)
        ):
            raise ValueError(
                f"n_neighbors must be a positive integer, None or 'auto', got {self.n_neighbors!r}."
            )
        if not _is_int(self.n_jobs) or (self.n_jobs < 1 and self.n_jobs != -1):
            raise ValueError(f"n_jobs must be a positive integer or -1, got {self.n_jobs!r}.")

    def _resolve_n_neighbors(self, n: int) -> int | None:
        """Number of neighbours to keep, or ``None`` when that is all of them."""
        n_neighbors = self.n_neighbors
        if isinstance(n_neighbors, str):  # "auto"
            n_neighbors = None if n <= _AUTO_EXACT_LIMIT else _AUTO_NEIGHBORS
        if n_neighbors is None or n_neighbors >= n - 1:
            return None
        return int(n_neighbors)


def gist_sample(X: Any, n_samples: int, **params: Any) -> Any:
    """Return a diverse, representative subset of the rows of ``X``.

    Shorthand for ``GISTSelector(n_samples, **params).fit_transform(X)``; see
    :class:`GISTSelector` for the parameters.

    Examples
    --------
    >>> import numpy as np
    >>> from gist_sampling import gist_sample
    >>> X = np.random.default_rng(0).normal(size=(100, 2))
    >>> gist_sample(X, n_samples=10).shape
    (10, 2)
    """
    return GISTSelector(n_samples, **params).fit_transform(X)


def _is_int(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _is_finite(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _as_points(X: Any, columns: str | Sequence[str] | None, metric: int) -> np.ndarray:
    """Validate ``X`` and return the float matrix the kernels read distances from."""
    if hasattr(X, "iloc") and hasattr(X, "columns"):  # a DataFrame, without importing pandas
        if columns is None:
            frame = X.select_dtypes(include="number")
            if frame.shape[1] == 0:
                raise ValueError("X has no numeric columns.")
        else:
            columns = [columns] if isinstance(columns, str) else list(columns)
            missing = [column for column in columns if column not in X.columns]
            if missing:
                raise ValueError(f"columns not found in X: {missing}.")
            if not columns:
                raise ValueError("columns is empty.")
            frame = X[columns]
        X = frame.to_numpy()
        if X.dtype == object:  # nullable or mixed numeric columns; missing values become NaN
            with suppress(TypeError, ValueError):
                X = frame.to_numpy(dtype=np.float64, na_value=np.nan)
    elif columns is not None:
        raise TypeError("columns can only be used when X is a DataFrame.")
    elif hasattr(X, "toarray"):
        raise TypeError("Sparse input is not supported; pass a dense array.")

    points = np.asarray(X)
    if points.dtype.kind not in "fiu":
        raise ValueError(f"X must be numeric, got dtype {points.dtype}.")
    if points.ndim == 1:
        points = points.reshape(-1, 1)
    if points.ndim != 2:
        raise ValueError(f"X must be 1-D or 2-D, got {points.ndim} dimensions.")
    if 0 in points.shape:
        raise ValueError(f"X is empty (shape {points.shape}).")
    dtype = np.float32 if points.dtype in (np.float16, np.float32) else np.float64
    points = np.ascontiguousarray(points, dtype=dtype)
    if not np.isfinite(points).all():
        raise ValueError("X contains NaN or infinity.")

    if metric == K.PRECOMPUTED:
        _check_distance_matrix(points)
    elif metric == K.COSINE:
        # Dividing each row by its largest entry first keeps the norm of very
        # long and very short rows within floating-point range.
        largest = np.abs(points).max(axis=1, keepdims=True)
        if not largest.all():
            raise ValueError("Cosine distance is undefined for all-zero rows of X.")
        points = points / largest
        points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points


def _check_distance_matrix(distances: np.ndarray) -> None:
    if distances.shape[0] != distances.shape[1]:
        raise ValueError(
            f"A precomputed distance matrix must be square, got shape {distances.shape}."
        )
    if (distances < 0).any():
        raise ValueError("A precomputed distance matrix must be non-negative.")
    # Allow for rounding in how the matrix was computed. Distances derived from
    # inner products are accurate to about the square root of machine epsilon.
    tolerance = 10 * math.sqrt(np.finfo(distances.dtype).eps) * distances.max()
    if distances.diagonal().max() > tolerance:
        raise ValueError(
            "A precomputed distance matrix must have a zero diagonal. If yours is only "
            "off by rounding, set it with np.fill_diagonal(D, 0)."
        )
    if not np.allclose(distances, distances.T, rtol=0, atol=tolerance):
        raise ValueError(
            "A precomputed distance matrix must be symmetric. If yours is only off by "
            "rounding, pass np.minimum(D, D.T)."
        )


@contextmanager
def _thread_limit(n_jobs: int) -> Generator[int, None, None]:
    """Cap Numba, BLAS and OpenMP threads at ``n_jobs``; yield the resulting count.

    Only one thread at a time may be inside.
    """
    available = numba.config.NUMBA_NUM_THREADS  # type: ignore[attr-defined]
    n_threads = available if n_jobs == -1 else min(n_jobs, available)
    with _KERNEL_LOCK:
        previous = numba.get_num_threads()
        numba.set_num_threads(n_threads)
        try:
            with threadpool_limits(limits=n_threads):
                yield n_threads
        finally:
            numba.set_num_threads(previous)
