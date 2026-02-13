"""GIST Selector for diversity-aware sampling."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from typing import Literal

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from gist_sampling.core.gist import gist
from gist_sampling.distance.metrics import (
    MetricType,
    compute_distance_matrix,
    compute_similarity_matrix,
)
from gist_sampling.utility.base import SubmodularFunction
from gist_sampling.utility.facility_location import FacilityLocationFunction
from gist_sampling.utils.validation import check_distance_matrix, validate_input_data

UtilityType = Literal["facility_location"]
ModeType = Literal["auto", "exact", "approximate"]


@contextmanager
def _numba_thread_limit(n_jobs: int) -> None:
    try:
        import numba  # type: ignore[import-not-found]
    except ImportError:
        yield
        return
    prev_threads = numba.get_num_threads()
    numba.set_num_threads(n_jobs)
    try:
        yield
    finally:
        numba.set_num_threads(prev_threads)


@contextmanager
def _sklearn_inner_thread_limit(n_jobs: int) -> None:
    if n_jobs == 1:
        yield
        return
    with threadpool_limits(limits=1), _numba_thread_limit(1):
        yield


# Default threshold for switching to approximate mode
# Benchmarks show approximate is faster above n~1000, using 2000 for safety margin
DEFAULT_APPROXIMATE_THRESHOLD = 2000


class GISTSelector:
    """
    GIST (Greedy Independent Set Thresholding) selector for diversity-aware sampling.

    This selector implements the GIST algorithm from arXiv:2405.18754 for
    downsampling datasets while maintaining diversity and optimizing a
    submodular utility function.

    Parameters
    ----------
    n_samples : int
        Number of samples to select (budget k).
    utility : UtilityType, default="facility_location"
        Utility function type. Currently only "facility_location" is supported.
    metric : MetricType, default="euclidean"
        Distance metric. One of "euclidean", "cosine", "manhattan", "chebyshev".
    epsilon : float, default=0.1
        Approximation parameter controlling threshold granularity.
        Smaller values give better approximation but slower runtime.
    lambda_diversity : float, default=1.0
        Diversity weight (λ) in f(S) = g(S) + λ·div(S).
    columns : list[str] | None, default=None
        Columns to use for distance computation. If None, uses all numeric columns.
    similarity : Literal["rbf", "inverse"], default="rbf"
        Similarity function for utility. One of "rbf", "inverse".
    gamma : float | None, default=None
        RBF kernel parameter. If None, uses median heuristic.
    distance_matrix : np.ndarray | None, default=None
        Precomputed distance matrix for exact mode only.
        If provided, metric parameter is ignored in exact mode.
    random_state : int | None, default=None
        Seed for randomized diameter estimation (approximate mode).
    mode : {"auto", "exact", "approximate"}, default="auto"
        Computation mode:
        - "auto": Use exact for small datasets (n <= approximate_threshold),
          approximate for large datasets.
        - "exact": Always use exact O(n²) computation (original behavior).
        - "approximate": Always use approximate O(n × k) computation.
    approximate_threshold : int, default=2000
        Dataset size threshold for switching to approximate mode when mode="auto".
    k_neighbors : int, default=100
        Number of nearest neighbors to use for sparse similarity in approximate mode.
    n_jobs : int, default=-1
        Max threads for internal parallelism (k-NN + BLAS/OpenMP).
        When k-NN runs with n_jobs != 1, BLAS/OpenMP threads are capped to 1
        during that step to avoid oversubscription (scikit-learn best practice).

    Attributes
    ----------
    selected_indices_ : np.ndarray
        Indices of selected samples after fitting.
    objective_value_ : float
        Objective value f(S) = g(S) + λ·div(S) of the selected set.
    diversity_ : float
        Minimum pairwise distance among selected samples.
    mode_used_ : str
        Actual mode used ("exact" or "approximate") after fitting.

    Examples
    --------
    >>> import pandas as pd
    >>> from gist_sampling import GISTSelector
    >>> df = pd.DataFrame({"x": [1, 2, 3, 10, 11, 12], "y": [1, 1, 1, 1, 1, 1]})
    >>> selector = GISTSelector(n_samples=2)
    >>> df_sampled = selector.fit_transform(df)
    >>> len(df_sampled)
    2

    For large datasets, approximate mode is faster and more memory-efficient:

    >>> selector = GISTSelector(n_samples=100, mode="approximate", k_neighbors=200)
    >>> result = selector.fit_transform(large_data)
    """

    def __init__(
        self,
        n_samples: int,
        utility: UtilityType = "facility_location",
        metric: MetricType = "euclidean",
        epsilon: float = 0.1,
        lambda_diversity: float = 1.0,
        columns: list[str] | None = None,
        similarity: Literal["rbf", "inverse"] = "rbf",
        gamma: float | None = None,
        distance_matrix: np.ndarray | None = None,
        random_state: int | None = None,
        mode: ModeType = "auto",
        approximate_threshold: int = DEFAULT_APPROXIMATE_THRESHOLD,
        k_neighbors: int = 100,
        n_jobs: int = -1,
    ) -> None:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        self.n_samples = n_samples
        self._fitted = False
        self.selected_indices_: np.ndarray | None = None

        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if lambda_diversity < 0:
            raise ValueError("lambda_diversity must be non-negative")
        if random_state is not None and not isinstance(random_state, (int, np.integer)):
            raise ValueError("random_state must be an int or None")
        if mode not in ("auto", "exact", "approximate"):
            raise ValueError(f"mode must be 'auto', 'exact', or 'approximate', got '{mode}'")
        if approximate_threshold <= 0:
            raise ValueError("approximate_threshold must be positive")
        if k_neighbors <= 0:
            raise ValueError("k_neighbors must be positive")
        if n_jobs == 0 or n_jobs < -1:
            raise ValueError("n_jobs must be -1 or a positive integer")

        self.utility = utility
        self.metric = metric
        self.epsilon = epsilon
        self.lambda_diversity = lambda_diversity
        self.columns = columns
        self.similarity = similarity
        self.gamma = gamma
        self.distance_matrix = distance_matrix
        self.random_state = int(random_state) if random_state is not None else None
        self.mode = mode
        self.approximate_threshold = approximate_threshold
        self.k_neighbors = k_neighbors
        self.n_jobs = n_jobs

        # Attributes set after fit
        self.objective_value_: float | None = None
        self.diversity_: float | None = None
        self.mode_used_: str | None = None

    def _get_utility_function(self, sparse: bool = False) -> SubmodularFunction:
        """Get the utility function instance."""
        if self.utility == "facility_location":
            if sparse:
                from gist_sampling.utility.sparse_facility_location import (
                    SparseFacilityLocationFunction,
                )

                return SparseFacilityLocationFunction()
            else:
                return FacilityLocationFunction()
        else:
            raise ValueError(f"Unknown utility function: {self.utility}")

    def _prepare_data(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Prepare data matrix for processing."""
        if not isinstance(X, pd.DataFrame):
            data = np.asarray(X)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            if data.ndim != 2:
                raise ValueError("Input array must be 1D or 2D")
            return data.astype(np.float64)

        if self.columns is not None:
            if len(self.columns) == 0:
                raise ValueError("columns must be a non-empty list of column names")

            missing = [col for col in self.columns if col not in X.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")

            data_frame = X[self.columns]
        else:
            data_frame = X.select_dtypes(include=[np.number])

        if data_frame.select_dtypes(include=[np.number]).shape[1] != data_frame.shape[1]:
            raise ValueError("All selected columns must be numeric")

        data = data_frame.to_numpy()
        return data.astype(np.float64)

    def _should_use_approximate(self, n: int) -> bool:
        """Determine whether to use approximate mode."""
        if self.mode == "exact":
            return False
        elif self.mode == "approximate":
            return True
        else:  # auto
            return n > self.approximate_threshold

    def _run_with_thread_limits(
        self, func: Callable[[], tuple[list[int], float, float]]
    ) -> tuple[list[int], float, float]:
        if self.n_jobs is None or self.n_jobs == -1:
            return func()
        with threadpool_limits(limits=self.n_jobs), _numba_thread_limit(self.n_jobs):
            return func()

    def _fit_exact(self, data: np.ndarray) -> tuple[list[int], float, float]:
        """Fit using exact O(n²) computation."""
        n = data.shape[0]

        # Use precomputed distance matrix if provided, otherwise compute
        if self.distance_matrix is not None:
            check_distance_matrix(self.distance_matrix)
            if self.distance_matrix.shape[0] != n:
                raise ValueError(
                    f"distance_matrix shape ({self.distance_matrix.shape[0]}) "
                    f"does not match data shape ({n})"
                )
            distance_matrix = self.distance_matrix
        else:
            distance_matrix = compute_distance_matrix(data, metric=self.metric)

        similarity_matrix = compute_similarity_matrix(
            distance_matrix, method=self.similarity, gamma=self.gamma
        )

        # Initialize utility function
        utility_fn = self._get_utility_function(sparse=False)
        utility_fn.initialize(data, distance_matrix, similarity_matrix)

        # Run GIST algorithm
        return gist(
            X=data,
            utility_fn=utility_fn,
            k=self.n_samples,
            epsilon=self.epsilon,
            lambda_diversity=self.lambda_diversity,
            distance_matrix=distance_matrix,
            random_state=self.random_state,
        )

    def _fit_approximate(self, data: np.ndarray) -> tuple[list[int], float, float]:
        """Fit using approximate O(n × k) computation."""
        from gist_sampling.distance.lazy import LazyDistanceMatrix
        from gist_sampling.distance.sparse import compute_sparse_similarity

        n = data.shape[0]

        # Create lazy distance matrix for on-demand computation
        lazy_distance = LazyDistanceMatrix(
            data,
            metric=self.metric,
            cache_size=min(1024, n),
            use_numba=True,
        )

        # Compute sparse similarity matrix
        k_neighbors = min(self.k_neighbors, n - 1)
        with _sklearn_inner_thread_limit(self.n_jobs):
            sparse_similarity = compute_sparse_similarity(
                data,
                k_neighbors=k_neighbors,
                metric=self.metric,
                similarity_method=self.similarity,
                gamma=self.gamma,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )

        # Initialize sparse utility function
        utility_fn = self._get_utility_function(sparse=True)
        utility_fn.initialize(data, None, sparse_similarity)

        # Run approximate GIST algorithm
        return gist(
            X=data,
            utility_fn=utility_fn,
            k=self.n_samples,
            epsilon=self.epsilon,
            lambda_diversity=self.lambda_diversity,
            distance_matrix=lazy_distance,
            random_state=self.random_state,
        )

    def fit(self, X: pd.DataFrame | np.ndarray) -> GISTSelector:
        """
        Fit the GIST selector to data.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input data. For DataFrames, numeric columns are used by default,
            or specify columns with the `columns` parameter.

        Returns
        -------
        self
        """
        validate_input_data(X, check_finite=False)

        data = self._prepare_data(X)
        n = data.shape[0]

        if n == 0:
            raise ValueError("Input data is empty")
        if not np.isfinite(data).all():
            raise ValueError("Input data contains NaN or Inf values")

        # Determine which mode to use
        use_approximate = self._should_use_approximate(n)
        if use_approximate and self.distance_matrix is not None:
            raise ValueError(
                "distance_matrix is only supported in exact mode. "
                "Use mode='exact' or omit distance_matrix."
            )

        if use_approximate:
            self.mode_used_ = "approximate"
            selected_indices, objective_value, diversity = self._run_with_thread_limits(
                lambda: self._fit_approximate(data)
            )
        else:
            self.mode_used_ = "exact"
            selected_indices, objective_value, diversity = self._run_with_thread_limits(
                lambda: self._fit_exact(data)
            )

        self.selected_indices_ = np.array(selected_indices)
        self.objective_value_ = objective_value
        self.diversity_ = diversity
        self._fitted = True

        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """
        Return selected samples from X.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input data (must be same as used in fit).

        Returns
        -------
        pd.DataFrame or np.ndarray
            Selected samples.
        """
        if not self._fitted:
            raise RuntimeError("Selector must be fitted before transform")

        if self.selected_indices_ is None:
            raise RuntimeError("No indices selected")

        n_rows = len(X)
        if np.any(self.selected_indices_ < 0) or np.any(self.selected_indices_ >= n_rows):
            raise ValueError(
                "selected_indices_ are out of bounds for the provided X; "
                "ensure transform() uses the same data as fit()."
            )

        if isinstance(X, pd.DataFrame):
            return X.iloc[self.selected_indices_]
        return X[self.selected_indices_]

    def fit_transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """
        Fit and transform in one step.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input data.

        Returns
        -------
        pd.DataFrame or np.ndarray
            Selected samples.
        """
        return self.fit(X).transform(X)
