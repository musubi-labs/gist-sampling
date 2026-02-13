"""Sparse Facility Location submodular function for large-scale datasets."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from gist_sampling.utility.base import SubmodularFunction


class SparseFacilityLocationFunction(SubmodularFunction):
    """
    Facility Location function using sparse similarity matrix.

    This implementation uses a sparse similarity matrix (CSC format) for
    memory-efficient computation on large datasets. Instead of storing
    O(n²) similarities, it only stores the top-k similarities per point,
    reducing memory to O(n × k).

    g(S) = Σ_{v∈V} max_{u∈S} similarity(v, u)
    """

    def __init__(self) -> None:
        super().__init__()
        self._similarity_matrix: sparse.csc_matrix | None = None
        self._coverage: np.ndarray | None = None

    def initialize(
        self,
        X: np.ndarray,
        distance_matrix: np.ndarray | None,
        similarity_matrix: sparse.spmatrix | np.ndarray,
    ) -> None:
        """
        Initialize with sparse similarity matrix.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (n, d).
        distance_matrix : np.ndarray or None
            Not used (included for API compatibility).
        similarity_matrix : sparse matrix or np.ndarray
            Pairwise similarity matrix. Converted to CSC format for fast column access.
        """
        self._n = X.shape[0]

        # Convert to CSC format for fast column slicing
        if isinstance(similarity_matrix, np.ndarray):
            self._similarity_matrix = sparse.csc_matrix(similarity_matrix)
        elif sparse.issparse(similarity_matrix):
            self._similarity_matrix = similarity_matrix.tocsc()
        else:
            raise TypeError(
                f"similarity_matrix must be ndarray or sparse matrix, got {type(similarity_matrix)}"
            )

        self._coverage = np.zeros(self._n)
        self._initialized = True

    def evaluate(self, S: list[int]) -> float:
        """
        Evaluate facility location objective.

        g(S) = Σ_{v∈V} max_{u∈S} similarity(v, u)
        """
        self._check_initialized()

        if len(S) == 0:
            return 0.0

        # Batch extract columns and compute max per row
        selected_cols = self._similarity_matrix[:, S]
        if sparse.issparse(selected_cols):
            max_similarities = selected_cols.max(axis=1).toarray().ravel()
        else:
            max_similarities = np.max(selected_cols, axis=1)

        return float(np.sum(max_similarities))

    def reset_coverage(self) -> None:
        """Reset coverage tracking to initial state (empty set)."""
        self._check_initialized()
        self._coverage = np.zeros(self._n)

    def marginal_gains_batch(self, candidates: np.ndarray) -> np.ndarray:
        """
        Compute marginal gains for all candidates using current coverage state.

        Uses batch column extraction for efficiency.

        Parameters
        ----------
        candidates : np.ndarray
            Array of candidate indices.

        Returns
        -------
        np.ndarray
            Array of marginal gains for each candidate.
        """
        self._check_initialized()

        if len(candidates) == 0:
            return np.array([])

        candidate_cols = self._similarity_matrix[:, candidates]
        if sparse.issparse(candidate_cols):
            candidate_cols = candidate_cols.tocsc()
            if candidate_cols.nnz == 0:
                return np.zeros(candidate_cols.shape[1])
            diff = candidate_cols.data - self._coverage[candidate_cols.indices]
            diff = np.maximum(diff, 0)
            col_indices = np.repeat(
                np.arange(candidate_cols.shape[1]),
                np.diff(candidate_cols.indptr),
            )
            return np.bincount(
                col_indices,
                weights=diff,
                minlength=candidate_cols.shape[1],
            )

        candidate_cols = np.asarray(candidate_cols)
        improvements = np.maximum(candidate_cols - self._coverage[:, np.newaxis], 0)
        return improvements.sum(axis=0)

    def marginal_gain_single(self, candidate: int) -> float:
        """
        Compute marginal gain for a single candidate.

        Optimized single-element version that operates on sparse column data
        without batch extraction overhead.

        Parameters
        ----------
        candidate : int
            Index of the candidate element.

        Returns
        -------
        float
            Marginal gain for the candidate.
        """
        self._check_initialized()
        col = self._similarity_matrix[:, candidate]
        if sparse.issparse(col):
            col = col.tocsc()
            if col.nnz == 0:
                return 0.0
            diff = np.maximum(col.data - self._coverage[col.indices], 0)
            return float(diff.sum())
        col_arr = np.asarray(col).ravel()
        return float(np.maximum(col_arr - self._coverage, 0).sum())

    def update_coverage(self, v: int) -> None:
        """
        Update coverage state after selecting element v.

        Parameters
        ----------
        v : int
            Index of selected element.
        """
        self._check_initialized()
        col = self._similarity_matrix[:, v]
        if sparse.issparse(col):
            col = col.tocsc()
            if col.nnz:
                self._coverage[col.indices] = np.maximum(self._coverage[col.indices], col.data)
        else:
            self._coverage = np.maximum(self._coverage, col)
