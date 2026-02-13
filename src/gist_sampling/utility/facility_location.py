"""Facility Location submodular function."""

from __future__ import annotations

import numpy as np

from gist_sampling.utility.base import SubmodularFunction


class FacilityLocationFunction(SubmodularFunction):
    """
    Facility Location submodular function.

    g(S) = Σ_{v∈V} max_{u∈S} similarity(v, u)

    This function measures how well the selected set S "covers" all elements
    in the ground set V, where coverage is measured by similarity.
    """

    def __init__(self) -> None:
        super().__init__()
        self._similarity_matrix: np.ndarray | None = None
        self._coverage: np.ndarray | None = None

    def initialize(
        self,
        X: np.ndarray,
        distance_matrix: np.ndarray,
        similarity_matrix: np.ndarray,
    ) -> None:
        """Initialize with similarity matrix."""
        self._n = X.shape[0]
        self._similarity_matrix = similarity_matrix
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

        # For each element, find max similarity to any selected element
        max_similarities = np.max(self._similarity_matrix[:, S], axis=1)
        return float(np.sum(max_similarities))

    def reset_coverage(self) -> None:
        """Reset coverage tracking to initial state (empty set)."""
        self._check_initialized()
        self._coverage = np.zeros(self._n)

    def marginal_gains_batch(self, candidates: np.ndarray) -> np.ndarray:
        """
        Compute marginal gains for all candidates using current coverage state.

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

        # Get similarity columns for all candidates at once: shape (n, len(candidates))
        candidate_similarities = self._similarity_matrix[:, candidates]

        # Compute improvement over current coverage for each candidate
        # improvement[i, j] = max(similarity[i, candidates[j]] - coverage[i], 0)
        improvements = np.maximum(candidate_similarities - self._coverage[:, np.newaxis], 0)

        # Sum over all elements to get marginal gain for each candidate
        gains = improvements.sum(axis=0)

        return gains

    def marginal_gain_single(self, candidate: int) -> float:
        """
        Compute marginal gain for a single candidate.

        Optimized single-element version that avoids the overhead of batch
        column extraction.

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
        return float(np.sum(np.maximum(col - self._coverage, 0)))

    def update_coverage(self, v: int) -> None:
        """
        Update coverage state after selecting element v.

        Parameters
        ----------
        v : int
            Index of selected element.
        """
        self._check_initialized()
        self._coverage = np.maximum(self._coverage, self._similarity_matrix[:, v])
