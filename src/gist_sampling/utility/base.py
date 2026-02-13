"""Base class for submodular utility functions."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SubmodularFunction(ABC):
    """Abstract base class for submodular utility functions."""

    def __init__(self) -> None:
        self._initialized = False
        self._n: int = 0

    @abstractmethod
    def initialize(
        self,
        X: np.ndarray,
        distance_matrix: np.ndarray,
        similarity_matrix: np.ndarray,
    ) -> None:
        """
        Initialize the function with data.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (n, d).
        distance_matrix : np.ndarray
            Pairwise distance matrix of shape (n, n).
        similarity_matrix : np.ndarray
            Pairwise similarity matrix of shape (n, n).
        """
        pass

    @abstractmethod
    def evaluate(self, S: list[int]) -> float:
        """
        Evaluate the utility function on a set.

        Parameters
        ----------
        S : list[int]
            Indices of selected elements.

        Returns
        -------
        float
            Utility value.
        """
        pass

    @abstractmethod
    def reset_coverage(self) -> None:
        """Reset coverage tracking to initial state (empty set)."""
        pass

    @abstractmethod
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
        pass

    def marginal_gain_single(self, candidate: int) -> float:
        """
        Compute marginal gain for a single candidate using current coverage state.

        Default implementation delegates to marginal_gains_batch.
        Subclasses may override for efficiency in lazy evaluation.

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
        gains = self.marginal_gains_batch(np.array([candidate]))
        return float(gains[0])

    @abstractmethod
    def update_coverage(self, v: int) -> None:
        """
        Update coverage state after selecting element v.

        Parameters
        ----------
        v : int
            Index of selected element.
        """
        pass

    def _check_initialized(self) -> None:
        """Raise error if not initialized."""
        if not self._initialized:
            raise RuntimeError("Function must be initialized before use")
