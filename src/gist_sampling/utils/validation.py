"""Validation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_input_data(X: pd.DataFrame | np.ndarray, *, check_finite: bool = True) -> None:
    """
    Validate input data.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Input data to validate.
    check_finite : bool, default=True
        Whether to reject NaN/Inf values.

    Raises
    ------
    TypeError
        If X is not a DataFrame or ndarray.
    ValueError
        If X is empty or not in a supported shape/type.
    """
    if not isinstance(X, (pd.DataFrame, np.ndarray)):
        raise TypeError(f"X must be a pandas DataFrame or numpy array, got {type(X).__name__}")

    if isinstance(X, pd.DataFrame):
        if X.empty:
            raise ValueError("Input DataFrame is empty")
        numeric = X.select_dtypes(include=[np.number])
        if numeric.empty:
            raise ValueError("Input DataFrame has no numeric columns")
        if check_finite and not np.isfinite(numeric.to_numpy()).all():
            raise ValueError("Input DataFrame contains NaN or Inf values")
    else:
        if X.size == 0:
            raise ValueError("Input array is empty")
        if X.ndim == 0 or X.ndim > 2:
            raise ValueError("Input array must be 1D or 2D")
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError("Input array must be numeric")
        if check_finite and not np.isfinite(X).all():
            raise ValueError("Input array contains NaN or Inf values")


def check_distance_matrix(distance_matrix: np.ndarray) -> None:
    """
    Validate a distance matrix.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Distance matrix to validate.

    Raises
    ------
    ValueError
        If the matrix is not valid.
    """
    if distance_matrix.ndim != 2:
        raise ValueError("Distance matrix must be 2-dimensional")

    n, m = distance_matrix.shape
    if n != m:
        raise ValueError("Distance matrix must be square")

    if not np.isfinite(distance_matrix).all():
        raise ValueError("Distance matrix must contain only finite values")

    if np.any(distance_matrix < 0):
        raise ValueError("Distance matrix must have non-negative entries")

    if not np.allclose(distance_matrix, distance_matrix.T):
        raise ValueError("Distance matrix must be symmetric")

    if not np.allclose(np.diag(distance_matrix), 0):
        raise ValueError("Distance matrix must have zero diagonal")
