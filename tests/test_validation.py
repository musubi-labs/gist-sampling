"""Tests for validation utilities."""

import numpy as np
import pandas as pd
import pytest

from gist_sampling.utils.validation import check_distance_matrix, validate_input_data


def test_validate_input_data_rejects_nan_array():
    """NaN values in arrays should be rejected."""
    data = np.array([[1.0, np.nan]])
    with pytest.raises(ValueError, match="NaN|Inf"):
        validate_input_data(data)


def test_validate_input_data_rejects_non_numeric_array():
    """Non-numeric arrays should be rejected."""
    data = np.array([["a", "b"]])
    with pytest.raises(ValueError, match="numeric"):
        validate_input_data(data)


def test_validate_input_data_rejects_nan_dataframe():
    """NaN values in DataFrames should be rejected."""
    df = pd.DataFrame({"x": [1.0, np.nan], "y": [2.0, 3.0]})
    with pytest.raises(ValueError, match="NaN|Inf"):
        validate_input_data(df)


def test_check_distance_matrix_rejects_nan():
    """Distance matrix must be finite."""
    matrix = np.array([[0.0, np.nan], [np.nan, 0.0]])
    with pytest.raises(ValueError, match="finite"):
        check_distance_matrix(matrix)
