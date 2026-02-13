"""Test fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def simple_data():
    """Simple 2D data with two clusters."""
    np.random.seed(42)
    cluster1 = np.random.randn(20, 2)
    cluster2 = np.random.randn(20, 2) + np.array([10, 10])
    return np.vstack([cluster1, cluster2])


@pytest.fixture
def simple_df(simple_data):
    """Simple DataFrame with two clusters."""
    return pd.DataFrame(simple_data, columns=["x", "y"])


@pytest.fixture
def collinear_data():
    """1D collinear data."""
    return np.array([[i] for i in range(10)], dtype=float)
