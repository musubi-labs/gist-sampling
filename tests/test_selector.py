"""Tests for GISTSelector."""

import numpy as np
import pandas as pd
import pytest

from gist_sampling import GISTSelector, gist_sample


class TestGISTSelector:
    """Tests for GISTSelector class."""

    def test_basic_usage_array(self, simple_data):
        """Should work with numpy arrays."""
        selector = GISTSelector(n_samples=10)
        result = selector.fit_transform(simple_data)

        assert len(result) <= 10
        assert isinstance(result, np.ndarray)
        assert selector.selected_indices_ is not None
        assert selector.objective_value_ > 0
        assert selector.diversity_ >= 0

    def test_basic_usage_dataframe(self, simple_df):
        """Should work with pandas DataFrames."""
        selector = GISTSelector(n_samples=10)
        result = selector.fit_transform(simple_df)

        assert len(result) <= 10
        assert isinstance(result, pd.DataFrame)

    def test_cluster_diversity(self):
        """Should select balanced samples from different clusters."""
        np.random.seed(42)
        cluster1 = np.random.randn(50, 2)
        cluster2 = np.random.randn(50, 2) + np.array([5, 5])
        data = np.vstack([cluster1, cluster2])

        selector = GISTSelector(n_samples=10)
        selector.fit(data)

        from_cluster1 = sum(1 for idx in selector.selected_indices_ if idx < 50)
        from_cluster2 = sum(1 for idx in selector.selected_indices_ if idx >= 50)

        assert from_cluster1 >= 3
        assert from_cluster2 >= 3

    def test_precomputed_distance_matrix(self, simple_data):
        """Should work with precomputed distance matrix."""
        from scipy.spatial.distance import cdist

        dist_matrix = cdist(simple_data, simple_data, metric="euclidean")

        selector1 = GISTSelector(n_samples=10, distance_matrix=dist_matrix)
        selector1.fit(simple_data)

        selector2 = GISTSelector(n_samples=10, metric="euclidean")
        selector2.fit(simple_data)

        np.testing.assert_array_equal(
            selector1.selected_indices_,
            selector2.selected_indices_,
        )

    def test_missing_columns(self, simple_df):
        """Should raise if specified columns are missing."""
        selector = GISTSelector(n_samples=5, columns=["x", "z"])
        with pytest.raises(ValueError, match="Missing columns"):
            selector.fit(simple_df)

    def test_non_numeric_columns(self):
        """Should raise if specified columns are non-numeric."""
        df = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        selector = GISTSelector(n_samples=1, columns=["y"])
        with pytest.raises(ValueError, match="numeric"):
            selector.fit(df)

    def test_transform_bounds(self, simple_data):
        """Transform should reject mismatched data sizes."""
        selector = GISTSelector(n_samples=5)
        selector.fit(simple_data)
        smaller = simple_data[:1]
        with pytest.raises(ValueError, match="out of bounds"):
            selector.transform(smaller)

    def test_invalid_inputs(self):
        """Should raise errors for invalid inputs."""
        with pytest.raises(ValueError):
            GISTSelector(n_samples=0)

        with pytest.raises(ValueError):
            GISTSelector(n_samples=10, epsilon=-0.1)

        with pytest.raises(ValueError):
            GISTSelector(n_samples=10).fit(np.array([]).reshape(0, 2))

    def test_scalar_array_rejected(self):
        """Scalar ndarrays should fail with a clear validation error."""
        with pytest.raises(ValueError, match="1D or 2D"):
            GISTSelector(n_samples=1).fit(np.array(1.23))

    def test_columns_only_validate_selected_features(self):
        """NaN/Inf in unselected numeric columns should not block fitting."""
        df = pd.DataFrame(
            {
                "x": [0.0, 1.0, 2.0],
                "y": [1.0, 2.0, 3.0],
                "unused_bad": [0.0, np.nan, 2.0],
            }
        )
        selector = GISTSelector(n_samples=2, columns=["x", "y"])
        selector.fit(df)
        assert len(selector.selected_indices_) <= 2

    def test_distance_matrix_rejected_in_approximate_mode(self):
        """Precomputed distances are only supported in exact mode."""
        X = np.array([[0.0], [1.0], [2.0]])
        distance_matrix = np.zeros((3, 3))
        selector = GISTSelector(
            n_samples=2,
            mode="approximate",
            distance_matrix=distance_matrix,
        )
        with pytest.raises(ValueError, match="only supported in exact mode"):
            selector.fit(X)


class TestGistSample:
    """Tests for gist_sample convenience function."""

    def test_basic_usage(self, simple_data):
        """Should work as a one-liner."""
        result = gist_sample(simple_data, n_samples=10)
        assert len(result) <= 10
