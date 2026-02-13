"""Tests for performance optimization features."""

import numpy as np
import pytest

from gist_sampling import GISTSelector


class TestLazyDistanceMatrix:
    """Tests for lazy distance matrix."""

    def test_basic_usage(self):
        """Should compute distances on demand."""
        from gist_sampling.distance.lazy import LazyDistanceMatrix

        X = np.random.randn(100, 5)
        lazy = LazyDistanceMatrix(X, metric="euclidean")

        assert lazy.shape == (100, 100)
        assert lazy.n == 100

        # Get a single row
        row = lazy.get_row(0)
        assert row.shape == (100,)
        assert row[0] == pytest.approx(0.0)  # Distance to self

    def test_indexing(self):
        """Should support various indexing operations."""
        from gist_sampling.distance.lazy import LazyDistanceMatrix

        X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        lazy = LazyDistanceMatrix(X, metric="euclidean")

        # Row indexing
        row = lazy[0]
        assert row.shape == (4,)

        # Element indexing
        assert lazy[0, 1] == pytest.approx(1.0)
        assert lazy[0, 2] == pytest.approx(1.0)
        assert lazy[0, 3] == pytest.approx(np.sqrt(2))

        # Slice indexing
        row_slice = lazy[0, :]
        np.testing.assert_array_almost_equal(row, row_slice)

    def test_cache(self):
        """Should cache computed rows."""
        from gist_sampling.distance.lazy import LazyDistanceMatrix

        X = np.random.randn(50, 5)
        lazy = LazyDistanceMatrix(X, metric="euclidean", cache_size=10)

        # Access same row multiple times
        for _ in range(5):
            lazy.get_row(0)

        info = lazy.cache_info()
        assert info.hits >= 4  # At least 4 cache hits

    def test_all_metrics(self):
        """Should support all distance metrics."""
        from gist_sampling.distance.lazy import LazyDistanceMatrix

        X = np.random.randn(20, 3)

        for metric in ["euclidean", "manhattan", "cosine", "chebyshev"]:
            lazy = LazyDistanceMatrix(X, metric=metric)
            row = lazy.get_row(0)
            assert row.shape == (20,)
            assert row[0] == pytest.approx(0.0, abs=1e-10)

    def test_to_dense(self):
        """Should convert to dense matrix correctly."""
        from scipy.spatial.distance import cdist

        from gist_sampling.distance.lazy import LazyDistanceMatrix

        X = np.random.randn(30, 4)
        lazy = LazyDistanceMatrix(X, metric="euclidean", use_numba=False)

        dense = lazy.to_dense()
        expected = cdist(X, X, metric="euclidean")

        np.testing.assert_array_almost_equal(dense, expected)


class TestSparseSimilarity:
    """Tests for sparse similarity computation."""

    @pytest.fixture
    def sklearn_available(self):
        """Check if sklearn is available."""
        try:
            import sklearn  # noqa: F401

            return True
        except ImportError:
            return False

    def test_basic_usage(self, sklearn_available):
        """Should compute sparse similarity matrix."""
        if not sklearn_available:
            pytest.skip("scikit-learn not installed")

        from gist_sampling.distance.sparse import compute_sparse_similarity

        X = np.random.randn(100, 5)
        sparse_sim = compute_sparse_similarity(X, k_neighbors=20)

        assert sparse_sim.shape == (100, 100)
        # Should be sparse
        assert sparse_sim.nnz < 100 * 100

    def test_symmetry(self, sklearn_available):
        """Sparse similarity should be symmetric."""
        if not sklearn_available:
            pytest.skip("scikit-learn not installed")

        from gist_sampling.distance.sparse import compute_sparse_similarity

        X = np.random.randn(50, 3)
        sparse_sim = compute_sparse_similarity(X, k_neighbors=10)

        # Check symmetry
        diff = sparse_sim - sparse_sim.T
        assert abs(diff).max() < 1e-10

    def test_self_similarity(self, sklearn_available):
        """Diagonal should have similarity 1.0 (RBF of distance 0)."""
        if not sklearn_available:
            pytest.skip("scikit-learn not installed")

        from gist_sampling.distance.sparse import compute_sparse_similarity

        X = np.random.randn(30, 4)
        sparse_sim = compute_sparse_similarity(X, k_neighbors=10)

        diag = sparse_sim.diagonal()
        np.testing.assert_array_almost_equal(diag, np.ones(30))


class TestSparseFacilityLocation:
    """Tests for sparse facility location function."""

    @pytest.fixture
    def sklearn_available(self):
        """Check if sklearn is available."""
        try:
            import sklearn  # noqa: F401

            return True
        except ImportError:
            return False

    def test_basic_usage(self, sklearn_available):
        """Should work with sparse similarity matrix."""
        if not sklearn_available:
            pytest.skip("scikit-learn not installed")

        from gist_sampling.distance.sparse import compute_sparse_similarity
        from gist_sampling.utility.sparse_facility_location import SparseFacilityLocationFunction

        X = np.random.randn(50, 3)
        sparse_sim = compute_sparse_similarity(X, k_neighbors=20)

        fl = SparseFacilityLocationFunction()
        fl.initialize(X, None, sparse_sim)

        # Evaluate on a subset
        obj = fl.evaluate([0, 10, 20])
        assert obj > 0

    def test_marginal_gains(self, sklearn_available):
        """Should compute marginal gains correctly."""
        if not sklearn_available:
            pytest.skip("scikit-learn not installed")

        from gist_sampling.distance.sparse import compute_sparse_similarity
        from gist_sampling.utility.sparse_facility_location import SparseFacilityLocationFunction

        X = np.random.randn(30, 3)
        sparse_sim = compute_sparse_similarity(X, k_neighbors=15)

        fl = SparseFacilityLocationFunction()
        fl.initialize(X, None, sparse_sim)
        fl.reset_coverage()

        gains = fl.marginal_gains_batch(np.arange(30))
        assert len(gains) == 30
        assert all(g >= 0 for g in gains)

    def test_coverage_update(self, sklearn_available):
        """Coverage should increase after adding elements."""
        if not sklearn_available:
            pytest.skip("scikit-learn not installed")

        from gist_sampling.distance.sparse import compute_sparse_similarity
        from gist_sampling.utility.sparse_facility_location import SparseFacilityLocationFunction

        X = np.random.randn(30, 3)
        sparse_sim = compute_sparse_similarity(X, k_neighbors=15)

        fl = SparseFacilityLocationFunction()
        fl.initialize(X, None, sparse_sim)
        fl.reset_coverage()

        # Initial objective
        obj0 = fl.evaluate([])

        # Add element
        fl.update_coverage(0)
        obj1 = fl.evaluate([0])

        assert obj1 > obj0


class TestHybridMode:
    """Tests for hybrid mode in GISTSelector."""

    def test_exact_mode_small_data(self, simple_data):
        """Should use exact mode for small datasets."""
        selector = GISTSelector(n_samples=5, mode="exact")
        selector.fit(simple_data)

        assert selector.mode_used_ == "exact"
        assert len(selector.selected_indices_) <= 5

    def test_auto_mode_small_data(self, simple_data):
        """Auto mode should use exact for small datasets."""
        selector = GISTSelector(n_samples=5, mode="auto", approximate_threshold=100)
        selector.fit(simple_data)

        assert selector.mode_used_ == "exact"

    @pytest.fixture
    def sklearn_available(self):
        """Check if sklearn is available."""
        try:
            import sklearn  # noqa: F401

            return True
        except ImportError:
            return False

    def test_approximate_mode(self, sklearn_available):
        """Should use approximate mode when requested."""
        if not sklearn_available:
            pytest.skip("scikit-learn not installed")

        np.random.seed(42)
        X = np.random.randn(200, 5)

        selector = GISTSelector(n_samples=10, mode="approximate", k_neighbors=50)
        selector.fit(X)

        assert selector.mode_used_ == "approximate"
        assert len(selector.selected_indices_) <= 10
        assert selector.objective_value_ > 0

    def test_auto_mode_large_data(self, sklearn_available):
        """Auto mode should use approximate for large datasets."""
        if not sklearn_available:
            pytest.skip("scikit-learn not installed")

        np.random.seed(42)
        X = np.random.randn(200, 5)

        selector = GISTSelector(n_samples=10, mode="auto", approximate_threshold=100)
        selector.fit(X)

        assert selector.mode_used_ == "approximate"

    def test_approximate_mode_quality(self, sklearn_available):
        """Approximate mode should produce reasonable results."""
        if not sklearn_available:
            pytest.skip("scikit-learn not installed")

        np.random.seed(42)
        # Create clustered data
        cluster1 = np.random.randn(100, 2)
        cluster2 = np.random.randn(100, 2) + np.array([10, 10])
        X = np.vstack([cluster1, cluster2])

        selector = GISTSelector(n_samples=20, mode="approximate", k_neighbors=50)
        selector.fit(X)

        # Should select from both clusters
        from_cluster1 = sum(1 for idx in selector.selected_indices_ if idx < 100)
        from_cluster2 = sum(1 for idx in selector.selected_indices_ if idx >= 100)

        assert from_cluster1 >= 5
        assert from_cluster2 >= 5

    def test_backward_compatibility(self, simple_data):
        """New parameters should have sensible defaults."""
        # Without new parameters (should work like before)
        selector1 = GISTSelector(n_samples=5)
        selector1.fit(simple_data)

        # With explicit exact mode
        selector2 = GISTSelector(n_samples=5, mode="exact")
        selector2.fit(simple_data)

        # Should produce same results
        np.testing.assert_array_equal(selector1.selected_indices_, selector2.selected_indices_)

    def test_invalid_mode(self):
        """Should reject invalid mode values."""
        with pytest.raises(ValueError, match="mode must be"):
            GISTSelector(n_samples=5, mode="invalid")

    def test_invalid_k_neighbors(self):
        """Should reject invalid k_neighbors values."""
        with pytest.raises(ValueError, match="k_neighbors must be positive"):
            GISTSelector(n_samples=5, k_neighbors=0)

    def test_invalid_threshold(self):
        """Should reject invalid threshold values."""
        with pytest.raises(ValueError, match="approximate_threshold must be positive"):
            GISTSelector(n_samples=5, approximate_threshold=0)


class TestNumbaKernels:
    """Tests for Numba JIT-compiled kernels."""

    def test_numba_availability_check(self):
        """Should correctly detect Numba availability."""
        from gist_sampling.distance._numba_kernels import is_numba_available

        result = is_numba_available()
        assert isinstance(result, bool)

    def test_distance_kernels_fallback(self):
        """Distance kernels should work without Numba."""
        from gist_sampling.distance._numba_kernels import get_distance_kernel

        X = np.random.randn(20, 3).astype(np.float64)

        for metric in ["euclidean", "manhattan", "cosine", "chebyshev"]:
            kernel = get_distance_kernel(metric)
            row = kernel(X, 0)
            assert row.shape == (20,)
            assert row[0] == pytest.approx(0.0, abs=1e-10)

    def test_invalid_metric(self):
        """Should raise error for unsupported metric."""
        from gist_sampling.distance._numba_kernels import get_distance_kernel

        with pytest.raises(ValueError, match="Unsupported metric"):
            get_distance_kernel("invalid_metric")
