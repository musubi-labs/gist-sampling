"""Tests for utility functions."""

import numpy as np
import pytest

from gist_sampling.distance.metrics import compute_distance_matrix, compute_similarity_matrix
from gist_sampling.utility.facility_location import FacilityLocationFunction


class TestFacilityLocationFunction:
    """Tests for FacilityLocationFunction."""

    @pytest.fixture
    def setup_utility(self):
        """Create initialized utility function."""
        data = np.array([[0, 0], [1, 0], [0, 1], [5, 5]], dtype=float)
        distance_matrix = compute_distance_matrix(data)
        similarity_matrix = compute_similarity_matrix(distance_matrix)
        utility = FacilityLocationFunction()
        utility.initialize(data, distance_matrix, similarity_matrix)
        return utility, data

    def test_monotone(self, setup_utility):
        """Utility should be monotone (adding elements never decreases value)."""
        utility, data = setup_utility
        n = len(data)

        for _ in range(10):
            indices = list(np.random.permutation(n))
            current_set = []
            prev_value = 0.0

            for idx in indices:
                current_set.append(idx)
                new_value = utility.evaluate(current_set)
                assert new_value >= prev_value
                prev_value = new_value

    def test_submodular(self, setup_utility):
        """Utility should satisfy diminishing returns property."""
        utility, data = setup_utility
        n = len(data)

        # Test: g(A ∪ {v}) - g(A) >= g(B ∪ {v}) - g(B) when A ⊆ B
        for _ in range(20):
            perm = np.random.permutation(n)
            split = np.random.randint(0, n)
            A = list(perm[:split])
            B = list(perm)

            remaining = set(range(n)) - set(B)
            if remaining:
                v = list(remaining)[0]
                gain_A = utility.evaluate(A + [v]) - utility.evaluate(A)
                gain_B = utility.evaluate(B + [v]) - utility.evaluate(B)
                assert gain_A >= gain_B - 1e-10


class TestMarginalGainSingle:
    """Tests for marginal_gain_single consistency with marginal_gains_batch."""

    @pytest.fixture
    def dense_utility(self):
        """Create initialized dense facility location function."""
        data = np.array([[0, 0], [1, 0], [0, 1], [5, 5], [3, 2]], dtype=float)
        distance_matrix = compute_distance_matrix(data)
        similarity_matrix = compute_similarity_matrix(distance_matrix)
        utility = FacilityLocationFunction()
        utility.initialize(data, distance_matrix, similarity_matrix)
        return utility, data

    def test_single_matches_batch_initial(self, dense_utility):
        """marginal_gain_single should match marginal_gains_batch with empty coverage."""
        utility, data = dense_utility
        n = len(data)
        utility.reset_coverage()

        batch_gains = utility.marginal_gains_batch(np.arange(n))
        for i in range(n):
            single_gain = utility.marginal_gain_single(i)
            assert single_gain == pytest.approx(batch_gains[i], abs=1e-10)

    def test_single_matches_batch_after_updates(self, dense_utility):
        """marginal_gain_single should match batch after coverage updates."""
        utility, data = dense_utility
        utility.reset_coverage()

        # Update coverage with a couple of elements
        utility.update_coverage(0)
        utility.update_coverage(3)

        remaining = np.array([1, 2, 4])
        batch_gains = utility.marginal_gains_batch(remaining)
        for i, idx in enumerate(remaining):
            single_gain = utility.marginal_gain_single(int(idx))
            assert single_gain == pytest.approx(batch_gains[i], abs=1e-10)

    def test_gains_decrease_after_coverage_update(self, dense_utility):
        """Gains should decrease (submodularity) after adding an element."""
        utility, data = dense_utility
        utility.reset_coverage()

        # Get initial gain for element 1
        gain_before = utility.marginal_gain_single(1)

        # Update coverage with element 0
        utility.update_coverage(0)

        # Gain should decrease or stay the same
        gain_after = utility.marginal_gain_single(1)
        assert gain_after <= gain_before + 1e-10

    def test_sparse_single_matches_batch(self):
        """Sparse marginal_gain_single should match batch."""
        from gist_sampling.distance.sparse import compute_sparse_similarity
        from gist_sampling.utility.sparse_facility_location import (
            SparseFacilityLocationFunction,
        )

        np.random.seed(42)
        X = np.random.randn(50, 3)
        sparse_sim = compute_sparse_similarity(X, k_neighbors=20)

        fl = SparseFacilityLocationFunction()
        fl.initialize(X, None, sparse_sim)
        fl.reset_coverage()

        # Check initial gains
        all_indices = np.arange(50)
        batch_gains = fl.marginal_gains_batch(all_indices)
        for i in range(50):
            single_gain = fl.marginal_gain_single(i)
            assert single_gain == pytest.approx(batch_gains[i], abs=1e-10)

        # After coverage update
        fl.update_coverage(0)
        fl.update_coverage(25)
        remaining = np.array([5, 10, 15, 30, 40])
        batch_gains = fl.marginal_gains_batch(remaining)
        for i, idx in enumerate(remaining):
            single_gain = fl.marginal_gain_single(int(idx))
            assert single_gain == pytest.approx(batch_gains[i], abs=1e-10)


class TestSimilarityMatrix:
    """Tests for similarity matrix computation."""

    def test_similarity_properties(self):
        """Similarity should be bounded and have unit diagonal."""
        data = np.random.randn(20, 5)
        distance_matrix = compute_distance_matrix(data)
        similarity = compute_similarity_matrix(distance_matrix, method="rbf")

        assert np.all(similarity >= 0)
        assert np.all(similarity <= 1)
        np.testing.assert_array_almost_equal(np.diag(similarity), 1.0)

    def test_similarity_inverse(self):
        """Inverse similarity should be bounded and have unit diagonal."""
        data = np.random.randn(20, 5)
        distance_matrix = compute_distance_matrix(data)
        similarity = compute_similarity_matrix(distance_matrix, method="inverse")

        assert np.all(similarity >= 0)
        assert np.all(similarity <= 1)
        np.testing.assert_array_almost_equal(np.diag(similarity), 1.0)
