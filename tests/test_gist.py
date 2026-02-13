"""Tests for GIST algorithm internals."""

import numpy as np
import pytest

from gist_sampling.core.gist import gist
from gist_sampling.core.greedy_independent_set import greedy_independent_set
from gist_sampling.distance.metrics import compute_distance_matrix, compute_similarity_matrix
from gist_sampling.utility.base import SubmodularFunction
from gist_sampling.utility.facility_location import FacilityLocationFunction


def _compute_diversity(indices, distance_matrix):
    if len(indices) <= 1:
        return float(np.max(distance_matrix))
    min_dist = float("inf")
    for i, idx_i in enumerate(indices):
        row = distance_matrix[idx_i]
        for idx_j in indices[i + 1 :]:
            dist = row[idx_j]
            if dist < min_dist:
                min_dist = dist
    return min_dist


class WeightedSumUtility(SubmodularFunction):
    """Simple linear utility: g(S) = sum of weights for S."""

    def __init__(self, weights: np.ndarray) -> None:
        super().__init__()
        self._weights = np.asarray(weights, dtype=float)

    def initialize(
        self,
        X: np.ndarray,
        distance_matrix: np.ndarray,
        similarity_matrix: np.ndarray,
    ) -> None:
        self._n = X.shape[0]
        if self._weights.shape[0] != self._n:
            raise ValueError("weights length must match data size")
        self._initialized = True

    def evaluate(self, S: list[int]) -> float:
        self._check_initialized()
        if len(S) == 0:
            return 0.0
        return float(np.sum(self._weights[S]))

    def reset_coverage(self) -> None:
        self._check_initialized()

    def marginal_gains_batch(self, candidates: np.ndarray) -> np.ndarray:
        self._check_initialized()
        if len(candidates) == 0:
            return np.array([])
        return self._weights[candidates]

    def update_coverage(self, v: int) -> None:
        self._check_initialized()


class TestGreedyIndependentSet:
    """Tests for greedy_independent_set function."""

    def test_respects_distance_threshold(self, collinear_data):
        """Selected elements should respect minimum distance constraint."""
        distance_matrix = compute_distance_matrix(collinear_data)
        similarity = compute_similarity_matrix(distance_matrix)
        utility = FacilityLocationFunction()
        utility.initialize(collinear_data, distance_matrix, similarity)

        threshold = 3.0
        result = greedy_independent_set(
            n=len(collinear_data),
            utility_fn=utility,
            distance_threshold=threshold,
            k=10,
            distance_matrix=distance_matrix,
        )

        # Check all pairs have distance >= threshold
        for i, idx_i in enumerate(result):
            for idx_j in result[i + 1 :]:
                assert distance_matrix[idx_i, idx_j] >= threshold


class TestLazyGreedy:
    """Tests for lazy evaluation in greedy_independent_set."""

    def test_selects_highest_gain_element_first(self, collinear_data):
        """First selected element should have the highest initial marginal gain."""
        distance_matrix = compute_distance_matrix(collinear_data)
        similarity = compute_similarity_matrix(distance_matrix)
        utility = FacilityLocationFunction()
        utility.initialize(collinear_data, distance_matrix, similarity)

        result = greedy_independent_set(
            n=len(collinear_data),
            utility_fn=utility,
            distance_threshold=0.0,
            k=1,
            distance_matrix=distance_matrix,
        )

        # Verify first element has the highest marginal gain
        utility.reset_coverage()
        all_gains = utility.marginal_gains_batch(np.arange(len(collinear_data)))
        expected_best = int(np.argmax(all_gains))
        assert result[0] == expected_best

    def test_lazy_respects_distance_threshold(self, collinear_data):
        """Lazy greedy should still respect the distance threshold constraint."""
        distance_matrix = compute_distance_matrix(collinear_data)
        similarity = compute_similarity_matrix(distance_matrix)
        utility = FacilityLocationFunction()
        utility.initialize(collinear_data, distance_matrix, similarity)

        threshold = 2.5
        result = greedy_independent_set(
            n=len(collinear_data),
            utility_fn=utility,
            distance_threshold=threshold,
            k=10,
            distance_matrix=distance_matrix,
        )

        # All pairwise distances must respect the threshold
        for i, idx_i in enumerate(result):
            for idx_j in result[i + 1 :]:
                assert distance_matrix[idx_i, idx_j] >= threshold

    def test_lazy_k_equals_one(self, collinear_data):
        """Should correctly handle k=1 (only batch step, no lazy steps)."""
        distance_matrix = compute_distance_matrix(collinear_data)
        similarity = compute_similarity_matrix(distance_matrix)
        utility = FacilityLocationFunction()
        utility.initialize(collinear_data, distance_matrix, similarity)

        result = greedy_independent_set(
            n=len(collinear_data),
            utility_fn=utility,
            distance_threshold=0.0,
            k=1,
            distance_matrix=distance_matrix,
        )

        assert len(result) == 1
        assert 0 <= result[0] < len(collinear_data)

    def test_lazy_high_threshold_limits_selection(self):
        """Very high threshold should select fewer elements than k."""
        data = np.array([[0.0], [0.1], [0.2], [10.0]], dtype=float)
        distance_matrix = compute_distance_matrix(data)
        similarity = compute_similarity_matrix(distance_matrix)
        utility = FacilityLocationFunction()
        utility.initialize(data, distance_matrix, similarity)

        result = greedy_independent_set(
            n=len(data),
            utility_fn=utility,
            distance_threshold=5.0,
            k=4,
            distance_matrix=distance_matrix,
        )

        # Only 2 elements can be >= 5.0 apart (points 0-2 are too close to each other)
        assert len(result) <= 2

    def test_lazy_diversity_tracking(self, collinear_data):
        """Diversity should be correctly tracked with lazy evaluation."""
        distance_matrix = compute_distance_matrix(collinear_data)
        similarity = compute_similarity_matrix(distance_matrix)
        utility = FacilityLocationFunction()
        utility.initialize(collinear_data, distance_matrix, similarity)

        result, diversity = greedy_independent_set(
            n=len(collinear_data),
            utility_fn=utility,
            distance_threshold=0.0,
            k=3,
            distance_matrix=distance_matrix,
            return_diversity=True,
            d_max=float(np.max(distance_matrix)),
        )

        assert len(result) == 3
        assert diversity >= 0

        # Verify diversity matches actual minimum pairwise distance
        actual_min = float("inf")
        for i, idx_i in enumerate(result):
            for idx_j in result[i + 1 :]:
                actual_min = min(actual_min, distance_matrix[idx_i, idx_j])
        assert diversity == pytest.approx(actual_min)

    def test_lazy_with_clustered_data(self):
        """Lazy greedy should select from multiple clusters."""
        np.random.seed(42)
        cluster1 = np.random.randn(20, 2)
        cluster2 = np.random.randn(20, 2) + np.array([20, 20])
        data = np.vstack([cluster1, cluster2])

        distance_matrix = compute_distance_matrix(data)
        similarity = compute_similarity_matrix(distance_matrix)
        utility = FacilityLocationFunction()
        utility.initialize(data, distance_matrix, similarity)

        result = greedy_independent_set(
            n=len(data),
            utility_fn=utility,
            distance_threshold=0.0,
            k=10,
            distance_matrix=distance_matrix,
        )

        # Should pick from both clusters
        from_c1 = sum(1 for idx in result if idx < 20)
        from_c2 = sum(1 for idx in result if idx >= 20)
        assert from_c1 >= 2
        assert from_c2 >= 2


class TestGistAlgorithm:
    """Tests for the main gist() function."""

    def test_gist_basic(self, collinear_data):
        """GIST should return a valid solution for basic input."""
        distance_matrix = compute_distance_matrix(collinear_data)
        similarity = compute_similarity_matrix(distance_matrix)
        utility = FacilityLocationFunction()
        utility.initialize(collinear_data, distance_matrix, similarity)
        lambda_diversity = 1.0

        indices, objective, diversity = gist(
            X=collinear_data,
            utility_fn=utility,
            k=3,
            epsilon=0.1,
            lambda_diversity=lambda_diversity,
            distance_matrix=distance_matrix,
        )

        assert len(indices) <= 3
        assert objective >= 0
        assert diversity >= 0

    def test_gist_k_zero(self, collinear_data):
        """k=0 should return empty selection with diameter diversity."""
        distance_matrix = compute_distance_matrix(collinear_data)
        similarity = compute_similarity_matrix(distance_matrix)
        utility = FacilityLocationFunction()
        utility.initialize(collinear_data, distance_matrix, similarity)
        lambda_diversity = 1.0
        d_max = float(np.max(distance_matrix))

        indices, objective, diversity = gist(
            X=collinear_data,
            utility_fn=utility,
            k=0,
            epsilon=0.1,
            lambda_diversity=lambda_diversity,
            distance_matrix=distance_matrix,
        )

        assert indices == []
        assert objective == lambda_diversity * d_max
        assert diversity == d_max

    def test_gist_k_ge_n(self, collinear_data):
        """k >= n should return all elements."""
        distance_matrix = compute_distance_matrix(collinear_data)
        similarity = compute_similarity_matrix(distance_matrix)
        utility = FacilityLocationFunction()
        utility.initialize(collinear_data, distance_matrix, similarity)
        lambda_diversity = 1.0

        n = collinear_data.shape[0]
        indices, objective, diversity = gist(
            X=collinear_data,
            utility_fn=utility,
            k=n + 5,
            epsilon=0.1,
            lambda_diversity=lambda_diversity,
            distance_matrix=distance_matrix,
        )

        assert indices == list(range(n))
        expected_diversity = _compute_diversity(indices, distance_matrix)
        expected_objective = utility.evaluate(indices) + lambda_diversity * expected_diversity
        assert objective == expected_objective
        assert diversity == expected_diversity

    def test_gist_identical_points(self):
        """Identical points should yield zero diversity and finite objective."""
        data = np.zeros((5, 2), dtype=float)
        distance_matrix = compute_distance_matrix(data)
        similarity = compute_similarity_matrix(distance_matrix)
        utility = FacilityLocationFunction()
        utility.initialize(data, distance_matrix, similarity)
        lambda_diversity = 1.0

        indices, objective, diversity = gist(
            X=data,
            utility_fn=utility,
            k=3,
            epsilon=0.1,
            lambda_diversity=lambda_diversity,
            distance_matrix=distance_matrix,
        )

        assert len(indices) == 3
        assert objective == data.shape[0] + lambda_diversity * 0.0
        assert diversity == 0.0

    def test_gist_single_point(self):
        """Single-point datasets should select the only element with diameter diversity."""
        data = np.array([[1.0, 2.0]])
        distance_matrix = compute_distance_matrix(data)
        similarity = compute_similarity_matrix(distance_matrix)
        utility = FacilityLocationFunction()
        utility.initialize(data, distance_matrix, similarity)
        lambda_diversity = 1.0
        d_max = float(np.max(distance_matrix))

        indices, objective, diversity = gist(
            X=data,
            utility_fn=utility,
            k=5,
            epsilon=0.1,
            lambda_diversity=lambda_diversity,
            distance_matrix=distance_matrix,
        )

        assert indices == [0]
        assert objective == utility.evaluate(indices) + lambda_diversity * d_max
        assert diversity == d_max

    def test_lambda_diversity_changes_selection(self):
        """Large lambda should favor diversity over utility."""
        data = np.array([[0.0], [0.1], [10.0]])
        weights = np.array([10.0, 9.0, 1.0])
        distance_matrix = compute_distance_matrix(data)
        utility = WeightedSumUtility(weights)
        utility.initialize(data, distance_matrix, distance_matrix)

        indices_low_lambda, _, _ = gist(
            X=data,
            utility_fn=utility,
            k=2,
            epsilon=0.1,
            lambda_diversity=0.0,
            distance_matrix=distance_matrix,
        )

        indices_high_lambda, _, _ = gist(
            X=data,
            utility_fn=utility,
            k=2,
            epsilon=0.1,
            lambda_diversity=100.0,
            distance_matrix=distance_matrix,
        )

        assert set(indices_low_lambda) == {0, 1}
        assert set(indices_high_lambda) == {0, 2}


class TestEarlyTermination:
    """Tests for early termination in the GIST threshold sweep."""

    def test_still_finds_good_solution(self, collinear_data):
        """Early termination should not prevent finding a good solution."""
        distance_matrix = compute_distance_matrix(collinear_data)
        similarity = compute_similarity_matrix(distance_matrix)
        utility = FacilityLocationFunction()
        utility.initialize(collinear_data, distance_matrix, similarity)

        indices, objective, diversity = gist(
            X=collinear_data,
            utility_fn=utility,
            k=3,
            epsilon=0.1,
            lambda_diversity=1.0,
            distance_matrix=distance_matrix,
        )

        assert len(indices) <= 3
        assert len(indices) >= 1
        assert objective > 0
        assert diversity >= 0

    def test_result_valid_with_small_epsilon(self, collinear_data):
        """Small epsilon (many thresholds) should still produce valid results."""
        distance_matrix = compute_distance_matrix(collinear_data)
        similarity = compute_similarity_matrix(distance_matrix)
        utility = FacilityLocationFunction()
        utility.initialize(collinear_data, distance_matrix, similarity)

        # Small epsilon generates many thresholds — early termination should help
        indices, objective, diversity = gist(
            X=collinear_data,
            utility_fn=utility,
            k=4,
            epsilon=0.01,
            lambda_diversity=1.0,
            distance_matrix=distance_matrix,
        )

        assert len(indices) <= 4
        assert len(indices) >= 1
        assert objective > 0

    def test_identical_points_terminates(self):
        """All-identical points should terminate quickly (d_max=0 fast path)."""
        data = np.zeros((10, 2), dtype=float)
        distance_matrix = compute_distance_matrix(data)
        similarity = compute_similarity_matrix(distance_matrix)
        utility = FacilityLocationFunction()
        utility.initialize(data, distance_matrix, similarity)

        indices, objective, diversity = gist(
            X=data,
            utility_fn=utility,
            k=3,
            epsilon=0.1,
            lambda_diversity=1.0,
            distance_matrix=distance_matrix,
        )

        assert len(indices) == 3
        assert diversity == 0.0
