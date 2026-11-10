"""The GIST driver, checked against the reference transcription of the paper."""

from __future__ import annotations

import numpy as np
import pytest

import reference
from gist_sampling import _kernels as K
from gist_sampling import _similarity
from gist_sampling._gist import gist, thresholds
from helpers import kernel_points, make_blobs, reference_matrices


def run_gist(X, metric, k, lambda_diversity, epsilon, gamma):
    """GIST on all pairwise similarities, as the selector runs it."""
    code = K.METRICS[metric]
    points = kernel_points(X, metric)
    sim, farthest_pair = _similarity.dense_similarity(code, points, K.RBF, gamma)
    return gist(
        code,
        points,
        sim,
        k,
        lambda_diversity=lambda_diversity,
        epsilon=epsilon,
        farthest_pair=farthest_pair,
    )


@pytest.mark.parametrize("epsilon", [1.0, 0.5, 0.1, 0.03])
def test_thresholds_match_the_paper(epsilon):
    np.testing.assert_allclose(thresholds(7.0, epsilon), reference.thresholds(7.0, epsilon))


def test_thresholds_span_the_diameter():
    grid = thresholds(10.0, 0.1)
    assert grid[0] == pytest.approx(0.5)
    assert grid[-1] <= 10.0 < grid[-1] * 1.1
    np.testing.assert_allclose(grid[1:] / grid[:-1], 1.1)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("lambda_diversity", [0.3, 1.0, 30.0, 1e4])
def test_gist_matches_reference(metric, seed, lambda_diversity):
    rng = np.random.default_rng(seed)
    n, d, k = int(rng.integers(30, 150)), int(rng.integers(2, 6)), int(rng.integers(2, 25))
    epsilon = float(rng.choice([0.05, 0.1, 0.3]))
    X = make_blobs(n, d, seed=seed)
    D, S, gamma = reference_matrices(X, metric)

    expected, expected_objective = reference.gist(D, S, k, lambda_diversity, epsilon)
    best = run_gist(X, metric, k, lambda_diversity, epsilon, gamma)

    assert best.indices.tolist() == expected
    assert best.objective(lambda_diversity) == pytest.approx(expected_objective, rel=1e-5)
    assert best.utility == pytest.approx(reference.utility(S, expected), rel=1e-5)
    assert best.diversity == pytest.approx(reference.diversity(D, expected), rel=1e-9)


@pytest.mark.parametrize("seed", range(10))
def test_zero_lambda_is_plain_greedy(seed):
    X = make_blobs(100, 2, seed=seed)
    D, S, gamma = reference_matrices(X)
    best = run_gist(X, "euclidean", 10, 0.0, 0.1, gamma)
    assert best.indices.tolist() == reference.greedy_independent_set(D, S, 0.0, 10)
    assert best.threshold == 0.0


def test_zero_lambda_fills_the_budget_with_duplicates():
    X = np.repeat([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]], 5, axis=0)
    best = run_gist(X, "euclidean", 5, 0.0, 0.1, gamma=0.1)
    assert len(best.indices) == 5
    assert best.diversity == 0.0


def test_many_thresholds_are_run_in_batches():
    X = make_blobs(60, 2, seed=7)
    D, S, gamma = reference_matrices(X)
    expected, objective = reference.gist(D, S, 6, 2.0, 0.01)  # several hundred thresholds
    best = run_gist(X, "euclidean", 6, 2.0, 0.01, gamma)
    assert best.indices.tolist() == expected
    assert best.objective(2.0) == pytest.approx(objective, rel=1e-5)


def test_large_lambda_trades_size_for_distance():
    X = make_blobs(100, 3, seed=2)
    D, _, gamma = reference_matrices(X)
    small = run_gist(X, "euclidean", 10, 0.0, 0.1, gamma)
    large = run_gist(X, "euclidean", 10, 1e6, 0.1, gamma)

    assert large.diversity > small.diversity
    assert len(large.indices) < len(small.indices)
    assert large.utility < small.utility
    # With diversity all that matters, nothing beats the diameter, which the
    # farthest pair attains and which a single point is defined to have.
    assert large.diversity == pytest.approx(D.max())
    assert len(large.indices) <= 2


def test_selection_respects_its_threshold():
    X = make_blobs(200, 2, seed=3)
    D, _, gamma = reference_matrices(X)
    best = run_gist(X, "euclidean", 20, 200.0, 0.1, gamma)
    assert best.threshold > 0
    assert reference.diversity(D, best.indices.tolist()) >= best.threshold


def test_budget_of_one_selects_the_most_central_point():
    X = make_blobs(60, 2, seed=4)
    D, S, gamma = reference_matrices(X)
    best = run_gist(X, "euclidean", 1, 5.0, 0.1, gamma)
    assert best.indices.tolist() == [int(np.argmax(S.sum(axis=1)))]
    assert best.diversity == pytest.approx(D.max())  # by convention, the diameter


def test_budget_beyond_the_data_is_clipped():
    X = make_blobs(25, 2, seed=5)
    D, _, gamma = reference_matrices(X)
    best = run_gist(X, "euclidean", 1000, 0.0, 0.1, gamma)
    assert sorted(best.indices.tolist()) == list(range(25))
    assert best.utility == pytest.approx(25.0)
    assert best.diversity == pytest.approx(D[np.triu_indices(25, k=1)].min())


def test_identical_points():
    X = np.ones((10, 3))
    best = run_gist(X, "euclidean", 4, 1.0, 0.1, gamma=1.0)
    assert best.indices.tolist() == [0, 1, 2, 3]
    assert best.diversity == 0.0
    assert best.utility == pytest.approx(10.0)


def test_single_point():
    best = run_gist(np.zeros((1, 2)), "euclidean", 3, 1.0, 0.1, gamma=1.0)
    assert best.indices.tolist() == [0]
    assert best.utility == pytest.approx(1.0)
    assert best.diversity == 0.0


def test_duplicates_are_dropped_when_diversity_matters():
    base = make_blobs(15, 2, seed=6)
    X = np.concatenate([base, base])  # every point twice
    _, _, gamma = reference_matrices(base)
    best = run_gist(X, "euclidean", 30, 1.0, 0.1, gamma)
    assert best.diversity > 0
    assert len({tuple(X[i]) for i in best.indices}) == len(best.indices)
