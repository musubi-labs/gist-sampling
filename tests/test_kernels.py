"""The compiled kernels, checked against NumPy/SciPy and the reference greedy."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import cdist

import reference
from gist_sampling import _kernels as K
from helpers import (
    SCIPY_METRIC,
    as_csr,
    greedy,
    kernel_points,
    make_blobs,
    reference_matrices,
)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distances_match_scipy(metric, dtype):
    X = make_blobs(60, 5, seed=0).astype(dtype)
    points = kernel_points(X, metric)
    code = K.METRICS[metric]

    got = np.array([K.distances_from(code, points, u) for u in range(len(X))])

    assert got.dtype == dtype
    tolerance = 1e-5 if dtype == np.float32 else 1e-12
    expected = cdist(X.astype(np.float64), X.astype(np.float64), metric=SCIPY_METRIC[metric])
    np.testing.assert_allclose(got, expected, rtol=tolerance, atol=tolerance * expected.max())
    assert (got >= 0).all()


def test_precomputed_distances_are_read_from_the_matrix():
    D = cdist(*[make_blobs(30, 3, seed=1)] * 2)
    np.testing.assert_array_equal(K.distances_from(K.PRECOMPUTED, D, 7), D[7])


def test_paired_distances(metric):
    X = make_blobs(40, 4, seed=2)
    left, right = np.array([0, 5, 9, 39]), np.array([1, 5, 30, 0])
    got = K.paired_distances(K.METRICS[metric], kernel_points(X, metric), left, right)
    expected = cdist(X, X, metric=SCIPY_METRIC[metric])[left, right]
    np.testing.assert_allclose(got, expected, atol=1e-12)


@pytest.mark.parametrize("kind", ["rbf", "inverse"])
def test_dense_similarity(metric, kind):
    X = make_blobs(50, 4, seed=3)
    D = cdist(X, X, metric=SCIPY_METRIC[metric])
    gamma = 0.3

    sim, farthest, farthest_distance = K.dense_similarity(
        K.METRICS[metric], kernel_points(X, metric), K.SIMILARITIES[kind], gamma
    )

    expected = np.exp(-gamma * D**2) if kind == "rbf" else 1 / (1 + D)
    assert sim.dtype == np.float32
    np.testing.assert_allclose(sim, expected, atol=1e-6)
    np.testing.assert_array_equal(sim.diagonal(), 1)
    np.testing.assert_allclose(farthest_distance, D.max(axis=1), atol=1e-12)
    np.testing.assert_allclose(D[np.arange(50), farthest], D.max(axis=1), atol=1e-12)


@pytest.mark.parametrize("representation", ["dense", "sparse"])
def test_gains_and_coverage(representation):
    rng = np.random.default_rng(4)
    S = rng.random((30, 30)).astype(np.float32)
    S = np.maximum(S, S.T) * (rng.random((30, 30)) < 0.5)  # symmetric with zeros
    S = np.maximum(S, S.T)
    np.fill_diagonal(S, 1)
    sim = S if representation == "dense" else as_csr(S)
    subset = np.array([3, 17, 8])

    covered = K.coverage(sim, 30, subset)
    np.testing.assert_array_equal(covered, S[subset].max(axis=0))

    expected = np.maximum(S - covered, 0).sum(axis=1, dtype=np.float64)
    np.testing.assert_allclose(K.gains(sim, covered), expected, rtol=1e-6)


def test_first_step_orders_candidates_by_gain_then_index():
    S = np.eye(6, dtype=np.float32)
    S[0, 1] = S[1, 0] = 0.5  # points 0 and 1 gain the most and tie; the rest tie below
    first, first_gain, order, bounds = K.first_step(S, 6)

    assert first == 0
    assert first_gain == 1.5
    assert order.tolist() == [2, 3, 4, 5, 1]
    np.testing.assert_array_equal(bounds[1:], [0.5, 1, 1, 1, 1])


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("representation", ["dense", "sparse"])
def test_greedy_matches_reference(metric, seed, dtype, representation):
    X = make_blobs(120, 3, seed=seed).astype(dtype)
    D, S, _ = reference_matrices(X.astype(np.float64), metric)
    sim = S if representation == "dense" else as_csr(S)
    points = kernel_points(X, metric)
    k = 15

    for fraction in [0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]:
        threshold = fraction * D.max()
        expected = reference.greedy_independent_set(D, S, threshold, k)

        selected, utility, diversity = greedy(metric, points, sim, threshold, k)

        assert selected == expected, f"threshold = {fraction} * diameter"
        assert utility == pytest.approx(reference.utility(S, expected), rel=1e-5)
        if len(expected) > 1:
            assert diversity == pytest.approx(reference.diversity(D, expected), rel=1e-5)
            assert diversity >= threshold
        else:
            assert diversity == np.inf


def test_greedy_with_truncated_similarity_matches_reference():
    """A genuinely sparse similarity (most pairs dropped) still gives exact greedy."""
    X = make_blobs(150, 2, seed=5)
    D, S, _ = reference_matrices(X)
    S[S < 0.6] = 0
    for threshold in [0, 0.05 * D.max(), 0.3 * D.max()]:
        expected = reference.greedy_independent_set(D, S, threshold, 40)
        selected, _, _ = greedy("euclidean", X, as_csr(S), threshold, 40)
        assert selected == expected


def test_greedy_breaks_ties_by_lowest_index():
    X = np.repeat(make_blobs(12, 2, seed=6), 3, axis=0)  # every point appears three times
    D, S, _ = reference_matrices(X)
    for threshold in [0, 0.1 * D.max()]:
        expected = reference.greedy_independent_set(D, S, threshold, 20)
        selected, _, _ = greedy("euclidean", X, S, threshold, 20)
        assert selected == expected


def test_greedy_stops_when_no_point_is_far_enough():
    X = make_blobs(80, 2, seed=7)
    D, S, _ = reference_matrices(X)
    selected, _, diversity = greedy("euclidean", X, S, 0.5 * D.max(), 80)
    assert 1 < len(selected) < 80
    assert diversity >= 0.5 * D.max()
    # maximal: every other point is too close to something selected
    assert (D[selected].min(axis=0) < 0.5 * D.max()).all()


def test_sweep_runs_each_threshold_independently():
    X = make_blobs(100, 3, seed=8)
    D, S, _ = reference_matrices(X)
    thresholds = np.array([0, 0.1, 0.3, 0.6]) * D.max()
    start = K.first_step(S, 100)

    selected, sizes, utilities, diversities = K.sweep(K.EUCLIDEAN, X, S, thresholds, 12, start)

    for t, threshold in enumerate(thresholds):
        expected, utility, diversity = greedy("euclidean", X, S, threshold, 12)
        assert selected[t, : sizes[t]].tolist() == expected
        assert utilities[t] == utility
        assert diversities[t] == diversity
