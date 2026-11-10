"""Similarity construction: bandwidth, nearest-neighbour graph and diameter."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from gist_sampling import _kernels as K
from gist_sampling import _similarity
from helpers import SCIPY_METRIC, as_dense, kernel_points, make_blobs


def test_median_distance_is_exact_for_small_inputs(metric):
    X = make_blobs(80, 3, seed=0)
    D = cdist(X, X, metric=SCIPY_METRIC[metric])
    expected = np.median(D[np.triu_indices(80, k=1)])
    got = _similarity.median_distance(K.METRICS[metric], kernel_points(X, metric))
    assert got == pytest.approx(expected, rel=1e-9)


def test_median_distance_is_sampled_for_large_inputs():
    X = make_blobs(2000, 4, seed=1)
    D = cdist(X, X)
    expected = np.median(D[np.triu_indices(2000, k=1)])
    got = _similarity.median_distance(K.EUCLIDEAN, X)
    assert got == pytest.approx(expected, rel=0.02)
    assert got == _similarity.median_distance(K.EUCLIDEAN, X)  # deterministic


def test_median_distance_ignores_duplicates():
    X = np.concatenate([np.zeros((50, 2)), [[3.0, 4.0]]])
    assert _similarity.median_distance(K.EUCLIDEAN, X) == pytest.approx(5.0)
    assert _similarity.median_distance(K.EUCLIDEAN, np.zeros((5, 2))) == 1.0


def test_dense_similarity_finds_the_farthest_pair(metric):
    X = make_blobs(70, 3, seed=2)
    D = cdist(X, X, metric=SCIPY_METRIC[metric])
    _, (u, v, distance) = _similarity.dense_similarity(
        K.METRICS[metric], kernel_points(X, metric), K.RBF, 1.0
    )
    assert distance == pytest.approx(D.max())
    assert D[u, v] == pytest.approx(D.max())


@pytest.mark.parametrize("kind", ["rbf", "inverse"])
def test_sparse_similarity_keeps_nearest_neighbours(metric, kind):
    X = make_blobs(90, 3, seed=3)
    D = cdist(X, X, metric=SCIPY_METRIC[metric])
    full = np.exp(-0.2 * D**2) if kind == "rbf" else 1 / (1 + D)
    n_neighbors = 7

    sim = _similarity.sparse_similarity(
        K.METRICS[metric], kernel_points(X, metric), K.SIMILARITIES[kind], 0.2, n_neighbors, 1
    )
    S = as_dense(sim)

    assert [part.dtype for part in sim] == [np.int64, np.int32, np.float32]
    np.testing.assert_array_equal(S, S.T)
    np.testing.assert_array_equal(S.diagonal(), 1)
    kept = S > 0
    np.testing.assert_allclose(S[kept], full[kept], atol=1e-5)
    # each point keeps at least its nearest neighbours, and itself
    nearest = np.argsort(D, axis=1)[:, : n_neighbors + 1]
    assert kept[np.arange(90)[:, None], nearest].all()
    assert (kept.sum(axis=1) >= n_neighbors + 1).all()
    assert kept.sum() < 90 * 90 / 2


def test_sparse_similarity_from_precomputed_distances():
    X = make_blobs(60, 3, seed=4)
    D = cdist(X, X)
    from_features = _similarity.sparse_similarity(K.EUCLIDEAN, X, K.RBF, 0.1, 5, 1)
    from_distances = _similarity.sparse_similarity(K.PRECOMPUTED, D, K.RBF, 0.1, 5, 1)
    np.testing.assert_allclose(as_dense(from_distances), as_dense(from_features), atol=1e-6)


def test_farthest_pair_is_a_tight_lower_bound(metric):
    X = make_blobs(500, 4, seed=5)
    D = cdist(X, X, metric=SCIPY_METRIC[metric])
    u, v, distance = _similarity.farthest_pair(K.METRICS[metric], kernel_points(X, metric))
    assert D[u, v] == pytest.approx(distance)
    assert 0.9 * D.max() <= distance <= D.max() * (1 + 1e-12)


def test_farthest_pair_of_identical_points():
    assert _similarity.farthest_pair(K.EUCLIDEAN, np.ones((8, 2))) == (0, 0, 0.0)
