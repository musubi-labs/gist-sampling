"""The public interface: GISTSelector and gist_sample."""

from __future__ import annotations

import numba
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.spatial.distance import cdist
from sklearn.base import clone

import reference
from gist_sampling import GISTSelector, _selector, gist_sample
from helpers import SCIPY_METRIC, make_blobs, reference_matrices

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


def test_fit_reports_a_consistent_selection():
    X = make_blobs(200, 3, seed=0)
    selector = GISTSelector(n_samples=6, lambda_diversity=2.0).fit(X)
    chosen = selector.selected_indices_
    D, S, _ = reference_matrices(X, gamma=selector.gamma_)

    assert chosen.dtype == np.intp
    assert len(chosen) == len(set(chosen.tolist())) == 6
    assert selector.utility_ == pytest.approx(reference.utility(S, chosen.tolist()), rel=1e-5)
    assert selector.diversity_ == pytest.approx(reference.diversity(D, chosen.tolist()))
    assert selector.objective_value_ == pytest.approx(
        selector.utility_ + 2.0 * selector.diversity_ / selector.diameter_
    )
    assert selector.diversity_ >= selector.threshold_
    assert selector.diameter_ == pytest.approx(D.max())
    assert selector.n_neighbors_ is None
    assert selector.n_samples_in_ == 200


@pytest.mark.parametrize("seed", range(3))
def test_fit_matches_reference_gist(metric, seed):
    X = make_blobs(90, 3, seed=seed)
    selector = GISTSelector(n_samples=9, metric=metric, lambda_diversity=5.0, epsilon=0.2).fit(X)
    D, S, _ = reference_matrices(X, metric, gamma=selector.gamma_)

    expected, objective = reference.gist(D, S, 9, lambda_diversity=5.0 / D.max(), epsilon=0.2)

    assert selector.selected_indices_.tolist() == expected
    assert selector.objective_value_ == pytest.approx(objective, rel=1e-5)


@pytest.mark.parametrize("lambda_diversity", [1.0, 40.0])
def test_selection_does_not_depend_on_the_units_of_the_data(lambda_diversity):
    X = make_blobs(120, 3, seed=0)
    small = GISTSelector(n_samples=10, lambda_diversity=lambda_diversity).fit(X)
    large = GISTSelector(n_samples=10, lambda_diversity=lambda_diversity).fit(X * 1000)

    D = cdist(X, X)
    median = np.median(D[np.triu_indices(120, k=1)])
    assert small.gamma_ == pytest.approx(1 / median**2)
    assert large.gamma_ == pytest.approx(small.gamma_ / 1e6)
    assert large.selected_indices_.tolist() == small.selected_indices_.tolist()
    assert large.objective_value_ == pytest.approx(small.objective_value_, rel=1e-5)
    assert large.diversity_ == pytest.approx(small.diversity_ * 1000)


def test_one_pick_per_cluster(two_clusters):
    chosen = GISTSelector(n_samples=2).fit(two_clusters).selected_indices_
    assert sorted(chosen // 20) == [0, 1]


def test_rare_clusters_are_not_drowned_out():
    """One cluster holds 60% of the points; five picks still cover all five."""
    rng = np.random.default_rng(0)
    centers = rng.normal(scale=10, size=(5, 8))
    labels = np.repeat(np.arange(5), [300, 50, 50, 50, 50])
    X = centers[labels] + rng.normal(size=(500, 8))

    chosen = GISTSelector(n_samples=5).fit(X).selected_indices_

    assert sorted(labels[chosen]) == [0, 1, 2, 3, 4]


def test_more_diversity_weight_spreads_the_selection():
    X = make_blobs(300, 2, seed=1)
    plain = GISTSelector(n_samples=30, lambda_diversity=0).fit(X)
    spread = GISTSelector(n_samples=30, lambda_diversity=100).fit(X)

    assert plain.threshold_ == 0
    assert spread.diversity_ > plain.diversity_
    assert spread.utility_ <= plain.utility_


def test_returns_fewer_points_when_more_would_be_redundant():
    X = np.repeat(make_blobs(10, 2, seed=2), 5, axis=0)  # ten distinct points, five copies each
    selector = GISTSelector(n_samples=30).fit(X)
    chosen = selector.selected_indices_

    assert len(chosen) <= 10
    assert len({tuple(X[i]) for i in chosen}) == len(chosen)  # no two copies of a point
    assert selector.diversity_ > 0

    # Without a diversity term every extra point is welcome, redundant or not.
    assert len(GISTSelector(n_samples=30, lambda_diversity=0).fit(X).selected_indices_) == 30


def test_budget_larger_than_the_data():
    X = make_blobs(12, 2, seed=3)
    selector = GISTSelector(n_samples=100, lambda_diversity=0).fit(X)
    assert sorted(selector.selected_indices_.tolist()) == list(range(12))


def test_single_point():
    selector = GISTSelector(n_samples=3).fit(np.array([[1.0, 2.0]]))
    assert selector.selected_indices_.tolist() == [0]
    assert selector.diversity_ == selector.diameter_ == 0.0


def test_inverse_similarity(two_clusters):
    selector = GISTSelector(n_samples=4, similarity="inverse").fit(two_clusters)
    D = cdist(two_clusters, two_clusters)
    S = (1 / (1 + D)).astype(np.float32)
    assert selector.gamma_ is None
    assert selector.selected_indices_.tolist() == reference.gist(D, S, 4, 1 / D.max(), 0.1)[0]


def test_explicit_gamma(two_clusters):
    selector = GISTSelector(n_samples=4, gamma=0.5).fit(two_clusters)
    D, S, _ = reference_matrices(two_clusters, gamma=0.5)
    assert selector.gamma_ == 0.5
    assert selector.selected_indices_.tolist() == reference.gist(D, S, 4, 1 / D.max(), 0.1)[0]


# --------------------------------------------------------------------------
# Determinism and threads
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_neighbors", [None, 10])
def test_selection_is_deterministic_and_thread_independent(n_neighbors):
    X = make_blobs(400, 4, seed=4)
    runs = [
        GISTSelector(n_samples=25, n_neighbors=n_neighbors, lambda_diversity=20, n_jobs=n_jobs)
        .fit(X)
        .selected_indices_.tolist()
        for n_jobs in (-1, 1, 2, -1)
    ]
    assert all(run == runs[0] for run in runs)


def test_concurrent_fits_take_turns():
    from concurrent.futures import ThreadPoolExecutor

    X = make_blobs(300, 3, seed=12)
    expected = GISTSelector(n_samples=20).fit(X).selected_indices_.tolist()
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda _: GISTSelector(n_samples=20, n_jobs=2).fit(X), range(12)))
    assert all(result.selected_indices_.tolist() == expected for result in results)


def test_thread_count_is_restored():
    before = numba.get_num_threads()
    GISTSelector(n_samples=5, n_jobs=1).fit(make_blobs(50, 2, seed=5))
    assert numba.get_num_threads() == before


# --------------------------------------------------------------------------
# Nearest-neighbour approximation
# --------------------------------------------------------------------------


def test_auto_keeps_all_pairs_for_small_data(two_clusters):
    assert GISTSelector(n_samples=3).fit(two_clusters).n_neighbors_ is None


def test_auto_switches_to_neighbours_for_large_data(monkeypatch):
    monkeypatch.setattr(_selector, "_AUTO_EXACT_LIMIT", 100)
    monkeypatch.setattr(_selector, "_AUTO_NEIGHBORS", 15)
    assert GISTSelector(n_samples=3).fit(make_blobs(100, 2, seed=6)).n_neighbors_ is None
    assert GISTSelector(n_samples=3).fit(make_blobs(101, 2, seed=6)).n_neighbors_ == 15


def test_enough_neighbours_means_all_pairs(two_clusters):
    assert GISTSelector(n_samples=3, n_neighbors=38).fit(two_clusters).n_neighbors_ == 38
    assert GISTSelector(n_samples=3, n_neighbors=39).fit(two_clusters).n_neighbors_ is None
    assert GISTSelector(n_samples=3, n_neighbors=500).fit(two_clusters).n_neighbors_ is None


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_neighbour_approximation_is_close_to_exact(metric, dtype):
    X = make_blobs(1500, 6, seed=7, n_clusters=8).astype(dtype)
    exact = GISTSelector(n_samples=40, metric=metric, n_neighbors=None).fit(X)
    approx = GISTSelector(n_samples=40, metric=metric, n_neighbors=50, gamma=exact.gamma_).fit(X)

    D, S, _ = reference_matrices(X.astype(np.float64), metric, gamma=exact.gamma_)
    true_utility = reference.utility(S, approx.selected_indices_.tolist())

    assert approx.n_neighbors_ == 50
    assert approx.gamma_ == exact.gamma_
    assert true_utility >= 0.95 * exact.utility_
    assert approx.diversity_ == pytest.approx(
        reference.diversity(D, approx.selected_indices_.tolist()), rel=1e-4
    )
    assert 0.9 * exact.diameter_ <= approx.diameter_ <= exact.diameter_ * (1 + 1e-6)


def test_default_gamma_does_not_depend_on_the_approximation():
    X = make_blobs(600, 3, seed=8)
    exact = GISTSelector(n_samples=10, n_neighbors=None).fit(X)
    approx = GISTSelector(n_samples=10, n_neighbors=20).fit(X)
    assert approx.gamma_ == exact.gamma_


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def test_precomputed_distances_give_the_same_selection(metric):
    X = make_blobs(80, 3, seed=9)
    D = cdist(X, X, metric=SCIPY_METRIC[metric])
    from_features = GISTSelector(n_samples=8, metric=metric, lambda_diversity=3).fit(X)
    from_distances = GISTSelector(n_samples=8, metric="precomputed", lambda_diversity=3).fit(D)

    assert from_distances.selected_indices_.tolist() == from_features.selected_indices_.tolist()
    assert from_distances.gamma_ == pytest.approx(from_features.gamma_)
    assert from_distances.diversity_ == pytest.approx(from_features.diversity_)


def test_precomputed_distances_with_neighbours():
    X = make_blobs(200, 3, seed=10)
    from_features = GISTSelector(n_samples=8, n_neighbors=12).fit(X)
    from_distances = GISTSelector(n_samples=8, n_neighbors=12, metric="precomputed").fit(
        cdist(X, X)
    )
    assert from_distances.selected_indices_.tolist() == from_features.selected_indices_.tolist()


def test_precomputed_distances_may_carry_rounding_error():
    # Distances from the Gram matrix in float32, as GPU libraries compute them:
    # the diagonal is not exactly zero and the matrix not exactly symmetric.
    X = (make_blobs(300, 64, seed=11) + 5).astype(np.float32)
    squares = (X * X).sum(axis=1)
    D = np.sqrt(np.maximum(squares[:, None] + squares[None, :] - 2 * X @ X.T, 0))
    assert D.dtype == np.float32
    assert D.diagonal().max() > 0

    from_features = GISTSelector(n_samples=8).fit(X)
    from_distances = GISTSelector(n_samples=8, metric="precomputed").fit(D)
    assert from_distances.selected_indices_.tolist() == from_features.selected_indices_.tolist()


@pytest.mark.parametrize("lambda_diversity", [0.0, 1.0])
def test_a_flood_of_duplicates_is_selected_once_with_neighbours(lambda_diversity):
    # More exact copies of one row than a point has neighbours.
    rng = np.random.default_rng(12)
    X = np.concatenate([rng.normal(size=(400, 6)), np.tile(rng.normal(size=6), (200, 1))])
    order = rng.permutation(len(X))
    X, is_copy = X[order], order >= 400

    selector = GISTSelector(n_samples=30, n_neighbors=20, lambda_diversity=lambda_diversity)
    chosen = selector.fit(X).selected_indices_
    assert len(chosen) == 30
    assert is_copy[chosen].sum() == 1


@pytest.mark.parametrize(
    "convert",
    [
        lambda X: X.tolist(),
        lambda X: X.astype(np.float32),
        lambda X: np.asfortranarray(X),
        lambda X: X[::-1][::-1],
        lambda X: pd.DataFrame(X, columns=["a", "b"]),
    ],
    ids=["list", "float32", "fortran", "view", "dataframe"],
)
def test_input_containers(two_clusters, convert):
    expected = GISTSelector(n_samples=5).fit(two_clusters).selected_indices_.tolist()
    assert (
        GISTSelector(n_samples=5).fit(convert(two_clusters)).selected_indices_.tolist() == expected
    )


def test_integer_and_one_dimensional_input():
    values = np.array([0, 1, 2, 50, 51, 52, 100, 101, 102])
    selector = GISTSelector(n_samples=3).fit(values)
    assert sorted(selector.selected_indices_ // 3) == [0, 1, 2]
    assert selector.transform(values).shape == (3,)


def test_input_is_not_modified():
    X = make_blobs(50, 3, seed=11)
    original = X.copy()
    GISTSelector(n_samples=5, metric="cosine").fit(X)
    np.testing.assert_array_equal(X, original)


def test_dataframe_uses_numeric_columns_and_keeps_the_rest(two_clusters):
    frame = pd.DataFrame(two_clusters, columns=["x", "y"], index=[f"row{i}" for i in range(40)])
    frame["label"] = ["a", "b"] * 20
    selector = GISTSelector(n_samples=4)

    sampled = selector.fit_transform(frame)

    expected = GISTSelector(n_samples=4).fit(two_clusters).selected_indices_
    assert selector.selected_indices_.tolist() == expected.tolist()
    assert isinstance(sampled, pd.DataFrame)
    assert list(sampled.columns) == ["x", "y", "label"]
    assert list(sampled.index) == [f"row{i}" for i in expected]


def test_dataframe_columns_restrict_the_features(two_clusters):
    frame = pd.DataFrame(two_clusters, columns=["x", "y"])
    frame["noise"] = np.random.default_rng(0).normal(scale=100, size=40)
    frame["broken"] = np.nan

    selector = GISTSelector(n_samples=4, columns=["x", "y"]).fit(frame)

    expected = GISTSelector(n_samples=4).fit(two_clusters).selected_indices_
    assert selector.selected_indices_.tolist() == expected.tolist()


def test_transform_subsets_anything_aligned_with_the_data(two_clusters):
    selector = GISTSelector(n_samples=4).fit(two_clusters)
    chosen = selector.selected_indices_.tolist()
    labels = [f"label{i}" for i in range(40)]

    assert selector.transform(labels) == [labels[i] for i in chosen]
    assert selector.transform(tuple(labels)) == [labels[i] for i in chosen]
    np.testing.assert_array_equal(selector.transform(np.arange(40)), chosen)
    assert selector.transform(pd.Series(labels)).tolist() == [labels[i] for i in chosen]
    np.testing.assert_array_equal(selector.transform(two_clusters), two_clusters[chosen])


def test_gist_sample(two_clusters):
    selector = GISTSelector(n_samples=6, metric="manhattan", lambda_diversity=3).fit(two_clusters)
    sampled = gist_sample(two_clusters, 6, metric="manhattan", lambda_diversity=3)
    np.testing.assert_array_equal(sampled, two_clusters[selector.selected_indices_])


# --------------------------------------------------------------------------
# scikit-learn conventions
# --------------------------------------------------------------------------


def test_parameters_follow_scikit_learn_conventions(two_clusters):
    selector = GISTSelector(n_samples=4, metric="cosine")
    assert selector.get_params()["metric"] == "cosine"
    assert repr(selector) == "GISTSelector(metric='cosine', n_samples=4)"

    copy = clone(selector).set_params(n_samples=7)
    assert (copy.n_samples, copy.metric) == (7, "cosine")
    assert len(copy.fit(two_clusters).selected_indices_) == 7
    assert not hasattr(selector, "selected_indices_")


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


def test_transform_before_fit(two_clusters):
    with pytest.raises(RuntimeError, match="not fitted"):
        GISTSelector(n_samples=3).transform(two_clusters)


def test_transform_with_misaligned_data(two_clusters):
    selector = GISTSelector(n_samples=3).fit(two_clusters)
    with pytest.raises(ValueError, match="39 rows but the selector was fitted on 40"):
        selector.transform(two_clusters[:39])


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"n_samples": 0}, "n_samples"),
        ({"n_samples": -3}, "n_samples"),
        ({"n_samples": 2.5}, "n_samples"),
        ({"n_samples": True}, "n_samples"),
        ({"metric": "minkowski"}, "metric"),
        ({"lambda_diversity": -1}, "lambda_diversity"),
        ({"lambda_diversity": float("nan")}, "lambda_diversity"),
        ({"lambda_diversity": float("inf")}, "lambda_diversity"),
        ({"epsilon": 0}, "epsilon"),
        ({"epsilon": 1e-9}, "epsilon"),
        ({"epsilon": 1.5}, "epsilon"),
        ({"epsilon": "small"}, "epsilon"),
        ({"similarity": "linear"}, "similarity"),
        ({"gamma": 0}, "gamma"),
        ({"gamma": -1.0}, "gamma"),
        ({"gamma": float("inf")}, "gamma"),
        ({"n_neighbors": 0}, "n_neighbors"),
        ({"n_neighbors": 2.5}, "n_neighbors"),
        ({"n_neighbors": "all"}, "n_neighbors"),
        ({"n_jobs": 0}, "n_jobs"),
        ({"n_jobs": -2}, "n_jobs"),
        ({"n_jobs": 1.5}, "n_jobs"),
    ],
)
def test_invalid_parameters(two_clusters, params, message):
    with pytest.raises(ValueError, match=message):
        GISTSelector(**{"n_samples": 3, **params}).fit(two_clusters)


@pytest.mark.parametrize(
    ("X", "error", "message"),
    [
        (np.array([[1.0, np.nan], [0.0, 1.0]]), ValueError, "NaN or infinity"),
        (np.array([[1.0, np.inf], [0.0, 1.0]]), ValueError, "NaN or infinity"),
        (np.empty((0, 3)), ValueError, "empty"),
        (np.empty((3, 0)), ValueError, "empty"),
        (np.zeros((2, 2, 2)), ValueError, "1-D or 2-D"),
        (np.array(5.0), ValueError, "1-D or 2-D"),
        (np.array([["a", "b"], ["c", "d"]]), ValueError, "numeric"),
        (np.array([[True, False], [False, True]]), ValueError, "numeric"),
        (sparse.eye_array(4, format="csr"), TypeError, "Sparse input"),
        (pd.DataFrame({"a": ["x", "y"]}), ValueError, "no numeric columns"),
    ],
    ids=[
        "nan",
        "inf",
        "no-rows",
        "no-columns",
        "3d",
        "scalar",
        "strings",
        "bool",
        "sparse",
        "text",
    ],
)
def test_invalid_data(X, error, message):
    with pytest.raises(error, match=message):
        GISTSelector(n_samples=1).fit(X)


def test_invalid_columns(two_clusters):
    frame = pd.DataFrame(two_clusters, columns=["x", "y"]).assign(label="a")
    with pytest.raises(ValueError, match=r"not found in X: \['z'\]"):
        GISTSelector(n_samples=2, columns=["x", "z"]).fit(frame)
    with pytest.raises(ValueError, match="columns is empty"):
        GISTSelector(n_samples=2, columns=[]).fit(frame)
    with pytest.raises(ValueError, match="numeric"):
        GISTSelector(n_samples=2, columns=["x", "label"]).fit(frame)
    with pytest.raises(TypeError, match="DataFrame"):
        GISTSelector(n_samples=2, columns=["x"]).fit(two_clusters)


@pytest.mark.parametrize(
    ("scale", "dtype", "metric"),
    [
        (1e20, np.float32, "euclidean"),
        (1e-30, np.float32, "euclidean"),
        (1e200, np.float64, "euclidean"),
        (1e-200, np.float64, "manhattan"),
    ],
)
def test_distances_outside_floating_point_range_are_rejected(scale, dtype, metric):
    X = (np.random.default_rng(0).normal(size=(50, 3)) * scale).astype(dtype)
    with pytest.raises(ValueError, match="overflow or underflow"):
        GISTSelector(n_samples=5, metric=metric).fit(X)


@pytest.mark.parametrize("scale", [1e-12, 1e12])
def test_large_and_small_units_are_fine(scale):
    X = make_blobs(80, 3, seed=13)
    expected = GISTSelector(n_samples=8).fit(X).selected_indices_.tolist()
    for dtype in (np.float32, np.float64):
        scaled = (X * scale).astype(dtype)
        assert GISTSelector(n_samples=8).fit(scaled).selected_indices_.tolist() == expected


@pytest.mark.parametrize("scale", [1e-30, 1e-12, 1e12, 1e30])
def test_cosine_ignores_the_length_of_the_rows(scale):
    X = make_blobs(80, 3, seed=13).astype(np.float32)
    expected = GISTSelector(n_samples=8, metric="cosine").fit(X).selected_indices_.tolist()
    scaled = X * np.float32(scale)
    selector = GISTSelector(n_samples=8, metric="cosine").fit(scaled)
    assert selector.selected_indices_.tolist() == expected


def test_identical_points_are_not_a_scale_error():
    selector = GISTSelector(n_samples=3).fit(np.full((6, 2), 7.0))
    assert selector.selected_indices_.tolist() == [0, 1, 2]
    assert selector.diameter_ == 0.0


def test_single_column_name_and_nullable_columns(two_clusters):
    frame = pd.DataFrame(two_clusters, columns=["x", "y"])
    expected = GISTSelector(n_samples=3).fit(two_clusters[:, 0]).selected_indices_.tolist()
    assert GISTSelector(n_samples=3, columns="x").fit(frame).selected_indices_.tolist() == expected

    nullable = frame.astype("Float64")
    expected = GISTSelector(n_samples=3).fit(two_clusters).selected_indices_.tolist()
    assert GISTSelector(n_samples=3).fit(nullable).selected_indices_.tolist() == expected
    nullable.iloc[0, 0] = pd.NA
    with pytest.raises(ValueError, match="NaN or infinity"):
        GISTSelector(n_samples=3).fit(nullable)


def test_cosine_rejects_zero_vectors():
    X = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="all-zero rows"):
        GISTSelector(n_samples=2, metric="cosine").fit(X)


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (lambda D: D[:, :-1], "square"),
        (lambda D: -D, "non-negative"),
        (lambda D: D + np.eye(len(D)), "zero diagonal"),
        (lambda D: np.triu(D), "symmetric"),
    ],
    ids=["rectangular", "negative", "diagonal", "asymmetric"],
)
def test_invalid_distance_matrix(two_clusters, corrupt, message):
    D = cdist(two_clusters, two_clusters)
    with pytest.raises(ValueError, match=message):
        GISTSelector(n_samples=2, metric="precomputed").fit(corrupt(D))
