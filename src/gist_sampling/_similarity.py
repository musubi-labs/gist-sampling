"""Build the similarity structure that the facility-location utility needs."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from gist_sampling import _kernels as K

# Pairs sampled to estimate the median pairwise distance. The sample is drawn
# with a fixed seed so that selection is a deterministic function of the data.
_MEDIAN_PAIRS = 100_000
_MEDIAN_SEED = 0

# Points visited to estimate the diameter without comparing all pairs.
_DIAMETER_VISITS = 32


def median_distance(metric: int, X: np.ndarray) -> float:
    """Median distance between distinct points, ignoring exact duplicates.

    Exact for small inputs, estimated from random pairs otherwise. Returns 1.0
    when all points coincide.
    """
    n = X.shape[0]
    if n * (n - 1) // 2 <= _MEDIAN_PAIRS:
        left, right = np.triu_indices(n, k=1)
    else:
        rng = np.random.default_rng(_MEDIAN_SEED)
        left = rng.integers(0, n, _MEDIAN_PAIRS)
        right = (left + rng.integers(1, n, _MEDIAN_PAIRS)) % n  # never equal to left
    distances = K.paired_distances(metric, X, left, right)
    distances = distances[distances > 0]
    return float(np.median(distances)) if distances.size else 1.0


def dense_similarity(
    metric: int, X: np.ndarray, kind: int, gamma: float
) -> tuple[np.ndarray, tuple[int, int, float]]:
    """All-pairs similarity matrix and the exact farthest pair ``(u, v, distance)``."""
    sim, farthest, farthest_distance = K.dense_similarity(metric, X, kind, gamma)
    u = int(np.argmax(farthest_distance))
    return sim, (u, int(farthest[u]), float(farthest_distance[u]))


def sparse_similarity(
    metric: int, X: np.ndarray, kind: int, gamma: float, n_neighbors: int, n_jobs: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Similarity restricted to each point's nearest neighbours, as CSR arrays.

    Point ``i`` keeps its similarity to ``j`` when either is among the other's
    ``n_neighbors`` nearest neighbours, so the result is symmetric. Every point
    is fully similar to itself.
    """
    n = X.shape[0]
    if metric == K.PRECOMPUTED:
        search = NearestNeighbors(n_neighbors=n_neighbors, metric="precomputed", n_jobs=n_jobs)
    elif metric == K.COSINE:
        # Rows are unit vectors, where cosine distance is half the squared
        # Euclidean distance. Searching in Euclidean space gives the same
        # neighbours and lets scikit-learn use its fastest code paths.
        search = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", n_jobs=n_jobs)
    else:
        name = {K.EUCLIDEAN: "euclidean", K.MANHATTAN: "manhattan", K.CHEBYSHEV: "chebyshev"}
        search = NearestNeighbors(n_neighbors=n_neighbors, metric=name[metric], n_jobs=n_jobs)
    distances, neighbors = search.fit(X).kneighbors()
    if metric == K.COSINE:
        distances = distances**2 / 2
    values = K.similarities(kind, gamma, distances)

    indptr = np.arange(0, n * n_neighbors + 1, n_neighbors)
    directed = sparse.csr_array((values.ravel(), neighbors.ravel(), indptr), shape=(n, n))
    sim = directed.maximum(directed.T) + sparse.eye_array(n, dtype=np.float32, format="csr")
    sim.sort_indices()
    return sim.indptr.astype(np.int64), sim.indices.astype(np.int32), sim.data


def farthest_pair(metric: int, X: np.ndarray) -> tuple[int, int, float]:
    """Approximate farthest pair ``(u, v, distance)`` without comparing all pairs.

    Visits a few points by farthest-first traversal (each new point is the one
    farthest from all points visited so far, which finds the extremes of the
    data) and returns the largest distance seen from any of them. Each visit
    costs one pass over the data. The result is a lower bound on the diameter,
    never less than half of it and usually within a few percent.
    """
    best = (0, 0, 0.0)
    nearest_visited = np.full(X.shape[0], np.inf)
    u = 0
    for _ in range(_DIAMETER_VISITS):
        distances = K.distances_from(metric, X, u)
        v = int(np.argmax(distances))
        if distances[v] > best[2]:
            best = (u, v, float(distances[v]))
        np.minimum(nearest_visited, distances, out=nearest_visited)
        u = int(np.argmax(nearest_visited))
        if nearest_visited[u] == 0:  # every distinct point has been visited
            break
    return best
