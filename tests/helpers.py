"""Data generators and conversions shared by the tests."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist

from gist_sampling import _kernels as K

SCIPY_METRIC = {
    "euclidean": "euclidean",
    "cosine": "cosine",
    "manhattan": "cityblock",
    "chebyshev": "chebyshev",
}


def make_blobs(n: int, d: int, seed: int, n_clusters: int = 4, spread: float = 4.0) -> np.ndarray:
    """Gaussian clusters of unequal size."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=spread, size=(n_clusters, d))
    labels = rng.choice(n_clusters, size=n, p=rng.dirichlet(np.ones(n_clusters)))
    return centers[labels] + rng.normal(size=(n, d))


def reference_matrices(X: np.ndarray, metric: str = "euclidean", gamma: float | None = None):
    """Distance matrix (SciPy, float64), RBF similarity matrix (float32) and its gamma."""
    D = cdist(X, X, metric=SCIPY_METRIC[metric])
    if gamma is None:
        gamma = 1 / np.median(D[np.triu_indices(len(D), k=1)]) ** 2
    S = np.exp(-gamma * D**2).astype(np.float32)
    np.fill_diagonal(S, 1)
    return D, S, gamma


def kernel_points(X: np.ndarray, metric: str) -> np.ndarray:
    """``X`` as the kernels expect it: contiguous, with unit rows for cosine."""
    X = np.ascontiguousarray(X)
    if metric == "cosine":
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X


def as_csr(S: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A similarity matrix as the CSR triple the kernels accept (zeros dropped)."""
    csr = sparse.csr_array(S)
    return csr.indptr.astype(np.int64), csr.indices.astype(np.int32), csr.data.astype(np.float32)


def as_dense(sim: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """The CSR triple the kernels accept as a dense matrix."""
    indptr, indices, data = sim
    n = indptr.size - 1
    return sparse.csr_array((data, indices, indptr), shape=(n, n)).toarray()


def greedy(metric: str, points: np.ndarray, sim, threshold: float, k: int):
    """Run the compiled greedy kernel; returns ``(selected, utility, diversity)``."""
    out = np.empty(k, np.int64)
    start = K.first_step(sim, points.shape[0])
    size, utility, diversity = K.greedy_independent_set(
        K.METRICS[metric], points, sim, threshold, start, out
    )
    return out[:size].tolist(), utility, diversity
