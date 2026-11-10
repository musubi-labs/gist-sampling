"""GIST: Greedy Independent Set Thresholding (Fahrbach et al., arXiv:2405.18754).

GIST selects at most ``k`` points maximising

    f(S) = g(S) + lambda * div(S)

where ``g`` is a monotone submodular utility (here facility location) and
``div(S)`` is the minimum pairwise distance in ``S``, defined as the diameter
of the data when ``S`` has fewer than two points.

A set whose points are all at least ``d`` apart has ``div(S) >= d``, and for a
fixed ``d`` a greedy algorithm finds such a set of near-maximal utility. GIST
runs that greedy algorithm for a geometric grid of thresholds ``d`` and keeps
the best result, which is within a factor ``1/2 - epsilon`` of optimal.

With ``lambda = 0`` the objective is the utility alone, for which plain greedy
is the standard algorithm (within ``1 - 1/e`` of optimal), so GIST returns it.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from gist_sampling import _kernels as K

# Thresholds run per batch; bounds the memory held for their selections.
_BATCH = 64

SimilarityMatrix = np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]


class Selection(NamedTuple):
    """A selected subset and the terms of its objective."""

    indices: np.ndarray
    utility: float
    diversity: float
    threshold: float  # distance threshold that produced it; NaN for the farthest pair

    def objective(self, lambda_diversity: float) -> float:
        return self.utility + lambda_diversity * self.diversity


def thresholds(diameter: float, epsilon: float) -> np.ndarray:
    """Distance thresholds ``(1 + eps)^i * eps * diameter / 2`` up to the diameter."""
    count = int(np.floor(np.log(2 / epsilon) / np.log1p(epsilon) + 1e-9)) + 1
    return epsilon * diameter / 2 * (1 + epsilon) ** np.arange(max(count, 0))


def gist(
    metric: int,
    X: np.ndarray,
    sim: SimilarityMatrix,
    k: int,
    *,
    lambda_diversity: float,
    epsilon: float,
    farthest_pair: tuple[int, int, float],
) -> Selection:
    """Run GIST and return the best subset found.

    Parameters
    ----------
    metric, X
        Metric code and the matrix that :func:`gist_sampling._kernels.distance`
        reads distances from.
    sim
        Symmetric similarity defining the facility-location utility: a dense
        ``(n, n)`` array or CSR arrays ``(indptr, indices, data)``.
    k
        Maximum number of points to select.
    lambda_diversity
        Weight of the diversity term, per unit of distance.
    epsilon
        Spacing of the distance thresholds.
    farthest_pair
        ``(u, v, distance)`` for the two farthest points. The distance is used
        as the diameter of the data.
    """
    n = X.shape[0]
    k = min(k, n)
    far_u, far_v, diameter = farthest_pair
    start = K.first_step(sim, n)

    def greedy(distance_thresholds: np.ndarray) -> list[Selection]:
        """One greedy selection per threshold, computed in parallel."""
        selected, sizes, utilities, diversities = K.sweep(
            metric, X, sim, distance_thresholds, k, start
        )
        return [
            Selection(
                indices=selected[t, : sizes[t]],
                utility=float(utilities[t]),
                diversity=float(diversities[t]) if sizes[t] > 1 else diameter,
                threshold=float(distance_thresholds[t]),
            )
            for t in range(distance_thresholds.size)
        ]

    def objective(selection: Selection) -> float:
        return selection.objective(lambda_diversity)

    # Plain greedy, which ignores distances.
    (plain,) = greedy(np.zeros(1))
    if lambda_diversity == 0:
        return plain
    best = plain

    # The two farthest points: the most diverse pair there is.
    if k >= 2 and diameter > 0:
        pair = np.array([far_u, far_v])
        utility = float(K.coverage(sim, n, pair).sum(dtype=np.float64))
        candidate = Selection(pair, utility, diameter, float("nan"))
        if objective(candidate) > objective(best):
            best = candidate

    # Greedy under each distance threshold. A threshold no larger than the
    # diversity plain greedy already achieves would reproduce plain greedy
    # exactly, so only the larger ones are worth running.
    grid = thresholds(diameter, epsilon)
    grid = grid[grid > plain.diversity]
    for batch in range(0, grid.size, _BATCH):
        for candidate in greedy(grid[batch : batch + _BATCH]):
            if objective(candidate) >= objective(best):
                # Copy, so that the winner does not keep its whole batch alive.
                best = candidate._replace(indices=candidate.indices.copy())
    return best
