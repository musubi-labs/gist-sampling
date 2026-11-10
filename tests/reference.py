"""A direct, unoptimised transcription of GIST from the paper (arXiv:2405.18754).

The package is tested against this. It shares no code with the package: it
recomputes every marginal gain at every step from dense matrices.
"""

from __future__ import annotations

import numpy as np


def greedy_independent_set(D: np.ndarray, S: np.ndarray, threshold: float, k: int) -> list[int]:
    """Algorithm 1: greedy by marginal gain among points >= threshold from the selection."""
    n = len(D)
    covered = np.zeros(n)
    nearest = np.full(n, np.inf)
    selected: list[int] = []
    while len(selected) < k:
        feasible = nearest >= threshold
        feasible[selected] = False
        if not feasible.any():
            break
        gains = np.maximum(S - covered, 0).sum(axis=1)
        gains[~feasible] = -np.inf
        v = int(np.argmax(gains))
        selected.append(v)
        covered = np.maximum(covered, S[v])
        nearest = np.minimum(nearest, D[v])
    return selected


def utility(S: np.ndarray, selected: list[int]) -> float:
    """Facility location: each point's similarity to its most similar selected point."""
    return float(S[:, selected].max(axis=1).sum()) if selected else 0.0


def diversity(D: np.ndarray, selected: list[int]) -> float:
    """Minimum pairwise distance; the diameter for fewer than two points."""
    if len(selected) < 2:
        return float(D.max())
    sub = D[np.ix_(selected, selected)]
    return float(sub[np.triu_indices(len(selected), k=1)].min())


def thresholds(diameter: float, epsilon: float) -> list[float]:
    out = []
    scale = 1.0
    while scale <= 2 / epsilon:
        out.append(scale * epsilon * diameter / 2)
        scale *= 1 + epsilon
    return out


def gist(
    D: np.ndarray, S: np.ndarray, k: int, lambda_diversity: float, epsilon: float
) -> tuple[list[int], float]:
    """Algorithm 2. Returns the selected points and their objective value."""

    def objective(selected: list[int]) -> float:
        return utility(S, selected) + lambda_diversity * diversity(D, selected)

    k = min(k, len(D))
    best = greedy_independent_set(D, S, 0.0, k)
    diameter = float(D.max())
    if diameter > 0:
        if k >= 2:
            pair = [int(i) for i in np.unravel_index(np.argmax(D), D.shape)]
            if objective(pair) > objective(best):
                best = pair
        for threshold in thresholds(diameter, epsilon):
            candidate = greedy_independent_set(D, S, threshold, k)
            if objective(candidate) >= objective(best):
                best = candidate
    return best, objective(best)
