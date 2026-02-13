"""GIST (Greedy Independent Set Thresholding) algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gist_sampling.core.greedy_independent_set import DistanceProvider, greedy_independent_set

if TYPE_CHECKING:
    from gist_sampling.utility.base import SubmodularFunction


def gist(
    X: np.ndarray,
    utility_fn: SubmodularFunction,
    k: int,
    epsilon: float,
    lambda_diversity: float,
    distance_matrix: DistanceProvider,
    skip_max_pair: bool = False,
    random_state: int | None = None,
) -> tuple[list[int], float, float]:
    """
    GIST algorithm for diversity-aware submodular maximization.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n, d).
    utility_fn : SubmodularFunction
        Submodular utility function.
    k : int
        Maximum number of elements to select.
    epsilon : float
        Approximation parameter controlling threshold granularity.
    lambda_diversity : float
        Diversity weight (λ) in f(S) = g(S) + λ·div(S).
    distance_matrix : DistanceProvider
        Distance matrix (dense ndarray or LazyDistanceMatrix).
    skip_max_pair : bool, default=False
        Skip the max-distance pair check (faster for large datasets).
    random_state : int | None, default=None
        Seed for randomized diameter estimation (approximate mode).

    Returns
    -------
    tuple[list[int], float, float]
        - Selected indices
        - Objective value (g(S) + λ·div(S))
        - Minimum pairwise distance (diversity)
    """
    n = X.shape[0]

    if lambda_diversity < 0:
        raise ValueError("lambda_diversity must be non-negative")

    if n == 0:
        return [], 0.0, 0.0

    rng = np.random.default_rng(random_state) if random_state is not None else None
    d_max, max_pair = _get_diameter_info(distance_matrix, n, rng=rng)

    if k <= 0:
        empty_obj = utility_fn.evaluate([]) + lambda_diversity * d_max
        return [], empty_obj, d_max

    if n <= k:
        indices = list(range(n))
        obj_val, diversity = _evaluate_set(
            indices, utility_fn, distance_matrix, lambda_diversity, d_max
        )
        return indices, obj_val, diversity

    # Baseline: greedy without distance constraint
    best_solution, best_diversity = greedy_independent_set(
        n,
        utility_fn,
        distance_threshold=0.0,
        k=k,
        distance_matrix=distance_matrix,
        return_diversity=True,
        d_max=d_max,
    )
    best_objective = utility_fn.evaluate(best_solution) + lambda_diversity * best_diversity

    if d_max == 0:
        return best_solution, best_objective, best_diversity

    # Check max-distance pair for k >= 2 when available
    if k >= 2 and not skip_max_pair and max_pair is not None:
        pair_indices = list(max_pair)
        if len(set(pair_indices)) == 2:
            pair_diversity = float(distance_matrix[pair_indices[0]][pair_indices[1]])
            pair_obj = utility_fn.evaluate(pair_indices) + lambda_diversity * pair_diversity
            if pair_obj > best_objective:
                best_solution = pair_indices
                best_objective = pair_obj
                best_diversity = pair_diversity

    # Generate threshold sweep values
    thresholds = _generate_thresholds(d_max, epsilon)

    # Early termination: stop after `patience` consecutive non-improving thresholds.
    # Since thresholds increase monotonically, the objective typically peaks then
    # declines as the distance constraint becomes too restrictive. Continuing past
    # the peak is wasted work since we already keep the best solution seen.
    patience = 3
    no_improvement_count = 0

    for d in thresholds:
        candidate, candidate_diversity = greedy_independent_set(
            n,
            utility_fn,
            distance_threshold=d,
            k=k,
            distance_matrix=distance_matrix,
            return_diversity=True,
            d_max=d_max,
        )

        if len(candidate) == 0:
            break  # thresholds are monotonically increasing; all subsequent will also be empty

        obj_val = utility_fn.evaluate(candidate) + lambda_diversity * candidate_diversity

        if obj_val > best_objective:
            best_solution = candidate
            best_objective = obj_val
            best_diversity = candidate_diversity
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                break

    return best_solution, best_objective, best_diversity


def _generate_thresholds(d_max: float, epsilon: float) -> list[float]:
    """Generate threshold values for the GIST sweep."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    thresholds = []
    base = epsilon * d_max / 2.0
    multiplier = 1.0
    max_multiplier = 2.0 / epsilon

    while multiplier <= max_multiplier:
        threshold = multiplier * base
        if threshold <= d_max:
            thresholds.append(threshold)
        multiplier *= 1.0 + epsilon

    return thresholds


def _compute_diversity(
    indices: list[int],
    distance_matrix: DistanceProvider,
    d_max: float,
) -> float:
    """Compute minimum pairwise distance among selected elements."""
    if len(indices) <= 1:
        return d_max

    min_dist = float("inf")
    for i, idx_i in enumerate(indices):
        row = distance_matrix[idx_i]
        for idx_j in indices[i + 1 :]:
            dist = row[idx_j]
            if dist < min_dist:
                min_dist = dist

    return min_dist


def _get_diameter_info(
    distance_matrix: DistanceProvider,
    n: int,
    sample_size: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[float, tuple[int, int] | None]:
    """
    Get or estimate the diameter (maximum pairwise distance) and a max pair.

    For dense matrices, computes exact max. For lazy matrices, samples rows.
    """
    if n <= 1:
        return 0.0, None

    if isinstance(distance_matrix, np.ndarray):
        max_pair = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)
        return float(distance_matrix[max_pair]), (int(max_pair[0]), int(max_pair[1]))

    # For lazy distance matrix, sample to estimate
    sample_size = min(sample_size, n)
    if n <= sample_size:
        sample_indices = np.arange(n)
    else:
        if rng is None:
            rng = np.random.default_rng()
        sample_indices = rng.choice(n, size=sample_size, replace=False)

    max_dist = 0.0
    max_pair: tuple[int, int] | None = None
    for i in sample_indices:
        row = distance_matrix[int(i)]
        j = int(np.argmax(row))
        dist = float(row[j])
        if dist > max_dist:
            max_dist = dist
            max_pair = (int(i), j)

    return max_dist, max_pair


def _evaluate_set(
    indices: list[int],
    utility_fn: SubmodularFunction,
    distance_matrix: DistanceProvider,
    lambda_diversity: float,
    d_max: float,
) -> tuple[float, float]:
    """Compute objective value and diversity for a selected set."""
    diversity = _compute_diversity(indices, distance_matrix, d_max)
    obj_val = utility_fn.evaluate(indices) + lambda_diversity * diversity
    return obj_val, diversity
