"""Greedy Independent Set algorithm for diversity-aware selection."""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from gist_sampling.utility.base import SubmodularFunction


@runtime_checkable
class DistanceProvider(Protocol):
    """Protocol for objects that can provide distance rows."""

    def __getitem__(self, idx: int) -> np.ndarray: ...


def _lazy_pop_best(
    heap: list[tuple[float, int]],
    utility_fn: SubmodularFunction,
    selected_mask: np.ndarray,
    dist_to_selected: np.ndarray,
    distance_threshold: float,
) -> int | None:
    """
    Pop the best valid candidate from the lazy evaluation heap.

    Uses submodularity (gains can only decrease) to avoid recomputing
    gains for all candidates. Upper bounds from previous steps are used
    to identify the true maximizer with minimal recomputations.

    Returns the index of the best candidate, or None if no valid candidate.
    """
    while heap:
        neg_upper, idx = heapq.heappop(heap)

        # Skip already-selected or distance-invalidated candidates
        if selected_mask[idx]:
            continue
        if distance_threshold > 0 and dist_to_selected[idx] < distance_threshold:
            continue

        # Recompute actual gain under current coverage
        actual_gain = utility_fn.marginal_gain_single(idx)

        # If gain >= next upper bound (or heap is empty), this is the best
        if not heap or actual_gain >= -heap[0][0]:
            return idx

        # Not the best anymore — push back with tighter bound
        heapq.heappush(heap, (-actual_gain, idx))

    return None


def greedy_independent_set(
    n: int,
    utility_fn: SubmodularFunction,
    distance_threshold: float,
    k: int,
    distance_matrix: DistanceProvider,
    *,
    return_diversity: bool = False,
    d_max: float | None = None,
) -> list[int] | tuple[list[int], float]:
    """
    Greedy Independent Set algorithm with lazy evaluation.

    Selects up to k elements that are at least distance_threshold apart
    from each other, maximizing the marginal gain of the utility function.

    Uses lazy evaluation (accelerated greedy) to avoid recomputing marginal
    gains for all candidates at every step. Since the utility function is
    submodular (gains can only decrease), upper bounds from previous steps
    are used to prune unnecessary evaluations.

    Parameters
    ----------
    n : int
        Total number of elements.
    utility_fn : SubmodularFunction
        Submodular utility function.
    distance_threshold : float
        Minimum distance required between any two selected elements.
    k : int
        Maximum number of elements to select.
    distance_matrix : DistanceProvider
        Object supporting row indexing (dense ndarray or LazyDistanceMatrix).
    return_diversity : bool, default=False
        If True, also return the minimum pairwise distance of the selected set.
    d_max : float | None, default=None
        Dataset diameter for |S| <= 1 diversity; required when return_diversity=True.

    Returns
    -------
    list[int] | tuple[list[int], float]
        Indices of selected elements, and optionally the diversity value.
    """
    if return_diversity and d_max is None:
        raise ValueError("d_max is required when return_diversity=True")

    selected: list[int] = []
    dist_to_selected = np.full(n, np.inf)
    selected_mask = np.zeros(n, dtype=bool)
    min_diversity = d_max if d_max is not None else float("inf")
    heap: list[tuple[float, int]] = []

    # Reset coverage for incremental tracking
    utility_fn.reset_coverage()

    # --- Step 1: batch evaluation to select first element and seed the heap ---
    if k >= 1:
        if distance_threshold > 0:
            candidates_mask = dist_to_selected >= distance_threshold
        else:
            candidates_mask = np.ones(n, dtype=bool)
        candidates_mask &= ~selected_mask
        candidates = np.where(candidates_mask)[0]

        if len(candidates) > 0:
            gains = utility_fn.marginal_gains_batch(candidates)
            best_local_idx = int(np.argmax(gains))
            best_idx = int(candidates[best_local_idx])

            selected.append(best_idx)
            selected_mask[best_idx] = True
            utility_fn.update_coverage(best_idx)
            dist_to_selected = np.minimum(dist_to_selected, distance_matrix[best_idx])

            # Build max-heap with upper bounds for remaining candidates
            # Negate gains because heapq is a min-heap
            heap = [
                (-float(gains[i]), int(candidates[i]))
                for i in range(len(candidates))
                if int(candidates[i]) != best_idx
            ]
            heapq.heapify(heap)

    # --- Steps 2..k: lazy evaluation from the heap ---
    for _ in range(k - 1):
        if not heap:
            break

        best_idx = _lazy_pop_best(
            heap, utility_fn, selected_mask, dist_to_selected, distance_threshold
        )
        if best_idx is None:
            break

        if return_diversity:
            min_diversity = min(min_diversity, float(dist_to_selected[best_idx]))

        selected.append(best_idx)
        selected_mask[best_idx] = True
        utility_fn.update_coverage(best_idx)
        dist_to_selected = np.minimum(dist_to_selected, distance_matrix[best_idx])

    if return_diversity:
        if len(selected) <= 1 and d_max is not None:
            return selected, d_max
        return selected, min_diversity
    return selected
