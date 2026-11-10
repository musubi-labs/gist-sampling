"""Compiled kernels: distances, facility-location gains and the lazy greedy sweep.

Everything here is Numba ``nopython`` code operating on plain arrays. Two
conventions keep the kernels small:

* A *metric* is an integer code (see :data:`METRICS`) and distances are
  computed on demand from the feature matrix, so no ``n x n`` distance matrix
  is ever stored. For ``PRECOMPUTED`` the "feature matrix" is the distance
  matrix itself.
* A *similarity* is either a dense ``(n, n)`` array or a CSR triple
  ``(indptr, indices, data)``. Both are symmetric, so row ``v`` lists the
  points that ``v`` covers. :func:`gain` and :func:`cover` dispatch on the
  representation at compile time.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange, types
from numba.extending import overload

EUCLIDEAN, COSINE, MANHATTAN, CHEBYSHEV, PRECOMPUTED = range(5)
METRICS = {
    "euclidean": EUCLIDEAN,
    "cosine": COSINE,
    "manhattan": MANHATTAN,
    "chebyshev": CHEBYSHEV,
    "precomputed": PRECOMPUTED,
}

RBF, INVERSE = range(2)
SIMILARITIES = {"rbf": RBF, "inverse": INVERSE}

# Allow the compiler to reorder floating-point reductions (needed for SIMD)
# without assuming the absence of NaN or infinity.
_FAST = {"reassoc", "nsz", "contract"}


# --------------------------------------------------------------------------
# Distances
# --------------------------------------------------------------------------


@njit(fastmath=_FAST, cache=True)
def distance(metric, X, u, v):
    """Distance between points ``u`` and ``v``.

    For ``COSINE`` the rows of ``X`` must already be unit-normalised.
    """
    if metric == PRECOMPUTED:
        return X[u, v]
    a, b = X[u], X[v]
    acc = a[0] * 0  # a zero of the data's own precision
    if metric == EUCLIDEAN:
        for j in range(a.size):
            diff = a[j] - b[j]
            acc += diff * diff
        return np.sqrt(acc)
    if metric == COSINE:
        for j in range(a.size):
            acc += a[j] * b[j]
        return max(1 - acc, acc * 0)
    if metric == MANHATTAN:
        for j in range(a.size):
            acc += abs(a[j] - b[j])
        return acc
    for j in range(a.size):
        acc = max(acc, abs(a[j] - b[j]))
    return acc


@njit(parallel=True, cache=True)
def distances_from(metric, X, u):
    """Distances from point ``u`` to every point."""
    out = np.empty(X.shape[0], X.dtype)
    for v in prange(X.shape[0]):
        out[v] = distance(metric, X, u, v)
    return out


@njit(parallel=True, cache=True)
def paired_distances(metric, X, left, right):
    """Distances between ``left[i]`` and ``right[i]`` for every ``i``."""
    out = np.empty(left.size, X.dtype)
    for i in prange(left.size):
        out[i] = distance(metric, X, left[i], right[i])
    return out


# --------------------------------------------------------------------------
# Similarity
# --------------------------------------------------------------------------


@njit(cache=True)
def similarity(kind, gamma, dist):
    """Map a distance to a similarity in ``(0, 1]``."""
    if kind == RBF:
        return np.exp(-gamma * dist * dist)
    return 1 / (1 + dist)


@njit(parallel=True, cache=True)
def similarities(kind, gamma, distances):
    """:func:`similarity` of every entry of a 2-D array of distances."""
    out = np.empty(distances.shape, np.float32)
    for i in prange(distances.shape[0]):
        for j in range(distances.shape[1]):
            out[i, j] = similarity(kind, gamma, distances[i, j])
    return out


@njit(parallel=True, cache=True)
def dense_similarity(metric, X, kind, gamma):
    """Full similarity matrix, plus each point's farthest neighbour.

    Returns ``(sim, farthest, farthest_distance)`` where ``farthest[u]`` is the
    point farthest from ``u``; the caller reduces these to the exact diameter.
    """
    n = X.shape[0]
    sim = np.empty((n, n), np.float32)
    farthest = np.empty(n, np.int64)
    farthest_distance = np.empty(n, X.dtype)
    for u in prange(n):
        best = 0
        best_distance = X[0, 0] * 0
        for v in range(n):
            dist = distance(metric, X, u, v)
            sim[u, v] = similarity(kind, gamma, dist)
            if dist > best_distance:
                best = v
                best_distance = dist
        sim[u, u] = 1
        farthest[u] = best
        farthest_distance[u] = best_distance
    return sim, farthest, farthest_distance


# --------------------------------------------------------------------------
# Facility location: g(S) = sum_i max_{u in S} sim(i, u)
#
# ``covered[i]`` holds max_{u in S} sim(i, u) for the current set S.
# --------------------------------------------------------------------------


def gain(sim, covered, v):
    """Marginal gain g(S + v) - g(S)."""
    raise NotImplementedError  # compiled implementations are registered below


def cover(sim, covered, v):
    """Update ``covered`` in place after adding ``v`` to S."""
    raise NotImplementedError


@overload(gain, jit_options={"fastmath": _FAST})
def _gain(sim, covered, v):
    if isinstance(sim, types.Array):

        def dense(sim, covered, v):
            row = sim[v]
            total = 0.0
            for i in range(row.size):
                total += max(row[i] - covered[i], 0)
            return total

        return dense

    def sparse(sim, covered, v):
        indptr, indices, data = sim
        total = 0.0
        for j in range(indptr[v], indptr[v + 1]):
            total += max(data[j] - covered[indices[j]], 0)
        return total

    return sparse


@overload(cover)
def _cover(sim, covered, v):
    if isinstance(sim, types.Array):

        def dense(sim, covered, v):
            row = sim[v]
            for i in range(row.size):
                covered[i] = max(covered[i], row[i])

        return dense

    def sparse(sim, covered, v):
        indptr, indices, data = sim
        for j in range(indptr[v], indptr[v + 1]):
            covered[indices[j]] = max(covered[indices[j]], data[j])

    return sparse


def gain_cost(sim, v):
    """Number of similarities read by ``gain(sim, covered, v)``."""
    raise NotImplementedError


@overload(gain_cost)
def _gain_cost(sim, v):
    if isinstance(sim, types.Array):
        return lambda sim, v: sim.shape[1]
    return lambda sim, v: sim[0][v + 1] - sim[0][v]


@njit(parallel=True, cache=True)
def gains(sim, covered):
    """Marginal gain of every point with respect to ``covered``."""
    out = np.empty(covered.size, np.float64)
    for v in prange(covered.size):
        out[v] = gain(sim, covered, v)
    return out


@njit(cache=True)
def coverage(sim, n, subset):
    """Coverage vector of ``subset``; its sum is the facility-location value."""
    covered = np.zeros(n, np.float32)
    for v in subset:
        cover(sim, covered, v)
    return covered


# --------------------------------------------------------------------------
# Lazy greedy
# --------------------------------------------------------------------------


@njit(cache=True)
def _ranks_before(bound, a, b):
    """Heap order: larger bound first, ties broken by smaller index."""
    return bound[a] > bound[b] or (bound[a] == bound[b] and a < b)


@njit(cache=True)
def _sift_down(heap, size, bound):
    """Restore the heap after the root's bound decreased."""
    item, i = heap[0], 0
    while True:
        child = 2 * i + 1
        if child >= size:
            break
        if child + 1 < size and _ranks_before(bound, heap[child + 1], heap[child]):
            child += 1
        if not _ranks_before(bound, heap[child], item):
            break
        heap[i] = heap[child]
        i = child
    heap[i] = item


@njit(cache=True)
def _pop(heap, size, bound):
    """Remove the root and return the new heap size."""
    size -= 1
    if size > 0:
        heap[0] = heap[size]
        _sift_down(heap, size, bound)
    return size


@njit(cache=True)
def first_step(sim, n):
    """The first greedy step, which every run shares because it ignores distances.

    Returns ``(first, first_gain, order, bounds)``: the point of largest gain,
    that gain, and for the remaining points their gains given ``{first}``
    (``bounds``) and their indices sorted by those gains, ties by index
    (``order``). A sorted array is a valid heap.
    """
    initial = gains(sim, np.zeros(n, np.float32))
    first = np.argmax(initial)
    bounds = gains(sim, coverage(sim, n, np.array([first])))
    order = np.argsort(-bounds, kind="mergesort")  # stable: ties stay in index order
    return first, initial[first], order[order != first], bounds


@njit(cache=True)
def greedy_independent_set(metric, X, sim, threshold, start, out):
    """GreedyIndependentSet (Algorithm 1 of the GIST paper) with lazy evaluation.

    Repeatedly adds the point of largest marginal gain among those at distance
    at least ``threshold`` from every point already selected, until ``out`` is
    full or no such point remains.

    ``start`` is the result of :func:`first_step`.

    Two kinds of laziness keep a run far below ``O(n)`` work per step:

    * Gains only shrink as the set grows (submodularity), so a stale gain is
      an upper bound and a point is re-evaluated only when it reaches the top
      of the heap.
    * Distances to the selected points are computed only for the point at the
      top of the heap, and each (point, selected point) pair at most once.

    The point at the top of the heap may need both; the cheaper is done first,
    since either can remove the point from contention and spare the other.

    Selected points are written to ``out``. Returns ``(size, utility,
    diversity)``, where ``diversity`` is the minimum pairwise distance among
    the selected points (infinite if fewer than two were selected).
    """
    first, first_gain, order, bounds = start
    n = bounds.size
    bound = bounds.copy()  # upper bound on each point's marginal gain
    fresh_at = np.ones(n, np.int64)  # |S| for which bound[v] is exact
    checked = np.zeros(n, np.int64)  # how many selected points v was compared to
    nearest = np.full(n, np.inf)  # distance from v to the nearest of those
    heap = order.copy()
    heap_size = heap.size
    distance_cost = 1 if metric == PRECOMPUTED else X.shape[1]

    covered = np.zeros(n, np.float32)
    cover(sim, covered, first)
    out[0] = first
    size = 1
    utility = first_gain
    diversity = np.inf

    while size < out.size and heap_size > 0:
        v = heap[0]
        stale = fresh_at[v] != size
        unchecked = size - checked[v] if threshold > 0 else 0

        if unchecked and (not stale or unchecked * distance_cost < gain_cost(sim, v)):
            while checked[v] < size and nearest[v] >= threshold:
                nearest[v] = min(nearest[v], distance(metric, X, v, out[checked[v]]))
                checked[v] += 1
            if nearest[v] < threshold:  # too close to the selection, now and forever
                heap_size = _pop(heap, heap_size, bound)
        elif stale:
            bound[v] = gain(sim, covered, v)
            fresh_at[v] = size
            _sift_down(heap, heap_size, bound)
        else:  # exact gain, no smaller than any other bound, and far enough
            heap_size = _pop(heap, heap_size, bound)
            while checked[v] < size:  # only pending when threshold == 0
                nearest[v] = min(nearest[v], distance(metric, X, v, out[checked[v]]))
                checked[v] += 1
            diversity = min(diversity, nearest[v])
            utility += bound[v]
            cover(sim, covered, v)
            out[size] = v
            size += 1

    return size, utility, diversity


@njit(parallel=True, cache=True)
def sweep(metric, X, sim, thresholds, k, start):
    """Run :func:`greedy_independent_set` once per threshold, in parallel.

    Returns ``(selected, sizes, utilities, diversities)``; row ``t`` of
    ``selected`` holds the ``sizes[t]`` points chosen for ``thresholds[t]``.
    """
    runs = thresholds.size
    selected = np.empty((runs, k), np.int64)
    sizes = np.empty(runs, np.int64)
    utilities = np.empty(runs, np.float64)
    diversities = np.empty(runs, np.float64)
    for t in prange(runs):
        sizes[t], utilities[t], diversities[t] = greedy_independent_set(
            metric, X, sim, thresholds[t], start, selected[t]
        )
    return selected, sizes, utilities, diversities
