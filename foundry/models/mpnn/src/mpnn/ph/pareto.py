"""Multi-objective utilities: dominance, sorting, selection, weights, hypervolume.

All objectives are minimised. Inputs are ``[N, M]`` arrays (N designs, M objectives).
"""

from __future__ import annotations

import itertools
import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """True if ``a`` is no worse than ``b`` everywhere and strictly better somewhere."""
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated_mask(points: np.ndarray) -> np.ndarray:
    """Boolean mask of points no other point dominates (duplicates are all kept)."""
    points = np.asarray(points, dtype=float)
    keep = np.ones(len(points), dtype=bool)
    for i in range(len(points)):
        no_worse = np.all(points <= points[i], axis=1)
        strictly_better = np.any(points < points[i], axis=1)
        if np.any(no_worse & strictly_better):
            keep[i] = False
    return keep


def fast_non_dominated_sort(points: np.ndarray) -> List[np.ndarray]:
    """Successive Pareto fronts as index arrays; front 0 is the non-dominated set."""
    points = np.asarray(points, dtype=float)
    remaining = np.arange(len(points))
    fronts: List[np.ndarray] = []
    while len(remaining):
        mask = non_dominated_mask(points[remaining])
        fronts.append(remaining[mask])
        remaining = remaining[~mask]
    return fronts


def crowding_distance(points: np.ndarray) -> np.ndarray:
    """NSGA-II crowding distance; boundary points get ``inf``."""
    points = np.asarray(points, dtype=float)
    n, m = points.shape
    distance = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for k in range(m):
        order = np.argsort(points[:, k], kind="stable")
        span = points[order[-1], k] - points[order[0], k]
        if span == 0:  # a constant objective has no extremes to protect
            continue
        distance[order[0]] = distance[order[-1]] = np.inf
        gaps = (points[order[2:], k] - points[order[:-2], k]) / span
        distance[order[1:-1]] += gaps
    return distance


def select_by_crowding(points: np.ndarray, n_select: int) -> np.ndarray:
    """Pick ``n_select`` indices: whole fronts first, the last one thinned by crowding."""
    points = np.asarray(points, dtype=float)
    chosen: List[int] = []
    for front in fast_non_dominated_sort(points):
        room = n_select - len(chosen)
        if room <= 0:
            break
        if len(front) <= room:
            chosen.extend(front.tolist())
        else:
            spread = crowding_distance(points[front])
            order = np.argsort(-spread, kind="stable")[:room]
            chosen.extend(front[order].tolist())
    return np.array(chosen, dtype=int)


def das_dennis(n_objectives: int, divisions: int) -> np.ndarray:
    """Evenly spaced weight vectors on the simplex (rows sum to 1, vertices included)."""
    if n_objectives < 1 or divisions < 1:
        raise ValueError("n_objectives and divisions must be >= 1.")
    rows = []
    for cuts in itertools.combinations(
        range(divisions + n_objectives - 1), n_objectives - 1
    ):
        edges = (-1, *cuts, divisions + n_objectives - 1)
        rows.append([edges[i + 1] - edges[i] - 1 for i in range(n_objectives)])
    return np.array(rows, dtype=float) / divisions


def tchebycheff(
    values: np.ndarray, weights: np.ndarray, ideal: np.ndarray
) -> np.ndarray:
    """``max_k w_k (z_k - ideal_k)`` over the objectives with positive weight.

    Unlike a weighted sum, minimising this can reach points on non-convex parts of the
    front.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    active = weights > 0
    return np.max(
        weights[active] * (values[..., active] - np.asarray(ideal)[active]), axis=-1
    )


def hypervolume(points: np.ndarray, reference: np.ndarray) -> float:
    """Volume dominated by ``points`` and bounded by ``reference`` (minimisation).

    Exact slicing recursion; fine for the few hundred points of a design set.
    """
    points = np.asarray(points, dtype=float)
    reference = np.asarray(reference, dtype=float)
    inside = points[np.all(points < reference, axis=1)]
    if len(inside) == 0:
        return 0.0
    return _hypervolume(inside, reference)


def _hypervolume(points: np.ndarray, reference: np.ndarray) -> float:
    if points.shape[1] == 1:
        return float(reference[0] - points[:, 0].min())
    points = points[np.argsort(points[:, -1], kind="stable")]
    volume = 0.0
    for i in range(len(points)):
        floor = points[i, -1]
        ceiling = points[i + 1, -1] if i + 1 < len(points) else reference[-1]
        if ceiling > floor:
            volume += _hypervolume(points[: i + 1, :-1], reference[:-1]) * (
                ceiling - floor
            )
    return volume
