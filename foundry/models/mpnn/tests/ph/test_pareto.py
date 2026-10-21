"""Tests for the multi-objective utilities."""

import itertools
from math import comb

import numpy as np
import pytest

from mpnn.ph.pareto import (
    crowding_distance,
    das_dennis,
    dominates,
    fast_non_dominated_sort,
    hypervolume,
    non_dominated_mask,
    select_by_crowding,
    tchebycheff,
)


def test_dominance_is_strict_and_directional():
    assert dominates(np.array([1, 1]), np.array([2, 1]))
    assert not dominates(np.array([1, 1]), np.array([1, 1]))
    assert not dominates(np.array([1, 3]), np.array([2, 1]))


def test_non_dominated_mask_and_fronts():
    pts = np.array([[1, 4], [2, 2], [4, 1], [3, 3], [5, 5], [2, 2]])
    mask = non_dominated_mask(pts)
    assert mask.tolist() == [True, True, True, False, False, True]  # duplicate kept
    fronts = fast_non_dominated_sort(pts)
    assert sorted(fronts[0].tolist()) == [0, 1, 2, 5]
    assert fronts[1].tolist() == [3]
    assert fronts[2].tolist() == [4]
    assert sorted(np.concatenate(fronts).tolist()) == list(range(len(pts)))


def test_crowding_distance_marks_extremes_and_prefers_isolated_points():
    pts = np.array([[0.0, 4.0], [1.0, 3.0], [1.1, 2.9], [4.0, 0.0]])
    dist = crowding_distance(pts)
    assert np.isinf(dist[0]) and np.isinf(dist[3])
    assert dist[1] > 0 and dist[2] > 0
    # point 1 sits in a tight cluster, point 2 spans a wide gap
    assert dist[2] > dist[1]


def test_select_by_crowding_keeps_extremes_and_whole_fronts():
    pts = np.array([[0, 5], [1, 3], [1.05, 2.95], [2, 2], [5, 0], [6, 6]])
    chosen = select_by_crowding(pts, 4)
    assert len(chosen) == 4 and 5 not in chosen
    assert {0, 4} <= set(chosen.tolist())
    assert sorted(select_by_crowding(pts, 6).tolist()) == list(range(6))


@pytest.mark.parametrize("n_obj, divisions", [(2, 4), (3, 4), (3, 6), (4, 3)])
def test_das_dennis_count_sum_and_vertices(n_obj, divisions):
    w = das_dennis(n_obj, divisions)
    assert len(w) == comb(divisions + n_obj - 1, n_obj - 1)
    assert np.allclose(w.sum(axis=1), 1.0)
    assert len({tuple(row) for row in w}) == len(w)
    for k in range(n_obj):  # every single-objective vertex is present
        assert any(row[k] == 1.0 for row in w)
    assert das_dennis(1, 5).tolist() == [[1.0]]


def test_tchebycheff_reaches_a_point_the_weighted_sum_cannot():
    pts = np.array([[0.0, 1.0], [1.0, 0.0], [0.6, 0.6]])  # last is on a non-convex part
    w = np.array([0.5, 0.5])
    assert np.argmin(pts @ w) != 2
    assert np.argmin(tchebycheff(pts, w, np.zeros(2))) == 2


def test_tchebycheff_ignores_zero_weight_objectives():
    pts = np.array([[1.0, 100.0], [2.0, 0.0]])
    assert np.argmin(tchebycheff(pts, np.array([1.0, 0.0]), np.zeros(2))) == 0


def test_hypervolume_2d_known_value_and_edge_cases():
    pts = np.array([[1.0, 2.0], [2.0, 1.0]])
    ref = np.array([3.0, 3.0])
    assert hypervolume(pts, ref) == pytest.approx(3.0)
    assert hypervolume(np.array([[3.0, 1.0]]), ref) == 0.0  # not strictly inside
    assert hypervolume(np.empty((0, 2)), ref) == 0.0
    dominated = np.vstack([pts, [2.5, 2.5]])
    assert hypervolume(dominated, ref) == pytest.approx(3.0)
    assert hypervolume(np.vstack([pts, [0.5, 2.9]]), ref) > 3.0


def box_union_by_inclusion_exclusion(points, ref):
    total = 0.0
    for k in range(1, len(points) + 1):
        for subset in itertools.combinations(range(len(points)), k):
            corner = np.max(points[list(subset)], axis=0)
            total += (-1) ** (k + 1) * np.prod(ref - corner)
    return total


@pytest.mark.parametrize("n_obj, seed", [(2, 0), (3, 1), (3, 2), (4, 3)])
def test_hypervolume_matches_inclusion_exclusion(n_obj, seed):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.0, 1.0, size=(6, n_obj))
    ref = np.ones(n_obj) * 1.1
    assert hypervolume(pts, ref) == pytest.approx(
        box_union_by_inclusion_exclusion(pts, ref)
    )


def test_a_constant_objective_does_not_decide_which_points_are_extremes():
    points = np.array([[1.0, 2.0], [0.0, 3.0], [3.0, 0.0], [2.0, 1.0]])
    with_constant = np.column_stack([points, np.zeros(len(points))])
    assert select_by_crowding(with_constant, 2).tolist() == (
        select_by_crowding(points, 2).tolist()
    )
    assert np.isfinite(crowding_distance(with_constant)).sum() == 2
