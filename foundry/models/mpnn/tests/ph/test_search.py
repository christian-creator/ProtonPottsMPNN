"""Tests for single-objective runs and the opt-in Pareto sweep."""

import itertools

import numpy as np
import pytest
import torch
from objective_fixtures import BASE, DESIGN_BLOCK, make_context, make_objective

from mpnn.ph.objective import plan_search
from mpnn.ph.pareto import dominates, hypervolume
from mpnn.ph.search import run_search

VALID = torch.tensor([True, True, False, False])
THREE = {"stability": 1.0, "potency": 1.0, "switch": 1.0}


@pytest.fixture
def objective():
    return make_objective(make_context())


def valid_designs():
    for values in itertools.product([0, 1], repeat=len(DESIGN_BLOCK)):
        tokens = BASE.clone()
        tokens[DESIGN_BLOCK] = torch.tensor(values)
        yield tokens


def global_minimum(objective, term_name):
    term = objective.terms[term_name]
    return min(term.value(objective.ctx, t) for t in valid_designs())


def search(objective, plan, **kwargs):
    options = {"block_size": 2, **kwargs}
    return run_search(objective, BASE, DESIGN_BLOCK, plan, valid_mask=VALID, **options)


def test_one_active_term_is_a_single_run_at_its_exact_optimum(objective):
    plan = plan_search({"switch": 1.0}, search="pareto")  # collapses to single
    result = search(objective, plan)
    assert plan.mode == "single" and len(plan.weight_vectors) == 1
    best = result.records[0]
    assert best.front is None and best.selected
    assert best.term_values["switch"] == pytest.approx(
        global_minimum(objective, "switch")
    )
    assert result.hypervolume is None


def test_explicit_weights_give_one_ranked_run(objective):
    plan = plan_search({"stability": 1.0, "switch": 2.0})
    result = search(objective, plan, n_select=1)
    assert result.plan.mode == "single"
    assert [r.scalarised for r in result.records] == sorted(
        r.scalarised for r in result.records
    )
    assert len(result.selected()) == 1 and result.records[0].selected
    assert set(result.records[0].weights) == {"stability", "switch"}


def test_pareto_sweep_finds_each_terms_exact_extreme(objective):
    plan = plan_search(THREE, search="pareto", divisions=2)  # vertices + edge midpoints
    result = search(objective, plan)
    assert plan.mode == "pareto" and len(plan.weight_vectors) == 6
    for name in THREE:
        pooled_best = min(r.term_values[name] for r in result.records)
        assert pooled_best == pytest.approx(global_minimum(objective, name))


def test_pareto_annotation_is_consistent(objective):
    plan = plan_search(THREE, search="pareto", divisions=3)
    result = search(objective, plan)
    names = list(plan.term_names)
    matrix = np.array([[r.term_values[n] for n in names] for r in result.records])
    for i, record in enumerate(result.records):
        dominated = any(
            dominates(matrix[j], matrix[i]) for j in range(len(matrix)) if j != i
        )
        assert (record.front == 0) == (not dominated)
    fronts = [r.front for r in result.records]
    assert fronts == sorted(fronts)
    reference = np.array([result.reference_point[n] for n in names])
    front0 = matrix[[i for i, r in enumerate(result.records) if r.front == 0]]
    assert result.hypervolume == pytest.approx(hypervolume(front0, reference))
    assert result.hypervolume > 0


def test_designs_are_pooled_without_duplicates(objective):
    plan = plan_search(THREE, search="pareto", divisions=3)
    result = search(objective, plan)
    keys = [tuple(int(t) for t in r.tokens) for r in result.records]
    n_runs = len(plan.weight_vectors)
    assert len(keys) == len(set(keys)) < n_runs  # several weight vectors share a design


def test_selection_is_thinned_to_n_select(objective):
    plan = plan_search(THREE, search="pareto", divisions=2)
    everything = search(objective, plan)
    thinned = search(objective, plan, n_select=2)
    assert len(everything.selected()) == len(everything.records)
    assert len(thinned.selected()) == 2


def test_runs_are_reproducible(objective):
    plan = plan_search(THREE, search="pareto", divisions=1)  # the three vertices
    first = search(objective, plan, temperature=1.0, n_seeds=2, base_seed=5)
    second = search(
        make_objective(make_context()), plan, temperature=1.0, n_seeds=2, base_seed=5
    )
    assert [tuple(r.tokens.tolist()) for r in first.records] == [
        tuple(r.tokens.tolist()) for r in second.records
    ]


def test_initialiser_supplies_each_seeds_starting_sequence(objective):
    """Starts that differ at a fixed position stay distinct designs; scales use BASE."""
    starts = [BASE.clone() for _ in range(3)]
    starts[1][5], starts[2][5] = 1, 2  # a fixed receptor position
    plan = plan_search({"stability": 1.0, "potency": 1.0})
    result = search(objective, plan, n_seeds=3, initialiser=lambda s: starts[s])
    assert sorted(int(r.tokens[5]) for r in result.records) == [0, 1, 2]
    assert sorted(r.seed_index for r in result.records) == [0, 1, 2]


def test_unknown_terms_are_reported(objective):
    plan = plan_search({"nonexistent": 1.0})
    with pytest.raises(KeyError, match="nonexistent"):
        search(objective, plan)
