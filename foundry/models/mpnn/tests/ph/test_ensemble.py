"""One binder, several target states."""

import dataclasses
import itertools

import numpy as np
import pytest
import torch
from objective_fixtures import BASE, DESIGN_BLOCK, NAMES
from potts_fixtures import IdealComplex
from test_session import make_config, make_inputs

from mpnn.ph.session import run_switch_design, run_switch_design_ensemble
from mpnn.ph.vocab_meta import TokenTable

TERMS = ("stability", "potency", "switch")
WEIGHTS = (0.3, 0.3, 0.4)


def two_states():
    human = make_inputs(IdealComplex(1, 2, 3))
    mouse_tokens = BASE.clone()
    mouse_tokens[5] = 1  # the mouse target has a different residue at position 5
    mouse = dataclasses.replace(
        make_inputs(IdealComplex(4, 5, 6)),
        tokens=mouse_tokens,
        parents=["HIS", "ALA", "ALA", "ALA", "HIS", "HIS"],
    )
    return {"human": human, "mouse": mouse}


def config(**overrides):
    return make_config(allow_bare_titratable=["HIS"], **overrides)


def valid_designs():
    for values in itertools.product([0, 1], repeat=len(DESIGN_BLOCK)):
        tokens = BASE.clone()
        tokens[DESIGN_BLOCK] = torch.tensor(values)
        yield tokens


@pytest.mark.parametrize(
    "reduce, combine", [("max", max), ("mean", lambda v: sum(v) / len(v))]
)
def test_terms_reduce_the_per_state_values(reduce, combine):
    states = two_states()
    run = run_switch_design_ensemble(states, config(target_reduce=reduce))
    for record in run.result.records:
        assert set(record.per_state) == {"human", "mouse"}
        for term in TERMS:
            per_state = [record.per_state[s][term] for s in ("human", "mouse")]
            assert record.term_values[term] == pytest.approx(combine(per_state))


def test_per_state_values_match_separate_single_state_evaluations():
    states = two_states()
    run = run_switch_design_ensemble(states, config())
    record = run.result.records[0]
    human = run_switch_design(states["human"], config()).objective
    assert record.per_state["human"] == pytest.approx(human.term_values(record.tokens))
    mouse = run_switch_design(states["mouse"], config()).objective
    mouse_tokens = states["mouse"].tokens.clone()
    mouse_tokens[DESIGN_BLOCK] = record.tokens[DESIGN_BLOCK]  # the shared binder
    assert record.per_state["mouse"] == pytest.approx(mouse.term_values(mouse_tokens))
    assert not torch.equal(mouse_tokens, record.tokens)  # the targets do differ


@pytest.mark.parametrize("reduce", ["max", "mean"])
def test_ensemble_run_reaches_the_exact_optimum(reduce):
    run = run_switch_design_ensemble(two_states(), config(target_reduce=reduce))
    brute = min(
        run.objective.scalarised_value(t, TERMS, WEIGHTS) for t in valid_designs()
    )
    assert run.result.records[0].scalarised == pytest.approx(brute)


def test_a_one_state_ensemble_equals_the_plain_run():
    states = two_states()
    plain = run_switch_design(states["human"], config())
    single = run_switch_design_ensemble({"human": states["human"]}, config())
    assert [r.tokens.tolist() for r in single.result.records] == [
        r.tokens.tolist() for r in plain.result.records
    ]
    for a, b in zip(single.result.records, plain.result.records):
        assert a.term_values == pytest.approx(b.term_values)


def test_worst_state_is_what_max_optimises():
    states = two_states()
    run = run_switch_design_ensemble(states, config(target_reduce="max"))
    best = run.result.records[0]
    for term in TERMS:
        worst = max(best.per_state[s][term] for s in best.per_state)
        assert best.term_values[term] == pytest.approx(worst)


def test_pareto_search_works_over_states():
    run = run_switch_design_ensemble(two_states(), config(search="pareto", divisions=2))
    assert run.result.plan.mode == "pareto" and run.result.hypervolume > 0
    assert all(r.per_state is not None for r in run.result.records)


def test_rows_and_site_report_cover_every_state():
    run = run_switch_design_ensemble(two_states(), config())
    row = run.to_rows("A")[0]
    assert {"term_potency@human", "term_potency@mouse", "term_potency"} <= set(row)
    rows = run.site_report(run.result.records[0])
    assert {r["state"] for r in rows} == {"human", "mouse"}
    assert len(rows) == 2 * 2 * 2  # states x conditions x sites
    assert run.describe(run.result.records[0], "A")["chain"] == "A"


@pytest.mark.parametrize(
    "change, error, message",
    [
        (
            lambda s: dataclasses.replace(
                s, parents=["ALA", "ALA", "ALA", "ALA", "HIS", "ALA"]
            ),
            ValueError,
            "binder sequence differs",
        ),
        (
            lambda s: dataclasses.replace(s, res_ids=[1, 2, 9, 1, 2, 3]),
            ValueError,
            "different binder residues",
        ),
        (
            lambda s: dataclasses.replace(s, table=TokenTable((*NAMES, "GLY"))),
            ValueError,
            "different vocabulary",
        ),
        (
            lambda s: dataclasses.replace(s, res_ids=[1, 2, 3, 1, 9, 3]),
            KeyError,
            "mouse",
        ),
    ],
)
def test_the_binder_must_be_shared_and_sites_must_exist_in_every_state(
    change, error, message
):
    states = two_states()
    states["mouse"] = change(states["mouse"])
    with pytest.raises(error, match=message):
        run_switch_design_ensemble(states, config())


def test_designing_a_non_shared_position_is_refused():
    states = two_states()
    with pytest.raises(ValueError, match="shared binder"):
        run_switch_design_ensemble(states, config(designable={"chains": ["B"]}))


def test_target_reduce_is_validated_and_defaults_to_the_worst_state():
    assert config().target_reduce == "max"
    with pytest.raises(ValueError, match="target_reduce"):
        config(target_reduce="median")


def contact_table(nearest_binder_of_target_site):
    """Toy contact table; target site B:2 (position 4) lists the given binder position first."""
    table = [[0, 1, 4], [1, 2, 3], [2, 1, 3], [3, 2, 5], [4, 1, 5], [5, 4, 3]]
    table[4] = [4, nearest_binder_of_target_site, 5]
    return np.array(table)


def test_max_mutations_caps_the_shared_designable_set_not_each_state():
    """Two states that pick different nearest residues must still respect one total cap."""
    states = two_states()
    states["human"] = dataclasses.replace(
        states["human"], neighbour_index=contact_table(1)
    )
    states["mouse"] = dataclasses.replace(
        states["mouse"], neighbour_index=contact_table(2)
    )
    near = {"sites": ["B:2"], "max_mutations": 1}
    run = run_switch_design_ensemble(
        states, config(designable={"chains": ["A"], "near": near})
    )
    assert run.designable == [1]  # ties on rank go to the lower position
    uncapped = run_switch_design_ensemble(
        states, config(designable={"chains": ["A"], "near": {"sites": ["B:2"]}})
    )
    assert uncapped.designable == [1, 2]  # without the cap the states differ


def test_each_state_keeps_its_own_fixed_binder_tokens():
    """A fixed His that is HIS-P in one complex and HIS-S in another stays so in both."""
    states = two_states()
    for name, token in [("human", 2), ("mouse", 3)]:
        tokens = states[name].tokens.clone()
        tokens[0] = token
        states[name] = dataclasses.replace(states[name], tokens=tokens)
    conditions = {  # only the target site is imposed; A:1 is neither designed nor imposed
        "on": {"states": {"B:2": "deprotonated"}},
        "off": {"states": {"B:2": "protonated"}},
    }
    run = run_switch_design_ensemble(
        states, config(conditions=conditions, designable={"include": ["A:2", "A:3"]})
    )
    record = run.result.records[0]
    expected = states["mouse"].tokens.clone()
    expected[run.designable] = record.tokens[run.designable]
    actual = run.objective.tokens_for(run.members["mouse"], record.tokens)
    assert torch.equal(actual, expected)
    assert int(actual[0]) == 3


def test_a_state_without_neighbours_of_the_site_still_joins_the_ensemble():
    """The union over states decides whether anything can be designed."""
    states = two_states()
    states["human"] = dataclasses.replace(
        states["human"], neighbour_index=contact_table(1)
    )
    states["mouse"] = dataclasses.replace(  # B:2 only touches receptor residues here
        states["mouse"],
        neighbour_index=np.array(
            [[0, 1, 2], [1, 0, 2], [2, 0, 1], [3, 4, 5], [4, 3, 5], [5, 3, 4]]
        ),
    )
    run = run_switch_design_ensemble(
        states, config(designable={"chains": ["A"], "near": {"sites": ["B:2"]}})
    )
    assert run.designable == [1]
    empty = {
        name: dataclasses.replace(s, neighbour_index=states["mouse"].neighbour_index)
        for name, s in states.items()
    }
    with pytest.raises(ValueError, match="No designable positions in any state"):
        run_switch_design_ensemble(
            empty, config(designable={"chains": ["A"], "near": {"sites": ["B:2"]}})
        )
