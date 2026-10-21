"""End-to-end design runs on the toy complex, from a plain config dict."""

import itertools
from copy import deepcopy

import pytest
import torch
from objective_fixtures import BASE, DESIGN_BLOCK, NAMES, PARENTS, SPEC
from potts_fixtures import FunctionSystem, IdealComplex

from mpnn.ph.config import DesignableSpec, SwitchDesignConfig
from mpnn.ph.session import DesignInputs, resolve_designable, run_switch_design
from mpnn.ph.vocab_meta import TokenTable

CHAINS = ["A", "A", "A", "B", "B", "B"]
RES_IDS = [1, 2, 3, 1, 2, 3]
THREE_TERMS = {
    "stability": {"weight": 0.3},
    "potency": {"weight": 0.3},
    "switch": {"weight": 0.4},
}


def make_inputs(ideal=None):
    ideal = ideal or IdealComplex(1, 2, 3)
    return DesignInputs(
        tokens=BASE,
        table=TokenTable(NAMES),
        chain_ids=CHAINS,
        res_ids=RES_IDS,
        parents=PARENTS,
        complex_scorer=ideal,
        binder_scorer=ideal.binder,
        receptor_scorer=ideal.receptor,
        binder_positions=[0, 1, 2],
        receptor_positions=[3, 4, 5],
    )


def make_config(**overrides):
    cfg = {
        "conditions": SPEC["conditions"],
        "terms": THREE_TERMS,
        "designable": {"chains": ["A"]},
        "block_size": 2,
    }
    return SwitchDesignConfig.from_dict({**cfg, **overrides})


def valid_designs():
    for values in itertools.product([0, 1], repeat=len(DESIGN_BLOCK)):  # ALA, bare HIS
        tokens = BASE.clone()
        tokens[DESIGN_BLOCK] = torch.tensor(values)
        yield tokens


def test_single_objective_run_reaches_the_exact_optimum():
    run = run_switch_design(make_inputs(), make_config(allow_bare_titratable=["HIS"]))
    assert run.designable == DESIGN_BLOCK  # the His site is set by the conditions
    best = run.result.records[0]
    weights = (0.3, 0.3, 0.4)
    names = ("stability", "potency", "switch")
    brute = min(
        run.objective.scalarised_value(t, names, weights) for t in valid_designs()
    )
    assert best.scalarised == pytest.approx(brute)
    assert run.result.plan.mode == "single"
    assert int(best.tokens[0]) == int(
        BASE[0]
    )  # a condition-set site keeps its input token


def test_design_space_excludes_titratable_residues_unless_allowed():
    default = run_switch_design(
        make_inputs(), make_config(search="pareto", divisions=2)
    )
    for record in default.result.records:
        assert all(int(t) == 0 for t in record.tokens[DESIGN_BLOCK])  # Ala only
    allowed = run_switch_design(
        make_inputs(),
        make_config(search="pareto", divisions=2, allow_bare_titratable=["HIS"]),
    )
    seen = {int(t) for r in allowed.result.records for t in r.tokens[DESIGN_BLOCK]}
    assert seen <= {0, 1} and 1 in seen  # bare His appears; no state tokens (2, 3)


def test_pareto_runs_only_when_requested():
    single = run_switch_design(make_inputs(), make_config())
    pareto = run_switch_design(make_inputs(), make_config(search="pareto", divisions=2))
    assert single.result.plan.mode == "single" and single.result.hypervolume is None
    assert pareto.result.plan.mode == "pareto" and pareto.result.hypervolume > 0
    assert all(r.front is not None for r in pareto.result.records)


def test_swapping_on_and_off_reverses_the_designed_switch():
    options = dict(terms={"switch": {"weight": 1.0}}, allow_bare_titratable=["HIS"])
    forward = run_switch_design(make_inputs(), make_config(**options))
    reverse = run_switch_design(
        make_inputs(), make_config(**options, on="off", off="on")
    )
    fwd_values = [
        forward.objective.terms["switch"].value(forward.objective.ctx, t)
        for t in valid_designs()
    ]
    assert forward.result.records[0].term_values["switch"] == pytest.approx(
        min(fwd_values)
    )
    assert reverse.result.records[0].term_values["switch"] == pytest.approx(
        -max(fwd_values)
    )


def test_linked_equilibrium_runs_from_the_same_config():
    config = make_config(
        binding_model="linked_equilibrium", binding_options={"beta": 0.7}
    )
    run = run_switch_design(make_inputs(), config)
    best = run.result.records[0]
    assert best.term_values["potency"] == pytest.approx(
        run.objective.terms["potency"].value(run.objective.ctx, best.tokens)
    )
    assert run.objective.ctx.binding.name == "linked_equilibrium"


def test_complex_gap_and_option_validation():
    run = run_switch_design(make_inputs(), make_config(binding_model="complex_gap"))
    assert run.objective.ctx.binding.name == "complex_gap"
    with pytest.raises(ValueError, match="takes no binding_options"):
        run_switch_design(make_inputs(), make_config(binding_options={"beta": 1.0}))
    with pytest.raises(ValueError, match="Unknown linked_equilibrium options"):
        run_switch_design(
            make_inputs(),
            make_config(
                binding_model="linked_equilibrium", binding_options={"temp": 1}
            ),
        )
    with pytest.raises(ValueError, match="Unknown binding model"):
        run_switch_design(make_inputs(), make_config(binding_model="magic"))


def test_resolve_designable_chains_includes_excludes_and_condition_sites(caplog):
    spec_sites = []
    assert resolve_designable(DesignableSpec(chains=["A"]), CHAINS, RES_IDS) == [
        0,
        1,
        2,
    ]
    assert resolve_designable(
        DesignableSpec(chains=["A"], exclude=["A:1"], include=["B:3"]), CHAINS, RES_IDS
    ) == [1, 2, 5]
    from mpnn.ph.states import SiteKey

    with caplog.at_level("WARNING", logger="mpnn.ph.session"):
        got = resolve_designable(
            DesignableSpec(chains=["A"]),
            CHAINS,
            RES_IDS,
            [SiteKey("A", 1)] + spec_sites,
        )
    assert got == [1, 2] and "set by a condition" in caplog.text


@pytest.mark.parametrize(
    "spec",
    [
        DesignableSpec(chains=["Z"]),
        DesignableSpec(include=["A:99"]),
        DesignableSpec(),
    ],
)
def test_resolve_designable_rejects_bad_input(spec):
    with pytest.raises(ValueError):
        resolve_designable(spec, CHAINS, RES_IDS)


def test_describe_renders_one_chain_of_a_design():
    run = run_switch_design(make_inputs(), make_config(allow_bare_titratable=["HIS"]))
    best = run.result.records[0]
    binder = run.describe(best, "A")
    assert binder["res_ids"] == [1, 2, 3]
    assert (
        len(binder["canonical_sequence"]) == 3 and len(binder["extended_tokens"]) == 3
    )
    assert (
        binder["canonical_sequence"][0] == "H"
    )  # the condition-set His keeps its token
    assert set(binder["extended_tokens"][1:]) <= {"ALA", "HIS"}
    assert run.describe(best, "B")["canonical_sequence"] == "AHA"
    with pytest.raises(ValueError, match="not in the structure"):
        run.describe(best, "Z")


def test_to_rows_lists_ranked_designs_with_terms_and_mutations():
    config = make_config(allow_bare_titratable=["HIS"], search="pareto", divisions=2)
    run = run_switch_design(make_inputs(), config)
    rows = run.to_rows("A")
    assert [r["rank"] for r in rows] == list(range(len(run.result.records)))
    for row in rows:
        assert {"canonical_sequence", "n_mutations", "front", "selected"} <= set(row)
        assert {"term_stability", "term_potency", "term_switch"} <= set(row)
        assert row["mode"] == "pareto"
    input_sequence = "HAA"  # the binder tokens of the toy structure
    first = rows[0]["canonical_sequence"]
    assert rows[0]["n_mutations"] == sum(a != b for a, b in zip(input_sequence, first))
    with pytest.raises(ValueError, match="chain length"):
        run.to_rows("A", reference_sequence="HA")


def identical_conditions(ph_on=7.4, ph_off=6.5):
    states = {"A:1": "deprotonated", "B:2": "deprotonated"}
    return {
        "on": {"ph": ph_on, "states": dict(states)},
        "off": {"ph": ph_off, "states": dict(states)},
    }


def test_identical_on_and_off_states_cannot_make_a_switch():
    """With the same states in both conditions a state-based switch term is always zero."""
    with pytest.raises(ValueError, match="identical states"):
        run_switch_design(make_inputs(), make_config(conditions=identical_conditions()))
    # no switch term: a single condition's states are all that is used
    run_switch_design(
        make_inputs(),
        make_config(
            conditions=identical_conditions(), terms={"potency": {"weight": 1.0}}
        ),
    )


def test_identical_states_are_fine_for_linked_equilibrium_when_the_ph_differs():
    options = dict(binding_model="linked_equilibrium", binding_options={"beta": 0.7})
    run_switch_design(
        make_inputs(), make_config(conditions=identical_conditions(), **options)
    )
    with pytest.raises(ValueError, match="same pH"):
        run_switch_design(
            make_inputs(),
            make_config(conditions=identical_conditions(7.0, 7.0), **options),
        )


def test_linked_equilibrium_tells_conditions_apart_only_by_ph():
    """It sums over every imposed site, so different explicit states do not make a switch."""
    conditions = deepcopy(SPEC["conditions"])
    for body in conditions.values():
        body["ph"] = 7.0
    options = dict(
        binding_model="linked_equilibrium", terms={"switch": {"weight": 1.0}}
    )
    with pytest.raises(ValueError, match="same pH"):
        run_switch_design(make_inputs(), make_config(conditions=conditions, **options))
    no_sites = {"on": {"ph": 7.4, "states": {}}, "off": {"ph": 6.5, "states": {}}}
    with pytest.raises(ValueError, match="no state on any site"):
        run_switch_design(make_inputs(), make_config(conditions=no_sites, **options))


def test_the_switch_check_uses_the_conditions_the_term_selects():
    selected = {"terms": {"switch": {"on": "actual_on", "off": "actual_off"}}}
    # the spec's own on/off are identical, the term's are not: a valid switch
    conditions = identical_conditions()
    conditions["actual_on"] = {"states": {"A:1": "deprotonated", "B:2": "deprotonated"}}
    conditions["actual_off"] = {"states": {"A:1": "protonated", "B:2": "protonated"}}
    run_switch_design(make_inputs(), make_config(conditions=conditions, **selected))
    # the spec's on/off differ, the term's are identical: refused
    conditions = deepcopy(SPEC["conditions"])
    conditions["actual_on"] = {"states": {"A:1": "protonated", "B:2": "protonated"}}
    conditions["actual_off"] = {"states": {"A:1": "protonated", "B:2": "protonated"}}
    with pytest.raises(ValueError, match="'actual_on' and 'actual_off'.*identical"):
        run_switch_design(make_inputs(), make_config(conditions=conditions, **selected))


def _design_with_forbidden_energy(forbidden_stability):
    """Only Ala and Gly may be designed; the other three tokens only differ in stability."""
    stability = [0.0, 1.0] + [forbidden_stability] * 3
    potency = [1.0, 0.0, 1.0, 1.0, 1.0]
    inputs = DesignInputs(
        tokens=torch.tensor([0, 4]),
        table=TokenTable(["ALA", "GLY", "HIS", "HIS-P", "HIS-S"]),
        chain_ids=["A", "B"],
        res_ids=[1, 1],
        parents=["ALA", "HIS"],
        complex_scorer=FunctionSystem(
            lambda t: stability[int(t[0])] + potency[int(t[0])], 5
        ),
        binder_scorer=FunctionSystem(lambda t: stability[int(t[0])], 5),
        receptor_scorer=FunctionSystem(lambda t: 0, 5),
        binder_positions=[0],
        receptor_positions=[1],
    )
    config = SwitchDesignConfig.from_dict(
        {
            "conditions": {"on": {"states": {"B:1": "deprotonated"}}},
            "terms": {"stability": {"weight": 0.6}, "potency": {"weight": 0.4}},
            "designable": {"chains": ["A"]},
            "block_size": 1,
        }
    )
    run = run_switch_design(inputs, config)
    return run.describe(run.result.records[0], "A")["canonical_sequence"]


def test_energies_of_forbidden_tokens_do_not_change_the_design():
    assert _design_with_forbidden_energy(0) == _design_with_forbidden_energy(10000)
