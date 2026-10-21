"""Wiring of PottsMPNNPHEngine.run_switch_design, with real scorers on synthetic data.

Only the step that needs HBPLUS (re-encoding a chain alone) is replaced by a fake that
returns real ``_PottsScorer`` objects built from random tables.
"""

import itertools

import pytest
import torch
from engine_fixtures import CHAINS, RES_IDS, bare_engine, make_ctx
from objective_fixtures import BASE, DESIGN_BLOCK, NAMES, PARENTS, SPEC

CONFIG = {
    "conditions": SPEC["conditions"],
    "terms": {
        "stability": {"weight": 0.3},
        "potency": {"weight": 0.3},
        "switch": {"weight": 0.4},
    },
    "designable": {"chains": ["A"]},
    "allow_bare_titratable": ["HIS"],
    "block_size": 2,
}


@pytest.fixture
def wired(engine):
    ctx = make_ctx(engine)
    return bare_engine(engine, ctx), ctx


def test_design_inputs_describe_the_complex(wired):
    engine_obj, ctx = wired
    inputs = engine_obj._design_inputs(ctx, "A")
    assert torch.equal(inputs.tokens, BASE)
    assert inputs.table.names == tuple(NAMES)
    assert inputs.chain_ids == CHAINS and inputs.res_ids == RES_IDS
    assert inputs.parents == PARENTS  # states would fold onto their residue
    assert inputs.binder_positions == [0, 1, 2] and inputs.receptor_positions == [
        3,
        4,
        5,
    ]
    assert inputs.partner_rank(0) == [int(j) for j in ctx.eidx_np[0] if j != 0]
    assert (inputs.neighbour_index == ctx.eidx_np).all()  # for designable.near


def test_design_inputs_need_a_receptor_chain(wired):
    engine_obj, ctx = wired
    with pytest.raises(ValueError, match="at least one other chain"):
        engine_obj._design_inputs(ctx, "Z")


def valid_designs():
    for values in itertools.product([0, 1], repeat=len(DESIGN_BLOCK)):
        tokens = BASE.clone()
        tokens[DESIGN_BLOCK] = torch.tensor(values)
        yield tokens


def test_run_switch_design_reaches_the_exact_optimum_with_real_scorers(wired):
    engine_obj, _ = wired
    run = engine_obj.run_switch_design(atom_array=None, binder_chain="A", config=CONFIG)
    assert run.designable == DESIGN_BLOCK
    best = run.result.records[0]
    names, weights = ("stability", "potency", "switch"), (0.3, 0.3, 0.4)
    brute = min(
        run.objective.scalarised_value(t, names, weights) for t in valid_designs()
    )
    assert best.scalarised == pytest.approx(brute, abs=1e-6)
    fixed = [p for p in range(6) if p not in DESIGN_BLOCK]
    assert torch.equal(best.tokens[fixed], BASE[fixed])


def test_run_switch_design_accepts_the_pareto_option(wired):
    engine_obj, _ = wired
    run = engine_obj.run_switch_design(
        atom_array=None,
        binder_chain="A",
        config={**CONFIG, "search": "pareto", "divisions": 2},
    )
    assert run.result.plan.mode == "pareto" and run.result.hypervolume > 0


def test_ensemble_design_builds_inputs_for_each_structure(engine):
    contexts = {"human": make_ctx(engine), "mouse": make_ctx(engine)}
    runner = bare_engine(engine, contexts["human"])
    runner._build_context = (
        lambda atom_array, chain, base_seed=0, protonation=None: contexts[atom_array]
    )
    run = runner.run_switch_design_ensemble(
        structures={"human": "human", "mouse": "mouse"},
        binder_chain="A",
        config=CONFIG,
    )
    assert list(run.members) == ["human", "mouse"]
    assert run.designable == DESIGN_BLOCK
    assert set(run.result.records[0].per_state) == {"human", "mouse"}


@pytest.mark.parametrize("method", ["single", "ensemble"])
@pytest.mark.parametrize(
    "seed, base_seed, expected",
    [(7, 3, 7), (None, 3, 3), (0, 3, 0), (None, 0, 0)],  # an explicit 0 is explicit
)
def test_the_seed_argument_overrides_the_configs_base_seed(
    engine, monkeypatch, method, seed, base_seed, expected
):
    ctx = make_ctx(engine)
    runner = bare_engine(engine, ctx)
    context_seeds, search_configs = [], []
    runner._build_context = lambda atom_array, chain, base_seed=0, protonation=None: (
        context_seeds.append(base_seed) or ctx
    )

    def spy(inputs, config):
        search_configs.append(config)
        return "run"

    monkeypatch.setattr(engine, "_run_switch_design", spy)
    monkeypatch.setattr(engine, "_run_switch_design_ensemble", spy)
    config = {**CONFIG, "base_seed": base_seed}
    kwargs = {} if seed is None else {"seed": seed}
    if method == "single":
        runner.run_switch_design(
            atom_array=None, binder_chain="A", config=config, **kwargs
        )
    else:
        runner.run_switch_design_ensemble(
            structures={"human": None}, binder_chain="A", config=config, **kwargs
        )
    assert [c.base_seed for c in search_configs] == [expected]  # the search stream
    assert context_seeds == [expected]  # and the featurisation use the same seed
