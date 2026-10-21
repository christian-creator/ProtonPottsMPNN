"""Tests for objective terms, scalarisation and search planning (atomworks-free)."""

import logging

import numpy as np
import pytest
import torch
from objective_fixtures import (
    BASE,
    DESIGN_BLOCK,
    NAMES_ACTIVE,
    assignments,
    make_context,
    make_objective,
)
from potts_fixtures import V

from mpnn.ph.objective import (
    PotencyTerm,
    StabilityTerm,
    SwitchTerm,
    TermContext,
    TermScale,
    build_terms,
    choose_assignment,
    estimate_term_scale,
    mask_joint,
    plan_search,
)


@pytest.fixture
def ctx():
    return make_context()


ALL_TERMS = [
    StabilityTerm(),
    StabilityTerm(reduce="mean"),
    StabilityTerm(where="complex"),
    PotencyTerm(),
    SwitchTerm(),
    SwitchTerm(off_margin=2.0),
]


@pytest.mark.parametrize("term", ALL_TERMS, ids=lambda t: f"{t.name}-{id(t) % 97}")
def test_block_values_equal_full_sequence_values(ctx, term):
    joint = term.block_values(ctx, BASE, DESIGN_BLOCK)
    for values, tokens in assignments():
        assert joint[values].item() == pytest.approx(term.value(ctx, tokens), abs=1e-9)


def test_potency_is_the_binding_energy_in_the_on_condition(ctx):
    on_tokens = ctx.tokens_in(BASE, "on")
    assert PotencyTerm().value(ctx, BASE) == pytest.approx(
        ctx.binding.energy(on_tokens)
    )
    assert PotencyTerm(condition="off").value(ctx, BASE) == pytest.approx(
        ctx.binding.energy(ctx.tokens_in(BASE, "off"))
    )


def test_stability_reduction_over_conditions(ctx):
    per_condition = [
        ctx.binder_view.block_potentials(ctx.tokens_in(BASE, c), DESIGN_BLOCK).joint()
        for c in ("on", "off")
    ]
    worst = StabilityTerm(reduce="max").block_values(ctx, BASE, DESIGN_BLOCK)
    mean = StabilityTerm(reduce="mean").block_values(ctx, BASE, DESIGN_BLOCK)
    only_on = StabilityTerm(conditions=["on"]).block_values(ctx, BASE, DESIGN_BLOCK)
    assert torch.allclose(worst, torch.maximum(*per_condition))
    assert torch.allclose(mean, (per_condition[0] + per_condition[1]) / 2)
    assert torch.allclose(only_on, per_condition[0])
    assert (worst >= mean - 1e-12).all()


def test_swapping_on_and_off_negates_the_switch():
    forward, reverse = make_context(), make_context(swap_roles=True)
    term = SwitchTerm()
    assert term.value(reverse, BASE) == pytest.approx(-term.value(forward, BASE))
    assert torch.allclose(
        term.block_values(reverse, BASE, DESIGN_BLOCK),
        -term.block_values(forward, BASE, DESIGN_BLOCK),
    )


def test_switch_hinge_is_bounded_and_stops_rewarding(ctx):
    margin = 0.2  # the widest gap G_off - G_on in this system is about 0.28
    plain = SwitchTerm().block_values(ctx, BASE, DESIGN_BLOCK)  # = -(G_off - G_on)
    hinged = SwitchTerm(off_margin=margin).block_values(ctx, BASE, DESIGN_BLOCK)
    assert (hinged >= 0).all()
    assert torch.allclose(hinged, torch.relu(margin + plain))
    wide_gap = plain <= -margin
    assert wide_gap.any() and (hinged[wide_gap] == 0).all()
    with pytest.raises(ValueError):
        SwitchTerm(off_margin=0.0)


def test_missing_roles_raise():
    ctx = make_context()
    no_roles = TermContext(
        ctx.binding, ctx.binder_view, ctx.complex_view, ctx.conditions
    )
    with pytest.raises(ValueError, match="'on'"):
        PotencyTerm().value(no_roles, BASE)
    with pytest.raises(KeyError, match="nowhere"):
        PotencyTerm(condition="nowhere").value(ctx, BASE)


# ---- scalarisation against brute force ----------------------------------------
def brute_force_joint(objective, weights, scalarisation):
    joint = torch.zeros((V,) * len(DESIGN_BLOCK), dtype=torch.float64)
    for values, tokens in assignments():
        z = []
        for name, weight in zip(NAMES_ACTIVE, weights):
            scale = objective.scales[name]
            raw = objective.terms[name].value(objective.ctx, tokens)
            z.append((weight, (raw - scale.center) / scale.scale, scale.ideal))
        if scalarisation == "weighted_sum":
            joint[values] = sum(w * zk for w, zk, _ in z)
        else:
            joint[values] = max(w * (zk - ideal) for w, zk, ideal in z if w > 0)
    return joint


@pytest.mark.parametrize("scalarisation", ["weighted_sum", "tchebycheff"])
@pytest.mark.parametrize("weights", [(0.2, 0.3, 0.5), (1.0, 0.0, 0.0), (0.0, 0.5, 0.5)])
def test_scalarised_joint_matches_brute_force(ctx, scalarisation, weights):
    objective = make_objective(ctx)
    joint = objective.scalarised_joint(
        BASE, DESIGN_BLOCK, NAMES_ACTIVE, weights, scalarisation
    )
    expected = brute_force_joint(objective, weights, scalarisation)
    assert torch.allclose(joint, expected, atol=1e-9)
    mask = torch.tensor([True, True, False, False])  # keep titratable states out
    chosen = objective.best_assignment(
        BASE, DESIGN_BLOCK, NAMES_ACTIVE, weights, scalarisation, valid_mask=mask
    )
    assert chosen == choose_assignment(expected, mask)


def test_one_active_term_is_single_objective(ctx):
    objective = make_objective(ctx)
    plain = objective.terms["switch"].block_values(ctx, BASE, DESIGN_BLOCK)
    for names, weights in ((["switch"], [1.0]), (NAMES_ACTIVE, [0.0, 0.0, 1.0])):
        chosen = objective.best_assignment(BASE, DESIGN_BLOCK, names, weights)
        assert chosen == choose_assignment(plain)


def test_reversing_the_roles_reverses_the_objective():
    forward, reverse = (
        make_objective(make_context()),
        make_objective(make_context(True)),
    )
    joint_f = forward.scalarised_joint(BASE, DESIGN_BLOCK, ["switch"], [1.0])
    joint_r = reverse.scalarised_joint(BASE, DESIGN_BLOCK, ["switch"], [1.0])
    assert torch.allclose(joint_r, -joint_f, atol=1e-9)


def test_term_matrix_feeds_pareto_selection(ctx):
    objective = make_objective(ctx)
    designs = [tokens for _, tokens in assignments()]
    matrix = objective.term_matrix(designs, NAMES_ACTIVE)
    assert matrix.shape == (len(designs), 3)
    assert np.isfinite(matrix).all()


# ---- search planning ------------------------------------------------------------
def test_one_active_term_collapses_even_when_pareto_is_requested(caplog):
    with caplog.at_level(logging.INFO, logger="mpnn.ph.objective"):
        plan = plan_search({"stability": 0.0, "switch": 2.0}, search="pareto")
    assert (plan.mode, plan.term_names) == ("single", ("switch",))
    assert plan.weight_vectors.tolist() == [[1.0]]
    assert plan.scalarisation == "weighted_sum"
    assert "single-objective" in caplog.text


def test_explicit_weights_make_one_run():
    plan = plan_search({"stability": 1.0, "potency": 0.0, "switch": 3.0})
    assert plan.mode == "single"
    assert plan.term_names == ("stability", "switch")
    assert plan.weight_vectors.tolist() == [[0.25, 0.75]]


def test_pareto_sweeps_the_weight_simplex_over_active_terms():
    weights = {"stability": 1.0, "potency": 1.0, "switch": 1.0}
    plan = plan_search(weights, search="pareto")
    assert plan.mode == "pareto" and plan.scalarisation == "tchebycheff"
    assert plan.weight_vectors.shape == (15, 3)
    assert np.allclose(plan.weight_vectors.sum(axis=1), 1.0)
    coarse = plan_search(
        weights, search="pareto", scalarisation="weighted_sum", divisions=2
    )
    assert (
        coarse.weight_vectors.shape == (6, 3) and coarse.scalarisation == "weighted_sum"
    )
    two = plan_search({"potency": 1.0, "switch": 1.0}, search="pareto")
    assert two.weight_vectors.shape == (5, 2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weights": {"switch": 0.0}},
        {"weights": {}},
        {"weights": {"switch": -1.0}},
        {"weights": {"switch": 1.0}, "search": "genetic"},
        {"weights": {"switch": 1.0}, "scalarisation": "product"},
    ],
)
def test_plan_search_rejects_bad_input(kwargs):
    with pytest.raises(ValueError):
        plan_search(**kwargs)


def test_build_terms_keeps_only_active_terms():
    weights, terms = build_terms(
        {
            "stability": {"weight": 0.3, "reduce": "mean"},
            "potency": {"weight": 0},
            "switch": {"weight": 0.4, "off_margin": 2.0},
        }
    )
    assert weights == {"stability": 0.3, "switch": 0.4}
    assert set(terms) == {"stability", "switch"}
    assert terms["stability"].reduce == "mean" and terms["switch"].off_margin == 2.0
    for bad in (
        {"nonsense": {}},
        {"switch": {"weight": -1}},
        {"stability": {"reduce": "median"}},
    ):
        with pytest.raises(ValueError):
            build_terms(bad)


# ---- choosing and scaling ----------------------------------------------------------
def test_choose_assignment_argmin_mask_and_empty():
    joint = torch.tensor([[3.0, 1.0, 2.0], [0.5, 4.0, 5.0], [6.0, 7.0, 8.0]])
    assert choose_assignment(joint) == (1, 0)
    assert choose_assignment(joint, torch.tensor([True, True, True])) == (1, 0)
    # forbidding token 0 at every position leaves rows/cols 1-2, whose best is (1, 1)
    assert choose_assignment(joint, torch.tensor([False, True, True])) == (1, 1)
    with pytest.raises(ValueError, match="No valid"):
        choose_assignment(joint, torch.tensor([False, False, False]))
    assert choose_assignment(torch.tensor(1.5)) == ()


def sample_first_tokens(joint, temperature, first_seed, n=60):
    return [
        choose_assignment(
            joint,
            temperature=temperature,
            generator=torch.Generator().manual_seed(first_seed + k),
        )[0]
        for k in range(n)
    ]


def test_sampling_is_reproducible_and_sharpens_as_temperature_drops():
    joint = torch.tensor([0.0, 0.0, 5.0])
    assert sample_first_tokens(joint, 1.0, 0) == sample_first_tokens(joint, 1.0, 0)
    assert {0, 1} <= set(sample_first_tokens(joint, 1.0, 0))
    assert 2 not in sample_first_tokens(joint, 0.05, 0)


def test_estimate_term_scale_uses_finite_spread_and_best_value():
    scale = estimate_term_scale(
        [torch.tensor([0.0, 2.0, 4.0]), torch.tensor([float("inf")])], center=1.0
    )
    spread = float(torch.tensor([0.0, 2.0, 4.0]).std(unbiased=False))
    assert scale.center == 1.0 and scale.scale == pytest.approx(spread)
    assert scale.ideal == pytest.approx((0.0 - 1.0) / spread)
    assert estimate_term_scale([torch.tensor([3.0])], center=3.0) == TermScale(
        center=3.0
    )


@pytest.mark.parametrize(
    "term", [PotencyTerm(), SwitchTerm(), SwitchTerm(off_margin=0.2)]
)
def test_linked_equilibrium_drives_terms_through_each_conditions_ph(term):
    ctx = make_context(binding_model="linked_equilibrium")
    joint = term.block_values(ctx, BASE, DESIGN_BLOCK)
    for values, tokens in assignments():
        assert joint[values].item() == pytest.approx(term.value(ctx, tokens), abs=1e-9)
    on_ph = ctx.ph_of("on")
    assert on_ph == 7.4
    expected = ctx.binding.energy(ctx.tokens_in(BASE, "on"), ph=on_ph)
    assert PotencyTerm().value(ctx, BASE) == pytest.approx(expected)


def test_plan_search_refuses_non_finite_weights():
    with pytest.raises(ValueError, match="finite"):
        plan_search({"stability": 1.0, "switch": float("nan")})


def test_mask_joint_forbids_a_token_at_every_block_position():
    joint = torch.arange(9, dtype=torch.float64).reshape(3, 3)
    masked = mask_joint(joint, torch.tensor([True, False, True]))
    assert torch.isinf(masked[1]).all() and torch.isinf(masked[:, 1]).all()
    assert masked[0, 0] == 0 and masked[2, 2] == 8
    assert mask_joint(joint, None) is joint
