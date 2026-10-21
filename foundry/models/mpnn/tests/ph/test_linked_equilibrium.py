"""Closed-form and limit tests for the linked-equilibrium binding model."""

import itertools
import math

import pytest
import torch
from potts_fixtures import FunctionSystem, IdealComplex, V

from mpnn.ph.binding import (
    LinkedEquilibrium,
    StateBinding,
    SystemView,
    TitratableSite,
    linked_sites_from_spec,
    make_binding_model,
)
from mpnn.ph.states import StateSpec, site_index_from_arrays
from mpnn.ph.vocab_meta import TokenTable

P, D = 2, 3  # HIS-P, HIS-S in the four-token test vocabulary
LN10 = math.log(10.0)
PH_GRID = [2.0, 5.0, 6.5, 7.4, 9.0, 12.0]


def two_state(table):
    """Energy lookup by the token at local position 0 (zero for any other token)."""
    return lambda tokens: table.get(int(tokens[0]), 0.0)


def single_site_model(g_p, g_d, pka=6.5, beta=1.0, intrinsic_p=0.0):
    """Site 0 binds a fixed receptor residue (position 1) with G_bind = g_p / g_d.

    ``intrinsic_p`` adds the same energy to the protonated state in the complex and in
    the free binder, so it changes the unbound populations but not G_bind.
    """
    binder = FunctionSystem(two_state({P: intrinsic_p}))
    complex_ = FunctionSystem(
        lambda t: two_state({P: g_p + intrinsic_p, D: g_d})(t[[0]])
    )
    receptor = FunctionSystem(lambda t: 0.0)
    return LinkedEquilibrium(
        SystemView(complex_, [0, 1], V),
        SystemView(binder, [0], V),
        SystemView(receptor, [1], V),
        [TitratableSite(0, P, D, pka)],
        beta=beta,
    )


def closed_form(g_p, g_d, ph, pka=6.5, beta=1.0):
    frac_p = 1.0 / (1.0 + 10.0 ** (ph - pka))
    return (
        -math.log(frac_p * math.exp(-beta * g_p) + (1 - frac_p) * math.exp(-beta * g_d))
        / beta
    )


TOKENS = torch.tensor([D, 0])


@pytest.mark.parametrize("beta", [1.0, 2.5])
@pytest.mark.parametrize("ph", PH_GRID)
def test_single_site_matches_the_two_state_formula(ph, beta):
    model = single_site_model(-3.0, -0.5, beta=beta)
    assert model.energy(TOKENS, ph=ph) == pytest.approx(
        closed_form(-3.0, -0.5, ph, beta=beta), abs=1e-9
    )


def test_ph_limits_recover_the_single_state_energies():
    model = single_site_model(-3.0, -0.5)
    assert model.energy(TOKENS, ph=-60.0) == pytest.approx(-3.0)  # all protonated
    assert model.energy(TOKENS, ph=60.0) == pytest.approx(-0.5)  # all deprotonated


@pytest.mark.parametrize("ph", PH_GRID)
def test_intrinsic_state_preferences_do_not_create_binding(ph):
    """A state the free binder already prefers shifts populations, not binding."""
    model = single_site_model(0.0, 0.0, intrinsic_p=-4.0)
    assert model.energy(TOKENS, ph=ph) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("ph", PH_GRID)
def test_decoupled_partners_have_zero_binding_at_any_ph(ph):
    ideal = IdealComplex(1, 2, 3, interface_scale=0.0)
    sites = [TitratableSite(0, P, D, 6.5), TitratableSite(4, P, D, 6.5)]
    model = LinkedEquilibrium(*ideal.views(), sites)
    tokens = torch.tensor([1, 0, 0, 0, 1, 0])
    assert model.energy(tokens, ph=ph) == pytest.approx(0.0, abs=1e-9)


def test_independent_sites_add():
    gaps = {0: (-2.0, 0.5), 1: (-0.3, -1.4)}  # (g_p, g_d) per site
    pkas = {0: 6.5, 1: 4.0}

    def interface(tokens):
        return sum(
            {P: gaps[i][0], D: gaps[i][1]}.get(int(tokens[i]), 0.0) for i in (0, 1)
        )

    complex_ = FunctionSystem(interface)
    zero = FunctionSystem(lambda t: 0.0)
    model = LinkedEquilibrium(
        SystemView(complex_, [0, 1, 2], V),
        SystemView(zero, [0, 1], V),
        SystemView(zero, [2], V),
        [TitratableSite(0, P, D, pkas[0]), TitratableSite(1, P, D, pkas[1])],
    )
    tokens = torch.tensor([D, D, 0])
    for ph in PH_GRID:
        expected = sum(closed_form(*gaps[i], ph, pka=pkas[i]) for i in (0, 1))
        assert model.energy(tokens, ph=ph) == pytest.approx(expected, abs=1e-9)


def test_tighter_binding_state_raises_its_pka_on_binding():
    """Textbook linkage: pKa_bound - pKa_free = beta * (g_d - g_p) / ln 10."""
    g_p, g_d, beta, ph = -3.0, -0.5, 1.7, 6.0
    model = single_site_model(g_p, g_d, beta=beta)
    pops = model.site_populations(TOKENS, ph)[0]

    def apparent_pka(fraction):
        return ph + math.log10(fraction / (1 - fraction))

    shift = apparent_pka(pops["bound"]) - apparent_pka(pops["free"])
    assert shift == pytest.approx(beta * (g_d - g_p) / LN10)
    assert pops["bound"] > pops["free"]
    assert pops["free"] == pytest.approx(1.0 / (1.0 + 10.0 ** (ph - 6.5)))


def test_extreme_ph_equals_state_binding_with_every_site_set():
    ideal = IdealComplex(1, 2, 3)
    complex_view, binder_view, receptor_view = ideal.views()
    sites = [TitratableSite(0, P, D, 6.5), TitratableSite(4, P, D, 6.5)]
    model = LinkedEquilibrium(complex_view, binder_view, receptor_view, sites)
    reference = StateBinding(complex_view, binder_view, receptor_view)
    base = torch.tensor([1, 0, 0, 0, 1, 0])
    for ph, token in ((-80.0, P), (80.0, D)):
        forced = base.clone()
        forced[[0, 4]] = token
        assert model.energy(base, ph=ph) == pytest.approx(
            reference.energy(forced), abs=1e-9
        )


def test_block_values_match_per_assignment_energies():
    ideal = IdealComplex(1, 2, 3)
    sites = [TitratableSite(0, P, D, 6.5), TitratableSite(4, P, D, 6.5)]
    model = LinkedEquilibrium(*ideal.views(), sites, beta=0.8)
    base = torch.tensor([1, 0, 0, 0, 1, 0])
    block = [1, 2]
    joint = model.block_values(base, block, ph=6.9)
    for values in itertools.product(range(V), repeat=2):
        tokens = base.clone()
        tokens[block] = torch.tensor(values)
        assert joint[values].item() == pytest.approx(
            model.energy(tokens, ph=6.9), abs=1e-9
        )


def test_validation_errors():
    ideal = IdealComplex(1, 2, 3)
    views = ideal.views()
    site = TitratableSite(0, P, D, 6.5)
    model = LinkedEquilibrium(*views, [site])
    base = torch.tensor([1, 0, 0, 0, 1, 0])
    with pytest.raises(ValueError, match="needs the pH"):
        model.energy(base)
    with pytest.raises(ValueError, match="titratable sites"):
        model.block_values(base, [0, 1], ph=7.0)
    with pytest.raises(ValueError, match="beta"):
        LinkedEquilibrium(*views, [site], beta=0.0)
    with pytest.raises(ValueError, match="cap"):
        LinkedEquilibrium(*views, [site, TitratableSite(4, P, D, 6.5)], max_sites=1)
    with pytest.raises(ValueError, match="distinct"):
        LinkedEquilibrium(*views, [site, site])
    with pytest.raises(ValueError, match="inside"):
        LinkedEquilibrium(*views, [TitratableSite(9, P, D, 6.5)])
    with pytest.raises(NotImplementedError):
        model.block_potentials(base, [1])


def test_sites_come_from_the_spec_with_reference_pka():
    table = TokenTable(("ALA", "HIS", "HIS-P", "HIS-S", "ASP", "ASP-P", "ASP-D"))
    index = site_index_from_arrays(["A", "A", "B"], [1, 2, 1])
    parents = ["HIS", "ASP", "HIS"]
    spec = StateSpec.from_dict(
        {
            "conditions": {
                "on": {"states": {"A:1": "deprotonated", "A:2": "ASP-D"}},
                "off": {"states": {"A:1": "protonated", "B:1": "HIS-P"}},
            }
        }
    )
    sites = linked_sites_from_spec(spec, table, index, parents)
    assert [(s.position, s.pka) for s in sites] == [(0, 6.5), (1, 3.7), (2, 6.5)]
    assert sites[1].protonated_token == table.index("ASP-P")
    with pytest.raises(KeyError, match="reference pKa"):
        linked_sites_from_spec(spec, table, index, parents, pka_by_parent={"HIS": 6.5})


def test_factory_builds_the_model():
    ideal = IdealComplex(1, 2, 3)
    complex_view, binder_view, receptor_view = ideal.views()
    model = make_binding_model(
        "linked_equilibrium",
        complex_view=complex_view,
        binder_view=binder_view,
        receptor_view=receptor_view,
        sites=[TitratableSite(0, P, D, 6.5)],
    )
    assert model.name == "linked_equilibrium"
