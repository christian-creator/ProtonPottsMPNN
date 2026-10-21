"""Characterisation of the engine's Potts scorer and selectivity helpers.

These run the real ``potts_mpnn_ph`` code (with missing third-party packages stubbed by
``engine_imports``) on synthetic Potts tables. They pin current behaviour so that the
state-spec refactor can be checked against it.
"""

import itertools
from types import SimpleNamespace

import pytest
import torch
from potts_fixtures import V, random_graph_potts

from mpnn.ph.potentials import BlockPotentials, probe_block_potentials

L = 8


@pytest.fixture
def scorer(engine):
    etab, e_idx = random_graph_potts(L, 4, V, seed=3)
    return engine._PottsScorer(etab, e_idx)


@pytest.fixture
def tokens():
    return torch.tensor([0, 1, 2, 3, 1, 0, 2, 3])


def test_row_slice_equals_full_conditional_table(scorer, tokens):
    full = scorer.cond_energy(tokens)
    for i in range(L):
        assert torch.allclose(scorer.cond_energy_at(tokens, i), full[i], atol=1e-10)


def test_conditional_energy_differences_are_hamiltonian_differences(scorer, tokens):
    full = scorer.cond_energy(tokens)
    base = scorer.H_of(tokens)
    for p, a in itertools.product(range(L), range(V)):
        mutated = tokens.clone()
        mutated[p] = a
        assert scorer.H_of(mutated) - base == pytest.approx(
            float(full[p, a] - full[p, tokens[p]]), abs=1e-9
        )


@pytest.mark.parametrize("block", [[3], [1, 5], [0, 2, 6], [4, 5, 6, 7]])
def test_block_decomposition_matches_the_probing_oracle(scorer, tokens, block):
    unary, edges = scorer.block_stability_potentials(tokens, block)
    analytic = BlockPotentials.from_pairs(unary.double(), edges)
    reference = [int(tokens[p]) for p in block]
    analytic = analytic.with_reference(reference, scorer.H_of(tokens))
    oracle = probe_block_potentials(scorer.H_of, tokens, block, V)
    assert torch.allclose(analytic.joint(), oracle.joint(), atol=1e-9)


def test_reweighting_scales_self_and_pair_terms_separately(scorer, tokens):
    etab = scorer.etab
    self_energy = sum(float(etab[i, 0, tokens[i], tokens[i]]) for i in range(L))
    pair_energy = scorer.H_of(tokens) - self_energy
    weighted = scorer.reweighted(0.5, 2.0)
    assert weighted.H_of(tokens) == pytest.approx(
        0.5 * self_energy + 2.0 * pair_energy, abs=1e-9
    )
    assert scorer.H_of(tokens) == pytest.approx(
        self_energy + pair_energy
    )  # original intact


def make_ctx(scorer):
    return SimpleNamespace(scorer=scorer)


def test_selective_gap_is_the_hamiltonian_difference_between_states(
    engine, scorer, tokens
):
    """e_P - e_D at a centre equals H(centre=P) - mean H(centre=D); lower favours P."""
    center, prot, deps = 2, 2, [3]
    with_p, with_d = tokens.clone(), tokens.clone()
    with_p[center], with_d[center] = prot, deps[0]
    gap = engine._selective_energy_of(make_ctx(scorer), tokens, center, prot, deps)
    assert gap == pytest.approx(scorer.H_of(with_p) - scorer.H_of(with_d), abs=1e-9)


def test_selective_gap_averages_over_several_contrast_tokens(engine, scorer, tokens):
    center, prot, deps = 1, 2, [0, 3]
    energies = []
    for token in [prot, *deps]:
        probe = tokens.clone()
        probe[center] = token
        energies.append(scorer.H_of(probe))
    gap = engine._selective_energy_of(make_ctx(scorer), tokens, center, prot, deps)
    assert gap == pytest.approx(energies[0] - sum(energies[1:]) / 2, abs=1e-9)


def test_selective_sum_measures_each_centre_with_the_others_protonated(
    engine, scorer, tokens
):
    pins = [
        engine.PlacementPin(
            position=pos, protonation_type="HIS-P", prot_idx=2, dep_idxs=[3], res_id=pos
        )
        for pos in (1, 5)
    ]
    all_protonated = tokens.clone()
    all_protonated[[1, 5]] = 2
    expected = 0.0
    for pin in pins:
        flipped = all_protonated.clone()
        flipped[pin.position] = 3
        expected += scorer.H_of(all_protonated) - scorer.H_of(flipped)
    got = engine._selective_energy_sum(make_ctx(scorer), tokens, pins)
    assert got == pytest.approx(expected, abs=1e-9)


V6_NAMES = ["ALA", "HIS", "UNK", "HIS-P", "HIS-S", "HIS-A", "ASP-P", "ASP-D", "ASP-A"]


def test_default_dep_map_still_names_v3_v4_tautomers(engine):
    """Documents the stale default: HIS-P is contrasted with HID/HIE, absent from v6."""
    assert engine.DEFAULT_DEP_MAP["HIS-P"] == ["HID", "HIE"]
    ctx = SimpleNamespace(
        encoding=SimpleNamespace(token_to_idx={n: i for i, n in enumerate(V6_NAMES)})
    )
    crit = engine.PHDesignCriteria()
    with pytest.raises(KeyError, match="HID"):
        engine.PottsMPNNPHEngine._pin_idxs(None, ctx, crit, "HIS-P")
    crit.dep_map = {"HIS-P": ["HIS-S"]}
    prot, deps = engine.PottsMPNNPHEngine._pin_idxs(None, ctx, crit, "HIS-P")
    assert (prot, deps) == (V6_NAMES.index("HIS-P"), [V6_NAMES.index("HIS-S")])


def test_criteria_validation_and_legacy_default_method(engine):
    assert engine.PHDesignCriteria().method == "converged_mcmc"
    with pytest.raises(ValueError, match="Unknown method"):
        engine.PHDesignCriteria(method="simulated_annealing")
    with pytest.raises(ValueError, match="requires backend"):
        engine.PHDesignCriteria(method="autoregressive", backend="potts")
