"""Tests for the binding-energy models (atomworks-free)."""

import itertools

import pytest
import torch
from potts_fixtures import IdealComplex, SyntheticPotts, V, all_pairs

from mpnn.ph.binding import (
    BindingModel,
    ComplexGap,
    StateBinding,
    SystemView,
    make_binding_model,
)


@pytest.fixture
def ideal():
    return IdealComplex(1, 2, 3)


TOKENS = torch.tensor([0, 1, 2, 3, 1, 0])


def test_state_binding_keeps_only_the_interface(ideal):
    complex_view, binder_view, receptor_view = ideal.views()
    model = StateBinding(complex_view, binder_view, receptor_view)
    assert model.energy(TOKENS) == pytest.approx(ideal.H_interface(TOKENS))
    assert ComplexGap(complex_view).energy(TOKENS) == pytest.approx(ideal.H_of(TOKENS))
    assert ideal.H_of(TOKENS) != pytest.approx(ideal.H_interface(TOKENS))


def test_state_binding_ignores_intramolecular_energy_scale():
    small = IdealComplex(1, 2, 3, internal_scale=0.1)
    large = IdealComplex(1, 2, 3, internal_scale=50.0)
    energies = []
    gaps = []
    for system in (small, large):
        views = system.views()
        energies.append(StateBinding(*views).energy(TOKENS))
        gaps.append(ComplexGap(views[0]).energy(TOKENS))
    assert energies[0] == pytest.approx(energies[1])
    assert gaps[0] != pytest.approx(gaps[1])


@pytest.mark.parametrize("block", [[0, 2], [1, 4], [3, 5], [0, 1, 2]])
def test_state_binding_block_potentials_match_brute_force(ideal, block):
    model = StateBinding(*ideal.views())
    joint = model.block_potentials(TOKENS, block).joint()
    for assignment in itertools.product(range(V), repeat=len(block)):
        probe = TOKENS.clone()
        for position, token in zip(block, assignment):
            probe[position] = token
        assert joint[assignment].item() == pytest.approx(
            ideal.H_interface(probe), abs=1e-9
        )


@pytest.mark.parametrize("block", [[0, 2], [1, 4]])
def test_complex_gap_block_potentials_match_brute_force(ideal, block):
    model = ComplexGap(ideal.views()[0])
    joint = model.block_potentials(TOKENS, block).joint()
    for assignment in itertools.product(range(V), repeat=len(block)):
        probe = TOKENS.clone()
        for position, token in zip(block, assignment):
            probe[position] = token
        assert joint[assignment].item() == pytest.approx(ideal.H_of(probe), abs=1e-9)


def test_receptor_residue_flip_changes_gap_but_not_binding_through_its_own_energy():
    """Flipping a receptor residue's state: its intramolecular change cancels in (b)."""
    flipped = TOKENS.clone()
    flipped[4] = 2
    shifts_state_binding, shifts_complex_gap = [], []
    for receptor_seed in (2, 99):  # different receptor internals, same interface
        system = IdealComplex(1, receptor_seed, 3)
        complex_view, binder_view, receptor_view = system.views()
        model = StateBinding(complex_view, binder_view, receptor_view)
        shifts_state_binding.append(model.energy(flipped) - model.energy(TOKENS))
        gap = ComplexGap(complex_view)
        shifts_complex_gap.append(gap.energy(flipped) - gap.energy(TOKENS))
    assert shifts_state_binding[0] == pytest.approx(shifts_state_binding[1])
    assert shifts_complex_gap[0] != pytest.approx(shifts_complex_gap[1])


def test_system_view_handles_permuted_positions():
    system = SyntheticPotts(3, V, all_pairs(range(3)), seed=5)
    permuted = [2, 0, 1]  # local position k is complex position permuted[k]
    view = SystemView(system, permuted, V)
    tokens = torch.tensor([3, 1, 0, 2, 2, 2])
    local_tokens = tokens[permuted]
    assert view.energy(tokens) == pytest.approx(system.H_of(local_tokens))
    joint = view.block_potentials(tokens, [0, 2]).joint()  # complex positions 0 and 2
    for a, b in itertools.product(range(V), repeat=2):
        probe = tokens.clone()
        probe[0], probe[2] = a, b
        assert joint[a, b].item() == pytest.approx(
            system.H_of(probe[permuted]), abs=1e-9
        )


def test_system_view_outside_block_is_constant():
    system = SyntheticPotts(3, V, all_pairs(range(3)), seed=5)
    view = SystemView(system, [0, 1, 2], V)
    tokens = torch.tensor([0, 1, 2, 3, 3, 3])
    joint = view.block_potentials(tokens, [4, 5]).joint()
    expected = torch.full((V, V), system.H_of(tokens[:3]), dtype=torch.float64)
    assert torch.allclose(joint, expected)


def test_state_binding_validates_its_systems(ideal):
    complex_view, binder_view, _ = ideal.views()
    with pytest.raises(ValueError, match="share"):
        StateBinding(complex_view, binder_view, binder_view)
    outside = SystemView(ideal.receptor, [3, 4, 9], V)
    with pytest.raises(ValueError, match="inside"):
        StateBinding(complex_view, binder_view, outside)


def test_factory_names_and_errors(ideal):
    complex_view, binder_view, receptor_view = ideal.views()
    assert (
        make_binding_model("complex_gap", complex_view=complex_view).name
        == "complex_gap"
    )
    model = make_binding_model(
        "state_binding",
        complex_view=complex_view,
        binder_view=binder_view,
        receptor_view=receptor_view,
    )
    assert model.name == "state_binding"
    with pytest.raises(ValueError, match="Unknown binding model"):
        make_binding_model("nonsense", complex_view=complex_view)


def test_block_values_default_to_the_potentials_joint(ideal):
    model = StateBinding(*ideal.views())
    values = model.block_values(TOKENS, [0, 2], ph=7.4)  # ph is ignored by this model
    assert torch.equal(values, model.block_potentials(TOKENS, [0, 2]).joint())


def test_model_without_decomposition_says_so():
    class Opaque(BindingModel):
        name = "opaque"

        def energy(self, tokens, ph=None):
            return 0.0

    with pytest.raises(NotImplementedError, match="block_values"):
        Opaque().block_potentials(TOKENS, [0])
