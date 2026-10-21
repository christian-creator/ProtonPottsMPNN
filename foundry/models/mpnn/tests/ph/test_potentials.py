"""Tests for block potentials and the probing oracle (atomworks-free)."""

import itertools

import pytest
import torch
from potts_fixtures import SyntheticPotts, all_pairs

from mpnn.ph.potentials import BlockPotentials, probe_block_potentials

V = 4
N = 6


@pytest.fixture
def system():
    # sparse, asymmetric graph: not every ordered pair is an edge
    edges = [(i, j) for i, j in all_pairs(range(N)) if (i * 3 + j) % 4 != 0]
    return SyntheticPotts(N, V, edges, seed=7)


@pytest.fixture
def tokens():
    return torch.tensor([0, 1, 2, 3, 1, 2])


@pytest.mark.parametrize("block", [[2], [1, 4], [0, 3, 5]])
def test_joint_matches_brute_force_energy(system, tokens, block):
    pot = system.block_potentials(tokens, block)
    joint = pot.joint()
    assert joint.shape == (V,) * len(block)
    for assignment in itertools.product(range(V), repeat=len(block)):
        probe = tokens.clone()
        for slot, token in zip(block, assignment):
            probe[slot] = token
        assert joint[assignment].item() == pytest.approx(system.H_of(probe), abs=1e-9)
        assert pot.energy(assignment) == pytest.approx(system.H_of(probe), abs=1e-9)


def test_probed_const_is_the_energy_of_the_reference(system, tokens):
    pot = system.block_potentials(tokens, [1, 4])
    assert pot.const == pytest.approx(system.H_of(tokens))
    assert pot.energy([int(tokens[1]), int(tokens[4])]) == pytest.approx(pot.const)


def test_algebra_acts_on_the_joint_tensor(system, tokens):
    other = SyntheticPotts(N, V, all_pairs(range(N)), seed=11)
    p = system.block_potentials(tokens, [0, 2])
    q = other.block_potentials(tokens, [0, 2])
    assert torch.allclose((p + q).joint(), p.joint() + q.joint())
    assert torch.allclose((p - q).joint(), p.joint() - q.joint())
    assert torch.allclose(p.scaled(-2.5).joint(), -2.5 * p.joint())


def test_incompatible_blocks_cannot_be_combined(system, tokens):
    with pytest.raises(ValueError):
        system.block_potentials(tokens, [0]) + system.block_potentials(tokens, [0, 1])


def test_from_pairs_normalises_orientation():
    unary = torch.zeros(2, V, dtype=torch.float64)
    table = torch.arange(V * V, dtype=torch.float64).reshape(V, V)
    flipped = BlockPotentials.from_pairs(unary, [(1, 0, table)])
    direct = BlockPotentials.from_pairs(unary, [(0, 1, table.T)])
    assert torch.equal(flipped.joint(), direct.joint())
    assert flipped.energy([2, 3]) == table[3, 2].item()
    with pytest.raises(ValueError):
        BlockPotentials.from_pairs(unary, [(1, 1, table)])


def test_embedding_adds_empty_slots_without_changing_energies(system, tokens):
    pot = system.block_potentials(tokens, [1, 4])
    wide = pot.embedded(4, [1, 3])
    for assignment in itertools.product(range(V), repeat=4):
        expected = pot.energy([assignment[1], assignment[3]])
        assert wide.energy(assignment) == pytest.approx(expected)
    with pytest.raises(ValueError):
        pot.embedded(4, [3, 1])


def test_with_reference_restores_absolute_energies(system, tokens):
    pot = system.block_potentials(tokens, [1, 4])
    shifted = BlockPotentials(pot.unary, pot.edges, const=123.0)
    fixed = shifted.with_reference(
        [int(tokens[1]), int(tokens[4])], system.H_of(tokens)
    )
    assert torch.allclose(fixed.joint(), pot.joint())


def test_joint_enumeration_is_capped(system, tokens):
    pot = system.block_potentials(tokens, [0, 1, 2])
    with pytest.raises(ValueError, match="exceeds the cap"):
        pot.joint(max_elements=V**3 - 1)


def test_empty_block_is_a_scalar(system, tokens):
    pot = probe_block_potentials(system.H_of, tokens, [], V)
    assert pot.joint().shape == ()
    assert pot.joint().item() == pytest.approx(system.H_of(tokens))


def meta_block(n_block=2, vocab=V):
    """Potentials on the 'meta' device: a CPU-only stand-in for a second device (CUDA)."""
    unary = torch.zeros(n_block, vocab, dtype=torch.float64, device="meta")
    table = torch.zeros(vocab, vocab, dtype=torch.float64, device="meta")
    return BlockPotentials(unary, {(0, 1): table}, const=0.5)


def test_every_tensor_stays_on_the_device_of_the_potentials():
    wide = meta_block().embedded(4, [1, 3])
    assert wide.unary.device.type == "meta"
    assert all(t.device.type == "meta" for t in wide.edges.values())
    assert wide.joint().device.type == "meta"
    assert BlockPotentials.zeros(2, V, device="meta").unary.device.type == "meta"


def test_adding_potentials_keeps_the_device():
    total = meta_block() + meta_block()
    assert total.unary.device.type == "meta" and total.joint().device.type == "meta"
