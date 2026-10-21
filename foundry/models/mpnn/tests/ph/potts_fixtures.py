"""Synthetic Potts systems for atomworks-free tests."""

from typing import Callable, Dict, List, Sequence, Tuple

import torch

from mpnn.ph.binding import SystemView
from mpnn.ph.potentials import BlockPotentials, probe_block_potentials

V = 4
BINDER = [0, 1, 2]
RECEPTOR = [3, 4, 5]


class SyntheticPotts:
    """A random directed pairwise energy ``sum_i h_i[S_i] + sum_(i->j) J_ij[S_i, S_j]``.

    Its block potentials come from probing, so they do not depend on any analytic
    decomposition. Positions are 0..n-1 of the *system*, not of a larger complex.
    """

    def __init__(
        self,
        n_positions: int,
        vocab_size: int,
        edges: Sequence[Tuple[int, int]],
        seed: int = 0,
        scale: float = 1.0,
    ) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.n_positions = n_positions
        self.vocab_size = vocab_size
        self.fields = scale * torch.randn(
            n_positions, vocab_size, generator=gen, dtype=torch.float64
        )
        self.couplings: Dict[Tuple[int, int], torch.Tensor] = {
            (i, j): scale
            * torch.randn(vocab_size, vocab_size, generator=gen, dtype=torch.float64)
            for i, j in edges
        }

    def H_of(self, tokens: torch.Tensor) -> float:
        total = sum(
            float(self.fields[i, int(tokens[i])]) for i in range(self.n_positions)
        )
        for (i, j), table in self.couplings.items():
            total += float(table[int(tokens[i]), int(tokens[j])])
        return total

    def block_potentials(
        self, tokens: torch.Tensor, block: Sequence[int]
    ) -> BlockPotentials:
        return probe_block_potentials(self.H_of, tokens, block, self.vocab_size)


def all_pairs(positions: Sequence[int]) -> List[Tuple[int, int]]:
    """Directed edges i->j for every ordered pair of distinct positions."""
    return [(i, j) for i in positions for j in positions if i != j]


class IdealComplex:
    """H_complex = H_binder + H_receptor + H_interface, exactly separable by design."""

    def __init__(
        self,
        binder_seed,
        receptor_seed,
        interface_seed,
        internal_scale=1.0,
        interface_scale=1.0,
    ):
        self.binder = SyntheticPotts(
            3, V, all_pairs(range(3)), binder_seed, internal_scale
        )
        self.receptor = SyntheticPotts(
            3, V, all_pairs(range(3)), receptor_seed, internal_scale
        )
        gen = torch.Generator().manual_seed(interface_seed)
        self.interface = {
            (i, j): interface_scale
            * torch.randn(V, V, generator=gen, dtype=torch.float64)
            for i in BINDER
            for j in RECEPTOR
        }

    def H_interface(self, tokens):
        return sum(
            float(t[int(tokens[i]), int(tokens[j])])
            for (i, j), t in self.interface.items()
        )

    def H_of(self, tokens):
        return (
            self.binder.H_of(tokens[BINDER])
            + self.receptor.H_of(tokens[RECEPTOR])
            + self.H_interface(tokens)
        )

    def block_potentials(self, tokens, block):
        return probe_block_potentials(self.H_of, tokens, block, V)

    def views(self):
        return (
            SystemView(self, range(6), V),
            SystemView(self.binder, BINDER, V),
            SystemView(self.receptor, RECEPTOR, V),
        )


class FunctionSystem:
    """A system defined by an arbitrary pairwise energy function of its own tokens."""

    def __init__(self, energy_fn: Callable[[torch.Tensor], float], vocab_size: int = V):
        self.energy_fn = energy_fn
        self.vocab_size = vocab_size

    def H_of(self, tokens: torch.Tensor) -> float:
        return float(self.energy_fn(tokens))

    def block_potentials(
        self, tokens: torch.Tensor, block: Sequence[int]
    ) -> BlockPotentials:
        return probe_block_potentials(self.H_of, tokens, block, self.vocab_size)


def random_graph_potts(
    n_positions: int = 8, n_neighbours: int = 4, vocab_size: int = V, seed: int = 0
):
    """Random ``etab [1, L, K, V, V]`` and an asymmetric kNN ``E_idx [1, L, K]``.

    Slot 0 of ``E_idx`` is the position itself, as in the model's own output.
    """
    gen = torch.Generator().manual_seed(seed)
    etab = torch.randn(
        1,
        n_positions,
        n_neighbours,
        vocab_size,
        vocab_size,
        generator=gen,
        dtype=torch.float64,
    )
    e_idx = torch.empty(1, n_positions, n_neighbours, dtype=torch.long)
    for i in range(n_positions):
        others = [j for j in range(n_positions) if j != i]
        chosen = torch.randperm(len(others), generator=gen)[: n_neighbours - 1]
        e_idx[0, i] = torch.tensor([i] + [others[int(c)] for c in chosen])
    return etab, e_idx
