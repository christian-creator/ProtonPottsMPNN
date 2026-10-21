"""Exact energies of a block of positions, for V**B joint enumeration.

For a pairwise (Potts) Hamiltonian, the part that depends on ``B`` chosen positions,
with the rest of the sequence fixed, is a sum of per-position tables and per-pair tables.
:class:`BlockPotentials` stores that decomposition so every joint assignment of the
block can be scored by lookup. Block descent evaluates all ``V**B`` assignments this way.
"""

from __future__ import annotations

import itertools
import logging
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

DEFAULT_MAX_JOINT_ELEMENTS = 50_000_000

EdgeKey = Tuple[int, int]


class BlockPotentials:
    """``E(a) = const + sum_i unary[i, a_i] + sum_(i<j) edges[(i, j)][a_i, a_j]``.

    ``unary`` is ``[B, V]``; each edge table is ``[V, V]`` indexed (position i, position
    j) with ``i < j``. Lower energy is better throughout.

    Every tensor lives on the device of ``unary``: edge tables are moved there on
    construction, and new tensors are allocated there.
    """

    def __init__(
        self,
        unary: torch.Tensor,
        edges: Optional[Mapping[EdgeKey, torch.Tensor]] = None,
        const: float = 0.0,
    ) -> None:
        if unary.ndim != 2:
            raise ValueError("unary must have shape [B, V].")
        self.unary = unary
        self.edges: Dict[EdgeKey, torch.Tensor] = {}
        for (i, j), table in (edges or {}).items():
            if not 0 <= i < j < unary.shape[0]:
                raise ValueError(f"Edge key {(i, j)} must satisfy 0 <= i < j < B.")
            if table.shape != (unary.shape[1], unary.shape[1]):
                raise ValueError("Edge tables must be [V, V].")
            self.edges[(i, j)] = table.to(unary.device)
        self.const = float(const)

    @property
    def n_block(self) -> int:
        return int(self.unary.shape[0])

    @property
    def vocab_size(self) -> int:
        return int(self.unary.shape[1])

    @classmethod
    def zeros(cls, n_block: int, vocab_size: int, dtype=torch.float32, device=None):
        return cls(torch.zeros(n_block, vocab_size, dtype=dtype, device=device))

    @classmethod
    def from_pairs(
        cls,
        unary: torch.Tensor,
        edges: Sequence[Tuple[int, int, torch.Tensor]],
        const: float = 0.0,
    ) -> "BlockPotentials":
        """Build from a ``(unary, [(i, j, M), ...])`` decomposition in either orientation.

        An edge given as ``i > j`` is transposed so that stored keys always satisfy
        ``i < j``; repeated pairs are summed.
        """
        merged: Dict[EdgeKey, torch.Tensor] = {}
        for i, j, table in edges:
            if i == j:
                raise ValueError("An edge cannot join a position to itself.")
            key, table = ((i, j), table) if i < j else ((j, i), table.T)
            merged[key] = merged[key] + table if key in merged else table
        return cls(unary, merged, const)

    # ---- algebra ------------------------------------------------------------
    def _check_compatible(self, other: "BlockPotentials") -> None:
        if (self.n_block, self.vocab_size) != (other.n_block, other.vocab_size):
            raise ValueError(
                "Block potentials differ in block size or vocabulary size."
            )

    def __add__(self, other: "BlockPotentials") -> "BlockPotentials":
        self._check_compatible(other)
        device = self.unary.device
        edges = dict(self.edges)
        for key, table in other.edges.items():
            table = table.to(device)
            edges[key] = edges[key] + table if key in edges else table
        return BlockPotentials(
            self.unary + other.unary.to(device), edges, self.const + other.const
        )

    def scaled(self, factor: float) -> "BlockPotentials":
        return BlockPotentials(
            self.unary * factor,
            {key: table * factor for key, table in self.edges.items()},
            self.const * factor,
        )

    def __sub__(self, other: "BlockPotentials") -> "BlockPotentials":
        return self + other.scaled(-1.0)

    # ---- evaluation ---------------------------------------------------------
    def energy(self, assignment: Sequence[int]) -> float:
        """Energy of one joint assignment (one token index per block position)."""
        if len(assignment) != self.n_block:
            raise ValueError("Assignment length must equal the block size.")
        total = self.const
        for i, a in enumerate(assignment):
            total += float(self.unary[i, a])
        for (i, j), table in self.edges.items():
            total += float(table[assignment[i], assignment[j]])
        return total

    def joint(self, max_elements: int = DEFAULT_MAX_JOINT_ELEMENTS) -> torch.Tensor:
        """All ``V**B`` energies as a tensor of shape ``[V] * B``."""
        n, v = self.n_block, self.vocab_size
        if v**n > max_elements:
            raise ValueError(
                f"Joint enumeration of {v}**{n} = {v**n:,} assignments exceeds the cap "
                f"of {max_elements:,}; use a smaller block."
            )
        out = torch.full(
            [v] * n, self.const, dtype=self.unary.dtype, device=self.unary.device
        )
        for i in range(n):
            shape = [1] * n
            shape[i] = v
            out = out + self.unary[i].reshape(shape)
        for (i, j), table in self.edges.items():
            shape = [1] * n
            shape[i] = v
            shape[j] = v
            out = out + table.reshape(shape)
        return out

    # ---- re-indexing --------------------------------------------------------
    def embedded(self, n_block: int, slots: Sequence[int]) -> "BlockPotentials":
        """Place this block's positions into slots ``slots`` of a larger block.

        Slots not listed carry zero energy. ``slots`` must be strictly increasing so the
        ``i < j`` orientation of edge tables is preserved.
        """
        slots = [int(s) for s in slots]
        if len(slots) != self.n_block or slots != sorted(set(slots)):
            raise ValueError(
                "slots must be strictly increasing, one per block position."
            )
        if slots and not (0 <= slots[0] and slots[-1] < n_block):
            raise ValueError("slots out of range.")
        unary = torch.zeros(
            n_block, self.vocab_size, dtype=self.unary.dtype, device=self.unary.device
        )
        for k, slot in enumerate(slots):
            unary[slot] = self.unary[k]
        edges = {(slots[i], slots[j]): t for (i, j), t in self.edges.items()}
        return BlockPotentials(unary, edges, self.const)

    def with_reference(
        self, reference: Sequence[int], total_energy: float
    ) -> "BlockPotentials":
        """Set ``const`` so that ``energy(reference) == total_energy``.

        Scorers that drop terms among fixed residues give potentials valid only up to a
        constant. Anchoring them to the full-system energy at the current assignment
        makes absolute values (needed by hinge terms) correct.
        """
        without_const = BlockPotentials(self.unary, self.edges, 0.0)
        return BlockPotentials(
            self.unary,
            self.edges,
            float(total_energy) - without_const.energy(reference),
        )


def probe_block_potentials(
    energy_fn: Callable[[torch.Tensor], float],
    tokens: torch.Tensor,
    block: Sequence[int],
    vocab_size: int,
) -> BlockPotentials:
    """Decompose any *pairwise* energy by probing it; exact for Potts Hamiltonians.

    Evaluates ``energy_fn`` on single and double substitutions around the current
    assignment, so it needs ``O(B*V + B**2 * V**2)`` evaluations. The result carries the
    absolute energy (``const`` is the energy of ``tokens`` itself). Useful as an
    independent oracle for scorers that decompose analytically.
    """
    block = [int(p) for p in block]
    base = float(energy_fn(tokens))

    def with_tokens(substitutions: Dict[int, int]) -> float:
        probe = tokens.clone()
        for slot, token in substitutions.items():
            probe[block[slot]] = token
        return float(energy_fn(probe))

    n = len(block)
    unary = torch.zeros(n, vocab_size, dtype=torch.float64)
    for i in range(n):
        for a in range(vocab_size):
            unary[i, a] = with_tokens({i: a}) - base
    edges: Dict[EdgeKey, torch.Tensor] = {}
    for i, j in itertools.combinations(range(n), 2):
        table = torch.zeros(vocab_size, vocab_size, dtype=torch.float64)
        for a in range(vocab_size):
            for b in range(vocab_size):
                table[a, b] = (
                    with_tokens({i: a, j: b}) - base - unary[i, a] - unary[j, b]
                )
        edges[(i, j)] = table
    return BlockPotentials(unary, edges, base)
