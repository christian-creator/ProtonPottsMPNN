"""Adapters between the engine's scorer objects and the ``mpnn.ph`` protocols."""

from __future__ import annotations

import logging
from typing import Callable, List, Mapping, Sequence

import numpy as np
import torch

from mpnn.ph.potentials import BlockPotentials
from mpnn.ph.states import SiteKey

logger = logging.getLogger(__name__)


class PottsScorerAdapter:
    """Present the engine's ``_PottsScorer`` as an :class:`~mpnn.ph.binding.EnergyScorer`.

    The engine scorer returns ``(unary, [(i, j, M), ...])`` from
    ``block_stability_potentials``; this wraps that into :class:`BlockPotentials`
    (summing repeated pairs, so edges i->j and j->i inside a block are merged).
    """

    def __init__(self, scorer) -> None:
        self.scorer = scorer
        self.vocab_size = int(scorer.V)

    def H_of(self, tokens: torch.Tensor) -> float:
        return float(self.scorer.H_of(tokens))

    def block_potentials(
        self, tokens: torch.Tensor, block: Sequence[int]
    ) -> BlockPotentials:
        unary, edges = self.scorer.block_stability_potentials(tokens, block)
        return BlockPotentials.from_pairs(unary, edges)


def positions_in_complex(
    chain_ids: Sequence[str],
    res_ids: Sequence[int],
    site_index: Mapping[SiteKey, int],
) -> List[int]:
    """Complex positions of an isolated chain's residues, in the chain's own order.

    Re-encoding a chain alone yields its own residue order; mapping each residue back by
    ``(chain, res_id)`` lets complex-wide token tensors be sliced for that scorer.
    """
    positions, missing = [], []
    for chain, res_id in zip(chain_ids, res_ids):
        key = SiteKey(str(chain), int(res_id))
        if key in site_index:
            positions.append(site_index[key])
        else:
            missing.append(str(key))
    if missing:
        raise KeyError(
            f"{len(missing)} residues of the isolated chain are not in the complex: "
            f"{missing[:5]}"
        )
    if len(set(positions)) != len(positions):
        raise ValueError("Two isolated residues map to the same complex position.")
    return positions


def parent_names(
    token_names: Sequence[str],
    parent_index: Sequence[int],
    tokens: Sequence[int],
) -> List[str]:
    """Bare residue name at each position, from the vocabulary's token-to-parent map.

    ``parent_index[t]`` is the index of token ``t``'s bare parent (the engine's canonical
    map, or ``TokenTable.parent_index()``); a protonation-state token maps to its residue.
    """
    return [token_names[int(parent_index[int(t)])] for t in tokens]


def knn_partner_rank(neighbour_index: np.ndarray) -> Callable[[int], List[int]]:
    """Partner ranking from the model's kNN table ``E_idx[L, K]`` (slot 0 is the residue).

    Closest first, as the model orders its neighbours; the residue itself is skipped.
    """

    def rank(position: int) -> List[int]:
        return [int(j) for j in neighbour_index[position] if int(j) != position]

    return rank
