"""Redesign neighbourhoods from the model's contact graph.

The Potts model orders each residue's neighbours closest first (``E_idx``, slot 0 is the
residue itself). A residue is *coupled* to a centre when one lists the other, in either
direction. This is the neighbourhood the original engine redesigns around a pinned
centre (``neighbour_k`` / ``max_mutations``); here a centre may be any residue,
including one on the target.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

UNRANKED = 10**6  # rank of a neighbour the centre does not list itself


def cap_by_rank(
    positions: Sequence[int], ranks: Mapping[int, int], max_mutations: int
) -> List[int]:
    """Keep the ``max_mutations`` best-ranked positions (0 = keep all), ties by position."""
    positions = sorted(int(p) for p in positions)
    if max_mutations > 0 and len(positions) > max_mutations:
        order = sorted(positions, key=lambda p: (ranks.get(p, UNRANKED), p))
        positions = sorted(order[:max_mutations])
    return positions


def near_ranks(
    neighbour_index: np.ndarray,
    centres: Sequence[int],
    pool: Sequence[int],
    k: int = 0,
) -> Dict[int, int]:
    """Positions of ``pool`` coupled to any of ``centres``, each with its closeness rank.

    The rank is the best (smallest) place the position holds in a centre's own neighbour
    list, ``UNRANKED`` when only the position lists the centre. Centres are not returned.
    """
    table = np.asarray(neighbour_index)
    n_slots = table.shape[1]
    reach = n_slots if k <= 0 else min(n_slots, k + 1)
    centres = [int(c) for c in centres]
    allowed = {int(p) for p in pool} - set(centres)
    found = set()
    for centre in centres:
        found.update(int(j) for j in table[centre, 1:reach])
        for i in np.nonzero((table[:, 1:reach] == centre).any(axis=1))[0]:
            found.add(int(i))
    ranks = {p: UNRANKED for p in found & allowed}
    for centre in centres:
        for slot, j in enumerate(table[centre, 1:]):
            if int(j) in ranks:
                ranks[int(j)] = min(ranks[int(j)], slot)
    return ranks


def near_positions(
    neighbour_index: np.ndarray,
    centres: Sequence[int],
    pool: Sequence[int],
    k: int = 0,
    max_mutations: int = 0,
) -> List[int]:
    """Positions of ``pool`` coupled to any of ``centres``, sorted.

    ``k`` > 0 keeps only each centre's ``k`` nearest listed neighbours (and the residues
    that list the centre among *their* ``k`` nearest); 0 keeps the full list.
    ``max_mutations`` > 0 caps the result, keeping the positions closest to a centre:
    those the centre itself lists come first, in list order, then the rest by position.
    Centres are never returned.
    """
    ranks = near_ranks(neighbour_index, centres, pool, k)
    return cap_by_rank(list(ranks), ranks, max_mutations)
