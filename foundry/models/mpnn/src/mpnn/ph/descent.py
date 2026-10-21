"""Block descent on a scalarised :class:`~mpnn.ph.objective.Objective`.

Each step enumerates every joint assignment of a small block of designable positions
(``V**B``) with the rest of the sequence fixed, and moves to the best one. With
``temperature == 0`` the scalarised objective never increases, and a block that covers
all designable positions returns the exact optimum.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import torch

from mpnn.ph.objective import Objective

logger = logging.getLogger(__name__)

SWEEP_ORDERS = ("position", "reverse")


@dataclass
class DescentResult:
    """Outcome of one descent run."""

    tokens: torch.Tensor
    value: float  # scalarised objective of ``tokens``
    trace: List[float] = field(default_factory=list)  # value after every block step
    rounds: int = 0
    converged: bool = False
    n_changes: int = 0


def build_block(
    position: int,
    designable: Sequence[int],
    block_size: int,
    partner_rank: Optional[Callable[[int], Sequence[int]]],
) -> List[int]:
    """The block around ``position``: itself plus its closest designable partners.

    ``partner_rank(p)`` lists positions by closeness to ``p``; without it the partners
    are the next designable positions in order (wrapping around).
    """
    designable = list(designable)
    if partner_rank is not None:
        ranked = [q for q in partner_rank(position) if q in set(designable)]
    else:
        start = designable.index(position)
        ranked = designable[start + 1 :] + designable[:start]
    partners = [q for q in ranked if q != position][: block_size - 1]
    return sorted([position, *partners])


def block_descent(
    objective: Objective,
    tokens: torch.Tensor,
    designable: Sequence[int],
    names: Sequence[str],
    weights: Sequence[float],
    scalarisation: str = "weighted_sum",
    valid_mask: Optional[torch.Tensor] = None,
    block_size: int = 2,
    partner_rank: Optional[Callable[[int], Sequence[int]]] = None,
    temperature: float = 0.0,
    generator: Optional[torch.Generator] = None,
    max_rounds: int = 10,
    sweep: str = "position",
) -> DescentResult:
    """Descend from ``tokens`` over the ``designable`` positions.

    At ``temperature > 0`` blocks are sampled, so the last state is not necessarily the
    best; the best state seen is returned. Pass a seeded ``generator`` to reproduce
    a run.

    Only a sequence whose designable positions all hold allowed tokens (``valid_mask``)
    can be returned: a forbidden token in ``tokens`` (for example a native ``HIS-S``) is
    replaced during the first sweep even when it scores better than every allowed
    replacement, and partly corrected sequences never qualify.
    """
    if block_size < 1:
        raise ValueError("block_size must be >= 1.")
    if sweep not in SWEEP_ORDERS:
        raise ValueError(f"sweep must be one of {SWEEP_ORDERS}.")
    designable = sorted(int(p) for p in designable)
    order = designable if sweep == "position" else designable[::-1]
    current = tokens.clone()

    def value_of(seq: torch.Tensor) -> float:
        return objective.scalarised_value(seq, names, weights, scalarisation)

    def is_valid(seq: torch.Tensor) -> bool:
        if valid_mask is None or not designable:
            return True
        return bool(valid_mask.to(seq.device)[seq[designable]].all())

    value = value_of(current)
    best, best_value = None, float("inf")
    if is_valid(current):
        best, best_value = current.clone(), value
    result = DescentResult(tokens=current, value=value, trace=[value])
    for round_index in range(1, max_rounds + 1):
        changed = False
        for position in order:
            block = build_block(position, designable, block_size, partner_rank)
            choice = objective.best_assignment(
                current,
                block,
                names,
                weights,
                scalarisation,
                valid_mask=valid_mask,
                temperature=temperature,
                generator=generator,
            )
            if any(int(current[p]) != c for p, c in zip(block, choice)):
                for p, c in zip(block, choice):
                    current[p] = c
                changed = True
                result.n_changes += 1
            value = value_of(current)
            result.trace.append(value)
            if is_valid(current) and value < best_value:
                best, best_value = current.clone(), value
        result.rounds = round_index
        if not changed:
            result.converged = True
            break
    if (
        best is None
    ):  # cannot happen after one sweep: every designable position was re-chosen
        raise RuntimeError(
            "Descent ended without reaching a sequence of allowed tokens."
        )
    result.tokens, result.value = best, best_value
    return result
