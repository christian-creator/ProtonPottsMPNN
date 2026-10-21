"""Run a :class:`~mpnn.ph.objective.SearchPlan`: one scalarised run, or a Pareto sweep.

A plan with ``mode == "single"`` runs block descent once per seed with one weight
vector. A ``"pareto"`` plan runs it once per weight vector and seed, pools the designs
(deduplicated by sequence), sorts them into Pareto fronts on the *active* terms and
optionally thins the set by crowding distance. Seeds are explicit: run ``r``, seed ``s``
uses ``torch.Generator().manual_seed(base_seed + 1000 * r + s)``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from mpnn.ph.descent import block_descent, build_block
from mpnn.ph.objective import Objective, SearchPlan
from mpnn.ph.pareto import (
    fast_non_dominated_sort,
    hypervolume,
    select_by_crowding,
)

logger = logging.getLogger(__name__)


@dataclass
class DesignRecord:
    """One design with the raw value of every term and where it came from."""

    tokens: torch.Tensor
    term_values: Dict[str, float]
    weights: Dict[str, float]  # weight vector of the producing run (active terms)
    run_index: int
    seed_index: int
    scalarised: float  # value under the producing run's weights
    front: Optional[int] = None  # Pareto front index (pareto mode); 0 = non-dominated
    selected: bool = False
    # term values in each target state (ensemble runs); term_values holds the reduced ones
    per_state: Optional[Dict[str, Dict[str, float]]] = None


@dataclass
class SearchResult:
    plan: SearchPlan
    records: List[DesignRecord] = field(default_factory=list)
    hypervolume: Optional[float] = None
    reference_point: Optional[Dict[str, float]] = None

    def selected(self) -> List[DesignRecord]:
        return [r for r in self.records if r.selected]


def _scale_blocks(designable, block_size, partner_rank):
    seen, blocks = set(), []
    for position in sorted(designable):
        block = tuple(build_block(position, designable, block_size, partner_rank))
        if block not in seen:
            seen.add(block)
            blocks.append(list(block))
    return blocks


def run_search(
    objective: Objective,
    tokens: torch.Tensor,
    designable: Sequence[int],
    plan: SearchPlan,
    *,
    valid_mask: Optional[torch.Tensor] = None,
    block_size: int = 2,
    partner_rank: Optional[Callable[[int], Sequence[int]]] = None,
    temperature: float = 0.0,
    n_seeds: int = 1,
    base_seed: int = 0,
    max_rounds: int = 10,
    n_select: Optional[int] = None,
    initialiser: Optional[Callable[[int], torch.Tensor]] = None,
) -> SearchResult:
    """Execute ``plan`` and return the pooled, annotated designs.

    Scales are frozen once on the starting sequence before any run, so every run and
    every weight vector shares one normalisation. ``initialiser(seed)`` may supply a
    different starting sequence per seed.
    """
    names = list(plan.term_names)
    if any(name not in objective.terms for name in names):
        raise KeyError(
            f"The objective lacks terms {sorted(set(names) - set(objective.terms))}."
        )
    objective.freeze_scales(
        tokens,
        _scale_blocks(designable, block_size, partner_rank),
        names,
        valid_mask=valid_mask,
    )
    pooled: Dict[tuple, DesignRecord] = {}
    for run_index, vector in enumerate(plan.weight_vectors):
        weights = [float(w) for w in vector]
        for seed_index in range(n_seeds):
            generator = torch.Generator().manual_seed(
                base_seed + 1000 * run_index + seed_index
            )
            start = initialiser(seed_index) if initialiser else tokens
            result = block_descent(
                objective,
                start,
                designable,
                names,
                weights,
                plan.scalarisation,
                valid_mask=valid_mask,
                block_size=block_size,
                partner_rank=partner_rank,
                temperature=temperature,
                generator=generator,
                max_rounds=max_rounds,
            )
            key = tuple(int(t) for t in result.tokens)
            if key in pooled:
                continue
            pooled[key] = DesignRecord(
                tokens=result.tokens,
                term_values=objective.term_values(result.tokens),
                weights=dict(zip(names, weights)),
                run_index=run_index,
                seed_index=seed_index,
                scalarised=result.value,
                per_state=(
                    objective.per_state_values(result.tokens)
                    if hasattr(objective, "per_state_values")
                    else None
                ),
            )
    records = list(pooled.values())
    out = SearchResult(plan=plan, records=records)
    if plan.mode == "pareto" and records:
        _annotate_pareto(out, names, n_select)
    else:
        records.sort(key=lambda r: r.scalarised)
        for record in records[: n_select if n_select else len(records)]:
            record.selected = True
    logger.info(
        "%s search over %s: %d designs from %d weight vectors",
        plan.mode,
        names,
        len(records),
        len(plan.weight_vectors),
    )
    return out


def _annotate_pareto(result: SearchResult, names: List[str], n_select) -> None:
    records = result.records
    matrix = np.array([[r.term_values[n] for n in names] for r in records])
    for front_index, front in enumerate(fast_non_dominated_sort(matrix)):
        for i in front:
            records[i].front = front_index
    span = matrix.max(axis=0) - matrix.min(axis=0)
    reference = matrix.max(axis=0) + 0.1 * np.where(span > 0, span, 1.0)
    first_front = matrix[[i for i, r in enumerate(records) if r.front == 0]]
    result.hypervolume = hypervolume(first_front, reference)
    result.reference_point = dict(zip(names, reference.tolist()))
    chosen = (
        select_by_crowding(matrix, n_select) if n_select else np.arange(len(records))
    )
    for i in chosen:
        records[i].selected = True
    order = sorted(range(len(records)), key=lambda i: (records[i].front, i))
    result.records = [records[i] for i in order]
