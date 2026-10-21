"""Per-site report: how binding changes a titratable residue's preference between states.

For every residue a condition names, in each condition's context, the report compares the
protonated-minus-deprotonated energy of the residue in the complex with the same gap in the
free partners. A negative ``binding_gap`` means binding favours the protonated state, which
raises the residue's apparent pKa on binding; ``delta_pka_model`` converts the gap with the
model's ``beta`` (model units, not calibrated). With the ``linked_equilibrium`` model the report
also gives the protonated fraction and apparent pKa of the free and the bound partners at the
condition's pH, and the shift between them.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from mpnn.ph.binding import LinkedEquilibrium
from mpnn.ph.vocab_meta import Protonation, TokenTable

logger = logging.getLogger(__name__)

LOG10 = math.log(10.0)


REPORT_STATES = ("protonated", "deprotonated")


def validate_report_token_shapes(report_tokens: Mapping[str, Any]) -> None:
    """Check the shape of ``{parent: {state: token | [tokens]}}`` (no vocabulary needed)."""
    for parent, states in report_tokens.items():
        if not isinstance(states, Mapping):
            raise ValueError(
                f"report_tokens[{parent!r}] must be a mapping of state to token(s)."
            )
        if set(states) - set(REPORT_STATES):
            raise ValueError(
                f"report_tokens[{parent!r}]: states must be 'protonated' and/or 'deprotonated', "
                f"got {sorted(states)}."
            )
        for state, names in states.items():
            listed = [names] if isinstance(names, str) else names
            if not isinstance(listed, (list, tuple)) or not listed:
                raise ValueError(
                    f"report_tokens[{parent!r}][{state!r}] needs at least one token."
                )
            if not all(isinstance(n, str) for n in listed):
                raise ValueError(
                    f"report_tokens[{parent!r}][{state!r}] must name tokens as strings."
                )


def validate_report_tokens(table: TokenTable, report_tokens: Mapping[str, Any]) -> None:
    """Check every named token exists and belongs to its residue."""
    validate_report_token_shapes(report_tokens)
    for parent, states in report_tokens.items():
        for names in states.values():
            for name in [names] if isinstance(names, str) else names:
                table.resolve(
                    parent, name
                )  # raises for a wrong residue or unknown token


def compared_tokens(
    table: TokenTable,
    parent: str,
    state: str,
    report_tokens: Optional[Mapping[str, Any]] = None,
) -> List[int]:
    """Token indices that stand for ``state`` of ``parent`` in a report.

    A policy in ``report_tokens`` wins; otherwise the state must name exactly one token.
    Several tokens (the neutral tautomers HID/HIE in v3/v4) are ambiguous and need a policy.
    """
    chosen = (report_tokens or {}).get(parent, {}).get(state)
    if chosen is not None:
        names = [chosen] if isinstance(chosen, str) else list(chosen)
        return [table.index(table.resolve(parent, name)) for name in names]
    candidates = table.tokens_for(parent, Protonation(state))
    if not candidates:
        raise ValueError(f"The vocabulary has no {state} token for {parent}.")
    if len(candidates) > 1:
        raise ValueError(
            f"The {state} state of {parent} is ambiguous in this vocabulary "
            f"{list(candidates)}; say how the report should compare it with report_tokens, "
            f"for example {{'{parent}': {{'{state}': {list(candidates)}}}}} "
            "(a list is averaged, a single name picks one token)."
        )
    return [table.index(candidates[0])]


def mean_energy(
    view, base: torch.Tensor, position: int, token_ids: Sequence[int]
) -> float:
    """Energy of ``view`` with ``position`` set to each token, averaged."""
    total = 0.0
    for token in token_ids:
        probe = base.clone()
        probe[position] = token
        total += view.energy(probe)
    return total / len(token_ids)


def apparent_pka(ph: float, protonated_fraction: Optional[float]) -> Optional[float]:
    """pKa implied by a protonated fraction at ``ph`` (None at 0 or 1, where it is undefined)."""
    if protonated_fraction is None or not 0.0 < protonated_fraction < 1.0:
        return None
    return ph + math.log10(protonated_fraction / (1.0 - protonated_fraction))


def site_report(
    ctx,
    conditions: Mapping[str, Any],
    inputs,
    tokens: torch.Tensor,
    beta: Optional[float] = None,
    report_tokens: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """One row per (condition, titratable site) for the design ``tokens``.

    ``ctx`` is the run's :class:`~mpnn.ph.objective.TermContext`, ``conditions`` its resolved
    conditions and ``inputs`` the :class:`~mpnn.ph.session.DesignInputs`. ``beta`` converts
    gaps to a model pKa shift; it defaults to the binding model's own ``beta`` when it has one.
    ``report_tokens`` says which tokens stand for a state a vocabulary names ambiguously
    (see :func:`compared_tokens`); each row lists the tokens it compared.
    """
    if ctx.receptor_view is None:
        raise ValueError("The term context carries no receptor system.")
    table = inputs.table
    if beta is None:
        beta = getattr(ctx.binding, "beta", None)
    positions = sorted({p for c in conditions.values() for p in c.token_by_position})
    rows: List[Dict[str, Any]] = []
    for name, condition in conditions.items():
        state = condition.apply(tokens)
        populations = None
        if isinstance(ctx.binding, LinkedEquilibrium) and condition.ph is not None:
            populations = ctx.binding.site_populations(state, condition.ph)
        for position in positions:
            parent = inputs.parents[position]
            protonated = compared_tokens(table, parent, "protonated", report_tokens)
            deprotonated = compared_tokens(table, parent, "deprotonated", report_tokens)

            def gap(view) -> float:
                return mean_energy(view, state, position, protonated) - mean_energy(
                    view, state, position, deprotonated
                )

            gap_complex = gap(ctx.complex_view)
            gap_free = gap(ctx.binder_view) + gap(ctx.receptor_view)
            binding_gap = gap_complex - gap_free
            row: Dict[str, Any] = {
                "condition": name,
                "ph": condition.ph,
                "site": f"{inputs.chain_ids[position]}:{inputs.res_ids[position]}",
                "parent": parent,
                "state_token": table.names[int(state[position])],
                "set_by_condition": position in condition.token_by_position,
                "protonated_tokens": [table.names[t] for t in protonated],
                "deprotonated_tokens": [table.names[t] for t in deprotonated],
                "gap_complex": gap_complex,
                "gap_free": gap_free,
                "binding_gap": binding_gap,
                "delta_pka_model": (
                    None if beta is None else -beta * binding_gap / LOG10
                ),
                "fraction_protonated_free": None,
                "fraction_protonated_bound": None,
                "apparent_pka_free": None,
                "apparent_pka_bound": None,
                "apparent_pka_shift": None,
            }
            if populations is not None and position in populations:
                free = populations[position]["free"]
                bound = populations[position]["bound"]
                pka_free = apparent_pka(condition.ph, free)
                pka_bound = apparent_pka(condition.ph, bound)
                row.update(
                    fraction_protonated_free=free,
                    fraction_protonated_bound=bound,
                    apparent_pka_free=pka_free,
                    apparent_pka_bound=pka_bound,
                    apparent_pka_shift=(
                        None
                        if pka_free is None or pka_bound is None
                        else pka_bound - pka_free
                    ),
                )
            rows.append(row)
    return rows
