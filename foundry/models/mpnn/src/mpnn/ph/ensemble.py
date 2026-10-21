"""One binder designed against several target states (for example human and mouse).

Each target state is its own complex with its own Potts tables and its own energies. The
binder is shared: the same residues, with the same sequence being designed. The ensemble
objective evaluates every term in every state with the shared binder tokens written into
that state's sequence, then reduces over states by ``max`` (the worst state) or ``mean``.
Scaling, scalarisation and selection are those of :class:`~mpnn.ph.objective.Objective`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from mpnn.ph.objective import Objective
from mpnn.ph.states import ResolvedCondition

logger = logging.getLogger(__name__)

REDUCTIONS = ("max", "mean")


def reduce_states(stacked: torch.Tensor, how: str) -> torch.Tensor:
    """Reduce over the leading (state) axis."""
    return stacked.amax(dim=0) if how == "max" else stacked.mean(dim=0)


@dataclass
class EnsembleMember:
    """One target state: its objective and how the shared binder maps into it."""

    name: str
    objective: Objective
    conditions: Mapping[str, ResolvedCondition]
    inputs: Any  # mpnn.ph.session.DesignInputs
    base_tokens: torch.Tensor  # this state's own tokens; the binder is overwritten
    position_map: Dict[int, int]  # reference binder position -> position in this state


class EnsembleObjective(Objective):
    """Objective over several target states; design tokens use the first state's indexing.

    ``designable`` lists the shared binder positions (first state's indexing) that the
    design writes into every state. Every other binder position keeps the token its own
    state has, so a fixed residue is not given another state's protonation. Without it,
    all shared binder positions are written.
    """

    def __init__(
        self,
        members: Sequence[EnsembleMember],
        reduce: str = "max",
        scales=None,
        designable: Optional[Sequence[int]] = None,
    ) -> None:
        if not members:
            raise ValueError("An ensemble needs at least one target state.")
        if reduce not in REDUCTIONS:
            raise ValueError(f"reduce must be one of {REDUCTIONS}.")
        reference = members[0]
        names = set(reference.objective.terms)
        for member in members[1:]:
            if set(member.objective.terms) != names:
                raise ValueError(f"State {member.name!r} has different terms.")
        super().__init__(reference.objective.ctx, reference.objective.terms, scales)
        self.members = list(members)
        self.reduce = reduce
        self.designable = None if designable is None else sorted(set(designable))

    def tokens_for(self, member: EnsembleMember, tokens: torch.Tensor) -> torch.Tensor:
        """The state's sequence with the designed binder tokens written in."""
        out = member.base_tokens.clone()
        positions = member.position_map if self.designable is None else self.designable
        for reference_position in positions:
            out[member.position_map[reference_position]] = tokens[reference_position]
        return out

    @staticmethod
    def _mapped_block(member: EnsembleMember, block: Sequence[int]) -> List[int]:
        try:
            return [member.position_map[int(p)] for p in block]
        except KeyError as err:
            raise ValueError(
                f"Position {err.args[0]} is not a binder residue shared by all target "
                "states; only shared binder residues can be designed."
            ) from None

    def block_values(self, name, tokens, block):
        joints = [
            m.objective.block_values(
                name, self.tokens_for(m, tokens), self._mapped_block(m, block)
            )
            for m in self.members
        ]
        return reduce_states(torch.stack(joints), self.reduce)

    def value(self, name, tokens):
        values = torch.tensor(
            [m.objective.value(name, self.tokens_for(m, tokens)) for m in self.members],
            dtype=torch.float64,
        )
        return float(reduce_states(values, self.reduce))

    def per_state_values(
        self, tokens: torch.Tensor, names: Optional[Sequence[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Raw term values in each state, before the reduction."""
        return {
            m.name: m.objective.term_values(self.tokens_for(m, tokens), names)
            for m in self.members
        }
