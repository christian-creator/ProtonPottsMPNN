"""Per-condition protonation states for residues on either side of an interface.

A *condition* (for example "pH 7.4") assigns a protonation state to chosen residues,
binder or receptor alike. A :class:`StateSpec` holds the conditions and names which one
binding should be kept in (``on``) and which one it should be lost in (``off``), so a
7.4-on/6.5-off switch and its reverse differ only by swapping those two names.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Set

import torch

from mpnn.ph.vocab_meta import TokenTable

logger = logging.getLogger(__name__)


_SPEC_KEYS = ("conditions", "on", "off")
_CONDITION_KEYS = ("states", "ph", "label")


@dataclass(frozen=True, order=True)
class SiteKey:
    """A residue identified by chain id and residue number (no insertion codes yet)."""

    chain: str
    res_id: int

    @classmethod
    def parse(cls, text: str) -> "SiteKey":
        chain, sep, number = text.partition(":")
        if not sep or not chain or not number.lstrip("-").isdigit():
            raise ValueError(f"Site {text!r} must look like 'B:57' (chain:res_id).")
        return cls(chain, int(number))

    def __str__(self) -> str:
        return f"{self.chain}:{self.res_id}"


@dataclass(frozen=True)
class Condition:
    """Protonation states to impose on selected residues; others keep their input state.

    ``states`` maps a site to a token name ("HIS-S") or a state word ("protonated").
    ``ph`` is informational here and only used by pH-continuous binding models.
    """

    name: str
    states: Mapping[SiteKey, str]
    ph: Optional[float] = None
    label: str = ""


@dataclass(frozen=True)
class StateSpec:
    """Named conditions plus the roles ``on`` (binds) and ``off`` (binding lost)."""

    conditions: Mapping[str, Condition]
    on: Optional[str] = None
    off: Optional[str] = None

    def __post_init__(self) -> None:
        for role, name in (("on", self.on), ("off", self.off)):
            if name is not None and name not in self.conditions:
                raise ValueError(
                    f"Role {role!r} names condition {name!r}, which is not defined "
                    f"(have {sorted(self.conditions)})."
                )
        if self.on is not None and self.on == self.off:
            raise ValueError("'on' and 'off' must name different conditions.")

    @classmethod
    def from_dict(cls, cfg: Mapping) -> "StateSpec":
        """Build from a plain dict, e.g. parsed YAML::

        {"conditions": {"on": {"ph": 7.4, "states": {"B:57": "HIS-S"}},
                        "off": {"ph": 6.5, "states": {"B:57": "HIS-P"}}},
         "on": "on", "off": "off"}

        ``on``/``off`` default to conditions of those names when they exist.
        """
        unknown = set(cfg) - set(_SPEC_KEYS)
        if unknown:
            raise ValueError(
                f"State spec has unknown keys {sorted(unknown)}; allowed: {list(_SPEC_KEYS)}."
            )
        raw = cfg.get("conditions")
        if not raw:
            raise ValueError("A state spec needs at least one condition.")
        conditions: Dict[str, Condition] = {}
        for name, body in raw.items():
            if not isinstance(body, Mapping):
                raise ValueError(f"Condition {name!r} must be a mapping.")
            unknown = set(body) - set(_CONDITION_KEYS)
            if unknown:
                raise ValueError(
                    f"Condition {name!r} has unknown keys {sorted(unknown)}; "
                    f"allowed: {list(_CONDITION_KEYS)}."
                )
            if "states" not in body:
                raise ValueError(
                    f"Condition {name!r} needs a 'states' mapping (it may be empty)."
                )
            if not isinstance(body["states"], Mapping):
                raise ValueError(
                    f"Condition {name!r}: 'states' must be a mapping of site to state."
                )
            states = {
                site if isinstance(site, SiteKey) else SiteKey.parse(site): spec
                for site, spec in dict(body.get("states", {})).items()
            }
            conditions[name] = Condition(
                name=name,
                states=states,
                ph=body.get("ph"),
                label=body.get("label", ""),
            )
        on = cfg.get("on", "on" if "on" in conditions else None)
        off = cfg.get("off", "off" if "off" in conditions else None)
        return cls(conditions=conditions, on=on, off=off)

    def sites(self) -> Set[SiteKey]:
        return {site for cond in self.conditions.values() for site in cond.states}

    def uncovered_sites(self) -> Dict[str, Set[SiteKey]]:
        """Per condition, the sites some other condition sets but this one leaves alone."""
        everything = self.sites()
        return {
            name: everything - set(cond.states)
            for name, cond in self.conditions.items()
            if everything - set(cond.states)
        }


@dataclass(frozen=True)
class ResolvedCondition:
    """A condition expressed as token indices at sequence positions."""

    name: str
    token_by_position: Mapping[int, int]
    ph: Optional[float] = None

    def apply(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return a copy of the 1-D token tensor ``tokens`` with the overrides applied."""
        out = tokens.clone()
        for position, token in self.token_by_position.items():
            out[position] = token
        return out


def site_index_from_arrays(
    chain_ids: Sequence[str], res_ids: Sequence[int]
) -> Dict[SiteKey, int]:
    """Map each residue to its sequence position; refuse ambiguous (duplicate) keys."""
    index: Dict[SiteKey, int] = {}
    duplicates = []
    for position, (chain, res_id) in enumerate(zip(chain_ids, res_ids)):
        key = SiteKey(str(chain), int(res_id))
        if key in index:
            duplicates.append(str(key))
        index[key] = position
    if duplicates:
        raise ValueError(
            f"Residue keys are not unique (insertion codes or symmetry copies?): "
            f"{sorted(set(duplicates))[:10]}"
        )
    return index


def resolve_condition(
    condition: Condition,
    table: TokenTable,
    site_index: Mapping[SiteKey, int],
    parents: Sequence[str],
) -> ResolvedCondition:
    """Resolve one condition onto sequence positions.

    ``parents[i]`` is the bare residue name (``HIS``, ``ASP``, ...) at position ``i``.
    Raises on unknown sites, non-titratable residues and state/residue mismatches.
    """
    unknown = [str(site) for site in condition.states if site not in site_index]
    if unknown:
        raise KeyError(
            f"Condition {condition.name!r}: sites not in the structure: {unknown}"
        )
    titratable = set(table.titratable_parents())
    token_by_position: Dict[int, int] = {}
    for site, spec in condition.states.items():
        position = site_index[site]
        parent = parents[position]
        if parent not in titratable:
            raise ValueError(
                f"Condition {condition.name!r}: {site} is {parent}, which has no "
                f"protonation states in this vocabulary (titratable: {sorted(titratable)})."
            )
        token = table.resolve(parent, spec)
        token_by_position[position] = table.index(token)
    return ResolvedCondition(condition.name, token_by_position, condition.ph)


def resolve_spec(
    spec: StateSpec,
    table: TokenTable,
    site_index: Mapping[SiteKey, int],
    parents: Sequence[str],
) -> Dict[str, ResolvedCondition]:
    """Resolve every condition; warn when conditions set different sites."""
    for name, missing in spec.uncovered_sites().items():
        logger.warning(
            "Condition %r leaves %s at their input state while another condition sets them.",
            name,
            sorted(str(site) for site in missing),
        )
    return {
        name: resolve_condition(cond, table, site_index, parents)
        for name, cond in spec.conditions.items()
    }
