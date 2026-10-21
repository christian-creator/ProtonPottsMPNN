"""Direction-neutral metadata for protonation-state vocabulary tokens.

Vocabularies name residue states with a suffix (``HIS-P``, ``ASP-D``, ...). Code that
needs the protonated or deprotonated form of a residue should ask :class:`TokenTable`
instead of parsing suffixes or hard-coding dicts, so that no caller assumes which state
is "on". The table is built from plain token names and needs no atomworks.

Suffix convention (v6): ``P`` protonated, ``D`` deprotonated acid, ``S`` neutral His
(the deprotonated form of ``HIS-P``), ``A`` ambiguous. Legacy v3/v4 names ``HID`` and
``HIE`` (neutral tautomers) and ``HIS-D`` (imidazolate) are recognised too.
"""

from __future__ import annotations

import enum
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

STANDARD_RESIDUES: Tuple[str, ...] = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
)  # fmt: skip
UNKNOWN_TOKEN = "UNK"
# STANDARD_RESIDUES is alphabetical by three-letter code, so this string follows it.
THREE_TO_ONE: Dict[str, str] = dict(zip(STANDARD_RESIDUES, "ARNDCQEGHILKMFPSTWYV"))


class Protonation(str, enum.Enum):
    """Physical protonation state of a side chain, independent of pH direction."""

    PROTONATED = "protonated"
    DEPROTONATED = "deprotonated"
    AMBIGUOUS = "ambiguous"
    UNSPECIFIED = "unspecified"  # bare residue token: state not stated


# Side-chain formal charge per titratable parent and state.
TITRATABLE_CHARGE: Dict[str, Dict[Protonation, int]] = {
    "HIS": {Protonation.PROTONATED: 1, Protonation.DEPROTONATED: 0},
    "ASP": {Protonation.PROTONATED: 0, Protonation.DEPROTONATED: -1},
    "GLU": {Protonation.PROTONATED: 0, Protonation.DEPROTONATED: -1},
}

_SUFFIX_STATE: Dict[str, Protonation] = {
    "P": Protonation.PROTONATED,
    "S": Protonation.DEPROTONATED,
    "D": Protonation.DEPROTONATED,
    "A": Protonation.AMBIGUOUS,
}
_LEGACY_NEUTRAL_HIS = ("HID", "HIE")
_CHARGE_OVERRIDE: Dict[str, int] = {"HIS-D": -1}  # imidazolate, not a neutral His


@dataclass(frozen=True)
class TokenMeta:
    """What a vocabulary token means: its parent residue, state and side-chain charge."""

    name: str
    parent: str
    state: Protonation
    charge: Optional[int]  # None when the state carries no stated charge


def parse_token(name: str) -> TokenMeta:
    """Parse one vocabulary token name; fail loudly on names it does not know."""
    if name == UNKNOWN_TOKEN or name in STANDARD_RESIDUES:
        return TokenMeta(name, name, Protonation.UNSPECIFIED, None)
    if name in _LEGACY_NEUTRAL_HIS:
        return TokenMeta(name, "HIS", Protonation.DEPROTONATED, 0)
    parent, sep, suffix = name.partition("-")
    if not sep or parent not in TITRATABLE_CHARGE or suffix not in _SUFFIX_STATE:
        raise ValueError(f"Cannot interpret vocabulary token {name!r}.")
    state = _SUFFIX_STATE[suffix]
    if state is Protonation.AMBIGUOUS:
        charge = None
    else:
        charge = _CHARGE_OVERRIDE.get(name, TITRATABLE_CHARGE[parent][state])
    return TokenMeta(name, parent, state, charge)


class TokenTable:
    """Metadata for an ordered vocabulary; token index = position in ``token_names``."""

    def __init__(self, token_names: Sequence[str]) -> None:
        names = list(token_names)
        if len(set(names)) != len(names):
            raise ValueError("Duplicate token names in vocabulary.")
        self._names = names
        self._index = {name: i for i, name in enumerate(names)}
        self._meta = [parse_token(name) for name in names]
        by_role: Dict[Tuple[str, Protonation], List[str]] = defaultdict(list)
        for meta in self._meta:
            by_role[(meta.parent, meta.state)].append(meta.name)
        self._by_role = {role: tuple(tokens) for role, tokens in by_role.items()}

    def __len__(self) -> int:
        return len(self._names)

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(self._names)

    def index(self, name: str) -> int:
        try:
            return self._index[name]
        except KeyError:
            raise KeyError(f"Token {name!r} is not in the vocabulary.") from None

    def meta(self, token: Union[str, int]) -> TokenMeta:
        idx = self.index(token) if isinstance(token, str) else int(token)
        return self._meta[idx]

    def tokens_for(self, parent: str, state: Protonation) -> Tuple[str, ...]:
        """All tokens of ``parent`` in ``state`` (several for legacy His tautomers)."""
        return self._by_role.get((parent, Protonation(state)), ())

    def titratable_parents(self) -> Tuple[str, ...]:
        """Parents that have both a protonated and a deprotonated token."""
        parents = {parent for parent, _ in self._by_role}
        return tuple(
            sorted(
                p
                for p in parents
                if self.tokens_for(p, Protonation.PROTONATED)
                and self.tokens_for(p, Protonation.DEPROTONATED)
            )
        )

    def parent_index(self) -> List[int]:
        """For each token index, the index of its bare parent residue token."""
        out = []
        for meta in self._meta:
            if meta.parent not in self._index:
                raise KeyError(
                    f"Bare parent {meta.parent!r} of {meta.name!r} is missing."
                )
            out.append(self._index[meta.parent])
        return out

    def design_mask(
        self,
        allow_bare_titratable: Sequence[str] = (),
        forbid: Sequence[str] = (UNKNOWN_TOKEN,),
    ) -> List[bool]:
        """Which tokens a freely designed position may take.

        Protonation-state tokens are never allowed: a designed residue has no stated
        state, and letting the optimiser pick one lets it dodge the condition states.
        Bare His/Asp/Glu are excluded too unless listed in ``allow_bare_titratable``
        (their bare-token energies mean "state unspecified" and are less calibrated).
        """
        titratable = set(self.titratable_parents())
        unknown = set(allow_bare_titratable) - titratable
        if unknown:
            raise ValueError(f"Not titratable in this vocabulary: {sorted(unknown)}")
        mask = []
        for meta in self._meta:
            allowed = meta.state is Protonation.UNSPECIFIED and meta.name not in forbid
            if allowed and meta.parent in titratable:
                allowed = meta.parent in allow_bare_titratable
            mask.append(allowed)
        return mask

    def canonical_sequence(self, tokens: Sequence[int]) -> str:
        """One-letter sequence of the parent residues (state tokens fold onto them)."""
        parents = self.parent_index()
        return "".join(
            THREE_TO_ONE.get(self._names[parents[int(t)]], "X") for t in tokens
        )

    def token_names(self, tokens: Sequence[int]) -> List[str]:
        """Token names, including protonation states, for a list of token indices."""
        return [self._names[int(t)] for t in tokens]

    def resolve(self, parent: str, spec: str) -> str:
        """Turn a token name or a state word into one token of ``parent``.

        ``spec`` is either a token name (checked against ``parent``) or a state word
        ("protonated", "deprotonated", ...), which must name exactly one token.
        """
        if spec in self._index:
            meta = self.meta(spec)
            if meta.parent != parent:
                raise ValueError(f"Token {spec!r} is a {meta.parent}, not a {parent}.")
            return spec
        try:
            state = Protonation(spec.lower())
        except ValueError:
            raise ValueError(
                f"{spec!r} is neither a vocabulary token nor a protonation state."
            ) from None
        tokens = self.tokens_for(parent, state)
        if not tokens:
            raise ValueError(f"The vocabulary has no {state.value} token for {parent}.")
        if len(tokens) > 1:
            raise ValueError(
                f"{state.value} {parent} is ambiguous in this vocabulary {tokens}; "
                "name the token explicitly."
            )
        return tokens[0]
