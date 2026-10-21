"""Protonation labels and reference pKa values from PROPKA (an optional dependency).

The default labeller (EV6, ``mpnn.transforms.ev6``) reads a structure's geometry and gives each
His, Asp and Glu one call that does not depend on pH. PROPKA gives each residue a pKa instead, so
a state needs a pH: a residue is protonated when its pKa is above the pH, neutral when below, and
ambiguous (``-A``) when the two are closer than ``margin``. The checkpoint was trained on EV6
labels, so PROPKA labels are a different, pH-dependent labelling to compare against, not a
drop-in replacement.

PROPKA is also the source of per-site reference pKa values for the ``linked_equilibrium`` binding
model (see ``mpnn.ph.binding``); for that use it is run on each free partner, since the pKa shift
from binding comes from the Potts energies.

PROPKA is imported only when it is used: ``pip install propka``.

References: Olsson MHM et al., J Chem Theory Comput 7:525 (2011), PROPKA3;
Søndergaard CR et al., J Chem Theory Comput 7:2284 (2011).
"""

from __future__ import annotations

import contextlib
import io
import logging
import math
import string
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_CAVEAT = (
    "Starting states come from PROPKA, but the checkpoint was trained on EV6 labels. Against "
    "experiment PROPKA labels are weak (neutron structures: ROC AUC 0.62 for His, 0.78 for "
    "Asp/Glu); see the README, Protonation source."
)
_WORDS = {"P": "protonated", "S": "neutral", "D": "deprotonated", "A": "ambiguous"}
_caveat_shown = False

# PROPKA reports a carboxylate group for every Asp, Glu and C terminus, so a side chain is told
# apart by the atom its group is anchored on (the C terminus is anchored on "C").
_SIDE_CHAIN_ANCHOR = {"ASP": "CG", "GLU": "CD", "HIS": "CG"}
_NEUTRAL_SUFFIX = {"HIS": "S", "ASP": "D", "GLU": "D"}
_PDB_CHAIN_CHARS = string.ascii_uppercase + string.ascii_lowercase + string.digits

ResidueKey = Tuple[str, int, str]  # (chain_id, res_id, res_name)


def _import_propka():
    try:
        import propka.run
    except ImportError as err:
        raise ImportError(
            "PROPKA is not installed. It is optional and only needed when the protonation source "
            "is 'propka'; install it with `pip install propka`."
        ) from err
    return propka.run


def require_propka() -> None:
    """Raise now, with the install instruction, if PROPKA is missing."""
    _import_propka()


def _single_char_chains(atom_array):
    """A copy with one-character chain ids (the PDB format needs them) and the way back."""
    original = [str(c) for c in atom_array.chain_id]
    chains = list(dict.fromkeys(original))
    if all(len(c) == 1 for c in chains):
        return atom_array, {c: c for c in chains}
    if len(chains) > len(_PDB_CHAIN_CHARS):
        raise ValueError(f"PROPKA reads at most {len(_PDB_CHAIN_CHARS)} chains, got {len(chains)}.")
    to_char = dict(zip(chains, _PDB_CHAIN_CHARS))
    renamed = atom_array.copy()
    renamed.chain_id = np.array([to_char[c] for c in original], dtype=atom_array.chain_id.dtype)
    return renamed, {v: k for k, v in to_char.items()}


def propka_pka(atom_array) -> Dict[ResidueKey, float]:
    """PROPKA's pKa of every side-chain His, Asp and Glu: ``{(chain_id, res_id, res_name): pKa}``.

    Hetero atoms are dropped first, so ligands and metals do not shift the values.
    """
    import biotite.structure.io.pdb as pdb_io

    run = _import_propka()
    protein = atom_array[~atom_array.hetero]
    protein, back = _single_char_chains(protein)
    buffer = io.StringIO()
    pdb_file = pdb_io.PDBFile()
    pdb_io.set_structure(pdb_file, protein)
    pdb_file.write(buffer)
    buffer.seek(0)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            molecule = run.single("structure.pdb", stream=buffer, write_pka=False)
    except Exception as err:
        raise RuntimeError(f"PROPKA failed on this structure: {err}") from err
    conformations = [c for name, c in molecule.conformations.items() if name != "AVR"]
    if not conformations:
        raise RuntimeError("PROPKA returned no conformation for this structure.")
    pka: Dict[ResidueKey, float] = {}
    for group in conformations[0].get_titratable_groups():
        name = str(group.atom.res_name).strip()
        if _SIDE_CHAIN_ANCHOR.get(name) != str(group.atom.name).strip():
            continue
        chain = back[str(group.atom.chain_id).strip()]
        pka[(chain, int(group.atom.res_num), name)] = float(group.pka_value)
    return pka


def propka_site_pka(atom_array) -> Dict[Tuple[str, int], float]:
    """As :func:`propka_pka`, keyed ``(chain_id, res_id)`` for a structure of free partners."""
    return {(chain, res_id): value for (chain, res_id, _), value in propka_pka(atom_array).items()}


def token_from_pka(res_name: str, pka: float, ph: float, margin: float) -> str:
    """The protonation token of one residue at ``ph``.

    ``-A`` when the pKa is within ``margin`` of the pH (or missing), ``-P`` when it is above, else
    the neutral token (``-S`` for His, ``-D`` for Asp and Glu). A pKa exactly at the pH with
    ``margin`` 0 counts as neutral.
    """
    if not math.isfinite(pka) or abs(pka - ph) < margin:
        return f"{res_name}-A"
    if pka > ph:
        return f"{res_name}-P"
    return f"{res_name}-{_NEUTRAL_SUFFIX[res_name]}"


@dataclass(frozen=True)
class PropkaLabeller:
    """Drop-in for a vocabulary's ``classify``: ``atom_array -> {(chain, res_id, res_name): token}``.

    Frozen, so it can be part of the key under which the featurisation pipeline is cached.
    """

    ph: float
    margin: float = 0.5

    def __post_init__(self) -> None:
        if not math.isfinite(self.ph):
            raise ValueError("The PROPKA labeller needs a finite pH.")
        if not math.isfinite(self.margin) or self.margin < 0:
            raise ValueError("margin must be a finite number >= 0.")

    def __call__(self, atom_array, **_ignored) -> Dict[ResidueKey, str]:
        # The vocabulary classifiers receive options such as margin= and hbond_records=; they
        # belong to the geometric labellers and are ignored here.
        pka = propka_pka(atom_array)
        titratable = atom_array[np.isin(atom_array.res_name, list(_NEUTRAL_SUFFIX))]
        residues = dict.fromkeys(
            (str(c), int(r), str(n))
            for c, r, n in zip(titratable.chain_id, titratable.res_id, titratable.res_name)
        )
        unscored = [key for key in residues if key not in pka]
        if unscored:
            logger.warning(
                "PROPKA gave no pKa for %d His/Asp/Glu residue(s), so they are labelled ambiguous. First: %s.",
                len(unscored),
                unscored[:5],
            )
        tokens = {
            key: token_from_pka(key[2], pka.get(key, math.nan), self.ph, self.margin) for key in residues
        }
        self._report(tokens, sorted({str(c) for c in atom_array.chain_id}))
        return tokens

    def _report(self, tokens: Dict[ResidueKey, str], chains) -> None:
        global _caveat_shown
        if not _caveat_shown:
            logger.warning(_CAVEAT)
            _caveat_shown = True
        by_type: Dict[str, Dict[str, int]] = {}
        for (_, _, name), token in tokens.items():
            counts = by_type.setdefault(name, {})
            counts[token[-1]] = counts.get(token[-1], 0) + 1
        summary = ", ".join(
            f"{name} {n} {_WORDS[letter]}"
            for name, counts in sorted(by_type.items())
            for letter, n in sorted(counts.items())
        )
        logger.info(
            "PROPKA states for chain(s) %s at pH %.2f, margin %.2f: %s.",
            ",".join(chains),
            self.ph,
            self.margin,
            summary or "no His, Asp or Glu",
        )
