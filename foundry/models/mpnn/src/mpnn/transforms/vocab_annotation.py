"""The protonation-labelling Transform.

This module used to CONTAIN the labelling logic. It no longer does: each vocabulary now lives in its
own frozen module (``extended_vocab_v3.py`` / ``extended_vocab_v4.py``) and this file only dispatches
to the one named by ``extended_vocab``. Keeping a second, live copy of the classifier here would
defeat the point of freezing them — an edit to it would silently do nothing.

See :mod:`mpnn.transforms.extended_vocab` for the registry and what each vocabulary is.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import biotite.structure as struc
from atomworks.ml.transforms.base import Transform

from mpnn.transforms.extended_vocab import get_vocab

_TITRATABLE = ("HIS", "ASP", "GLU")
# sd_cut (the ambiguous -A gate) is FIXED — only prob_thr is swept. Read from the same JSON the
# precompute build used, so the -A band is identical across every threshold in a sweep.
_THRESHOLDS_JSON = (Path(__file__).parent / "ev6" / "weights" / "thresholds.json")

# Kept for backwards compatibility: scripts/13_charge_network_prototype.py imports these.
# The vocabularies each carry their OWN copy of _hb_quality — changing this one does not change them.
_ASP_OXYGENS = {"OD1", "OD2"}
_GLU_OXYGENS = {"OE1", "OE2"}


def _hb_quality(dist: float, angle: float, beta: float) -> float:
    """H-bond quality (higher = better): shorter distance + straighter D-H..A angle.
    Distance in Angstrom; angle penalty is beta*(180-angle)/100 (skipped if angle NaN)."""
    q = -float(dist)
    if not np.isnan(angle):
        q -= beta * (180.0 - float(angle)) / 100.0
    return q


class AnnotateProtonationStates(Transform):
    """Classify HIS, ASP and GLU by protonation state; store the result as the per-atom string
    annotation ``protonation_label``. Must run after CalculateHbondsPlus and AnnotateSaltBridges.

    Labels:
        HIS  -> HIS-P | HID | HIE | HIS-D | HIS-A
        ASP  -> ASP-P | ASP-D | ASP-A
        GLU  -> GLU-P | GLU-D | GLU-A
        everything else -> "" (empty string)

    Parameters
    ----------
    extended_vocab : str
        WHICH vocabulary to label with -- ``"v3"`` or ``"v4"`` (see
        :mod:`mpnn.transforms.extended_vocab`). Each is a frozen, self-contained labeller, so a
        checkpoint can always be scored with exactly the labels it was trained on. The vocabulary
        also fixes ``cutoff_HA_dist`` / ``filter_capability``, which the PIPELINE applies upstream --
        they decide which bonds this classifier ever sees, so they are part of the vocabulary rather
        than free knobs.
    use_salt_bridge : bool
        Whether the PLIP salt-bridge proximity prior is used. Both vocabularies support it, but they
        consult it at different points: v3 inside the decision tree (whenever geometry is not fully
        clear), v4 as a post-pass over residues still ``*-A`` -- a much smaller trigger set.
    deterministic : bool
        For donor/acceptor ties geometry cannot separate: True -> leave it undecided; False -> sample
        the role with a probability set by the quality gap (training-time augmentation). Training uses
        the vocabulary's ``train_deterministic``; validation/inference is always True (stable S).
    margin : float
        Bond-quality gap (Angstrom-equivalent) above which a donor/acceptor role is the clear winner.
    temperature : float
        Sampling sharpness when deterministic=False (smaller = harder decisions).
    beta : float
        Weight of the D-H..A angle penalty in the bond-quality score (0 = distance only).
    seed : int | None
        Seed for the sampling RNG. None -> fresh randomness each call (re-sampled per epoch).
        Only matters when deterministic=False.
    classify : callable | None
        A labeller to use instead of the vocabulary's own; see ``self.classify`` below.
    """

    def __init__(
        self,
        extended_vocab: str = "v4",
        deterministic: bool = True,
        margin: float = 0.3,
        temperature: float = 0.2,
        beta: float = 1.0,
        seed: int | None = None,
        use_salt_bridge: bool = True,
        persist_scores: bool = False,
        classify=None,
    ):
        self.extended_vocab = extended_vocab
        vocab = get_vocab(extended_vocab)   # raises on an unknown name
        # ``classify`` replaces the vocabulary's labeller (same call: atom_array -> {(chain, res_id,
        # res_name): token}), for example mpnn.transforms.propka_labeller.PropkaLabeller. The tokens
        # must belong to the vocabulary, since the encoder reads them.
        self.classify = classify if classify is not None else vocab["classify"]
        # v6-only labeller that returns the raw FLAML score alongside the token, so the snapshot can carry
        # the continuous p/sd/metal and the threshold be swept at train time. None for v3/v4.
        self.classify_scores = vocab.get("classify_scores")
        self.deterministic = deterministic
        self.margin = margin
        self.temperature = temperature
        self.beta = beta
        self.seed = seed
        self.use_salt_bridge = use_salt_bridge
        # When True, ALSO persist per-atom flaml_p / flaml_sd / flaml_metal annotations (broadcast across
        # each residue like protonation_label). Requires a vocabulary that exposes continuous scores (v6).
        self.persist_scores = persist_scores
        if persist_scores and self.classify_scores is None:
            raise ValueError(
                f"persist_scores=True needs a vocabulary with continuous FLAML scores (v6); "
                f"extended_vocab={extended_vocab!r} has none."
            )

    def forward(self, data: dict) -> dict:
        aa = data["atom_array"]
        n_atoms = len(aa)
        res_starts = struc.get_residue_starts(aa)

        # Persist path (v6): one predict() returns the DEFAULT-threshold token AND the raw p/sd/metal, so we
        # avoid a second FLAML pass and the stored token stays env-independent. Labelling path: the usual
        # token-only classifier (env-aware operating point for v6).
        if self.persist_scores:
            scored = self.classify_scores(aa, hbond_records=data.get("hbond_records"))
            classifications = {k: v[0] for k, v in scored.items()}
        else:
            rng = np.random.default_rng(self.seed)
            classifications = self.classify(
                aa,
                deterministic=self.deterministic,
                margin=self.margin,
                temperature=self.temperature,
                beta=self.beta,
                rng=rng,
                use_salt_bridge=self.use_salt_bridge,
                # The bond pool CalculateHbondsPlus already built, for labellers that want the raw geometry
                # rather than the per-atom annotations. v3/v4 read the annotations and ignore this; v6's
                # feature models consume it instead of re-running HBPLUS. Absent -> v6 recomputes it.
                hbond_records=data.get("hbond_records"),
            )

        labels = np.full(n_atoms, "", dtype=object)
        if self.persist_scores:
            flaml_p = np.full(n_atoms, np.nan, dtype=np.float32)
            flaml_sd = np.full(n_atoms, np.nan, dtype=np.float32)
            flaml_metal = np.zeros(n_atoms, dtype=bool)

        for k, start in enumerate(res_starts):
            end = res_starts[k + 1] if k + 1 < len(res_starts) else n_atoms
            key = (str(aa.chain_id[start]), int(aa.res_id[start]), str(aa.res_name[start]))
            if key in classifications:
                labels[start:end] = classifications[key]
                if self.persist_scores:
                    _, p, sd, metal = scored[key]
                    flaml_p[start:end] = p
                    flaml_sd[start:end] = sd
                    flaml_metal[start:end] = metal

        aa.set_annotation("protonation_label", labels)
        if self.persist_scores:
            aa.set_annotation("flaml_p", flaml_p)
            aa.set_annotation("flaml_sd", flaml_sd)
            aa.set_annotation("flaml_metal", flaml_metal)
        data["atom_array"] = aa
        return data


class ApplyProtonationThreshold(Transform):
    """Re-derive ``protonation_label`` from the persisted FLAML scores at a configurable ``prob_thr``.

    The precompute cache stores the raw ensemble scores (``flaml_p`` / ``flaml_sd`` / ``flaml_metal``,
    persisted by ``AnnotateProtonationStates(persist_scores=True)``). This transform re-runs ONLY the cheap
    two-cut decision (``discretize_protonation``) at a per-run ``prob_thr``, overwriting the baked
    ``protonation_label`` before the sequence ``S`` is built. That makes the protonation operating point a
    train-time knob to sweep, without re-running FLAML (pure NumPy → fork-safe in DataLoader workers).

    ``sd_cut`` (the ambiguous ``-A`` gate) is held FIXED at the ``thresholds.json`` value — only ``prob_thr``
    (the P-vs-D cut) moves, by design. NO-OP when ``flaml_p`` is absent (old snapshots, v3/v4), so existing
    caches and non-v6 pipelines are untouched.

    Parameters
    ----------
    his_prob_thr, acid_prob_thr : float | None
        The P-vs-(S/D) probability cut for His and the acids. None -> the ``thresholds.json`` default (so the
        transform reproduces the baked label exactly, a useful identity/regression check).
    """

    def __init__(self, his_prob_thr: float | None = None, acid_prob_thr: float | None = None):
        cfg = json.loads(_THRESHOLDS_JSON.read_text())
        self._his_sd_cut = cfg["his"]["sd_cut"]
        self._acid_sd_cut = cfg["acid"]["sd_cut"]
        self._his_thr = his_prob_thr if his_prob_thr is not None else cfg["his"]["prob_thr"]
        self._acid_thr = acid_prob_thr if acid_prob_thr is not None else cfg["acid"]["prob_thr"]

    def forward(self, data: dict) -> dict:
        from mpnn.transforms.ev6 import discretize_protonation

        aa = data["atom_array"]
        cats = aa.get_annotation_categories()
        # Nothing to threshold on: leave the baked label as-is (old snapshots / v3 / v4).
        if not {"flaml_p", "flaml_sd", "flaml_metal"} <= set(cats):
            return data

        n_atoms = len(aa)
        res_starts = struc.get_residue_starts(aa)
        labels = (aa.protonation_label.copy()
                  if "protonation_label" in cats else np.full(n_atoms, "", dtype=object))
        p, sd, metal = aa.flaml_p, aa.flaml_sd, aa.flaml_metal

        for k, start in enumerate(res_starts):
            stem = str(aa.res_name[start])
            if stem not in _TITRATABLE or np.isnan(p[start]):
                continue
            end = res_starts[k + 1] if k + 1 < len(res_starts) else n_atoms
            sd_cut = self._his_sd_cut if stem == "HIS" else self._acid_sd_cut
            thr = self._his_thr if stem == "HIS" else self._acid_thr
            labels[start:end] = discretize_protonation(
                float(p[start]), float(sd[start]), bool(metal[start]), stem, sd_cut, thr
            )

        aa.set_annotation("protonation_label", labels)
        data["atom_array"] = aa
        return data
