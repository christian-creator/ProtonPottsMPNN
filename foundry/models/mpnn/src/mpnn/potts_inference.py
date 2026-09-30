"""
PottsMPNN inference script.

Four steps:
    model     = load_model(CHECKPOINT)
    batch     = prepare_potts_input(STRUCTURE, fixed_chains=["B"])
    potts_out = run_forward(model, batch)
    seq_opt, energies = run_potts_optimize(potts_out, *get_masks(model, batch))
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from biotite.structure import AtomArray

from mpnn.collate.feature_collator import FeatureCollator
from mpnn.model.pottsmpnn import PottsMPNN, PottsOutput
from mpnn.model.layers.graph_embeddings import PottsProteinFeatures
from mpnn.pipelines.potts_mpnn import build_mpnn_transform_pipeline
from mpnn.transforms.feature_aggregation.token_encodings import MPNN_TOKEN_ENCODING
from mpnn.utils.inference import MPNNInferenceInput


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULTS = {
    "num_sequences":    10,
    "temperature":      0.01,    # T→0 = greedy; T>0 = stochastic
    "max_iters":        1000,
    "convergence_mode": True,
    "structure_noise":  0.0,
    "extended_vocab":   False,
}


# ---------------------------------------------------------------------------
# Step 1 — load model
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str | Path,
    extended_vocab: bool | str | None = DEFAULTS["extended_vocab"],
    etab_source: str | None = None,
) -> PottsMPNN:
    """Load a PottsMPNN checkpoint.

    ``etab_source`` ("edge" vs "node_edge_node") is auto-detected from the checkpoint's
    ``etab_out`` weight shape when left as None, so old and new checkpoints both just work.

    ``extended_vocab`` may be a vocabulary NAME ("v3"/"v4"/"v6") or a legacy bool. Checkpoints now
    RECORD their own vocabulary (``train_cfg.extended_vocab``), and that name is used automatically --
    which closes the old silent-mismatch hazard, since the token set changes no weight shape for v3-vs-v4
    (both 32 tokens) and nothing could catch it. An explicit name still wins, for checkpoints written
    before the vocabulary was recorded. The loaded model carries ``extended_vocab_name`` so downstream
    scoring (PKAD, recovery) converts through the right ``aa_protonated``/``aa_deprotonated`` maps.
    """
    device = _resolve_device()
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if etab_source is None:
        etab_source = PottsMPNN.infer_etab_source(ckpt["model"])

    try:                                        # the vocabulary the checkpoint says it was trained on
        ckpt_vocab = ckpt.get("train_cfg", {}).get("extended_vocab")
    except Exception:
        ckpt_vocab = None
    vocab_name = extended_vocab if isinstance(extended_vocab, str) else ckpt_vocab

    # Reproduce the checkpoint's LABEL OPERATING POINT (train_cfg.{his,acid}_prob_thr) in the live EV6
    # labeller, so prepare_potts_input builds S at the cut the model was trained at rather than the
    # thresholds.json default. An explicit EV6_*_PROB_THR env var still wins. See
    # extended_vocab_v6.set_checkpoint_operating_point.
    try:
        _tc = ckpt.get("train_cfg") or {}
        _his_thr, _acid_thr = _tc.get("his_prob_thr"), _tc.get("acid_prob_thr")
    except Exception:
        _his_thr = _acid_thr = None
    if _his_thr is not None or _acid_thr is not None:
        from mpnn.transforms import extended_vocab_v6
        extended_vocab_v6.set_checkpoint_operating_point(_his_thr, _acid_thr)

    if vocab_name:
        from mpnn.transforms.extended_vocab import get_vocab
        feats = PottsProteinFeatures(token_encoding=get_vocab(vocab_name)["token_encoding"])
        model = PottsMPNN(graph_featurization_module=feats, etab_source=etab_source)
    elif extended_vocab:                        # legacy bool: the 32-token default
        model = PottsMPNN(graph_featurization_module=PottsProteinFeatures(), etab_source=etab_source)
    else:
        model = PottsMPNN(etab_source=etab_source)
    model.extended_vocab_name = vocab_name

    model.load_state_dict(ckpt["model"], strict=True)
    return model.eval().to(device)


# ---------------------------------------------------------------------------
# Step 2 — prepare input
# ---------------------------------------------------------------------------

_PIPELINE_CACHE: dict = {}


def _get_potts_pipeline(device, build_bond_labels: bool = False, hbond_scope: str = "sc_any",
                        extended_vocab: str = "v4",
                        use_salt_bridge: bool = True,
                        deterministic: bool = True,
                        protonation_seed: int | None = None):
    """Cache the (structure-independent) Potts featurization pipeline by device.

    The build args don't depend on the input structure, so rebuilding it on every call only
    wasted work AND churned ~30k objects per structure — which progressively slowed long loops
    over prepare_potts_input (e.g. scripts/07_assess_rf3_exp3.py: a GC sweep over the accumulated
    objects stalled for tens of seconds). Build once, reuse — same pattern as callers that cache.

    ``build_bond_labels=True`` also persists the HBPLUS / PLIP ground-truth bonds as the E_idx-alignable
    ``hbond_donates_to`` / ``hbond_accepts_from`` / ``salt_partners`` token-pair partner lists (the
    annotation is already computed for protonation labelling; this just keeps it).
    """
    key = (str(device), bool(build_bond_labels), str(hbond_scope), str(extended_vocab),
           bool(use_salt_bridge), bool(deterministic), protonation_seed)
    if key not in _PIPELINE_CACHE:
        _PIPELINE_CACHE[key] = build_mpnn_transform_pipeline(
            model_type="potts_mpnn", is_inference=True, minimal_return=True, device=device,
            build_bond_labels=build_bond_labels, hbond_scope=hbond_scope,
            extended_vocab=extended_vocab,
            use_salt_bridge=use_salt_bridge,
            deterministic=deterministic, protonation_seed=protonation_seed,
        )
    return _PIPELINE_CACHE[key]


def prepare_potts_input(
    structure: str | Path | AtomArray,
    *,
    fixed_residues: list[str] | None = None,
    fixed_chains:   list[str] | None = None,
    designed_chains: list[str] | None = None,
    structure_noise: float = DEFAULTS["structure_noise"],
    build_bond_labels: bool = False,
    hbond_scope: str = "sc_any",
    extended_vocab: str = "v4",      # WHICH protonation vocabulary: "v3" (ev3's original labeller) or
                                     # "v4" (what potts_sb_* trains on). MUST MATCH THE CHECKPOINT --
                                     # the encoder sees these tokens in S, and unlike field_source there
                                     # is no state_dict key to catch a mismatch. It brings its own
                                     # H-bond cutoff + capability filter. See transforms/extended_vocab.py.
    use_salt_bridge: bool = True,    # PLIP salt-bridge proximity prior (both vocabularies support it).
    deterministic: bool = True,      # False -> SAMPLE ambiguous roles + symmetric shared-proton dyads.
    protonation_seed: int | None = None,  # seed the sampling; None -> fresh each call
) -> dict:
    """Featurize a PDB/CIF file or AtomArray.

    Returns a dict with keys ``network_input`` and ``atom_array``. With ``build_bond_labels=True`` the
    returned ``network_input["input_features"]`` also carries the HBPLUS / PLIP ground-truth bonds as
    ``hbond_donates_to`` / ``hbond_accepts_from`` / ``salt_partners`` token-pair partner lists.
    """
    device = _resolve_device()

    settings: dict[str, Any] = {
        "fixed_residues":   fixed_residues,
        "fixed_chains":     fixed_chains,
        "designed_chains":  designed_chains,
        "structure_noise":  structure_noise,
        "decode_type":      "auto_regressive",
        "causality_pattern": "auto_regressive",
        "temperature":      0.1,
        "batch_size":       1,
        "number_of_batches": 1,
        "atomize_side_chains": False,
        "initialize_sequence_embedding_with_ground_truth": False,
        "features_to_return": None,
        "omit": [],
        "occupancy_threshold_sidechain": None,
        "occupancy_threshold_backbone":  None,
        "undesired_res_names": None,
    }

    if isinstance(structure, (str, Path)):
        inf = MPNNInferenceInput.from_atom_array_and_dict(
            input_dict={"structure_path": str(structure), **settings}
        )
    else:
        inf = MPNNInferenceInput.from_atom_array_and_dict(
            atom_array=structure, input_dict=settings
        )

    pipeline = _get_potts_pipeline(device, build_bond_labels=build_bond_labels, hbond_scope=hbond_scope,
                                   extended_vocab=extended_vocab, use_salt_bridge=use_salt_bridge,
                                   deterministic=deterministic, protonation_seed=protonation_seed)
    pipeline_out = pipeline({
        "atom_array":         inf.atom_array.copy(),
        "structure_noise":    inf.input_dict["structure_noise"],
        "decode_type":        inf.input_dict["decode_type"],
        "causality_pattern":  inf.input_dict["causality_pattern"],
        "initialize_sequence_embedding_with_ground_truth": False,
        "atomize_side_chains": False,
        "repeat_sample_num":  None,
        "features_to_return": None,
    })

    return {
        "network_input": FeatureCollator()([pipeline_out]),
        "atom_array":    pipeline_out["atom_array"],
    }


# ---------------------------------------------------------------------------
# Step 3 — run PottsMPNN forward
# ---------------------------------------------------------------------------

def run_forward(
    model: PottsMPNN,
    batch: dict,
    num_sequences: int = DEFAULTS["num_sequences"],
) -> PottsOutput:
    """Run the full PottsMPNN forward pass and return a PottsOutput."""
    ni = batch["network_input"]
    ni["input_features"]["repeat_sample_num"] = num_sequences

    with torch.no_grad():
        out = model(ni)

    return PottsOutput(
        S_sampled = out["decoder_features"]["S_sampled"],   # [N, L]
        S_argmax  = out["decoder_features"]["S_argmax"],    # [1, L]
        etab_out  = out["potts_context"].etab_out,          # [1, L, K, V, V]
        E_idx     = out["potts_context"].E_idx,             # [1, L, K]
        log_probs = out["decoder_features"].get("log_probs"),  # [*, L, V] decode-pass log-probs
    )


def compute_log_probs(
    model: PottsMPNN,
    batch: dict,
    *,
    causality: str = "conditional_minus_self",
    init_with_gt: bool = True,
    seq: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-position decoder log-probabilities under teacher forcing.

    Runs a single teacher-forced decode with the given ``causality`` pattern
    and returns ``log_probs`` of shape ``[L, V]`` (batch dim squeezed). The
    input batch is not mutated.

    With ``causality="conditional_minus_self"`` each position is scored on the
    rest of the (ground-truth) sequence and not its own identity, giving an
    order-independent per-position pseudo-likelihood — useful for asking where a
    particular token (e.g. a protonated microstate HIS-P/ASP-P/GLU-P) is
    favoured along the chain.

    Parameters
    ----------
    seq : torch.Tensor | None
        Optional sequence override (``[L]`` or ``[1, L]``). If given, the batch
        sequence ``S`` is replaced by it before scoring — e.g. to score the
        chain with a particular residue forced into a protonated/deprotonated
        state. The batch itself is left untouched.
    """
    import copy

    ni = copy.deepcopy(batch["network_input"])
    if seq is not None:
        s = seq if seq.dim() == 2 else seq.unsqueeze(0)
        ni["input_features"]["S"] = s.to(ni["input_features"]["S"].device)
    ni["input_features"].update({
        "decode_type": "teacher_forcing",
        "causality_pattern": causality,
        "initialize_sequence_embedding_with_ground_truth": init_with_gt,
        "repeat_sample_num": 1,
    })
    with torch.no_grad():
        out = model(ni)
    return out["decoder_features"]["log_probs"].squeeze(0)   # [L, V]


# ---------------------------------------------------------------------------
# Step 4 — Gibbs optimisation
# ---------------------------------------------------------------------------

def get_masks(
    model: PottsMPNN,
    batch: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(free_mask [L], valid_aa_mask [V])`` from a batch.

    ``free_mask``      — True for positions that Gibbs is allowed to mutate.
    ``valid_aa_mask``  — True for amino acid tokens that may be proposed
                         (excludes UNK and other special tokens).
    """
    ni = batch["network_input"]["input_features"]
    designed = ni.get("designed_residue_mask")
    if designed is None:
        # No fixed_chains / designed_chains / *_residues were given, so the featuriser left the
        # annotation off entirely (utils/inference.py::_annotate_design_scope returns early) and relies
        # on the model's implicit "design every residue". Mirror that here instead of raising an
        # AttributeError on None -- which is what the module's own __main__ demo below used to do.
        free_mask = torch.ones(int(ni["S"].shape[-1]), dtype=torch.bool,
                               device=next(model.parameters()).device)
    else:
        free_mask = designed.squeeze(0).bool()
    V = model.potts_vocab_size
    device = next(model.parameters()).device
    valid_aa_mask = torch.ones(V, dtype=torch.bool, device=device)
    for idx in model.unknown_token_indices:
        valid_aa_mask[idx] = False

    return free_mask, valid_aa_mask


def run_potts_optimize(
    potts_out: PottsOutput,
    free_mask: torch.Tensor,
    valid_aa_mask: torch.Tensor,
    *,
    temperature:      float = DEFAULTS["temperature"],
    max_iters:        int   = DEFAULTS["max_iters"],
    convergence_mode: bool  = DEFAULTS["convergence_mode"],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimise sequences with Potts Gibbs sweeps.

    Takes the output of ``run_forward`` and the masks from ``get_masks``,
    runs Gibbs independently on each sequence in ``potts_out.S_sampled``,
    and returns results sorted lowest energy first.

    Returns
    -------
    seq_opt  : [N, L]  optimised integer-encoded sequences.
    energies : [N]     Potts Hamiltonian H(s) for each sequence.
    """
    seq_opt, energies = PottsMPNN.potts_gibbs_optimize(
        etab_out=potts_out.etab_out,
        E_idx=potts_out.E_idx,
        seq_init=potts_out.S_sampled.clone(),
        free_mask=free_mask,
        temperature=temperature,
        max_iters=max_iters,
        convergence_mode=convergence_mode,
        valid_aa_mask=valid_aa_mask,
    )
    order = energies.argsort()
    return seq_opt[order], energies[order]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def decode_sequences(seq_int: torch.Tensor, model: PottsMPNN | None = None) -> list[str]:
    """Convert integer-encoded [N, L] tensor to one-letter amino-acid strings.

    Pass ``model`` whenever the sequence was built in an EXTENDED vocabulary: the token indices of the
    v6 (30-token) and v3/v4 (32-token) sets differ from the base 21-token map, so decoding a v6
    sequence against the default encoding silently renders the wrong residues, and every protonation
    microstate ("ASP-P", "HIS-S", …) renders as the unknown-AA letter because ``DICT_THREE_TO_ONE`` has
    no entry for it. With ``model`` the model's own encoding is used and each microstate is folded to
    its canonical parent first. ``model=None`` keeps the historical base-vocabulary behaviour.
    """
    from atomworks.constants import DICT_THREE_TO_ONE, UNKNOWN_AA
    idx_to_token = (
        model.graph_featurization_module.TOKEN_ENCODING.idx_to_token
        if model is not None else MPNN_TOKEN_ENCODING.idx_to_token
    )
    unk = DICT_THREE_TO_ONE[UNKNOWN_AA]
    result = []
    for row in seq_int.detach().cpu().numpy():
        three = [str(idx_to_token[int(i)]) for i in row]
        # "ASP-P"/"ASP-D"/"HIS-S"/… -> the canonical parent before the 1-letter lookup.
        parents = [t.split("-", 1)[0] if "-" in t else t for t in three]
        result.append("".join(DICT_THREE_TO_ONE.get(r, unk) for r in parents))
    return result


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    CHECKPOINT = "/path/to/potts_mpnn.ckpt"
    STRUCTURE  = "/path/to/protein.pdb"
    FIXED_CHAINS   = None   # e.g. ["B"]
    FIXED_RESIDUES = None   # e.g. ["A5", "A12"]

    model     = load_model(CHECKPOINT)
    batch     = prepare_potts_input(STRUCTURE, fixed_chains=FIXED_CHAINS,
                                    fixed_residues=FIXED_RESIDUES)
    potts_out = run_forward(model, batch, num_sequences=DEFAULTS["num_sequences"])
    free_mask, valid_aa_mask = get_masks(model, batch)

    seq_opt, energies = run_potts_optimize(potts_out, free_mask, valid_aa_mask)

    sequences = decode_sequences(seq_opt, model)
    print(f"\n{'Rank':>4}  {'Energy':>14}  Sequence")
    print("-" * 80)
    for rank, (seq, e) in enumerate(zip(sequences, energies.tolist()), 1):
        print(f"{rank:>4}  {e:>14.4f}  {seq}")
