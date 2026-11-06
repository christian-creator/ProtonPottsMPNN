# %% [markdown]
# Sequence-recovery benchmark on 300 validation structures (NOT training); writes val_recovery_all.csv for
# benchmark_barplot.py. Ported from ph/sandbox/ev6/val_recovery_benchmark.py; only paths and RESUME changed.
#
# All three predictors score the SAME residues from the SAME
# shared input_features (one v6 featurization per structure); only S is collapsed to each model's vocab.
# This makes the comparison provably apples-to-apples (verified: ProteinMPNN recovery on the shared v6
# features == on its own pipeline).
#
# Sequence recovery is reported under BOTH decode protocols (the two decoder-based models change; the Potts
# head is decode-order invariant so it is computed once):
#   auto_regressive        : each position decoded from previously-decoded true tokens (native ProteinMPNN).
#   conditional_minus_self : each position from the full true context of every OTHER residue (masked).
#   PottsMPNN MPNN head : argmax(decoder log_probs)         — both protocols
#   ProteinMPNN         : argmax(decoder log_probs)         — both protocols (S collapsed to 21-token)
#   PottsMPNN Potts head: argmax(log_softmax(-h_full)), h_full = h_i + Σ_k J_context — protocol-free (once)
# Scored over residue_mask:
#   exact     : predicted 30-token token == true (PottsMPNN heads only)
#   canonical : canonical_parent(pred) == canonical_parent(true)  — the 20-AA metric, common to all three.

# %%
import sys, random
import os
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
import contextlib, os
from tqdm import tqdm

os.environ.pop("DEBUG", None)                      # a stray non-boolean DEBUG breaks environs/foundry
_P = Path(os.environ.get("PROTON_ROOT", Path(__file__).resolve().parents[1]))   # mpnn/foundry from the venv
from atomworks.io.parser import STANDARD_PARSER_ARGS
from atomworks.ml.datasets.pandas_dataset import PandasDataset, StructuralDatasetWrapper
from atomworks.ml.datasets.parsers.default_metadata_row_parsers import GenericDFParser
from mpnn.pipelines.potts_mpnn import build_mpnn_transform_pipeline as build_potts_pipe
from mpnn.transforms.extended_vocab import get_vocab
from mpnn.model.pottsmpnn import PottsMPNN
from mpnn.model.layers.graph_embeddings import PottsProteinFeatures
from mpnn.metrics.sequence_recovery import PottsSequenceRecovery
from mpnn.inference_engines.mpnn import MPNNInferenceEngine

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXTENDED_VOCAB = os.environ.get("EV6_VOCAB", "v6")   # MUST match EV6_MODEL_DIR's checkpoint: it sizes the token encoding AND labels the features
# Same checkpoints and output folder as stability_benchmark.py, so benchmark_barplot.py reads all three together.
CKPT = Path(os.environ.get("POTTS_CKPT", _P / "checkpoints/potts_v6_afdb_edge_his0.3_acid0.06/epoch-0125.ckpt"))
PROTEINMPNN_CKPT = os.environ.get("PROTEINMPNN_CKPT", "/novo/projects/departments/rdd/cpjb/foundry/model_ckpts/proteinmpnn_v_48_020.pt")
# The validation split is not shipped (80 MB; its structure paths point at a PDB mirror). Same convention as
# labeller/01_hbond_dump.py's PROTON_TRAIN_DF.
VAL_PARQUET = Path(os.environ.get("PROTON_VAL_DF", _P / "data/mpnn_split/val_df_filtered.parquet"))
if not VAL_PARQUET.exists():
    sys.exit(f"validation split not found at {VAL_PARQUET}; set PROTON_VAL_DF=/path/to/val_df_filtered.parquet")
OUT_DIR = _P / "benchmarks/results" / os.environ.get("EV6_OUT_SUBDIR", "his0.3_acid0.06")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "val_recovery_all.csv"
N_SUBSAMPLE = 300
MIN_RES = 20
SEED = 0

# ── models ────────────────────────────────────────────────────────────────────────
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
potts = PottsMPNN(graph_featurization_module=PottsProteinFeatures(token_encoding=get_vocab(EXTENDED_VOCAB)["token_encoding"]),
                  etab_source=PottsMPNN.infer_etab_source(ck["model"]))
potts.apply(potts.init_weights); potts.load_state_dict(ck["model"]); potts.eval().to(DEVICE)
eng = MPNNInferenceEngine(model_type="protein_mpnn", checkpoint_path=PROTEINMPNN_CKPT, is_legacy_weights=True,
                          out_directory=None, write_fasta=False, write_structures=False, device=DEVICE)
pmpnn = eng.model.eval()
print(f"loaded potts {CKPT.name} (epoch {ck['current_epoch']}, {potts.potts_vocab_size} tok) + ProteinMPNN on {DEVICE}", flush=True)

# canonical-parent maps
enc = get_vocab(EXTENDED_VOCAB)["token_encoding"]
canon_map = PottsSequenceRecovery(token_encoding=enc)._get_canonical_map(torch.device(DEVICE))  # [V] v6 tok -> v6 canonical
v6_idx2tok = {v: k for k, v in enc.token_to_idx.items()}
_unk = pmpnn.token_to_idx.get("UNK", max(pmpnn.token_to_idx.values()))
v6_to_p = torch.full((len(enc.token_to_idx),), _unk, dtype=torch.long, device=DEVICE)  # v6 tok -> proteinmpnn 21-token
for _tok, _i in enc.token_to_idx.items():
    v6_to_p[_i] = pmpnn.token_to_idx.get(v6_idx2tok[int(canon_map[_i])], _unk)

# %%
val = pd.read_parquet(VAL_PARQUET)
rng = random.Random(SEED); idx = list(range(len(val))); rng.shuffle(idx)
sub = val.iloc[idx[:N_SUBSAMPLE]].reset_index(drop=True)
manifest = pd.DataFrame({"example_id": [str(e) for e in sub["example_id"]],
                         "path": [str(p) for p in sub["path"]],
                         "assembly_id": [str(a) for a in sub.get("assembly_id", pd.Series(["1"] * len(sub)))]})
print(f"val {len(val):,} rows -> subsample {len(manifest)} structures (seed {SEED})", flush=True)
pipe = build_potts_pipe(model_type="potts_mpnn", is_inference=True, minimal_return=False,
                        train_structure_noise_default=0.0, extended_vocab=EXTENDED_VOCAB)  # no noise: same coords both protocols
ds = StructuralDatasetWrapper(
    dataset=PandasDataset(data=manifest, id_column="example_id", name="val_rec"),
    dataset_parser=GenericDFParser(example_id_colname="example_id", path_colname="path", assembly_id_colname="assembly_id"),
    transform=pipe,
    cif_parser_args={**STANDARD_PARSER_ARGS, "add_bond_types_from_struct_conn": (),
                     "load_from_cache": False, "save_to_cache": False, "cache_dir": None})


@torch.no_grad()
def recovery(od):
    inp = {k: (v.unsqueeze(0).to(DEVICE) if torch.is_tensor(v) else v) for k, v in od["input_features"].items()}
    inp["decode_type"] = "teacher_forcing"
    S = inp["S"][0].clone()                                          # true v6 tokens [L]
    mask = inp["residue_mask"][0].bool() if "residue_mask" in inp else torch.ones_like(S, dtype=torch.bool)
    if mask.sum() < MIN_RES:
        return None
    m = mask; n = m.sum().item(); Sc = canon_map[S]                 # true canonical-parent AA
    res = {"n_res": int(n)}

    # --- PottsMPNN, auto_regressive: decoder head (AR) + Potts head (protocol-free, from etab) ---
    inp["S"] = S.unsqueeze(0); inp["causality_pattern"] = "auto_regressive"
    out = potts({"input_features": inp})
    pred = out["decoder_features"]["log_probs"][0].argmax(-1)
    res["decoder_exact_ar"] = ((pred == S) & m).sum().item() / n
    res["decoder_canonical_ar"] = ((canon_map[pred] == Sc) & m).sum().item() / n
    etab = out["potts_context"].etab_out; E_idx = out["potts_context"].E_idx; V = etab.shape[-1]
    h_i = torch.diagonal(etab[:, :, 0:1, :, :], 0, -2, -1).squeeze(2)
    S_neigh = S.unsqueeze(0)[torch.arange(1, device=S.device)[:, None, None], E_idx]
    s_idx = S_neigh[:, :, 1:].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, V, 1)
    J = etab[:, :, 1:].gather(-1, s_idx).squeeze(-1)
    pred_pot = F.log_softmax(-(h_i + J.sum(dim=2)), dim=-1)[0].argmax(-1)
    res["potts_exact"] = ((pred_pot == S) & m).sum().item() / n
    res["potts_canonical"] = ((canon_map[pred_pot] == Sc) & m).sum().item() / n

    # --- PottsMPNN, conditional_minus_self: decoder head (CMS) ---
    inp["S"] = S.unsqueeze(0); inp["causality_pattern"] = "conditional_minus_self"
    pred = potts({"input_features": inp})["decoder_features"]["log_probs"][0].argmax(-1)
    res["decoder_exact_cms"] = ((pred == S) & m).sum().item() / n
    res["decoder_canonical_cms"] = ((canon_map[pred] == Sc) & m).sum().item() / n

    # --- ProteinMPNN on the SAME features, S collapsed to 21-token, both protocols ---
    Sp = v6_to_p[S]
    for tag, caus in (("ar", "auto_regressive"), ("cms", "conditional_minus_self")):
        inp["S"] = Sp.unsqueeze(0); inp["causality_pattern"] = caus
        predp = pmpnn({"input_features": inp})["decoder_features"]["log_probs"][0].argmax(-1)
        res[f"proteinmpnn_canonical_{tag}"] = ((predp == Sp) & m).sum().item() / n   # native 20-AA == canonical
    return res


# %%
rows, done = [], set()   # every run recomputes from scratch (no rows are read back) and overwrites OUT_CSV
torch.manual_seed(SEED)
for i in tqdm(range(len(manifest)), desc="recovery"):
    eid = str(manifest.example_id.iloc[i])
    if eid in done:
        continue
    try:
        with open(os.devnull, "w") as _n, contextlib.redirect_stdout(_n), contextlib.redirect_stderr(_n):
            od = ds[i]
        r = recovery(od)
    except Exception as e:
        tqdm.write(f"  skip {eid[:36]}: {type(e).__name__}: {str(e)[:70]}"); continue
    if r is None:
        continue
    r["example_id"] = eid; rows.append(r); done.add(eid)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    if len(rows) % 25 == 0:
        d = pd.DataFrame(rows)
        tqdm.write(f"  n={len(rows)} | canonical AR: pmpnn={d.proteinmpnn_canonical_ar.mean():.3f} "
                   f"mpnn={d.decoder_canonical_ar.mean():.3f} | CMS: pmpnn={d.proteinmpnn_canonical_cms.mean():.3f} "
                   f"mpnn={d.decoder_canonical_cms.mean():.3f} | potts={d.potts_canonical.mean():.3f}")

D = pd.DataFrame(rows)
print(f"\nscored {len(D)} structures -> {OUT_CSV}", flush=True)
print("\n=== canonical (20-AA) recovery, mean over structures ===")
print("  auto_regressive:        "
      f"ProteinMPNN={D.proteinmpnn_canonical_ar.mean():.4f}  "
      f"MPNN-head={D.decoder_canonical_ar.mean():.4f}")
print("  conditional_minus_self: "
      f"ProteinMPNN={D.proteinmpnn_canonical_cms.mean():.4f}  "
      f"MPNN-head={D.decoder_canonical_cms.mean():.4f}")
print(f"  Potts head (protocol-free): {D.potts_canonical.mean():.4f}   "
      f"[exact 30-token: potts={D.potts_exact.mean():.4f}]")
