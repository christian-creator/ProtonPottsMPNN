#!/usr/bin/env python
"""
Standalone PKAD benchmark evaluation across PottsMPNN checkpoints.

Replicates PKADBenchmarkCallback outside the training loop for post-hoc
evaluation of any saved checkpoints. Data paths are hardcoded to match
train.py.

For each checkpoint:
  - Runs the model with teacher_forcing + conditional_minus_self (same as the
    callback) on every PKAD structure.
  - Computes log10(P_dep / P_prot) from the decoder head and the Potts
    single-site fields (diagonal of the self-edge energy table, k=0).
  - Combines: signal(α) = α·decoder + (1−α)·Potts, for α in --alphas.
  - Reports Pearson / Spearman vs experimental pKa for HIS, ASP, GLU, all.

Outputs (in --model_dir by default):
  pkad_per_residue.csv          per residue × per epoch raw signals
  pkad_summary.csv              per epoch × per alpha correlation metrics
  pkad_correlation_evolution.png  Pearson r vs epoch line plot
  pkad_scatter_best.png         scatter at the best-performing checkpoint

Usage:
    python -m mpnn.scripts.eval_pkad \
        --model_dir mpnn_output_potts_mpnn_ev2 \
        [--alphas 0.0 0.5 1.0] [--ckpt_stride 1] [--device cuda:0]
"""

import argparse
import contextlib
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from tqdm import tqdm

from atomworks.io.parser import STANDARD_PARSER_ARGS
from atomworks.ml.datasets.pandas_dataset import PandasDataset, StructuralDatasetWrapper
from atomworks.ml.datasets.parsers.default_metadata_row_parsers import GenericDFParser
from atomworks.ml.utils.token import token_iter
from mpnn.callbacks.pkad_benchmark import (
    _log10_ratio,
    _pearson,
    _r2,
    _spearman,
    _total_titratable_prob,
)
from mpnn.transforms.extended_vocab import get_vocab
from mpnn.model.layers.graph_embeddings import PottsProteinFeatures
from mpnn.model.pottsmpnn import PottsMPNN
from mpnn.pipelines.potts_mpnn import build_mpnn_transform_pipeline as build_potts_pipeline

# ── PKAD data paths ───────────────────────────────────────────────────────────
# Resolved, in order: $PROTON_ROOT/benchmarks/data/PKAD (the documented override), the
# ProtonPottsMPNN checkout this package is installed from (the CSV + the 164-entry pdb_cache SHIP
# there), then the original internal tree. Previously only the last of these existed, so the
# standalone script could not find its data from a clean checkout even though the data was in-repo
# -- and README's "benchmark scripts derive the package root from their location (override with
# PROTON_ROOT)" did not hold for this script. --pkad_csv / --pdb_dir still override everything.
_LEGACY_ROOT = Path("/novo/users/cpjb/PHD/conditional_binding/ph")


def _resolve_pkad_paths() -> tuple[Path, Path]:
    _csv_name = "PKAD-R-v1.0_2026-04-22T16_0755.441Z.csv"
    _candidates = []
    _env_root = os.environ.get("PROTON_ROOT", "").strip()
    if _env_root:
        _candidates.append(Path(_env_root) / "benchmarks/data/PKAD")
    # .../<root>/foundry/models/mpnn/src/mpnn/scripts/eval_pkad.py -> parents[6] == <root>
    try:
        _candidates.append(Path(__file__).resolve().parents[6] / "benchmarks/data/PKAD")
    except IndexError:
        pass
    _candidates.append(_LEGACY_ROOT / "data/PKAD")
    for _d in _candidates:
        if (_d / _csv_name).exists() and (_d / "pdb_cache").is_dir():
            return _d / _csv_name, _d / "pdb_cache"
    _d = _candidates[-1]
    return _d / _csv_name, _d / "pdb_cache"


_PKAD_CSV, _PKAD_PDB_DIR = _resolve_pkad_paths()

_TITRATABLE = {"HIS", "ASP", "GLU"}


def _make_token_idx_tensor(
    token_names: frozenset[str],
    token_to_idx: dict[str, int],
) -> torch.Tensor:
    return torch.tensor(
        [token_to_idx[t] for t in token_names if t in token_to_idx],
        dtype=torch.long,
    )


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path: Path, device: str, extended_vocab: str | None = None) -> tuple["PottsMPNN", int]:
    """Load a checkpoint, sizing the featuriser to the checkpoint's OWN vocabulary.

    This used to always build a default 32-token ``PottsProteinFeatures()``, so a 30-token v6
    checkpoint failed the strict ``load_state_dict`` on ``W_s``/``W_out``/``etab_out``. The vocabulary
    is taken from the checkpoint's recorded ``train_cfg.extended_vocab`` when present (the authoritative
    source), else from ``extended_vocab``; a mismatch between the two is an error rather than a silent
    out-of-distribution score, mirroring the guard in ``callbacks/pkad_benchmark.py``.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    try:
        ckpt_vocab = ckpt.get("train_cfg", {}).get("extended_vocab")
    except Exception:
        ckpt_vocab = None
    if ckpt_vocab and extended_vocab and str(ckpt_vocab) != str(extended_vocab):
        raise ValueError(
            f"{ckpt_path} was trained with extended_vocab={ckpt_vocab!r} but --extended_vocab="
            f"{extended_vocab!r} was requested. The encoder sees these tokens in S, so scoring across "
            "vocabularies is out of distribution; re-run with --extended_vocab "
            f"{ckpt_vocab!r}.")
    vocab_name = str(ckpt_vocab) if ckpt_vocab else extended_vocab
    if vocab_name:
        feats = PottsProteinFeatures(token_encoding=get_vocab(vocab_name)["token_encoding"])
    else:
        feats = PottsProteinFeatures()
    model = PottsMPNN(
        graph_featurization_module=feats,
        etab_source=PottsMPNN.infer_etab_source(ckpt["model"]),
    )
    model.apply(model.init_weights)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.extended_vocab_name = vocab_name
    return model.to(device), ckpt["current_epoch"]


def resolve_ckpt_dir(model_dir: Path) -> Path:
    """Resolve a training output directory to its checkpoint directory."""
    candidates = [
        model_dir / "ckpt",
        model_dir / "checkpoints",
        model_dir,
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("epoch-*.ckpt")):
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"No epoch-*.ckpt files found. Looked in: {searched}"
    )


# ── Dataset / featurization ───────────────────────────────────────────────────

def _build_position_map(out: dict) -> dict[tuple, int]:
    """Map (chain_id, res_id, res_name) → linear token index."""
    atom_array = out["atom_array"]
    non_atomized = atom_array[~atom_array.atomize]
    pos_map: dict[tuple, int] = {}
    for i, token in enumerate(token_iter(non_atomized)):
        key = (token.chain_id[0], int(token.res_id[0]), token.res_name[0])
        pos_map[key] = i
    return pos_map


def build_pkad_cache(
    pkad_csv: Path,
    pdb_dir: Path,
    extended_vocab: str = "v4",
) -> tuple[dict[str, pd.DataFrame], dict[str, tuple[dict, dict]]]:
    """Load and filter the PKAD CSV, featurize all structures, cache results.

    ``extended_vocab`` must match the checkpoints being scored: it decides which protonation labels the
    structures are featurized with, and S is teacher-forced into the model."""
    pka_df = pd.read_csv(pkad_csv)
    pka_df = pka_df[
        (pka_df["pKa Classification"] == "Main")
        & (pka_df["ResName"].isin(_TITRATABLE))
    ].copy()
    # Drop censored pKa values (stored as strings like "<2.5" or ">10.5");
    # their true value is unknown so they cannot be used in correlation metrics.
    pka_df["Expt. pKa"] = pd.to_numeric(pka_df["Expt. pKa"], errors="coerce")
    n_censored = pka_df["Expt. pKa"].isna().sum()
    if n_censored:
        print(f"[PKAD] dropping {n_censored} censored pKa rows (<X / >X)", flush=True)
    pka_df = pka_df.dropna(subset=["Expt. pKa"])
    pdb_groups = {pdb: grp for pdb, grp in pka_df.groupby("PDB")}

    pipeline = build_potts_pipeline(
        model_type="potts_mpnn",
        is_inference=True,
        minimal_return=False,  # need potts_context (etab_out)
        extended_vocab=extended_vocab,   # label + encode in the checkpoint's own vocabulary
    )

    pdb_ids = list(pdb_groups.keys())
    benchmark_df = pd.DataFrame({
        "example_id": pdb_ids,
        "path": [str(pdb_dir / f"{pid}.pdb") for pid in pdb_ids],
        "assembly_id": ["1"] * len(pdb_ids),
    })

    dataset = StructuralDatasetWrapper(
        dataset=PandasDataset(data=benchmark_df, id_column="example_id",
                              name="pkad_benchmark"),
        dataset_parser=GenericDFParser(
            example_id_colname="example_id",
            path_colname="path",
            assembly_id_colname="assembly_id",
        ),
        transform=pipeline,
        cif_parser_args={
            **STANDARD_PARSER_ARGS,
            "add_bond_types_from_struct_conn": (),
            "load_from_cache": False,
            "save_to_cache": False,
            "cache_dir": None,
        },
    )

    pdb_id_to_idx = {pid: i for i, pid in enumerate(pdb_ids)}
    cache: dict[str, tuple[dict, dict]] = {}
    n_fail = 0

    print(f"Featurizing {len(pdb_ids)} PKAD structures…", flush=True)
    for pid in tqdm(pdb_ids, desc="Featurizing", file=sys.stderr):
        idx = pdb_id_to_idx[pid]
        # try:
        with open(os.devnull, "w") as _null, \
                contextlib.redirect_stdout(_null), \
                contextlib.redirect_stderr(_null):
            out = dataset[idx]
        pos_map = _build_position_map(out)
        input_features = {
            k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
            for k, v in out["input_features"].items()
        }
        cache[pid] = (input_features, pos_map)
        # except Exception as e:
        #     tqdm.write(f"[warning] featurize failed for {pid}: {e}", file=sys.stderr)
        #     n_fail += 1

    print(f"Featurized {len(cache)}/{len(pdb_ids)} structures ({n_fail} failures)",
          flush=True)
    return pdb_groups, cache


# ── Per-checkpoint evaluation ─────────────────────────────────────────────────

def evaluate_checkpoint(
    model: "PottsMPNN",
    epoch: int,
    pdb_groups: dict[str, pd.DataFrame],
    cache: dict[str, tuple[dict, dict]],
    device: str,
    prot_idx: dict[str, torch.Tensor],
    deprot_idx: dict[str, torch.Tensor],
) -> list[dict]:
    records: list[dict] = []
    n_miss = n_fail = n_skip = 0

    with torch.no_grad():
        for pdb_id, df in tqdm(
            pdb_groups.items(), total=len(pdb_groups),
            desc=f"  epoch {epoch}", leave=False, file=sys.stderr,
        ):
            cached = cache.get(pdb_id)
            if cached is None:
                n_miss += 1
                continue

            raw_features, pos_map = cached

            rows: list[tuple[int, str, float]] = []
            for _, row in df.iterrows():
                key = (str(row["Chain"]), int(row["ResID in PDB"]), str(row["ResName"]))
                pos = pos_map.get(key)
                
                if pos is None:
                    n_skip += 1
                    continue
                rows.append((pos, str(row["ResName"]), float(row["Expt. pKa"])))

            if not rows:
                continue

            input_features = {
                k: v.clone().to(device) if isinstance(v, torch.Tensor) else v
                for k, v in raw_features.items()
            }
            input_features["decode_type"] = "teacher_forcing"
            input_features["causality_pattern"] = "conditional_minus_self"

            # try:
            out = model({"input_features": input_features})
            # except Exception as e:
            #     tqdm.write(
            #         f"[warning] forward failed for {pdb_id} epoch {epoch}: {e}",
            #         file=sys.stderr,
            #     )
            #     n_fail += 1
            #     continue

            log_probs_dec = out["decoder_features"]["log_probs"]   # [1, L, V]
            etab_out = out["potts_context"].etab_out               # [1, L, K, V, V]
            E_idx    = out["potts_context"].E_idx                  # [1, L, K]
            S        = input_features["S"]                         # [1, L]  token indices

            # Self-edge (k=0) was masked to diagonal-only in PottsMPNN.compute_potts_context,
            # so the diagonal is exactly h_i[v] = single-site field.
            B, L_seq, K = E_idx.shape
            V = etab_out.shape[-1]
            h_i = torch.diagonal(
                etab_out[:, :, 0:1, :, :], offset=0, dim1=-2, dim2=-1
            ).squeeze(2)                                           # [1, L, V]

            # Full Potts energy: add pairwise couplings at the current sequence context.
            # For each neighbour k=1..K-1 and each candidate token v, pick the coupling
            # energy column corresponding to the neighbour's actual current token.
            #   etab_out[:, :, k, v, S[neigh_k]]  →  summed over k
            S_neigh = S[torch.arange(B, device=S.device)[:, None, None], E_idx]
            # [1, L, K];  k=0 is self-edge, k≥1 are structural neighbours
            s_idx = (
                S_neigh[:, :, 1:]                    # [B, L, K-1]
                .unsqueeze(-1)                        # [B, L, K-1, 1]
                .unsqueeze(-1)                        # [B, L, K-1, 1, 1]
                .expand(-1, -1, -1, V, 1)            # [B, L, K-1, V, 1]
            )
            J_context = etab_out[:, :, 1:].gather(-1, s_idx).squeeze(-1)  # [1, L, K-1, V]
            h_full = h_i + J_context.sum(dim=2)                  # [1, L, V]

            log_probs_pot = F.log_softmax(-h_full, dim=-1)        # Boltzmann over full energy

            for pos, res_name, expt_pka in rows:
                dep_idx = deprot_idx[res_name].to(device)
                pro_idx = prot_idx[res_name].to(device)
                lp_dec = log_probs_dec[0, pos, :]
                lp_pot = log_probs_pot[0, pos, :]
                records.append({
                    "epoch": epoch,
                    "pdb": pdb_id,
                    "res_name": res_name,
                    "expt_pka": expt_pka,
                    "log_ratio_dec": _log10_ratio(lp_dec, dep_idx, pro_idx),
                    "log_ratio_pot": _log10_ratio(lp_pot, dep_idx, pro_idx),
                    # confidence = total softmax mass on titratable tokens
                    "confidence_dec": _total_titratable_prob(lp_dec, dep_idx, pro_idx),
                    "confidence_pot": _total_titratable_prob(lp_pot, dep_idx, pro_idx),
                })

    print(
        f"    epoch {epoch}: records={len(records)}"
        f"  miss={n_miss}  fail={n_fail}  skip={n_skip}",
        flush=True,
    )
    return records


# ── Aggregate metrics ─────────────────────────────────────────────────────────

def compute_summary_rows(
    epoch_records: list[dict],
    epoch: int,
    alphas: list[float],
) -> list[dict]:
    df = pd.DataFrame(epoch_records)
    summary_rows = []
    for alpha in alphas:
        df["signal"] = alpha * df["log_ratio_dec"] + (1.0 - alpha) * df["log_ratio_pot"]
        row: dict = {"epoch": epoch, "alpha": alpha, "n": len(df)}
        for res in ("HIS", "ASP", "GLU"):
            mask = df["res_name"] == res
            sig_r = df.loc[mask, "signal"].tolist()
            pka_r = df.loc[mask, "expt_pka"].tolist()
            row[f"pearson_{res}"]  = _pearson(sig_r, pka_r)
            row[f"spearman_{res}"] = _spearman(sig_r, pka_r)
            row[f"r2_{res}"]       = _r2(sig_r, pka_r)
        sig_all = df["signal"].tolist()
        pka_all = df["expt_pka"].tolist()
        row["pearson_all"]  = _pearson(sig_all, pka_all)
        row["spearman_all"] = _spearman(sig_all, pka_all)
        row["r2_all"]       = _r2(sig_all, pka_all)
        summary_rows.append(row)
        print(
            f"    alpha={alpha:.1f}  pearson_all={row['pearson_all']:.4f}"
            f"  HIS={row['pearson_HIS']:.4f}"
            f"  ASP={row['pearson_ASP']:.4f}"
            f"  GLU={row['pearson_GLU']:.4f}",
            flush=True,
        )
    return summary_rows


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_correlation_evolution(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """One PNG per alpha; each shows Pearson r for HIS / ASP / GLU / all vs epoch."""
    alphas = sorted(summary_df["alpha"].unique())
    res_colors = {"HIS": "#6495ED", "ASP": "#DC143C", "GLU": "#FF4444", "all": "#333333"}

    for alpha in alphas:
        sub = summary_df[summary_df["alpha"] == alpha].sort_values("epoch")
        fig, ax = plt.subplots(figsize=(7, 4))
        for res, color in res_colors.items():
            ax.plot(sub["epoch"], sub[f"pearson_{res}"],
                    color=color, linewidth=1.5, marker="o", markersize=4, label=res)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.set_title(f"PKAD Pearson r  (α={alpha:.1f})", fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Pearson r")
        ax.legend(fontsize=9, title="Residue")
        fig.tight_layout()
        fname = f"pkad_correlation_alpha_{alpha:.1f}.png".replace("-", "neg")
        fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_scatter_per_alpha(
    per_residue_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
    alphas: list[float],
) -> None:
    """One scatter PNG per alpha at the last epoch; HIS / ASP / GLU panels only."""
    last_epoch = int(summary_df["epoch"].max())
    sub_epoch = per_residue_df[per_residue_df["epoch"] == last_epoch].copy()
    res_colors = {"HIS": "#6495ED", "ASP": "#DC143C", "GLU": "#FF4444"}
    res_list = [r for r in ("HIS", "ASP", "GLU") if r in sub_epoch["res_name"].values]

    for alpha in alphas:
        sub = sub_epoch.copy()
        sub["signal"] = alpha * sub["log_ratio_dec"] + (1.0 - alpha) * sub["log_ratio_pot"]

        fig, axes = plt.subplots(1, len(res_list), figsize=(4.5 * len(res_list), 4))
        if len(res_list) == 1:
            axes = [axes]

        for ax, res in zip(axes, res_list):
            df_res = sub[sub["res_name"] == res]
            r   = _pearson(df_res["signal"].tolist(), df_res["expt_pka"].tolist())
            r2  = _r2(df_res["signal"].tolist(), df_res["expt_pka"].tolist())
            rho = _spearman(df_res["signal"].tolist(), df_res["expt_pka"].tolist())
            conf = (
                alpha * df_res["confidence_dec"]
                + (1.0 - alpha) * df_res["confidence_pot"]
            )
            max_c = float(conf.max()) if len(conf) > 0 and float(conf.max()) > 0 else 1.0
            pt_sizes = 15 + 85 * (conf / max_c)
            ax.scatter(
                df_res["expt_pka"], df_res["signal"],
                c=res_colors[res], alpha=0.75, s=pt_sizes,
                edgecolors="white", linewidths=0.3,
            )
            ax.set_title(
                f"{res}  r={r:.3f}  ρ={rho:.3f}  R²={r2:.3f}  n={len(df_res)}",
                fontsize=9,
            )
            ax.set_xlabel("Expt. pKa")
            ax.set_ylabel("Predicted signal")

        # Size legend: three representative confidence levels
        size_handles = [
            mlines.Line2D([], [], linestyle="None", marker="o", color="gray",
                          markersize=np.sqrt(15 + 85 * cv), alpha=0.75,
                          label=f"{cv:.1f}")
            for cv in (0.1, 0.5, 1.0)
        ]
        fig.legend(
            handles=size_handles, title="Confidence", fontsize=8,
            title_fontsize=8, loc="lower right",
            bbox_to_anchor=(1.0, 0.0), frameon=True, framealpha=0.9,
        )
        fig.suptitle(
            f"PKAD scatter — last epoch ({last_epoch}), α={alpha:.1f}  |  size ∝ confidence",
            fontsize=10,
        )
        fig.tight_layout()
        fname = f"pkad_scatter_alpha_{alpha:.1f}.png".replace("-", "neg")
        fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_confidence_evolution(per_residue_df: pd.DataFrame, output_dir: Path) -> None:
    """Mean P_total per epoch for each residue type; separate panels for decoder and Potts."""
    res_colors = {"HIS": "#6495ED", "ASP": "#DC143C", "GLU": "#FF4444", "all": "#333333"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

    for ax, col, head in [
        (axes[0], "confidence_dec", "Decoder"),
        (axes[1], "confidence_pot", "Potts"),
    ]:
        all_mean = per_residue_df.groupby("epoch")[col].mean()
        ax.plot(all_mean.index, all_mean.values, color=res_colors["all"],
                linewidth=1.5, marker="o", markersize=3, label="all")
        for res in ("HIS", "ASP", "GLU"):
            df_r = per_residue_df[per_residue_df["res_name"] == res]
            if df_r.empty:
                continue
            res_mean = df_r.groupby("epoch")[col].mean()
            ax.plot(res_mean.index, res_mean.values, color=res_colors[res],
                    linewidth=1.5, marker="o", markersize=3, label=res)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean P(titratable tokens)")
        ax.set_title(f"{head} head")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9, title="Residue")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

    fig.suptitle("Confidence (P_prot + P_deprot) over training", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / "pkad_confidence_evolution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confidence_distribution(per_residue_df: pd.DataFrame, output_dir: Path) -> None:
    """KDE of confidence values at the last epoch, per residue type and model head."""
    last_epoch = int(per_residue_df["epoch"].max())
    sub = per_residue_df[per_residue_df["epoch"] == last_epoch]
    res_colors = {"HIS": "#6495ED", "ASP": "#DC143C", "GLU": "#FF4444"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, col, head in [
        (axes[0], "confidence_dec", "Decoder"),
        (axes[1], "confidence_pot", "Potts"),
    ]:
        for res, color in res_colors.items():
            data = sub[sub["res_name"] == res][col].dropna()
            if len(data) < 3:
                continue
            sns.kdeplot(
                data, ax=ax, color=color, linewidth=1.5,
                fill=True, alpha=0.15,
                label=f"{res} (n={len(data)})", warn_singular=False,
            )
            ax.axvline(float(data.mean()), color=color, linewidth=0.8,
                       linestyle="--", alpha=0.7)
        ax.set_xlabel("Confidence (P_prot + P_deprot)")
        ax.set_ylabel("Density")
        ax.set_title(f"{head} head — epoch {last_epoch}")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)

    fig.suptitle("Confidence distribution at last epoch", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / "pkad_confidence_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_vs_confidence(
    per_residue_df: pd.DataFrame,
    output_dir: Path,
    alphas: list[float],
) -> None:
    """Pearson r and Spearman ρ vs confidence threshold at last epoch, one PNG per alpha."""
    last_epoch = int(per_residue_df["epoch"].max())
    sub_base = per_residue_df[per_residue_df["epoch"] == last_epoch].copy()
    res_colors = {"HIS": "#6495ED", "ASP": "#DC143C", "GLU": "#FF4444"}
    thresholds = np.linspace(0.0, 0.8, 17)

    for alpha in alphas:
        sub = sub_base.copy()
        sub["signal"] = (
            alpha * sub["log_ratio_dec"] + (1.0 - alpha) * sub["log_ratio_pot"]
        )
        sub["confidence"] = (
            alpha * sub["confidence_dec"] + (1.0 - alpha) * sub["confidence_pot"]
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # n remaining on twin axis of the right panel
        ax_n = ax2.twinx()
        ns_all = [int((sub["confidence"] >= t).sum()) for t in thresholds]
        ax_n.plot(thresholds, ns_all, color="#999999", linewidth=1.0,
                  linestyle="--", alpha=0.6, zorder=1)
        ax_n.set_ylabel("n remaining (all)", color="#999999", fontsize=9)
        ax_n.tick_params(axis="y", labelcolor="#999999", labelsize=8)
        ax_n.set_ylim(bottom=0)

        for ax, metric_fn, metric_name in [
            (ax1, _pearson,  "Pearson r"),
            (ax2, _spearman, "Spearman ρ"),
        ]:
            for res, color in res_colors.items():
                ys = []
                for t in thresholds:
                    filt = sub[
                        (sub["res_name"] == res) & (sub["confidence"] >= t)
                    ]
                    ys.append(
                        metric_fn(filt["signal"].tolist(), filt["expt_pka"].tolist())
                    )
                ax.plot(thresholds, ys, color=color, linewidth=1.5,
                        marker="o", markersize=3, label=res, zorder=3)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle=":", zorder=2)
            ax.set_xlabel("Min confidence threshold  (P_total ≥ x)")
            ax.set_ylabel(metric_name)
            ax.set_title(f"{metric_name} vs confidence threshold")
            ax.legend(fontsize=9, title="Residue")

        fig.suptitle(
            f"Correlation vs confidence — last epoch ({last_epoch}), α={alpha:.1f}",
            fontsize=11,
        )
        fig.tight_layout()
        fname = f"pkad_corr_vs_confidence_alpha_{alpha:.1f}.png".replace("-", "neg")
        fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone PKAD benchmark evaluation")
    parser.add_argument("--model_dir", type=Path, required=False,
                        help="Training output directory; checkpoints are found under model_dir/ckpt")
    parser.add_argument("--ckpt_dir", type=Path, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--output_dir", type=Path, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.5, 1.0],
                        help="Decoder/Potts mixing weights (default: 0.0 0.5 1.0)")
    parser.add_argument("--ckpt_stride", type=int, default=1,
                        help="Evaluate every N-th checkpoint (default: 1 = all)")
    parser.add_argument("--device",
                        default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Torch device (default: cuda:0 if available)")
    # Allow overriding the hardcoded paths for flexibility
    parser.add_argument("--pkad_csv", type=Path, default=_PKAD_CSV,
                        help=f"PKAD CSV (default: {_PKAD_CSV})")
    parser.add_argument("--pdb_dir", type=Path, default=_PKAD_PDB_DIR,
                        help=f"PKAD PDB directory (default: {_PKAD_PDB_DIR})")
    parser.add_argument("--extended_vocab", default="v4",
                        help="Vocabulary the checkpoints were trained on ('v3'/'v4'/'v6'). MUST match: "
                             "it sets both the labels the structures are featurized with and which "
                             "tokens count as protonated / deprotonated (default: v4)")
    args = parser.parse_args()

    if args.model_dir is None and args.ckpt_dir is None:
        parser.error("--model_dir is required")

    model_dir = args.model_dir
    if model_dir is None:
        model_dir = args.ckpt_dir.parent if args.ckpt_dir.name == "ckpt" else args.ckpt_dir

    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else resolve_ckpt_dir(model_dir)
    output_dir = args.output_dir or (model_dir / "eval_pkad")

    all_ckpts = sorted(ckpt_dir.glob("epoch-*.ckpt"))
    if not all_ckpts:
        sys.exit(f"No checkpoints found in {ckpt_dir}")
    selected = all_ckpts[:: args.ckpt_stride]
    print(f"Model directory:      {model_dir}", flush=True)
    print(f"Checkpoint directory: {ckpt_dir}", flush=True)
    print(f"Output directory:     {output_dir}", flush=True)
    print(
        f"Evaluating {len(selected)}/{len(all_ckpts)} checkpoint(s) "
        f"(stride={args.ckpt_stride})",
        flush=True,
    )

    pdb_groups, cache = build_pkad_cache(args.pkad_csv, args.pdb_dir, args.extended_vocab)

    _probe, _ = load_model(selected[0], "cpu", args.extended_vocab)
    token_to_idx = _probe.token_to_idx
    # The token groups come from the checkpoint's vocabulary, never hardcoded: v3/v4 call neutral His
    # HID/HIE, v6 calls it HIS-S, and naming the wrong one silently drops that probability mass.
    _vocab = get_vocab(args.extended_vocab)
    prot_idx = {
        res: _make_token_idx_tensor(frozenset(_vocab["aa_protonated"][res]), token_to_idx)
        for res in _TITRATABLE
    }
    deprot_idx = {
        res: _make_token_idx_tensor(frozenset(_vocab["aa_deprotonated"][res]), token_to_idx)
        for res in _TITRATABLE
    }
    del _probe

    all_per_residue: list[dict] = []
    all_summary: list[dict] = []

    # ── Random-init baseline (epoch = 0) ─────────────────────────────────────
    # epoch 0 = untrained network; trained checkpoints are shifted to epoch N+1
    # so the x-axis reads "epochs of training completed".
    print("Evaluating random-init baseline (epoch = 0)…", flush=True)
    random_model = PottsMPNN(graph_featurization_module=PottsProteinFeatures())
    random_model.apply(random_model.init_weights)
    random_model.eval()
    random_model = random_model.to(args.device)
    baseline_records = evaluate_checkpoint(
        random_model, 0, pdb_groups, cache, args.device, prot_idx, deprot_idx
    )
    if baseline_records:
        df_rec = pd.DataFrame(baseline_records)
        for alpha in args.alphas:
            df_rec[f"signal_alpha_{alpha:.1f}"] = (
                alpha * df_rec["log_ratio_dec"]
                + (1.0 - alpha) * df_rec["log_ratio_pot"]
            )
        all_per_residue.extend(df_rec.to_dict("records"))
        all_summary.extend(compute_summary_rows(baseline_records, 0, args.alphas))
    del random_model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    for ckpt_path in tqdm(selected, desc="Checkpoints", file=sys.stderr):
        model, ckpt_epoch = load_model(ckpt_path, args.device, args.extended_vocab)
        epoch = ckpt_epoch + 1  # shift so epoch 0 is reserved for the random baseline
        print(f"  → {ckpt_path.name}  (ckpt_epoch={ckpt_epoch} → plot_epoch={epoch})", flush=True)

        records = evaluate_checkpoint(
            model, epoch, pdb_groups, cache, args.device, prot_idx, deprot_idx
        )

        if records:
            df_rec = pd.DataFrame(records)
            for alpha in args.alphas:
                col = f"signal_alpha_{alpha:.1f}"
                df_rec[col] = (
                    alpha * df_rec["log_ratio_dec"]
                    + (1.0 - alpha) * df_rec["log_ratio_pot"]
                )
            all_per_residue.extend(df_rec.to_dict("records"))
            all_summary.extend(compute_summary_rows(records, epoch, args.alphas))

        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    if not all_per_residue:
        sys.exit("No records collected — check PKAD CSV / PDB directory / pipeline.")

    output_dir.mkdir(parents=True, exist_ok=True)

    per_residue_df = pd.DataFrame(all_per_residue)
    per_residue_path = output_dir / "pkad_per_residue.csv"
    per_residue_df.to_csv(per_residue_path, index=False)
    print(f"Saved per-residue records → {per_residue_path}", flush=True)

    summary_df = pd.DataFrame(all_summary)
    summary_path = output_dir / "pkad_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary metrics → {summary_path}", flush=True)

    print("Generating plots…", flush=True)
    plot_correlation_evolution(summary_df, output_dir)
    plot_scatter_per_alpha(per_residue_df, summary_df, output_dir, args.alphas)
    plot_confidence_evolution(per_residue_df, output_dir)
    plot_confidence_distribution(per_residue_df, output_dir)
    plot_correlation_vs_confidence(per_residue_df, output_dir, args.alphas)
    print(f"All outputs saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
