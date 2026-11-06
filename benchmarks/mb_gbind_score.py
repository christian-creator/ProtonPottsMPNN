"""Benchmark scorer: v6 Potts G_bind on the ON-TARGET intended folds of the MB design sources, for a
binder-vs-non-binder benchmark. Multi-source, shardable (SLURM), resumable.

Sources (intended-target folds only; the *_RF3_output off-target specificity panels are NOT used):
  - metaanalysis : meta_AF3_outputs/<design>/<name>_model.cif   (AF3, binder = chain A)
  - boltzgen     : boltz2 best_model_path_n10 .cif               (boltz2, binder = chain A)
  - bindcraft    : structures TBD (not in MB_benchmark/data) — plugs in once BINDCRAFT_DIR is set.

  # one shard:
  SHARD_ID=0 NUM_SHARDS=8 PYTHONPATH=. HBPLUS_PATH=/path/to/hbplus \
      ./dpo_potts/bin/python sandbox/gbind/mb_gbind_score.py
  # smoke a few of one source:
  ... sandbox/gbind/mb_gbind_score.py --source metaanalysis --shard 0 --num-shards 1 --limit 5
  # aggregate shards + join labels/iptm → benchmark parquet:
  ./dpo_potts/bin/python sandbox/gbind/mb_gbind_score.py --aggregate
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path("/novo/users/cpjb/PHD/diffusion_DPO_gefion")
sys.path.insert(0, str(PROJECT_ROOT))

MB = Path("/novo/users/cpjb/rdd/cpjb/MB_benchmark/data")
SPEC = Path("/novo/projects/departments/rdd/cpjb/tmp/MB_specificity")
# MB_MODEL picks the scoring checkpoint. Checkpoint and vocabulary are ONE unit: the vocab sizes the
# token encoding AND labels the structures that go into S, and the ev3 checkpoint carries no
# train_cfg.extended_vocab to fall back on, so it is named explicitly here. Shards and the aggregate
# parquet are tagged with the model — without that an ev3 run overwrites the v6 benchmark table.
import os
_PH = "/novo/users/cpjb/PHD/conditional_binding/ph/trained_models"
MODELS = {
    "v6":  (f"{_PH}/mpnn_output_potts_v6_afdb_edge_his0.3_acid0.06/ckpt/epoch-0125.ckpt", "v6"),
    "ev3": (f"{_PH}/mpnn_output_potts_mpnn_SB_0_2_lcomplex_fixed_ev3/ckpt/epoch-0043.ckpt", "v3"),
}
MB_MODEL = os.environ.get("MB_MODEL", "v6")
POTTS_CKPT, POTTS_VOCAB = MODELS[MB_MODEL]
_MTAG = "" if MB_MODEL == "v6" else f"_{MB_MODEL}"     # v6 keeps its existing paths
OUT_DIR = PROJECT_ROOT / "data/crossbinding"
OUT_PARQUET = OUT_DIR / f"mb_gbind_benchmark{_MTAG}.parquet"
SHARD_DIR = OUT_DIR / f"mb_gbind_shards{_MTAG}"


# ── per-source enumeration → [(binder_id, structure_path)] ─────────────────────────────────────
# metaanalysis + bindcraft are scored on their TEMPLATED Boltz-2 structures (same oracle as boltzgen),
# indexed by sandbox/gbind/build_fold_index.py; boltzgen keeps its own Boltz-2 structures.
def _enum_from_index(source):
    df = pd.read_csv(OUT_DIR / f"boltz_tpl_index_{source}.csv")
    return [(str(r.binder_id), str(r.structure_path)) for r in df.itertuples()]


def enum_metaanalysis():
    return _enum_from_index("metaanalysis")


def enum_boltzgen():
    m = pd.read_csv(MB / "boltzgen/boltz2_outputs/boltz2_metrics_combined.csv")
    out = []
    for _, r in m.iterrows():
        p = next((str(r[c]) for c in ("best_model_path_n10", "best_model_path_n6", "best_model_path_n3")
                  if c in m.columns and isinstance(r.get(c), str) and Path(str(r[c])).exists()), None)
        if p:
            out.append((str(r["binder_id"]), p))
    return out


def enum_bindcraft():
    return _enum_from_index("bindcraft")


# label/iptm/target join config per source (applied in aggregate).
# baselines = other predicted binder/non-binder scores to compare G_bind against; each is
# {parquet_name: (source_column, sign)} with sign=+1 (higher=better) / -1 (lower=better, e.g. an
# interface ΔG → stored as −ΔG so higher always = more likely a binder, like −G_bind).
SOURCES = {
    "metaanalysis": dict(binder_chain="A", enumerate=enum_metaanalysis,   # templated Boltz-2: binder = chain A
                         label_csv=MB / "metaanalysis_paper/final_dataset.csv",
                         key="binder_id", label_col="binder", target_col="target_id",
                         iptm_csv=OUT_DIR / "boltz_tpl_index_metaanalysis.csv",   # templated Boltz-2 ipTM
                         iptm_key="binder_id", iptm_col="iptm",
                         baselines={
                             "boltz1_iptm": ("boltz1_iptm_model_0", +1),
                             "colab_iptm": ("colab_iptm_model_0", +1),
                             "colab_actifptm": ("colab_actifptm_model_0", +1),
                             "af3_dockQ": ("af3_dockQ", +1),
                             "af2_plddt_binder": ("af2_plddt_binder", +1),
                             "neg_af3_rosetta_dG": ("af3_rosetta_interface_dG", -1),
                             "neg_boltz1_rosetta_dG": ("boltz1_rosetta_interface_dG", -1),
                             "neg_af2_rosetta_dG": ("af2_rosetta_interface_dG", -1),
                             "neg_input_rosetta_dG": ("input_rosetta_interface_dG", -1),
                         }),
    "boltzgen": dict(binder_chain="A", enumerate=enum_boltzgen,
                     label_csv=MB / "boltzgen/boltzgen_data.csv",
                     key="binder_id", label_col="binding", target_col="target_id",
                     iptm_csv=MB / "boltzgen/boltz2_outputs/boltz2_metrics_combined.csv",
                     iptm_key="binder_id", iptm_col="iptm_n10",
                     meta_cols=["binder_type"],   # scaffold class: nano | prot (confounds MPNN logP)
                     baselines={
                         "boltz1_iptm": ("boltz_iptm_model_0", +1),
                         "af3_iptm": ("af3_iptm_model_0", +1),
                         "boltz1_plddt": ("boltz_complex_plddt_model_0", +1),
                         "af3_ipSAE_max": ("af3_ipSAE_max", +1),
                         "boltz1_ipSAE_max": ("boltz1_ipSAE_max", +1),
                         "af3_LIS": ("af3_LIS", +1),
                     }),
    "bindcraft": dict(binder_chain="B", enumerate=enum_bindcraft,   # templated Boltz-2: binder = chain B
                      label_csv=MB / "bindcraft/bindcraft_summary.csv",
                      key="DesignName", label_col="Binding", target_col="Target",
                      iptm_csv=OUT_DIR / "boltz_tpl_index_bindcraft.csv",   # templated Boltz-2 ipTM
                      iptm_key="binder_id", iptm_col="iptm",
                      baselines={}),
}


def score_shard(source, shard_id, num_shards, limit=None, metric="gbind"):
    """metric='gbind'  → v6 Potts G_bind (3-system thermodynamic cycle)
       metric='mpnn_ll' → MPNN decoder log-likelihood of the binder sequence (second benchmark score)."""
    import torch
    from src.foundry_wrappers.pottsmpnn_ph import PottsMPNNPHEngine
    cfg = SOURCES[source]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    eng = PottsMPNNPHEngine(checkpoint_path=POTTS_CKPT, extended_vocab=POTTS_VOCAB, device=dev,
                            write_fasta=False, write_structures=False)
    comps = cfg["enumerate"]()
    mine = comps[shard_id::num_shards]
    if limit:
        mine = mine[:limit]
    print(f"[mbg] {source} [{metric}] shard {shard_id}/{num_shards}: {len(mine)}/{len(comps)} on {dev} "
          f"(binder_chain={cfg['binder_chain']})", flush=True)

    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    tag = "" if metric == "gbind" else "_mpnnll"
    out_path = SHARD_DIR / f"{source}{tag}_shard_{shard_id:03d}_of_{num_shards:03d}.parquet"
    rows, done = [], set()
    if out_path.exists():
        prev = pd.read_parquet(out_path)
        rows, done = prev.to_dict("records"), set(prev["binder_id"])

    for i, (binder_id, path) in enumerate(mine, 1):
        if binder_id in done:
            continue
        try:
            if metric == "gbind":
                g = eng.compute_g_bind(path, binder_chain=cfg["binder_chain"])
                row = dict(g_bind=g["g_bind"], neg_g_bind=-g["g_bind"], E_complex=g["E_complex"],
                           E_binder=g["E_binder"], E_target=g["E_target"],
                           n_binder=g["n_binder"], n_target=g["n_target"])
                note = f"g_bind={g['g_bind']:.1f}"
            else:
                m = eng.compute_mpnn_loglik(path, binder_chain=cfg["binder_chain"])
                row = dict(mpnn_ll_mean=m["mpnn_ll_mean"], mpnn_ll_sum=m["mpnn_ll_sum"],
                           n_binder=m["n_binder"])
                note = f"mpnn_ll={m['mpnn_ll_mean']:.3f}"
        except Exception as e:
            print(f"[mbg] SKIP {source}:{binder_id}: {type(e).__name__}: {e}", flush=True)
            continue
        rows.append(dict(source=source, binder_id=binder_id, structure_path=path, **row))
        if i % 25 == 0:
            pd.DataFrame(rows).to_parquet(out_path, index=False)
            print(f"[mbg] {i}/{len(mine)}  {binder_id}  {note}", flush=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"[mbg] {source} [{metric}] shard done → {out_path} ({len(rows)} rows)", flush=True)


_LABEL_MAP = {True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0, 1.0: 1, 0.0: 0}


def aggregate():
    shards = [s for s in sorted(SHARD_DIR.glob("*_shard_*.parquet")) if "_mpnnll_" not in s.name]
    if not shards:
        raise SystemExit(f"no G_bind shard parquets under {SHARD_DIR}")
    df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True).drop_duplicates(["source", "binder_id"])
    # second benchmark score: MPNN decoder log-likelihood (its own shards) → merged as extra columns
    ml_shards = sorted(SHARD_DIR.glob("*_mpnnll_shard_*.parquet"))
    if ml_shards:
        ml = (pd.concat([pd.read_parquet(s) for s in ml_shards], ignore_index=True)
              .drop_duplicates(["source", "binder_id"])[["source", "binder_id", "mpnn_ll_mean", "mpnn_ll_sum"]])
        df = df.merge(ml, on=["source", "binder_id"], how="left")
        print(f"[mbg] merged MPNN log-lik for {df['mpnn_ll_mean'].notna().sum()}/{len(df)} complexes")
    parts = []
    for source, sub in df.groupby("source"):
        cfg = SOURCES[source]
        lcols = [cfg["key"], cfg["label_col"], cfg["target_col"]] + list(cfg.get("meta_cols", []))
        ren = {cfg["key"]: "binder_id", cfg["label_col"]: "label", cfg["target_col"]: "target_id"}
        if "iptm_csv" not in cfg:                                 # iptm lives in the same label table
            lcols.append(cfg["iptm_col"]); ren[cfg["iptm_col"]] = "iptm"
        lab = pd.read_csv(cfg["label_csv"])[lcols].drop_duplicates(cfg["key"]).rename(columns=ren)
        sub = sub.merge(lab, on="binder_id", how="left")
        if "iptm_csv" in cfg:                                     # iptm from the scored-structure's metrics
            it = (pd.read_csv(cfg["iptm_csv"])[[cfg["iptm_key"], cfg["iptm_col"]]]
                  .drop_duplicates(cfg["iptm_key"]).rename(columns={cfg["iptm_key"]: "binder_id", cfg["iptm_col"]: "iptm"}))
            sub = sub.merge(it, on="binder_id", how="left")
        # join the other predicted baseline scores (signed so higher = more likely a binder)
        bl = cfg.get("baselines", {})
        if bl:
            raw = pd.read_csv(cfg["label_csv"]).drop_duplicates(cfg["key"]).set_index(cfg["key"])
            for name, (col, sign) in bl.items():
                if col in raw.columns:
                    sub[name] = sign * pd.to_numeric(raw[col].reindex(sub["binder_id"]).values, errors="coerce")
        parts.append(sub)
    out = pd.concat(parts, ignore_index=True)
    out["y"] = out["label"].map(_LABEL_MAP)
    out["neg_g_bind_per_binder_res"] = out["neg_g_bind"] / out["n_binder"]
    # PottsEnergy of the complex (size-normalized): total E_complex is negative (favourable), so -E_complex
    # rises with stability; per-residue divides by complex length to make it target/size-comparable.
    out["neg_E_complex"] = -out["E_complex"]
    out["neg_E_complex_per_res"] = -out["E_complex"] / (out["n_binder"] + out["n_target"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PARQUET, index=False)
    lab = out.dropna(subset=["y"])
    print(f"[mbg] aggregated {len(out)} rows → {OUT_PARQUET}")
    print(f"[mbg] labelled: {len(lab)} ({int(lab.y.sum())} bind / {int((lab.y==0).sum())} not); by source:")
    print(lab.groupby("source")["y"].agg(["size", "sum", "mean"]).round(3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--source", choices=list(SOURCES), default=os.environ.get("MBG_SOURCE", "metaanalysis"))
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--metric", choices=["gbind", "mpnn_ll"], default=os.environ.get("MBG_METRIC", "gbind"))
    a = ap.parse_args()
    if a.aggregate:
        aggregate()
    else:
        sid = a.shard if a.shard is not None else int(os.environ.get("SHARD_ID", 0))
        nsh = a.num_shards if a.num_shards is not None else int(os.environ.get("NUM_SHARDS", 1))
        score_shard(a.source, sid, nsh, limit=a.limit, metric=a.metric)
