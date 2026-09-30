"""CHARGE-CLASH annotation of the de-novo RF3 folds: for each pinned protonation centre, how many
same-sign charged residues sit within 4 A (heavy-atom) of it — i.e. how electrostatically clashed the
centre's CHARGED microstate is in the actual predicted structure.

  HIS-S / HIS-P centre  -> count POSITIVE neighbours (ARG, LYS, HIS)  — the clash the protonated His feels.
        HIS is INCLUDED: at the low-pH condition where the His centre is charged, nearby His protonate too
        (pKa ~6), so His+...His+ (incl. another pinned HIS-P centre) is a genuine same-sign clash.
  ASP-P / GLU-P centre  -> count NEGATIVE neighbours (ASP, GLU)       — the clash the deprotonated acid feels.
        (His is NEUTRAL at the high-pH deprotonated condition, so it is not counted for acids.)

Neighbours are counted across BOTH chains (binder + target); a residue is a neighbour if any of its heavy
atoms lies within 4 A of any heavy atom of the centre residue (the centre itself is excluded). Per design
we report the MEAN over its pinned centres (n_centres-invariant). Pure geometry (biotite) — no model, no
HBPLUS. Reads the same FOLDS parquet + redesign-jsonl centre map as annotate_denovo_folds.py.

  SMOKE:  LIMIT=3 PYTHONPATH=. <mainvenv>/python sandbox/pdbval/annotate_denovo_charge_clash.py
  SHARD:  NUM_SHARDS=8 SHARD_ID=i ...      (writes ..._charge_clash_shard{i}.parquet)
  AGG:    --aggregate                      (-> pdbval_denovo_charge_clash.parquet)
  FOLDS / OUTDIR overridable via env (for the w=0 campaign).
"""
import glob
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd

MAIN = "/novo/users/cpjb/PHD/diffusion_DPO_gefion"
FOLDS = os.environ.get("FOLDS", f"{MAIN}/data/ph_sensitive/pdb_val_dimers/pdbval_denovo_redesign_folds.parquet")
JSONL_GLOBS = os.environ.get("JSONL_GLOBS", "|".join([
    f"{MAIN}/outputs/ph_sensitive_pdbval_denovo_redesign_potts_s*/*/iter_01/metadata/optimized_sequences.jsonl",
    f"{MAIN}/outputs/ph_sensitive_pdbval_denovo_redesign_decoder_s*/*/iter_01/metadata/optimized_sequences.jsonl",
    f"{MAIN}/outputs/ph_sensitive_pdbval_denovo_redesign_potts_w0_s*/*/iter_01/metadata/optimized_sequences.jsonl",
])).split("|")
OUTDIR = os.environ.get("OUTDIR", f"{MAIN}/data/ph_sensitive/pdb_val_dimers/phannot")
OUTNAME = os.environ.get("OUTNAME", "pdbval_denovo_charge_clash")
SHARD_ID = int(os.environ.get("SHARD_ID", "0"))
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "1"))
LIMIT = int(os.environ.get("LIMIT", "0"))
POS, NEG = {"ARG", "LYS", "HIS"}, {"ASP", "GLU"}     # HIS in POS: protonates at the low-pH His-charged condition
CLASH_DIST = 4.0
SALT_DIST = 5.0                                       # opposite-sign salt-bridge partner range (pH-dependent stabilizing bond)


def charge_clash(cif, binder_len, centers):
    """Mean over the design's pinned centres of (# same-sign charged residues within 4 A, both chains).

    Returns ``(clash, salt_bridge, n_centers_scored)``. When no chain's CA count matches
    ``binder_len`` the binder cannot be identified and it returns ``(nan, nan, 0)`` -- a THREE-tuple,
    like every other exit, because all callers unpack three
    (``annotate_charge_clash.py`` __main__, ``scoring/score_example.py``, ``inference/design_ph.py``).
    """
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx
    aa = pdbx.get_structure(pdbx.CIFFile.read(cif), model=1)
    aa = aa[struc.filter_amino_acids(aa)]
    aa = aa[~np.isnan(aa.coord).any(axis=1)]
    chain = np.asarray(aa.chain_id); resid = np.asarray(aa.res_id).astype(int)
    resn = np.asarray(aa.res_name); is_ca = np.asarray(aa.atom_name) == "CA"
    binder_ch = None
    for c in sorted(set(chain.tolist())):
        if int((is_ca & (chain == c)).sum()) == binder_len:
            binder_ch = c; break
    if binder_ch is None:
        return np.nan, np.nan, 0
    cell = struc.CellList(aa, cell_size=SALT_DIST)
    clash, salt = [], []
    for rid, ptype in centers:
        same = POS if str(ptype) in ("HIS-S", "HIS-P") else NEG      # same-sign as the charged microstate (CLASH)
        opp = NEG if str(ptype) in ("HIS-S", "HIS-P") else POS       # opposite sign = SALT-BRIDGE partner (stabilizing)
        cmask = (chain == binder_ch) & (resid == int(rid))
        if not cmask.any():
            continue

        def _count(radius, cls):
            idx = np.unique(cell.get_atoms(aa.coord[cmask], radius=radius))
            idx = idx[idx >= 0]
            seen, cnt = set(), 0
            for a in idx:
                if chain[a] == binder_ch and int(resid[a]) == int(rid):
                    continue
                key = (chain[a], int(resid[a]))
                if key in seen:
                    continue
                seen.add(key)
                if str(resn[a]) in cls:
                    cnt += 1
            return cnt
        clash.append(_count(CLASH_DIST, same))                       # same-sign within 4 Å
        salt.append(_count(SALT_DIST, opp))                          # opposite-sign within 5 Å (pH-dependent salt bridge)
    cc = float(np.mean(clash)) if clash else np.nan
    sb = float(np.mean(salt)) if salt else np.nan
    return cc, sb, len(clash)


def _center_map():
    cm = {}
    for g in JSONL_GLOBS:
        for f in glob.glob(g):
            for l in open(f):
                r = json.loads(l)
                if r.get("source") != "sequence_optimization" or not r.get("sequence"):
                    continue
                rids = r.get("center_res_ids") or []
                types = r.get("center_protonation_types") or []
                cm[(r["design_id"], r["sequence"])] = list(zip([int(x) for x in rids], [str(t) for t in types]))
    return cm


if __name__ == "__main__":
    if "--aggregate" in sys.argv:
        parts = sorted(glob.glob(f"{OUTDIR}/{OUTNAME}_shard*.parquet"))
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        out = f"{OUTDIR}/{OUTNAME}.parquet"
        df.to_parquet(out, index=False)
        print(f"[aggregate] {len(df)} folds from {len(parts)} shards -> {out}")
        print(f"  charge_clash: mean {df.charge_clash.mean():.3f} | salt_bridge: mean {df.salt_bridge.mean():.3f} | notna {df.charge_clash.notna().sum()}")
        sys.exit(0)

    os.makedirs(OUTDIR, exist_ok=True)
    folds = pd.read_parquet(FOLDS).dropna(subset=["refold_cif"]).reset_index(drop=True)
    folds = folds.iloc[SHARD_ID::NUM_SHARDS].reset_index(drop=True)
    if LIMIT:
        folds = folds.head(LIMIT)
    cm = _center_map()
    print(f"[clash] shard {SHARD_ID}/{NUM_SHARDS}: {len(folds)} folds | centre map {len(cm)}", flush=True)
    rows = []
    for i, r in folds.iterrows():
        centers = cm.get((r.design_id, r.sequence), [])
        try:
            cc, sb, ncc = charge_clash(r.refold_cif, int(r.binder_len), centers)
        except Exception as e:
            print(f"  {i} FAIL {r.sequence_id}: {type(e).__name__}: {str(e)[:80]}", flush=True)
            cc, sb, ncc = np.nan, np.nan, 0
        rows.append(dict(sequence_id=r.sequence_id, design_id=r.design_id, sequence=r.sequence,
                         charge_clash=cc, salt_bridge=sb, n_centers_clash=ncc))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(folds)}", flush=True)
    out = f"{OUTDIR}/{OUTNAME}_shard{SHARD_ID}.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"[clash] wrote {len(rows)} -> {out}", flush=True)
