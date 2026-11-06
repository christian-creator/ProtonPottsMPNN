"""Structure-derived pH annotation of the de-novo RF3 folds: raw HBPLUS/PLIP geometric bonds + v6
protonation labels, per fold. For every folded redesign (all campaigns, deprot HIS-S + prot HIS-P/acid):

  prepare_potts_input(clean_pdb, extended_vocab="v6", build_bond_labels=True)  →
    - atom_array.protonation_label per residue (the v6 FLAML+HBPLUS labeller re-run on the FOLD), and
    - hbond_donates_to / hbond_accepts_from / salt_partners : per-token partner-index lists (−1 padded)
      = the raw HBPLUS H-bonds + PLIP salt bridges of the actual predicted structure.

From these we count pH-SENSITIVE bonds (binder scope: ≥1 endpoint on the binder chain) with a titratable
His / Asp-Glu endpoint → his_ph_{bonds,hbonds,saltbridges}, acid_ph_{bonds,hbonds,saltbridges}; and the
MICROSTATE MATCH = fraction of the design's pinned centres whose re-predicted label equals the pinned type.

CPU only (HBPLUS + FLAML); FLAML is fork-unsafe → one process per shard, serial. RF3 cifs carry NaN
b-factors that break HBPLUS's PDB write, so each fold is cleaned to a temp PDB first.

  SMOKE:  LIMIT=1 HBPLUS_PATH=/path/to/hbplus PYTHONPATH=. ./dpo_potts/bin/python sandbox/pdbval/annotate_denovo_folds.py
  SHARD:  NUM_SHARDS=40 SHARD_ID=i ... (writes pdbval_denovo_phannot_shard{i}.parquet; then --aggregate)
"""
import glob
import json
import os
import sys
import tempfile

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd

FOLDS = os.environ.get("FOLDS", "data/ph_sensitive/pdb_val_dimers/pdbval_denovo_redesign_folds.parquet")
JSONL_GLOBS = ["outputs/ph_sensitive_pdbval_denovo_redesign_potts_s*/*/iter_01/metadata/optimized_sequences.jsonl",
               "outputs/ph_sensitive_pdbval_denovo_redesign_decoder_s*/*/iter_01/metadata/optimized_sequences.jsonl",
               "outputs/ph_sensitive_pdbval_denovo_redesign_potts_w0_s*/*/iter_01/metadata/optimized_sequences.jsonl"]
OUTDIR = "data/ph_sensitive/pdb_val_dimers/phannot"
OUTNAME = os.environ.get("OUTNAME", "pdbval_denovo_phannot")
SHARD_ID = int(os.environ.get("SHARD_ID", "0"))
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "1"))
LIMIT = int(os.environ.get("LIMIT", "0"))
_HIS, _ACID = {"HIS"}, {"ASP", "GLU"}


def _np(x):
    """A NumPy array from a tensor on any device (the features sit on the GPU when one is visible)."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _clean_to_pdb(cif):
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx
    import biotite.structure.io.pdb as pdbio
    aa = pdbx.get_structure(pdbx.CIFFile.read(cif), model=1)
    aa = aa[struc.filter_amino_acids(aa)]
    aa = aa[~np.isnan(aa.coord).any(axis=1)]
    aa.b_factor = np.zeros(aa.array_length())
    tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False).name
    f = pdbio.PDBFile(); f.set_structure(aa); f.write(tmp)
    return tmp


def _edges(partner_lists):
    """Unordered edge set {(min,max)} from a [L, K] partner-index array (−1 padded)."""
    e = set()
    for i, row in enumerate(partner_lists):
        for j in row:
            if j >= 0:
                e.add((min(i, int(j)), max(i, int(j))))
    return e


def annotate(cif, binder_len, centers):
    """(dict of pH-bond counts + microstate_match) for one fold. centers = [(res_id, pinned_type)]."""
    from mpnn.potts_inference import prepare_potts_input
    pdb = _clean_to_pdb(cif)
    out = prepare_potts_input(pdb, extended_vocab="v6", build_bond_labels=True)
    ni = out["network_input"]["input_features"]
    arr = out["atom_array"]
    ca = arr[arr.atom_name == "CA"]                                   # token order == CA order
    chain = np.asarray(ca.chain_id); resid = np.asarray(ca.res_id).astype(int)
    resn = np.asarray(ca.res_name); plabel = np.asarray(ca.get_annotation("protonation_label"))
    L = len(ca)
    # binder chain = the chain whose residue count == binder_len (target is the other; lengths differ here)
    binder_ch = None
    for c in sorted(set(chain)):
        if int((chain == c).sum()) == binder_len:
            binder_ch = c; break
    if binder_ch is None:
        # Previously this fell back to `np.ones(L, bool)`, i.e. "every residue is binder", which made
        # _count score EVERY bond in the dimer as binder-scoped and silently inflated *_ph_bonds;
        # center_gated_bonds went nan and microstate_match scored 0.0 against a `None` chain key.
        # A fold whose binder cannot be identified has no meaningful read-out, so fail loudly --
        # both callers (design_ph.py, the --aggregate driver) already catch and skip.
        raise ValueError(
            f"no chain has binder_len={binder_len} residues in {cif} "
            f"(chain residue counts: {{{', '.join(f'{c!r}: {int((chain == c).sum())}' for c in sorted(set(chain)))}}}) "
            "-- pass the binder_len that matches this fold")
    is_binder = (chain == binder_ch)

    hb = _edges(_np(ni["hbond_donates_to"]).squeeze(0)) | _edges(_np(ni["hbond_accepts_from"]).squeeze(0))
    sb = _edges(_np(ni["salt_partners"]).squeeze(0))

    def _count(edges, cls):
        n = 0
        for i, j in edges:
            if not (is_binder[i] or is_binder[j]):                   # binder scope: ≥1 endpoint on binder
                continue
            if str(resn[i]) in cls or str(resn[j]) in cls:           # ≥1 titratable endpoint of this class
                n += 1
        return n

    res = dict(
        his_ph_hbonds=_count(hb, _HIS), his_ph_saltbridges=_count(sb, _HIS),
        acid_ph_hbonds=_count(hb, _ACID), acid_ph_saltbridges=_count(sb, _ACID))
    res["his_ph_bonds"] = res["his_ph_hbonds"] + res["his_ph_saltbridges"]
    res["acid_ph_bonds"] = res["acid_ph_hbonds"] + res["acid_ph_saltbridges"]

    # pH-GATED bonds at the pinned CENTRES (bi-directional, state-specific): the H-bonds the centre's TARGET
    # protonation state uniquely enables — protonated (HIS-P/ASP-P/GLU-P) → the titratable proton DONATES;
    # neutral His (HIS-S) → the lone pair ACCEPTS (lost on protonation). Partner unrestricted (Thr/acid/…).
    don = _np(ni["hbond_donates_to"]).squeeze(0)
    acc = _np(ni["hbond_accepts_from"]).squeeze(0)
    tok_at = {int(rd): t for t, (ch, rd) in enumerate(zip(chain, resid)) if ch == binder_ch}
    gated = []
    for rid, pt in centers:
        t = tok_at.get(int(rid))
        if t is None:
            continue
        arr = acc if str(pt) == "HIS-S" else don            # HIS-S accepts; protonated states donate
        gated.append(int((arr[t] >= 0).sum()))
    res["center_gated_bonds"] = float(np.sum(gated)) if gated else np.nan       # total over centres
    res["center_gated_bonds_mean"] = float(np.mean(gated)) if gated else np.nan  # per-centre (n_centers-invariant)

    # microstate match: re-predicted label at each pinned centre (binder chain, res_id) == pinned type
    lab_at = {(str(binder_ch), int(rid)): str(pl) for rid, pl in zip(resid[is_binder], plabel[is_binder])}
    matches = [1.0 if lab_at.get((str(binder_ch), int(rid))) == pt else 0.0 for rid, pt in centers]
    res["microstate_match"] = float(np.mean(matches)) if matches else np.nan
    res["n_centers_checked"] = len(matches)
    return res


# %% centre map (design_id, sequence) -> [(res_id, pinned_type)] from the redesign jsonl
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
        print(f"[aggregate] {len(df)} annotated folds from {len(parts)} shards -> {out}")
        for c in ["his_ph_bonds", "acid_ph_bonds", "microstate_match"]:
            print(f"  {c}: mean {df[c].mean():.3f}")
        sys.exit(0)

    os.makedirs(OUTDIR, exist_ok=True)
    folds = pd.read_parquet(FOLDS).dropna(subset=["refold_cif"]).reset_index(drop=True)
    folds = folds.iloc[SHARD_ID::NUM_SHARDS].reset_index(drop=True)
    if LIMIT:
        folds = folds.head(LIMIT)
    cm = _center_map()
    print(f"[annot] shard {SHARD_ID}/{NUM_SHARDS}: {len(folds)} folds | centre map {len(cm)}", flush=True)
    rows = []
    for i, r in folds.iterrows():
        centers = cm.get((r.design_id, r.sequence), [])
        try:
            a = annotate(r.refold_cif, int(r.binder_len), centers)
        except Exception as e:
            print(f"  {i} FAIL {r.sequence_id}: {type(e).__name__}: {str(e)[:80]}", flush=True)
            a = {}
        rows.append(dict(sequence_id=r.sequence_id, design_id=r.design_id, sequence=r.sequence,
                         refold_cif=r.refold_cif, **a))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(folds)}", flush=True)
    out = f"{OUTDIR}/{OUTNAME}_shard{SHARD_ID}.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"[annot] wrote {len(rows)} -> {out}", flush=True)
