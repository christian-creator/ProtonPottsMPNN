"""EV5 stage 13 — dump the raw CHARGE CONTEXT, so the charge field can be FITTED, not assumed.

Two things are wrong with Phi as I measured it in stages 1-11:

  1. IT WAS CIRCULAR. `charge_network.residue_charge` reads each neighbour's `protonation_label` --
     which, in my feature build, was v4's own answer. So Phi was partly a readout of the vocabulary it
     was supposed to be judged against. Here every charge is A PRIORI (from residue identity alone);
     no label is ever consulted.

  2. ITS PARAMETERS WERE NEVER FITTED. ev4 sets them from a charge-scale argument, not from data:
     cutoff 8.0, shifted 1/r kernel, no screening, his_w=+0.1, acid=-1.0. Reasonable, but untested --
     and the module's own docstring says the acid thresholds "CANNOT be calibrated" because the only
     "known protonated" acids came from the override pass. With neutron ground truth that objection
     disappears: we know which acids carry a proton.

So dump the raw context once -- every charged neighbour and its distance -- and then Phi can be
recomputed for ANY (kernel, cutoff, charge weights) in-memory, and the parameters chosen by grouped CV.

Kernels the sweep will consider:
    shifted    q * (1/d - 1/rc)          ev4's; continuous at the cutoff
    bare       q / d                     Coulomb, hard truncation
    inv2       q / d^2                   distance-dependent dielectric (eps ~ r)
    debye      q * exp(-d/lam) / d       screened Coulomb -- the physically right one for water+salt,
                                         and the one ev4 never tried

Output: labeller/data/charge_ctx.parquet  (one row per (titratable atom, charged neighbour) pair)
"""
import gc, gzip, os, sys, warnings
gc.disable()
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile

_PKG = Path(__file__).resolve().parent
DATA = _PKG / "data"; DATA.mkdir(exist_ok=True)
TRAIN_DF = os.environ.get("PROTON_TRAIN_DF", str(_PKG.parent / "data/mpnn_split/train_df_filtered.parquet"))
OUT = DATA / "charge_ctx.parquet"
N_PDBS = int(sys.argv[1]) if len(sys.argv) > 1 else 250
RMAX = 16.0                                   # dump generously; the cutoff is a fitted parameter

AA20 = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS",
        "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}
FUNC = {"HIS": ("ND1", "NE2"), "ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2")}
# charge sites: the functional atoms of every ionisable residue (ev4's _charge_sites, a priori only)
SITES = {"HIS": ("ND1", "NE2"), "ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2"),
         "LYS": ("NZ",), "ARG": ("NE", "NH1", "NH2")}
HIS_H, ASP_H, GLU_H = {"HD1", "DD1"}, {"HD1", "DD1", "HD2", "DD2"}, {"HE1", "DE1", "HE2", "DE2"}
HIS_E = {"HE2", "DE2"}


def truth_labels(aa):
    out = {}
    st = struc.get_residue_starts(aa)
    for s, e in zip(st, np.append(st[1:], len(aa))):
        rn = str(aa.res_name[s])
        if rn not in FUNC:
            continue
        names = {str(x).strip().lstrip("0123456789") for x in aa.atom_name[s:e]}
        if not np.isin(aa.element[s:e], ["H", "D"]).any():
            continue
        key = (str(aa.chain_id[s]), int(aa.res_id[s]), rn)
        if rn == "HIS":
            d1, e2 = bool(names & HIS_H), bool(names & HIS_E)
            if d1 and e2:
                out[key] = "HIS-P"
            elif d1 or e2:
                out[key] = "HIS-S"
        else:
            out[key] = f"{rn}-P" if (names & (ASP_H if rn == "ASP" else GLU_H)) else f"{rn}-D"
    return out


def build_one(pdb, path):
    with gzip.open(path, "rt") as fh:
        raw = PDBFile.read(fh).get_structure(model=1)
    truth = truth_labels(raw)
    if not truth:
        return []
    heavy = raw[~np.isin(raw.element, ["H", "D"])]
    prot = heavy[np.isin([str(x) for x in heavy.res_name], list(AA20))]

    nm = np.array([str(x).strip() for x in prot.atom_name])
    rn = np.array([str(x) for x in prot.res_name])
    ch = np.array([str(x) for x in prot.chain_id])
    ri = np.array(prot.res_id)

    # every a-priori charge site, grouped by residue: (chain, res_id, res_name) -> coords
    site_mask = np.array([(r in SITES and n in SITES[r]) for r, n in zip(rn, nm)])
    keys, coords = [], []
    for c, i, r in {(a, int(b), c_) for a, b, c_ in zip(ch[site_mask], ri[site_mask], rn[site_mask])}:
        m = site_mask & (ch == c) & (ri == i)
        keys.append((c, i, r)); coords.append(prot.coord[m])

    rows = []
    st = struc.get_residue_starts(prot)
    for s, e in zip(st, np.append(st[1:], len(prot))):
        r = str(prot.res_name[s])
        if r not in FUNC:
            continue
        key = (str(prot.chain_id[s]), int(prot.res_id[s]), r)
        if key not in truth:
            continue
        for an in FUNC[r]:
            sel = (nm[s:e] == an)
            if not sel.any():
                continue
            xyz = prot.coord[s:e][sel][0]
            for (c, i, nrn), ncoord in zip(keys, coords):
                if (c, i) == key[:2]:
                    continue                                  # a residue does not charge itself
                d = float(np.linalg.norm(ncoord - xyz, axis=-1).min())   # to its CLOSEST site atom
                if d <= RMAX:
                    rows.append(dict(pdb=pdb, chain=key[0], res_id=key[1], res_name=r, atom=an,
                                     truth=truth[key], n_res_name=nrn, n_chain=c, n_res_id=i, d=d))
    return rows


df = pd.read_parquet(TRAIN_DF)
neut = df[df["method"].str.contains("NEUTRON", na=False)].drop_duplicates("pdb_id").head(N_PDBS)
print(f"{len(neut)} neutron PDBs -- a-priori charge context to {RMAX} A, protein atoms only\n", flush=True)

all_rows, ok = [], 0
for _, r in neut.iterrows():
    try:
        rows = build_one(r["pdb_id"], r["path"])
    except Exception as ex:
        print(f"  [skip] {r['pdb_id']}: {type(ex).__name__}", flush=True)
        continue
    if rows:
        all_rows += rows
        ok += 1
        if ok % 25 == 0:
            print(f"  {ok} PDBs, {len(all_rows)} pairs", flush=True)

C = pd.DataFrame(all_rows)
C.to_parquet(OUT)
print(f"\n=== {ok} PDBs -> {len(C)} (atom, charged-neighbour) pairs -> {OUT} ===")
print(C.n_res_name.value_counts().to_string())
