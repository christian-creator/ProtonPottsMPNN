"""EV5 stage 35 — a RICH, EXPLICIT feature table for HIS-P vs HIS-S.

Built to be LEARNED FROM, not just predicted with: every column is a named, physically interpretable
quantity, so that whatever the model finds can be read back as chemistry.

  METAL-COORDINATING AND METAL-ADJACENT HIS ARE DROPPED (any metal within 3.5 A of a ring N).
  171 residues go, of which only 2 are HIS-P. They are 99% neutral for a reason the model can never
  observe -- there is no Zn at design time -- and left in, a tree model WILL find them: converging
  ligating side chains are visible in protein coordinates alone, so it would quietly learn a metal-site
  detector and be rewarded for it. Dropping them raises the base rate 30% -> 34% and makes the task
  HARDER, which is the honest thing.

Feature families (all protein heavy atoms only):

  A. PER-RING-NITROGEN H-BONDS   ND1 and NE2 kept SEPARATE and also as a symmetric summary. For each:
     does it donate / accept, the best DHA angle, best H-A and D-A distance, and the TYPE of partner
     (backbone C=O, carboxylate, hydroxyl, amide, His N, cation, sulfur). This is where the chemistry is.
  B. NEIGHBOURHOOD COMPOSITION   how many of each residue class sit within 6 / 8 / 10 A -- acidic,
     basic, polar, aromatic, aliphatic, Gly/Pro, His. The pocket's identity, not just its size.
  C. DENSITY / BURIAL            heavy-atom counts at 4/6/8/10/12 A, side-chain SASA, and the ratio of
     side-chain to backbone neighbours (is it packed against protein or hanging in solvent?).
  D. ELECTROSTATICS              the fitted charge field Phi at each ring N (and their DIFFERENCE, which
     is what picks the tautomer), plus net-charge counts at 6/8/10 A.
  E. GEOMETRY                    chi1/chi2, distance to the nearest aromatic ring centroid (stacking),
     secondary structure, relative position in the chain.
  F. SEQUENCE CONTEXT            the residue classes at i-2..i+2.

Output: sandbox/ev5/his_rich.parquet
"""
import gc, gzip, os, sys, warnings
gc.disable()
warnings.filterwarnings("ignore")

import os
from pathlib import Path

import numpy as np
import pandas as pd
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile

HERE = Path(__file__).parent           # this stage's own folder — figures/CSVs land here
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)            # the shared parquet inputs, built once by stages 00/12/13/35/38/43/44/47/49
MODELS = HERE / "models"; MODELS.mkdir(exist_ok=True)        # the trained AutoML artefacts
sys.path.insert(0, str(HERE))
KEY = ["pdb", "chain", "res_id"]
OUT = DATA / "his_rich.parquet"
METAL_CUT = 3.5                      # drop His with any metal this close to a ring N

AA20 = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS",
        "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}
METALS = {"ZN", "CU", "FE", "MN", "NI", "CO", "MG", "CA", "CD", "HG"}
RING = ("ND1", "NE2")
CLASS = {**{r: "acidic" for r in ("ASP", "GLU")},
         **{r: "basic" for r in ("LYS", "ARG")},
         **{r: "polar" for r in ("SER", "THR", "ASN", "GLN", "TYR", "CYS")},
         **{r: "aromatic" for r in ("PHE", "TRP", "TYR")},
         **{r: "aliphatic" for r in ("ALA", "VAL", "LEU", "ILE", "MET")},
         **{r: "glypro" for r in ("GLY", "PRO")}, "HIS": "his"}
CLASSES = ("acidic", "basic", "polar", "aromatic", "aliphatic", "glypro", "his")
CARBOX_O = {"OD1", "OD2", "OE1", "OE2"}
CATION_N = {"NZ", "NE", "NH1", "NH2"}
HYDROXYL = {("SER", "OG"), ("THR", "OG1"), ("TYR", "OH")}
AMIDE_O = {("ASN", "OD1"), ("GLN", "OE1")}
SULFUR = {("CYS", "SG"), ("MET", "SD")}
AROM_RINGS = {"PHE": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
              "TYR": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
              "TRP": ("CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2")}
HIS_H, HIS_E = {"HD1", "DD1"}, {"HE2", "DE2"}


def partner_class(prn, pat):
    if pat in CARBOX_O:
        return "carboxylate"
    if pat in ("O", "OXT"):
        return "bbcarbonyl"
    if pat == "N":
        return "bbamide"
    if pat in CATION_N:
        return "cation"
    if (prn, pat) in HYDROXYL:
        return "hydroxyl"
    if (prn, pat) in AMIDE_O:
        return "amide"
    if (prn, pat) in SULFUR:
        return "sulfur"
    if (prn, pat) in {("HIS", "ND1"), ("HIS", "NE2")}:
        return "hisN"
    return "other"


PCLS = ("carboxylate", "bbcarbonyl", "bbamide", "cation", "hydroxyl", "amide", "sulfur", "hisN", "other")


def build_one(pdb, path):
    with gzip.open(path, "rt") as fh:
        raw = PDBFile.read(fh).get_structure(model=1)
    heavy = raw[~np.isin(raw.element, ["H", "D"])]
    rn_all = np.array([str(x) for x in heavy.res_name])
    prot = heavy[np.isin(rn_all, list(AA20))]
    if not len(prot):
        return []
    met_xyz = heavy.coord[np.isin([str(x).strip().upper() for x in heavy.element], list(METALS))]

    nm = np.array([str(x).strip() for x in prot.atom_name])
    rn = np.array([str(x) for x in prot.res_name])
    ch = np.array([str(x) for x in prot.chain_id])
    ri = np.array(prot.res_id)
    xyz = prot.coord
    is_sc = ~np.isin(nm, ["N", "CA", "C", "O", "OXT"])

    # SASA of the whole protein, summed per residue side chain
    try:
        sasa = struc.sasa(prot, vdw_radii="Single")
        sasa = np.nan_to_num(sasa)
    except Exception:
        sasa = np.zeros(len(prot))
    try:
        sse = struc.annotate_sse(prot)          # per-residue, only for amino acids
        sse_starts = struc.get_residue_starts(prot)
    except Exception:
        sse, sse_starts = None, None

    # aromatic ring centroids
    arom = []
    for (c, i, r) in {(a, int(b), c_) for a, b, c_ in zip(ch, ri, rn) if c_ in AROM_RINGS}:
        m = (ch == c) & (ri == i) & np.isin(nm, list(AROM_RINGS[r]))
        if m.sum() >= 5:
            arom.append(xyz[m].mean(0))
    arom = np.array(arom) if arom else np.zeros((0, 3))

    # ground truth from the deposited H/D
    truth = {}
    st_raw = struc.get_residue_starts(raw)
    for s, e in zip(st_raw, np.append(st_raw[1:], len(raw))):
        if str(raw.res_name[s]) != "HIS":
            continue
        names = {str(x).strip().lstrip("0123456789") for x in raw.atom_name[s:e]}
        if not np.isin(raw.element[s:e], ["H", "D"]).any():
            continue
        d1, e2 = bool(names & HIS_H), bool(names & HIS_E)
        if d1 and e2:
            truth[(str(raw.chain_id[s]), int(raw.res_id[s]))] = "HIS-P"
        elif d1 or e2:
            truth[(str(raw.chain_id[s]), int(raw.res_id[s]))] = "HIS-S"

    rows = []
    starts = struc.get_residue_starts(prot)
    ends = np.append(starts[1:], len(prot))
    res_keys = [(str(prot.chain_id[s]), int(prot.res_id[s])) for s in starts]
    for idx, (s, e) in enumerate(zip(starts, ends)):
        if str(prot.res_name[s]) != "HIS":
            continue
        key = (str(prot.chain_id[s]), int(prot.res_id[s]))
        if key not in truth:
            continue
        own = (ch == key[0]) & (ri == key[1])
        ring_xyz = {a: xyz[own & (nm == a)] for a in RING}
        if any(len(v) == 0 for v in ring_xyz.values()):
            continue
        cen = np.vstack(list(ring_xyz.values())).mean(0)

        # metal veto — the ONLY use of a metal anywhere: to EXCLUDE the residue
        dmet = 99.0
        if len(met_xyz):
            dmet = float(min(np.linalg.norm(met_xyz - ring_xyz[a][0], axis=1).min() for a in RING))
        if dmet < METAL_CUT:
            continue

        rec = dict(pdb=pdb, chain=key[0], res_id=key[1], truth=truth[key], d_metal=dmet)

        # ── C. density / burial ────────────────────────────────────────────────────────
        d_cen = np.linalg.norm(xyz - cen, axis=1)
        for r_ in (4, 6, 8, 10, 12):
            rec[f"dens{r_}"] = int((d_cen < r_).sum())
        near8 = d_cen < 8
        rec["sc_frac8"] = float(is_sc[near8].mean()) if near8.any() else 0.0
        rec["sasa_sc"] = float(sasa[own & is_sc].sum())
        rec["sasa_ring"] = float(sasa[own & np.isin(nm, RING)].sum())

        # ── B. neighbourhood composition ───────────────────────────────────────────────
        for r_ in (6, 8, 10):
            near = d_cen < r_
            seen = {(c, int(i)): rr for c, i, rr, k in zip(ch[near], ri[near], rn[near], near[near])}
            cnt = {c_: 0 for c_ in CLASSES}
            for (c, i), rr in seen.items():
                if (c, i) == key:
                    continue
                cnt[CLASS.get(rr, "aliphatic")] += 1
            for c_ in CLASSES:
                rec[f"n_{c_}{r_}"] = cnt[c_]
            rec[f"n_res{r_}"] = sum(cnt.values())
            rec[f"net_q{r_}"] = cnt["basic"] - cnt["acidic"]

        # ── E. geometry ────────────────────────────────────────────────────────────────
        def near_to(mask, coords=None):
            src = coords if coords is not None else xyz[mask]
            if not len(src):
                return 99.0
            return float(min(np.linalg.norm(src - ring_xyz[a][0], axis=1).min() for a in RING))

        rec["d_carbox"] = near_to(np.isin(nm, list(CARBOX_O)) & ~own)
        rec["d_cation"] = near_to(np.isin(nm, list(CATION_N)) & ~own)
        rec["d_hydroxyl"] = near_to(np.array([(a, b) in HYDROXYL for a, b in zip(rn, nm)]) & ~own)
        rec["d_amide"] = near_to(np.array([(a, b) in AMIDE_O for a, b in zip(rn, nm)]) & ~own)
        rec["d_sulfur"] = near_to(np.array([(a, b) in SULFUR for a, b in zip(rn, nm)]) & ~own)
        rec["d_bbO"] = near_to(np.isin(nm, ["O", "OXT"]) & ~own)
        rec["d_bbN"] = near_to((nm == "N") & ~own)
        rec["d_hisN"] = near_to(np.isin(nm, RING) & (rn == "HIS") & ~own)
        rec["d_arom"] = near_to(None, arom) if len(arom) else 99.0
        if sse is not None and idx < len(sse):
            rec["sse"] = str(sse[idx])
        else:
            rec["sse"] = "c"
        n_res_chain = (np.array([k[0] for k in res_keys]) == key[0]).sum()
        rec["rel_pos"] = float(idx) / max(len(res_keys), 1)
        rec["chain_len"] = int(n_res_chain)

        # ── F. sequence context ────────────────────────────────────────────────────────
        for off in (-2, -1, 1, 2):
            j = idx + off
            rec[f"seq{off:+d}"] = (CLASS.get(str(prot.res_name[starts[j]]), "aliphatic")
                                   if 0 <= j < len(starts) else "none")
        rows.append(rec)
    return rows


df = pd.read_parquet(os.environ.get("PROTON_TRAIN_DF", str(HERE.parent / "data/mpnn_split/train_df_filtered.parquet")))
neut = df[df["method"].str.contains("NEUTRON", na=False)].drop_duplicates("pdb_id")

# Restrict to the PDBs the stage-01/02 dumps actually cover. Those stages take an N_PDBS cap
# (default 250); this one used to iterate EVERY neutron PDB, so any structure beyond the cap
# survived to the joins below with no H-bond and no charge row and was SENTINEL-FILLED
# (*_n=0, *_ha/_da=9.0, *_dha=0, phi_*=0.0) -- i.e. fed to FLAML as "no H-bonds, neutral
# pocket" rather than dropped. Intersecting with the dumps keeps the four stages consistent
# whatever cap stage 01/02 ran with.
_covered = None
for _dump in ("hbonds.parquet", "charge_ctx.parquet"):
    _p = DATA / _dump
    if not _p.exists():
        raise FileNotFoundError(f"{_p} is missing -- run 01_hbond_dump.py and 02_charge_dump.py first.")
    _pdbs = set(pd.read_parquet(_p, columns=["pdb"])["pdb"].unique())
    _covered = _pdbs if _covered is None else (_covered & _pdbs)
_n_all = len(neut)
neut = neut[neut["pdb_id"].isin(_covered)]
if len(neut) < _n_all:
    print(f"  [cap] {_n_all} neutron PDBs in the split, {len(neut)} covered by the stage-01/02 dumps "
          f"-- skipping {_n_all - len(neut)} uncovered (re-run 01/02 with a larger N_PDBS to include them)",
          flush=True)
print(f"{len(neut)} neutron PDBs; dropping any His with a metal within {METAL_CUT} A of a ring N\n",
      flush=True)

rows, ok = [], 0
for _, r in neut.iterrows():
    try:
        rs = build_one(r["pdb_id"], r["path"])
    except Exception as ex:
        print(f"  [skip] {r['pdb_id']}: {type(ex).__name__}: {ex}", flush=True)
        continue
    if rs:
        rows += rs
        ok += 1
        if ok % 40 == 0:
            print(f"  {ok} PDBs, {len(rows)} His", flush=True)

G = pd.DataFrame(rows)

# ── A. per-ring-N H-bond features, from the raw HBPLUS dump ────────────────────────────
B = pd.read_parquet(DATA / "hbonds.parquet")
hb = B[(B.res_name == "HIS") & (B["mode"] == "default")].copy()
hb["pcls"] = [partner_class(a, b) for a, b in zip(hb.p_res_name, hb.p_atom)]
G = G.set_index(KEY)
for tag, atom in (("nd1", "ND1"), ("ne2", "NE2")):
    for role in ("donor", "acceptor"):
        s = hb[(hb.atom == atom) & (hb.role == role)]
        g = s.groupby(KEY).agg(**{f"{tag}_{role}_n": ("dha", "size"),
                                  f"{tag}_{role}_dha": ("dha", "max"),
                                  f"{tag}_{role}_ha": ("ha", "min"),
                                  f"{tag}_{role}_da": ("da", "min")})
        G = G.join(g)
        # partner types, per role
        for pc in PCLS:
            k = s[s.pcls == pc].groupby(KEY).size().rename(f"{tag}_{role}_{pc}")
            G = G.join(k)
G = G.fillna({c: 0 for c in G.columns if c.endswith(tuple(["_n"] + list(PCLS)))})
G = G.fillna({c: 0 for c in G.columns if c.endswith("_dha")})
G = G.fillna({c: 9.0 for c in G.columns if c.endswith(("_ha", "_da"))})

# symmetric summaries: the rule works on "both / neither", not on ND1 vs NE2 identity
for role in ("donor", "acceptor"):
    a = (G[f"nd1_{role}_n"] > 0).astype(int)
    b = (G[f"ne2_{role}_n"] > 0).astype(int)
    G[f"n_{role}_atoms"] = a + b
    G[f"best_{role}_dha"] = G[[f"nd1_{role}_dha", f"ne2_{role}_dha"]].max(axis=1)
    G[f"best_{role}_ha"] = G[[f"nd1_{role}_ha", f"ne2_{role}_ha"]].min(axis=1)
    for pc in PCLS:
        G[f"{role}_{pc}"] = G[f"nd1_{role}_{pc}"] + G[f"ne2_{role}_{pc}"]
G["n_hbonds"] = G[[c for c in G.columns if c.endswith("_n") and ("donor" in c or "acceptor" in c)]].sum(1)

# ── D. the fitted charge field, per ring N ─────────────────────────────────────────────
from ev5_rules import charge_field
Cx = pd.read_parquet(DATA / "charge_ctx.parquet")
q = np.where(Cx.n_res_name.isin(("LYS", "ARG")), 1.0,
             np.where(Cx.n_res_name == "HIS", 1.0,
                      np.where(Cx.n_res_name.isin(("ASP", "GLU")), -0.5, 0.0)))
for rc in (6.0, 8.0):
    v = np.where(Cx.d.values <= rc, q / Cx.d.values, 0.0)
    per_atom = (pd.DataFrame(dict(pdb=Cx.pdb, chain=Cx.chain, res_id=Cx.res_id, atom=Cx.atom, v=v))
                .groupby(KEY + ["atom"]).v.sum().unstack("atom"))
    G[f"phi_nd1_{rc:.0f}"] = per_atom.get("ND1")
    G[f"phi_ne2_{rc:.0f}"] = per_atom.get("NE2")
    G[f"phi_min_{rc:.0f}"] = per_atom[["ND1", "NE2"]].min(axis=1)
    G[f"phi_max_{rc:.0f}"] = per_atom[["ND1", "NE2"]].max(axis=1)
    G[f"phi_diff_{rc:.0f}"] = (per_atom["ND1"] - per_atom["NE2"]).abs()   # picks the TAUTOMER
G = G.fillna({c: 0.0 for c in G.columns if c.startswith("phi_")})

G["y"] = (G.truth == "HIS-P").astype(int)
G.reset_index().to_parquet(OUT)
print(f"\n=== {ok} PDBs -> {len(G)} His ({int(G.y.sum())} HIS-P, {100*G.y.mean():.0f}%) -> {OUT} ===")
print(f"    {G.shape[1] - 3} features")
print(f"    metal-coordinating/adjacent His EXCLUDED (< {METAL_CUT} A)")
