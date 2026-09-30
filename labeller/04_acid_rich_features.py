"""EV5 stage 43 — a RICH, EXPLICIT feature table for ASP-P/GLU-P vs ASP-D/GLU-D.

The exact mirror of stage 35 (the HIS table), with the imidazole ring swapped for the carboxyl group:
the two ring nitrogens ND1/NE2 become the two carboxyl oxygens OD1/OD2 (Asp) or OE1/OE2 (Glu). They are
kept as `o1` / `o2` so ASP and GLU pool into one table.

  METAL-COORDINATING AND METAL-ADJACENT ACIDS ARE DROPPED (any metal within 3.5 A of a carboxyl O), for
  the same reason as the His: a metal-bound carboxylate is deprotonated because of something no
  design-time model can see, and left in, a tree would learn a metal-site detector and be rewarded.

THE HARD PART IS THE BASE RATE. Only ~1.5% of acids are protonated (vs 34% of His), so:
  * average precision must be read against 0.015, not against 0.5;
  * ~100 positives across ~45 PDBs is the true sample size, and the PR curve will be visibly noisy;
  * a model that "does nothing" scores 98.5% accuracy. Accuracy is meaningless here.

Feature families (identical structure to the HIS table, so the two are directly comparable):
  A  per-carboxyl-oxygen H-bond roles & geometry, and the TYPE of partner
  B  neighbourhood composition by residue class at 6 / 8 / 10 A
  C  density / burial / SASA
  D  electrostatics: the fitted charge field Phi per oxygen, plus net-charge counts
  E  distances to functional groups -- including the DYAD (nearest OTHER carboxylate O)
  F  geometry / secondary structure / position
  G  sequence context

Output: sandbox/ev5/acid_rich.parquet
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
OUT = DATA / "acid_rich.parquet"
METAL_CUT = 3.5

AA20 = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS",
        "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}
METALS = {"ZN", "CU", "FE", "MN", "NI", "CO", "MG", "CA", "CD", "HG"}
CARBOX = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2")}
ALL_CARBOX_O = {"OD1", "OD2", "OE1", "OE2"}
CLASS = {**{r: "acidic" for r in ("ASP", "GLU")},
         **{r: "basic" for r in ("LYS", "ARG")},
         **{r: "polar" for r in ("SER", "THR", "ASN", "GLN", "TYR", "CYS")},
         **{r: "aromatic" for r in ("PHE", "TRP", "TYR")},
         **{r: "aliphatic" for r in ("ALA", "VAL", "LEU", "ILE", "MET")},
         **{r: "glypro" for r in ("GLY", "PRO")}, "HIS": "his"}
CLASSES = ("acidic", "basic", "polar", "aromatic", "aliphatic", "glypro", "his")
CATION_N = {"NZ", "NE", "NH1", "NH2"}
HYDROXYL = {("SER", "OG"), ("THR", "OG1"), ("TYR", "OH")}
AMIDE_O = {("ASN", "OD1"), ("GLN", "OE1")}
SULFUR = {("CYS", "SG"), ("MET", "SD")}
RING_N = {("HIS", "ND1"), ("HIS", "NE2")}
AROM_RINGS = {"PHE": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
              "TYR": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
              "TRP": ("CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2")}
ASP_H = {"HD1", "DD1", "HD2", "DD2"}
GLU_H = {"HE1", "DE1", "HE2", "DE2"}


def partner_class(prn, pat):
    if pat in ALL_CARBOX_O:
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
    if (prn, pat) in RING_N:
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

    try:
        sasa = np.nan_to_num(struc.sasa(prot, vdw_radii="Single"))
    except Exception:
        sasa = np.zeros(len(prot))
    try:
        sse = struc.annotate_sse(prot)
    except Exception:
        sse = None

    arom = []
    for (c, i, r) in {(a, int(b), c_) for a, b, c_ in zip(ch, ri, rn) if c_ in AROM_RINGS}:
        m = (ch == c) & (ri == i) & np.isin(nm, list(AROM_RINGS[r]))
        if m.sum() >= 5:
            arom.append(xyz[m].mean(0))
    arom = np.array(arom) if arom else np.zeros((0, 3))

    # ground truth: a carboxyl O carrying a deposited H/D  ->  protonated
    truth = {}
    st_raw = struc.get_residue_starts(raw)
    for s, e in zip(st_raw, np.append(st_raw[1:], len(raw))):
        r = str(raw.res_name[s])
        if r not in CARBOX:
            continue
        names = {str(x).strip().lstrip("0123456789") for x in raw.atom_name[s:e]}
        if not np.isin(raw.element[s:e], ["H", "D"]).any():
            continue                                   # no hydrogens modelled -> no evidence
        oh = ASP_H if r == "ASP" else GLU_H
        truth[(str(raw.chain_id[s]), int(raw.res_id[s]))] = f"{r}-P" if (names & oh) else f"{r}-D"

    rows = []
    starts = struc.get_residue_starts(prot)
    ends = np.append(starts[1:], len(prot))
    for idx, (s, e) in enumerate(zip(starts, ends)):
        r = str(prot.res_name[s])
        if r not in CARBOX:
            continue
        key = (str(prot.chain_id[s]), int(prot.res_id[s]))
        if key not in truth:
            continue
        own = (ch == key[0]) & (ri == key[1])
        o_names = CARBOX[r]
        o_xyz = {a: xyz[own & (nm == a)] for a in o_names}
        if any(len(v) == 0 for v in o_xyz.values()):
            continue
        cen = np.vstack(list(o_xyz.values())).mean(0)
        o1, o2 = o_xyz[o_names[0]][0], o_xyz[o_names[1]][0]

        # the metal veto — the ONLY use of a metal: to EXCLUDE the residue
        dmet = 99.0
        if len(met_xyz):
            dmet = float(min(np.linalg.norm(met_xyz - o1, axis=1).min(),
                             np.linalg.norm(met_xyz - o2, axis=1).min()))
        if dmet < METAL_CUT:
            continue

        rec = dict(pdb=pdb, chain=key[0], res_id=key[1], res_name=r, truth=truth[key], d_metal=dmet,
                   is_glu=int(r == "GLU"))

        # ── C. density / burial ────────────────────────────────────────────────────────
        d_cen = np.linalg.norm(xyz - cen, axis=1)
        for r_ in (4, 6, 8, 10, 12):
            rec[f"dens{r_}"] = int((d_cen < r_).sum())
        near8 = d_cen < 8
        rec["sc_frac8"] = float(is_sc[near8].mean()) if near8.any() else 0.0
        rec["sasa_sc"] = float(sasa[own & is_sc].sum())
        rec["sasa_o"] = float(sasa[own & np.isin(nm, list(o_names))].sum())

        # ── B. neighbourhood composition ───────────────────────────────────────────────
        for r_ in (6, 8, 10):
            near = d_cen < r_
            seen = {(c, int(i)): rr for c, i, rr in zip(ch[near], ri[near], rn[near])}
            cnt = {c_: 0 for c_ in CLASSES}
            for (c, i), rr in seen.items():
                if (c, i) == key:
                    continue
                cnt[CLASS.get(rr, "aliphatic")] += 1
            for c_ in CLASSES:
                rec[f"n_{c_}{r_}"] = cnt[c_]
            rec[f"n_res{r_}"] = sum(cnt.values())
            rec[f"net_q{r_}"] = cnt["basic"] - cnt["acidic"]

        # ── E. distances (from either carboxyl O) ──────────────────────────────────────
        def near_to(mask, coords=None):
            src = coords if coords is not None else xyz[mask]
            if not len(src):
                return 99.0
            return float(min(np.linalg.norm(src - o1, axis=1).min(),
                             np.linalg.norm(src - o2, axis=1).min()))

        rec["d_dyad"] = near_to(np.isin(nm, list(ALL_CARBOX_O)) & ~own)     # THE dyad — EV5's key term
        rec["d_cation"] = near_to(np.isin(nm, list(CATION_N)))
        rec["d_hydroxyl"] = near_to(np.array([(a, b) in HYDROXYL for a, b in zip(rn, nm)]))
        rec["d_amide"] = near_to(np.array([(a, b) in AMIDE_O for a, b in zip(rn, nm)]))
        rec["d_sulfur"] = near_to(np.array([(a, b) in SULFUR for a, b in zip(rn, nm)]))
        rec["d_hisN"] = near_to(np.array([(a, b) in RING_N for a, b in zip(rn, nm)]))
        rec["d_bbO"] = near_to(np.isin(nm, ["O", "OXT"]) & ~own)
        rec["d_bbN"] = near_to((nm == "N") & ~own)
        rec["d_arom"] = near_to(None, arom) if len(arom) else 99.0
        rec["d_oo"] = float(np.linalg.norm(o1 - o2))                        # the carboxyl's own O–O

        # ── F. geometry / SSE / position ───────────────────────────────────────────────
        rec["sse"] = str(sse[idx]) if (sse is not None and idx < len(sse)) else "c"
        rec["rel_pos"] = float(idx) / max(len(starts), 1)
        rec["chain_len"] = int((np.array([str(prot.chain_id[t]) for t in starts]) == key[0]).sum())

        # ── G. sequence context ────────────────────────────────────────────────────────
        for off in (-2, -1, 1, 2):
            j = idx + off
            rec[f"seq{off:+d}"] = (CLASS.get(str(prot.res_name[starts[j]]), "aliphatic")
                                   if 0 <= j < len(starts) else "none")
        rows.append(rec)
    return rows


df = pd.read_parquet(os.environ.get("PROTON_TRAIN_DF", str(HERE.parent / "data/mpnn_split/train_df_filtered.parquet")))
neut = df[df["method"].str.contains("NEUTRON", na=False)].drop_duplicates("pdb_id")

# Restrict to the PDBs the stage-01/02 dumps actually cover -- see the same block in
# 03_his_rich_features.py. Without it, any neutron PDB beyond stage 01/02's N_PDBS cap reached
# the joins below with no H-bond and no charge row and was sentinel-filled rather than dropped.
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
print(f"{len(neut)} neutron PDBs; dropping any acid with a metal within {METAL_CUT} A of a carboxyl O\n",
      flush=True)

rows, ok = [], 0
for _, r in neut.iterrows():
    try:
        rs = build_one(r["pdb_id"], r["path"])
    except Exception as ex:
        print(f"  [skip] {r['pdb_id']}: {type(ex).__name__}", flush=True)
        continue
    if rs:
        rows += rs
        ok += 1
        if ok % 40 == 0:
            print(f"  {ok} PDBs, {len(rows)} acids", flush=True)

G = pd.DataFrame(rows)

# ── A. per-carboxyl-oxygen H-bond features, from the raw HBPLUS dump ───────────────────
# The OVERRIDE passes matter here and nowhere else: HBPLUS's default chemistry never lets a carboxyl O
# DONATE, so without `-E ASP OD1 1` the donor role -- the only direct evidence of a proton -- cannot exist.
B = pd.read_parquet(DATA / "hbonds.parquet")
hb = B[B.res_name.isin(("ASP", "GLU"))].copy()
hb["pcls"] = [partner_class(a, b) for a, b in zip(hb.p_res_name, hb.p_atom)]
hb["slot"] = np.where(hb.atom.isin(("OD1", "OE1")), "o1", "o2")
G = G.set_index(KEY)
for tag in ("o1", "o2"):
    for role in ("donor", "acceptor"):
        s = hb[(hb.slot == tag) & (hb.role == role)]
        g = s.groupby(KEY).agg(**{f"{tag}_{role}_n": ("dha", "size"),
                                  f"{tag}_{role}_dha": ("dha", "max"),
                                  f"{tag}_{role}_ha": ("ha", "min"),
                                  f"{tag}_{role}_da": ("da", "min")})
        G = G.join(g)
        for pc in PCLS:
            G = G.join(s[s.pcls == pc].groupby(KEY).size().rename(f"{tag}_{role}_{pc}"))
G = G.fillna({c: 0 for c in G.columns if c.endswith(tuple(["_n", "_dha"] + list(PCLS)))})
G = G.fillna({c: 9.0 for c in G.columns if c.endswith(("_ha", "_da"))})

for role in ("donor", "acceptor"):
    a = (G[f"o1_{role}_n"] > 0).astype(int)
    b = (G[f"o2_{role}_n"] > 0).astype(int)
    G[f"n_{role}_atoms"] = a + b                       # how many of the TWO oxygens play this role
    G[f"best_{role}_dha"] = G[[f"o1_{role}_dha", f"o2_{role}_dha"]].max(axis=1)
    G[f"best_{role}_ha"] = G[[f"o1_{role}_ha", f"o2_{role}_ha"]].min(axis=1)
    for pc in PCLS:
        G[f"{role}_{pc}"] = G[f"o1_{role}_{pc}"] + G[f"o2_{role}_{pc}"]
G["n_hbonds"] = G[[c for c in G.columns if c.endswith("_n")]].sum(1)
# the strict COOH signature: one O donates AND the other accepts
G["strict_cooh"] = ((G.n_donor_atoms >= 1) & (G.n_acceptor_atoms >= 1)).astype(int)

# ── D. the fitted charge field, per oxygen ─────────────────────────────────────────────
Cx = pd.read_parquet(DATA / "charge_ctx.parquet")
q = np.where(Cx.n_res_name.isin(("LYS", "ARG")), 1.0,
             np.where(Cx.n_res_name == "HIS", 1.0,
                      np.where(Cx.n_res_name.isin(("ASP", "GLU")), -0.5, 0.0)))
Cx = Cx.assign(slot=np.where(Cx.atom.isin(("OD1", "OE1")), "o1", "o2"))
for rc in (6.0, 8.0):
    v = np.where(Cx.d.values <= rc, q / Cx.d.values, 0.0)
    pa = (pd.DataFrame(dict(pdb=Cx.pdb, chain=Cx.chain, res_id=Cx.res_id, slot=Cx.slot, v=v))
          .groupby(KEY + ["slot"]).v.sum().unstack("slot"))
    if "o1" in pa and "o2" in pa:
        G[f"phi_o1_{rc:.0f}"] = pa["o1"]
        G[f"phi_o2_{rc:.0f}"] = pa["o2"]
        G[f"phi_min_{rc:.0f}"] = pa[["o1", "o2"]].min(axis=1)
        G[f"phi_max_{rc:.0f}"] = pa[["o1", "o2"]].max(axis=1)
        G[f"phi_diff_{rc:.0f}"] = (pa["o1"] - pa["o2"]).abs()
G = G.fillna({c: 0.0 for c in G.columns if c.startswith("phi_")})

G["y"] = G.truth.str.endswith("-P").astype(int)
G.reset_index().to_parquet(OUT)
n_pos_pdb = (G.reset_index().groupby("pdb").y.max() > 0).sum()
print(f"\n=== {ok} PDBs -> {len(G)} acids ({int(G.y.sum())} protonated, {100*G.y.mean():.2f}%) -> {OUT} ===")
print(f"    {G.shape[1] - 4} features")
print(f"    ASP {int((G.res_name == 'ASP').sum())} · GLU {int((G.res_name == 'GLU').sum())}")
print(f"    EFFECTIVE SAMPLE SIZE: {int(G.y.sum())} positives in {n_pos_pdb} PDBs "
      f"(of {G.reset_index().pdb.nunique()}) — the rest are pure negatives")
print(f"    metal-coordinating/adjacent acids EXCLUDED (< {METAL_CUT} A)")
