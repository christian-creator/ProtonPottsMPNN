"""EV5 stage 12 — dump RAW H-bond geometry, so the H-bond criteria can be FITTED rather than assumed.

The complaint against stage 1-11 is fair. What I called "the H-bond evidence" was never raw geometry:
it was `active_donor` / `active_acceptor`, roles ALREADY resolved through v4's own cutoffs (H-A <= 2.5 A,
D-A <= 3.5 A, capability filter). Concluding "H-bonds are uninformative" from one badly-tuned
instantiation is not a conclusion about H-bonds -- it is a conclusion about those cutoffs.

So dump every candidate bond at HBPLUS's most permissive setting and let the thresholds be chosen by the
data. Per bond we keep all three geometric quantities -- the production parser only ever reads two:

    D-A distance  [27:32]      donor heavy atom to acceptor heavy atom
    DHA angle     [46:51]      linearity; a real H-bond is ~160-180 deg, a fake one ~95-120
    H-A distance  [52:57]      <-- NEVER PARSED by bond_annotation.py, and it is the sharpest of the three

Three passes, tagged, so the override can be switched on and off in the sweep:
    default    HBPLUS's own chemistry. A carboxyl O can only ACCEPT (it is not in HBPLUS's donor list),
               so this pass alone can never produce an ASP-P hypothesis.
    od1_oe1    -E ASP OD1 1 -E GLU OE1 1  -> lets OD1/OE1 donate
    od2_oe2    -E ASP OD2 1 -E GLU OE2 1  -> lets OD2/OE2 donate

Protein heavy atoms ONLY -- no waters, no ligands, no metals. A designed protein has none of them, and
neutron structures model D2O explicitly, which would otherwise hand the labeller free water bridges.

Output: sandbox/ev5/hbonds.parquet  (one row per candidate bond at a titratable atom)
"""
import gc, gzip, os, subprocess, sys, tempfile, warnings
gc.disable()
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile

HBPLUS = os.environ.get("HBPLUS_PATH") or sys.exit(
    "set HBPLUS_PATH=/path/to/hbplus (see the README, section 'Getting HBPLUS')")
_PKG = Path(__file__).resolve().parent
DATA = _PKG / "data"; DATA.mkdir(exist_ok=True)
TRAIN_DF = os.environ.get("PROTON_TRAIN_DF", str(_PKG.parent / "data/mpnn_split/train_df_filtered.parquet"))
OUT = DATA / "hbonds.parquet"
N_PDBS = int(sys.argv[1]) if len(sys.argv) > 1 else 250

AA20 = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS",
        "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}
FUNC = {("HIS", "ND1"), ("HIS", "NE2"), ("ASP", "OD1"), ("ASP", "OD2"),
        ("GLU", "OE1"), ("GLU", "OE2")}
# permissive: well past every cutoff we will ever want to test
HA_MAX, DA_MAX, DHA_MIN = 3.2, 4.0, 90.0
PASSES = {
    "default": [],
    "od1_oe1": ["-E", "ASP", " OD1", "1", "-E", "GLU", " OE1", "1"],
    "od2_oe2": ["-E", "ASP", " OD2", "1", "-E", "GLU", " OE2", "1"],
}


def run_pass(pdb_path, tmpdir, chain_map, extra):
    cmd = [HBPLUS, "-h", str(HA_MAX), "-d", str(DA_MAX), "-a", str(DHA_MIN)] + extra
    cmd += [pdb_path, pdb_path]
    subprocess.run(cmd, cwd=tmpdir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    out = []
    for ln in open(pdb_path.replace(".pdb", ".hb2")).readlines()[8:]:
        try:
            d = dict(d_chain=chain_map[ln[0]], d_resi=int(ln[1:5]), d_resn=ln[6:9].strip(),
                     d_atom=ln[9:13].strip(), a_chain=chain_map[ln[14]], a_resi=int(ln[15:19]),
                     a_resn=ln[20:23].strip(), a_atom=ln[23:27].strip(),
                     da=float(ln[27:32]), dha=float(ln[46:51]), ha=float(ln[52:57]))
        except (ValueError, IndexError, KeyError):
            continue
        for v in ("dha", "ha", "da"):                    # HBPLUS writes -1.0 for "undefined"
            if d[v] < 0:
                d[v] = np.nan
        out.append(d)
    return out


def build_one(pdb, path):
    with gzip.open(path, "rt") as fh:
        raw = PDBFile.read(fh).get_structure(model=1)
    heavy = raw[~np.isin(raw.element, ["H", "D"])]
    prot = heavy[np.isin([str(x) for x in heavy.res_name], list(AA20))]
    if not len(prot):
        return []

    # HBPLUS wants single-character chain ids; blanks confuse its parser
    prot = prot.copy()
    prot.chain_id = np.array([(str(c).strip() or "A")[0] for c in prot.chain_id])
    chain_map = {c: c for c in set(prot.chain_id)}

    rows = []
    with tempfile.TemporaryDirectory() as td:
        pp = os.path.join(td, "x.pdb")
        f = PDBFile(); f.set_structure(prot); f.write(pp)
        for mode, extra in PASSES.items():
            for b in run_pass(pp, td, chain_map, extra):
                # keep the bond ONLY if a titratable functional atom is one of its two ends
                for side, oside in (("d", "a"), ("a", "d")):
                    rn, at = b[f"{side}_resn"], b[f"{side}_atom"]
                    if (rn, at) not in FUNC:
                        continue
                    rows.append(dict(
                        pdb=pdb, chain=b[f"{side}_chain"], res_id=b[f"{side}_resi"], res_name=rn,
                        atom=at, role="donor" if side == "d" else "acceptor",
                        p_chain=b[f"{oside}_chain"], p_res_id=b[f"{oside}_resi"],
                        p_res_name=b[f"{oside}_resn"], p_atom=b[f"{oside}_atom"],
                        da=b["da"], dha=b["dha"], ha=b["ha"], mode=mode))
    return rows


df = pd.read_parquet(TRAIN_DF)
neut = df[df["method"].str.contains("NEUTRON", na=False)].drop_duplicates("pdb_id").head(N_PDBS)
print(f"{len(neut)} neutron PDBs -- HBPLUS at -h {HA_MAX} -d {DA_MAX} -a {DHA_MIN}, protein atoms only\n",
      flush=True)

all_rows, ok = [], 0
for _, r in neut.iterrows():
    try:
        rows = build_one(r["pdb_id"], r["path"])
    except Exception as ex:
        print(f"  [skip] {r['pdb_id']}: {type(ex).__name__}: {ex}", flush=True)
        continue
    if rows:
        all_rows += rows
        ok += 1
        if ok % 25 == 0:
            print(f"  {ok} PDBs, {len(all_rows)} bonds", flush=True)

B = pd.DataFrame(all_rows).drop_duplicates(
    subset=["pdb", "chain", "res_id", "atom", "role", "p_chain", "p_res_id", "p_atom", "mode"])
B.to_parquet(OUT)
print(f"\n=== {ok} PDBs -> {len(B)} candidate bonds -> {OUT} ===")
print(B.groupby(["res_name", "role", "mode"]).size().to_string())
print("\ngeometry of the raw pool:")
print(B[["da", "dha", "ha"]].describe().loc[["count", "min", "25%", "50%", "75%", "max"]].to_string())
