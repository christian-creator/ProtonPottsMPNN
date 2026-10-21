"""PROPKA against experiment, for the two things the design engine can take from it.

  pkad     REFERENCE pKA (linked_equilibrium, protonation source "propka"): PROPKA's pKa of every His,
           Asp and Glu of PKAD (experimental pKa values in monomeric proteins) next to the constant per
           residue type that linked_equilibrium uses otherwise. Reads the PROPKA values that ship in
           data/PKAD/pkad_propka_results.csv; fast, needs no HBPLUS.
  neutron  STARTING STATES (protonation source "propka"): PROPKA's pKa read at a pH, against the
           experimental protonation state of every His/Asp/Glu of the neutron benchmark, next to EV6,
           which was trained on these structures (an upper bound, not a fair rival). Runs the pipeline
           on 226 structures: needs HBPLUS_PATH and CCD_MIRROR_PATH, minutes with --n-jobs 10.

    python benchmarks/propka_vs_experiment.py pkad
    python benchmarks/propka_vs_experiment.py neutron --n-jobs 10 --cache /path/to/neutron_labels.json
"""

import argparse
import csv
import json
import math
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

DATA = Path(__file__).resolve().parent / "data"
PKAD_CSV = DATA / "PKAD" / "pkad_propka_results.csv"
NEUTRON = DATA / "neutron_benchmark"
CONSTANT_PKA = {"HIS": 6.5, "ASP": 3.7, "GLU": 4.3}  # mpnn.ph.binding.DEFAULT_MODEL_PKA


def finite(text):
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


# ---- reference pKa: PKAD ---------------------------------------------------------------------
def run_pkad(args):
    rows = []
    for row in csv.DictReader(open(PKAD_CSV, encoding="utf-8-sig")):
        experimental, propka = finite(row["Expt. pKa"]), finite(row["propka_pka"])
        if row["ResName"] in CONSTANT_PKA and experimental is not None and propka is not None:
            rows.append((row["ResName"], experimental, propka))
    print(f"His/Asp/Glu with an experimental pKa and a PROPKA value: {len(rows)}\n")
    print("type    n | MAE constant  PROPKA | RMSE constant  PROPKA | PROPKA vs experiment: Pearson  Spearman")
    for name in CONSTANT_PKA:
        exp = np.array([r[1] for r in rows if r[0] == name])
        pka = np.array([r[2] for r in rows if r[0] == name])
        const = np.full_like(exp, CONSTANT_PKA[name])
        print(
            f"{name}  {len(exp):4d} |     {np.abs(const - exp).mean():5.2f}   {np.abs(pka - exp).mean():5.2f} |"
            f"      {np.sqrt(((const - exp) ** 2).mean()):5.2f}   {np.sqrt(((pka - exp) ** 2).mean()):5.2f} |"
            f"              {pearsonr(pka, exp)[0]:5.2f}     {spearmanr(pka, exp)[0]:5.2f}"
        )


# ---- starting states: neutron structures -----------------------------------------------------
def label_structure(pdb):
    """EV6 label and PROPKA pKa of every His/Asp/Glu of one neutron structure."""
    try:
        from mpnn.potts_inference import prepare_potts_input
        from mpnn.transforms.propka_labeller import propka_pka

        atoms = prepare_potts_input(str(NEUTRON / "heavy" / pdb), extended_vocab="v6")["atom_array"]
        pka = propka_pka(atoms)
        rows, seen = [], set()
        for chain, res_id, name, label in zip(
            atoms.chain_id, atoms.res_id, atoms.res_name, atoms.protonation_label
        ):
            key = (str(chain), int(res_id), str(name))
            if name in CONSTANT_PKA and key not in seen:
                seen.add(key)
                rows.append([pdb[:-4], *key, str(label), pka.get(key)])
        return rows, None
    except Exception as err:  # a structure the pipeline cannot read is reported, not fatal
        return [], f"{pdb}: {type(err).__name__}: {str(err)[:100]}"


def label_all(n_jobs, limit):
    pdbs = sorted(p.name for p in (NEUTRON / "heavy").glob("*.pdb"))[:limit]
    with Pool(n_jobs) as pool:
        results = pool.map(label_structure, pdbs)
    return {"rows": [r for rows, _ in results for r in rows], "errors": [e for _, e in results if e]}


def score(name, called_protonated, called_ambiguous, truth):
    confident = ~called_ambiguous
    hits = (called_protonated & truth & confident).sum()
    accuracy = ((called_protonated == truth) & confident).sum() / max(confident.sum(), 1)
    print(
        f"  {name:24s} coverage {confident.mean():6.1%}  accuracy {accuracy:6.1%}  protonated: "
        f"precision {hits / max((called_protonated & confident).sum(), 1):6.1%}  recall {hits / max(truth.sum(), 1):6.1%}"
    )


def run_neutron(args):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    if args.cache and Path(args.cache).exists():
        labelled = json.load(open(args.cache))
    else:
        labelled = label_all(args.n_jobs, args.limit)
        if args.cache:
            json.dump(labelled, open(args.cache, "w"))
    truth = {
        (r["pdb_id"], r["chain"], int(r["res_id"]), r["res_name"]): r["truth"]
        for r in csv.DictReader(open(NEUTRON / "truth.csv"))
    }
    data = [
        (row[3], truth[tuple(row[:3]) + (row[3],)].endswith("-P"), row[4], row[5])
        for row in ((r[0], r[1], r[2], r[3], r[4], r[5]) for r in labelled["rows"])
        if (row[0], row[1], row[2], row[3]) in truth and row[5] is not None
    ]
    print(f"{len(data)} residues with an experimental state, an EV6 label and a PROPKA pKa; "
          f"{len(labelled['errors'])} structures the pipeline could not read\n")
    for group, names in (("His", {"HIS"}), ("Asp/Glu", {"ASP", "GLU"})):
        items = [d for d in data if d[0] in names]
        y = np.array([d[1] for d in items])
        pka = np.array([d[3] for d in items])
        print(f"{group}: n={len(items)}, {y.mean():.1%} protonated in the experiments")
        print(f"  PROPKA pKa as a score: ROC AUC {roc_auc_score(y, pka):.3f}, average precision "
              f"{average_precision_score(y, pka):.3f} (chance {y.mean():.3f})")
        ev6 = [d[2] for d in items]
        score("EV6 (trained on these)", np.array([t.endswith("-P") for t in ev6]),
              np.array([t.endswith("-A") for t in ev6]), y)
        for ph in args.ph:
            for margin in args.margin:
                score(f"PROPKA pH {ph:g} margin {margin:g}", pka > ph, np.abs(pka - ph) < margin, y)
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="benchmark", required=True)
    sub.add_parser("pkad").set_defaults(run=run_pkad)
    neutron = sub.add_parser("neutron")
    neutron.set_defaults(run=run_neutron)
    neutron.add_argument("--ph", type=float, nargs="+", default=[5.5, 6.5, 7.0, 7.5])
    neutron.add_argument("--margin", type=float, nargs="+", default=[0.0, 0.5])
    neutron.add_argument("--n-jobs", type=int, default=4)
    neutron.add_argument("--limit", type=int, help="first N structures only (a quick check)")
    neutron.add_argument("--cache", help="JSON file: reuse it if present, else write the labelled residues there")
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
