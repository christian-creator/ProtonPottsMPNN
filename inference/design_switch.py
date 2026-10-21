"""Switch design with explicit protonation states on both chains (README, "Switch design").

Redesigns the binder of one complex so that it binds in an ``on`` condition and not in an ``off``
condition. A condition names the protonation state of one binder site and one target site, and
the objective weighs the binder's stability, its binding in ``on`` (potency) and the gap between
``on`` and ``off`` (switch). The defaults reproduce the shipped PD-L1 example: the binder His
A:32 and the target Asp B:105 are deprotonated at pH 7.4 (binding on) and protonated at pH 6.5
(binding off). Swap the states or the pH values to design the opposite switch.

    python inference/design_switch.py
    python inference/design_switch.py --binding-model linked_equilibrium --reference-pka propka

PROPKA is opt-in, for two separate things: --states propka labels the structure's other His, Asp and
Glu from PROPKA at --label-ph instead of EV6, and --reference-pka propka (linked_equilibrium only) takes
each named site's pKa from PROPKA on the free partners instead of one constant per residue type. Read the
README section "Protonation source" for what each is worth before using them.

Needs HBPLUS_PATH unless --states propka. Writes the ranked
designs as a table (``--out``) and prints the best ones.
"""
import argparse
import json
import logging
import os
from pathlib import Path

os.environ.pop("DEBUG", None)  # a stray non-boolean DEBUG in your shell breaks environs/foundry

import pandas as pd
from biotite.structure.io.pdb import PDBFile

from mpnn.inference_engines.potts_mpnn_ph import PottsMPNNPHEngine

PKG = Path(__file__).resolve().parent.parent
CKPT = PKG / "checkpoints" / "potts_v6_afdb_edge_his0.3_acid0.06" / "epoch-0125.ckpt"
EXAMPLES = PKG / "inference" / "examples"
META = json.loads((EXAMPLES / "example_meta.json").read_text())
EXAMPLE_SITES = {"binder": "A:32", "target": "B:105"}  # the closest titratable pair across the interface


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pdb", type=Path, default=EXAMPLES / META["pdb"], help="complex to design (PDB)")
    p.add_argument("--binder-chain", default=META["binder_chain"])
    p.add_argument("--binder-site", help="titratable binder residue, like A:32 (default for the example)")
    p.add_argument("--target-site", help="titratable target residue, like B:105 (default for the example)")
    p.add_argument("--ph-on", type=float, default=7.4)
    p.add_argument("--ph-off", type=float, default=6.5)
    p.add_argument("--on-state", choices=["protonated", "deprotonated"], default="deprotonated")
    p.add_argument("--off-state", choices=["protonated", "deprotonated"], default="protonated")
    p.add_argument("--binding-model", default="state_binding",
                   choices=["state_binding", "complex_gap", "linked_equilibrium"])
    p.add_argument("--states", default="ev6", choices=["ev6", "propka"],
                   help="where the starting states of residues no condition names come from")
    p.add_argument("--label-ph", type=float, help="pH of the PROPKA states (default: --ph-on)")
    p.add_argument("--label-margin", type=float, default=0.5,
                   help="PROPKA states within this many pH units of the pKa are ambiguous")
    p.add_argument("--reference-pka", default="constant", choices=["constant", "propka"],
                   help="pKa of each named site in linked_equilibrium: one constant per residue type, or PROPKA")
    p.add_argument("--weights", type=float, nargs=3, default=[0.3, 0.3, 0.4],
                   metavar=("STABILITY", "POTENCY", "SWITCH"))
    p.add_argument("--near-k", type=int, default=16, help="binder neighbours of the target site that may change")
    p.add_argument("--max-mutations", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--top", type=int, default=5, help="designs to print")
    p.add_argument("--out", type=Path, default=PKG / "inference" / "outputs" / "switch_designs.tsv")
    return p.parse_args(argv)


def build_config(args):
    binder_site, target_site = args.binder_site, args.target_site
    if args.pdb.resolve() == (EXAMPLES / META["pdb"]).resolve():
        binder_site = binder_site or EXAMPLE_SITES["binder"]
        target_site = target_site or EXAMPLE_SITES["target"]
    if not (binder_site and target_site):
        raise SystemExit("give --binder-site and --target-site (residues like A:32) for your own complex")
    stability, potency, switch = args.weights
    config = {
        "conditions": {
            "on": {"ph": args.ph_on, "states": {binder_site: args.on_state, target_site: args.on_state}},
            "off": {"ph": args.ph_off, "states": {binder_site: args.off_state, target_site: args.off_state}},
        },
        "terms": {"stability": {"weight": stability}, "potency": {"weight": potency},
                  "switch": {"weight": switch, "off_margin": 2.0}},
        "designable": {"chains": [args.binder_chain],
                       "near": {"sites": [target_site], "k": args.near_k, "max_mutations": args.max_mutations}},
        "binding_model": args.binding_model,
        "search": "single",
        "base_seed": args.seed,
    }
    protonation = {}
    if args.states == "propka":
        protonation |= {"states": "propka", "ph": args.label_ph if args.label_ph is not None else args.ph_on,
                        "margin": args.label_margin}
    if args.reference_pka == "propka":
        protonation["reference_pka"] = "propka"
    if protonation:
        config["protonation"] = protonation
    return config


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("mpnn").setLevel(logging.INFO)  # which protonation source is used, and where each pKa comes from
    engine = PottsMPNNPHEngine(checkpoint_path=str(CKPT), extended_vocab="v6", device=args.device,
                               out_directory=None, write_fasta=False, write_structures=False)
    atom_array = PDBFile.read(str(args.pdb)).get_structure(model=1)
    run = engine.run_switch_design(atom_array=atom_array, binder_chain=args.binder_chain,
                                   config=build_config(args), seed=args.seed)
    table = pd.DataFrame(run.to_rows(args.binder_chain))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, sep="\t", index=False)
    columns = ["rank", "n_mutations"] + [c for c in table.columns if c.startswith("term_")]
    print(table[columns].head(args.top).to_string(index=False))
    print(f"\nbest design: {table.loc[0, 'canonical_sequence']}\nwrote {len(table)} design(s) to {args.out}")


if __name__ == "__main__":
    main()
