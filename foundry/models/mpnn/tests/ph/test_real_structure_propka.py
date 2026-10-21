"""Switch design with PROPKA for the starting states and for the reference pKa, on the shipped PD-L1 complex.

The PROPKA source labels residues from pKa values and skips the H-bond passes, so these tests
run without HBPLUS (they remove it from the environment). They need atomworks, propka and the
packaged checkpoint; each test is skipped when one is missing.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("atomworks")
pytest.importorskip("biotite")
pytest.importorskip("propka")

REPO = Path(__file__).resolve().parents[5]
EXAMPLES = REPO / "inference" / "examples"
CHECKPOINT = (
    REPO / "checkpoints" / "potts_v6_afdb_edge_his0.3_acid0.06" / "epoch-0125.ckpt"
)
META = json.loads((EXAMPLES / "example_meta.json").read_text())
BINDER = META["binder_chain"]
BINDER_SITE, TARGET_SITE = "A:32", "B:105"
PROPKA = {
    "states": "propka",
    "ph": 7.4,
}  # starting states from PROPKA, constant reference pKa

CONFIG = {
    "conditions": {
        "on": {
            "ph": 7.4,
            "states": {BINDER_SITE: "deprotonated", TARGET_SITE: "deprotonated"},
        },
        "off": {
            "ph": 6.5,
            "states": {BINDER_SITE: "protonated", TARGET_SITE: "protonated"},
        },
    },
    "terms": {"stability": {"weight": 0.5}, "potency": {"weight": 0.5}},
    "designable": {
        "chains": [BINDER],
        "near": {"sites": [TARGET_SITE], "k": 16, "max_mutations": 8},
    },
    "binding_model": "linked_equilibrium",
    "protonation": {**PROPKA, "reference_pka": "propka"},
    "search": "single",
    "block_size": 2,
    "max_rounds": 2,
}


def _missing_requirement():
    if not CHECKPOINT.is_file() or not (EXAMPLES / META["pdb"]).is_file():
        return "the packaged checkpoint or example structure is missing"
    return None


pytestmark = pytest.mark.skipif(
    _missing_requirement() is not None, reason=str(_missing_requirement())
)


@pytest.fixture(autouse=True)
def no_hbplus(monkeypatch):
    monkeypatch.delenv("HBPLUS_PATH", raising=False)


@pytest.fixture(scope="module")
def atoms():
    from biotite.structure.io.pdb import PDBFile

    return PDBFile.read(str(EXAMPLES / META["pdb"])).get_structure(model=1)


@pytest.fixture(scope="module")
def engine():
    from mpnn.inference_engines.potts_mpnn_ph import PottsMPNNPHEngine

    return PottsMPNNPHEngine(
        checkpoint_path=str(CHECKPOINT),
        extended_vocab="v6",
        device="cpu",
        out_directory=None,
        write_fasta=False,
        write_structures=False,
    )


def run_example_script(tmp_path, *flags):
    """inference/design_switch.py as a user runs it: its table and what it printed."""
    import pandas as pd

    out = tmp_path / "designs.tsv"
    done = subprocess.run(
        [
            sys.executable,
            str(REPO / "inference" / "design_switch.py"),
            "--out",
            str(out),
            *flags,
        ],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert done.returncode == 0, done.stderr[-1500:]
    return pd.read_csv(out, sep="\t"), done.stdout


def labelled(ctx, tokens_by_residue):
    """The context's native token at each His/Asp/Glu, by (chain, res_id, res_name)."""
    names = ctx.encoding.idx_to_token
    token_aa = ctx.token_aa
    found = {}
    for i, (chain, res_id, res_name) in enumerate(
        zip(token_aa.chain_id, token_aa.res_id, token_aa.res_name)
    ):
        key = (str(chain), int(res_id), str(res_name))
        if key in tokens_by_residue:
            found[key] = str(names[int(ctx.S_native[i])])
    return found


def test_the_context_carries_propka_labels_and_needs_no_hbplus(engine, atoms):
    from mpnn.transforms.propka_labeller import PropkaLabeller

    for ph in (7.4, 3.0):
        expected = PropkaLabeller(ph)(atoms)
        ctx = engine._build_context(
            atoms, BINDER, protonation={"states": "propka", "ph": ph}
        )
        assert labelled(ctx, expected) == expected
    # the pipeline is cached per labeller: the two pH values gave different tokens
    low = engine._build_context(
        atoms, BINDER, protonation={"states": "propka", "ph": 3.0}
    )
    high = engine._build_context(atoms, BINDER, protonation=PROPKA)
    assert not np.array_equal(low.S_native.numpy(), high.S_native.numpy())


def test_the_linked_equilibrium_model_uses_the_pka_of_the_free_partners(engine, atoms):
    from mpnn.transforms.propka_labeller import propka_pka, propka_site_pka

    run = engine.run_switch_design(
        atom_array=atoms, binder_chain=BINDER, config=CONFIG, seed=0
    )
    chain_of = np.array([str(c) for c in atoms.chain_id])
    free = {
        "A": propka_site_pka(atoms[chain_of == "A"]),
        "B": propka_site_pka(atoms[chain_of == "B"]),
    }
    chosen = {
        (
            str(run.inputs.chain_ids[s.position]),
            int(run.inputs.res_ids[s.position]),
        ): s.pka
        for s in run.objective.ctx.binding.sites
    }
    assert chosen == {
        ("A", 32): pytest.approx(free["A"][("A", 32)]),
        ("B", 105): pytest.approx(free["B"][("B", 105)]),
    }
    # the per-site values carry the structure: the binder His is not the 6.5 of a His in water
    assert abs(chosen[("A", 32)] - 6.5) > 0.5
    # and they are those of the FREE partner: inside the complex PROPKA gives this His another pKa
    assert abs(chosen[("A", 32)] - propka_pka(atoms)[("A", 32, "HIS")]) > 1.0
    report = run.site_report(run.result.records[0])
    assert {row["site"] for row in report} == {BINDER_SITE, TARGET_SITE}
    assert all(np.isfinite(row["apparent_pka_free"]) for row in report)


def test_run_ph_redesign_takes_the_propka_source_too(engine, atoms):
    from mpnn.inference_engines.potts_mpnn_ph import PHDesignCriteria

    criteria = PHDesignCriteria(
        method="block_descent",
        backend="potts",
        temperature=0.05,
        samples_per_site=1,
        block_size=3,
        combined_lambda=0.3,
        seed_source="native",
        center_types=["HIS-P", "ASP-P", "GLU-P"],
        dep_map={"HIS-P": ["HIS-S"], "ASP-P": ["ASP-D"], "GLU-P": ["GLU-D"]},
        forbidden_tokens=["HIS-A", "ASP-A", "GLU-A", "UNK"],
        placement_by="scan_potts",
        neighbour_k=8,
        max_mutations=8,
    )
    found = engine.run_ph_redesign(
        atom_array=atoms,
        binder_chain=BINDER,
        criteria_list=[criteria],
        seed=0,
        protonation=PROPKA,
    )
    assert len(found) >= 1


def test_the_example_script_takes_the_propka_source(tmp_path):
    table, printed = run_example_script(
        tmp_path,
        "--states",
        "propka",
        "--reference-pka",
        "propka",
        "--binding-model",
        "linked_equilibrium",
    )
    assert len(table) >= 1 and "best design" in printed
    assert {
        "rank",
        "canonical_sequence",
        "term_stability",
        "term_potency",
        "term_switch",
    } <= set(table.columns)


def test_propka_states_alone_leave_the_reference_pka_at_the_constants(engine, atoms):
    from mpnn.ph.binding import DEFAULT_MODEL_PKA

    config = {**CONFIG, "protonation": PROPKA}
    run = engine.run_switch_design(
        atom_array=atoms, binder_chain=BINDER, config=config, seed=0
    )
    assert sorted(site.pka for site in run.objective.ctx.binding.sites) == sorted(
        [DEFAULT_MODEL_PKA["HIS"], DEFAULT_MODEL_PKA["ASP"]]
    )
