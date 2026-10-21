"""Switch design on the shipped PD-L1 example, through the real parser and checkpoint.

Everything else in ``tests/ph`` runs on synthetic Potts tables. These tests run the engine
as a user would: ``inference/examples/pdl1_seed_binder.pdb`` (binder chain A, PD-L1 on chain
B), the packaged v6 checkpoint, the structure pipeline and the protonation labeller.

They need the full stack (atomworks, biotite) and an HBPLUS executable (``HBPLUS_PATH``);
each test is skipped when one is missing. They run on CPU, and one compares with CUDA when
it is available.

    cd foundry/models/mpnn
    HBPLUS_PATH=... PYTHONPATH=src \
        python -m pytest tests/ph/test_real_structure.py --confcutdir=tests/ph -v

The assertions hold for any HBPLUS output, so they check how the pieces are wired together
(position mapping, determinism, design space), not the biology.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("atomworks")
pytest.importorskip("biotite")

REPO = Path(__file__).resolve().parents[5]
EXAMPLES = REPO / "inference" / "examples"
CHECKPOINT = (
    REPO / "checkpoints" / "potts_v6_afdb_edge_his0.3_acid0.06" / "epoch-0125.ckpt"
)
META = json.loads((EXAMPLES / "example_meta.json").read_text())
BINDER = META["binder_chain"]

# The titratable pair closest across the interface in the example (CA-CA 9.4 A).
BINDER_SITE, BINDER_RESIDUE = "A:32", "HIS"
TARGET_SITE, TARGET_RESIDUE = "B:105", "ASP"

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
    "terms": {
        "stability": {"weight": 0.3},
        "potency": {"weight": 0.3},
        "switch": {"weight": 0.4, "off_margin": 2.0},
    },
    "designable": {
        "chains": [BINDER],
        "near": {"sites": [TARGET_SITE], "k": 16, "max_mutations": 20},
    },
    "binding_model": "state_binding",
    "search": "single",
    "block_size": 2,
    "max_rounds": 3,
}


def _missing_requirement():
    hbplus = os.environ.get("HBPLUS_PATH")
    if not hbplus or not Path(hbplus).is_file():
        return "HBPLUS_PATH does not point at the hbplus executable"
    if not CHECKPOINT.is_file() or not (EXAMPLES / META["pdb"]).is_file():
        return "the packaged checkpoint or example structure is missing"
    return None


pytestmark = pytest.mark.skipif(
    _missing_requirement() is not None, reason=str(_missing_requirement())
)


def make_engine(device):
    from mpnn.inference_engines.potts_mpnn_ph import PottsMPNNPHEngine

    return PottsMPNNPHEngine(
        checkpoint_path=str(CHECKPOINT),
        extended_vocab="v6",
        device=device,
        out_directory=None,
        write_fasta=False,
        write_structures=False,
    )


@pytest.fixture(scope="module")
def atoms():
    from biotite.structure.io.pdb import PDBFile

    structure = PDBFile.read(str(EXAMPLES / META["pdb"])).get_structure(model=1)
    for site, residue in ((BINDER_SITE, BINDER_RESIDUE), (TARGET_SITE, TARGET_RESIDUE)):
        chain, res_id = site.split(":")
        names = structure.res_name[
            (structure.chain_id == chain) & (structure.res_id == int(res_id))
        ]
        assert set(names) == {residue}, f"{site} is no longer a {residue}"
    return structure


@pytest.fixture(scope="module")
def engine():
    return make_engine("cpu")


@pytest.fixture(scope="module")
def run(engine, atoms):
    return engine.run_switch_design(
        atom_array=atoms, binder_chain=BINDER, config=CONFIG, seed=0
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


def sequences(design_run):
    return [row["canonical_sequence"] for row in design_run.to_rows(BINDER)]


def test_the_context_covers_the_example_complex(engine, atoms):
    ctx = engine._build_context(atoms, BINDER)
    assert ctx.L == META["binder_len"] + META["target_len"]
    assert int(ctx.chA_free_idx.numel()) == META["binder_len"]


def test_state_binding_ranks_designs_like_compute_g_bind(engine, atoms, run):
    """The binder-alone and receptor-alone energies are mapped onto the right residues.

    ``compute_g_bind`` re-encodes the isolated chains on its own code path and counts each
    pair twice, so the two differ by a factor of about two. Random designs on the fixed
    backbone must still be ordered alike; a mis-mapped residue would break that.
    """
    inputs = engine._design_inputs(engine._build_context(atoms, BINDER), BINDER)
    allowed = [i for i, ok in enumerate(inputs.table.design_mask([])) if ok]
    binder_order = sorted(inputs.binder_positions, key=lambda p: int(inputs.res_ids[p]))
    rng = np.random.default_rng(0)
    designs, name_sets = [], []
    for _ in range(8):
        tokens = inputs.tokens.clone()
        for position in rng.choice(binder_order, size=12, replace=False):
            tokens[position] = int(rng.choice(allowed))
        designs.append(tokens)
        name_sets.append([inputs.table.names[int(tokens[p])] for p in binder_order])
    binding = run.objective.ctx.binding
    ours = np.array([binding.energy(tokens) for tokens in designs])
    reference = np.array(
        [
            row["g_bind"]
            for row in engine.compute_g_bind_batch(
                atoms, BINDER, binder_token_sets=name_sets
            )
        ]
    )
    assert np.corrcoef(ours, reference)[0, 1] > 0.99
    assert np.argsort(ours).tolist() == np.argsort(reference).tolist()


def test_a_run_is_reproducible_and_stays_in_the_design_space(engine, atoms, run):
    again = engine.run_switch_design(
        atom_array=atoms, binder_chain=BINDER, config=CONFIG, seed=0
    )
    assert sequences(again) == sequences(run)
    inputs, best = run.inputs, run.result.records[0]
    designed = set(run.designable)
    changed = {i for i in range(len(best.tokens)) if best.tokens[i] != inputs.tokens[i]}
    assert changed <= designed
    allowed = torch.tensor(inputs.table.design_mask([]), dtype=torch.bool)
    assert allowed[best.tokens[sorted(designed)]].all()
    site = inputs.res_ids.index(int(BINDER_SITE.split(":")[1]))
    assert site not in designed  # a residue named in a condition is never redesigned


def test_an_ensemble_of_one_complex_twice_equals_the_single_run(engine, atoms, run):
    ensemble = engine.run_switch_design_ensemble(
        structures={"first": atoms, "second": atoms},
        binder_chain=BINDER,
        config=CONFIG,
        seed=0,
    )
    assert sequences(ensemble) == sequences(run)
    for row, single in zip(ensemble.to_rows(BINDER), run.to_rows(BINDER)):
        for term in CONFIG["terms"]:
            assert row[f"term_{term}"] == pytest.approx(single[f"term_{term}"])
            assert row[f"term_{term}@first"] == row[f"term_{term}@second"]


def test_the_site_report_has_a_row_per_condition_and_site(run):
    rows = run.site_report(run.result.records[0])
    assert len(rows) == 4
    imposed = {(r["condition"], r["site"]): r["state_token"] for r in rows}
    assert imposed == {
        ("on", BINDER_SITE): "HIS-S",
        ("off", BINDER_SITE): "HIS-P",
        ("on", TARGET_SITE): "ASP-D",
        ("off", TARGET_SITE): "ASP-P",
    }
    for row in rows:
        assert row["set_by_condition"]
        assert np.isfinite(row["gap_complex"]) and np.isfinite(row["gap_free"])


def test_propka_reference_pka_can_be_used_with_ev6_states(engine, atoms, run):
    pytest.importorskip("propka")
    from mpnn.transforms.propka_labeller import propka_site_pka

    config = {
        **CONFIG,
        "binding_model": "linked_equilibrium",
        "protonation": {"reference_pka": "propka"},
    }
    linked = engine.run_switch_design(
        atom_array=atoms, binder_chain=BINDER, config=config, seed=0
    )
    assert torch.equal(
        linked.inputs.tokens, run.inputs.tokens
    )  # the starting states are still EV6's
    chain_of = np.array([str(c) for c in atoms.chain_id])
    free = {c: propka_site_pka(atoms[chain_of == c]) for c in ("A", "B")}
    chosen = {
        (
            str(linked.inputs.chain_ids[s.position]),
            int(linked.inputs.res_ids[s.position]),
        ): s.pka
        for s in linked.objective.ctx.binding.sites
    }
    assert chosen == {
        ("A", 32): pytest.approx(free["A"][("A", 32)]),
        ("B", 105): pytest.approx(free["B"][("B", 105)]),
    }


def test_linked_equilibrium_runs_on_the_real_complex(engine, atoms):
    config = {**CONFIG, "binding_model": "linked_equilibrium"}
    linked = engine.run_switch_design(
        atom_array=atoms, binder_chain=BINDER, config=config, seed=0
    )
    best = linked.result.records[0]
    assert all(np.isfinite(value) for value in best.term_values.values())
    assert {row["site"] for row in linked.site_report(best)} == {
        BINDER_SITE,
        TARGET_SITE,
    }


def test_the_original_redesign_returns_the_same_designs_for_any_n_jobs(engine, atoms):
    from mpnn.inference_engines.potts_mpnn_ph import PHDesignCriteria

    criteria = PHDesignCriteria(
        method="block_descent",
        backend="potts",
        temperature=0.05,
        samples_per_site=2,
        block_size=3,
        combined_lambda=0.3,
        seed_source="native",
        center_types=["HIS-P", "ASP-P", "GLU-P"],
        dep_map={"HIS-P": ["HIS-S"], "ASP-P": ["ASP-D"], "GLU-P": ["GLU-D"]},
        forbidden_tokens=["HIS-A", "ASP-A", "GLU-A", "UNK"],
        placement_by="scan_potts",
        neighbour_k=16,
        max_mutations=20,
    )

    def designs(n_jobs):
        found = engine.run_ph_redesign(
            atom_array=atoms,
            binder_chain=BINDER,
            criteria_list=[criteria],
            seed=0,
            n_jobs=n_jobs,
        )
        return [(d.canonical_sequence, d.center_res_ids) for d in found]

    serial = designs(1)
    assert serial and designs(2) == serial


def test_the_example_script_designs_the_shipped_complex(tmp_path):
    table, printed = run_example_script(tmp_path, "--seed", "0")
    assert len(table) >= 1 and "best design" in printed
    assert {
        "rank",
        "canonical_sequence",
        "term_stability",
        "term_potency",
        "term_switch",
    } <= set(table.columns)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_cuda_gives_the_same_design_values_as_cpu(atoms, run):
    on_gpu = make_engine("cuda").run_switch_design(
        atom_array=atoms, binder_chain=BINDER, config=CONFIG, seed=0
    )
    cpu_best, gpu_best = run.result.records[0], on_gpu.result.records[0]
    for term, value in cpu_best.term_values.items():
        assert gpu_best.term_values[term] == pytest.approx(value, rel=1e-2, abs=0.5)
