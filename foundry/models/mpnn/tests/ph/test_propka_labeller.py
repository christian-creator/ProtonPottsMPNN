"""PROPKA as a source of protonation labels (needs atomworks and propka for most tests)."""

import importlib.util
import logging
import math
import sys
from pathlib import Path

import pytest
from engine_imports import engine_module

# Decided at import, before any stub exists: the stubs of engine_module() would make a missing
# package look installed.
INSTALLED = {
    name: importlib.util.find_spec(name) is not None
    for name in ("biotite", "atomworks", "propka")
}


def require(*names):
    missing = [n for n in names if not INSTALLED[n]]
    if missing:
        pytest.skip(f"needs {', '.join(missing)}")


REPO = Path(__file__).resolve().parents[5]
EXAMPLE = REPO / "inference" / "examples" / "pdl1_seed_binder.pdb"
HIS_32 = ("A", 32, "HIS")  # the binder's histidine in the example


@pytest.fixture(scope="module")
def labeller_module():
    with engine_module():
        from mpnn.transforms import propka_labeller

        yield propka_labeller


@pytest.fixture(scope="module")
def atoms():
    require("biotite")
    from biotite.structure.io.pdb import PDBFile

    return PDBFile.read(str(EXAMPLE)).get_structure(model=1)


@pytest.mark.parametrize(
    "res_name, pka, ph, margin, expected",
    [
        ("HIS", 6.5, 7.4, 0.5, "HIS-S"),
        ("HIS", 6.5, 6.0, 0.5, "HIS-P"),
        ("ASP", 3.8, 7.4, 0.5, "ASP-D"),
        ("GLU", 6.7, 6.5, 0.5, "GLU-A"),  # within the margin of the pH
        ("GLU", 6.7, 6.5, 0.0, "GLU-P"),  # no ambiguous class
        ("ASP", 5.0, 5.0, 0.0, "ASP-D"),  # a tie goes to the neutral state
        ("HIS", math.nan, 7.4, 0.5, "HIS-A"),  # PROPKA gave no value
    ],
)
def test_the_token_follows_the_pka_and_the_ph(
    labeller_module, res_name, pka, ph, margin, expected
):
    assert labeller_module.token_from_pka(res_name, pka, ph, margin) == expected


def test_a_labeller_can_key_a_cache_and_refuses_nonsense(labeller_module):
    PropkaLabeller = labeller_module.PropkaLabeller
    assert PropkaLabeller(7.4) == PropkaLabeller(7.4, 0.5)
    assert len({PropkaLabeller(7.4), PropkaLabeller(7.4), PropkaLabeller(6.0)}) == 2
    with pytest.raises(ValueError, match="finite pH"):
        PropkaLabeller(math.nan)
    with pytest.raises(ValueError, match="margin"):
        PropkaLabeller(7.0, margin=-1.0)


def test_a_missing_propka_says_how_to_install_it(labeller_module, atoms, monkeypatch):
    monkeypatch.setitem(sys.modules, "propka", None)
    monkeypatch.setitem(sys.modules, "propka.run", None)
    with pytest.raises(ImportError, match="pip install propka"):
        labeller_module.propka_pka(atoms)


@pytest.fixture(scope="module")
def pka(labeller_module, atoms):
    require("propka")
    return labeller_module.propka_pka(atoms)


def test_propka_scores_every_side_chain_and_nothing_else(pka, atoms):
    wanted = {
        (str(c), int(r), str(n))
        for c, r, n in zip(atoms.chain_id, atoms.res_id, atoms.res_name)
        if n in ("HIS", "ASP", "GLU")
    }
    assert set(pka) == wanted  # no C-terminal carboxylate, no other residue type
    assert all(0.0 < value < 14.0 for value in pka.values())
    assert pka[HIS_32] < 6.5  # buried: well below the 6.5 of a His in water


def test_the_labels_move_with_the_ph(labeller_module, atoms):
    PropkaLabeller = labeller_module.PropkaLabeller

    def count(ph, suffix):
        tokens = PropkaLabeller(ph, margin=0.0)(atoms).values()
        return sum(t.endswith(suffix) for t in tokens)

    counts = [count(ph, "-P") for ph in (0.0, 3.0, 5.0, 7.4, 14.0)]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] == len(PropkaLabeller(0.0)(atoms)) and counts[-1] == 0
    assert set(PropkaLabeller(7.4)(atoms)) == set(PropkaLabeller(3.0)(atoms))


def test_the_labeller_replaces_the_vocabulary_classifier(atoms):
    require("atomworks")
    from mpnn.transforms.vocab_annotation import AnnotateProtonationStates

    seen = {}

    def fixed(atom_array, **options):
        seen.update(options)
        return {HIS_32: "HIS-P"}

    transform = AnnotateProtonationStates(extended_vocab="v6", classify=fixed)
    out = transform.forward({"atom_array": atoms.copy()})["atom_array"]
    his = (out.chain_id == "A") & (out.res_id == 32)
    assert set(out.protonation_label[his]) == {"HIS-P"}
    assert "hbond_records" in seen  # the vocabulary's call is unchanged


def test_the_labeller_logs_what_it_did_and_warns_once(
    labeller_module, atoms, caplog, monkeypatch
):
    require("propka")
    monkeypatch.setattr(labeller_module, "_caveat_shown", False)
    labeller = labeller_module.PropkaLabeller(7.4)
    with caplog.at_level(logging.INFO, logger=labeller_module.logger.name):
        labeller(atoms)
        labeller(atoms)
    ours = [
        r for r in caplog.records if r.name == labeller_module.logger.name
    ]  # not PROPKA's own
    warnings = [r.getMessage() for r in ours if r.levelno == logging.WARNING]
    assert (
        len(warnings) == 1
        and "trained on EV6 labels" in warnings[0]
        and "0.62" in warnings[0]
    )
    summaries = [r.getMessage() for r in ours if r.levelno == logging.INFO]
    assert len(summaries) == 2
    assert "pH 7.40" in summaries[0] and "HIS 3 neutral" in summaries[0]
