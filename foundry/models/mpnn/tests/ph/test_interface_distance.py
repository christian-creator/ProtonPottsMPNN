"""The interface cutoff is a criteria field and region masks follow it."""

from pathlib import Path

import pytest
import torch
from engine_fixtures import make_ctx

ENGINE_FILE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "mpnn"
    / "inference_engines"
    / "potts_mpnn_ph.py"
)


def test_interface_distance_default_override_and_validation(engine):
    assert engine.PHDesignCriteria().interface_distance == 6.0
    crit = engine.PHDesignCriteria.from_params({"interface_distance": 8})
    assert crit.interface_distance == 8
    for bad in (0, -1.5):
        with pytest.raises(ValueError, match="interface_distance"):
            engine.PHDesignCriteria(interface_distance=bad)


def test_region_masks_are_recomputed_per_cutoff_and_cached(engine, monkeypatch, caplog):
    ctx = make_ctx(engine, proc_atom_array=object())
    calls = []

    def masks_at(distance):
        interface = torch.zeros(6, dtype=torch.bool)
        interface[0 if distance < 7 else 2] = True  # a wider cutoff reaches position 2
        empty = torch.zeros(6, dtype=torch.bool)
        return {"interface": interface, "core": empty, "surface": empty}

    def fake(atoms, token_aa, chain, chain_mask, device, length, interface_dist=6.0):
        calls.append(interface_dist)
        return masks_at(interface_dist)

    monkeypatch.setattr(engine, "_binder_region_masks", fake)
    ctx.region_masks = masks_at(6.0)

    assert ctx.region_mask(["interface"]).nonzero().flatten().tolist() == [0]
    assert ctx.region_mask(["interface"], 6.0).nonzero().flatten().tolist() == [0]
    assert calls == []  # the default cutoff is built with the context
    assert ctx.region_mask(["interface"], 8.0).nonzero().flatten().tolist() == [2]
    ctx.region_mask(["interface"], 8.0)
    assert calls == [8.0]  # computed once per cutoff

    with caplog.at_level("WARNING", logger=engine.__name__):
        assert not ctx.region_mask(["core"], 8.0).any()
    assert "selects no free binder position" in caplog.text


def test_a_non_default_cutoff_needs_the_processed_atom_array(engine):
    ctx = make_ctx(engine, proc_atom_array=None)
    with pytest.raises(RuntimeError, match="processed atom array"):
        ctx.region_mask(["interface"], 8.0)


def test_every_placement_call_passes_the_criteria_cutoff():
    assert "region_mask(crit.placement_region)" not in ENGINE_FILE.read_text()
