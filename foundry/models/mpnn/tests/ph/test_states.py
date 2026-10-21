"""Tests for condition/state specs (atomworks-free)."""

import logging

import pytest
import torch

from mpnn.ph.states import (
    Condition,
    SiteKey,
    StateSpec,
    resolve_condition,
    resolve_spec,
    site_index_from_arrays,
)
from mpnn.ph.vocab_meta import STANDARD_RESIDUES, TokenTable

V6 = TokenTable(
    (
        *STANDARD_RESIDUES,
        "UNK",
        *("HIS-P", "HIS-S", "HIS-A", "ASP-P", "ASP-D", "ASP-A"),
        *("GLU-P", "GLU-D", "GLU-A"),
    )
)

# A toy complex: binder chain A (HIS ALA ASP), receptor chain B (GLU HIS LYS).
CHAINS = ["A", "A", "A", "B", "B", "B"]
RES_IDS = [10, 11, 12, 57, 58, 59]
PARENTS = ["HIS", "ALA", "ASP", "GLU", "HIS", "LYS"]


@pytest.fixture
def sites():
    return site_index_from_arrays(CHAINS, RES_IDS)


def spec_dict(on_state="HIS-S", off_state="HIS-P"):
    return {
        "conditions": {
            "on": {"ph": 7.4, "states": {"A:10": on_state, "B:58": on_state}},
            "off": {"ph": 6.5, "states": {"A:10": off_state, "B:58": off_state}},
        }
    }


def test_site_key_round_trip_and_errors():
    assert SiteKey.parse("B:57") == SiteKey("B", 57)
    assert str(SiteKey("A", -3)) == "A:-3"
    assert SiteKey.parse("A:-3") == SiteKey("A", -3)
    for bad in ("B57", ":5", "B:", "B:x"):
        with pytest.raises(ValueError):
            SiteKey.parse(bad)


def test_from_dict_defaults_on_off():
    spec = StateSpec.from_dict(spec_dict())
    assert (spec.on, spec.off) == ("on", "off")
    assert spec.conditions["on"].ph == 7.4
    assert spec.sites() == {SiteKey("A", 10), SiteKey("B", 58)}


def test_role_names_are_validated():
    cfg = spec_dict()
    cfg["on"], cfg["off"] = "on", "nowhere"
    with pytest.raises(ValueError, match="nowhere"):
        StateSpec.from_dict(cfg)
    cfg["off"] = "on"
    with pytest.raises(ValueError, match="different"):
        StateSpec.from_dict(cfg)
    with pytest.raises(ValueError):
        StateSpec.from_dict({})


def test_resolution_covers_both_sides_of_the_interface(sites):
    spec = StateSpec.from_dict(spec_dict())
    resolved = resolve_spec(spec, V6, sites, PARENTS)
    his_p, his_s = V6.index("HIS-P"), V6.index("HIS-S")
    assert resolved["on"].token_by_position == {0: his_s, 4: his_s}  # binder + receptor
    assert resolved["off"].token_by_position == {0: his_p, 4: his_p}


def test_reversing_the_spec_swaps_tokens_only(sites):
    forward = resolve_spec(StateSpec.from_dict(spec_dict()), V6, sites, PARENTS)
    reverse = resolve_spec(
        StateSpec.from_dict(spec_dict("HIS-P", "HIS-S")), V6, sites, PARENTS
    )
    assert forward["on"].token_by_position == reverse["off"].token_by_position
    assert forward["off"].token_by_position == reverse["on"].token_by_position


def test_state_words_resolve_per_residue_type(sites):
    cond = Condition(
        "x",
        {SiteKey("A", 10): "protonated", SiteKey("A", 12): "deprotonated"},
    )
    out = resolve_condition(cond, V6, sites, PARENTS)
    assert out.token_by_position == {0: V6.index("HIS-P"), 2: V6.index("ASP-D")}


def test_apply_clones_and_overrides_only_listed_positions(sites):
    cond = Condition("x", {SiteKey("B", 58): "HIS-S"})
    resolved = resolve_condition(cond, V6, sites, PARENTS)
    base = torch.arange(6)
    out = resolved.apply(base)
    assert out[4] == V6.index("HIS-S")
    assert torch.equal(out[[0, 1, 2, 3, 5]], base[[0, 1, 2, 3, 5]])
    assert torch.equal(base, torch.arange(6))  # input untouched


@pytest.mark.parametrize(
    "state, site, message",
    [
        ("HIS-P", "A:12", "not a ASP"),  # token of the wrong residue type
        ("protonated", "A:11", "no protonation states"),  # alanine
        ("protonated", "B:59", "no protonation states"),  # lysine is not in the vocab
    ],
)
def test_bad_assignments_fail_loudly(sites, state, site, message):
    cond = Condition("x", {SiteKey.parse(site): state})
    with pytest.raises(ValueError, match=message):
        resolve_condition(cond, V6, sites, PARENTS)


def test_unknown_site_is_reported(sites):
    cond = Condition("x", {SiteKey("Z", 1): "protonated"})
    with pytest.raises(KeyError, match="Z:1"):
        resolve_condition(cond, V6, sites, PARENTS)


def test_duplicate_residue_keys_are_refused():
    with pytest.raises(ValueError, match="not unique"):
        site_index_from_arrays(["A", "A"], [5, 5])


def test_mismatched_coverage_is_flagged(sites, caplog):
    cfg = spec_dict()
    cfg["conditions"]["off"]["states"].pop("B:58")
    spec = StateSpec.from_dict(cfg)
    assert spec.uncovered_sites() == {"off": {SiteKey("B", 58)}}
    with caplog.at_level(logging.WARNING, logger="mpnn.ph.states"):
        resolve_spec(spec, V6, sites, PARENTS)
    assert "B:58" in caplog.text


def test_resolved_conditions_carry_their_ph(sites):
    resolved = resolve_spec(StateSpec.from_dict(spec_dict()), V6, sites, PARENTS)
    assert (resolved["on"].ph, resolved["off"].ph) == (7.4, 6.5)


@pytest.mark.parametrize(
    "body, message",
    [
        (
            {"state": {"A:10": "HIS-S"}},
            "unknown keys .*'state'",
        ),  # the typo of 'states'
        ({"states": {"A:10": "HIS-S"}, "pH": 7.4}, "unknown keys .*'pH'"),
        ({"ph": 7.4}, "needs a 'states' mapping"),
        (["A:10"], "must be a mapping"),
        ({"states": ["A:10"]}, "'states' must be a mapping"),
    ],
)
def test_condition_bodies_are_validated_as_strictly_as_the_config(body, message):
    with pytest.raises(ValueError, match=message):
        StateSpec.from_dict({"conditions": {"on": body}})


def test_the_spec_rejects_unknown_top_level_keys():
    with pytest.raises(ValueError, match="unknown keys .*'condition'"):
        StateSpec.from_dict({"condition": {}, "conditions": {"on": {"states": {}}}})


def test_an_explicitly_empty_condition_is_allowed():
    spec = StateSpec.from_dict({"conditions": {"base": {"states": {}}}})
    assert spec.conditions["base"].states == {}
