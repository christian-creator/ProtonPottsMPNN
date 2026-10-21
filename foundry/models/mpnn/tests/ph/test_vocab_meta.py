"""Tests for the protonation token table (atomworks-free)."""

import pytest

from mpnn.ph.vocab_meta import (
    STANDARD_RESIDUES,
    Protonation,
    TokenTable,
    parse_token,
)

# Copy of the v6 token order (token_encodings.py): 20 residues, UNK, 9 states.
V6_TOKENS = (
    *STANDARD_RESIDUES,
    "UNK",
    "HIS-P", "HIS-S", "HIS-A",
    "ASP-P", "ASP-D", "ASP-A",
    "GLU-P", "GLU-D", "GLU-A",
)  # fmt: skip
V4_EXTRA = ("HID", "HIE", "HIS-P", "HIS-D", "HIS-A", "ASP-P", "ASP-D", "ASP-A")


@pytest.fixture
def v6():
    return TokenTable(V6_TOKENS)


def test_v6_has_thirty_tokens(v6):
    assert len(v6) == 30


@pytest.mark.parametrize(
    "name, parent, state, charge",
    [
        ("HIS-P", "HIS", Protonation.PROTONATED, 1),
        ("HIS-S", "HIS", Protonation.DEPROTONATED, 0),
        ("ASP-P", "ASP", Protonation.PROTONATED, 0),
        ("ASP-D", "ASP", Protonation.DEPROTONATED, -1),
        ("GLU-D", "GLU", Protonation.DEPROTONATED, -1),
        ("GLU-A", "GLU", Protonation.AMBIGUOUS, None),
        ("LYS", "LYS", Protonation.UNSPECIFIED, None),
        ("UNK", "UNK", Protonation.UNSPECIFIED, None),
    ],
)
def test_parse_token(name, parent, state, charge):
    meta = parse_token(name)
    assert (meta.parent, meta.state, meta.charge) == (parent, state, charge)


def test_legacy_tokens_are_recognised():
    assert parse_token("HID").state is Protonation.DEPROTONATED
    assert parse_token("HIE").charge == 0
    assert parse_token("HIS-D").charge == -1


@pytest.mark.parametrize("bad", ["FOO", "HIS-X", "LYS-P", "ALA-D", "HIS-"])
def test_unknown_token_names_fail_loudly(bad):
    with pytest.raises(ValueError):
        parse_token(bad)


def test_duplicate_names_rejected():
    with pytest.raises(ValueError):
        TokenTable(("ALA", "ALA"))


def test_titratable_parents(v6):
    assert v6.titratable_parents() == ("ASP", "GLU", "HIS")


def test_tokens_for_each_direction(v6):
    assert v6.tokens_for("HIS", Protonation.PROTONATED) == ("HIS-P",)
    assert v6.tokens_for("HIS", Protonation.DEPROTONATED) == ("HIS-S",)
    assert v6.tokens_for("ASP", Protonation.DEPROTONATED) == ("ASP-D",)
    assert v6.tokens_for("ALA", Protonation.PROTONATED) == ()


def test_resolve_state_word_and_token_name(v6):
    assert v6.resolve("HIS", "protonated") == "HIS-P"
    assert v6.resolve("HIS", "Deprotonated") == "HIS-S"
    assert v6.resolve("ASP", "ASP-D") == "ASP-D"
    assert v6.resolve("GLU", "ambiguous") == "GLU-A"


def test_resolve_rejects_parent_mismatch_and_nonsense(v6):
    with pytest.raises(ValueError, match="not a ASP"):
        v6.resolve("ASP", "HIS-P")
    with pytest.raises(ValueError, match="neither"):
        v6.resolve("HIS", "very-acidic")
    with pytest.raises(ValueError, match="no protonated token"):
        v6.resolve("ALA", "protonated")


def test_legacy_vocab_needs_explicit_his_tautomer():
    legacy = TokenTable((*STANDARD_RESIDUES, "UNK", *V4_EXTRA[:-3], "ASP-P", "ASP-D"))
    with pytest.raises(ValueError, match="ambiguous"):
        legacy.resolve("HIS", "deprotonated")
    assert legacy.resolve("HIS", "HID") == "HID"


def test_parent_index_maps_states_onto_bare_residue(v6):
    parent = v6.parent_index()
    assert parent[v6.index("HIS-P")] == v6.index("HIS")
    assert parent[v6.index("ASP-D")] == v6.index("ASP")
    assert parent[v6.index("ALA")] == v6.index("ALA")


def test_parent_index_needs_bare_parent():
    table = TokenTable(("ALA", "HIS-P"))
    with pytest.raises(KeyError):
        table.parent_index()


def test_index_error_names_the_token(v6):
    with pytest.raises(KeyError, match="HID"):
        v6.index("HID")


def test_design_mask_blocks_state_tokens_and_titratable_residues(v6):
    mask = dict(zip(v6.names, v6.design_mask()))
    assert not mask["UNK"]
    assert not any(mask[t] for t in v6.names if "-" in t)  # no state tokens
    assert not any(mask[t] for t in ("HIS", "ASP", "GLU"))
    assert all(mask[t] for t in ("ALA", "LYS", "TRP", "GLY"))
    assert sum(mask.values()) == 17


def test_design_mask_can_admit_bare_titratable_residues(v6):
    mask = dict(zip(v6.names, v6.design_mask(allow_bare_titratable=["HIS"])))
    assert mask["HIS"] and not mask["ASP"] and not mask["HIS-P"]
    with pytest.raises(ValueError, match="Not titratable"):
        v6.design_mask(allow_bare_titratable=["LYS"])


def test_canonical_sequence_folds_states_onto_residues(v6):
    tokens = [v6.index(t) for t in ("ALA", "HIS-P", "ASP-D", "GLU-A", "UNK", "TRP")]
    assert v6.canonical_sequence(tokens) == "AHDEXW"
    assert v6.token_names(tokens) == ["ALA", "HIS-P", "ASP-D", "GLU-A", "UNK", "TRP"]
    assert v6.canonical_sequence([]) == ""
