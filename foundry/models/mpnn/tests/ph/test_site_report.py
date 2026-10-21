"""The per-site report: complex versus free-partner preference for each titratable site."""

import dataclasses
import math

import pytest
from potts_fixtures import IdealComplex
from test_session import make_config, make_inputs

from mpnn.ph.binding import StateBinding
from mpnn.ph.report import apparent_pka, site_report
from mpnn.ph.session import run_switch_design
from mpnn.ph.vocab_meta import TokenTable

P, D = 2, 3  # HIS-P and HIS-S in the four-token test vocabulary
POSITION = {"A:1": 0, "B:2": 4}  # a His on each chain
LN10 = math.log(10.0)


def make_run(**overrides):
    ideal = IdealComplex(1, 2, 3)
    config = make_config(allow_bare_titratable=["HIS"], **overrides)
    run = run_switch_design(make_inputs(ideal), config)
    return ideal, run, run.result.records[0]


def flipped(run, record, row):
    state = run.conditions[row["condition"]].apply(record.tokens)
    position = POSITION[row["site"]]
    as_p, as_d = state.clone(), state.clone()
    as_p[position], as_d[position] = P, D
    return as_p, as_d


def test_one_row_per_condition_and_site():
    _, run, record = make_run()
    rows = run.site_report(record)
    assert {(r["condition"], r["site"]) for r in rows} == {
        ("on", "A:1"),
        ("on", "B:2"),
        ("off", "A:1"),
        ("off", "B:2"),
    }
    assert {r["parent"] for r in rows} == {"HIS"}


def test_binding_gap_is_the_bound_state_energy_difference_between_protonation_states():
    ideal, run, record = make_run()
    reference = StateBinding(*ideal.views())
    for row in run.site_report(record):
        as_p, as_d = flipped(run, record, row)
        expected = reference.energy(as_p) - reference.energy(as_d)
        assert row["binding_gap"] == pytest.approx(expected, abs=1e-9)
        assert row["gap_complex"] - row["gap_free"] == pytest.approx(row["binding_gap"])


def test_the_free_gap_of_a_receptor_site_ignores_the_binder():
    ideal, run, record = make_run()
    for row in (r for r in run.site_report(record) if r["site"] == "B:2"):
        as_p, as_d = flipped(run, record, row)
        expected = ideal.receptor.H_of(as_p[3:6]) - ideal.receptor.H_of(as_d[3:6])
        assert row["gap_free"] == pytest.approx(expected, abs=1e-9)


def test_rows_state_what_each_condition_imposes():
    _, run, record = make_run()
    tokens = {(r["condition"], r["site"]): r for r in run.site_report(record)}
    assert tokens[("on", "A:1")]["state_token"] == "HIS-S"
    assert tokens[("off", "B:2")]["state_token"] == "HIS-P"
    assert all(r["set_by_condition"] for r in tokens.values())
    assert (tokens[("on", "A:1")]["ph"], tokens[("off", "A:1")]["ph"]) == (7.4, 6.5)


def test_model_pka_shift_needs_a_beta_and_uses_the_models_own_when_it_has_one():
    _, run, record = make_run()
    assert all(r["delta_pka_model"] is None for r in run.site_report(record))
    for row in run.site_report(record, beta=0.5):
        assert row["delta_pka_model"] == pytest.approx(-0.5 * row["binding_gap"] / LN10)
    _, linked, linked_record = make_run(
        binding_model="linked_equilibrium", binding_options={"beta": 0.7}
    )
    for row in linked.site_report(linked_record):
        assert row["delta_pka_model"] == pytest.approx(-0.7 * row["binding_gap"] / LN10)


def test_linked_equilibrium_adds_populations_and_apparent_pka():
    _, run, record = make_run(
        binding_model="linked_equilibrium", binding_options={"beta": 0.7}
    )
    model = run.objective.ctx.binding
    for row in run.site_report(record):
        state = run.conditions[row["condition"]].apply(record.tokens)
        populations = model.site_populations(state, row["ph"])[POSITION[row["site"]]]
        assert row["fraction_protonated_free"] == pytest.approx(populations["free"])
        assert row["fraction_protonated_bound"] == pytest.approx(populations["bound"])
        assert row["apparent_pka_shift"] == pytest.approx(
            row["apparent_pka_bound"] - row["apparent_pka_free"]
        )
    plain = make_run()[1].site_report(make_run()[2])
    assert all(r["apparent_pka_shift"] is None for r in plain)


def test_apparent_pka_from_a_fraction():
    assert apparent_pka(7.0, 0.5) == pytest.approx(7.0)
    assert apparent_pka(6.0, 10 / 11) == pytest.approx(7.0)
    assert apparent_pka(7.0, 0.0) is None and apparent_pka(7.0, 1.0) is None
    assert apparent_pka(7.0, None) is None


def test_the_report_needs_the_receptor_system():
    _, run, record = make_run()
    ctx = dataclasses.replace(run.objective.ctx, receptor_view=None)
    with pytest.raises(ValueError, match="receptor system"):
        site_report(ctx, run.conditions, run.inputs, record.tokens)


# ---- legacy (v3/v4-style) vocabularies: two neutral histidine tautomers -------------
LEGACY = ("ALA", "HID", "HIE", "HIS-P")  # HID = 1, HIE = 2, HIS-P = 3
LEGACY_CONDITIONS = {
    "on": {"ph": 7.4, "states": {"A:1": "HID", "B:2": "HID"}},
    "off": {"ph": 6.5, "states": {"A:1": "HIS-P", "B:2": "HIS-P"}},
}


def make_legacy_run(**overrides):
    ideal = IdealComplex(1, 2, 3)
    inputs = dataclasses.replace(make_inputs(ideal), table=TokenTable(LEGACY))
    config = make_config(conditions=LEGACY_CONDITIONS, **overrides)
    run = run_switch_design(inputs, config)
    return ideal, run, run.result.records[0]


def legacy_gap(ideal, run, record, row, deprotonated):
    """Independent binding gap: protonated minus the mean over the given tokens."""
    reference = StateBinding(*ideal.views())
    state = run.conditions[row["condition"]].apply(record.tokens)
    position = POSITION[row["site"]]
    energies = []
    for token in (3, *deprotonated):
        probe = state.clone()
        probe[position] = token
        energies.append(reference.energy(probe))
    return energies[0] - sum(energies[1:]) / len(deprotonated)


def test_a_legacy_run_works_but_the_ambiguous_state_needs_an_explicit_policy():
    _, run, record = make_legacy_run()
    assert run.result.records  # the explicit HID / HIS-P conditions are valid
    with pytest.raises(ValueError, match="ambiguous.*report_tokens"):
        run.site_report(record)


def test_report_tokens_say_which_tautomers_the_gap_compares():
    tautomers = {"HIS": {"deprotonated": ["HID", "HIE"]}}  # averaged, as dep_map did
    ideal, run, record = make_legacy_run(report_tokens=tautomers)
    rows = run.site_report(record)
    assert len(rows) == 4
    for row in rows:
        assert row["protonated_tokens"] == ["HIS-P"]
        assert row["deprotonated_tokens"] == ["HID", "HIE"]
        assert row["binding_gap"] == pytest.approx(
            legacy_gap(ideal, run, record, row, (1, 2)), abs=1e-9
        )
    single = make_legacy_run(report_tokens={"HIS": {"deprotonated": "HIE"}})
    for row in single[1].site_report(single[2]):
        assert row["binding_gap"] == pytest.approx(
            legacy_gap(single[0], single[1], single[2], row, (2,)), abs=1e-9
        )


@pytest.mark.parametrize(
    "policy, message",
    [
        ({"HIS": {"neutral": "HID"}}, "protonated.*deprotonated"),
        ({"HIS": {"deprotonated": "ALA"}}, "not a HIS"),  # another residue's token
        ({"HIS": {"deprotonated": "ASP-D"}}, "neither a vocabulary token"),
        ({"HIS": {"deprotonated": []}}, "at least one token"),
        ({"HIS": ["HID"]}, "mapping"),
    ],
)
def test_report_tokens_are_validated_up_front(policy, message):
    with pytest.raises(ValueError, match=message):
        make_legacy_run(report_tokens=policy)


def test_unambiguous_vocabularies_need_no_policy():
    _, run, record = make_run()
    row = run.site_report(record)[0]
    assert row["protonated_tokens"] == ["HIS-P"] and row["deprotonated_tokens"] == [
        "HIS-S"
    ]
