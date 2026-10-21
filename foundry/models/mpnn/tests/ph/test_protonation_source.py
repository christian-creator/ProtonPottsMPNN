"""The two PROPKA settings: parsing, per-site reference pKa, the engine helpers and the logs."""

import dataclasses
import logging
import sys

import pytest
from objective_fixtures import NAMES, PARENTS, SPEC
from test_session import make_config, make_inputs

from mpnn.ph.binding import DEFAULT_MODEL_PKA, linked_sites_from_spec
from mpnn.ph.config import ProtonationSources, SwitchDesignConfig
from mpnn.ph.session import run_switch_design
from mpnn.ph.states import SiteKey, StateSpec, site_index_from_arrays
from mpnn.ph.vocab_meta import TokenTable

A1, B2 = SiteKey("A", 1), SiteKey("B", 2)  # the two titratable sites of the toy complex
LINKED = dict(binding_model="linked_equilibrium", binding_options={"beta": 0.7})
PROPKA_PKA = {"reference_pka": "propka"}
PROPKA_STATES = {"states": "propka", "ph": 7.4}
CONSTANT = DEFAULT_MODEL_PKA["HIS"]


def test_both_settings_default_to_not_using_propka():
    assert ProtonationSources() == ProtonationSources("ev6", "constant")
    assert make_config().protonation == ProtonationSources()
    for bad in ({"ph": 7.4}, {"margin": 0.5}):
        with pytest.raises(
            ValueError, match="only apply to protonation.states 'propka'"
        ):
            ProtonationSources(**bad)


def test_propka_states_need_a_ph_and_have_a_default_margin():
    sources = ProtonationSources.from_dict(PROPKA_STATES)
    assert (sources.ph, sources.margin) == (7.4, 0.5)
    assert ProtonationSources("propka", ph=6.0, margin=0.0).margin == 0.0
    for bad in ({"states": "propka"}, {"states": "propka", "ph": float("nan")}):
        with pytest.raises(ValueError, match="finite protonation.ph"):
            ProtonationSources.from_dict(bad)
    with pytest.raises(ValueError, match="margin"):
        ProtonationSources.from_dict(dict(PROPKA_STATES, margin=-0.1))


def test_names_and_values_are_checked():
    with pytest.raises(ValueError, match="states must be one of"):
        ProtonationSources.from_dict({"states": "pkacalc"})
    with pytest.raises(ValueError, match="reference_pka must be one of"):
        ProtonationSources.from_dict({"reference_pka": "ev6"})
    with pytest.raises(ValueError, match="Unknown protonation keys"):
        ProtonationSources.from_dict(
            {"source": "propka"}
        )  # the old name of the setting


def test_the_two_settings_are_independent_and_parse_from_the_config():
    only_states = make_config(protonation=PROPKA_STATES).protonation
    assert only_states == ProtonationSources("propka", "constant", 7.4, 0.5)
    only_pka = make_config(protonation=PROPKA_PKA, **LINKED).protonation
    assert only_pka == ProtonationSources("ev6", "propka")
    both = make_config(
        protonation=dict(PROPKA_STATES, **PROPKA_PKA), **LINKED
    ).protonation
    assert (both.states, both.reference_pka) == ("propka", "propka")
    with pytest.raises(ValueError, match="Unknown protonation keys"):
        make_config(protonation={"sauce": "propka"})


def test_a_propka_reference_pka_needs_the_model_that_uses_a_pka():
    with pytest.raises(ValueError, match="linked_equilibrium"):
        make_config(protonation=PROPKA_PKA)  # state_binding has no pKa to set
    assert isinstance(make_config(protonation=PROPKA_PKA, **LINKED), SwitchDesignConfig)


def sites(by_site=None, by_parent=None):
    spec = StateSpec.from_dict(SPEC)
    index = site_index_from_arrays(["A", "A", "A", "B", "B", "B"], [1, 2, 3, 1, 2, 3])
    found = linked_sites_from_spec(
        spec, TokenTable(NAMES), index, PARENTS, by_parent, pka_by_site=by_site
    )
    return {site.position: site.pka for site in found}


def test_a_sites_own_pka_beats_the_value_for_its_residue_type(caplog):
    assert sites() == {0: CONSTANT, 4: CONSTANT}
    assert sites(by_parent={"HIS": 6.0}) == {0: 6.0, 4: 6.0}
    assert sites(by_site={A1: 5.2, B2: 6.9}, by_parent={"HIS": 6.0}) == {0: 5.2, 4: 6.9}
    with caplog.at_level(logging.WARNING):  # one site lacks its own value
        assert sites(by_site={A1: 5.2}, by_parent={"HIS": 6.0}) == {0: 5.2, 4: 6.0}
    assert "No per-site pKa for B:2 (HIS), using the per-type value 6.0." in caplog.text
    with pytest.raises(ValueError, match="must be finite"):
        sites(by_site={A1: float("nan"), B2: 6.9})


def chosen_pka(run):
    return {site.position: site.pka for site in run.objective.ctx.binding.sites}


def with_pka():
    return dataclasses.replace(make_inputs(), site_pka={A1: 5.2, B2: 6.9})


def test_propka_values_reach_the_linked_equilibrium_model():
    run = run_switch_design(with_pka(), make_config(protonation=PROPKA_PKA, **LINKED))
    assert chosen_pka(run) == {0: 5.2, 4: 6.9}
    # the user's own value for one site wins over PROPKA's
    by_hand = dict(LINKED, binding_options={"beta": 0.7, "pka_by_site": {"B:2": 4.0}})
    run = run_switch_design(with_pka(), make_config(protonation=PROPKA_PKA, **by_hand))
    assert chosen_pka(run) == {0: 5.2, 4: 4.0}


def test_the_reference_pka_is_the_constant_unless_it_is_asked_for():
    expected = {0: CONSTANT, 4: CONSTANT}
    run = run_switch_design(with_pka(), make_config(**LINKED))
    assert chosen_pka(run) == expected
    # PROPKA states say nothing about the reference pKa
    run = run_switch_design(
        with_pka(), make_config(protonation=PROPKA_STATES, **LINKED)
    )
    assert chosen_pka(run) == expected


def test_propka_pka_values_are_required_when_they_are_asked_for():
    with pytest.raises(ValueError, match="DesignInputs.site_pka"):
        run_switch_design(make_inputs(), make_config(protonation=PROPKA_PKA, **LINKED))
    run_switch_design(
        make_inputs(), make_config(protonation=PROPKA_STATES)
    )  # not needed here


def pkad_caveats(caplog):
    return [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "PKAD" in r.getMessage()
    ]


def test_the_logs_say_where_each_reference_pka_comes_from(caplog):
    with caplog.at_level(logging.INFO, logger="mpnn.ph.session"):
        run_switch_design(with_pka(), make_config(protonation=PROPKA_PKA, **LINKED))
    text = caplog.text
    assert (
        "Reference pKa of A:1 (HIS): 5.20 from PROPKA on the free partner; the per-type constant is 6.5."
        in text
    )
    assert len(pkad_caveats(caplog)) == 1  # His: PROPKA did not beat the constant
    caplog.clear()
    by_hand = dict(
        LINKED, binding_options={"beta": 0.7, "pka_by_site": {"A:1": 5.9, "B:2": 6.1}}
    )
    with caplog.at_level(logging.INFO, logger="mpnn.ph.session"):
        run_switch_design(make_inputs(), make_config(**by_hand))
    assert "from binding_options.pka_by_site" in caplog.text
    assert not pkad_caveats(caplog)  # chosen by hand: no caveat
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="mpnn.ph.session"):
        run_switch_design(make_inputs(), make_config(**LINKED))
    assert (
        "Reference pKa of" not in caplog.text
    )  # the constants are the default: nothing to explain


def test_a_missing_propka_stops_the_run_before_featurising(engine, monkeypatch):
    monkeypatch.setitem(sys.modules, "propka", None)
    monkeypatch.setitem(sys.modules, "propka.run", None)
    with pytest.raises(ImportError, match="pip install propka"):
        engine._protonation_classifier(ProtonationSources("propka", ph=7.4))
    for protonation in (PROPKA_STATES, PROPKA_PKA):
        config = make_config(protonation=protonation, **LINKED)
        with pytest.raises(ImportError, match="pip install propka"):
            engine._check_propka_available(config)
    engine._check_propka_available(make_config(**LINKED))  # the defaults need none
    assert engine._protonation_classifier(ProtonationSources()) is None


def test_engine_helpers_choose_the_labeller_and_the_pka_runs(engine):
    from mpnn.transforms.propka_labeller import PropkaLabeller

    labeller = engine._protonation_classifier(
        ProtonationSources("propka", ph=6.0, margin=0.25)
    )
    assert labeller == PropkaLabeller(6.0, 0.25)
    assert engine._as_protonation(None) == ProtonationSources()
    assert engine._as_protonation({"states": "propka", "ph": 6.0}).margin == 0.5

    def needs(**overrides):
        return engine._needs_site_pka(make_config(**overrides))

    assert needs(protonation=PROPKA_PKA, **LINKED)
    assert not needs(protonation=PROPKA_STATES, **LINKED)  # PROPKA states alone
    assert not needs(**LINKED)  # the constants

    text = engine._describe_protonation(
        ProtonationSources("propka", "propka", 7.4, 0.5)
    )
    assert (
        "PROPKA (pH 7.4, margin 0.5)" in text and "PROPKA on the free partners" in text
    )
    assert (
        "starting states EV6, reference pKa per-type constants"
        in engine._describe_protonation(ProtonationSources())
    )


def test_run_ph_redesign_refuses_a_reference_pka_it_cannot_use(engine):
    bare = object.__new__(engine.PottsMPNNPHEngine)
    with pytest.raises(ValueError, match="only applies to run_switch_design"):
        bare.run_ph_redesign(
            atom_array=None,
            binder_chain="A",
            criteria_list=[],
            protonation=PROPKA_PKA,
        )
