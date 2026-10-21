"""Tests for the strict design-config parser."""

import pytest
from objective_fixtures import SPEC

from mpnn.ph.config import SwitchDesignConfig

FULL = {
    "conditions": SPEC["conditions"],
    "on": "on",
    "off": "off",
    "binding_model": "state_binding",
    "terms": {
        "stability": {"weight": 0.3, "reduce": "max"},
        "potency": {"weight": 0.3},
        "switch": {"weight": 0.4, "off_margin": 2.0},
    },
    "search": "single",
    "block_size": 3,
    "temperature": 0.05,
    "n_seeds": 2,
    "designable": {"chains": ["A"], "exclude": ["A:3"]},
}


def parse(**overrides):
    return SwitchDesignConfig.from_dict({**FULL, **overrides})


def test_full_example_parses_with_defaults_filled_in():
    config = parse()
    assert config.block_size == 3 and config.n_seeds == 2 and config.max_rounds == 10
    assert config.designable.chains == ["A"] and config.designable.exclude == ["A:3"]
    assert (config.spec.on, config.spec.off) == ("on", "off")
    weights, terms = config.weights_and_terms()
    assert weights == {"stability": 0.3, "potency": 0.3, "switch": 0.4}
    assert terms["switch"].off_margin == 2.0
    assert config.plan().mode == "single"


def test_pareto_is_requested_explicitly():
    plan = parse(search="pareto", divisions=2).plan()
    assert plan.mode == "pareto" and plan.weight_vectors.shape == (6, 3)
    one_term = parse(search="pareto", terms={"switch": {"weight": 1.0}}).plan()
    assert one_term.mode == "single"


@pytest.mark.parametrize(
    "bad, message",
    [
        ({"blocksize": 2}, "Unknown config keys"),
        ({"designable": {"chain": ["A"]}}, "Unknown designable keys"),
        ({"terms": {}}, "terms"),
        ({"terms": {"stabilty": {"weight": 1.0}}}, "Unknown term"),
        ({"search": "genetic"}, "search"),
        ({"block_size": 0}, "block_size"),
        ({"temperature": -1.0}, "temperature"),
        ({"n_select": 0}, "n_select"),
        ({"divisions": 0}, "divisions"),
        ({"terms": {"switch": {"weight": 0.0}}}, "positive weight"),
    ],
)
def test_bad_configs_fail_loudly(bad, message):
    with pytest.raises(ValueError, match=message):
        parse(**bad)


def test_missing_conditions_fail():
    cfg = {k: v for k, v in FULL.items() if k not in ("conditions", "on", "off")}
    with pytest.raises(ValueError):
        SwitchDesignConfig.from_dict(cfg)


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_weights_are_refused_instead_of_dropping_the_term(weight):
    with pytest.raises(ValueError, match="finite"):
        parse(terms={"stability": {"weight": 1.0}, "switch": {"weight": weight}})
