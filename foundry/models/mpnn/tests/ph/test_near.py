"""designable.near: redesign only the residues coupled to chosen sites, on either chain."""

import dataclasses

import numpy as np
import pytest
from objective_fixtures import BASE, SPEC
from test_session import make_config, make_inputs

from mpnn.ph.config import DesignableSpec, NearSpec, SwitchDesignConfig
from mpnn.ph.neighbourhood import near_positions
from mpnn.ph.session import resolve_designable, run_switch_design
from mpnn.ph.states import SiteKey

# Contact table of the toy complex: row i lists i, then its neighbours, closest first.
TABLE = np.array(
    [
        [0, 1, 4],
        [1, 2, 3],
        [2, 1, 3],
        [3, 2, 5],
        [4, 1, 5],
        [5, 4, 3],
    ]
)
CHAINS = ["A", "A", "A", "B", "B", "B"]
CONDITION_SITES = [SiteKey("A", 1), SiteKey("B", 2)]  # the sites the spec sets
RES_IDS = [1, 2, 3, 1, 2, 3]


def test_neighbourhood_is_outgoing_plus_incoming_couplings():
    # centre 3 lists 2 and 5; positions 1, 2 and 5 list 3 (position 1 only one way)
    assert near_positions(TABLE, [3], pool=range(6)) == [1, 2, 5]
    assert near_positions(TABLE, [3], pool=[0, 1, 2]) == [
        1,
        2,
    ]  # restricted to the pool


def test_k_keeps_only_the_nearest_listed_neighbours():
    assert near_positions(TABLE, [3], pool=range(6), k=1) == [2]
    assert near_positions(TABLE, [3], pool=range(6), k=2) == [1, 2, 5]
    assert near_positions(TABLE, [3], pool=range(6), k=99) == [1, 2, 5]


def test_max_mutations_keeps_the_closest_and_ranks_unlisted_last():
    # centre 3 lists 2 (rank 0) then 5 (rank 1); position 1 only lists the centre
    assert near_positions(TABLE, [3], pool=range(6), max_mutations=1) == [2]
    assert near_positions(TABLE, [3], pool=range(6), max_mutations=2) == [2, 5]
    assert near_positions(TABLE, [3], pool=range(6), max_mutations=3) == [1, 2, 5]


def test_several_centres_union_and_never_return_a_centre():
    got = near_positions(TABLE, [3, 4], pool=range(6))
    assert got == [0, 1, 2, 5] and 3 not in got and 4 not in got


def test_near_config_parsing_and_validation():
    cfg = {
        "conditions": SPEC["conditions"],
        "terms": {"switch": {"weight": 1.0}},
        "designable": {
            "chains": ["A"],
            "near": {"sites": ["B:2"], "k": 4, "max_mutations": 20},
        },
    }
    near = SwitchDesignConfig.from_dict(cfg).designable.near
    assert near == NearSpec(sites=["B:2"], k=4, max_mutations=20)
    for bad, message in (
        ({"sites": []}, "at least one site"),
        ({"sites": ["B:2"], "k": -1}, "must be >= 0"),
        ({"sites": ["B:2"], "radius": 8}, "Unknown designable.near keys"),
    ):
        broken = {**cfg, "designable": {"chains": ["A"], "near": bad}}
        with pytest.raises(ValueError, match=message):
            SwitchDesignConfig.from_dict(broken)


def test_target_site_restricts_the_binder_positions_that_may_change():
    kwargs = dict(neighbour_index=TABLE, spec_sites=CONDITION_SITES)
    spec = DesignableSpec(chains=["A"], near=NearSpec(sites=["B:2"]))
    # free binder positions are 1 and 2; only position 1 is coupled to the target residue B:2
    assert resolve_designable(spec, CHAINS, RES_IDS, **kwargs) == [1]
    narrow = DesignableSpec(chains=["A"], near=NearSpec(sites=["B:2"], k=1))
    assert resolve_designable(narrow, CHAINS, RES_IDS, **kwargs) == [1]


def test_near_needs_a_contact_table_and_known_sites():
    spec = DesignableSpec(chains=["A"], near=NearSpec(sites=["B:2"]))
    with pytest.raises(ValueError, match="contact table"):
        resolve_designable(spec, CHAINS, RES_IDS)
    unknown = DesignableSpec(chains=["A"], near=NearSpec(sites=["B:99"]))
    with pytest.raises(ValueError, match="not in the structure"):
        resolve_designable(unknown, CHAINS, RES_IDS, neighbour_index=TABLE)
    nearest_only = DesignableSpec(chains=["A"], near=NearSpec(sites=["B:1"], k=1))
    assert resolve_designable(nearest_only, CHAINS, RES_IDS, neighbour_index=TABLE) == [
        2
    ]
    nothing = DesignableSpec(
        chains=["A"], exclude=["A:1", "A:2", "A:3"], near=NearSpec(sites=["B:2"])
    )
    with pytest.raises(ValueError, match="No designable positions"):
        resolve_designable(nothing, CHAINS, RES_IDS, neighbour_index=TABLE)


def test_a_run_changes_only_the_neighbourhood_of_the_target_centre():
    inputs = dataclasses.replace(make_inputs(), neighbour_index=TABLE)
    config = make_config(
        allow_bare_titratable=["HIS"],
        designable={"chains": ["A"], "near": {"sites": ["B:2"]}},
    )
    run = run_switch_design(inputs, config)
    assert run.designable == [1]
    for record in run.result.records:
        unchanged = [p for p in range(6) if p != 1]
        assert record.tokens[unchanged].tolist() == BASE[unchanged].tolist()


def test_ranks_and_the_total_cap_are_separate_steps():
    from mpnn.ph.neighbourhood import UNRANKED, cap_by_rank, near_ranks

    ranks = near_ranks(TABLE, [3], pool=range(6))
    # centre 3 lists 2 then 5; position 1 only lists the centre, so it is unranked
    assert ranks == {2: 0, 5: 1, 1: UNRANKED}
    assert cap_by_rank([1, 2, 5], ranks, 2) == [2, 5]
    assert cap_by_rank([1, 2, 5], ranks, 0) == [1, 2, 5]  # 0 = no cap
    assert cap_by_rank([4, 7], {}, 1) == [4]  # no ranks: ties go to the lower position
