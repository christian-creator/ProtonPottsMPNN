"""The engine scorer behind the ph protocols, on complex + isolated-chain tables."""

import itertools

import pytest
import torch
from potts_fixtures import V, random_graph_potts

from mpnn.ph.binding import StateBinding, SystemView
from mpnn.ph.engine_adapter import PottsScorerAdapter
from mpnn.ph.potentials import probe_block_potentials

TOKENS = torch.tensor([0, 1, 2, 3, 1, 0, 2, 3])


def adapter(engine, n_positions, n_neighbours, seed):
    etab, e_idx = random_graph_potts(n_positions, n_neighbours, V, seed)
    return PottsScorerAdapter(engine._PottsScorer(etab, e_idx))


def test_adapter_block_potentials_match_the_probing_oracle(engine):
    scorer = adapter(engine, 8, 4, seed=1)
    for block in ([2], [1, 6], [0, 3, 5]):
        reference = [int(TOKENS[p]) for p in block]
        analytic = scorer.block_potentials(TOKENS, block).with_reference(
            reference, scorer.H_of(TOKENS)
        )
        oracle = probe_block_potentials(scorer.H_of, TOKENS, block, V)
        assert torch.allclose(analytic.joint().double(), oracle.joint(), atol=1e-4)


def test_state_binding_over_real_scorers_matches_per_assignment_energy(engine):
    """Complex and each isolated chain have their own graph, as after re-encoding."""
    complex_view = SystemView(adapter(engine, 8, 4, 1), range(8), V)
    binder_view = SystemView(adapter(engine, 4, 3, 2), range(4), V)
    receptor_view = SystemView(adapter(engine, 4, 3, 3), range(4, 8), V)
    model = StateBinding(complex_view, binder_view, receptor_view)
    for block in ([1, 2], [0, 3], [5, 6]):
        joint = model.block_potentials(TOKENS, block).joint()
        for values in itertools.product(range(V), repeat=len(block)):
            probe = TOKENS.clone()
            probe[block] = torch.tensor(values)
            assert float(joint[values]) == pytest.approx(model.energy(probe), abs=1e-4)


def test_positions_in_complex_follow_the_isolated_chains_order():
    from mpnn.ph.engine_adapter import positions_in_complex
    from mpnn.ph.states import site_index_from_arrays

    index = site_index_from_arrays(["A", "A", "B", "B"], [1, 2, 1, 2])
    assert positions_in_complex(["B", "B"], [2, 1], index) == [3, 2]
    with pytest.raises(KeyError, match="not in the complex"):
        positions_in_complex(["B"], [9], index)
    with pytest.raises(ValueError, match="same complex position"):
        positions_in_complex(["A", "A"], [1, 1], index)


def test_parent_names_map_states_onto_their_residue():
    from mpnn.ph.engine_adapter import parent_names
    from mpnn.ph.vocab_meta import TokenTable

    table = TokenTable(("ALA", "HIS", "HIS-P", "HIS-S"))
    names = parent_names(table.names, table.parent_index(), [0, 2, 3, 1])
    assert names == ["ALA", "HIS", "HIS", "HIS"]


def test_knn_partner_rank_skips_the_residue_itself():
    import numpy as np

    from mpnn.ph.engine_adapter import knn_partner_rank

    rank = knn_partner_rank(np.array([[0, 2, 1], [1, 0, 2], [2, 1, 0]]))
    assert rank(0) == [2, 1] and rank(1) == [0, 2]
