"""Synthetic engine contexts for tests that exercise real engine code (atomworks-free)."""

from types import SimpleNamespace

import numpy as np
import torch
from objective_fixtures import BASE, NAMES, PARENTS
from potts_fixtures import V, random_graph_potts

from mpnn.ph.engine_adapter import PottsScorerAdapter
from mpnn.ph.vocab_meta import TokenTable

CHAINS = ["A", "A", "A", "B", "B", "B"]
RES_IDS = [1, 2, 3, 1, 2, 3]


def make_ctx(engine, proc_atom_array=None):
    """A real ``_PHContext`` over random Potts tables for the toy complex."""
    table = TokenTable(NAMES)
    etab, e_idx = random_graph_potts(6, 4, V, seed=11)
    encoding = SimpleNamespace(
        idx_to_token=dict(enumerate(NAMES)),
        token_to_idx={n: i for i, n in enumerate(NAMES)},
    )
    return engine._PHContext(
        device=torch.device("cpu"),
        encoding=encoding,
        extended_vocab="v6",
        canonical_map=torch.tensor(table.parent_index()),
        token_aa=SimpleNamespace(
            chain_id=np.array(CHAINS),
            res_id=np.array(RES_IDS),
            res_name=np.array(PARENTS),
        ),
        S_native=BASE.clone(),
        scorer=engine._PottsScorer(etab, e_idx),
        field_potts=None,
        field_mpnn=None,
        free_mask=torch.tensor([True] * 3 + [False] * 3),
        V=V,
        L=6,
        unknown_indices=[],
        eidx_np=e_idx[0].numpy(),
        K=4,
        chainA_t=torch.tensor([True] * 3 + [False] * 3),
        binder_chain="A",
        network_input={},
        proc_atom_array=proc_atom_array,
    )


def bare_engine(engine, ctx):
    """An engine instance without model loading, whose context and isolated chains are synthetic."""
    engine_obj = object.__new__(engine.PottsMPNNPHEngine)
    isolated = {
        ("A",): (
            PottsScorerAdapter(engine._PottsScorer(*random_graph_potts(3, 3, V, 21))),
            [0, 1, 2],
        ),
        ("B",): (
            PottsScorerAdapter(engine._PottsScorer(*random_graph_potts(3, 3, V, 22))),
            [3, 4, 5],
        ),
    }
    engine_obj._isolated_scorer = lambda c, chains, site_index: isolated[tuple(chains)]
    engine_obj._build_context = (
        lambda atom_array, chain, base_seed=0, protonation=None: ctx
    )
    engine_obj.device = torch.device("cpu")
    return engine_obj


def make_output(engine, sample, energy, tokens, seed_idx, **extra):
    return engine.PHDesignOutput(
        binder_chain="A",
        canonical_sequence="".join(t[0] for t in tokens),
        extended_tokens=tokens,
        extended_vocab=True,
        scheme="s",
        sample=sample,
        final_potts_energy=energy,
        seed_idx=seed_idx,
        **extra,
    )
