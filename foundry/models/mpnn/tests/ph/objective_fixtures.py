"""Shared setup for objective, descent and search tests (atomworks-free).

A toy complex: binder chain A (positions 0-2) and receptor chain B (positions 3-5), each
with one histidine (a titratable site) and designable alanine positions on the binder.
"""

import itertools

import torch
from potts_fixtures import IdealComplex, V

from mpnn.ph.binding import LinkedEquilibrium, StateBinding, linked_sites_from_spec
from mpnn.ph.objective import (
    Objective,
    PotencyTerm,
    StabilityTerm,
    SwitchTerm,
    TermContext,
)
from mpnn.ph.states import StateSpec, resolve_spec, site_index_from_arrays
from mpnn.ph.vocab_meta import TokenTable

NAMES = ("ALA", "HIS", "HIS-P", "HIS-S")  # token indices 0..3
CHAINS = ["A", "A", "A", "B", "B", "B"]
RES_IDS = [1, 2, 3, 1, 2, 3]
PARENTS = ["HIS", "ALA", "ALA", "ALA", "HIS", "ALA"]  # a His on each side
BASE = torch.tensor([1, 0, 0, 0, 1, 0])
DESIGN_BLOCK = [1, 2]  # designable binder positions
SPEC = {
    "conditions": {
        "on": {"ph": 7.4, "states": {"A:1": "deprotonated", "B:2": "deprotonated"}},
        "off": {"ph": 6.5, "states": {"A:1": "protonated", "B:2": "protonated"}},
    }
}


def make_context(swap_roles=False, binding_model="state_binding"):
    ideal = IdealComplex(1, 2, 3)
    complex_view, binder_view, receptor_view = ideal.views()
    table = TokenTable(NAMES)
    site_index = site_index_from_arrays(CHAINS, RES_IDS)
    spec = StateSpec.from_dict(SPEC)
    resolved = resolve_spec(spec, table, site_index, PARENTS)
    if binding_model == "state_binding":
        binding = StateBinding(complex_view, binder_view, receptor_view)
    else:
        sites = linked_sites_from_spec(spec, table, site_index, PARENTS)
        binding = LinkedEquilibrium(complex_view, binder_view, receptor_view, sites)
    on, off = ("off", "on") if swap_roles else ("on", "off")
    return TermContext(
        binding=binding,
        binder_view=binder_view,
        complex_view=complex_view,
        conditions=resolved,
        on=on,
        off=off,
    )


def assignments(block=DESIGN_BLOCK):
    for values in itertools.product(range(V), repeat=len(block)):
        tokens = BASE.clone()
        for position, token in zip(block, values):
            tokens[position] = token
        yield values, tokens


NAMES_ACTIVE = ("stability", "potency", "switch")


def make_objective(ctx):
    terms = {
        "stability": StabilityTerm(),
        "potency": PotencyTerm(),
        "switch": SwitchTerm(),
    }
    objective = Objective(ctx, terms)
    objective.freeze_scales(BASE, [DESIGN_BLOCK, [1], [2]])
    return objective
