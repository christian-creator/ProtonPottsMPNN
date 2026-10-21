"""Assemble and run a pH-switch design from scorers, a structure's labels and a config.

This is the engine-independent core of a design run: it needs three Potts scorers
(complex, binder alone, receptor alone) that follow the :class:`~mpnn.ph.binding.EnergyScorer`
protocol, and plain arrays describing the residues. The engine supplies those from a real
structure (see ``PottsMPNNPHEngine.run_switch_design``); tests supply synthetic ones.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from mpnn.ph.binding import (
    DEFAULT_MODEL_PKA,
    BindingModel,
    EnergyScorer,
    SystemView,
    linked_sites_from_spec,
    make_binding_model,
)
from mpnn.ph.config import DesignableSpec, SwitchDesignConfig
from mpnn.ph.ensemble import EnsembleMember, EnsembleObjective
from mpnn.ph.neighbourhood import cap_by_rank, near_ranks
from mpnn.ph.objective import Objective, TermContext
from mpnn.ph.report import site_report, validate_report_tokens
from mpnn.ph.search import SearchResult, run_search
from mpnn.ph.states import (
    ResolvedCondition,
    SiteKey,
    resolve_spec,
    site_index_from_arrays,
)
from mpnn.ph.vocab_meta import TokenTable

logger = logging.getLogger(__name__)

_LINKED_OPTIONS = {"beta", "max_sites", "pka_by_parent", "pka_by_site"}


@dataclass
class DesignInputs:
    """What a design run needs to know about one structure.

    ``tokens`` is the complex's token tensor ``[L]``; ``chain_ids``, ``res_ids`` and
    ``parents`` (bare residue names) describe each position. The binder-alone and
    receptor-alone scorers cover ``binder_positions`` / ``receptor_positions``, listed
    in each scorer's own residue order.
    """

    tokens: torch.Tensor
    table: TokenTable
    chain_ids: Sequence[str]
    res_ids: Sequence[int]
    parents: Sequence[str]
    complex_scorer: EnergyScorer
    binder_scorer: EnergyScorer
    receptor_scorer: EnergyScorer
    binder_positions: Sequence[int]
    receptor_positions: Sequence[int]
    partner_rank: Optional[Callable[[int], Sequence[int]]] = None
    # the model's contact table E_idx [L, K] (slot 0 = the residue itself); needed by designable.near
    neighbour_index: Optional[np.ndarray] = None
    # reference pKa of each titratable site in the free partners (PROPKA, filled by the engine);
    # used by linked_equilibrium when config.protonation.reference_pka == "propka"
    site_pka: Optional[Mapping[SiteKey, float]] = None


@dataclass
class SwitchDesignRun:
    """A finished run with the pieces needed to inspect or re-score it."""

    result: SearchResult
    objective: Objective
    designable: List[int]
    conditions: Mapping[str, ResolvedCondition]
    inputs: DesignInputs
    report_tokens: Dict[str, Any] = field(default_factory=dict, kw_only=True)

    def site_report(self, record, beta: Optional[float] = None) -> List[Dict[str, Any]]:
        """Per-condition, per-site comparison of the complex with the free partners.

        See :func:`mpnn.ph.report.site_report`; one row per condition and titratable site.
        """
        return site_report(
            self.objective.ctx,
            self.conditions,
            self.inputs,
            record.tokens,
            beta,
            self.report_tokens,
        )

    def to_rows(
        self, chain: str, reference_sequence: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Plain dicts, one per design in ranked order, for tables and downstream code.

        ``n_mutations`` is the Hamming distance of ``chain``'s canonical sequence to
        ``reference_sequence``, by default the input structure's own sequence for that
        chain. Term values appear as ``term_<name>``.
        """
        positions = [i for i, c in enumerate(self.inputs.chain_ids) if str(c) == chain]
        if reference_sequence is None:
            reference_sequence = self.inputs.table.canonical_sequence(
                [int(t) for t in self.inputs.tokens[positions]]
            )
        rows = []
        for rank, record in enumerate(self.result.records):
            design = self.describe(record, chain)
            if len(reference_sequence) != len(design["canonical_sequence"]):
                raise ValueError("reference_sequence does not match the chain length.")
            rows.append(
                {
                    "rank": rank,
                    "chain": chain,
                    "canonical_sequence": design["canonical_sequence"],
                    "extended_tokens": " ".join(design["extended_tokens"]),
                    "n_mutations": sum(
                        a != b
                        for a, b in zip(
                            reference_sequence, design["canonical_sequence"]
                        )
                    ),
                    "mode": self.result.plan.mode,
                    "front": record.front,
                    "selected": record.selected,
                    "scalarised": record.scalarised,
                    "weights": dict(record.weights),
                    "run_index": record.run_index,
                    "seed_index": record.seed_index,
                    **{f"term_{k}": v for k, v in record.term_values.items()},
                }
            )
        return rows

    def describe(self, record, chain: str) -> Dict[str, Any]:
        """One chain of a design as a canonical sequence and its token names.

        ``extended_tokens`` keeps protonation states for residues that carry them
        (residues set by a condition show the state of the *input* structure).
        """
        positions = [i for i, c in enumerate(self.inputs.chain_ids) if str(c) == chain]
        if not positions:
            raise ValueError(f"Chain {chain!r} is not in the structure.")
        tokens = [int(t) for t in record.tokens[positions]]
        table = self.inputs.table
        return {
            "chain": chain,
            "res_ids": [int(self.inputs.res_ids[i]) for i in positions],
            "canonical_sequence": table.canonical_sequence(tokens),
            "extended_tokens": table.token_names(tokens),
        }


def designable_with_ranks(
    spec: DesignableSpec,
    chain_ids: Sequence[str],
    res_ids: Sequence[int],
    spec_sites: Sequence[SiteKey] = (),
    neighbour_index: Optional[np.ndarray] = None,
    allow_empty: bool = False,
) -> Tuple[List[int], Dict[int, int]]:
    """Designable positions before any ``near.max_mutations`` cap, with their closeness ranks.

    Residues set by a condition are never designable. With ``spec.near`` the set is
    restricted to the residues coupled to the named sites in ``neighbour_index`` (the
    model's contact table), and ``ranks`` orders them from closest; without it ``ranks`` is
    empty. The cap is applied separately so that several target states can share one.
    An empty set raises unless ``allow_empty``: in an ensemble a state may have no
    neighbours of a site while the union over states still has some.
    """
    index = site_index_from_arrays(chain_ids, res_ids)
    chains = set(spec.chains)
    unknown = chains - {str(c) for c in chain_ids}
    if unknown:
        raise ValueError(f"Designable chains not in the structure: {sorted(unknown)}")
    positions = {i for i, c in enumerate(chain_ids) if str(c) in chains}
    for text in spec.include:
        positions.add(_lookup(index, text))
    for text in spec.exclude:
        positions.discard(_lookup(index, text))
    fixed_by_spec = {index[site] for site in spec_sites if site in index}
    overlap = positions & fixed_by_spec
    if overlap:
        logger.warning(
            "Removing %d residue(s) set by a condition from the designable set.",
            len(overlap),
        )
        positions -= fixed_by_spec
    if spec.near is not None and positions:
        if neighbour_index is None:
            raise ValueError(
                "designable.near needs the model's contact table (DesignInputs.neighbour_index)."
            )
        centres = [_lookup(index, text) for text in spec.near.sites]
        ranks = near_ranks(neighbour_index, centres, positions, k=spec.near.k)
        positions = set(ranks)
    else:
        ranks = {}
    if not positions and not allow_empty:
        raise ValueError("No designable positions: check 'designable' in the config.")
    return sorted(positions), ranks


def _cap(spec: DesignableSpec) -> int:
    return spec.near.max_mutations if spec.near is not None else 0


def resolve_designable(
    spec: DesignableSpec,
    chain_ids: Sequence[str],
    res_ids: Sequence[int],
    spec_sites: Sequence[SiteKey] = (),
    neighbour_index: Optional[np.ndarray] = None,
) -> List[int]:
    """Positions that may change: :func:`designable_with_ranks` with the cap applied."""
    positions, ranks = designable_with_ranks(
        spec, chain_ids, res_ids, spec_sites, neighbour_index
    )
    return cap_by_rank(positions, ranks, _cap(spec))


def _lookup(index: Mapping[SiteKey, int], text: str) -> int:
    key = SiteKey.parse(text)
    if key not in index:
        raise ValueError(f"Site {text} is not in the structure.")
    return index[key]


PKAD_HIS_CAVEAT = (
    "PROPKA's reference pKa for His did not beat the per-type constant on PKAD "
    "(Pearson r 0.15, mean error 1.03 against 0.80). Set His sites with "
    "binding_options.pka_by_site if you prefer another value."
)


def _reference_pka_by_site(
    config: SwitchDesignConfig,
    inputs: DesignInputs,
    explicit: Optional[Mapping[str, float]],
) -> Optional[Dict[SiteKey, float]]:
    """Per-site reference pKa for linked_equilibrium: PROPKA's when ``protonation.reference_pka`` is
    ``"propka"``, then the user's own ``binding_options.pka_by_site`` (``{"A:32": 5.9}``) on top.
    Logs where each value of a site the conditions name comes from."""
    by_site: Dict[SiteKey, float] = {}
    origin: Dict[SiteKey, str] = {}
    if config.protonation.reference_pka == "propka":
        if inputs.site_pka is None:
            raise ValueError(
                "protonation.reference_pka 'propka' needs the free partners' PROPKA pKa values "
                "(DesignInputs.site_pka); the engine computes them."
            )
        by_site.update(inputs.site_pka)
        origin.update({key: "PROPKA on the free partner" for key in inputs.site_pka})
    for text, value in (explicit or {}).items():
        key = SiteKey.parse(text)
        by_site[key] = float(value)
        origin[key] = "binding_options.pka_by_site"
    site_index = site_index_from_arrays(inputs.chain_ids, inputs.res_ids)
    from_propka_his = False
    for key in sorted(config.spec.sites()):
        if key not in by_site or key not in site_index:
            continue
        parent = inputs.parents[site_index[key]]
        constant = DEFAULT_MODEL_PKA.get(parent)
        logger.info(
            "Reference pKa of %s (%s): %.2f from %s; the per-type constant is %s.",
            key,
            parent,
            by_site[key],
            origin[key],
            constant if constant is not None else "not defined",
        )
        from_propka_his |= parent == "HIS" and origin[key].startswith("PROPKA")
    if from_propka_his:
        logger.warning(PKAD_HIS_CAVEAT)
    return by_site or None


def _binding_model(
    config: SwitchDesignConfig,
    inputs: DesignInputs,
    site_index: Mapping[SiteKey, int],
    complex_view: SystemView,
    binder_view: SystemView,
    receptor_view: SystemView,
) -> BindingModel:
    name = config.binding_model
    options: Dict[str, Any] = dict(config.binding_options)
    if name == "complex_gap":
        _no_options(name, options)
        return make_binding_model(name, complex_view=complex_view)
    views = dict(
        complex_view=complex_view,
        binder_view=binder_view,
        receptor_view=receptor_view,
    )
    if name == "state_binding":
        _no_options(name, options)
        return make_binding_model(name, **views)
    if name == "linked_equilibrium":
        unknown = set(options) - _LINKED_OPTIONS
        if unknown:
            raise ValueError(f"Unknown linked_equilibrium options: {sorted(unknown)}")
        sites = linked_sites_from_spec(
            config.spec,
            inputs.table,
            site_index,
            inputs.parents,
            options.pop("pka_by_parent", None),
            pka_by_site=_reference_pka_by_site(
                config, inputs, options.pop("pka_by_site", None)
            ),
        )
        return make_binding_model(name, sites=sites, **views, **options)
    return make_binding_model(name, **views, **options)  # raises for unknown names


def _no_options(name: str, options: Mapping[str, Any]) -> None:
    if options:
        raise ValueError(f"{name} takes no binding_options (got {sorted(options)}).")


def _check_switch_can_differ(
    config: SwitchDesignConfig, conditions: Mapping[str, ResolvedCondition]
) -> None:
    """Refuse a switch term whose ``on`` and ``off`` conditions cannot differ.

    The term's own ``on`` / ``off`` override the spec's, as when it is scored. With the
    state-based binding models the energies depend on a condition only through its
    states, so identical states make the switch exactly zero for every design.
    ``linked_equilibrium`` sums over every site a condition imposes, so only the pH
    tells the conditions apart (and it needs at least one such site).
    """
    _, terms = config.weights_and_terms()
    switch = terms.get("switch")
    if switch is None:
        return
    on, off = switch.on or config.spec.on, switch.off or config.spec.off
    if on not in conditions or off not in conditions:
        return  # the term reports a missing or unknown condition when it is scored
    if config.binding_model == "linked_equilibrium":
        if conditions[on].ph != conditions[off].ph and config.spec.sites():
            return
        reason = (
            "have the same pH"
            if conditions[on].ph == conditions[off].ph
            else "impose no state on any site"
        )
        hint = "give them different pH values and name the sites that titrate"
    else:
        if conditions[on].token_by_position != conditions[off].token_by_position:
            return
        reason = "impose identical states"
        hint = (
            "check the condition keys (for example 'states', not 'state') and the "
            "states you assigned"
        )
    raise ValueError(
        f"The {on!r} and {off!r} conditions {reason}, so the switch term is zero for "
        f"every design. {hint[0].upper()}{hint[1:]}."
    )


def _prepare_state(
    inputs: DesignInputs, config: SwitchDesignConfig, allow_empty: bool = False
):
    """Resolve one structure's states and build its objective and designable set."""
    n_positions = len(inputs.tokens)
    vocab_size = len(inputs.table)
    site_index = site_index_from_arrays(inputs.chain_ids, inputs.res_ids)
    conditions = resolve_spec(config.spec, inputs.table, site_index, inputs.parents)
    _check_switch_can_differ(config, conditions)
    validate_report_tokens(inputs.table, config.report_tokens)

    complex_view = SystemView(inputs.complex_scorer, range(n_positions), vocab_size)
    binder_view = SystemView(inputs.binder_scorer, inputs.binder_positions, vocab_size)
    receptor_view = SystemView(
        inputs.receptor_scorer, inputs.receptor_positions, vocab_size
    )
    binding = _binding_model(
        config, inputs, site_index, complex_view, binder_view, receptor_view
    )
    context = TermContext(
        binding=binding,
        binder_view=binder_view,
        complex_view=complex_view,
        conditions=conditions,
        on=config.spec.on,
        off=config.spec.off,
        receptor_view=receptor_view,
    )
    _, terms = config.weights_and_terms()
    objective = Objective(context, terms)
    designable, ranks = designable_with_ranks(
        config.designable,
        inputs.chain_ids,
        inputs.res_ids,
        sorted(config.spec.sites()),
        inputs.neighbour_index,
        allow_empty=allow_empty,
    )
    return objective, conditions, designable, ranks


def _search(objective, inputs, designable, config) -> SearchResult:
    valid_mask = torch.tensor(
        inputs.table.design_mask(config.allow_bare_titratable), dtype=torch.bool
    )
    return run_search(
        objective,
        inputs.tokens,
        designable,
        config.plan(),
        valid_mask=valid_mask,
        block_size=config.block_size,
        partner_rank=inputs.partner_rank,
        temperature=config.temperature,
        n_seeds=config.n_seeds,
        base_seed=config.base_seed,
        max_rounds=config.max_rounds,
        n_select=config.n_select,
    )


def run_switch_design(
    inputs: DesignInputs, config: SwitchDesignConfig
) -> SwitchDesignRun:
    """Resolve the states, build the objective and run the planned search."""
    objective, conditions, designable, ranks = _prepare_state(inputs, config)
    designable = cap_by_rank(designable, ranks, _cap(config.designable))
    result = _search(objective, inputs, designable, config)
    return SwitchDesignRun(
        result,
        objective,
        designable,
        conditions,
        inputs,
        report_tokens=dict(config.report_tokens),
    )


# ---- several target states sharing one binder ------------------------------------------
@dataclass
class EnsembleRun(SwitchDesignRun):
    """A run over several target states; ``inputs`` and ``conditions`` are the first state's."""

    members: Dict[str, EnsembleMember]

    def to_rows(self, chain: str, reference_sequence: Optional[str] = None):
        """As :meth:`SwitchDesignRun.to_rows`, plus ``term_<name>@<state>`` columns."""
        rows = super().to_rows(chain, reference_sequence)
        for row, record in zip(rows, self.result.records):
            for state, values in (record.per_state or {}).items():
                for term, value in values.items():
                    row[f"term_{term}@{state}"] = value
        return rows

    def site_report(self, record, beta: Optional[float] = None):
        """The per-site report of every target state, with a ``state`` column."""
        rows = []
        for member in self.members.values():
            tokens = self.objective.tokens_for(member, record.tokens)
            for row in site_report(
                member.objective.ctx,
                member.conditions,
                member.inputs,
                tokens,
                beta,
                self.report_tokens,
            ):
                rows.append({"state": member.name, **row})
        return rows


def _binder_keys(inputs: DesignInputs) -> Dict[SiteKey, int]:
    return {
        SiteKey(str(inputs.chain_ids[p]), int(inputs.res_ids[p])): int(p)
        for p in inputs.binder_positions
    }


def _check_shared_binder(
    reference_name: str,
    reference: DesignInputs,
    name: str,
    inputs: DesignInputs,
) -> Dict[int, int]:
    """Map reference binder positions to ``inputs``' and check the binder really is shared."""
    if inputs.table.names != reference.table.names:
        raise ValueError(
            f"State {name!r} uses a different vocabulary from {reference_name!r}."
        )
    ref_keys, keys = _binder_keys(reference), _binder_keys(inputs)
    if set(ref_keys) != set(keys):
        raise ValueError(
            f"State {name!r} has different binder residues from {reference_name!r}; "
            "the binder must be shared."
        )
    different = [
        str(k)
        for k in ref_keys
        if reference.parents[ref_keys[k]] != inputs.parents[keys[k]]
    ]
    if different:
        raise ValueError(
            f"State {name!r}: the binder sequence differs from {reference_name!r} at {different[:5]}."
        )
    return {ref_keys[k]: keys[k] for k in ref_keys}


def run_switch_design_ensemble(
    inputs_by_state: Mapping[str, DesignInputs], config: SwitchDesignConfig
) -> EnsembleRun:
    """Design one binder against several target states, one complex per state.

    The first state is the reference: designed tokens use its indexing and its starting
    binder sequence. Every state must contain the same binder (same residues and
    sequence); the conditions' sites must exist in every state. Terms are evaluated per
    state and reduced by ``config.target_reduce`` (``max`` = worst state).
    """
    if not inputs_by_state:
        raise ValueError("Need at least one target state.")
    names = list(inputs_by_state)
    reference_name, reference = names[0], inputs_by_state[names[0]]
    members: Dict[str, EnsembleMember] = {}
    designable_keys = set()
    shared_ranks: Dict[int, int] = {}
    reference_keys = _binder_keys(reference)
    for name in names:
        inputs = inputs_by_state[name]
        position_map = (
            {p: p for p in reference_keys.values()}
            if name == reference_name
            else _check_shared_binder(reference_name, reference, name, inputs)
        )
        try:
            objective, conditions, designable, ranks = _prepare_state(
                inputs, config, allow_empty=True
            )
        except (ValueError, KeyError) as err:
            raise type(err)(f"State {name!r}: {err}") from None
        reverse = {v: k for k, v in position_map.items()}
        outside = [p for p in designable if p not in reverse]
        if outside:
            raise ValueError(
                f"State {name!r}: designable positions {outside[:5]} are not shared binder "
                "residues; only the shared binder can be designed in an ensemble."
            )
        designable_keys |= {reverse[p] for p in designable}
        for p, rank in ranks.items():  # best rank over the states, on the shared binder
            if p in reverse:
                shared_ranks[reverse[p]] = min(rank, shared_ranks.get(reverse[p], rank))
        members[name] = EnsembleMember(
            name, objective, conditions, inputs, inputs.tokens.clone(), position_map
        )
    # one total cap on the shared set; capping each state and uniting would exceed it
    designable = cap_by_rank(
        sorted(designable_keys), shared_ranks, _cap(config.designable)
    )
    if not designable:
        raise ValueError(
            "No designable positions in any state: check 'designable' in the config."
        )
    objective = EnsembleObjective(
        list(members.values()), reduce=config.target_reduce, designable=designable
    )
    result = _search(objective, reference, designable, config)
    return EnsembleRun(
        result,
        objective,
        designable,
        members[reference_name].conditions,
        reference,
        members=members,
        report_tokens=dict(config.report_tokens),
    )
