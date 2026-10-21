"""Binding-energy models: how strongly a state assignment holds two partners together.

All energies follow the engine convention, lower = better (tighter binding). A model
sees one token tensor for the whole complex (condition overrides already applied) and
returns either the absolute energy or the block potentials around chosen positions.

Two models ship today, behind one interface so objectives can swap them:

* ``complex_gap``    G = H_complex                         (what the engine does now)
* ``state_binding``  G = H_complex - H_binder - H_receptor (interface part only)
* ``linked_equilibrium``  pH-continuous: the free energy of binding averaged over the
  protonation states the unbound partners populate at that pH (experimental)

``state_binding`` removes everything that exists whether or not the partners touch,
including the energy change of a receptor residue flipping its own state. Both need
Potts scorers; the isolated-chain scorers come from re-encoding each chain alone.
"""

from __future__ import annotations

import abc
import itertools
import logging
import math
from dataclasses import dataclass
from typing import (
    ClassVar,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Type,
)

import torch

from mpnn.ph.potentials import BlockPotentials
from mpnn.ph.states import SiteKey, StateSpec
from mpnn.ph.vocab_meta import TokenTable

logger = logging.getLogger(__name__)


class EnergyScorer(Protocol):
    """What a Potts scorer must offer (the engine's scorer is wrapped to match)."""

    def H_of(self, tokens: torch.Tensor) -> float: ...

    def block_potentials(
        self, tokens: torch.Tensor, block: Sequence[int]
    ) -> BlockPotentials: ...


class SystemView:
    """A scorer together with the complex positions it covers, in the scorer's order."""

    def __init__(
        self, scorer: EnergyScorer, positions: Sequence[int], vocab_size: int
    ) -> None:
        positions = [int(p) for p in positions]
        if len(set(positions)) != len(positions):
            raise ValueError("A system cannot list the same position twice.")
        self.scorer = scorer
        self.positions = positions
        self.vocab_size = int(vocab_size)
        self._local = {p: i for i, p in enumerate(positions)}

    def energy(self, tokens: torch.Tensor) -> float:
        return float(self.scorer.H_of(tokens[self.positions]))

    def block_potentials(
        self, tokens: torch.Tensor, block: Sequence[int]
    ) -> BlockPotentials:
        """Absolute potentials of this system over ``block`` (complex positions).

        Block positions outside the system contribute nothing; if none are inside, the
        result is the constant energy of the system.
        """
        n = len(block)
        sub_tokens = tokens[self.positions]
        total = float(self.scorer.H_of(sub_tokens))
        present = [
            (bi, self._local[p]) for bi, p in enumerate(block) if p in self._local
        ]
        if not present:
            return BlockPotentials(
                torch.zeros(
                    n, self.vocab_size, dtype=torch.float64, device=tokens.device
                ),
                {},
                total,
            )
        slots = [bi for bi, _ in present]
        sub_block = [local for _, local in present]
        potentials = self.scorer.block_potentials(sub_tokens, sub_block)
        reference = [int(sub_tokens[local]) for local in sub_block]
        potentials = potentials.with_reference(reference, total)
        return potentials.embedded(n, slots)


class BindingModel(abc.ABC):
    """Interface: absolute binding energy and its block values.

    ``ph`` is the pH of the condition being scored. State-based models ignore it (the
    condition's states are already in ``tokens``); pH-continuous models require it.
    """

    name: ClassVar[str]

    @abc.abstractmethod
    def energy(self, tokens: torch.Tensor, ph: Optional[float] = None) -> float:
        """Binding energy of the full token assignment ``tokens`` (lower = tighter)."""

    def block_potentials(
        self, tokens: torch.Tensor, block: Sequence[int]
    ) -> BlockPotentials:
        """Additive unary + pairwise decomposition over ``block``, when one exists."""
        raise NotImplementedError(
            f"{self.name} has no additive block decomposition; use block_values."
        )

    def block_values(
        self, tokens: torch.Tensor, block: Sequence[int], ph: Optional[float] = None
    ) -> torch.Tensor:
        """Binding energy for every joint assignment of ``block``: shape ``[V] * B``."""
        return self.block_potentials(tokens, block).joint()


class ComplexGap(BindingModel):
    """G = H_complex. Includes every intramolecular term, so it measures how much the
    complex likes a state, not how much that state changes binding."""

    name = "complex_gap"

    def __init__(self, complex_view: SystemView) -> None:
        self.complex_view = complex_view

    def energy(self, tokens: torch.Tensor, ph: Optional[float] = None) -> float:
        return self.complex_view.energy(tokens)

    def block_potentials(self, tokens, block) -> BlockPotentials:
        return self.complex_view.block_potentials(tokens, block)


class StateBinding(BindingModel):
    """G = H_complex - H_binder_alone - H_receptor_alone, all with the same tokens."""

    name = "state_binding"

    def __init__(
        self,
        complex_view: SystemView,
        binder_view: SystemView,
        receptor_view: SystemView,
    ) -> None:
        covered = set(complex_view.positions)
        binder, receptor = set(binder_view.positions), set(receptor_view.positions)
        if binder & receptor:
            raise ValueError("Binder and receptor systems must not share positions.")
        if not (binder | receptor) <= covered:
            raise ValueError("Binder/receptor positions must lie inside the complex.")
        self.complex_view = complex_view
        self.binder_view = binder_view
        self.receptor_view = receptor_view

    def energy(self, tokens: torch.Tensor, ph: Optional[float] = None) -> float:
        return (
            self.complex_view.energy(tokens)
            - self.binder_view.energy(tokens)
            - self.receptor_view.energy(tokens)
        )

    def block_potentials(self, tokens, block) -> BlockPotentials:
        return (
            self.complex_view.block_potentials(tokens, block)
            - self.binder_view.block_potentials(tokens, block)
            - self.receptor_view.block_potentials(tokens, block)
        )


# Approximate model-compound pKa values (e.g. Thurlkill et al., Protein Sci. 2006).
# Defaults only: review them against your own source before relying on pH predictions.
DEFAULT_MODEL_PKA: Dict[str, float] = {"ASP": 3.7, "GLU": 4.3, "HIS": 6.5}
LOG10 = math.log(10.0)
DEFAULT_MAX_MICROSTATE_ELEMENTS = 50_000_000


@dataclass(frozen=True)
class TitratableSite:
    """A residue whose protonation the linked-equilibrium model sums over."""

    position: int
    protonated_token: int
    deprotonated_token: int
    pka: float


def linked_sites_from_spec(
    spec: StateSpec,
    table: TokenTable,
    site_index: Mapping[SiteKey, int],
    parents: Sequence[str],
    pka_by_parent: Optional[Mapping[str, float]] = None,
    pka_by_site: Optional[Mapping[SiteKey, float]] = None,
) -> List[TitratableSite]:
    """One :class:`TitratableSite` per residue the spec mentions, on either chain.

    A site's reference pKa is ``pka_by_site[site]`` when given, else the value for its residue
    type in ``pka_by_parent`` (default :data:`DEFAULT_MODEL_PKA`). It is the pKa of the free
    partner; what binding adds comes from the Potts energies.
    """
    pka_by_parent = dict(DEFAULT_MODEL_PKA if pka_by_parent is None else pka_by_parent)
    sites = []
    for key in sorted(spec.sites()):
        if key not in site_index:
            raise KeyError(f"Site {key} is not in the structure.")
        position = site_index[key]
        parent = parents[position]
        if pka_by_site and key in pka_by_site:
            pka = float(pka_by_site[key])
        elif parent in pka_by_parent:
            pka = float(pka_by_parent[parent])
            if pka_by_site:
                logger.warning(
                    "No per-site pKa for %s (%s), using the per-type value %.1f.",
                    key,
                    parent,
                    pka,
                )
        else:
            raise KeyError(f"No reference pKa for {parent} (site {key}).")
        if not math.isfinite(pka):
            raise ValueError(f"The reference pKa of {key} must be finite (got {pka}).")
        sites.append(
            TitratableSite(
                position=position,
                protonated_token=table.index(table.resolve(parent, "protonated")),
                deprotonated_token=table.index(table.resolve(parent, "deprotonated")),
                pka=pka,
            )
        )
    return sites


class LinkedEquilibrium(BindingModel):
    """pH-continuous binding free energy over protonation microstates (experimental).

    For titratable sites with reference pKa values, a microstate ``m`` (a protonation
    vector) of the *unbound* partners has weight
    ``exp(-beta*(H_binder + H_receptor)(m)) * 10**sum_i x_i (pKa_i - pH)``, ``x_i = 1``
    when site ``i`` is protonated. The binding free energy at that pH is

        G(pH) = -(1/beta) ln sum_m p_free(m | pH) exp(-beta * G_bind(m)),

    with ``G_bind(m) = H_complex(m) - H_binder(m) - H_receptor(m)``: the Boltzmann
    average of the state-specific binding energy over the unbound ensemble. It reduces
    to ``state_binding`` at the pH extremes. Sites must be few (``2**K`` microstates),
    and ``beta`` (Potts units to 1/kT) is not calibrated: treat results as model units.
    Residues outside ``sites`` keep the tokens they have in ``tokens``.
    """

    name = "linked_equilibrium"

    def __init__(
        self,
        complex_view: SystemView,
        binder_view: SystemView,
        receptor_view: SystemView,
        sites: Sequence[TitratableSite],
        beta: float = 1.0,
        max_sites: int = 12,
    ) -> None:
        if beta <= 0:
            raise ValueError("beta must be positive.")
        if len(sites) > max_sites:
            raise ValueError(
                f"{len(sites)} titratable sites means 2**{len(sites)} microstates; "
                f"the cap is {max_sites} sites."
            )
        positions = [site.position for site in sites]
        if len(set(positions)) != len(positions):
            raise ValueError("Titratable sites must be distinct positions.")
        covered = set(complex_view.positions)
        if not set(positions) <= covered:
            raise ValueError("Titratable sites must lie inside the complex.")
        self.complex_view = complex_view
        self.binder_view = binder_view
        self.receptor_view = receptor_view
        self.sites = list(sites)
        self.beta = float(beta)
        self._states = list(itertools.product((0, 1), repeat=len(self.sites)))

    # ---- microstates --------------------------------------------------------
    def _microstate_tokens(
        self, tokens: torch.Tensor, state: Sequence[int]
    ) -> torch.Tensor:
        out = tokens.clone()
        for site, protonated in zip(self.sites, state):
            out[site.position] = (
                site.protonated_token if protonated else site.deprotonated_token
            )
        return out

    def _log_weight(self, state: Sequence[int], ph: float) -> float:
        return LOG10 * sum(x * (s.pka - ph) for x, s in zip(state, self.sites))

    @staticmethod
    def _need_ph(ph: Optional[float]) -> float:
        if ph is None:
            raise ValueError("linked_equilibrium needs the pH of the condition.")
        return float(ph)

    # ---- values ---------------------------------------------------------------
    def energy(self, tokens: torch.Tensor, ph: Optional[float] = None) -> float:
        ph = self._need_ph(ph)
        log_bound = log_free = None
        for state in self._states:
            micro = self._microstate_tokens(tokens, state)
            weight = self._log_weight(state, ph)
            h_complex = self.complex_view.energy(micro)
            h_free = self.binder_view.energy(micro) + self.receptor_view.energy(micro)
            bound = torch.tensor(-self.beta * h_complex + weight, dtype=torch.float64)
            free = torch.tensor(-self.beta * h_free + weight, dtype=torch.float64)
            log_bound = (
                bound if log_bound is None else torch.logaddexp(log_bound, bound)
            )
            log_free = free if log_free is None else torch.logaddexp(log_free, free)
        return float(-(log_bound - log_free) / self.beta)

    def block_values(
        self, tokens: torch.Tensor, block: Sequence[int], ph: Optional[float] = None
    ) -> torch.Tensor:
        ph = self._need_ph(ph)
        overlap = {s.position for s in self.sites} & {int(p) for p in block}
        if overlap:
            raise ValueError(f"Block positions {sorted(overlap)} are titratable sites.")
        vocab = self.complex_view.vocab_size
        if len(self._states) * vocab ** len(block) > DEFAULT_MAX_MICROSTATE_ELEMENTS:
            raise ValueError(
                f"{len(self._states)} microstates x {vocab}**{len(block)} block "
                "assignments is too large; use a smaller block or fewer sites."
            )
        log_bound = log_free = None
        for state in self._states:
            micro = self._microstate_tokens(tokens, state)
            weight = self._log_weight(state, ph)
            h_complex = self.complex_view.block_potentials(micro, block).joint()
            h_free = (
                self.binder_view.block_potentials(micro, block)
                + self.receptor_view.block_potentials(micro, block)
            ).joint()
            bound = -self.beta * h_complex + weight
            free = -self.beta * h_free + weight
            log_bound = (
                bound if log_bound is None else torch.logaddexp(log_bound, bound)
            )
            log_free = free if log_free is None else torch.logaddexp(log_free, free)
        return -(log_bound - log_free) / self.beta

    def site_populations(
        self, tokens: torch.Tensor, ph: float
    ) -> Dict[int, Dict[str, float]]:
        """Protonated fraction of each site in the unbound and the bound ensemble.

        A site that is more protonated when bound than when free has a raised pKa on
        binding; the opposite means binding lowers it.
        """
        ph = self._need_ph(ph)
        bound, free = [], []
        for state in self._states:
            micro = self._microstate_tokens(tokens, state)
            weight = self._log_weight(state, ph)
            h_free = self.binder_view.energy(micro) + self.receptor_view.energy(micro)
            bound.append(-self.beta * self.complex_view.energy(micro) + weight)
            free.append(-self.beta * h_free + weight)
        p_bound = torch.softmax(torch.tensor(bound, dtype=torch.float64), dim=0)
        p_free = torch.softmax(torch.tensor(free, dtype=torch.float64), dim=0)
        states = torch.tensor(self._states, dtype=torch.float64).reshape(
            len(self._states), len(self.sites)
        )
        return {
            site.position: {
                "free": float(p_free @ states[:, i]),
                "bound": float(p_bound @ states[:, i]),
            }
            for i, site in enumerate(self.sites)
        }


BINDING_MODELS: Dict[str, Type[BindingModel]] = {
    ComplexGap.name: ComplexGap,
    StateBinding.name: StateBinding,
    LinkedEquilibrium.name: LinkedEquilibrium,
}


def make_binding_model(name: str, **kwargs) -> BindingModel:
    """Build a binding model by config name from its constructor arguments."""
    try:
        cls = BINDING_MODELS[name]
    except KeyError:
        raise ValueError(
            f"Unknown binding model {name!r}; choose from {sorted(BINDING_MODELS)}."
        ) from None
    return cls(**kwargs)
