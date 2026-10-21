"""Configuration for a pH-switch design run, parsed strictly from a plain dict.

Unknown keys raise instead of being ignored, so a misspelt option cannot silently fall
back to a default. Defaults live here; command-line flags (or a YAML file) override them.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from mpnn.ph.objective import SearchPlan, build_terms, plan_search
from mpnn.ph.report import validate_report_token_shapes
from mpnn.ph.states import StateSpec

logger = logging.getLogger(__name__)

_TOP_LEVEL = {
    "conditions",
    "on",
    "off",
    "binding_model",
    "binding_options",
    "terms",
    "search",
    "scalarisation",
    "divisions",
    "block_size",
    "temperature",
    "max_rounds",
    "n_seeds",
    "base_seed",
    "n_select",
    "designable",
    "allow_bare_titratable",
    "target_reduce",
    "report_tokens",
    "protonation",
}
_PROTONATION = {"states", "reference_pka", "ph", "margin"}
STATE_SOURCES = ("ev6", "propka")
REFERENCE_PKA_SOURCES = ("constant", "propka")
DEFAULT_PROPKA_MARGIN = 0.5
_DESIGNABLE = {"chains", "include", "exclude", "near"}
_NEAR = {"sites", "k", "max_mutations"}


@dataclass(frozen=True)
class ProtonationSources:
    """Two separate choices of what PROPKA is used for. Both default to not using it.

    ``states``: the starting protonation state of every His, Asp and Glu that no condition names.
    ``"ev6"`` (default) is the geometric classifier the checkpoint was trained with: one call per
    residue, independent of pH. ``"propka"`` reads PROPKA's pKa of each residue at the pH ``ph``
    (required): protonated when pKa > pH + ``margin``, neutral when pKa < pH - ``margin``,
    ambiguous (``-A``) in between (``margin`` defaults to 0.5). Caveats: the checkpoint was
    trained on EV6 labels; against experimental states PROPKA is a weak labeller (ROC AUC of its
    pKa 0.62 for His, 0.78 for Asp/Glu on the neutron structures), and PROPKA is run on the
    structure as given, so a designed or predicted backbone gives pKa values only as good as it is.

    ``reference_pka``: the pKa that ``linked_equilibrium`` (the only model that uses one) gives each
    site it sums over. ``"constant"`` (default) is one value per residue type (Asp 3.7, Glu 4.3,
    His 6.5). ``"propka"`` runs PROPKA on the free binder and the free receptor and uses its value
    for each site: the pKa shift that binding brings is already in the Potts energies, so the
    complex must not be used. Caveat: on PKAD (experimental pKa in monomeric proteins) PROPKA beats
    the constant for Asp (Pearson r 0.81, mean error 0.71 against 1.04), slightly for Glu, and not
    for His (r 0.15, 1.03 against 0.80); ``binding_options.pka_by_site`` sets single sites by hand.

    The two are independent. Residues named in a condition keep the state the condition gives them
    whatever ``states`` is.
    """

    states: str = "ev6"
    reference_pka: str = "constant"
    ph: Optional[float] = None
    margin: Optional[float] = None

    def __post_init__(self) -> None:
        if self.states not in STATE_SOURCES:
            raise ValueError(f"protonation.states must be one of {STATE_SOURCES}.")
        if self.reference_pka not in REFERENCE_PKA_SOURCES:
            raise ValueError(
                f"protonation.reference_pka must be one of {REFERENCE_PKA_SOURCES}."
            )
        if self.states == "ev6":
            if self.ph is not None or self.margin is not None:
                raise ValueError(
                    "protonation.ph and .margin only apply to protonation.states 'propka'."
                )
            return
        if self.ph is None or not math.isfinite(self.ph):
            raise ValueError(
                "protonation.states 'propka' needs a finite protonation.ph."
            )
        if self.margin is None:
            object.__setattr__(self, "margin", DEFAULT_PROPKA_MARGIN)
        if not math.isfinite(self.margin) or self.margin < 0:
            raise ValueError("protonation.margin must be a finite number >= 0.")

    @classmethod
    def from_dict(cls, cfg: Optional[Mapping[str, Any]]) -> "ProtonationSources":
        cfg = dict(cfg or {})
        unknown = set(cfg) - _PROTONATION
        if unknown:
            raise ValueError(f"Unknown protonation keys: {sorted(unknown)}")
        return cls(**cfg)


@dataclass(frozen=True)
class NearSpec:
    """Keep only residues coupled to ``sites`` in the model's contact graph.

    ``k`` caps each site's neighbours (0 = its full list); ``max_mutations`` caps the total
    number of designable positions, keeping the closest (0 = no cap). The sites may be on
    either chain, for example a titratable residue on the target.
    """

    sites: List[str]
    k: int = 0
    max_mutations: int = 0

    def __post_init__(self) -> None:
        if not self.sites:
            raise ValueError("designable.near needs at least one site.")
        if self.k < 0 or self.max_mutations < 0:
            raise ValueError("designable.near: k and max_mutations must be >= 0.")


@dataclass(frozen=True)
class DesignableSpec:
    """Which residues may change: chains plus adds and removals, optionally restricted
    to the neighbourhood of chosen sites."""

    chains: List[str] = field(default_factory=list)
    include: List[str] = field(default_factory=list)  # sites like "A:12"
    exclude: List[str] = field(default_factory=list)
    near: Optional[NearSpec] = None


@dataclass(frozen=True)
class SwitchDesignConfig:
    spec: StateSpec
    terms: Dict[str, Dict[str, Any]]
    designable: DesignableSpec
    binding_model: str = "state_binding"
    binding_options: Dict[str, Any] = field(default_factory=dict)
    search: str = "single"
    scalarisation: Optional[str] = None
    divisions: int = 4
    block_size: int = 2
    temperature: float = 0.0
    max_rounds: int = 10
    n_seeds: int = 1
    base_seed: int = 0
    n_select: Optional[int] = None
    allow_bare_titratable: List[str] = field(default_factory=list)
    # several target states: "max" optimises the worst state, "mean" the average
    target_reduce: str = "max"
    # how site_report compares states a vocabulary names ambiguously (v3/v4 have two neutral
    # His tautomers): {"HIS": {"deprotonated": ["HID", "HIE"]}}; a list is averaged
    report_tokens: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # what PROPKA is used for: the starting states, the reference pKa values (both off by default)
    protonation: ProtonationSources = field(default_factory=ProtonationSources)

    def __post_init__(self) -> None:
        if self.block_size < 1 or self.max_rounds < 1 or self.n_seeds < 1:
            raise ValueError("block_size, max_rounds and n_seeds must be >= 1.")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0.")
        if self.n_select is not None and self.n_select < 1:
            raise ValueError("n_select must be >= 1 when given.")
        if self.divisions < 1:
            raise ValueError("divisions must be >= 1.")
        validate_report_token_shapes(self.report_tokens)
        if (
            self.protonation.reference_pka == "propka"
            and self.binding_model != "linked_equilibrium"
        ):
            raise ValueError(
                "protonation.reference_pka 'propka' only applies to the linked_equilibrium "
                f"binding model, the one that uses a pKa (binding_model is {self.binding_model!r})."
            )
        if self.target_reduce not in ("max", "mean"):
            raise ValueError("target_reduce must be 'max' or 'mean'.")
        self.plan()  # validates terms, search mode and scalarisation now

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any]) -> "SwitchDesignConfig":
        unknown = set(cfg) - _TOP_LEVEL
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        designable_cfg = dict(cfg.get("designable", {}))
        unknown = set(designable_cfg) - _DESIGNABLE
        if unknown:
            raise ValueError(f"Unknown designable keys: {sorted(unknown)}")
        if designable_cfg.get("near") is not None:
            near_cfg = dict(designable_cfg["near"])
            unknown = set(near_cfg) - _NEAR
            if unknown:
                raise ValueError(f"Unknown designable.near keys: {sorted(unknown)}")
            designable_cfg["near"] = NearSpec(**near_cfg)
        if not cfg.get("terms"):
            raise ValueError(
                "The config needs a 'terms' section with at least one term."
            )
        options = {
            key: cfg[key]
            for key in (
                "binding_model",
                "binding_options",
                "search",
                "scalarisation",
                "divisions",
                "block_size",
                "temperature",
                "max_rounds",
                "n_seeds",
                "base_seed",
                "n_select",
                "allow_bare_titratable",
                "target_reduce",
                "report_tokens",
            )
            if key in cfg
        }
        options["protonation"] = ProtonationSources.from_dict(cfg.get("protonation"))
        return cls(
            spec=StateSpec.from_dict(
                {k: cfg[k] for k in ("conditions", "on", "off") if k in cfg}
            ),
            terms={name: dict(body) for name, body in cfg["terms"].items()},
            designable=DesignableSpec(**designable_cfg),
            **options,
        )

    def weights_and_terms(self):
        """``(weights, term objects)`` of the active terms."""
        return build_terms(self.terms)

    def plan(self) -> SearchPlan:
        weights, _ = self.weights_and_terms()
        return plan_search(weights, self.search, self.scalarisation, self.divisions)
