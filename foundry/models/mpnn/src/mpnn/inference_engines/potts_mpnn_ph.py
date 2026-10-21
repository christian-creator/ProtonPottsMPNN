"""PottsMPNN-pH redesign engine (foundry wrapper).

Mirrors the ``MPNNWithLogProbs`` pattern: subclass the foundry PottsMPNN inference engine
(``mpnn.inference_engines.potts_mpnn.MPNNInferenceEngine``) so we reuse its checkpoint
loading, ``extended_vocab`` handling, transform-pipeline featurisation and device logic, and
add the protonation-aware redesign methods from the two research scripts
(``scripts/05_pmhc_binder_redesign_{potts,mpnn}.py`` in the conditional_binding/ph repo).

What this adds on top of the engine:
- ``PHDesignCriteria`` — config schema for a single design run (backend, method + knobs).
- design methods: ``autoregressive`` (masked decoder infill — the MPNN method),
  ``converged_mcmc``, ``two_phase`` and ``converged_mcmc_combined`` (placement-based, protonation-aware),
  plus the whole-chain baselines ``gibbs`` (Potts) and ``mpnn_sample`` (decoder).
- ``PHDesignOutput`` / ``PHDesignSet`` — the requested design-output classes that carry each
  design's sequence and its energy scores (final Potts H, centre selectivity, binder
  protonation ΔH, placement probability …), sortable lowest-energy-first.
- ``run_ph_redesign`` — featurise + forward ONCE, then run one or more criteria (a sweep over
  e.g. ``combined_lambda``) against the shared Potts tables and return a ``PHDesignSet``.

Starting sequence for placement (``seed_source``): ``native`` starts from the input-PDB / RFD3
sequence; ``inverse`` starts from externally supplied seeds (the inverse stage's sequences —
ProteinMPNN or the model's own MPNN sampling) passed in via ``initial_sequences``, running the
placement scan + neighbourhood redesign once per seed. MPNN seeds are produced by the inverse
stage, not re-sampled here — this engine only optimizes.

Conventions (unify the Potts energy head and the decoder log-prob field):
- the *design field* and all scores are ENERGIES with **lower = better**; the decoder field is
  ``-log P``. Invalid tokens are ``+inf``; sampling is ``softmax(-energy / T)``.
- vocab is the model's build-time choice; we read it back off the model
  (``graph_featurization_module.TOKEN_ENCODING``) and decode microstates to their canonical
  parent via ``mpnn.metrics.sequence_recovery._build_canonical_map`` + ``DICT_THREE_TO_ONE``
  (the foundry's own 21-token ``_decode_sequences`` cannot render the 11 protonation tokens).
"""

from __future__ import annotations

import copy
import logging
import multiprocessing as mp
import random
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

# Imported at runtime from the pH-sensitive foundry the GRPO .venv points at.
from atomworks.constants import DICT_THREE_TO_ONE, UNKNOWN_AA
from atomworks.ml.utils.token import get_token_starts
from biotite.structure import AtomArray
from mpnn.inference_engines.potts_mpnn import MPNNInferenceEngine
from mpnn.metrics.sequence_recovery import _build_canonical_map
from mpnn.model.layers.graph_embeddings import PottsProteinFeatures
from mpnn.model.pottsmpnn import PottsMPNN
from mpnn.ph.config import ProtonationSources, SwitchDesignConfig
from mpnn.ph.engine_adapter import (
    PottsScorerAdapter,
    knn_partner_rank,
    parent_names,
    positions_in_complex,
)
from mpnn.ph.session import DesignInputs, EnsembleRun, SwitchDesignRun
from mpnn.ph.session import run_switch_design as _run_switch_design
from mpnn.ph.session import run_switch_design_ensemble as _run_switch_design_ensemble
from mpnn.ph.states import SiteKey, site_index_from_arrays
from mpnn.ph.vocab_meta import TokenTable
from mpnn.potts_inference import prepare_potts_input

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Placement-based, protonation-aware methods (place a centre RES-P, redesign its neighbourhood):
#   autoregressive  — mask neighbours to UNK, decode one-by-one (the MPNN script's infill).
#   converged_mcmc  — random single-site mutations until the sequence settles.
#   two_phase       — freeze least-disruptive selective picks, then stability cleanup.
#   converged_mcmc_combined — converged MCMC on a LEGACY, unscaled (1+λ)·stability − λ·mean_dep blend.
#                     NOTE: this is NOT the canonical objective. block_descent (below) is the manuscript
#                     default and uses the z-scored Eq (6): O = (1−λ)·z(H_stab) + λ·z(Σ sel).
#   block_descent   — deterministic block coordinate descent: exact argmin over V**block_size joint
#                     assignments of each neighbour + its closest partners (near-exact MAP).
#   greedy_energy_block — CENTRE-FREE pure-stability variant of block_descent: each step re-ranks the
#                     designable positions by single-residue energy improvement, updates the top
#                     block_size jointly (exact V**block_size), and converges when nothing improves.
#                     No titratable centre is ever pinned (center_count=0, infill_scope=chain).
PLACEMENT_METHODS = ("autoregressive", "converged_mcmc", "two_phase", "converged_mcmc_combined",
                     "block_descent", "greedy_energy_block")
# Whole-chain baselines (no protonation placement; design the whole binder chain):
WHOLE_CHAIN_METHODS = ("gibbs", "mpnn_sample")
ALL_METHODS = PLACEMENT_METHODS + WHOLE_CHAIN_METHODS

# Protonated microstate -> deprotonated/neutral contrast partners (selective objective +
# placement selectivity). Mirrors DEP / RES_D in the 05 scripts.
DEFAULT_DEP_MAP: Dict[str, List[str]] = {
    "HIS-P": ["HID", "HIE"],
    "ASP-P": ["ASP-D"],
    "GLU-P": ["GLU-D"],
}
# Microstates forbidden as design *choices* (centre placement is set separately).
DEFAULT_FORBIDDEN_TOKENS: List[str] = ["HIS-A", "ASP-A", "GLU-A", "HIS-D"]


DEFAULT_INTERFACE_DISTANCE = 6.0   # angstroms, CA-CA; see PHDesignCriteria.interface_distance


@dataclass
class PHDesignCriteria:
    """Criteria for a single design run. Build from a hydra ``params`` dict via ``from_params``."""

    backend: str = "potts"                          # "potts" (energy head) | "mpnn" (decoder field)
    # Where the AUTOREGRESSIVE (method=autoregressive) designer takes its SELECTIVITY signal from:
    #   "potts"   — (default; current behaviour) the Potts centre gap Σ(e_P−e_D), z-scaled against the
    #               decoder naturalness term:  J = (1−λ)·z(−log p) + λ·z(Σ(e_P−e_D)).
    #   "decoder" — a PURE-decoder two-state probability contrast, no Potts and no z-scaling:
    #                   R(a) = p(a | all centers = TARGET) − λ · p(a | all centers = OFF)
    #               where both are the decoder's own softmax(log_probs) distributions on one simplex
    #               (target = pin at prot_idx; off = pin at dep_idxs, averaged over tautomers e.g.
    #               HID/HIE). λ=0 recovers the plain target-state decode ("no selective objective").
    #               Valid ONLY for method=autoregressive (block descent has no pluggable [V] row).
    selective_source: str = "potts"
    # Which protonated microstate to introduce at the (designed) centre, then redesign around.
    center_protonation_types: List[str] = field(default_factory=lambda: ["HIS-P", "ASP-P", "GLU-P"])
    dep_map: Dict[str, List[str]] = field(default_factory=lambda: dict(DEFAULT_DEP_MAP))
    topk_sites: int = 2
    # Placement-site selection (which binder position to introduce RES-P at):
    #   False (default) — score candidate sites against the current binder sequence (initial_sequence).
    #   True  — "structure-only" placement: mask ALL binder design positions to UNK first, so each
    #           site's selective score LL(RES-P) - LL(best RES-D) depends on the backbone (+ fixed
    #           target) and NOT the current binder sequence. With backend=mpnn this score is the
    #           decoder log-likelihood. Pair with method=autoregressive + backend=mpnn to "place the
    #           protonation by structure-only LL, then resample the site's neighbourhood with the
    #           MPNN head, the mutated site held fixed". (selective=True is required for the gap.)
    placement_seq_masked: bool = False

    method: str = "converged_mcmc"
    selective: bool = True
    combined_lambda: float = 1.0
    two_phase_frac: float = 0.5
    temperature: float = 0.1

    samples_per_site: int = 4                       # placement methods: designs per (centre type, site, seed)
    cv_patience: int = 3                           # converged: stop after patience*n_neigh no-change
    cv_max: int = 50                               # converged: hard cap = cv_max*n_neigh moves

    # block_descent (deterministic / sampled, near-exact MAP): each block = one neighbour + its
    # (block_size-1) closest coupled partners; enumerate ALL V**block_size joint assignments and
    # score them by the Z-SCALED combined objective — the MANUSCRIPT Eq (6):
    #     J = (1-combined_lambda)*z(H) + combined_lambda*z(selective) + global_weight*z(global)
    # where z(x) = (x - mean)/std of that term's single-mutation deltas (computed ONCE over the
    # design positions and frozen). Z-scaling puts stability and selectivity on a common spread, so
    # combined_lambda is a true RELATIVE weight (0 -> pure stability, 1 -> pure selectivity, 0.5 ->
    # balanced in std units). Readout temperature `temperature`: 0 -> argmin (deterministic), >0 ->
    # sample the block ~ softmax(-J/temperature) (Boltzmann, like sampling). block_size=1 -> greedy
    # ICM; 2 -> pairwise; 3 -> triples; larger -> closer to the joint optimum. Only STABILITY has
    # within-block pairwise terms (selective+global are exactly unary), so increasing block_size
    # changes the answer ONLY through stability coupling.
    block_size: int = 2
    global_weight: float = 0.0                       # extra z-scaled global term (0 = report-only)
    rank_normalize: bool = False                     # use percentile (rank) scaling instead of z-score
    # z-scale reference distribution for the frozen combined objective (block_descent):
    #   "block" (default) — std of the BLOCK (V**block_size) joint spread, pooled WITHIN-block; captures
    #                       the within-block pairwise stability variance (the honest scale for the objective).
    #   "single_mutation" — legacy cheap proxy: std of single-position deltas over the design set (kept
    #                       to compare against / fall back to; note it re-calibrates λ vs block mode).
    zscale_mode: str = "block"
    # Adjacent-repeat bias: +adjacent_repeat_weight per SEQUENCE-ADJACENT (i,i+1) pair that ends up the
    # SAME amino acid (canonical; ASP-P/ASP-D both = Asp). 0 = off. Discourages i,i+1 homopolymer repeats;
    # added raw to the (z-scaled) block objective, so it is a knob in J's z-units. See _block_descent.
    adjacent_repeat_weight: float = 0.0
    # Windowed repetitive-density penalty (anti-collapse; block_descent only, 0 = off). Penalises
    # clustering of a residue class: +weight per class residue within ±radius sequence positions.
    # ``repetitive_window_parents`` = 3-letter canonical parents to disperse, e.g. ["ASP","GLU"],
    # ["ARG","LYS"], ["HIS"]. ``repetitive_window_gate_types`` limits it to a pinned centre type (EMPTY
    # = always on when weight>0). See _block_descent / _parent_vocab_mask.
    repetitive_window_weight: float = 0.0
    repetitive_window_radius: int = 2
    repetitive_window_parents: List[str] = field(default_factory=list)
    repetitive_window_gate_types: List[str] = field(default_factory=list)
    # Relative weight of the Potts SELF-fields (single-site h_i) vs the pairwise couplings J_ij in the
    # STABILITY term: H_stab = self_weight·Σh_i + Σ J_ij (pairwise fixed at 1). 1.0 = current behaviour.
    # <1 down-weights the self bias (context-aware design; tests whether poly-acidic collapse is a
    # self-field artefact), >1 amplifies it. Scope is stability ONLY — selective/global/reported energies
    # stay on the true model (the self-field cancels in e_P−e_D anyway). Because H_stab is z-scored in J,
    # only the self:pair RATIO matters (uniform scale is divided out), so self_weight==1 is an exact
    # no-op. See _block_descent / _PottsScorer.reweighted.
    self_weight: float = 1.0
    block_max_rounds: int = 10                      # sweeps over all neighbours before giving up

    # Where the placement redesign starts from (the "initial_sequence" per the 05 scripts):
    #   "inverse" — externally supplied seeds = the inverse stage's sequences (ProteinMPNN, or the
    #               model's own MPNN sampling); passed into run_ph_redesign as initial_sequences.
    #               This is the decoupled "run the inverse stage, then only optimize" path.
    #   "native"  — the single native (input-PDB / RFD3) sequence.
    # MPNN-sampled seeds are produced by the INVERSE stage, never re-sampled inside the optimizer.
    seed_source: str = "inverse"

    forbidden_tokens: List[str] = field(default_factory=lambda: list(DEFAULT_FORBIDDEN_TOKENS))
    num_designs: int = 8                           # whole-chain (gibbs/mpnn_sample) count / run

    # Record the per-step energy trajectory (selective energy, global protonation ΔH, total Potts H)
    # of the placement redesign so you can see the optimization "under the hood". OFF by default:
    # each recorded step adds ~selective + 3×H_of + 1×H_of Potts evals, so it ~doubles the (already
    # CPU-heavy) per-move cost — use it with a small config (few seeds, samples_per_site=1).
    record_trajectory: bool = False

    # --- multi-center placement (pin a SET of protonated centers, redesign the union of their
    #     neighbourhoods). center_count == 1 reproduces the single-center behavior. ---
    center_count: int = 1
    # explicit_centers: [{res_id, protonation_type}, ...] -> exactly ONE plan with those pins,
    # bypassing the random site search (mix-and-match by hand). Its length overrides center_count.
    explicit_centers: List[Dict] = field(default_factory=list)
    # center_types: an EXACT protonation-type multiset e.g. ["ASP-P","ASP-P","HIS-P"] to place. Unlike
    # center_protonation_types (a type ALPHABET the combo logic mixes freely), this pins the exact
    # COMPOSITION while letting placement (placement_by/region) CHOOSE the positions — one distinct,
    # best-ranked site per type. Produces exactly ONE plan. [] = off (use the pool/combo logic).
    center_types: List[str] = field(default_factory=list)
    # placement region(s) candidate centers are drawn from (union): subset of
    # {interface, core, surface, all}. Region masks come from RASA + target-contact
    # (mirrors src/rewards/property_calculators.py). "all" == all free binder positions.
    placement_region: List[str] = field(default_factory=lambda: ["all"])
    # How candidate center positions are chosen within the region (decoupled from the redesign method):
    #   "random"     — uniform over the region.
    #   "scan_potts" — the existing _placement_sites ranked by Potts dE_p - dE_d.
    #   "scan_mpnn"  — the same finder ranked by the PottsMPNN decoder log-lik(prot) - log-lik(deprot).
    placement_by: str = "scan_potts"
    # Informational label for the drawn placement TYPE (e.g. "core"/"interface"/"random"/"scan"), set by
    # sample_criteria so each design carries one clean placement-attribution column. "" = unset.
    placement_label: str = ""
    # Candidate center positions come from the existing selective ranking (`_placement_sites`,
    # generalized over the region): the potts backend ranks by dE_p - dE_d; PottsMPNN's MPNN backend
    # ranks by log-lik(protonated) - log-lik(deprotonated). candidate_pool = top-ranked positions per
    # protonation type that feed the multi-center combinations.
    candidate_pool: int = 12
    n_plan_samples: int = 64          # cap on center combinations evaluated (random subsample if larger)
    max_plans_per_seed: int = 8       # plans kept after ranking combos by locked Σ_centers (e_P - e_D)
    infill_scope: str = "neighbourhood"   # "neighbourhood" (∪ pin kNN) | "chain" (whole free binder)
    # sweep_order: order block descent visits designable positions (greedy → order matters).
    #   "position" — ascending residue index; "knn" — closest-coupled first; "energy" — largest
    #   contribution to the centres' selective gap (e_P − e_D) first.
    sweep_order: str = "position"
    # neighbour_k: per-centre cap on how many coupled kNN neighbours become designable — controls the
    # redesign extent / mutation load. 0 = full neighbourhood (default). Ignored when infill_scope=="chain".
    neighbour_k: int = 0
    # HARD cap on the TOTAL number of designable positions (across all centres) → a direct MAX-MUTATIONS
    # budget (block descent redesigns the designable set, so n_mutations <= max_mutations). Unlike
    # neighbour_k (a PER-centre kNN cap), this bounds the whole redesign; when the union exceeds it, keep
    # the max_mutations positions CLOSEST-coupled to the centres (by _knn_rank). 0 = no cap (default no-op).
    # Sweepable (see _axis_expand / the sampling block).
    max_mutations: int = 0
    # Binder residues whose CA lies within this many angstroms of any target CA form the "interface"
    # placement region. 6 suits compact interfaces; loops that reach over the target (e.g. nanobody
    # CDRs) may need 8 or more, otherwise "interface" can come out empty and no site is found.
    interface_distance: float = DEFAULT_INTERFACE_DISTANCE

    def __post_init__(self) -> None:
        if self.interface_distance <= 0:
            raise ValueError(f"interface_distance must be > 0 (got {self.interface_distance}).")
        if self.method not in ALL_METHODS:
            raise ValueError(f"Unknown method '{self.method}'. Available: {ALL_METHODS}")
        if self.backend not in ("potts", "mpnn"):
            raise ValueError(f"Unknown backend '{self.backend}'. Use 'potts' or 'mpnn'.")
        if self.seed_source not in ("inverse", "native"):
            raise ValueError(
                f"Unknown seed_source '{self.seed_source}'. Use 'inverse' or 'native' "
                "(MPNN-sampled seeds come from the inverse stage, not the optimizer).")
        if self.method == "gibbs" and self.backend != "potts":
            raise ValueError("method='gibbs' requires backend='potts'.")
        if self.method in ("mpnn_sample", "autoregressive") and self.backend != "mpnn":
            raise ValueError(f"method='{self.method}' requires backend='mpnn'.")
        if self.selective_source not in ("potts", "decoder"):
            raise ValueError(f"Unknown selective_source '{self.selective_source}'. Use 'potts' or 'decoder'.")
        if self.selective_source == "decoder" and self.method != "autoregressive":
            raise ValueError(
                "selective_source='decoder' is only valid for method='autoregressive' "
                f"(got method='{self.method}'); block descent has no pluggable [V] selective row.")
        _centre_free = (self.method == "greedy_energy_block" and self.infill_scope == "chain"
                        and not self.center_types and not self.explicit_centers)
        if self.center_count < 1 and not (self.center_count == 0 and _centre_free):
            raise ValueError(
                f"center_count must be >= 1 (got {self.center_count}); center_count=0 is allowed only for "
                f"method='greedy_energy_block' with infill_scope='chain' and no center_types/explicit_centers.")
        _regions = {"interface", "core", "surface", "all"}
        bad = set(self.placement_region) - _regions
        if bad:
            raise ValueError(f"Unknown placement_region {bad}. Allowed: {_regions}.")
        if self.sweep_order not in ("position", "knn", "energy"):
            raise ValueError(f"Unknown sweep_order '{self.sweep_order}'. Use 'position', 'knn' or 'energy'.")
        if self.infill_scope not in ("neighbourhood", "chain"):
            raise ValueError(f"Unknown infill_scope '{self.infill_scope}'. Use 'neighbourhood' or 'chain'.")
        if self.placement_by not in ("random", "scan_potts", "scan_mpnn"):
            raise ValueError(f"Unknown placement_by '{self.placement_by}'. Use random/scan_potts/scan_mpnn.")
        if self.zscale_mode not in ("single_mutation", "block"):
            raise ValueError(f"Unknown zscale_mode '{self.zscale_mode}'. Use 'single_mutation' or 'block'.")
        if self.self_weight < 0:
            raise ValueError(f"self_weight must be >= 0 (got {self.self_weight}).")
        if self.explicit_centers and self.center_count not in (1, len(self.explicit_centers)):
            raise ValueError("center_count must equal len(explicit_centers) when explicit_centers is set.")
        if self.center_types:
            bad_t = set(self.center_types) - set(self.dep_map)
            if bad_t:
                raise ValueError(f"center_types {bad_t} not in dep_map keys {set(self.dep_map)}.")
            self.center_count = len(self.center_types)     # keep downstream center_count logic consistent

    def stability_weights(self) -> tuple:
        """(w_self, w_pair) applied to the STABILITY Potts energy. Pairwise fixed at 1; only the
        ratio matters (H_stab is z-scored), so self_weight==1 → (1,1) is an exact no-op."""
        return float(self.self_weight), 1.0

    @property
    def is_placement(self) -> bool:
        return self.method in PLACEMENT_METHODS

    def scheme_label(self) -> str:
        """Sweep-safe label identifying method+objective+key-param (mirrors the 05 scripts)."""
        obj = "selective" if self.selective else "non_selective"
        if self.method == "converged_mcmc_combined":
            base = f"converged_mcmc_combined_l{self.combined_lambda:g}"
        elif self.method == "two_phase":
            base = f"two_phase_f{self.two_phase_frac:g}"
        elif self.method == "converged_mcmc":
            base = f"{obj}_converged"
        elif self.method == "autoregressive":
            base = "selective" if self.selective else "likelihood"
            if self.selective_source == "decoder":       # keep 'potts' (default) label byte-identical
                base += "_dec"
        elif self.method == "block_descent":
            base = f"block_descent_b{self.block_size}_l{self.combined_lambda:g}"
            if self.zscale_mode == "single_mutation":     # default (block) keeps the clean label
                base += "_sm"
            if self.adjacent_repeat_weight:
                base += f"_ar{self.adjacent_repeat_weight:g}"
            if self.repetitive_window_weight:
                base += f"_rw{self.repetitive_window_weight:g}"
            if self.self_weight != 1.0:                   # default (1.0) keeps the clean label
                base += f"_sw{self.self_weight:g}"
        else:
            base = self.method  # gibbs / mpnn_sample
        # structure-only placement variant — tag so swept (masked vs not) designs don't collide on
        # design_id. Omitted by default so existing labels are unchanged.
        if self.placement_seq_masked and self.is_placement:
            base += "_smask"
        return base

    @classmethod
    def from_params(cls, params: Optional[Dict]) -> "PHDesignCriteria":
        """Build one criteria from a resolved hydra ``params`` dict (unknown keys ignored)."""
        params = dict(params or {})
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        crit = cls(**{k: v for k, v in params.items() if k in known})
        # OmegaConf may hand back ListConfig/DictConfig — coerce to plain containers.
        crit.center_protonation_types = list(crit.center_protonation_types)
        crit.forbidden_tokens = list(crit.forbidden_tokens)
        crit.dep_map = {k: list(v) for k, v in dict(crit.dep_map).items()}
        crit.placement_region = list(crit.placement_region)
        crit.explicit_centers = [dict(c) for c in crit.explicit_centers]
        crit.repetitive_window_parents = list(crit.repetitive_window_parents)
        crit.repetitive_window_gate_types = list(crit.repetitive_window_gate_types)
        crit.center_types = list(crit.center_types)
        return crit


def expand_criteria(params: Optional[Dict]) -> List[PHDesignCriteria]:
    """Expand a hydra ``params`` dict into a flat list of concrete ``PHDesignCriteria``.

    Two layers, so you can run several DIFFERENT methods in one job AND sweep a param within a
    method:

    1. ``schemes`` — an optional list of per-method overrides. Each entry inherits the shared
       top-level params and overrides (typically) ``method`` and its knob. Use this to run, say,
       a combined-λ sweep and a two-phase-fraction sweep as TWO separate design methods at once::

           schemes:
             - {method: converged_mcmc_combined, combined_lambda: [0.25, 0.5, 1.0]}
             - {method: two_phase, two_phase_frac: [0.25, 0.5, 0.75]}

    2. axis sweep — within each scheme (or the top level if no ``schemes``), any of
       ``method / selective / combined_lambda / two_phase_frac / temperature`` given as a LIST is
       expanded as a Cartesian product. So ``combined_lambda: [0.25, 0.5, 1.0]`` -> 3 criteria
       labelled ``converged_mcmc_combined_l0.25 / …_l0.5 / …_l1``.
    """
    params = dict(params or {})
    schemes = params.pop("schemes", None)
    base = params
    scheme_dicts = [dict(s) for s in schemes] if schemes else [{}]

    out: List[PHDesignCriteria] = []
    for scheme in scheme_dicts:
        out.extend(_axis_expand({**base, **scheme}))
    return out


def _axis_expand(params: Dict) -> List[PHDesignCriteria]:
    """Cartesian-product expansion over any list-valued sweep axes in a single param dict."""
    # Any of these may be given as a LIST inside a scheme dict (or top-level) to Cartesian-sweep it;
    # any OTHER PHDesignCriteria field may still be set per-scheme as a scalar (merged via from_params).
    sweep_keys = ("method", "backend", "selective", "selective_source", "combined_lambda", "two_phase_frac", "temperature",
                  "block_size", "global_weight", "cv_patience", "cv_max", "block_max_rounds",
                  "samples_per_site", "placement_seq_masked", "center_count", "infill_scope",
                  "candidate_pool", "n_plan_samples", "max_plans_per_seed", "placement_by", "zscale_mode",
                  "adjacent_repeat_weight", "self_weight",
                  "repetitive_window_weight", "repetitive_window_radius", "neighbour_k", "max_mutations")
    axes: Dict[str, list] = {}
    base = dict(params)
    for k in sweep_keys:
        v = params.get(k)
        if isinstance(v, (list, tuple)):
            axes[k] = list(v)
            base.pop(k)

    if not axes:
        return [PHDesignCriteria.from_params(params)]

    keys = list(axes)
    out: List[PHDesignCriteria] = []

    def _recurse(i: int, acc: Dict[str, Any]) -> None:
        if i == len(keys):
            out.append(PHDesignCriteria.from_params({**base, **acc}))
            return
        for val in axes[keys[i]]:
            _recurse(i + 1, {**acc, keys[i]: val})

    _recurse(0, {})
    return out


def sample_criteria(params: Optional[Dict], rng) -> List[PHDesignCriteria]:
    """STOCHASTIC criteria for a large sweep: when ``params`` has a ``sampling`` block, draw
    ``n_criteria`` random criteria instead of the Cartesian ``expand_criteria`` (which would explode
    over placement × composition × param). Falls back to ``expand_criteria`` when no ``sampling`` block.
    ``rng`` = ``random.Random`` (seed per backbone for reproducible diversity).

    Each criterion draws ONE value on each axis, so every design is cleanly attributable in the parquet:
      - ONE ``placement`` type — a NAMED (region, method) bundle, NOT independent region/method draws,
        so "which placement won" is a single categorical column (``placement_label``).
      - ONE ``center_types`` exact multiset (composition + count).
      - ONE strategy param (``repetitive_window_weight`` for block_descent, ``combined_lambda`` for mpnn).

        sampling:
          n_criteria: 40
          placement:                                   # ONE bundle drawn per criterion
            - {name: core,      placement_region: [core],      placement_by: scan_potts}
            - {name: interface, placement_region: [interface], placement_by: scan_potts}
            - {name: random,    placement_region: [all],       placement_by: random}
            - {name: scan,      placement_region: [all],       placement_by: scan_potts}
          center_types: [[ASP-P], [ASP-P,GLU-P,HIS-P], [ASP-P,ASP-P,HIS-P], ...]   # exact multisets
          repetitive_window_weight: [1, 2]             # (block_descent) OR  combined_lambda: [0.4, 0.5] (mpnn)
    Any other key in ``sampling`` whose value is a list is likewise sampled (one pick) per criterion;
    ``center_types`` picks a whole multiset. Duplicate draws are kept (dedup happens later by sequence).
    """
    params = dict(params or {})
    s = params.pop("sampling", None)
    if not s:
        return expand_criteria(params)
    s = dict(s)
    n = int(s.pop("n_criteria", 1))
    placements = list(s.pop("placement", []) or [])       # named (region, method) bundles — ONE per design
    out: List[PHDesignCriteria] = []
    for _ in range(n):
        pick = dict(params)
        if placements:                                    # draw ONE placement type (region+method together)
            pl = dict(placements[rng.randrange(len(placements))])
            reg = pl["placement_region"]
            pick["placement_region"] = list(reg) if isinstance(reg, (list, tuple)) else [reg]
            pick["placement_by"] = pl["placement_by"]
            pick["placement_label"] = str(pl.get("name") or f"{pl['placement_by']}@{pick['placement_region']}")
        for k, pool in s.items():                         # remaining sampled axes: one pick each
            pool = list(pool)
            if not pool:
                continue
            choice = pool[rng.randrange(len(pool))]
            pick[k] = list(choice) if k == "center_types" else choice
        out.append(PHDesignCriteria.from_params(pick))
    return out


# ---------------------------------------------------------------------------
# Placement plan: a SET of pinned protonated centers + the positions to redesign
# ---------------------------------------------------------------------------

@dataclass
class PlacementPin:
    """One pinned protonated center."""
    position: int           # binder POSITION index into the sequence tensor S (ctx coord), not res_id
    protonation_type: str   # "HIS-P" | "ASP-P" | "GLU-P"
    prot_idx: int           # ctx.encoding.token_to_idx[protonation_type]
    dep_idxs: List[int]     # deprotonated-alternative token idxs (ctx.encoding of crit.dep_map[type])
    res_id: int             # int(ctx.token_aa.res_id[position]) — for output + explicit matching


@dataclass
class PlacementPlan:
    """A set of pinned centers + the (union) positions to redesign around them."""
    pins: List[PlacementPin]
    designable: List[int]        # sorted positions to redesign (excludes pin positions)
    label: str                   # "+".join(sorted protonation types), e.g. "GLU-P+HIS-P"
    placement_score: float = 0.0 # locked Σ_centers (e_P - e_D) at selection time (lower = more selective)

    @property
    def seed_key(self) -> int:
        # deterministic per-plan RNG offset; reduces to the old single-center `center*97` at len==1.
        # centre-free plans (greedy_energy_block) have no pins -> derive a stable offset from the span.
        if not self.pins:
            return (self.designable[0] * 97 + self.designable[-1] * 7) if self.designable else 0
        return self.pins[0].position * 97 + sum(p.position for p in self.pins[1:]) * 7

    @staticmethod
    def placement_fn(placement_by: str) -> Callable:
        """Dispatch to the candidate-position finder chosen by ``placement_by`` (decoupled from the
        redesign method). Each finder has signature
        ``fn(ctx, crit, region_idx, prot_idx, dep_idxs, initial, rng) -> List[(position, score)]``
        ordered ascending (lower = better); ``random`` ignores the scan args and returns score 0.
          random     -> uniform over the region
          scan_potts -> _placement_sites ranked by Potts dE_p - dE_d
          scan_mpnn  -> _placement_sites ranked by PottsMPNN log-lik(prot) - log-lik(deprot)
        """
        fns = {"random": _place_random, "scan_potts": _place_scan_potts, "scan_mpnn": _place_scan_mpnn}
        try:
            return fns[placement_by]
        except KeyError:
            raise ValueError(f"Unknown placement_by '{placement_by}'. Available: {list(fns)}")


# ---------------------------------------------------------------------------
# Design output (the "score each design by its energies" class)
# ---------------------------------------------------------------------------

@dataclass
class PHDesignOutput:
    """One finished design: its binder sequence plus every energy score.

    ``final_potts_energy`` is the whole-system Potts Hamiltonian (lower = stabler).
    ``selective_energy`` is the centre RES-P-vs-RES-D gap (lower = RES-P preferred).
    ``global_protonation_dH`` is the binder's H(all titratable→prot) − H(→deprot)
    (lower = prefers protonated). Placement metrics describe the chosen site.
    """

    binder_chain: str
    canonical_sequence: str               # binder one-letter (RF3-foldable)
    extended_tokens: List[str]            # binder microstate token names (parallel to above)
    extended_vocab: bool
    scheme: str
    sample: int
    final_potts_energy: float
    # placement-based fields (None for whole-chain baselines)
    seed_idx: Optional[int] = None          # which MPNN seed sequence this was redesigned from
    protonation_type: Optional[str] = None      # single type OR the combo label "GLU-P+HIS-P"
    site_rank: Optional[int] = None
    center_res_id: Optional[int] = None          # primary (first) center res_id
    n_neighbours: Optional[int] = None
    selective_energy: Optional[float] = None      # total locked Σ_centers (e_P - e_D)
    # multi-center detail (None for whole-chain baselines):
    n_centers: Optional[int] = None
    center_res_ids: Optional[List[int]] = None
    center_protonation_types: Optional[List[str]] = None
    selective_energies: Optional[List[float]] = None   # per-center (e_P - e_D)
    global_protonation_dH: Optional[float] = None
    placement_prob: Optional[float] = None
    placement_entropy: Optional[float] = None
    sequence_decoded_prob_score: Optional[float] = None
    sequence_entropy: Optional[float] = None      # Shannon entropy (bits) of the redesigned region's AA composition
    # criteria echo (for parquet attribution in a stochastic sweep): the drawn strategy/placement/param.
    method: Optional[str] = None
    backend: Optional[str] = None
    selective_source: Optional[str] = None    # autoregressive arm: "potts" | "decoder" (see PHDesignCriteria)
    placement_label: Optional[str] = None
    placement_region: Optional[str] = None
    placement_by: Optional[str] = None
    combined_lambda: Optional[float] = None
    repetitive_window_weight: Optional[float] = None
    block_size: Optional[int] = None        # solver knob echoed for attribution (cost ~ V**block_size)
    sweep_order: Optional[str] = None       # solver knob echoed for attribution
    neighbour_k: Optional[int] = None          # redesign-extent cap drawn for this design (0 = full kNN)
    max_mutations: Optional[int] = None        # total designable/mutation cap drawn for this design (0 = none)
    # Per-step optimization trace (only when criteria.record_trajectory): one dict per MCMC/infill step
    # with {step, selective_energy, global_protonation_dH, potts_energy, canonical_sequence,
    # extended_tokens}. The last two are the binder sequence AT that step (1-letter + 3-letter/protonation
    # token string), so any step's sequence can be saved. step 0 = initial seq.
    energy_trajectory: Optional[List[Dict[str, Any]]] = None

    # Set by PHDesignSet.deduped() when two DIFFERENT sequences share a design_id (the id does not encode
    # every swept knob), so neither is silently dropped. Empty for the usual case: ids are unchanged.
    id_suffix: str = ""

    def design_id(self) -> str:
        if self.protonation_type is not None and self.center_res_ids:
            seed_tag = "" if self.seed_idx is None else f"_seed{self.seed_idx}"
            res_tag = "-".join(str(r) for r in self.center_res_ids)
            return f"{self.protonation_type}_{self.scheme}{seed_tag}_res{res_tag}_s{self.sample}{self.id_suffix}"
        return f"{self.scheme}_s{self.sample}{self.id_suffix}"

    def to_row(self, reference_sequence: Optional[str] = None) -> Dict[str, Any]:
        """``to_metadata()`` plus the fields tables usually want (see ``PHDesignSet.to_rows``)."""
        row = self.to_metadata()
        row["binder_chain"] = self.binder_chain
        row["canonical_sequence"] = self.canonical_sequence
        row["centers"] = [
            f"{res_id}:{ptype}"
            for res_id, ptype in zip(self.center_res_ids or [], self.center_protonation_types or [])
        ]
        if reference_sequence is not None:
            if len(reference_sequence) != len(self.canonical_sequence):
                raise ValueError(
                    f"reference_sequence has {len(reference_sequence)} residues, the design "
                    f"{len(self.canonical_sequence)}")
            row["n_mutations"] = sum(a != b for a, b in zip(reference_sequence, self.canonical_sequence))
        else:
            row["n_mutations"] = None
        return row

    def to_metadata(self) -> Dict[str, Any]:
        """Energy scores + provenance for InverseFoldOutputItem.metadata (rewards stage)."""
        return {
            "design_id": self.design_id(),
            "scheme": self.scheme,
            "seed_idx": self.seed_idx,
            "extended_vocab": self.extended_vocab,
            "extended_tokens": " ".join(self.extended_tokens),
            "potts_energy": self.final_potts_energy,
            "selective_energy": self.selective_energy,
            "global_protonation_dH": self.global_protonation_dH,
            "placement_prob": self.placement_prob,
            "placement_entropy": self.placement_entropy,
            "sequence_decoded_prob_score": self.sequence_decoded_prob_score,
            "sequence_entropy": self.sequence_entropy,
            "protonation_type": self.protonation_type,
            "site_rank": self.site_rank,
            "center_res_id": self.center_res_id,
            "n_neighbours": self.n_neighbours,
            "n_centers": self.n_centers,
            "center_res_ids": self.center_res_ids,
            "center_protonation_types": self.center_protonation_types,
            "selective_energies": self.selective_energies,
            "method": self.method,
            "backend": self.backend,
            "selective_source": self.selective_source,
            "placement_label": self.placement_label,
            "placement_region": self.placement_region,
            "placement_by": self.placement_by,
            "combined_lambda": self.combined_lambda,
            "repetitive_window_weight": self.repetitive_window_weight,
            "block_size": self.block_size,
            "sweep_order": self.sweep_order,
            "neighbour_k": self.neighbour_k,
            "max_mutations": self.max_mutations,
            "energy_trajectory": self.energy_trajectory,
        }


class PHDesignSet(list):
    """A list of ``PHDesignOutput`` with convenience selectors."""

    def sorted_by_energy(self) -> "PHDesignSet":
        return PHDesignSet(sorted(self, key=lambda d: d.final_potts_energy))

    def deduped(self) -> "PHDesignSet":
        """Drop exact repeats (same id AND same sequence); keep designs that share an id but differ
        in sequence under a ``~k`` id suffix. First occurrence wins, so the order of ``self`` decides.

        Every id already present in ``self`` is reserved, so a new suffix never lands on another
        design's id, and a record that needs a new id is copied: the inputs are not modified."""
        taken = {d.design_id() for d in self}
        by_id: Dict[str, set] = {}
        out = PHDesignSet()
        for d in self:
            did, seq = d.design_id(), tuple(d.extended_tokens)
            known = by_id.setdefault(did, set())
            if seq in known:
                continue
            if known:
                k = 2
                while f"{did}~{k}" in taken:
                    k += 1
                d = replace(d, id_suffix=d.id_suffix + f"~{k}")
                taken.add(d.design_id())
            known.add(seq)
            out.append(d)
        return out

    def to_rows(self, reference_sequence: Optional[str] = None) -> List[Dict[str, Any]]:
        """Plain dicts, one per design, for tables and downstream code.

        ``to_metadata()`` plus ``canonical_sequence``, ``centers`` (``"res_id:type"``) and, when the
        starting binder sequence is given as ``reference_sequence``, ``n_mutations`` (Hamming distance)."""
        return [d.to_row(reference_sequence) for d in self]

    def top(self, n: int) -> "PHDesignSet":
        return PHDesignSet(self.deduped().sorted_by_energy()[:n])


# ---------------------------------------------------------------------------
# Parallel solve: fork a pool over the independent per-seed tasks. The pool is
# created AFTER featurisation, so each worker inherits the finished ctx (etab
# tables) read-only via copy-on-write — concurrent lookups, zero copy, tiny
# per-worker memory. CPU only (fork after CUDA init is unsafe). The heavy ctx is
# shared through these module globals (inherited at fork), NOT pickled per task;
# only the small (ci, seq_i) task and the resulting PHDesignOutput list cross the pipe.
# ---------------------------------------------------------------------------
_POOL_CTX = None
_POOL_ENGINE = None
_POOL_CRITERIA = None
_POOL_SEED = 0
_POOL_THREADLIMIT = None   # holds the threadpoolctl limiter so it isn't GC'd (which would restore threads)


def _pool_init():
    # Pin EACH worker to a single compute thread. Parallelism here is across the (fork) PROCESS pool,
    # not within a worker — left uncapped, every worker spins up its own ~Ncore OpenBLAS/MKL/OMP
    # thread pool, so n_jobs workers × ~Ncore threads thrash the cores (futex contention) and
    # effective throughput collapses to <10 cores. torch.set_num_threads only caps torch's intra-op
    # pool; OpenBLAS/MKL/OMP (used by numpy and torch's BLAS) need separate capping. threadpool_limits
    # resizes the already-loaded pools at runtime (works even when the env vars weren't exported); the
    # env-var fallback covers libs that read their thread count only at import.
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    torch.set_num_threads(1)
    try:
        from threadpoolctl import threadpool_limits
        global _POOL_THREADLIMIT
        # Hold the limiter for the worker's lifetime — if it were GC'd it would restore the original
        # (uncapped) thread counts. Not a context manager here on purpose.
        _POOL_THREADLIMIT = threadpool_limits(limits=1)
    except Exception:
        pass


def _run_design_task(eng, ctx, criteria_list, base_seed, task):
    """Run one independent unit (criteria_index, seq_i) -> plain list[PHDesignOutput].
    seq_i is the placement seed index (unused for whole-chain methods)."""
    ci, seq_i = task
    crit = criteria_list[ci]
    field_fn, field_at, valid_mask = eng._criteria_fields(ctx, crit)
    seed = base_seed + 1009 * ci
    if crit.method in WHOLE_CHAIN_METHODS:
        return list(eng._run_whole_chain(ctx, crit, field_fn, valid_mask, seed=seed))
    initial_sequence, seeded = eng._placement_initial(ctx, crit, seq_i)
    return list(eng._run_placement_one_seed(
        ctx, crit, field_fn, field_at, valid_mask, seed, seq_i, initial_sequence, seeded))


def _pool_task(task):
    """Pool worker entry: reads the forked (COW-shared) globals and runs one task."""
    return _run_design_task(_POOL_ENGINE, _POOL_CTX, _POOL_CRITERIA, _POOL_SEED, task)


# ---------------------------------------------------------------------------
# Binder region classification (interface / core / surface) for center placement.
# interface: binder CA within interface_dist of a target CA, found with biotite's CellList
#   (native spatial search over the AtomArray — no scipy).
# core: mean per-residue RASA < threshold (Shrake-Rupley via CalculateRASA); surface = binder ∧ ¬core.
# Returns per-ctx-position boolean masks keyed onto token_aa by (chain_id, res_id). Fail-soft.
# ---------------------------------------------------------------------------
# Tien et al. 2013 (PLoS ONE 8:e80635) theoretical maximum accessible surface area per residue, A^2.
# The denominator for residue-level RASA in _binder_region_masks.
_MAX_ASA = {"ALA": 129, "ARG": 274, "ASN": 195, "ASP": 193, "CYS": 167, "GLN": 225, "GLU": 223,
            "GLY": 104, "HIS": 224, "ILE": 197, "LEU": 201, "LYS": 236, "MET": 224, "PHE": 240,
            "PRO": 159, "SER": 155, "THR": 172, "TRP": 285, "TYR": 263, "VAL": 174}


def _binder_region_masks(proc_atom_array, token_aa, binder_chain, chainA_t, device, L,
                         interface_dist: float = 6.0, rasa_threshold: float = 0.2
                         ) -> Dict[str, torch.Tensor]:
    aa = proc_atom_array
    iface = torch.zeros(L, dtype=torch.bool, device=device)
    core = torch.zeros(L, dtype=torch.bool, device=device)
    surface = torch.zeros(L, dtype=torch.bool, device=device)
    binder_ct = chainA_t.detach().cpu().numpy()
    pos_of = {(str(token_aa.chain_id[i]), int(token_aa.res_id[i])): i
              for i in range(L) if binder_ct[i]}
    binder_ca = (aa.chain_id == binder_chain) & (aa.atom_name == "CA")
    target_ca = (aa.chain_id != binder_chain) & (aa.atom_name == "CA")
    try:                                                      # interface via biotite CellList (CA-CA)
        from biotite.structure import CellList
        target_ca_arr = aa[target_ca]
        binder_ca_idx = np.where(binder_ca)[0]
        if target_ca_arr.array_length() and len(binder_ca_idx):
            cell_list = CellList(target_ca_arr, cell_size=float(interface_dist))
            contacts = cell_list.get_atoms(aa.coord[binder_ca], radius=float(interface_dist))
            has_contact = np.asarray(contacts != -1).reshape(len(binder_ca_idx), -1).any(axis=1)
            for k, ca_idx in enumerate(binder_ca_idx):
                if bool(has_contact[k]):
                    p = pos_of.get((str(aa.chain_id[ca_idx]), int(aa.res_id[ca_idx])))
                    if p is not None:
                        iface[p] = True
    except Exception:
        logger.warning("interface region could not be computed; it will be empty", exc_info=True)
    try:                                                      # core/surface: residue-level RASA
        # RESIDUE SASA / MAX ASA, which is what a 0.2 "buried" threshold means.
        #
        # This previously took the MEAN of calculate_atomwise_rasa over a residue's atoms, which is
        # not RASA: that helper normalises each atom's SASA by that ATOM's own isolated sphere area
        # (4*pi*(r+probe)^2), so an atom occluded by its own side chain reads ~0 and the per-atom
        # median across a structure is 0.000. Averaging those and thresholding at 0.2 classified
        # 99.5% of every binder as "core" (measured over 304 folded designs, 34,981 residues), so
        # placement_region="core" was effectively unconstrained while "surface" had almost no
        # candidates. Summing SASA over the residue and dividing by the residue type's maximum is
        # the standard definition and restores a sane core/surface split.
        from atomworks.ml.transforms.sasa import calculate_atomwise_sasa
        sasa = calculate_atomwise_sasa(aa, probe_radius=1.4, atom_radii="ProtOr", point_number=100)
        sasa = np.nan_to_num(np.asarray(sasa, dtype=float), nan=0.0)
        for ca_idx in np.where(binder_ca)[0]:
            chain, res_id = aa.chain_id[ca_idx], aa.res_id[ca_idx]
            p = pos_of.get((str(chain), int(res_id)))
            if p is None:
                continue
            sel = (aa.chain_id == chain) & (aa.res_id == res_id)
            max_asa = _MAX_ASA.get(str(aa.res_name[ca_idx]))
            if max_asa is None or not sel.any():
                continue
            if float(sasa[sel].sum()) / max_asa < rasa_threshold:
                core[p] = True
            else:
                surface[p] = True
    except Exception:
        logger.warning("core/surface regions could not be computed; they will be empty", exc_info=True)
    return {"interface": iface, "core": core, "surface": surface}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def _needs_site_pka(config: SwitchDesignConfig) -> bool:
    """PROPKA's reference pKa values are only needed when they are asked for (which the config
    only allows for the linked-equilibrium model, the one that uses a pKa)."""
    return config.protonation.reference_pka == "propka"


def _with_seed(config: SwitchDesignConfig, seed: Optional[int]) -> SwitchDesignConfig:
    """One seed for a switch-design run: ``seed``, when given, overrides ``config.base_seed``."""
    return config if seed is None else replace(config, base_seed=int(seed))


def _as_protonation(protonation) -> ProtonationSources:
    """ProtonationSources from one, a plain dict, or None (the defaults: EV6 states, constant pKa)."""
    if isinstance(protonation, ProtonationSources):
        return protonation
    return ProtonationSources.from_dict(protonation)


def _check_propka_available(config: SwitchDesignConfig) -> None:
    """Both PROPKA settings need it; fail now, before any structure is read."""
    if config.protonation.states == "propka" or _needs_site_pka(config):
        from mpnn.transforms.propka_labeller import require_propka

        require_propka()


def _describe_protonation(protonation: ProtonationSources) -> str:
    states = (
        f"PROPKA (pH {protonation.ph:g}, margin {protonation.margin:g})"
        if protonation.states == "propka"
        else "EV6"
    )
    pka = "PROPKA on the free partners" if protonation.reference_pka == "propka" else "per-type constants"
    return f"starting states {states}, reference pKa {pka}"


def _protonation_classifier(protonation: ProtonationSources):
    """The labeller the starting states ask for: None keeps the vocabulary's own (EV6)."""
    if protonation.states == "ev6":
        return None
    from mpnn.transforms.propka_labeller import PropkaLabeller, require_propka

    require_propka()  # fail here, not after the structure has been parsed
    return PropkaLabeller(ph=protonation.ph, margin=protonation.margin)


def _to_device(obj, device):
    """Move every tensor in a nested dict/list/tuple onto ``device``."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


class PottsMPNNPHEngine(MPNNInferenceEngine):
    """Foundry PottsMPNN engine + protonation-aware redesign methods."""

    model_type = "potts_mpnn"

    # ------------------------------------------------------------------ #
    # Model loading — vocab-aware (adds v6 / 30-token, HIS-S support)
    # ------------------------------------------------------------------ #
    def _resolve_vocab_name(self) -> Optional[str]:
        """Vocabulary NAME to build the featuriser with, or None. Only an EXPLICIT string
        ``extended_vocab`` (e.g. "v6") activates the vocab-aware path; a legacy bool / None keeps
        the historical 32-token default behaviour byte-for-byte."""
        return self.extended_vocab if isinstance(self.extended_vocab, str) else None

    def _build_and_load_model(self) -> torch.nn.Module:
        """Load the checkpoint, sizing the featuriser to the vocabulary.

        Overrides the base loader (which ALWAYS builds a 32-token ``PottsProteinFeatures()`` →
        a 30-token v6 checkpoint crashes on the strict ``W_s.weight`` load). When ``extended_vocab``
        is a vocab NAME, the featuriser is built from ``get_vocab(name)["token_encoding"]`` (30 for
        v6, 32 for v3/v4). A legacy bool / None reproduces the base behaviour exactly. ``etab_source``
        and ``field_source``/``*_hidden`` are threaded through unchanged.
        """
        from mpnn.transforms.extended_vocab import get_vocab

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise TypeError("Expected checkpoint to be a dict with a 'model' key.")
        etab_source = self.etab_source
        if etab_source is None:
            etab_source = PottsMPNN.infer_etab_source(checkpoint["model"])
        potts_kwargs = dict(
            field_source=self.field_source, etab_source=etab_source,
            etab_hidden=self.etab_hidden, field_hidden=self.field_hidden,
        )
        vocab_name = self._resolve_vocab_name()
        # remember the resolved name so _build_context featurises S in the SAME vocabulary
        self._vocab_name = vocab_name
        # Reproduce the LABEL OPERATING POINT this checkpoint was trained at. The packaged v6 checkpoint
        # records his_prob_thr=0.3, while transforms/ev6/weights/thresholds.json defaults to 0.40 — so
        # without this every design run labelled His at a different cut than training, silently and with
        # nothing to catch it (the token set and weight shapes are identical either way). An explicit
        # EV6_HIS_PROB_THR / EV6_ACID_PROB_THR still wins, so deliberate sweeps are unaffected.
        _tc = checkpoint.get("train_cfg") or {}
        try:
            _his_thr, _acid_thr = _tc.get("his_prob_thr"), _tc.get("acid_prob_thr")
        except Exception:
            _his_thr = _acid_thr = None
        if _his_thr is not None or _acid_thr is not None:
            from mpnn.transforms import extended_vocab_v6
            extended_vocab_v6.set_checkpoint_operating_point(_his_thr, _acid_thr)
        if vocab_name:
            feats = PottsProteinFeatures(token_encoding=get_vocab(vocab_name)["token_encoding"])
            model = PottsMPNN(graph_featurization_module=feats, **potts_kwargs)
        elif self.extended_vocab:            # legacy bool True → 32-token default (UNCHANGED)
            model = PottsMPNN(graph_featurization_module=PottsProteinFeatures(), **potts_kwargs)
        else:
            model = PottsMPNN(**potts_kwargs)
        model.extended_vocab_name = vocab_name
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        return model

    # ------------------------------------------------------------------ #
    # Binding-energy proxy: G_bind = E_complex − (E_binder + E_target)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def compute_g_bind(self, structure, binder_chain: str = "A", binder_tokens=None) -> Dict[str, float]:
        """Structure-based binding-energy proxy from the frozen Potts model (thermodynamic cycle):

            G_bind = E_complex − (E_binder + E_target)

        E(system) = Σ_i cond[i, S_i] is the native-token Potts energy (run_potts_encoder →
        potts_candidate_energies) summed over the system's residues, computed for the full complex and
        for each chain re-encoded ALONE (target atoms removed → no cross-interface graph edges). The
        sequence (incl. each titratable residue's EV6 protonation microstate) is held FIXED to the
        complex's assignment in all three passes: each isolated chain's tokens are overwritten with the
        complex token, matched by (chain_id, res_id), so only the present-chains geometry changes.

        More NEGATIVE g_bind ⇒ the complex is lower-energy than the two isolated chains ⇒ stronger
        predicted interaction (binder-propensity = −g_bind). Units are model-internal (Potts a.u.) and
        size-dependent — z-score / macro-average per target before pooling across targets.

        ``structure`` = a path (PDB/CIF) or an AtomArray of the complex; ``binder_chain`` = the binder's
        chain id (one internal campaign uses "B"), every other chain is the target.

        ``binder_tokens`` (optional) threads a DESIGN onto the FIXED backbone so you can score a designed
        sequence without a refold: a list of vocab token NAMES (e.g. a design's space-joined
        ``extended_tokens``, incl. HIS-S microstates; a str is split on whitespace), ONE per binder
        residue in ascending res_id order. They overwrite the binder chain's tokens in the complex AND
        binder-alone passes (the target is untouched). None = score the structure's own sequence.

        Returns dict(E_complex, E_binder, E_target, g_bind, n_binder, n_target)."""
        vocab = getattr(self, "_vocab_name", None) or "v4"

        def _to_dev(x):
            if torch.is_tensor(x):
                return x.to(self.device)
            if isinstance(x, dict):
                return {k: _to_dev(v) for k, v in x.items()}
            return x

        def _sys_energy(inp, aa):
            S = inp["S"][0]
            L = int(S.shape[0])
            etab, E_idx = self.model.run_potts_encoder(inp)
            cond = PottsMPNN.potts_candidate_energies(etab, E_idx, S.unsqueeze(0))[0]
            e = cond[torch.arange(L, device=cond.device), S]
            taa = aa[get_token_starts(aa)]
            return float(e.sum().cpu()), [str(c) for c in taa.chain_id]

        bc = prepare_potts_input(structure, extended_vocab=vocab)
        aa_c = bc["atom_array"]
        inp_c = _to_dev(bc["network_input"]["input_features"])
        taa = aa_c[get_token_starts(aa_c)]
        S_c = inp_c["S"][0]
        if binder_tokens is not None:                          # thread a DESIGN onto the fixed backbone
            if isinstance(binder_tokens, str):
                binder_tokens = binder_tokens.split()
            t2i = self.model.graph_featurization_module.TOKEN_ENCODING.token_to_idx
            missing = sorted({t for t in binder_tokens if isinstance(t, str) and t not in t2i})
            if missing:
                raise KeyError(f"binder_tokens not in the model vocabulary: {missing}")
            idxs = [t2i[t] if isinstance(t, str) else int(t) for t in binder_tokens]
            binder_pos = sorted((p for p, c in enumerate(taa.chain_id) if str(c) == binder_chain),
                                key=lambda p: int(taa.res_id[p]))   # design tokens are in res_id order
            if len(binder_pos) != len(idxs):
                raise ValueError(f"binder_tokens length {len(idxs)} != {len(binder_pos)} binder residues "
                                 f"(chain {binder_chain!r})")
            S_c = S_c.clone()
            for p, a in zip(binder_pos, idxs):
                S_c[p] = a
            inp_c = {**inp_c, "S": S_c.unsqueeze(0)}
        key_S = {(str(c), int(r)): int(s)                      # (design-overridden) token per (chain, res_id)
                 for c, r, s in zip(taa.chain_id, taa.res_id, S_c.cpu())}
        e_complex, ch_c = _sys_energy(inp_c, aa_c)
        n_binder = sum(c == binder_chain for c in ch_c)
        n_target = len(ch_c) - n_binder

        chain_all = np.array([str(c) for c in aa_c.chain_id])
        e_side = {}
        for side, mask in (("binder", chain_all == binder_chain),
                           ("target", chain_all != binder_chain)):
            if not mask.any():
                e_side[side] = 0.0
                continue
            bi = prepare_potts_input(aa_c[mask], extended_vocab=vocab)   # re-encode this chain ALONE
            aa_i = bi["atom_array"]
            inp_i = _to_dev(bi["network_input"]["input_features"])
            tii = aa_i[get_token_starts(aa_i)]
            Si = inp_i["S"][0].clone()
            for p, (c, r) in enumerate(zip(tii.chain_id, tii.res_id)):   # fix microstate to the complex
                k = (str(c), int(r))
                if k in key_S:
                    Si[p] = key_S[k]
            e_side[side], _ = _sys_energy({**inp_i, "S": Si.unsqueeze(0)}, aa_i)

        return dict(E_complex=e_complex, E_binder=e_side["binder"], E_target=e_side["target"],
                    g_bind=e_complex - (e_side["binder"] + e_side["target"]),
                    n_binder=int(n_binder), n_target=int(n_target))

    @torch.no_grad()
    def compute_mpnn_loglik(self, structure, binder_chain: str = "A") -> Dict[str, float]:
        """MPNN decoder log-likelihood of the BINDER sequence given the complex backbone.

        Uses the same decoder field the design methods use (``field_mpnn`` = teacher-forced
        ``causality_pattern="conditional_minus_self"``, i.e. each residue's log P given the structure and
        every OTHER residue — the standard inverse-folding conditional readout, not a left-to-right
        chain). Returns the mean log P per binder residue (length-normalised, the comparable score) and
        the raw sum. HIGHER = the binder sequence is more "native-like" on this backbone.

        ``structure`` = path or AtomArray of the complex; ``binder_chain`` = the binder's chain id.
        Returns dict(mpnn_ll_mean, mpnn_ll_sum, n_binder)."""
        ctx = self._build_context(structure, binder_chain)
        S = ctx.S_native
        logp = -ctx.field_mpnn(S)                       # [L, V] decoder log-probs
        idx = ctx.chainA_all_idx                        # binder positions in ctx coords
        ll = logp[idx, S[idx]]
        return dict(mpnn_ll_mean=float(ll.mean().cpu()), mpnn_ll_sum=float(ll.sum().cpu()),
                    n_binder=int(idx.numel()))

    @torch.no_grad()
    def compute_g_bind_batch(self, structure, binder_chain: str = "A",
                             binder_token_sets=None, chunk: int = 64) -> List[Dict[str, float]]:
        """Batched G_bind over MANY designs sharing ONE fixed backbone. Featurises the complex /
        binder-alone / target-alone ONCE (the encoder is backbone-only + E_target is constant), then
        scores each design by re-threading only the binder tokens — far faster than compute_g_bind per
        design. ``binder_token_sets`` = list where each entry is a design's binder tokens (space-joined
        ``extended_tokens`` str or a list of token names, ONE per binder residue in res_id order); a
        None entry scores the structure's own binder sequence (the original baseline). Returns a list of
        dicts (E_complex, E_binder, E_target, g_bind, n_binder, n_target) aligned to the input list.
        Identical numerically to compute_g_bind — only the featurisation is shared."""
        vocab = getattr(self, "_vocab_name", None) or "v4"
        t2i = self.model.graph_featurization_module.TOKEN_ENCODING.token_to_idx

        def _to_dev(x):
            if torch.is_tensor(x):
                return x.to(self.device)
            if isinstance(x, dict):
                return {k: _to_dev(v) for k, v in x.items()}
            return x

        def _feat(struct):
            b = prepare_potts_input(struct, extended_vocab=vocab)
            aa = b["atom_array"]
            inp = _to_dev(b["network_input"]["input_features"])
            taa = aa[get_token_starts(aa)]
            etab, E_idx = self.model.run_potts_encoder(inp)
            return aa, inp, taa, etab, E_idx

        def _binder_order(taa):                                # binder token positions, ascending res_id
            return sorted((p for p, c in enumerate(taa.chain_id) if str(c) == binder_chain),
                          key=lambda p: int(taa.res_id[p]))

        def _native_sum(etab, E_idx, S_batch):                # Σ_i cond[i, S_i] per row → [N] (double-counted; cancels in Δ)
            cond = PottsMPNN.potts_candidate_energies(etab, E_idx, S_batch)   # [N, L, V]
            N, L, _ = cond.shape
            ar_n = torch.arange(N, device=cond.device)[:, None]
            ar_l = torch.arange(L, device=cond.device)[None, :]
            return cond[ar_n, ar_l, S_batch].sum(1)

        def _tok_idxs(tokset):
            if isinstance(tokset, str):
                tokset = tokset.split()
            missing = sorted({t for t in tokset if isinstance(t, str) and t not in t2i})
            if missing:
                raise KeyError(f"binder_tokens not in the model vocabulary: {missing}")
            return [t2i[t] if isinstance(t, str) else int(t) for t in tokset]

        # ── featurise the three systems once ──
        aa_c, inp_c, taa_c, etab_c, E_idx_c = _feat(structure)
        S_c0 = inp_c["S"][0]
        chc = [str(c) for c in taa_c.chain_id]
        n_binder = sum(c == binder_chain for c in chc); n_target = len(chc) - n_binder
        bpos_c = _binder_order(taa_c)
        chain_all = np.array([str(c) for c in aa_c.chain_id])
        key_S = {(str(c), int(r)): int(s) for c, r, s in zip(taa_c.chain_id, taa_c.res_id, S_c0.cpu())}

        if not (chain_all == binder_chain).any() or not (chain_all != binder_chain).any():
            # degenerate (single-chain) — fall back to per-design compute_g_bind
            return [self.compute_g_bind(structure, binder_chain, ts) for ts in (binder_token_sets or [None])]

        aa_b, inp_b, taa_b, etab_b, E_idx_b = _feat(aa_c[chain_all == binder_chain])
        S_b0 = inp_b["S"][0]; bpos_b = _binder_order(taa_b)
        aa_t, inp_t, taa_t, etab_t, E_idx_t = _feat(aa_c[chain_all != binder_chain])
        S_t = inp_t["S"][0].clone()
        for p, (c, r) in enumerate(zip(taa_t.chain_id, taa_t.res_id)):     # target fixed to complex tokens
            k = (str(c), int(r))
            if k in key_S:
                S_t[p] = key_S[k]
        E_target = float(_native_sum(etab_t, E_idx_t, S_t.unsqueeze(0))[0])

        token_sets = list(binder_token_sets) if binder_token_sets is not None else [None]
        Sc_list, Sb_list = [], []
        for ts in token_sets:
            Sc = S_c0.clone(); Sb = S_b0.clone()
            if ts is not None:
                idxs = _tok_idxs(ts)
                if len(idxs) != len(bpos_c):
                    raise ValueError(f"binder_tokens length {len(idxs)} != {len(bpos_c)} binder residues")
                for p, a in zip(bpos_c, idxs):
                    Sc[p] = a
                for p, a in zip(bpos_b, idxs):
                    Sb[p] = a
            Sc_list.append(Sc); Sb_list.append(Sb)

        def _batched(etab, E_idx, S_list):
            outs = []
            for i in range(0, len(S_list), chunk):
                outs.append(_native_sum(etab, E_idx, torch.stack(S_list[i:i + chunk])))
            return torch.cat(outs) if outs else torch.empty(0)

        E_cplx = _batched(etab_c, E_idx_c, Sc_list).cpu().tolist()
        E_bind = _batched(etab_b, E_idx_b, Sb_list).cpu().tolist()
        return [dict(E_complex=ec, E_binder=eb, E_target=E_target, g_bind=ec - (eb + E_target),
                     n_binder=int(n_binder), n_target=int(n_target))
                for ec, eb in zip(E_cplx, E_bind)]

    # ------------------------------------------------------------------ #
    # State-specified switch design (mpnn.ph): any pH direction, both sides
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def run_switch_design(
        self,
        *,
        atom_array: AtomArray,
        binder_chain: str,
        config,
        seed: Optional[int] = None,
    ) -> SwitchDesignRun:
        """Design the binder against condition-specific protonation states.

        Unlike :meth:`run_ph_redesign` (which pins protonated binder centres and favours
        them in the complex), the conditions here name the state of any residue, binder
        or receptor, at each pH, and the objective combines stability, potency and
        switch terms (see ``mpnn.ph``). ``config`` is a ``SwitchDesignConfig`` or the
        plain dict it is parsed from. ``seed``, when given, overrides ``config.base_seed``;
        the search draws from ``base_seed + 1000 * run + seed_index``.

        Not yet exercised end to end: the wiring below needs HBPLUS and atomworks. The
        design core (``mpnn.ph``) and the glue helpers are unit-tested on synthetic data.
        """
        if not isinstance(config, SwitchDesignConfig):
            config = SwitchDesignConfig.from_dict(config)
        config = _with_seed(config, seed)
        _check_propka_available(config)
        logger.info("Protonation sources: %s.", _describe_protonation(config.protonation))
        ctx = self._build_context(
            atom_array, binder_chain, base_seed=config.base_seed, protonation=config.protonation
        )
        inputs = self._design_inputs(ctx, binder_chain, with_site_pka=_needs_site_pka(config))
        return _run_switch_design(inputs, config)

    @torch.no_grad()
    def run_switch_design_ensemble(
        self,
        *,
        structures: Dict[str, AtomArray],
        binder_chain: str,
        config,
        seed: Optional[int] = None,
    ) -> EnsembleRun:
        """Design one binder against several target states, one complex per state.

        ``structures`` maps a state name (for example ``"human"``, ``"mouse"``) to its
        binder-plus-target complex; every complex must hold the same binder (same residues
        and sequence) and, for the conditions' sites, the same residue numbering. The first
        entry is the reference. ``seed``, when given, overrides ``config.base_seed``. Terms
        are reduced over states by ``config.target_reduce``
        (``max`` optimises the worst state). Like :meth:`run_switch_design`, not yet run end
        to end on real structures.
        """
        if not isinstance(config, SwitchDesignConfig):
            config = SwitchDesignConfig.from_dict(config)
        config = _with_seed(config, seed)
        _check_propka_available(config)
        logger.info("Protonation sources: %s.", _describe_protonation(config.protonation))
        inputs = {}
        for name, atom_array in structures.items():
            ctx = self._build_context(
                atom_array, binder_chain, base_seed=config.base_seed, protonation=config.protonation
            )
            inputs[name] = self._design_inputs(
                ctx, binder_chain, with_site_pka=_needs_site_pka(config)
            )
        return _run_switch_design_ensemble(inputs, config)

    def _free_site_pka(self, ctx: "_PHContext", binder_chain: str) -> Dict[SiteKey, float]:
        """PROPKA's reference pKa of every His, Asp and Glu, run on the free binder and on the
        free receptor. The pKa shift that binding brings is not in these numbers: it comes from
        the Potts energies of the linked-equilibrium model."""
        from mpnn.transforms.propka_labeller import propka_site_pka

        atoms = ctx.proc_atom_array
        if atoms is None:
            raise RuntimeError("The context carries no processed atom array.")
        on_binder = np.array([str(c) == binder_chain for c in atoms.chain_id])
        site_pka: Dict[SiteKey, float] = {}
        for role, partner in (("binder", atoms[on_binder]), ("receptor", atoms[~on_binder])):
            logger.info(
                "Running PROPKA on the free %s (%d atoms) for the reference pKa.", role, len(partner)
            )
            for (chain, res_id), value in propka_site_pka(partner).items():
                site_pka[SiteKey(chain, res_id)] = value
        return site_pka

    def _design_inputs(
        self, ctx: "_PHContext", binder_chain: str, *, with_site_pka: bool = False
    ) -> DesignInputs:
        """Describe the featurised complex to ``mpnn.ph``: tokens, residues, 3 scorers."""
        names = [str(ctx.encoding.idx_to_token[i]) for i in range(ctx.V)]
        chain_ids = [str(c) for c in ctx.token_aa.chain_id]
        res_ids = [int(r) for r in ctx.token_aa.res_id]
        site_index = site_index_from_arrays(chain_ids, res_ids)
        receptor_chains = sorted(set(chain_ids) - {binder_chain})
        if binder_chain not in chain_ids or not receptor_chains:
            raise ValueError(
                f"Need the binder chain {binder_chain!r} and at least one other chain "
                f"(chains present: {sorted(set(chain_ids))})."
            )
        binder_scorer, binder_positions = self._isolated_scorer(
            ctx, [binder_chain], site_index
        )
        receptor_scorer, receptor_positions = self._isolated_scorer(
            ctx, receptor_chains, site_index
        )
        return DesignInputs(
            tokens=ctx.S_native,
            table=TokenTable(names),
            chain_ids=chain_ids,
            res_ids=res_ids,
            parents=parent_names(
                names, ctx.canonical_map.tolist(), ctx.S_native.tolist()
            ),
            complex_scorer=PottsScorerAdapter(ctx.scorer),
            binder_scorer=binder_scorer,
            receptor_scorer=receptor_scorer,
            binder_positions=binder_positions,
            receptor_positions=receptor_positions,
            partner_rank=knn_partner_rank(ctx.eidx_np),
            neighbour_index=ctx.eidx_np,
            site_pka=self._free_site_pka(ctx, binder_chain) if with_site_pka else None,
        )

    def _isolated_scorer(self, ctx: "_PHContext", chains, site_index):
        """Re-encode ``chains`` alone: ``(scorer, complex positions in its order)``.

        Mirrors :meth:`compute_g_bind`: removing the partner changes each chain's kNN
        graph, so its energy is not a slice of the complex's. The protonation labeller
        also runs on the isolated chain although its tokens are discarded (the complex's
        tokens are used), which is wasteful but keeps the featurisation identical.
        """
        atoms = ctx.proc_atom_array
        if atoms is None:
            raise RuntimeError("The context carries no processed atom array.")
        keep = np.isin(np.array([str(c) for c in atoms.chain_id]), list(chains))
        if not keep.any():
            raise ValueError(f"No atoms for chains {list(chains)}.")
        vocab = getattr(self, "_vocab_name", None) or "v4"
        batch = prepare_potts_input(
            atoms[keep],
            structure_noise=0.0,
            extended_vocab=vocab,
            protonation_classifier=ctx.protonation_classifier,
        )
        network_input = _to_device(batch["network_input"], self.device)
        out = run_forward_compat(self.model, network_input)
        isolated = batch["atom_array"]
        token_isolated = isolated[get_token_starts(isolated)]
        positions = positions_in_complex(
            token_isolated.chain_id, token_isolated.res_id, site_index
        )
        scorer = PottsScorerAdapter(_PottsScorer(out["etab_out"], out["E_idx"]))
        return scorer, positions

    # ------------------------------------------------------------------ #
    # Public entrypoint
    # ------------------------------------------------------------------ #
    def run_ph_redesign(
        self,
        *,
        atom_array: AtomArray,
        binder_chain: str,
        criteria_list: Sequence[PHDesignCriteria],
        seed: int = 0,
        initial_sequences: Optional[Sequence[str]] = None,
        n_jobs: int = 1,
        protonation=None,
    ) -> PHDesignSet:
        """Featurise + forward ONCE, then run each criteria against the shared Potts tables.

        ``protonation`` (a ``ProtonationSources`` or its dict) chooses where the input structure's
        protonation states come from: EV6 by default, or ``{"states": "propka", "ph": ...}``.
        Its ``reference_pka`` belongs to ``linked_equilibrium`` and is refused here.

        All criteria must share this engine's checkpoint/vocab; they may differ in method and
        knobs (the sweep). Returns a ``PHDesignSet`` (deduped, lowest energy first).

        ``initial_sequences`` are externally supplied binder seed sequences (one-letter strings,
        binder-chain only) used when a criteria has ``seed_source="inverse"`` — e.g. the
        ProteinMPNN sequences produced by the inverse stage. Each seed is encoded onto the binder
        positions (target positions stay native) and the placement redesign is run once per seed.

        ``n_jobs`` > 1 fans the independent (criteria × seed) solves across a fork pool that shares
        this backbone's featurised tables read-only (COW). CPU only; results are identical to the
        serial path (each design re-seeds its own RNG), just produced out of order then re-sorted.
        """
        protonation = _as_protonation(protonation)
        if protonation.reference_pka != "constant":
            raise ValueError(
                "protonation.reference_pka only applies to run_switch_design with the "
                "linked_equilibrium binding model; run_ph_redesign has no pKa to set."
            )
        logger.info("Protonation sources: %s.", _describe_protonation(protonation))
        ctx = self._build_context(atom_array, binder_chain, base_seed=seed, protonation=protonation)
        ctx.external_initial_sequences = list(initial_sequences) if initial_sequences else []
        # Score the ORIGINAL seeds (as supplied, no RES-P introduced) with the same featurised
        # tables — stability H + binder global protonation ΔH — so the seed energy is available for
        # the handler to persist. Indexed by seed position == seed_idx of the resulting designs.
        seed_energies = []
        for s in ctx.external_initial_sequences:
            try:
                S = ctx.encode_initial_sequence(s)
                g = _global_protonation_dH(ctx, S)
                seed_energies.append({"potts_energy": ctx.scorer.H_of(S),
                                      "global_protonation_dH": (float(g) if g is not None else None)})
            except Exception:
                seed_energies.append({"potts_energy": None, "global_protonation_dH": None})
        # Independent units of work: (criteria_index, seq_i). One per placement seed; whole-chain
        # methods are a single unit (seq_i=0, ignored).
        criteria_list = list(criteria_list)
        tasks = []
        for ci, crit in enumerate(criteria_list):
            if crit.method in WHOLE_CHAIN_METHODS:
                tasks.append((ci, 0))
            else:
                for seq_i in range(self._n_placement_seeds(ctx, crit) or 1):
                    tasks.append((ci, seq_i))

        results = PHDesignSet()
        n_jobs = max(1, int(n_jobs))
        if n_jobs > 1 and str(ctx.device) == "cpu" and len(tasks) > 1:
            results.extend(self._run_tasks_parallel(ctx, criteria_list, seed, tasks, n_jobs))
        else:
            for task in tasks:
                results.extend(_run_design_task(self, ctx, criteria_list, seed, task))
        results = results.deduped().sorted_by_energy()
        results.seed_energies = seed_energies          # attached for the adapter/handler to read
        return results

    def _run_tasks_parallel(self, ctx, criteria_list, seed, tasks, n_jobs) -> PHDesignSet:
        """Fork a pool over the (criteria × seed) tasks, sharing ctx read-only via COW (set as module
        globals before the fork). Returns all designs (order-independent; run_ph_redesign re-sorts)."""
        global _POOL_CTX, _POOL_ENGINE, _POOL_CRITERIA, _POOL_SEED
        _POOL_CTX, _POOL_ENGINE, _POOL_CRITERIA, _POOL_SEED = ctx, self, criteria_list, seed
        out = PHDesignSet()
        try:
            with mp.get_context("fork").Pool(min(n_jobs, len(tasks)), initializer=_pool_init) as pool:
                for sub in pool.imap(_pool_task, tasks):   # ordered: same result list as the serial path
                    out.extend(sub)
        finally:
            _POOL_CTX = _POOL_ENGINE = _POOL_CRITERIA = None
        return out

    # ------------------------------------------------------------------ #
    # Context: featurise once, expose design field + scorer + masks
    # ------------------------------------------------------------------ #
    def _build_context(
        self, atom_array: AtomArray, binder_chain: str, *, base_seed: int = 0, protonation=None
    ) -> "_PHContext":
        encoding = self.model.graph_featurization_module.TOKEN_ENCODING
        classifier = _protonation_classifier(_as_protonation(protonation))

        # Featurise with the protonation-aware Potts pipeline (mpnn.potts_inference), NOT the
        # engine's _build_network_input — that one is wired to the standard mpnn pipeline which
        # rejects model_type="potts_mpnn".
        # Featurise S in the SAME vocabulary the model was built with. A legacy bool/None keeps the
        # historical default ("v4", exactly what prepare_potts_input already used) so 32-token runs are
        # byte-identical; an explicit vocab name (e.g. "v6") makes S use that vocab's token indexing —
        # MANDATORY for a 30-token v6 model (its indices differ from the 32-token set).
        _vocab = getattr(self, "_vocab_name", None) or "v4"
        batch = prepare_potts_input(
            atom_array, designed_chains=[binder_chain], structure_noise=0.0,
            extended_vocab=_vocab, protonation_classifier=classifier,
        )
        network_input = batch["network_input"]
        # prepare_potts_input builds tensors on the default device (CUDA when a GPU is visible),
        # but the model lives on self.device. Move the input onto self.device so a forced
        # device="cpu" (needed for the n_jobs CPU fork-pool) doesn't hit a cpu/cuda mismatch in
        # the forward. No-op when self.device already matches the input's device.
        def _to_engine_device(obj):
            if isinstance(obj, torch.Tensor):
                return obj.to(self.device)
            if isinstance(obj, dict):
                return {k: _to_engine_device(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return type(obj)(_to_engine_device(v) for v in obj)
            return obj
        network_input = _to_engine_device(network_input)
        proc_atom_array = batch["atom_array"]
        token_aa = proc_atom_array[get_token_starts(proc_atom_array)]

        # Native sequence BEFORE forward (forward mutates network_input in place).
        S_native = network_input["input_features"]["S"].squeeze(0).clone().to(self.device)
        ni_tf = copy.deepcopy(network_input)  # decoder teacher-forced template

        po = run_forward_compat(self.model, network_input)
        etab_out, E_idx = po["etab_out"], po["E_idx"]
        scorer = _PottsScorer(etab_out, E_idx)

        free_mask = network_input["input_features"]["designed_residue_mask"].squeeze(0).bool()
        V = int(self.model.potts_vocab_size)
        eidx_np = E_idx.squeeze(0).cpu().numpy()

        # decoder design field template (conditional_minus_self), reused across calls.
        ni_tf["input_features"].update({
            "decode_type": "teacher_forcing",
            "causality_pattern": "conditional_minus_self",
            "initialize_sequence_embedding_with_ground_truth": True,
            "repeat_sample_num": 1,
        })

        def field_potts(S: torch.Tensor) -> torch.Tensor:
            return scorer.cond_energy(S)

        def field_mpnn(S: torch.Tensor) -> torch.Tensor:
            ni_tf["input_features"]["S"] = S.unsqueeze(0)
            with torch.no_grad():
                lp = self.model(ni_tf)["decoder_features"]["log_probs"].squeeze(0)
            return -lp

        # Pass the model's OWN encoding so protonation tokens decode to the right parent by index
        # (v6's 30-token indices differ from the 32-token default; for 32-token models this is the
        # same map as the no-arg default, so it stays byte-identical).
        canonical_map = (
            _build_canonical_map(encoding).to(torch.long)
            if self.extended_vocab else torch.arange(V, dtype=torch.long)
        )

        chainA = (token_aa.chain_id == binder_chain).astype(bool)
        chainA_t = torch.from_numpy(chainA).to(self.device)

        region_masks = _binder_region_masks(
            proc_atom_array, token_aa, binder_chain, chainA_t, self.device, int(S_native.shape[0]))

        return _PHContext(
            device=self.device, encoding=encoding, extended_vocab=self.extended_vocab,
            canonical_map=canonical_map, token_aa=token_aa, S_native=S_native,
            scorer=scorer, field_potts=field_potts, field_mpnn=field_mpnn,
            free_mask=free_mask, V=V, L=int(S_native.shape[0]),
            unknown_indices=list(self.model.unknown_token_indices),
            eidx_np=eidx_np, K=int(eidx_np.shape[1]),
            chainA_t=chainA_t, binder_chain=binder_chain,
            network_input=network_input, base_seed=int(base_seed),
            region_masks=region_masks,
            proc_atom_array=proc_atom_array,
            protonation_classifier=classifier,
        )

    # ------------------------------------------------------------------ #
    # Per-criteria dispatch
    # ------------------------------------------------------------------ #
    @staticmethod
    def _criteria_fields(ctx, crit):
        """(field_fn, field_at, valid_mask) for a criteria. field_fn = full [L,V] design field;
        field_at(S,i) = fast per-position scorer (potts: slice the fixed etab — ≈L× cheaper and
        identical to field_fn(S)[i]; mpnn: full field then index, no single-position shortcut)."""
        field_fn = ctx.field_potts if crit.backend == "potts" else ctx.field_mpnn
        field_at = (ctx.scorer.cond_energy_at if crit.backend == "potts"
                    else (lambda S, i: ctx.field_mpnn(S)[i]))
        return field_fn, field_at, ctx.valid_aa_mask(crit.forbidden_tokens)

    def _run_one_criteria(self, ctx: "_PHContext", crit: PHDesignCriteria, *, seed: int) -> PHDesignSet:
        field_fn, field_at, valid_mask = self._criteria_fields(ctx, crit)
        if crit.method in WHOLE_CHAIN_METHODS:
            return self._run_whole_chain(ctx, crit, field_fn, valid_mask, seed=seed)
        return self._run_placement(ctx, crit, field_fn, field_at, valid_mask, seed=seed)

    # --- whole-chain baselines -------------------------------------------------
    def _run_whole_chain(self, ctx, crit, field_fn, valid_mask, *, seed) -> PHDesignSet:
        torch.manual_seed(seed)
        N = crit.num_designs
        if crit.method == "gibbs":
            seq_init = ctx.S_native.unsqueeze(0).expand(N, -1).clone()
            seq_opt, _ = PottsMPNN.potts_gibbs_optimize(
                etab_out=ctx.scorer.etab_out, E_idx=ctx.scorer.E_idx,
                seq_init=seq_init, free_mask=ctx.free_mask,
                temperature=max(crit.temperature, 1e-3),
                max_iters=1000, convergence_mode=True, valid_aa_mask=valid_mask,
            )
            seqs = [seq_opt[i] for i in range(N)]
        else:  # mpnn_sample
            ni = copy.deepcopy(ctx.network_input)
            ni["input_features"]["repeat_sample_num"] = N
            with torch.no_grad():
                S_sampled = self.model(ni)["decoder_features"]["S_sampled"]  # [N, L]
            seqs = [S_sampled[i].to(ctx.device) for i in range(N)]

        out = PHDesignSet()
        for s, seq in enumerate(seqs):
            out.append(self._score_design(ctx, crit, seq, scheme=crit.scheme_label(), sample=s,
                                          field_fn=field_fn, valid_mask=valid_mask))
        return out

    # --- placement-based protonation redesign ----------------------------------
    @staticmethod
    def _n_placement_seeds(ctx, crit) -> int:
        """How many initial sequences the placement redesign runs over (inverse: one per supplied
        seed; native: one)."""
        if crit.seed_source == "inverse":
            return len(ctx.external_initial_sequences)
        return 1

    @staticmethod
    def _placement_initial(ctx, crit, seq_i):
        """Encode the seq_i-th initial sequence + whether it's a seeded run. inverse -> the seq_i-th
        external seed (encoded onto the binder positions); native -> the native/RFD3 sequence."""
        if crit.seed_source == "inverse":
            if not ctx.external_initial_sequences:
                raise ValueError(
                    "seed_source='inverse' but no initial_sequences provided — run the inverse "
                    "stage first (or use seed_source='native').")
            return ctx.encode_initial_sequence(ctx.external_initial_sequences[seq_i]), True
        return ctx.S_native, False

    def _run_placement(self, ctx, crit, field_fn, field_at, valid_mask, *, seed) -> PHDesignSet:
        out = PHDesignSet()
        for seq_i in range(self._n_placement_seeds(ctx, crit)):
            initial_sequence, seeded = self._placement_initial(ctx, crit, seq_i)
            out.extend(self._run_placement_one_seed(
                ctx, crit, field_fn, field_at, valid_mask, seed, seq_i, initial_sequence, seeded))
        return out

    def _run_placement_one_seed(self, ctx, crit, field_fn, field_at, valid_mask, seed, seq_i,
                                initial_sequence, seeded) -> PHDesignSet:
        """All designs for ONE initial sequence (the independent unit parallelised by run_ph_redesign):
        enumerate multi-center placement plans, redesign the union of their neighbourhoods. Self-contained
        + deterministic (each design re-seeds torch RNG from (seq_i, plan, sample)) so it forks cleanly."""
        rng = random.Random(seed * 100003 + seq_i * 7919)
        out = PHDesignSet()
        seed_idx = seq_i if seeded else None
        # placement distribution over THIS initial sequence (for placement_prob / entropy).
        P_init = _cond_dist(field_fn(initial_sequence), valid_mask, crit.temperature)
        plans = self.enumerate_placement_plans(ctx, crit, field_fn, valid_mask, initial_sequence, rng=rng)
        for plan in plans:
            if not plan.designable:
                continue
            for s in range(crit.samples_per_site):
                torch.manual_seed(seed * 100003 + seq_i * 7919 + plan.seed_key + s)
                seq, traj = self._design_one(ctx, crit, field_fn, field_at, valid_mask, plan, rng,
                                             initial_sequence, sample=s)
                out.append(self._score_design(
                    ctx, crit, seq, scheme=crit.scheme_label(), sample=s,
                    field_fn=field_fn, valid_mask=valid_mask, seed_idx=seed_idx,
                    plan=plan, P_native=P_init, energy_trajectory=traj))
        return out

    # --- placement plan enumeration (multi-center, mix-and-match, region + scan) --------------
    def _pin_idxs(self, ctx, crit, ptype):
        # Fail loudly (not with a cryptic KeyError) when a centre/contrast token is not in the
        # model's vocabulary — e.g. a v3/v4 dep_map (HID/HIE) used against a v6 (HIS-S) checkpoint.
        t2i = ctx.encoding.token_to_idx
        if ptype not in t2i:
            raise KeyError(
                f"centre protonation token {ptype!r} is not in the model vocabulary "
                f"{sorted(t2i)}. (v6 uses HIS-S for neutral His; HID/HIE/HIS-D exist only in v3/v4.)")
        deps = crit.dep_map.get(ptype)
        if not deps:
            raise KeyError(f"dep_map has no contrast tokens for centre {ptype!r} (dep_map={crit.dep_map!r}).")
        missing = [t for t in deps if t not in t2i]
        if missing:
            raise KeyError(
                f"dep_map[{ptype!r}] references tokens {missing} absent from the model vocabulary "
                f"{sorted(t2i)}. (v6 has no HID/HIE — use HIS-P as the charged contrast.)")
        return (t2i[ptype], [t2i[t] for t in deps])

    def _pin(self, ctx, crit, position, ptype) -> PlacementPin:
        prot_idx, dep_idxs = self._pin_idxs(ctx, crit, ptype)
        return PlacementPin(position=int(position), protonation_type=ptype, prot_idx=prot_idx,
                            dep_idxs=dep_idxs, res_id=int(ctx.token_aa.res_id[int(position)]))

    def score_native_baseline(self, ctx, crit, *, centers=None) -> Dict[str, Any]:
        """Score the NATIVE (input-PDB) binder sequence with the design energy recipe, as a baseline.

        ``final_potts_energy`` is H of the native sequence exactly as it comes off the backbone --
        nothing pinned -- so it is the honest "before any design" number, directly comparable to a
        design's ``final_potts_energy`` (both are ``ctx.scorer.H_of``).

        ``selective_energy`` is only meaningful relative to a SET OF CENTRES, so it is measured the
        way a design's is: the same centre positions locked to RES-P, each gap evaluated in the
        presence of the others (``_selective_energy_sum``, which is what ``_score_design`` reduces to
        -- a design's sequence already carries its centres pinned). Pass ``centers`` as
        ``(res_id, protonation_type)`` pairs, normally a design's ``center_res_ids`` /
        ``center_protonation_types``, to compare like with like; the number says what selective gap
        the NATIVE sequence context already afforded at those sites, before the redesign.

        """
        binder_idx = ctx.chainA_all_idx.tolist()
        seq = ctx.S_native
        res_to_pos = {int(ctx.token_aa.res_id[int(i)]): int(i) for i in binder_idx}
        pins = []
        for res_id, ptype in (centers or []):
            if int(res_id) not in res_to_pos:
                raise KeyError(
                    f"centre res_id {res_id} is not a position of binder chain {ctx.binder_chain!r} "
                    f"(binder res_ids {min(res_to_pos)}..{max(res_to_pos)}).")
            pins.append(self._pin(ctx, crit, res_to_pos[int(res_id)], ptype))
        return {
            "canonical_sequence": ctx.decode_canonical(seq, binder_idx),
            "extended_tokens": ctx.extended_tokens(seq, binder_idx),
            "final_potts_energy": ctx.scorer.H_of(seq),
            "selective_energy": _selective_energy_sum(ctx, seq, pins) if pins else None,
            "global_protonation_dH": _global_protonation_dH(ctx, seq),
            "centers": [(int(p.res_id), p.protonation_type) for p in pins],
        }

    def enumerate_placement_plans(self, ctx, crit, field_fn, valid_mask, initial_sequence, *, rng):
        """Placement plans for this seed. explicit_centers -> ONE hand-picked plan. Otherwise draw
        candidate positions from the chosen placement function (random / scan_potts / scan_mpnn) within
        the region, form center_count-combinations (mix-and-match types), and (for the scan strategies)
        rank them by the locked joint Σ_centers (e_P - e_D). Keeps max_plans_per_seed."""
        if crit.center_count == 0:                       # CENTRE-FREE (greedy_energy_block): no pins.
            # designable = free binder positions, restricted to placement_region (e.g. [interface] to
            # focus on the interacting residues; [all] = whole chain).
            free = set(int(x) for x in ctx.chA_free_idx.tolist())
            if crit.placement_region and "all" not in crit.placement_region:
                region = set(int(x) for x in ctx.region_mask(crit.placement_region, crit.interface_distance).nonzero().flatten().tolist())
                free = free & region
            designable = sorted(free)
            if not designable:
                return []
            return [PlacementPlan(pins=[], designable=designable, label="")]

        if crit.explicit_centers:
            pins = [self._pin(ctx, crit, ctx.binder_pos_of_res_id(int(c["res_id"])),
                              str(c["protonation_type"])) for c in crit.explicit_centers]
            plan = _finalize_plan(ctx, crit, pins)
            return [plan] if plan is not None else []

        if crit.center_types:
            # EXACT composition: place one distinct, best-ranked site per type (positions chosen by the
            # placement method within the region). Yields exactly ONE plan of that multiset.
            region_idx = ctx.region_mask(crit.placement_region, crit.interface_distance).nonzero().flatten()
            if int(region_idx.numel()) < len(crit.center_types):
                return []
            place = PlacementPlan.placement_fn(crit.placement_by)
            used, pins = set(), []
            for t in crit.center_types:
                prot_idx, dep_idxs = self._pin_idxs(ctx, crit, t)
                ranked = place(ctx, crit, region_idx, prot_idx, dep_idxs, initial_sequence, rng)
                pos = next((p for p, _ in ranked if int(p) not in used), None)
                if pos is None:
                    return []
                used.add(int(pos))
                pins.append(self._pin(ctx, crit, int(pos), t))
            plan = _finalize_plan(ctx, crit, pins)
            return [plan] if plan is not None else []

        region_idx = ctx.region_mask(crit.placement_region, crit.interface_distance).nonzero().flatten()
        if int(region_idx.numel()) < crit.center_count:
            return []
        place = PlacementPlan.placement_fn(crit.placement_by)
        pool = []                                              # (position, protonation_type)
        for t in crit.center_protonation_types:
            prot_idx, dep_idxs = self._pin_idxs(ctx, crit, t)
            ranked = place(ctx, crit, region_idx, prot_idx, dep_idxs, initial_sequence, rng)
            pool.extend((pos, t) for pos, _ in ranked[: max(1, crit.candidate_pool)])
        combos = _combos_or_sample(pool, crit.center_count, max(1, crit.n_plan_samples), rng)

        rank = crit.placement_by != "random"
        place_field = ctx.field_mpnn if crit.placement_by == "scan_mpnn" else ctx.field_potts
        cands = []
        for combo in combos:
            pins = [self._pin(ctx, crit, pos, t) for pos, t in combo]
            score = _locked_selective_score(ctx, place_field, initial_sequence, pins) if rank else 0.0
            cands.append((score, pins))
        if rank:
            cands.sort(key=lambda x: x[0])
        plans = []
        for score, pins in cands[: (crit.max_plans_per_seed or len(cands))]:
            plan = _finalize_plan(ctx, crit, pins)
            if plan is not None:
                plan.placement_score = float(score)
                plans.append(plan)
        return plans

    def _placement_sites(self, ctx, crit, field_fn, valid_mask, prot_idx, dep_idxs,
                         initial_sequence) -> List[int]:
        """Top-K binder sites for placing RES-P given ``initial_sequence`` (lower energy = better).

        ``initial_sequence`` is the full starting sequence to score against (the native/RFD3 sequence,
        or an MPNN seed). non-selective: design-field energy of RES-P minus the placement-sequence
        residue (ΔE-like). selective: that minus the best deprotonated alternative.

        ``crit.placement_seq_masked``: score against a binder-masked sequence (all design positions →
        UNK) instead of ``initial_sequence``, so each site's selective score depends only on the
        backbone (+ fixed target), not the current binder sequence ("structure-only" placement).
        The per-site ``base`` cancels in the selective score, so masking only affects the selective
        ranking through the field itself.
        """
        chA = ctx.chA_free_idx
        placement_seq = initial_sequence
        if crit.placement_seq_masked:
            placement_seq = initial_sequence.clone()
            placement_seq[chA] = ctx.encoding.token_to_idx["UNK"]
        ef = field_fn(placement_seq)                            # [L, V]
        base = ef[chA, placement_seq[chA]]                       # placement-sequence energy per site
        score = ef[chA, prot_idx] - base                     # ΔE(RES-P)
        if crit.selective:
            dep_dE = torch.stack([ef[chA, d] - base for d in dep_idxs], 0).min(0).values
            score = score - dep_dE
        order = torch.argsort(score)[: crit.topk_sites]
        return [int(chA[k]) for k in order]

    def _design_one(self, ctx, crit, field_fn, field_at, valid_mask, plan, rng, initial_sequence, *, sample):
        """Redesign the union of the plan's center neighbourhoods (all centers pinned to RES-P), starting
        from ``initial_sequence``. Returns ``(seq, trajectory)`` (trajectory None unless record_trajectory)."""
        trajectory = [] if crit.record_trajectory else None
        pins, neigh = plan.pins, plan.designable
        method = crit.method
        if method == "autoregressive":
            order = rng.sample(neigh, len(neigh))
            seq = _masked_infill(ctx, crit, valid_mask, pins, order, initial_sequence, trajectory=trajectory)
        elif method == "two_phase":
            seq = _two_phase(ctx, crit, field_at, valid_mask, pins, neigh, initial_sequence,
                             trajectory=trajectory)
        elif method == "block_descent":
            seq = _block_descent(ctx, crit, valid_mask, pins, neigh, initial_sequence, rng,
                                 trajectory=trajectory)
        elif method == "greedy_energy_block":
            seq = _greedy_energy_block(ctx, crit, valid_mask, pins, neigh, initial_sequence, rng,
                                       trajectory=trajectory)
        else:  # converged_mcmc / converged_mcmc_combined
            seq = _converge(ctx, crit, field_at, valid_mask, pins, neigh, initial_sequence,
                            trajectory=trajectory)
        return seq, trajectory

    # ------------------------------------------------------------------ #
    # Scoring + decode
    # ------------------------------------------------------------------ #
    def _score_design(self, ctx, crit, seq, *, scheme, sample, field_fn, valid_mask,
                      seed_idx=None, plan=None, P_native=None,
                      energy_trajectory=None) -> PHDesignOutput:
        binder_idx = ctx.chainA_all_idx.tolist()
        canonical = ctx.decode_canonical(seq, binder_idx)
        ext = ctx.extended_tokens(seq, binder_idx)
        final_e = ctx.scorer.H_of(seq)

        sel_e = glob = pl_prob = pl_ent = None
        n_neigh = center_res = proto = None
        n_centers = center_res_ids = center_types = sel_each = None
        score_positions = binder_idx
        if plan is not None:
            pins = plan.pins
            n_neigh = len(plan.designable)
            proto = plan.label                                 # single token OR combo label ("" if centre-free)
            n_centers = len(pins)
            glob = _global_protonation_dH(ctx, seq)
            score_positions = plan.designable or binder_idx
            if pins:
                primary = pins[0]
                # seq already carries every center pinned to RES-P (the redesign never touched them),
                # so each per-pin selective gap is evaluated in the presence of the others.
                sel_each = [_selective_energy_of(ctx, seq, p.position, p.prot_idx, p.dep_idxs) for p in pins]
                sel_e = float(sum(sel_each))
                valid_idx = valid_mask.nonzero().flatten()
                vp = {int(t): i for i, t in enumerate(valid_idx.tolist())}
                if P_native is not None and primary.prot_idx in vp:
                    pl_prob = float(P_native[primary.position, vp[primary.prot_idx]])
                    pl_ent = float(_entropy_bits(P_native[primary.position]))
                center_res = primary.res_id
                center_res_ids = [p.res_id for p in pins]
                center_types = [p.protonation_type for p in pins]
            else:                                              # centre-free (greedy_energy_block): no pins
                sel_each, sel_e = [], 0.0

        dec = _decoded_prob_score(ctx, crit, field_fn, valid_mask, seq, score_positions)
        seq_entropy = _seq_entropy_bits(ctx.decode_canonical(seq, score_positions))   # redesigned-region diversity

        return PHDesignOutput(
            binder_chain=ctx.binder_chain, canonical_sequence=canonical, extended_tokens=ext,
            extended_vocab=ctx.extended_vocab, scheme=scheme, sample=sample, seed_idx=seed_idx,
            final_potts_energy=final_e, protonation_type=proto, site_rank=None,
            center_res_id=center_res, n_neighbours=n_neigh, selective_energy=sel_e,
            n_centers=n_centers, center_res_ids=center_res_ids, center_protonation_types=center_types,
            selective_energies=sel_each, global_protonation_dH=glob, placement_prob=pl_prob,
            placement_entropy=pl_ent, sequence_decoded_prob_score=dec, sequence_entropy=seq_entropy,
            method=crit.method, backend=crit.backend, selective_source=crit.selective_source,
            placement_label=crit.placement_label,
            placement_region="+".join(crit.placement_region), placement_by=crit.placement_by,
            combined_lambda=crit.combined_lambda,
            repetitive_window_weight=crit.repetitive_window_weight, neighbour_k=crit.neighbour_k,
            max_mutations=crit.max_mutations,
            block_size=crit.block_size,     # solver knob: joint-move size (cost ~ V**block_size)
            sweep_order=crit.sweep_order,   # solver knob: visit order of the designable positions
            energy_trajectory=energy_trajectory,
        )


# ---------------------------------------------------------------------------
# Context + scorer
# ---------------------------------------------------------------------------

class _PottsScorer:
    """Holds the Potts tables (computed ONCE by the head) and scores sequences by slicing them.

    Two deliberately separate scoring functions:
    - ``cond_energy(S)``        → full [L, V] conditional table (every position, every AA).
    - ``cond_energy_at(S, i)``  → only position i's [V] row, sliced straight out of the fixed etab.

    The single-site Gibbs move changes one residue and only needs row i, so ``cond_energy_at`` drops
    the per-move cost from O(L·K·V) to O(K·V) (≈L× faster). Both read the same ``etab``/``E_idx``;
    the head is never re-run. ``cond_energy_at(S, i)`` is numerically identical to ``cond_energy(S)[i]``.
    """

    def __init__(self, etab_out: torch.Tensor, E_idx: torch.Tensor) -> None:
        self.etab_out = etab_out
        self.E_idx = E_idx
        # Precompute the per-position pieces of potts_candidate_energies, ONCE.
        etab = etab_out.squeeze(0)                                   # [L, K, V, V]
        self.etab = etab
        self.L, self.K, self.V, _ = etab.shape
        self.self_term = etab[:, 0].diagonal(dim1=-2, dim2=-1)       # [L, V]  single-site field
        self.nbr_idx = E_idx.squeeze(0)[:, 1:]                       # [L, K-1] outgoing neighbours
        self.pe = etab[:, 1:]                                        # [L, K-1, V, V] pair tables
        # Incoming edges grouped by TARGET position (graph transpose; the kNN graph is asymmetric,
        # so incoming edges are NOT the mirror of outgoing — required to match calc_potts_eners).
        src, slot, tgt = PottsMPNN._directed_pair_edges(E_idx)
        device = etab.device
        inc_src: List[List[int]] = [[] for _ in range(self.L)]
        inc_slot: List[List[int]] = [[] for _ in range(self.L)]
        for s_, sl_, t_ in zip(src.tolist(), slot.tolist(), tgt.tolist()):
            inc_src[t_].append(s_)
            inc_slot[t_].append(sl_)
        self._inc_src = [torch.tensor(x, dtype=torch.long, device=device) for x in inc_src]
        self._inc_slot = [torch.tensor(x, dtype=torch.long, device=device) for x in inc_slot]

    def cond_energy(self, S: torch.Tensor) -> torch.Tensor:
        """Full [L, V] conditional-energy table for sequence S (every position, every candidate AA)."""
        return PottsMPNN.potts_candidate_energies(self.etab_out, self.E_idx, S.unsqueeze(0))[0]

    def cond_energy_at(self, S: torch.Tensor, i: int) -> torch.Tensor:
        """Conditional energies [V] at position ``i`` only — equals ``cond_energy(S)[i]`` but slices
        position i's terms (self + outgoing i→j + incoming m→i) straight out of the fixed etab."""
        i = int(i)
        out = self.self_term[i].clone()                             # [V]  single-site field
        s_nbr = S[self.nbr_idx[i]]                                  # [K-1] outgoing neighbours' tokens
        gi = s_nbr.view(-1, 1, 1).expand(-1, self.V, 1)
        out += torch.gather(self.pe[i], -1, gi).squeeze(-1).sum(0)  # outgoing i→j pair terms
        msrc, mslot = self._inc_src[i], self._inc_slot[i]
        if msrc.numel():
            out += self.etab[msrc, mslot, S[msrc], :].sum(0)        # incoming m→i pair terms
        return out

    def H_of(self, S: torch.Tensor) -> float:
        return PottsMPNN.calc_potts_eners(self.etab_out, self.E_idx, S.unsqueeze(0)).item()

    def block_stability_potentials(self, S: torch.Tensor, block: Sequence[int]):
        """Decompose the part of the Potts Hamiltonian H that depends on the ``block`` positions
        (the rest of S fixed) into a UNARY table + within-block PAIRWISE matrices, so any joint
        assignment to the block can be scored by a sum (enabling exact V**|block| enumeration).

        Returns ``(unary, edges)`` where:
          unary[bi, a]              = self field of block[bi]=a + all its edges to FIXED context
          edges = [(bi, bj, M)]     M[a, b] = directed pair table for within-block edge block[bi]→block[bj],
                                    ALWAYS emitted with bi < bj (see below)
        For any block assignment ``a_B``:
            H_block(a_B) = Σ_bi unary[bi, a_B[bi]] + Σ_(bi,bj,M) M[a_B[bi], a_B[bj]]
        equals (up to a constant independent of the block) the full H restricted to the block.

        ORIENTATION GUARANTEE (bi < bj). Consumers add these tables into the [V]*B joint tensor by
        ``shape = [1]*B; shape[bi] = V; shape[bj] = V; J += M.reshape(shape)`` — but that ``shape``
        is identical whether bi<bj or bi>bj, and ``reshape`` always lays M's FIRST axis on the
        lower-numbered axis. The pair tables are genuinely asymmetric (the kNN graph is ~16%
        non-reciprocal), so an edge with bi>bj would be added TRANSPOSED — silently scoring a
        different objective than H. Emitting every edge with bi<bj (transposing the table, which
        leaves ``M[a_bi, a_bj]`` unchanged in meaning) makes that broadcast correct everywhere with
        no call-site change. Measured before this guarantee: half the within-block edges reversed,
        the block tensor off by up to 2.3 energy units, and _block_descent never terminating —
        every task hit block_max_rounds at block_size>=2, vs 3-4 sweeps once oriented
        (sandbox/blockdescent/compare_edge_orientation.py).
        """
        block = [int(p) for p in block]
        pos_of = {p: bi for bi, p in enumerate(block)}
        B = len(block)
        unary = torch.stack([self.self_term[p].clone() for p in block])      # [B, V]  self fields
        edges = []
        for bi, p in enumerate(block):
            # outgoing p -> j
            nbrs = self.nbr_idx[p].tolist()
            pep = self.pe[p]                                                 # [K-1, V, V]
            for t, j in enumerate(nbrs):
                if j == p:
                    continue
                bj = pos_of.get(j)
                if bj is None:                                              # j fixed -> unary in a_p
                    unary[bi] += pep[t][:, int(S[j])]
                elif bi < bj:                                                # j in block -> pairwise
                    edges.append((bi, bj, pep[t]))                          # M[a_p, a_j]
                else:                                                        # keep bi < bj: M.T[a_j, a_p] == M[a_p, a_j]
                    edges.append((bj, bi, pep[t].T))
            # incoming m -> p (only from FIXED context; within-block incoming = other end's outgoing)
            msrc, mslot = self._inc_src[p], self._inc_slot[p]
            for e in range(msrc.numel()):
                m = int(msrc[e])
                if m in pos_of:
                    continue
                unary[bi] += self.etab[m, int(mslot[e]), int(S[m]), :]
        return unary, edges

    def reweighted(self, w_self: float, w_pair: float) -> "_PottsScorer":
        """A copy of this scorer with the Potts energy rescaled to H = w_self·Σh_i + w_pair·ΣJ_ij.

        Slot 0 of ``etab`` is the pure single-site field h_i (its off-diagonal is zeroed by the head);
        slots ≥1 are the pure pairwise couplings J_ij. Scaling them independently cleanly reweights
        self vs pair. Shares the weight-INDEPENDENT index tensors (``nbr_idx``, ``_inc_src``,
        ``_inc_slot``, ``E_idx``, ``L/K/V``) via a shallow copy and only overwrites the energy tensors,
        so it avoids re-running ``__init__``'s O(edges) incoming-edge loop. Used by ``_block_descent``
        for the STABILITY term only (``block_stability_potentials`` reads self_term/pe/etab)."""
        sc = copy.copy(self)
        etab_w = self.etab.clone()                                   # preserves dtype/device
        etab_w[:, 0] *= float(w_self)                                # self field (slot 0 diagonal)
        etab_w[:, 1:] *= float(w_pair)                               # pure pairwise J (slots ≥1)
        sc.etab = etab_w
        sc.self_term = etab_w[:, 0].diagonal(dim1=-2, dim2=-1)       # derive views AFTER scaling
        sc.pe = etab_w[:, 1:]
        sc.etab_out = etab_w.unsqueeze(0)
        return sc


@dataclass
class _PHContext:
    device: torch.device
    encoding: Any
    extended_vocab: bool
    canonical_map: torch.Tensor
    token_aa: Any
    S_native: torch.Tensor
    scorer: _PottsScorer
    field_potts: Callable[[torch.Tensor], torch.Tensor]
    field_mpnn: Callable[[torch.Tensor], torch.Tensor]
    free_mask: torch.Tensor
    V: int
    L: int
    unknown_indices: List[int]
    eidx_np: np.ndarray
    K: int
    chainA_t: torch.Tensor
    binder_chain: str
    network_input: dict
    base_seed: int = 0
    external_initial_sequences: List[str] = field(default_factory=list)
    # per-ctx-position boolean masks (length L, True only on binder positions) for
    # {"interface","core","surface"}; empty when classification failed (fail-soft).
    region_masks: Dict[str, torch.Tensor] = field(default_factory=dict)
    # processed atom array of the complex; lets chains be re-encoded alone (mpnn.ph)
    proc_atom_array: Any = None
    # the labeller that produced the tokens (None = the vocabulary's own); chains re-encoded alone use it too
    protonation_classifier: Any = None

    def __post_init__(self) -> None:
        self.chA_free_idx = (self.free_mask & self.chainA_t).nonzero(as_tuple=False).squeeze(1)
        self.chainA_all_idx = self.chainA_t.nonzero(as_tuple=False).squeeze(1)
        # binder res_id -> ctx position, for explicit-center placement by res_id.
        self.res_id_to_pos: Dict[int, int] = {
            int(self.token_aa.res_id[int(p)]): int(p) for p in self.chainA_all_idx.tolist()}
        self._unk_one = DICT_THREE_TO_ONE.get(UNKNOWN_AA, "X")
        # one-letter -> base-token index, for encoding external (ProteinMPNN) seed sequences.
        self._one_to_idx: Dict[str, int] = {}
        for idx in range(self.V):
            tok3 = str(self.encoding.idx_to_token[idx])
            one = DICT_THREE_TO_ONE.get(tok3)
            if one is not None and one not in self._one_to_idx:
                self._one_to_idx[one] = idx

    def stability_scorer(self, w_self: float, w_pair: float) -> "_PottsScorer":
        """Scorer whose Potts energy is reweighted to w_self·Σh + w_pair·ΣJ, for the STABILITY term.
        ``w_self == w_pair`` is a uniform scale — removed by the z-scaling of H_stab in the objective —
        so it returns the unweighted ``self.scorer`` unchanged (exact no-op; e.g. the default
        self_weight=1.0). Otherwise a reweighted scorer is built once per distinct weight and cached
        (ctx is per-target and built before the fork, so each worker fills its own cache)."""
        if w_self == w_pair:
            return self.scorer
        key = (round(float(w_self), 6), round(float(w_pair), 6))
        cache = self.__dict__.setdefault("_stab_scorer_cache", {})
        sc = cache.get(key)
        if sc is None:
            sc = self.scorer.reweighted(w_self, w_pair)
            cache[key] = sc
        return sc

    def encode_initial_sequence(self, one_letter: str) -> torch.Tensor:
        """Full-length S tensor: native scaffold with the binder positions overwritten by
        ``one_letter`` (binder-chain seed, e.g. a ProteinMPNN sequence). Length must match the
        binder chain; unknown characters fall back to UNK."""
        binder_idx = self.chainA_all_idx.tolist()
        if len(one_letter) != len(binder_idx):
            raise ValueError(
                f"seed length {len(one_letter)} != binder length {len(binder_idx)} "
                f"(chain {self.binder_chain}) — seed must be the binder-chain sequence")
        unk = self.encoding.token_to_idx.get("UNK", self.unknown_indices[0] if self.unknown_indices else 0)
        S = self.S_native.clone()
        for pos, ch in zip(binder_idx, one_letter):
            S[pos] = self._one_to_idx.get(ch, unk)
        return S

    def valid_aa_mask(self, forbidden_tokens: Sequence[str]) -> torch.Tensor:
        mask = torch.ones(self.V, dtype=torch.bool, device=self.device)
        # UNK + any model-declared unknown tokens are invalid (matches foundry get_masks).
        for idx in self.unknown_indices:
            if idx < self.V:
                mask[idx] = False
        unk = self.encoding.token_to_idx.get("UNK")
        if unk is not None and unk < self.V:
            mask[unk] = False
        for tok in forbidden_tokens:
            idx = self.encoding.token_to_idx.get(tok)
            if idx is not None and idx < self.V:
                mask[idx] = False
        return mask

    def neighbour_mask(self, center: int, k_max: int = 0) -> torch.Tensor:
        """Designable neighbourhood of a pinned centre = its coupled kNN (both graph directions).

        ``k_max`` > 0 caps it to the k_max NEAREST coupled neighbours (the kNN prefix, applied to BOTH
        the outgoing and incoming edges). This is the direct control on HOW MANY positions get
        redesigned — i.e. the mutation load, since designable ≈ center_count × neighbourhood size.
        0 (default) = the full kNN neighbourhood, an exact no-op vs the previous behaviour."""
        K = self.K if k_max <= 0 else min(self.K, int(k_max) + 1)
        m = torch.zeros(self.L, dtype=torch.bool, device=self.device)
        for k in range(1, K):                                        # outgoing centre->j
            j = int(self.eidx_np[center, k])
            if j != center and bool(self.free_mask[j]) and bool(self.chainA_t[j]):
                m[j] = True
        for i, _ in np.argwhere(self.eidx_np[:, 1:K] == center):     # incoming i->centre (same cap)
            i = int(i)
            if i != center and bool(self.free_mask[i]) and bool(self.chainA_t[i]):
                m[i] = True
        return m

    def region_masks_at(self, interface_distance: float) -> Dict[str, torch.Tensor]:
        """Region masks for an interface cutoff in angstroms; each distinct cutoff is computed once.

        The default cutoff comes from ``_build_context``; others re-run the geometry on the
        processed atom array the context keeps."""
        distance = round(float(interface_distance), 6)
        if distance == DEFAULT_INTERFACE_DISTANCE:
            return self.region_masks
        cache = self.__dict__.setdefault("_region_cache", {})
        if distance not in cache:
            if self.proc_atom_array is None:
                raise RuntimeError("The context carries no processed atom array.")
            cache[distance] = _binder_region_masks(
                self.proc_atom_array, self.token_aa, self.binder_chain, self.chainA_t,
                self.device, self.L, interface_dist=distance)
        return cache[distance]

    def region_mask(self, names: Sequence[str], interface_distance: Optional[float] = None) -> torch.Tensor:
        """OR of the requested region masks ∩ free binder positions. 'all' (or empty / unknown names)
        -> all free binder positions."""
        free = self.free_mask & self.chainA_t
        if not names or "all" in names:
            return free.clone()
        masks = (self.region_masks if interface_distance is None
                 else self.region_masks_at(interface_distance))
        base = torch.zeros(self.L, dtype=torch.bool, device=self.device)
        for n in names:
            m = masks.get(n)
            if m is not None:
                base = base | m
        region = base & free
        if not bool(region.any()):
            logger.warning(
                "placement_region %s selects no free binder position (interface_distance=%s); "
                "no placement site can be found",
                list(names), interface_distance if interface_distance is not None else DEFAULT_INTERFACE_DISTANCE)
        return region

    def binder_pos_of_res_id(self, res_id: int) -> int:
        pos = self.res_id_to_pos.get(int(res_id))
        if pos is None:
            raise ValueError(
                f"res_id {res_id} is not a free binder position on chain {self.binder_chain}.")
        return pos

    def decode_canonical(self, seq: torch.Tensor, positions: Sequence[int]) -> str:
        """Foldable one-letter over ``positions``: microstate -> canonical parent -> 1-letter."""
        idx_to_token = self.encoding.idx_to_token
        out = []
        for i in positions:
            parent3 = str(idx_to_token[int(self.canonical_map[int(seq[i])])])
            out.append(DICT_THREE_TO_ONE.get(parent3, self._unk_one))
        return "".join(out)

    def extended_tokens(self, seq: torch.Tensor, positions: Sequence[int]) -> List[str]:
        idx_to_token = self.encoding.idx_to_token
        return [str(idx_to_token[int(seq[i])]) for i in positions]


# ---------------------------------------------------------------------------
# Field utilities (energy convention: lower = better)
# ---------------------------------------------------------------------------

def _pick(score: torch.Tensor, temperature: float) -> int:
    if temperature <= 0:
        return int(score.argmin())
    return int(torch.multinomial(F.softmax(-score / temperature, dim=-1), 1))


def _cond_dist(E_LV: torch.Tensor, valid_mask: torch.Tensor, temperature: float) -> torch.Tensor:
    """[L, n_valid] softmax over designable tokens (lower energy -> higher prob)."""
    t = temperature if temperature > 0 else 1e-3
    valid_idx = valid_mask.nonzero().flatten()
    return F.softmax(-E_LV[:, valid_idx] / t, dim=-1)


def _entropy_bits(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp_min(1e-12)
    return -(p * p.log2()).sum(-1)


def _score_at(ctx, field_at, valid_mask, S, i, pins, *, selective, lam=None):
    """Energy of each token at neighbour i with ALL centers pinned at RES-P (lower = better).

    ``field_at(S, i)`` returns position i's [V] conditional energies (the fast per-position scorer).
    selective: subtract the mean deprotonated-centre field, averaged over the pinned centers.
    converged_mcmc_combined (lam set): LEGACY unscaled blend (1+lam)*stab - lam*mean_dep — distinct from the
    z-scored manuscript Eq (6) objective O=(1-λ)*z(H)+λ*z(sel) that block_descent uses. Invalid tokens -> +inf.
    Every centre is restored to RES-P.
    """
    for p in pins:
        S[p.position] = p.prot_idx
    e_stab = field_at(S, i).clone()
    if not selective and lam is None:
        return e_stab.masked_fill(~valid_mask, float("inf"))
    ed = torch.zeros_like(e_stab)
    for p in pins:
        c = torch.zeros_like(e_stab)
        for d in p.dep_idxs:
            S[p.position] = d
            c += field_at(S, i)
        S[p.position] = p.prot_idx
        ed += c / len(p.dep_idxs)
    ed = ed / len(pins)
    sc = (1.0 + lam) * e_stab - lam * ed if lam is not None else e_stab - ed
    return sc.masked_fill(~valid_mask, float("inf"))


def _energy_snapshot(ctx, S, pins) -> Dict[str, Any]:
    """One trajectory point for S: total selective energy (Σ centers RES-P vs RES-D gap), the binder
    global protonation ΔH (metric), the total Potts Hamiltonian H, and the binder SEQUENCE at this step —
    both the one-letter ``canonical_sequence`` and the ``extended_tokens`` (3-letter + protonation state,
    e.g. ``HIS-P``/``ASP-D``), so the sequence can be saved/read off any step. Centers pinned at RES-P.
    Only built when a trajectory is being recorded (see _record)."""
    binder_idx = ctx.chainA_all_idx.tolist()
    return {
        "selective_energy": _selective_energy_sum(ctx, S, pins),
        "global_protonation_dH": _global_protonation_dH(ctx, S),
        "potts_energy": ctx.scorer.H_of(S),
        "canonical_sequence": ctx.decode_canonical(S, binder_idx),
        "extended_tokens": " ".join(ctx.extended_tokens(S, binder_idx)),
    }


def _record(trajectory, ctx, S, pins, *, initial=False) -> None:
    """Append an energy snapshot (0-based step index) if recording is on. ``initial`` only records when
    the trajectory is still empty, so two-phase's phase-2 converge doesn't duplicate the hand-off."""
    if trajectory is None:
        return
    if initial and trajectory:
        return
    snap = {"step": len(trajectory)}
    snap.update(_energy_snapshot(ctx, S, pins))
    trajectory.append(snap)


def _sampler_zscales(ctx, S, order, pins, valid_idx):
    """Freeze the std of single-mutation deltas for the MPNN sampler's two terms over ``order``:
    naturalness (PottsMPNN decoder -log p) and selectivity (Σ centers e_P - e_D). Centers locked."""
    Sloc = S.clone()
    for p in pins:
        Sloc[p.position] = p.prot_idx
    nat_full = ctx.field_mpnn(Sloc)                       # [L, V], one forward
    sel0 = _selective_energy_sum(ctx, Sloc, pins)
    nat_d, sel_d = [], []
    for j in order:
        cur = int(Sloc[j])
        base = float(nat_full[j, cur])
        for a in valid_idx:
            nat_d.append(float(nat_full[j, a]) - base)
            Sloc[j] = a
            sel_d.append(_selective_energy_sum(ctx, Sloc, pins) - sel0)
        Sloc[j] = cur
    sdNat = max(float(np.std(nat_d)) if nat_d else 1.0, 1e-6)
    sdSel = max(float(np.std(sel_d)) if sel_d else 1.0, 1e-6)
    return sdNat, sdSel


def _selective_row(ctx, valid_idx, S, j, pins) -> torch.Tensor:
    """[V] selectivity energy Σ centers (e_P - e_D) for each candidate AA at position j (others fixed,
    centers locked). Invalid AAs stay +inf. S[j] restored on exit."""
    out = torch.full((ctx.V,), float("inf"), device=ctx.device)
    cur = int(S[j])
    for a in valid_idx:
        S[j] = a
        out[a] = _selective_energy_sum(ctx, S, pins)
    S[j] = cur
    return out


def _selective_reward_decoder(ctx, valid_idx, S, j, pins, lam) -> torch.Tensor:
    """[V] PURE-DECODER selectivity reward at position j (higher = better), for the autoregressive
    designer when crit.selective_source=='decoder':
        R(a) = p(a | ALL centers = TARGET) − λ · p(a | ALL centers = OFF)
    Both p_* are the decoder's OWN distributions — exp(−field_mpnn) = softmax(log_probs) over the full
    vocab — so the two terms live on one simplex and subtract directly (no Potts, no z-scaling). λ=0
    ⇒ R = p_target, the plain target-state decode. The OFF decode pins every center at its dep token(s)
    (all centers flipped TOGETHER, matching the physical switch), averaged over tautomers (e.g. HID/HIE
    for HIS-P). Centers are assumed pre-pinned at prot_idx in S on entry and are restored on exit;
    invalid AAs stay −inf."""
    p_target = torch.exp(-ctx.field_mpnn(S)[j])                    # [V] distribution, centers = target
    n_off = max(len(p.dep_idxs) for p in pins)                    # off-token/tautomer count (2 for HIS-P)
    p_off = torch.zeros(ctx.V, device=ctx.device)
    saved = [int(S[p.position]) for p in pins]
    for t in range(n_off):
        for p in pins:
            S[p.position] = p.dep_idxs[min(t, len(p.dep_idxs) - 1)]
        p_off += torch.exp(-ctx.field_mpnn(S)[j])                 # [V] distribution, centers = off
    for p, tok in zip(pins, saved):                               # restore centers to target
        S[p.position] = tok
    reward = p_target - lam * (p_off / n_off)
    out = torch.full((ctx.V,), float("-inf"), device=ctx.device)
    vi = torch.as_tensor(valid_idx, device=ctx.device, dtype=torch.long)
    out[vi] = reward[vi]
    return out


def _masked_infill(ctx, crit, valid_mask, pins, order, initial_sequence, trajectory=None):
    """Autoregressive MPNN infill (the 'autoregressive'/MPNN method), generalized to a set of centers:
    lock ALL centers, mask the union of their neighbourhoods (``order``) to UNK, then decode positions
    one-by-one, SAMPLING each from softmax(-J/T). Two selectivity sources (crit.selective_source):
        "potts"   — J(a) = (1-λ)·z(-log p_MPNN(a)) + λ·z(Σ centers (e_P - e_D)(a));
                    naturalness = the PottsMPNN decoder (protonation-aware), selectivity = Potts gap.
        "decoder" — J(a) = -[ p(a|centers=target) - λ·p(a|centers=off) ]; a pure two-state decoder
                    probability contrast (see _selective_reward_decoder), no Potts, no z-scaling."""
    lam = float(crit.combined_lambda)
    valid_idx = valid_mask.nonzero().flatten().tolist()
    S = initial_sequence.clone()
    for p in pins:
        S[p.position] = p.prot_idx
    decoder_sel = (crit.selective_source == "decoder")
    if not decoder_sel:
        sdNat, sdSel = _sampler_zscales(ctx, S, order, pins, valid_idx)
    unk = ctx.encoding.token_to_idx["UNK"]
    for i in order:
        S[i] = unk
    _record(trajectory, ctx, S, pins, initial=True)
    T = max(float(crit.temperature), 1e-3)
    for i in order:
        if decoder_sel:
            Jv = -_selective_reward_decoder(ctx, valid_idx, S, i, pins, lam)   # R(a)=p_target−λ·p_off
        else:
            nat = ctx.field_mpnn(S)[i]                        # [V] -log p, lower = more natural
            sel = _selective_row(ctx, valid_idx, S, i, pins)
            Jv = (1.0 - lam) * (nat / sdNat) + lam * (sel / sdSel)
        S[i] = _pick(Jv.masked_fill(~valid_mask, float("inf")), T)
        _record(trajectory, ctx, S, pins)
    return S


def _converge(ctx, crit, field_at, valid_mask, pins, neigh, initial_sequence, S0=None, trajectory=None):
    """Stochastic single-site MCMC over random neighbours until the sequence settles.
    Starts from ``S0`` if given (two-phase's frozen phase-1 sequence), else ``initial_sequence``."""
    lam = crit.combined_lambda if crit.method == "converged_mcmc_combined" else None
    selective = crit.selective if lam is None else False
    S = (S0.clone() if S0 is not None else initial_sequence.clone())
    for p in pins:
        S[p.position] = p.prot_idx
    _record(trajectory, ctx, S, pins, initial=True)
    patience, cap = crit.cv_patience * len(neigh), crit.cv_max * len(neigh)
    since = step = 0
    while neigh and since < patience and step < cap:
        i = neigh[int(torch.randint(len(neigh), (1,)))]
        tok = _pick(_score_at(ctx, field_at, valid_mask, S, i, pins, selective=selective, lam=lam),
                    crit.temperature)
        since = 0 if tok != int(S[i]) else since + 1
        S[i] = tok
        step += 1
        _record(trajectory, ctx, S, pins)
    return S


def _two_phase(ctx, crit, field_at, valid_mask, pins, neigh, initial_sequence, trajectory=None):
    """Phase 1: commit+freeze the least-disruptive selective picks; phase 2: stability MCMC.
    Starts from ``initial_sequence`` (native/RFD3 or an MPNN seed)."""
    S = initial_sequence.clone()
    for p in pins:
        S[p.position] = p.prot_idx
    _record(trajectory, ctx, S, pins, initial=True)
    sel_pick, disruption = {}, {}
    for i in neigh:
        a = _pick(_score_at(ctx, field_at, valid_mask, S, i, pins, selective=True), crit.temperature)
        e_stab = field_at(S, i)
        sel_pick[i] = a
        disruption[i] = float(e_stab[a] - e_stab[int(S[i])])
    order = sorted(neigh, key=lambda i: disruption[i])
    k = max(1, min(len(neigh) - 1, int(round(crit.two_phase_frac * len(neigh)))))
    for i in order[:k]:
        S[i] = sel_pick[i]
        _record(trajectory, ctx, S, pins)
    rest = [i for i in neigh if i not in set(order[:k])]
    # phase 2 is non-selective stability cleanup on the rest.
    crit2 = copy.copy(crit)
    crit2.method = "converged_mcmc"
    crit2.selective = False
    return _converge(ctx, crit2, field_at, valid_mask, pins, rest, initial_sequence, S0=S,
                     trajectory=trajectory)


# ---------------------------------------------------------------------------
# block_descent: deterministic / sampled block coordinate descent on the
# z-scaled combined objective  J = (1-λ)·z(ΔH) + λ·z(Δsel) + global_weight·z(Δglob)
# ---------------------------------------------------------------------------

def _zscale(deltas: torch.Tensor, valid_mask: torch.Tensor) -> float:
    """Std of the finite, valid single-mutation deltas (the z-scale; mean is irrelevant — a constant
    shift cancels in both argmin and softmax, so only 1/std matters for the pick). Floored away from 0.
    deltas: [N, V] (rows = design positions, cols = candidate AAs). Invalid AAs are +inf."""
    vals = deltas[:, valid_mask]
    vals = vals[torch.isfinite(vals)]
    if vals.numel() == 0:
        return 1.0
    return max(float(vals.std(unbiased=False)), 1e-6)


def _block_mutation_tables(ctx, valid_mask, S, neigh, pins, want_global):
    """Single-mutation delta tables over the design positions, used to FREEZE the z-scale.

    Returns dH, dSel, dGlob each [N, V] (N=len(neigh)); entry [n,a] = change in that term when
    neigh[n] is set to AA a (rest of S fixed, ALL centers pinned at RES-P). dGlob zeros if not wanted.
    """
    N, V = len(neigh), ctx.V
    dH = torch.full((N, V), float("inf"), device=ctx.device)
    dSel = torch.zeros((N, V), device=ctx.device)
    dGlob = torch.zeros((N, V), device=ctx.device)
    valid_idx = valid_mask.nonzero().flatten().tolist()
    S = S.clone()
    for p in pins:
        S[p.position] = p.prot_idx
    sel0 = _selective_energy_sum(ctx, S, pins)
    glob0 = _global_protonation_dH(ctx, S) if want_global else None
    for n, i in enumerate(neigh):
        cur = int(S[i])
        e_i = ctx.scorer.cond_energy_at(S, i)                      # [V] absolute stability row
        row = (e_i - e_i[cur])
        for a in valid_idx:
            dH[n, a] = row[a]
        # selective / global only vary for the right positions; cheap per-candidate eval
        for a in valid_idx:
            if a == cur:
                continue
            S[i] = a
            dSel[n, a] = _selective_energy_sum(ctx, S, pins) - sel0
            if want_global:
                g = _global_protonation_dH(ctx, S)
                dGlob[n, a] = (g - glob0) if g is not None else 0.0
        S[i] = cur
    return dH, dSel, dGlob


def _block_partners(ctx, p, neigh_set, block_size):
    """`p` plus its (block_size-1) closest coupled partners drawn from the design set, in the
    model's own kNN order (E_idx) — i.e. the graph-nearest neighbours that are also design positions."""
    block = [p]
    if block_size <= 1:
        return block
    for j in ctx.scorer.nbr_idx[p].tolist():               # E_idx order = closest first
        if j != p and j in neigh_set and j not in block:
            block.append(j)
            if len(block) == block_size:
                break
    return block


def _block_zscales(ctx, crit, valid_mask, S, neigh, pins, want_global, stab_scorer=None):
    """Freeze the z-scales from the BLOCK (V**block_size) joint distribution instead of single
    mutations (crit.zscale_mode == 'block'). For every design position's block, build the stability
    H_block over all V**B assignments via block_stability_potentials + the SAME broadcast the sweep
    uses (minus weights) and pool its WITHIN-block variance -> sdH (this is what captures within-block
    pairwise variance). Selective/global are exactly unary, so their block tensor is the broadcast-sum
    of the per-position deltas (dSel0/dGlob0 from _block_mutation_tables), pooled over CENTRE-COUPLED
    blocks -> sdSel/sdGlob. Reuses the decomposition already in the sweep; ~tens of ms at N~120.

    ``stab_scorer`` supplies the (possibly self/pair-reweighted) scorer for the STABILITY tensor so the
    frozen sdH matches the reweighted H_stab the descent optimizes; defaults to the unweighted
    ctx.scorer. Selective/global stay on ctx.scorer (via dSel0/dGlob0)."""
    V = ctx.V
    stab_scorer = stab_scorer if stab_scorer is not None else ctx.scorer
    bsize = max(1, int(crit.block_size))
    neigh_set = set(int(x) for x in neigh)
    _dH0, dSel0, dGlob0 = _block_mutation_tables(ctx, valid_mask, S, neigh, pins, want_global)
    row_of = {int(p): n for n, p in enumerate(neigh)}
    inf_mask = torch.where(~valid_mask, torch.tensor(float("inf"), device=ctx.device),
                           torch.zeros(V, device=ctx.device))

    def _unary_block_var(vecs):
        """Variance of Σ_bi vecs[bi] over valid V**B assignments (invalid AAs -> +inf, dropped)."""
        B = len(vecs)
        J = vecs[0].clone().reshape([V] + [1] * (B - 1))
        for bi in range(1, B):
            shape = [1] * B; shape[bi] = V
            J = J + vecs[bi].reshape(shape)
        for bi in range(B):
            shape = [1] * B; shape[bi] = V
            J = J + inf_mask.reshape(shape)
        v = J[torch.isfinite(J)]
        return float(v.var(unbiased=False)) if v.numel() > 1 else None

    stab_v, sel_v, glob_v = [], [], []
    for p in neigh:
        block = _block_partners(ctx, p, neigh_set, bsize)
        B = len(block)
        # stability: unary su + within-block pairwise M (identical broadcast to _block_descent, no weights)
        su, sedges = stab_scorer.block_stability_potentials(S, block)
        J = su[0].clone().reshape([V] + [1] * (B - 1))
        for bi in range(1, B):
            shape = [1] * B; shape[bi] = V
            J = J + su[bi].reshape(shape)
        for (bi, bj, M) in sedges:
            shape = [1] * B; shape[bi] = V; shape[bj] = V
            J = J + M.reshape(shape)
        for bi in range(B):
            shape = [1] * B; shape[bi] = V
            J = J + inf_mask.reshape(shape)
        vals = J[torch.isfinite(J)]
        if vals.numel() > 1:
            stab_v.append(float(vals.var(unbiased=False)))
        # selective (additive/unary): only blocks that actually vary selectivity
        srows = [dSel0[row_of[int(b)]] for b in block]
        if any(float(r.abs().sum()) > 0 for r in srows):
            var = _unary_block_var(srows)
            if var is not None:
                sel_v.append(var)
        if want_global:
            grows = [dGlob0[row_of[int(b)]] for b in block]
            if any(float(r.abs().sum()) > 0 for r in grows):
                var = _unary_block_var(grows)
                if var is not None:
                    glob_v.append(var)

    def _pool(vs):                       # sqrt of the mean within-block variance, floored like _zscale
        return max((sum(vs) / len(vs)) ** 0.5, 1e-6) if vs else 1.0
    return _pool(stab_v), _pool(sel_v), (_pool(glob_v) if want_global else 1.0)


def _parent_vocab_mask(ctx, parents) -> torch.Tensor:
    """[V] bool over the vocab: True where a token's canonical parent 3-letter code is in ``parents``
    (e.g. {"ASP","GLU"} acids, {"ARG","LYS"} basics, {"HIS"} histidine). Every His microstate token
    (HIS/HIS-P/HIS-S/HIS-A) shares the parent "HIS", so {"HIS"} catches them all. Powers the windowed
    density penalties in _block_descent."""
    idx_to_token = ctx.encoding.idx_to_token
    cm = ctx.canonical_map
    pset = set(parents)
    m = torch.zeros(ctx.V, dtype=torch.bool, device=ctx.device)
    for a in range(ctx.V):
        if str(idx_to_token[int(cm[a])]) in pset:
            m[a] = True
    return m


def _centre_energy_rank(ctx, pins, S, neigh):
    """|contribution to the centres' SELECTIVE gap| per designable position, descending.

    For designable j and pinned centre c, j enters c's conditional energy through exactly two additive
    pair terms (c→j outgoing and j→c incoming), so its contribution to c's gap e(prot) - mean_d e(dep)
    is exact and cheap — no re-scoring of the whole sequence:

        contrib(j, c) = [pe_c[j, prot, S_j] + etab_inc(j→c)[S_j, prot]]
                      - mean_d [pe_c[j, d, S_j] + etab_inc(j→c)[S_j, d]]

    Summed over centres in absolute value: a position that strongly favours the protonated state and one
    that strongly opposes it are both IMPORTANT to get right, which is what the visit order is about.
    Evaluated ONCE on the seed sequence, so the order is fixed for the run (a dynamic re-rank after every
    commit would be a different method — and far more expensive).
    """
    sc = ctx.scorer
    score = {int(j): 0.0 for j in neigh}
    for pin in pins:
        c, prot, deps = int(pin.position), int(pin.prot_idx), [int(d) for d in pin.dep_idxs]
        out_nbrs = sc.nbr_idx[c].tolist()                       # c -> j
        slot_of = {int(j): t for t, j in enumerate(out_nbrs)}
        inc_src, inc_slot = sc._inc_src[c], sc._inc_slot[c]     # m -> c
        inc_of = {int(inc_src[e]): int(inc_slot[e]) for e in range(inc_src.numel())}
        for j in score:
            sj = int(S[j])
            e_p = e_d = 0.0
            if j in slot_of:                                     # c -> j term, indexed [a_c, a_j]
                t = slot_of[j]
                e_p += float(sc.pe[c][t][prot, sj])
                e_d += sum(float(sc.pe[c][t][d, sj]) for d in deps) / max(len(deps), 1)
            if j in inc_of:                                      # j -> c term, indexed [a_j, a_c]
                row = sc.etab[j, inc_of[j], sj, :]
                e_p += float(row[prot])
                e_d += sum(float(row[d]) for d in deps) / max(len(deps), 1)
            score[j] += abs(e_p - e_d)
    # descending importance; position index breaks ties so the order is deterministic
    return sorted(score, key=lambda j: (-score[j], j))


def _knn_rank(ctx, pins, neigh):
    """Designable positions ordered by their BEST (smallest) rank in any centre's coupled kNN list —
    i.e. closest-coupled first. Ties broken by position index for determinism."""
    best = {int(j): 10**6 for j in neigh}
    for pin in pins:
        c = int(pin.position)
        for r, j in enumerate(ctx.scorer.nbr_idx[c].tolist()):   # E_idx order = closest first
            j = int(j)
            if j in best:
                best[j] = min(best[j], r)
    return sorted(best, key=lambda j: (best[j], j))


def _block_descent(ctx, crit, valid_mask, pins, neigh, initial_sequence, rng, trajectory=None):
    """Deterministic (or sampled) block coordinate descent on the z-scaled combined objective, with a
    SET of centers pinned to RES-P (fixed context).

    Each sweep, for every design position p, form a block = p + its (block_size-1) closest coupled
    partners, build the EXACT combined score over all V**block_size joint assignments, and commit
    the argmin (temperature==0) or a Boltzmann sample (temperature>0). Iterate until a full sweep
    makes no change or block_max_rounds is reached. z-scale (std of single-mutation deltas) is
    computed ONCE and frozen so J = (1-λ)·z(H_stab) + λ·z(Σ centers (e_P - e_D)) [+ gw·z(global)] is
    a fixed objective (global term off by default; global_weight=0 -> dH is a report-only metric).
    """
    V = ctx.V
    lam = float(crit.combined_lambda)
    gw = float(crit.global_weight)
    want_global = gw != 0.0
    arw = float(crit.adjacent_repeat_weight)
    want_repeat = arw != 0.0
    # windowed repetitive-density penalty — THE ONE windowed density penalty. Disperses a configurable
    # residue class (repetitive_window_parents); acids are just parents ["ASP","GLU"]. Gated to a centre
    # type only if repetitive_window_gate_types is set; an EMPTY gate = active whenever weight>0 and a
    # parent set is given. See PHDesignCriteria.repetitive_window_*.
    rw = float(crit.repetitive_window_weight)
    rw_parents = tuple(crit.repetitive_window_parents)
    _rw_gate = set(crit.repetitive_window_gate_types)
    want_rep = rw != 0.0 and bool(rw_parents) and (
        not _rw_gate or any(p.protonation_type in _rw_gate for p in pins))
    rrad = max(1, int(crit.repetitive_window_radius))
    # self/pair reweight for the STABILITY term only (H_stab = w_self·Σh + w_pair·ΣJ). Default (1,1) →
    # ctx.scorer unchanged. Used for both the frozen sdH and the per-round stability potentials so they
    # stay consistent; selective/global/readouts remain on ctx.scorer.
    w_self, w_pair = crit.stability_weights()
    stab_scorer = ctx.stability_scorer(w_self, w_pair)
    bsize = max(1, int(crit.block_size))
    if bsize > 4:
        print(f"[block_descent] block_size={bsize} → V**{bsize} enumeration may be large/slow.")

    S = initial_sequence.clone()
    for pin in pins:
        S[pin.position] = pin.prot_idx
    if not neigh:
        return S

    # VISIT ORDER of the designable positions. Block descent is greedy and order-dependent, so this
    # changes the result even with an identical objective. Computed once, on the seed sequence.
    if crit.sweep_order == "energy":
        neigh = _centre_energy_rank(ctx, pins, S, neigh)
    elif crit.sweep_order == "knn":
        neigh = _knn_rank(ctx, pins, neigh)
    # "position" keeps the incoming order (ascending residue index, set by _finalize_plan)

    # --- freeze the z-scale: from the BLOCK joint spread (default) or single-mutation deltas ---
    if crit.zscale_mode == "block":
        sdH, sdSel, sdGlob = _block_zscales(ctx, crit, valid_mask, S, neigh, pins, want_global, stab_scorer)
    else:
        dH0, dSel0, dGlob0 = _block_mutation_tables(
            ctx, valid_mask, S, neigh, pins, want_global)
        sdH = _zscale(dH0, valid_mask)
        # selective/global std measured only where they actually vary (non-zero rows)
        sel_rows = dSel0.abs().sum(1) > 0
        sdSel = _zscale(dSel0[sel_rows] if sel_rows.any() else dSel0, valid_mask)
        glob_rows = dGlob0.abs().sum(1) > 0
        sdGlob = (_zscale(dGlob0[glob_rows], valid_mask) if (want_global and glob_rows.any()) else 1.0)
    wH, wSel, wGlob = (1.0 - lam) / sdH, lam / sdSel, (gw / sdGlob if want_global else 0.0)

    # adjacent-repeat bias: [V,V] "same canonical amino acid" mask (ASP-P==ASP-D, …); +arw is added per
    # sequence-adjacent (i,i+1) pair that lands on the same AA (built once, applied per block below).
    same_canon = (((ctx.canonical_map.view(-1, 1) == ctx.canonical_map.view(1, -1)).float()).to(ctx.device)
                  if want_repeat else None)
    rep_mask = _parent_vocab_mask(ctx, rw_parents).float() if want_rep else None  # [V] configured class
    res_id = ctx.token_aa.res_id

    invalid = ~valid_mask                                          # [V] forbidden/invalid AAs
    neigh_set = set(int(x) for x in neigh)

    def combined_unary(p, block_set):
        """z-scaled combined unary [V] for design position p: (1-λ)z(H)+λz(sel)+gw·z(glob).
        Selective/global pieces are exact single-position deltas (they are unary). Stability's
        unary part is folded in by the caller via block_stability_potentials; here we add only the
        sel+glob contribution (stability handled separately so its pairwise stays exact)."""
        cur = int(S[p])
        sel0 = _selective_energy_sum(ctx, S, pins)
        glob0 = _global_protonation_dH(ctx, S) if want_global else None
        u = torch.zeros(V, device=ctx.device)
        for a in range(V):
            if invalid[a] or a == cur:
                continue
            S[p] = a
            u[a] += wSel * (_selective_energy_sum(ctx, S, pins) - sel0)
            if want_global:
                g = _global_protonation_dH(ctx, S)
                if g is not None:
                    u[a] += wGlob * (g - glob0)
        S[p] = cur
        # windowed repetitive-density penalty (unary): +rw × (#class residues within ±rrad SEQUENCE
        # positions of p) read off the WORKING SEQUENCE S. Everything outside the current block counts —
        # frozen target/centre residues AND designable positions already assigned this sweep (or on a
        # previous round). Only the current block is excluded, because those pairs are charged exactly
        # by the within-block pairwise term below; counting them here too would double-charge.
        #
        # Previously this skipped ALL of neigh_set, i.e. every designable position whether or not it had
        # been assigned. That made designable-designable crowding visible ONLY when two positions
        # happened to share a block, so the penalty's effective strength varied with block_size and was
        # inert between designed residues at block_size=1 (class adjacency ~2x higher there).
        if want_rep:
            ri = int(res_id[p])
            n_ctx = 0
            for dd in range(-rrad, rrad + 1):
                if dd == 0:
                    continue
                q = ctx.res_id_to_pos.get(ri + dd)
                if q is not None and int(q) not in block_set and bool(rep_mask[int(S[int(q)])]):
                    n_ctx += 1
            if n_ctx:
                u = u + rw * n_ctx * rep_mask
        return u

    changed_any = True
    rounds = 0
    _record(trajectory, ctx, S, pins, initial=True)
    while changed_any and rounds < int(crit.block_max_rounds):
        changed_any = False
        rounds += 1
        for p in neigh:
            block = _block_partners(ctx, p, neigh_set, bsize)
            B = len(block)
            # stability: exact unary + within-block pairwise, z-scaled by wH (self/pair reweighted)
            su, sedges = stab_scorer.block_stability_potentials(S, block)    # su:[B,V]
            # combined unary per block position = wH·(stab unary) + (sel+glob z-scaled). Absolute
            # stability differs from z(ΔH) only by a per-position constant → irrelevant to argmin/softmax.
            cu = wH * su
            block_set = set(int(x) for x in block)
            for bi, p_b in enumerate(block):
                cu[bi] = cu[bi] + combined_unary(p_b, block_set)
            # build J over all V**B assignments by broadcasting
            J = cu[0].clone().reshape([V] + [1] * (B - 1))
            for bi in range(1, B):
                shape = [1] * B
                shape[bi] = V
                J = J + cu[bi].reshape(shape)
            for (bi, bj, M) in sedges:
                shape = [1] * B
                shape[bi] = V
                shape[bj] = V
                J = J + (wH * M).reshape(shape)
            # adjacent-repeat bias: +arw per sequence-adjacent (i,i+1) pair on the SAME canonical AA —
            # (a) within-block adjacent pair -> pairwise [V,V]; (b) i±1 neighbour in FIXED context -> unary.
            if want_repeat:
                bres = [int(res_id[p_b]) for p_b in block]
                bset = set(int(p_b) for p_b in block)
                for bi in range(B):
                    for bj in range(bi + 1, B):
                        if abs(bres[bi] - bres[bj]) == 1:              # (a) both i,i+1 vary in this block
                            shape = [1] * B; shape[bi] = V; shape[bj] = V
                            J = J + (arw * same_canon).reshape(shape)
                    for d in (-1, 1):
                        q = ctx.res_id_to_pos.get(bres[bi] + d)
                        if q is not None and int(q) not in bset:      # (b) i±1 neighbour is fixed (no dbl-count)
                            shape = [1] * B; shape[bi] = V
                            J = J + (arw * same_canon[:, int(S[int(q)])]).reshape(shape)
            # within-block repetitive pairwise: +rw per pair of block positions ≤rrad res-ids apart that
            # are BOTH in the configured class. Class residues OUTSIDE the block are handled by the
            # working-sequence unary in combined_unary, so nothing is double-counted and the total charge
            # is the same however the positions happen to be partitioned into blocks.
            if want_rep:
                bres_r = [int(res_id[p_b]) for p_b in block]
                for bi in range(B):
                    for bj in range(bi + 1, B):
                        if abs(bres_r[bi] - bres_r[bj]) <= rrad:
                            shape = [1] * B; shape[bi] = V; shape[bj] = V
                            J = J + (rw * torch.outer(rep_mask, rep_mask)).reshape(shape)
            # forbid invalid AAs on every block axis
            for bi in range(B):
                shape = [1] * B
                shape[bi] = V
                J = J + torch.where(invalid, torch.tensor(float("inf"), device=ctx.device),
                                    torch.zeros(V, device=ctx.device)).reshape(shape)
            # pick the joint assignment: argmin (T=0) or Boltzmann sample (T>0)
            flat = J.reshape(-1)
            T = float(crit.temperature)
            if T <= 0:
                choice = int(torch.argmin(flat))
            else:
                probs = F.softmax(-(flat - flat.min()) / T, dim=-1)
                choice = int(torch.multinomial(probs, 1))
            assign = list(np.unravel_index(choice, [V] * B))
            moved = False
            for bi, p_b in enumerate(block):
                a = int(assign[bi])
                if a != int(S[p_b]):
                    S[p_b] = a
                    moved = True
            if moved:
                changed_any = True
                _record(trajectory, ctx, S, pins)
    return S


def _greedy_energy_block(ctx, crit, valid_mask, pins, neigh, initial_sequence, rng, trajectory=None):
    """Centre-free, dynamically-reranked block coordinate descent on the pure Potts STABILITY energy H
    (no pins, no selective/λ term). Each step:
      (1) score every designable position's best single-residue improvement
          δ_i = min_a valid e_i[a] − e_i[S_i]  on the CURRENT sequence (recomputed every step);
      (2) take the ``block_size`` positions with the largest improvement;
      (3) update those jointly by exact V**block_size enumeration — argmin (temperature==0) or a
          Boltzmann sample (temperature>0) of the (1/sdH)-scaled stability energy + the optional
          repetitive_window penalty;
      (4) stop when NO position improves (or the step cap).
    Returns the LOWEST-H sequence visited (robust to temperature>0 wandering).

    This is the "same algorithm as block descent, but pick the positions that most improve the energy,
    recomputed at every step" variant: no titratable centre is ever placed (pins == []); forbidden_tokens
    keeps the design plain 20-AA. Reuses _block_descent's block machinery (block_stability_potentials +
    the V**B broadcast) and the repetitive_window unary/pairwise.
    """
    V = ctx.V
    K = max(1, int(crit.block_size))
    T = float(crit.temperature)
    tol = 1e-4
    w_self, w_pair = crit.stability_weights()
    stab_scorer = ctx.stability_scorer(w_self, w_pair)

    # windowed repetitive-density penalty. Two modes:
    #   parents == ["ALL"]  -> ANTI-REPETITION over EVERY amino acid: penalise a candidate AA by how many
    #                          nearby residues share its canonical identity (fights low-complexity / poly-X
    #                          collapse of raw Potts minimisation). Uses the [V,V] same-canonical mask.
    #   parents == [<class>] -> the class-dispersal penalty (e.g. ["ASP","GLU"] for acids), as in block_descent.
    rw = float(crit.repetitive_window_weight)
    rw_parents = tuple(crit.repetitive_window_parents)
    _rw_gate = set(crit.repetitive_window_gate_types)
    rep_all = "ALL" in rw_parents
    want_rep = rw != 0.0 and (rep_all or (bool(rw_parents) and (
        not _rw_gate or any(p.protonation_type in _rw_gate for p in pins))))
    rrad = max(1, int(crit.repetitive_window_radius))
    _cmap = ctx.canonical_map
    same_canon = ((_cmap.view(-1, 1) == _cmap.view(1, -1)).float().to(ctx.device)
                  if (want_rep and rep_all) else None)          # [V,V] 1 where two tokens share a parent AA
    rep_mask = _parent_vocab_mask(ctx, rw_parents).float() if (want_rep and not rep_all) else None
    res_id = ctx.token_aa.res_id

    invalid = ~valid_mask
    inf_vec = torch.where(invalid, torch.tensor(float("inf"), device=ctx.device),
                          torch.zeros(V, device=ctx.device))

    S = initial_sequence.clone()
    if not neigh:
        return S

    # project the designable region onto BARE canonical tokens: drop any protonation microstate the
    # backbone parse carried in (e.g. an acid encoded as ASP-D), so a forbidden-but-preexisting microstate
    # can't PERSIST unchanged (greedy only moves positions that improve). forbidden_tokens then keeps the
    # search on plain 20-AA. The folded sequence is unchanged (both ASP and ASP-D decode to "D").
    cmap = ctx.canonical_map
    for i in neigh:
        ci = int(cmap[int(S[int(i)])])
        if bool(valid_mask[ci]):
            S[int(i)] = ci

    # freeze the stability z-scale (so `temperature` is in std units, exactly as in _block_descent)
    dH0, _, _ = _block_mutation_tables(ctx, valid_mask, S, neigh, [], False)
    wH = 1.0 / _zscale(dH0, valid_mask)

    best_S = S.clone()
    best_H = stab_scorer.H_of(S)
    _record(trajectory, ctx, S, pins, initial=True)
    patience, cap = crit.cv_patience * len(neigh), crit.cv_max * len(neigh)
    since = step = 0
    while since < patience and step < cap:
        # (1) best single-residue improvement per designable position on the CURRENT S
        improving = []
        for i in neigh:
            e_i = stab_scorer.cond_energy_at(S, int(i))                # [V] absolute conditional energy
            cur = int(S[int(i)])
            e_valid = e_i.clone()
            e_valid[invalid] = float("inf")
            delta = float(e_valid.min()) - float(e_i[cur])
            if delta < -tol:
                improving.append((delta, int(i)))
        # (2) converge when nothing improves
        if not improving:
            break
        improving.sort(key=lambda t: (t[0], t[1]))                     # most-improving first (deterministic ties)
        block = [i for _, i in improving[:K]]
        B = len(block)

        # (3) exact V**B joint move over the block: (1/sdH)-scaled stability + repetitive_window
        su, sedges = stab_scorer.block_stability_potentials(S, block)  # su:[B,V]
        cu = wH * su
        if want_rep:
            block_set = set(int(x) for x in block)
            for bi, p_b in enumerate(block):
                ri = int(res_id[p_b])
                for dd in range(-rrad, rrad + 1):
                    if dd == 0:
                        continue
                    q = ctx.res_id_to_pos.get(ri + dd)
                    if q is None or int(q) in block_set:
                        continue
                    if rep_all:                                   # +rw for every candidate AA == this neighbour's AA
                        cu[bi] = cu[bi] + rw * same_canon[:, int(S[int(q)])]
                    elif bool(rep_mask[int(S[int(q)])]):          # class dispersal (acids): +rw on the class
                        cu[bi] = cu[bi] + rw * rep_mask
        J = cu[0].clone().reshape([V] + [1] * (B - 1))
        for bi in range(1, B):
            shape = [1] * B; shape[bi] = V
            J = J + cu[bi].reshape(shape)
        for (bi, bj, M) in sedges:
            shape = [1] * B; shape[bi] = V; shape[bj] = V
            J = J + (wH * M).reshape(shape)
        if want_rep:                                                   # within-block repetitive pairwise
            bres = [int(res_id[p_b]) for p_b in block]
            pair_pen = same_canon if rep_all else (
                torch.outer(rep_mask, rep_mask) if rep_mask is not None else None)
            if pair_pen is not None:
                for bi in range(B):
                    for bj in range(bi + 1, B):
                        if abs(bres[bi] - bres[bj]) <= rrad:
                            shape = [1] * B; shape[bi] = V; shape[bj] = V
                            J = J + (rw * pair_pen).reshape(shape)
        for bi in range(B):                                            # forbid invalid AAs on every axis
            shape = [1] * B; shape[bi] = V
            J = J + inf_vec.reshape(shape)
        flat = J.reshape(-1)
        if T <= 0:
            choice = int(torch.argmin(flat))
        else:
            probs = F.softmax(-(flat - flat.min()) / T, dim=-1)
            choice = int(torch.multinomial(probs, 1))
        assign = list(np.unravel_index(choice, [V] * B))
        for bi, p_b in enumerate(block):
            S[p_b] = int(assign[bi])
        step += 1

        # (4) track the lowest-H sequence (temperature>0 can wander; return the best seen)
        H = stab_scorer.H_of(S)
        if H < best_H - 1e-9:
            best_H, best_S, since = H, S.clone(), 0
        else:
            since += 1
        _record(trajectory, ctx, S, pins)
    return best_S


# ---------------------------------------------------------------------------
# Outcome metrics (use the Potts energy head via ctx.scorer)
# ---------------------------------------------------------------------------

def _selective_energy_of(ctx, seq, center, prot_idx, dep_idxs) -> float:
    s = seq.clone()
    s[center] = prot_idx
    e_p = float(ctx.scorer.cond_energy_at(s, center)[prot_idx])   # only the centre row is needed
    ed = 0.0
    for d in dep_idxs:
        s[center] = d
        ed += float(ctx.scorer.cond_energy_at(s, center)[d])
    return e_p - ed / len(dep_idxs)


def _selective_energy_sum(ctx, seq, pins) -> float:
    """Σ over pinned centers of (e_P - mean_d e_D), all centers locked protonated (each center's gap
    is measured in the presence of the others). Equals _selective_energy_of for a single pin."""
    s = seq.clone()
    for p in pins:
        s[p.position] = p.prot_idx
    return float(sum(_selective_energy_of(ctx, s, p.position, p.prot_idx, p.dep_idxs) for p in pins))


# ---------------------------------------------------------------------------
# Placement helpers: candidate finders (used by PlacementPlan.placement_fn), combination scoring,
# and plan finalize. Signatures kept uniform: fn(ctx, crit, region_idx, prot_idx, dep_idxs, seq, rng).
# ---------------------------------------------------------------------------
def _ranked_candidates(ctx, crit, field_fn, prot_idx, dep_idxs, initial_sequence, region_idx, mask_seq):
    """(position, score) over region_idx, ascending (lower = better). Generalizes _placement_sites:
    non-selective = ΔE(RES-P) vs the current residue; selective = that minus the best deprotonated
    alternative. field_fn = Potts conditional energy or the PottsMPNN decoder -log p. ``mask_seq``
    (MPNN only) masks the binder design positions to UNK first -> structure-only placement; the Potts
    head scans against the real sequence (masking is not meaningful for its energy)."""
    region_idx = region_idx if torch.is_tensor(region_idx) else torch.as_tensor(
        list(region_idx), device=ctx.device)
    placement_seq = initial_sequence
    if mask_seq:
        placement_seq = initial_sequence.clone()
        placement_seq[ctx.chA_free_idx] = ctx.encoding.token_to_idx["UNK"]
    ef = field_fn(placement_seq)                              # [L, V]
    base = ef[region_idx, placement_seq[region_idx]]
    score = ef[region_idx, prot_idx] - base
    if crit.selective:
        dep_dE = torch.stack([ef[region_idx, d] - base for d in dep_idxs], 0).min(0).values
        score = score - dep_dE
    order = torch.argsort(score).tolist()
    return [(int(region_idx[k]), float(score[k])) for k in order]


def _place_scan_potts(ctx, crit, region_idx, prot_idx, dep_idxs, initial_sequence, rng):
    # Potts energy head: scan against the REAL sequence (never mask).
    return _ranked_candidates(ctx, crit, ctx.field_potts, prot_idx, dep_idxs, initial_sequence,
                              region_idx, mask_seq=False)


def _place_scan_mpnn(ctx, crit, region_idx, prot_idx, dep_idxs, initial_sequence, rng):
    # PottsMPNN decoder: mask the binder design positions -> structure-only log-lik(prot) vs (deprot).
    return _ranked_candidates(ctx, crit, ctx.field_mpnn, prot_idx, dep_idxs, initial_sequence,
                              region_idx, mask_seq=True)


def _place_random(ctx, crit, region_idx, prot_idx, dep_idxs, initial_sequence, rng):
    idx = region_idx.tolist() if torch.is_tensor(region_idx) else list(region_idx)
    rng.shuffle(idx)
    return [(int(p), 0.0) for p in idx]


def _locked_selective_score(ctx, field_fn, seq, pins) -> float:
    """Locked joint Σ centers (e_P - min_d e_D) from ONE field_fn(S) call (all centers locked)."""
    S = seq.clone()
    for p in pins:
        S[p.position] = p.prot_idx
    ef = field_fn(S)                                          # [L, V]
    total = 0.0
    for p in pins:
        e_p = float(ef[p.position, p.prot_idx])
        dep = min(float(ef[p.position, d]) for d in p.dep_idxs)
        total += e_p - dep
    return total


def _combos_or_sample(pool, k, cap, rng):
    """k-combinations of (position, type) pool entries with DISTINCT positions. Enumerate + shuffle when
    small; rejection-sample up to ``cap`` distinct combos when large."""
    import itertools
    import math
    n = len(pool)
    if k > n:
        return []
    if math.comb(n, k) <= cap * 4:
        combos = [c for c in itertools.combinations(pool, k) if len({p for p, _ in c}) == k]
        rng.shuffle(combos)
        return combos[:cap]
    out, seen, tries = [], set(), 0
    while len(out) < cap and tries < cap * 40:
        tries += 1
        combo = tuple(rng.sample(pool, k))
        if len({p for p, _ in combo}) != k:
            continue
        key = tuple(sorted(combo))
        if key in seen:
            continue
        seen.add(key)
        out.append(combo)
    return out


def _finalize_plan(ctx, crit, pins) -> Optional[PlacementPlan]:
    """Build a PlacementPlan: sort pins by position (stable seed_key), compute the designable set
    (union of pin neighbourhoods, or the whole free binder chain), drop if there is nothing to design."""
    pins = sorted(pins, key=lambda p: p.position)
    pin_pos = {p.position for p in pins}
    if crit.infill_scope == "chain":
        designable = sorted(set(int(x) for x in ctx.chA_free_idx.tolist()) - pin_pos)
    else:
        dz = set()
        for p in pins:
            dz.update(int(x) for x in
                      ctx.neighbour_mask(p.position, crit.neighbour_k).nonzero().flatten().tolist())
        designable = sorted(dz - pin_pos)
    if not designable:
        return None
    # HARD max-mutations budget: if the designable union exceeds the cap, keep the closest-coupled ones
    # (best rank in any centre's kNN). Bounds n_mutations directly, across all centres. 0 = no cap.
    if crit.max_mutations and len(designable) > crit.max_mutations:
        designable = sorted(_knn_rank(ctx, pins, designable)[:crit.max_mutations])
    label = "+".join(sorted(p.protonation_type for p in pins))
    return PlacementPlan(pins=pins, designable=designable, label=label)


def _global_protonation_dH(ctx, seq) -> Optional[float]:
    """Binder pH-response: H(all binder titratable -> protonated) - H(-> deprotonated).

    Neutral His = mean over {HID, HIE} (v3/v4 tautomers) or the single HIS-S token (v6). Lower =
    the binder prefers its titratable residues protonated; higher = deprotonated/neutral. Block
    descent MINIMISES this term when ``global_weight`` > 0 (so a positive weight pushes toward
    protonation); the sign of the weight sets the direction. Returns None if the vocab lacks the
    needed microstates.
    """
    t2i = ctx.encoding.token_to_idx
    base_needed = ("HIS-P", "ASP-P", "ASP-D", "GLU-P", "GLU-D")
    if any(t not in t2i for t in base_needed):
        return None
    if "HID" in t2i and "HIE" in t2i:            # v3/v4: neutral His = tautomer mean
        his_dep_tokens = ("HID", "HIE")
    elif "HIS-S" in t2i:                         # v6: neutral His = single HIS-S token
        his_dep_tokens = ("HIS-S",)
    else:
        return None
    res_name = ctx.token_aa.res_name
    his = torch.from_numpy((res_name == "HIS")).to(ctx.device) & ctx.chainA_t
    asp = torch.from_numpy((res_name == "ASP")).to(ctx.device) & ctx.chainA_t
    glu = torch.from_numpy((res_name == "GLU")).to(ctx.device) & ctx.chainA_t

    sp = seq.clone()
    sp[his] = t2i["HIS-P"]; sp[asp] = t2i["ASP-P"]; sp[glu] = t2i["GLU-P"]
    h_prot = ctx.scorer.H_of(sp)
    sd = seq.clone(); sd[asp] = t2i["ASP-D"]; sd[glu] = t2i["GLU-D"]
    h_his_dep = 0.0
    for tok in his_dep_tokens:                   # mean over the neutral-His token(s)
        sd[his] = t2i[tok]
        h_his_dep += ctx.scorer.H_of(sd)
    return h_prot - h_his_dep / len(his_dep_tokens)


def _seq_entropy_bits(letters: str) -> float:
    """Shannon entropy (bits) of the amino-acid composition of ``letters`` (higher = more diverse).
    The sequence-diversity metric used throughout the collapse study (over the redesigned region)."""
    from collections import Counter
    n = len(letters)
    if n == 0:
        return 0.0
    p = np.array([v / n for v in Counter(letters).values()], dtype=float)
    return float(-(p * np.log2(p)).sum())


def _decoded_prob_score(ctx, crit, field_fn, valid_mask, seq, positions) -> float:
    """Mean conditional prob (at T) of the chosen token over ``positions``."""
    Pf = _cond_dist(field_fn(seq), valid_mask, crit.temperature)
    valid_idx = valid_mask.nonzero().flatten()
    vp = {int(t): i for i, t in enumerate(valid_idx.tolist())}
    vals = [float(Pf[p, vp[int(seq[p])]]) for p in positions if int(seq[p]) in vp]
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# Compatibility helper: forward -> Potts tables
# ---------------------------------------------------------------------------

def run_forward_compat(model: PottsMPNN, network_input: dict) -> Dict[str, torch.Tensor]:
    """Single forward pass returning the Potts tables (B=1)."""
    network_input["input_features"]["repeat_sample_num"] = 1
    with torch.no_grad():
        out = model(network_input)
    return {
        "etab_out": out["potts_context"].etab_out,
        "E_idx": out["potts_context"].E_idx,
    }
