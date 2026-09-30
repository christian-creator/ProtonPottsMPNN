# %% [markdown]
# # Proton-PottsMPNN — pH-switch binder design (example)
#
# This notebook designs one example binder with **Proton-PottsMPNN** and opens up the design
# **under the hood**: it shows where the protonated centres get *placed*, how many there are, and the
# **optimisation trajectory** (energy vs. step) as the block-descent optimiser converges.
#
# It runs the *exact* optimiser used in our internal design campaigns: Potts-head **block descent**.
#
# It uses only files inside this `ProtonPottsMPNN/` folder — the relocated design engine in the
# `mpnn` foundry package, the packaged v6 checkpoint, and one example dimer backbone.
#
# Pipeline context: see `../README.md` (§Pipeline — how the protonation labels this model was trained on
# are produced; and the companion `../labeller/label_pdb.py` example that labels a PDB with FLAML).

# %%
import os
import sys
import json
from pathlib import Path

os.environ.pop("DEBUG", None)                     # a stray non-boolean DEBUG in your shell breaks environs/foundry
# CPU is plenty for the DESIGN stage (one small binder). But RF3 folding at the end of this script needs
# a visible CUDA device -- `fold_rf3.rf3_available()` requires torch.cuda.is_available() -- so hiding the
# GPU unconditionally (as this used to) silently forced the no-RF3 branch even with RF3_CKPT set on a GPU
# node. Only hide the GPU when there is no fold to run.
if not os.environ.get("RF3_CKPT"):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd
import matplotlib
try:
    get_ipython()                                  # in a Jupyter kernel → inline backend, plots show inline
except NameError:
    matplotlib.use("Agg")                          # plain `python design_ph.py` → headless (figures saved to files)
import matplotlib.pyplot as plt
from biotite.structure.io.pdb import PDBFile

import mpnn                                         # resolved from the venv:  pip install -e ./foundry
print("mpnn package:", Path(mpnn.__file__).resolve())

# package root = the parent of this inference/ folder; local analysis modules onto the path
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
PKG = HERE.parent if HERE.name == "inference" else HERE
sys.path[:0] = [str(PKG / d) for d in ("inference", "scoring")]   # sibling helpers (not pip-installed)

CKPT = PKG / "checkpoints" / "potts_v6_afdb_edge_his0.3_acid0.06" / "epoch-0125.ckpt"
META = json.loads((PKG / "inference" / "examples" / "example_meta.json").read_text())
PDB = PKG / "inference" / "examples" / META["pdb"]
BINDER_CHAIN = META["binder_chain"]
OUT = PKG / "inference" / "outputs"
OUT.mkdir(exist_ok=True)
print(f"example: {META['pdb_id']}  binder chain {BINDER_CHAIN}  binder_len {META['binder_len']}")

# %%
# --- build the design engine on the packaged v6 checkpoint (30-token extended vocab) ---
from mpnn.inference_engines.potts_mpnn_ph import PottsMPNNPHEngine, PHDesignCriteria

engine = PottsMPNNPHEngine(
    checkpoint_path=str(CKPT), extended_vocab="v6",
    out_directory=None, write_fasta=False, write_structures=False,
)
atom_array = PDBFile.read(str(PDB)).get_structure(model=1)
ctx = engine._build_context(atom_array, BINDER_CHAIN)
binder_res_ids = [int(ctx.token_aa.res_id[int(p)]) for p in ctx.chA_free_idx.tolist()]
print(f"engine ready · {len(binder_res_ids)} free (designable) binder positions")

# %%
# --- design criteria: the EXACT optimiser used in our internal design campaigns ---
# Potts-head BLOCK DESCENT (deterministic/near-exact MAP per block; temperature 0.05 ≈ argmin with a
# little Boltzmann sampling). center_types pins the composition; placement (scan_potts) CHOOSES positions.
# These hyper-parameters mirror our internal redesign-sweep / optimize-only configs.
crit = PHDesignCriteria(
    method="block_descent",           # production optimiser (NOT converged_mcmc)
    backend="potts",
    temperature=0.05,                 # Sampling temperature
    samples_per_site=2,               # Number of designs to make per condition
    block_size=3,                     # triples
    combined_lambda=0.3,              # λ in the manuscript Eq (6): O = (1−λ)·zscore(H_stab) + λ·zscore(Σ sel)
                                      #   0 = pure stability, 1 = pure selectivity (z-scored, so λ is relative)
    seed_source="native",             # start from the native binder sequence (no external seeds)
    center_types=["HIS-P", "ASP-P", "GLU-P"],   # one His + one Asp + one Glu protonated centre
    # v6 deprotonated contrasts (v6 has no HID/HIE — neutral His is HIS-S):
    dep_map={"HIS-P": ["HIS-S"], "ASP-P": ["ASP-D"], "GLU-P": ["GLU-D"]},
    forbidden_tokens=["HIS-A", "ASP-A", "GLU-A", "UNK"],   # only the ambiguous microstates forbidden
    placement_by="scan_potts",        # rank candidate positions by Potts dE(protonated) - dE(deprotonated)
    placement_region=["all"],
    repetitive_window_parents=["ARG", "LYS", "HIS", "ASP", "GLU"],   # poly-charge collapse penalty
    repetitive_window_radius=2,
    repetitive_window_weight=1.0,
    neighbour_k=16,                   # per-centre redesign-extent cap
    max_mutations=20,                 # total mutation budget
    record_trajectory=True,           # records the block-descent optimisation path
)
design_set = engine.run_ph_redesign(
    atom_array=atom_array, binder_chain=BINDER_CHAIN, criteria_list=[crit], seed=0,
)
print(f"produced {len(design_set)} design(s)")
best = design_set[0]                    # PHDesignSet is sorted, lowest Potts energy first

# %% [markdown]
# ## Placement — where the pipeline places each protonation state (this binder)
# The engine ranks every free binder position by the **selective placement score** of each state (Potts
# field; lower = preferred) and pins the best. Below: the per-position score for every state + the
# state-agnostic **ΔE** (substitution-cost) scan; a ★ marks each state's top-1 placement, and the track
# colours every residue by structural class. Then this design's actually-pinned centres.

# %%
from design_placement_scan import plot_design_placement
fig, top = plot_design_placement(engine, atom_array, BINDER_CHAIN)
fig.savefig(OUT / "placement_scan.png", dpi=200, bbox_inches="tight")
plt.show()                                          # display inline in the notebook
print("top-1 placement per state (this binder):")
for stt, (rid, c, sc) in top.items():
    print(f"  {stt:6s} res {rid:4d}  class={c:9s}  score={sc:.3f}")
print(f"\nthis design's pinned centres: {best.n_centers} × "
      f"{list(zip(best.center_res_ids, best.center_protonation_types))}")
print("saved", OUT / "placement_scan.png")

# %% [markdown]
# ## Block-descent optimisation trajectory
# Each recorded step is one block-descent move. We plot the total **Potts energy** (stability) and the
# **selective energy** (the protonated-vs-deprotonated gap the design is driving down) versus step.

# %%
traj = best.energy_trajectory or []
steps = [t["step"] for t in traj]
potts = [t["potts_energy"] for t in traj]
selE = [t["selective_energy"] for t in traj]
print(f"trajectory length: {len(traj)} steps")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(steps, potts, color="#189486", lw=2)
ax[0].set_xlabel("block-descent step"); ax[0].set_ylabel("Potts energy  H(S)"); ax[0].set_title("Stability")
ax[1].plot(steps, selE, color="#4153A4", lw=2)
ax[1].set_xlabel("block-descent step"); ax[1].set_ylabel("selective energy  Σ(e_P − e_D)")
ax[1].set_title("Selectivity (protonated − deprotonated)")
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "optimisation_trajectory.png", dpi=200)
plt.show()                                          # display inline in the notebook
print("saved", OUT / "optimisation_trajectory.png")

# --- save the sequence AT EVERY STEP: 1-letter canonical + 3-letter/protonation tokens ---
# each recorded step now carries the working binder sequence, so the whole optimisation path is saveable.
traj_df = pd.DataFrame([{
    "step": t["step"],
    "potts_energy": t["potts_energy"],
    "selective_energy": t["selective_energy"],
    "canonical_sequence": t["canonical_sequence"],   # 1-letter (RF3-foldable)
    "extended_tokens": t["extended_tokens"],          # 3-letter + protonation state (e.g. HIS-P ASP-D …)
} for t in traj])
traj_df.to_csv(OUT / "trajectory.tsv", sep="\t", index=False)
print("saved", OUT / "trajectory.tsv", f"({len(traj_df)} steps × sequence)")
if len(traj_df):
    print(f"  step 0   tokens: {traj[0]['extended_tokens'][:70]} …")
    print(f"  step {traj[-1]['step']:<3} tokens: {traj[-1]['extended_tokens'][:70]} …")

# %% [markdown]
# ## The resulting design

# %%
print(f"design id       : {best.design_id()}")
print(f"final Potts H   : {best.final_potts_energy:.3f}")
print(f"selective energy: {best.selective_energy}")
print(f"global ΔH (prot): {best.global_protonation_dH}")
print(f"binder sequence : {best.canonical_sequence}")
print(f"extended tokens : {' '.join(best.extended_tokens)}")

# write the design(s) out — canonical FASTA (foldable), a parallel 3-letter/protonation FASTA, a flat TSV,
# and the metadata json. Everything stays inside inference/outputs/.
with open(OUT / "designs.fasta", "w") as fh:                       # 1-letter canonical (for folding)
    for d in design_set:
        fh.write(f">{d.design_id()} H={d.final_potts_energy:.3f} sel={d.selective_energy}\n"
                 f"{d.canonical_sequence}\n")
with open(OUT / "designs_states.fasta", "w") as fh:                # 3-letter + PROTONATION STATE per residue
    for d in design_set:
        fh.write(f">{d.design_id()} H={d.final_potts_energy:.3f} sel={d.selective_energy}\n"
                 f"{' '.join(d.extended_tokens)}\n")                # e.g. ALA ALA ARG … ASP-P … HIS-P …
pd.DataFrame([{                                                     # one flat table with both sequence forms
    "design_id": d.design_id(),
    "combined_lambda": d.combined_lambda,
    "potts_energy": d.final_potts_energy,
    "selective_energy": d.selective_energy,
    "centers": ";".join(f"{r}:{t}" for r, t in zip(d.center_res_ids or [], d.center_protonation_types or [])),
    "canonical_sequence": d.canonical_sequence,
    "extended_tokens": " ".join(d.extended_tokens),
} for d in design_set]).to_csv(OUT / "designs.tsv", sep="\t", index=False)
with open(OUT / "designs.json", "w") as fh:
    json.dump([d.to_metadata() for d in design_set], fh, indent=2, default=float)
print("wrote", OUT / "designs.fasta", "(canonical),", OUT.name + "/designs_states.fasta (3-letter+state),",
      OUT.name + "/designs.tsv,", OUT.name + "/designs.json")

# %% [markdown]
# ## Many designs at once — the stability ↔ selectivity Pareto front
# `run_ph_redesign` featurises the backbone **once** and then runs a whole **list** of criteria against the
# shared Potts tables — so generating N designs is just a list of criteria. Here we sweep `combined_lambda`
# from **0 (pure stability)** to **1 (pure selectivity)**: each λ is one design trading the two objectives
# differently, and together they trace the **Pareto front**. (`n_jobs>1` fans them across a CPU fork pool.)

# %%
from dataclasses import replace

N_DESIGNS = 12                                  # bump to 20+ for a denser front
lambdas = np.round(np.linspace(0.0, 1.0, N_DESIGNS), 3)
sweep_criteria = [
    replace(crit, combined_lambda=float(lam),
            samples_per_site=1,                 # one design per λ (deterministic-ish at T=0.05)
            record_trajectory=False)            # trajectories off → fast; we only need the endpoints
    for lam in lambdas
]
n_jobs = min(N_DESIGNS, (os.cpu_count() or 1))
print(f"running {N_DESIGNS} designs (λ = {lambdas[0]}…{lambdas[-1]}) on {n_jobs} workers…")
sweep = engine.run_ph_redesign(
    atom_array=atom_array, binder_chain=BINDER_CHAIN, criteria_list=sweep_criteria, seed=0, n_jobs=n_jobs,
)
sweep_df = pd.DataFrame([{
    "combined_lambda": d.combined_lambda,
    "potts_energy": d.final_potts_energy,        # stability  (lower = stabler)
    "selective_energy": d.selective_energy,      # selectivity (lower = protonated preferred)
    "design_id": d.design_id(),
    "canonical_sequence": d.canonical_sequence,
    "extended_tokens": " ".join(d.extended_tokens),
} for d in sweep]).sort_values("combined_lambda").reset_index(drop=True)
sweep_df.to_csv(OUT / "sweep_designs.tsv", sep="\t", index=False)   # every design + both sequence forms
print(f"produced {len(sweep_df)} designs → {OUT.name}/sweep_designs.tsv")

# %%
# --- Pareto front: minimise BOTH potts_energy and selective_energy (lower-left is best) ---
P = sweep_df[["potts_energy", "selective_energy"]].to_numpy()
dominated = np.zeros(len(P), bool)
for i in range(len(P)):
    for j in range(len(P)):
        if i != j and (P[j] <= P[i]).all() and (P[j] < P[i]).any():
            dominated[i] = True; break
sweep_df["pareto_optimal"] = ~dominated
front = sweep_df[~dominated].sort_values("potts_energy")

fig, ax = plt.subplots(figsize=(6.4, 5.0))
sc = ax.scatter(sweep_df.potts_energy, sweep_df.selective_energy,
                c=sweep_df.combined_lambda, cmap="viridis", s=70, zorder=3,
                edgecolor="white", linewidth=0.6)
ax.plot(front.potts_energy, front.selective_energy, "-", color="#189486", lw=2, zorder=2, label="Pareto front")
ax.scatter(front.potts_energy, front.selective_energy, s=150, facecolor="none",
           edgecolor="#189486", linewidth=2, zorder=4)
cbar = fig.colorbar(sc, ax=ax); cbar.set_label("combined_lambda  (0 = stability → 1 = selectivity)")
# axes are raw energies (not inverted): lower is better, so 'better' points LEFT (x) and DOWN (y).
# in the 90°-rotated y-label a '←' glyph renders pointing down — toward the more-selective (lower) end.
ax.set_xlabel("Potts energy  H(S)   (← more stable)")
ax.set_ylabel("selective energy  Σ(e_P − e_D)   (← more pH-selective)")
ax.set_title(f"{len(sweep_df)} pH-switch designs — stability ↔ selectivity trade-off")
ax.legend(frameon=False, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "pareto_front.png", dpi=200)
plt.show()                                          # display inline in the notebook
print(f"Pareto-optimal designs: {int((~dominated).sum())}/{len(sweep_df)}  →  saved {OUT.name}/pareto_front.png")

# %% [markdown]
# ## Fold the Pareto designs & score them — clashes + pH-sensitive H-bonds
# The designed sequences must be **folded** to check their structure. Folding uses **RF3 (from_target
# templating)** — the target chain is templated and the binder is folded fresh from its designed sequence.
# The RF3 *code* ships in this package (`foundry/models/rf3/`), but the **weights (~3 GB) are not bundled**
# and RF3 needs a **GPU** — so set `RF3_CKPT` and run on a GPU to fold (see `../README.md` and
# `fold_rf3.py`). We pick **N designs off the Pareto front**, export them for folding, and — when RF3 is
# available — fold each and score same-sign **charge clashes** (≤4 Å) and **pH-sensitive H-bonds** (HBPLUS)
# at its pinned centres. Without RF3 the notebook still runs: it exports the designs and scores a shipped
# real RF3 fold as the read-out demo.

# %%
from annotate_charge_clash import charge_clash
from fold_rf3 import rf3_available, fold_from_target

N_FOLD = 3                                                              # how many Pareto designs to fold
TARGET_CHAINS = sorted(set(np.asarray(atom_array.chain_id)) - {BINDER_CHAIN})

# select N Pareto-optimal designs (lowest selective energy first) as design objects, with their centres
by_id = {d.design_id(): d for d in sweep}
pick = sweep_df[sweep_df.pareto_optimal].sort_values("selective_energy")["design_id"].tolist()[:N_FOLD]
to_fold = [by_id[i] for i in pick]

# export them for folding (runnable everywhere — this is what you'd hand to RF3)
manifest = [{
    "design_id": d.design_id(), "combined_lambda": d.combined_lambda,
    "binder_chain": BINDER_CHAIN, "target_chains": TARGET_CHAINS,
    "centers": [[int(r), str(t)] for r, t in zip(d.center_res_ids, d.center_protonation_types)],
    "canonical_sequence": d.canonical_sequence,
} for d in to_fold]
(OUT / "pareto_fold_manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"selected {len(to_fold)} Pareto designs to fold → {OUT.name}/pareto_fold_manifest.json")

# %%
_hbplus = os.environ.get("HBPLUS_PATH") and Path(os.environ["HBPLUS_PATH"]).exists()
if rf3_available():
    print(f"RF3 available (RF3_CKPT set + GPU) — folding {len(to_fold)} Pareto designs from_target…")
    rows = []
    for d, m in zip(to_fold, manifest):
        centres = [(int(r), str(t)) for r, t in m["centers"]]
        cif = fold_from_target(PDB, d.canonical_sequence, BINDER_CHAIN, TARGET_CHAINS,
                               out_dir=OUT / "folds" / d.design_id())
        cc, sb, _ = charge_clash(str(cif), META["binder_len"], centres)          # CIF geometry (robust)
        row = {"design_id": d.design_id(), "combined_lambda": d.combined_lambda,
               "charge_clash": cc, "salt_bridge": sb, "fold": str(cif)}
        if _hbplus:
            try:                                                                 # needs a physical fold
                from annotate_folds import annotate
                a = annotate(str(cif), META["binder_len"], centres)
                row |= {"his_ph_hbonds": a["his_ph_hbonds"], "acid_ph_hbonds": a["acid_ph_hbonds"],
                        "microstate_match": a["microstate_match"]}
            except Exception as e:
                print(f"  (annotate skipped for {d.design_id()}: {type(e).__name__} — fold may be low quality)")
        rows.append(row); print(f"  {d.design_id()}: clash {cc:.2f} · salt {sb:.2f}")
    pd.DataFrame(rows).to_csv(OUT / "pareto_fold_scores.csv", index=False)
    print("wrote", OUT / "pareto_fold_scores.csv")
else:
    # No RF3 here (CPU / no weights): the designs above are exported for folding elsewhere. To still show the
    # read-outs, score a shipped REAL RF3 fold of a PD-L1 design (ipTM 0.92) — same geometry the manuscript uses.
    print("RF3 not available (set RF3_CKPT + run on a GPU to fold the exported Pareto designs).")
    fmeta = json.loads((PKG / "inference" / "examples" / "fold_meta.json").read_text())
    FOLD = str(PKG / "inference" / "examples" / fmeta["fold"])
    fcenters = [(int(r), str(t)) for r, t in fmeta["centers"]]
    print(f"read-out demo on shipped fold {fmeta['fold']} (ipTM {fmeta['fcx_iptm']}) · centres {fcenters}")
    cc, sb, _ = charge_clash(FOLD, fmeta["binder_len"], fcenters)
    print(f"  charge clashes (same-sign ≤4 Å, mean/centre) : {cc:.2f}")
    print(f"  salt bridges   (opp-sign  ≤5 Å, mean/centre) : {sb:.2f}")
    if _hbplus:
        from annotate_folds import annotate
        a = annotate(FOLD, fmeta["binder_len"], fcenters)
        print(f"  pH-sensitive H-bonds  : His {a['his_ph_hbonds']} · acid {a['acid_ph_hbonds']}")
        print(f"  microstate match      : {a['microstate_match']}")
    else:
        print("  (set HBPLUS_PATH to also count pH-sensitive H-bonds)")
