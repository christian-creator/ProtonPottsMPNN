# pH-sensitive binder design with Proton-PottsMPNN

A **PottsMPNN with an explicit protonation-state alphabet**, for designing **pH-switchable** binders.
Histidine is `HIS-P` (charged, +1) vs `HIS-S` (neutral); acids are `ASP-P`/`GLU-P` (protonated, neutral
COOH) vs `ASP-D`/`GLU-D` (deprotonated, −1). Because the learned Potts energy is protonation-aware, the
design engine can **pin protonated centres** and redesign around them so that binding **switches with pH**.

This is the code for *"pH-sensitive binder design with Proton-PottsMPNN"* (Jacobsen et al., 2026). It
extends **PottsMPNN** — the Potts-energy inverse-folding
model of **Birnbaum & Keating** ([github.com/KeatingLab/PottsMPNN](https://github.com/KeatingLab/PottsMPNN);
[PNAS 2026, 10.1073/pnas.2535494123](https://www.pnas.org/doi/10.1073/pnas.2535494123)) — by adding explicit
protonation-state tokens so that a single energy function scores alternative protonation assignments on a
fixed backbone, which is what makes pH-conditioned design possible.

![Figure 1 — Proton-PottsMPNN and pH-conditioned binder design](figures/Figure_1.png)

> **Figure 1.** **(a)** Local geometric features predict per-residue protonation states, encoded as sequence
> tokens to train Proton-PottsMPNN; backbone node/edge embeddings feed a shared encoder, then an
> autoregressive decoder (token likelihoods) and a Potts head (single-site fields + pairwise couplings).
> **(b)** Protonation-conditioned redesign for a fixed `ASP-P` centre (pink) vs `ASP-D` (blue): neighbours are
> mutated to balance global Potts energy against the selective gap `E_selective = E_P − E_D`, weighted by λ.
> **(c)** Flow cytometry of enriched yeast-display binders incubated with PD-L1 at pH 5.0 vs 7.4.

The project has two halves, and this folder is self-contained for both:

1. **Label** — a FLAML labeller assigns a protonation state to every titratable residue of every training
   structure (the supervision the model learns from).
2. **Design** — the trained Potts model drives a block-descent optimiser that places protonated centres and
   redesigns their neighbourhood.

---

## Install (once) — with uv

```bash
cd ProtonPottsMPNN
./install.sh                     # uv venv (Python 3.12) + uv pip install -e ./foundry + extras
source .venv/bin/activate
```

`install.sh` runs `uv venv --clear` then a single `uv pip install -e ./foundry -r requirements-extra.txt`
(the one resolution keeps both the foundry core — torch, lightning, atomworks[ml] — and the extras — the
FLAML stack, jupyter, propka), verifies `import mpnn` resolves inside this folder, and registers the venv as
a Jupyter kernel `ProtonPottsMPNN (.venv)`. Scripts just `import mpnn`; there are no `sys.path` hacks.

- **Python 3.12** is required (`mpnn`/`foundry` pin `>=3.12,<3.13`).
- **HBPLUS** is an external C binary (not pip-installable) used by the **labeller** and the **fold scoring**
  to read H-bond geometry. Point `HBPLUS_PATH` at your build (`export HBPLUS_PATH=/path/to/hbplus`). The
  design engine needs it too: `_build_context` featurises through `prepare_potts_input`, and the `v6`
  feature pipeline computes H-bonds on every structure it reads.
- Let the install finish uninterrupted — a killed `uv pip install` can leave the venv half-written (package
  metadata present, module files missing). If imports fail oddly, repair in place with
  `uv pip install --python .venv/bin/python --reinstall -e ./foundry -r requirements-extra.txt`.

| part | needs mpnn/torch | needs HBPLUS | needs FLAML stack | needs external oracle weights |
|------|:---:|:---:|:---:|:---:|
| **label a PDB** (`labeller/`) | ✅ | ✅ | ✅ | — |
| **design a binder** (`inference/`) | ✅ | ✅ | — | — |
| **score a fold** (`scoring/`) | ✅ | ✅ (pH-bonds) | — | — |
| **benchmarks** (`benchmarks/`) | ✅ | — | — | ProteinMPNN arm only |
| **train** (`training/`, reference) | ✅ | ✅ | — | — |

---

## Two examples

### 1. Label a PDB with the protonation pipeline

Assign a protonation state to every His/Asp/Glu in a structure — the labels the Potts model is trained on and
the design engine consumes. This runs the **transformation pipeline** `prepare_potts_input(..., extended_vocab="v6")`
— the *same* code path the training data and `_build_context` use: it strips hydrogens, runs HBPLUS, applies
the FLAML labeller, and attaches a per-residue `protonation_label` token (`-P` protonated · `-S` neutral His ·
`-D` deprotonated acid · `-A` ambiguous).

```bash
HBPLUS_PATH=/path/to/hbplus python labeller/label_pdb.py     # -> labeller/outputs/protonation_labels.csv
```
```python
import numpy as np
from mpnn.potts_inference import prepare_potts_input

out = prepare_potts_input("inference/examples/pdl1_seed_binder.pdb", extended_vocab="v6")
ca  = out["atom_array"][out["atom_array"].atom_name == "CA"]     # one CA per residue, token order
labels = ca.get_annotation("protonation_label")                  # per-residue token: HIS-S / ASP-D / GLU-P / …
```
```text
chain  res_id res_name token
    A      32      HIS  HIS-S     # neutral at rest
    A      57      GLU  GLU-D     # deprotonated
    B      52      HIS  HIS-S
    …                             # 21 titratable residues, all neutral/deprotonated
```
Note the read-out: the apo seed binder carries **no** strongly-protonated residue at rest — which is exactly
why *design* (below) **pins** `HIS-P`/`ASP-P`/`GLU-P` centres deliberately rather than reading them off the
input. The pipeline emits the discrete token (the model-input form); for the raw labeller probabilities
(`p_protonated`, `sd`) call `mpnn.transforms.ev6.EV6Predictor` directly — the labeller `prepare_potts_input` wraps.

### 2. Design a pH-switch binder

`PottsMPNNPHEngine` (in the `mpnn` package at `inference_engines/potts_mpnn_ph.py`) places protonated centres
and redesigns their neighbourhood with **Potts-head block descent** — the *exact* optimiser used in our
internal design campaigns.

```bash
python inference/design_ph.py       # -> inference/outputs/ (see below)
# interactively:  jupyter lab inference/design_ph.ipynb   (pick the "ProtonPottsMPNN (.venv)" kernel)
```
```python
from mpnn.inference_engines.potts_mpnn_ph import PottsMPNNPHEngine, PHDesignCriteria

engine = PottsMPNNPHEngine(checkpoint_path=CKPT, extended_vocab="v6")   # 30-token v6 model
crit = PHDesignCriteria(
    method="block_descent", backend="potts",          # the internal-campaign optimiser
    combined_lambda=0.3,                               # Eq (6): O = (1−λ)·zscore(H_stab) + λ·zscore(Σ sel)
    center_types=["HIS-P", "ASP-P", "GLU-P"],          # composition to place …
    placement_by="scan_potts",                         # … placement chooses the positions
    dep_map={"HIS-P": ["HIS-S"], "ASP-P": ["ASP-D"], "GLU-P": ["GLU-D"]},  # v6 has no HID/HIE
    block_size=3, temperature=0.05, neighbour_k=16, max_mutations=20,
    record_trajectory=True,
)
design_set = engine.run_ph_redesign(atom_array=aa, binder_chain="A", criteria_list=[crit], seed=0)
```

The notebook opens up the internals: **placement** (which state goes where, and how many centres — acids to
the core, `HIS-P` to the surface) and the **block-descent trajectory** (stability + selectivity vs. step).
`_build_context` featurises through `prepare_potts_input` (the same transform pipeline the labeller/training
use) — the engine reads a path *or* an AtomArray.

**Many designs → the Pareto front.** `run_ph_redesign` takes a *list* of criteria, so N designs is just a
`combined_lambda` sweep from 0 (pure stability) to 1 (pure selectivity), fanned across a CPU fork pool
(`n_jobs`). The notebook runs `N_DESIGNS` of them and plots each in **(stability, selectivity)** space with the
**Pareto front** marked (`pareto_front.png`). Set `N_DESIGNS = 20` for a denser front.

**Fold the Pareto designs.** It then picks `N_FOLD` designs off the Pareto front and folds each with the
**packaged RF3 engine** (`fold_rf3.py`, from_target templating: the target chain is templated, the binder is
folded from its designed sequence), scoring charge clashes + pH-sensitive H-bonds at the pinned centres. RF3
**code** ships here, but the **weights (~3 GB) do not** — set `RF3_CKPT=/abs/rf3_*.ckpt` and run on a **GPU**
(with the RF3 atom-embedding cache) to fold. This path was **exercised end-to-end** (input build → checkpoint
load → RF3 forward → a 2-chain binder+target `.cif`) with an internal checkpoint; on a GPU node with the
embedding cache it produces a physical fold. Without RF3 the notebook still runs: it **exports** the selected
designs (`pareto_fold_manifest.json`) ready to fold elsewhere and scores a shipped real RF3 fold as the demo.

**Every design is saved with its protonation states.** Outputs in `inference/outputs/`:

| file | what |
|------|------|
| `designs.fasta` | 1-letter canonical sequence (RF3-foldable) |
| `designs_states.fasta` | the **3-letter + protonation-state** sequence (`… ASP-P … HIS-P …`) |
| `designs.tsv` / `designs.json` | both sequence forms + energies + pinned centres |
| `trajectory.tsv` | the binder sequence (1-letter **and** 3-letter/protonation) **at every optimisation step** |
| `sweep_designs.tsv` | the N sweep designs (λ, stability, selectivity, both sequences) |
| `pareto_fold_manifest.json` | the `N_FOLD` Pareto designs selected to fold (sequence, binder/target chains, centres) |
| `pareto_fold_scores.csv` | per-fold clash / pH-bond scores (**only when RF3 is available**) |
| `placement_scan.png` · `optimisation_trajectory.png` · `pareto_front.png` | the figures |

> Point it at your own backbone by editing `PDB` / `BINDER_CHAIN` (or `inference/examples/example_meta.json`).
> The checkpoint's `extended_vocab` **must** be `"v6"` or the 30-token weight load fails.

---

## Layout

| folder | what it holds |
|--------|---------------|
| [`foundry/`](foundry/) | a full verbatim copy of the `ph/foundry` monorepo. The `mpnn` package is the deliverable: the model (`model/pottsmpnn.py`), transforms, the **deployed FLAML labeller** (`transforms/ev6/`), the **design engine** (`inference_engines/potts_mpnn_ph.py`), and `prepare_potts_input` (`potts_inference.py`). `models/{rf3,rfd3,rfd3na}` come along but their multi-GB weights are not shipped. |
| [`labeller/`](labeller/) | the FLAML protonation labeller: `label_pdb.py` (example above), the train path `01…05_*.py`, `pr_curve.py` (AUPR), and the trained models. |
| [`inference/`](inference/) | `design_ph.ipynb` / `.py` (example above) + `design_placement_scan.py` (the engine's per-design placement plot) + `fold_rf3.py` (fold a design with the packaged RF3 engine; needs `RF3_CKPT` + GPU) + `examples/` (a PD-L1 seed binder + a real RF3 fold). |
| [`scoring/`](scoring/) | fold read-outs: `charge_clash` (geometry) + `annotate` (pH-sensitive H-bonds / salt bridges via HBPLUS+PLIP). `score_example.py` is a runnable demo. |
| [`benchmarks/`](benchmarks/) | PKAD (pKa), MegaScale/FireProt (stability), binding AP, within-backbone, and `placement_by_class.py` (library-wide placement propensity) — all data in-folder. |
| [`training/`](training/) | PottsMPNN + H-bond-head SLURM launchers — **reference** (cluster-specific; the trainer itself is `mpnn.train`). |
| [`checkpoints/`](checkpoints/) | the v6 design checkpoint (21 MB). |

### Where the core code lives

| component | path |
|-----------|------|
| **PottsMPNN implementation** (encoder → Potts field + pairwise couplings) | [`model/pottsmpnn.py`](foundry/models/mpnn/src/mpnn/model/pottsmpnn.py) · loss: [`loss/potts_loss.py`](foundry/models/mpnn/src/mpnn/loss/potts_loss.py) · train entry: [`train.py`](foundry/models/mpnn/src/mpnn/train.py) |
| **Transformation pipeline** (structure → model input: parse, HBPLUS/SASA features, protonation-vocab encoding) | [`potts_inference.py`](foundry/models/mpnn/src/mpnn/potts_inference.py) (`prepare_potts_input`) · transforms: [`transforms/`](foundry/models/mpnn/src/mpnn/transforms/) (FLAML labeller in [`transforms/ev6/`](foundry/models/mpnn/src/mpnn/transforms/ev6/)) |
| **Inference / design engine** (placement + block-descent pH-redesign) | [`inference_engines/potts_mpnn_ph.py`](foundry/models/mpnn/src/mpnn/inference_engines/potts_mpnn_ph.py) (`PottsMPNNPHEngine`, `PHDesignCriteria`) |

### Everything else, one command each

| I want to… | run | out |
|------------|-----|-----|
| **label a PDB** | `HBPLUS_PATH=… python labeller/label_pdb.py` | `labeller/outputs/protonation_labels.csv` |
| **design a binder** | `python inference/design_ph.py` | placement / trajectory / Pareto PNGs + `designs*.fasta` / `.tsv` (with protonation states) |
| **retrain the labeller** | `python labeller/05_train_one.py HIS features 1200` | `labeller/models/automl_feature_HIS/` |
| **labeller AUPR** | `python labeller/pr_curve.py` | `labeller/{his,acid}_pr.png` (AP HIS 0.70, acids 0.33) |
| **score a fold** | `HBPLUS_PATH=… python scoring/score_example.py` | pH-bond / charge-clash counts |
| **placement by class** (library-wide) | `python benchmarks/placement_by_class.py` | `benchmarks/placement_by_class.png` |
| **pKa benchmark** | `PROTON_ROOT=$PWD python -m mpnn.scripts.eval_pkad --checkpoints checkpoints/…/epoch-0125.ckpt` | `pkad_*.csv` + scatter |
| **stability benchmark** | `EV6_OUT_SUBDIR=his0.3_acid0.06 python benchmarks/stability_benchmark.py` | ΔΔG CSVs (GPU recommended) |
| **3-way bar plot** | `EV6_OUT_SUBDIR=his0.3_acid0.06 python benchmarks/benchmark_barplot.py` | recovery + MegaScale + FireProt figure |
| **binding AP** | `python benchmarks/binding_ap.py` | `benchmarks/results/summary_global_ap.png` |
| **within-backbone** | `python benchmarks/within_backbone.py` | `within_backbone_inverse_outcomes.png` |

Benchmark scripts derive the package root from their location (override with `PROTON_ROOT=/abs/path`). The
model-quality benchmarks (PKAD, stability) compute from raw structures; the design benchmarks (binding,
within-backbone) read shipped parquets (regenerating them needs the Boltz-2 / RF3 oracles).

---

## License

This project is released under the **MIT License** — see [`LICENSE`](LICENSE). It covers the
Proton-PottsMPNN code authored here (`labeller/`, `inference/`, `scoring/`, `benchmarks/`, `training/`,
`checkpoints/`, the docs, and the pH-design additions to the `mpnn` package). The bundled [`foundry/`](foundry/)
is a verbatim copy of IPD's rc-foundry and keeps its own **BSD 3-Clause License**
([`foundry/LICENSE.md`](foundry/LICENSE.md), © 2025 Institute for Protein Design, University of Washington).

If you use this in academic work, please cite the manuscript *"pH-sensitive binder design with
Proton-PottsMPNN"* (Jacobsen et al., 2026).
