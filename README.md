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

# or put the venv elsewhere (say, off a nearly full disk):
VENV_DIR=~/venvs/ppm ./install.sh
source ~/venvs/ppm/bin/activate
```

`install.sh` runs `uv venv --clear` then a single `uv pip install -e ./foundry -r requirements-extra.txt`
(the one resolution keeps both the foundry core — torch, lightning, atomworks[ml] — and the extras — the
FLAML stack, jupyter, propka), verifies `import mpnn` resolves inside this folder, and registers the venv as
a Jupyter kernel `ProtonPottsMPNN (venv)`. Scripts just `import mpnn`; there are no `sys.path` hacks. A `VENV_DIR`
that already exists is replaced only with `./install.sh --clear`, since uv would delete whatever it holds.
`atomworks` is kept below 3, because version 3.0 rejects atom arrays without a `charge` annotation and the
structure pipeline fails on it.

- **Python 3.12** is required (`mpnn`/`foundry` pin `>=3.12,<3.13`).
- **HBPLUS** is an external C binary (not pip-installable) used to read H-bond geometry. Point
  `HBPLUS_PATH` at your build (`export HBPLUS_PATH=/path/to/hbplus`). It is needed by **every** entry
  point that featurises a structure — the labeller, the fold scoring **and the design engine**: the
  engine's `_build_context` calls the same `prepare_potts_input` pipeline, whose `CalculateHbondsPlus`
  stage shells out to HBPLUS to produce the protonation labels. Only the pure read-only paths
  (loading a checkpoint, re-scoring a shipped parquet) run without it. *Getting HBPLUS* below says where
  to get it and how to check the build.
- Let the install finish uninterrupted — a killed `uv pip install` can leave the venv half-written (package
  metadata present, module files missing). If imports fail oddly, repair in place with
  `uv pip install --python .venv/bin/python --reinstall -e ./foundry -r requirements-extra.txt`.

| part | needs mpnn/torch | needs HBPLUS | needs FLAML stack | needs external oracle weights |
|------|:---:|:---:|:---:|:---:|
| **label a PDB** (`labeller/`) | ✅ | ✅ | ✅ | — |
| **design a binder** (`inference/`) | ✅ | ✅ | ✅ | — |
| **score a fold** (`scoring/`) | ✅ | ✅ (pH-bonds) | — | — |
| **benchmarks** (`benchmarks/`) | ✅ | — | — | ProteinMPNN arm only |
| **train** (`training/`, reference) | ✅ | ✅ | — | — |

### Getting HBPLUS

HBPLUS (McDonald & Thornton, *J Mol Biol* 238:777, 1994) is a C program for hydrogen-bond geometry, written
at UCL and distributed by the Thornton group at EMBL-EBI. It is not on PyPI or conda. The structure pipeline
runs it on **every** structure it featurises (`prepare_potts_input`), so the design engine needs it as well as
the labeller and the fold scoring.

1. Get `hbplus.tar.gz` from the [EBI download page](https://www.ebi.ac.uk/thornton-srv/software/HBPLUS/download.html)
   or the copy in [RomanLas/HBPLUS](https://github.com/RomanLas/HBPLUS). The archive carries a confidentiality
   agreement (`confid.txt`): use for your own research, and cite the paper above. Do not commit the source or the
   binary to this repository.
2. `tar -xzf hbplus.tar.gz && cd hbplus && make` (needs `gcc`; version 3.2 compiles with warnings only).
3. `export HBPLUS_PATH=/absolute/path/to/hbplus/hbplus` — the executable itself, not its directory.
4. Check it on the shipped example: in a scratch directory,
   `cp inference/examples/pdl1_seed_binder.pdb x.pdb && "$HBPLUS_PATH" -h 2.5 -d 3.5 x.pdb x.pdb | tail -1`
   ends with `181 hydrogen bonds found.`

A missing, mistyped or non-executable `HBPLUS_PATH` stops the run with a message that says which of these it is.

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
    A      26      ASP  ASP-D     # deprotonated
    A      32      HIS  HIS-S     # neutral at rest
    A      57      GLU  GLU-D
    B      52      HIS  HIS-P     # protonated (on the TARGET chain)
    B      61      HIS  HIS-P
    …                             # 21 titratable residues; the 4 on the binder are all neutral/deprotonated
```
Note the read-out: the apo seed **binder** (chain A) carries **no** strongly-protonated residue at rest —
which is exactly why *design* (below) **pins** `HIS-P`/`ASP-P`/`GLU-P` centres deliberately rather than
reading them off the input. (Two His on the PD-L1 **target** chain do come back `HIS-P`.)

> The P-vs-neutral cut is part of the labels. `transforms/ev6/weights/thresholds.json` defaults to
> `his prob_thr = 0.40`, but the packaged checkpoint was trained at `0.30` (recorded in its
> `train_cfg.his_prob_thr`), and at 0.40 this same structure returns **zero** protonated residues. The
> checkpoint loaders now register the checkpoint's own operating point with the live labeller
> automatically, so inference reproduces training; `labeller/label_pdb.py` reads it off the packaged
> checkpoint for the same reason. An explicit `EV6_HIS_PROB_THR` / `EV6_ACID_PROB_THR` still wins, so
> deliberate sweeps are unaffected. The pipeline emits the discrete token (the model-input form); for the raw labeller probabilities
(`p_protonated`, `sd`) call `mpnn.transforms.ev6.EV6Predictor` directly — the labeller `prepare_potts_input` wraps.

### 2. Design a pH-switch binder

`PottsMPNNPHEngine` (in the `mpnn` package at `inference_engines/potts_mpnn_ph.py`) places protonated centres
and redesigns their neighbourhood with **Potts-head block descent** — the *exact* optimiser used in our
internal design campaigns.

```bash
python inference/design_ph.py       # -> inference/outputs/ (see below)
# interactively:  jupyter lab inference/design_ph.ipynb   (pick the "ProtonPottsMPNN (venv)" kernel)
```
```python
from mpnn.inference_engines.potts_mpnn_ph import PottsMPNNPHEngine, PHDesignCriteria

engine = PottsMPNNPHEngine(checkpoint_path=CKPT, extended_vocab="v6")   # 30-token v6 model
crit = PHDesignCriteria(
    method="block_descent", backend="potts",          # the internal-campaign optimiser
    combined_lambda=0.3,                               # Eq (6): O = (1−λ)·zscore(H_stab) + λ·zscore(Σ sel)
    seed_source="native",                              # start from the input sequence (the default "inverse" needs initial_sequences)
    center_types=["HIS-P", "ASP-P", "GLU-P"],          # composition to place …
    placement_by="scan_potts",                         # … placement chooses the positions
    dep_map={"HIS-P": ["HIS-S"], "ASP-P": ["ASP-D"], "GLU-P": ["GLU-D"]},  # v6 has no HID/HIE
    block_size=3, temperature=0.05, neighbour_k=16, max_mutations=20,
    record_trajectory=True,
)
design_set = engine.run_ph_redesign(atom_array=aa, binder_chain="A", criteria_list=[crit], seed=0)
```

**Direction.** `selective_energy` = Σ(e_P − e_D) over the pinned centres, in the bound complex, and the
optimiser **lowers** it → the bound state prefers the protonated centre → binding is **stronger at low pH**
and weakens once the centre loses its proton. The opposite direction is not available here; use
`run_switch_design` (below) and swap `on` / `off`.

**Interface region.** `PHDesignCriteria.interface_distance` (default 6 Å, CA–CA) sets the `interface`
placement region — raise it when loops reach over the target, otherwise the region can come out empty (a
warning is logged).

**Plain rows.** `design_set.to_rows(reference_sequence=…)` → one dict per design (`canonical_sequence`,
`centers`, `n_mutations`, `selective_energy`, `potts_energy`, …).

The notebook opens up the internals: **placement** (which state goes where, and how many centres — acids to
the core, `HIS-P` to the surface) and the **block-descent trajectory** (stability + selectivity vs. step).
`_build_context` featurises through `prepare_potts_input` (the same transform pipeline the labeller/training
use) — the engine reads a path *or* an AtomArray.

**Many designs → the Pareto front.** `run_ph_redesign` takes a *list* of criteria, so N designs is just a
`combined_lambda` sweep from 0 (pure stability) to 1 (pure selectivity), fanned across a CPU fork pool
(`n_jobs`). The notebook runs `N_DESIGNS` of them and plots each in **(stability, selectivity)** space with the
**Pareto front** marked (`pareto_front.png`). Set `N_DESIGNS = 20` for a denser front.

**Reproducible.** The same `seed` returns the same designs for any `n_jobs` (results come back in task
order). Two designs that share an id but differ in sequence are both kept, the second as `<id>~2`.

**Fold the Pareto designs.** It then picks `N_FOLD` designs off the Pareto front and folds each with the
**packaged RF3 engine** (`fold_rf3.py`, from_target templating: the target chain is templated, the binder is
folded from its designed sequence), scoring charge clashes + pH-sensitive H-bonds at the pinned centres. RF3
**code** ships here, but the **weights (3.04 GB) do not**. `foundry install rf3 --checkpoint-dir DIR` fetches
`rf3_foundry_01_24_latest_remapped.ckpt`; check its size (3,038,876,446 bytes), because on a slow connection the
installer reported success after writing 0.5 MB, and `curl -C - -o … https://files.ipd.uw.edu/pub/rf3/rf3_foundry_01_24_latest_remapped.ckpt`
resumes such a download. Then set `RF3_CKPT=DIR/rf3_foundry_01_24_latest_remapped.ckpt` and keep a **GPU** visible
(`design_ph.py` designs on the CPU either way and uses the GPU only for the fold). This path was **exercised
end-to-end** (input build → checkpoint load → RF3 forward → a 2-chain binder+target `.cif` → clash and pH-bond
scores) on an RTX 3070 with 8 GB: the example binder folds in about 2 to 3 minutes per design (pLDDT 0.8 at 3
recycles and 20 steps), with no atom-embedding cache set up. Without RF3 the notebook still runs: it **exports**
the selected designs (`pareto_fold_manifest.json`) ready to fold elsewhere and scores a shipped real RF3 fold as
the demo.

**Every design is saved with its protonation states.** Outputs in `inference/outputs/`:

| file | what |
|------|------|
| `designs.fasta` | 1-letter canonical sequence (RF3-foldable). **First record is the original** (native input-PDB) binder |
| `designs_states.fasta` | the **3-letter + protonation-state** sequence (`… ASP-P … HIS-P …`), native record first |
| `designs.tsv` / `designs.json` | both sequence forms + energies + pinned centres, **plus the native baseline** (`native_potts_energy`, `native_sel_at_centres`, `native_canonical_sequence`) and the deltas `dH_vs_native` / `dsel_vs_native` |
| `trajectory.tsv` | the binder sequence (1-letter **and** 3-letter/protonation) **at every optimisation step** |
| `sweep_designs.tsv` | the N sweep designs (λ, stability, selectivity, both sequences) |
| `pareto_fold_manifest.json` | the `N_FOLD` Pareto designs selected to fold (sequence, binder/target chains, centres) |
| `pareto_fold_scores.csv` | per-fold clash / pH-bond scores (**only when RF3 is available**) |
| `placement_scan.png` · `optimisation_trajectory.png` · `pareto_front.png` | the figures |

**Reading the native (original) baseline.** Every design is reported against the untouched input-PDB
binder, scored with the *same* recipe (`PottsMPNNPHEngine.score_native_baseline`):

* `native_potts_energy` is `H` of the native sequence exactly as it comes off the backbone, nothing
  pinned — directly comparable to a design's `potts_energy` (both are `ctx.scorer.H_of`). Lower = stabler,
  so a **positive** `dH_vs_native` means the design traded stability away.
* `native_sel_at_centres` is **not** a property of the native residues. A RES-P-vs-RES-D gap is undefined
  without placing those tokens, so it is measured on the native sequence **with the design's centres
  substituted in** — which is exactly what makes it comparable to the design's `selective_energy`. The
  sequence printed in the FASTA record keeps its native residues at those positions; the number does not
  describe it.
* The two therefore decompose the result: `native_sel_at_centres` is what **site choice alone** bought,
  the design's `selective_energy` is site choice **+** surrounding redesign, and `dsel_vs_native` isolates
  the redesign's contribution with placement held fixed.

> Point it at your own backbone by editing `PDB` / `BINDER_CHAIN` (or `inference/examples/example_meta.json`).
> The checkpoint's `extended_vocab` **must** be `"v6"` or the 30-token weight load fails.

---

## Switch design with explicit states (`mpnn.ph`)

`run_ph_redesign` pins protonated centres on the **binder**. `PottsMPNNPHEngine.run_switch_design` takes
**conditions** instead (say pH 7.4 and pH 6.5) that name the protonation state of **any residue on either
chain** — so a binder that binds at 7.4 but not 6.5, and the reverse, differ only by which condition is `on`,
and titratable residues on the **target** switch just like the binder's.

**Three terms, each optional** (weight 0 = off), lower is better:

| term | what it scores |
|------|----------------|
| `stability` | Potts energy of the binder alone in each condition → `max` (worst condition, default) or `mean` |
| `potency` | binding energy in the `on` condition |
| `switch` | how much weaker binding is in `off` than in `on`; `off_margin` stops rewarding the gap once it is wide enough |

**Binding energy** (`binding_model`):

| model | binding energy |
|-------|----------------|
| `state_binding` (default) | complex − binder alone − receptor alone → only the **interface** contribution counts |
| `complex_gap` | the complex energy |
| `linked_equilibrium` | experimental — averages over the protonation states the free partners populate at each condition's pH; `beta` is uncalibrated |

**One run, or a Pareto sweep — your choice.** One active term, or one explicit weight vector → an ordinary
single-objective run. `search: "pareto"` with two or more active terms sweeps weight vectors and returns the
**Pareto set** (`front`, `selected`, hypervolume).

```python
config = {
    "conditions": {                                           # state of any residue, either chain
        "on":  {"ph": 7.4, "states": {"A:12": "deprotonated", "B:409": "deprotonated"}},
        "off": {"ph": 6.5, "states": {"A:12": "protonated",   "B:409": "protonated"}},
    },
    "terms": {"stability": {"weight": 0.3}, "potency": {"weight": 0.3},
              "switch": {"weight": 0.4, "off_margin": 2.0}},
    "designable": {"chains": ["A"],                           # residues named in a condition are never redesigned
                   "near": {"sites": ["B:409"], "k": 16, "max_mutations": 20}},   # optional: only around B:409
    "binding_model": "state_binding",      # | "complex_gap" | "linked_equilibrium"
    "search": "single",                    # | "pareto"
    "target_reduce": "max",                # several target states: "max" = worst state | "mean"
}
run = engine.run_switch_design(atom_array=aa, binder_chain="A", config=config)
best = run.result.records[0]
run.describe(best, "A")                    # canonical sequence + token names of the binder
run.to_rows("A")                           # one dict per design: sequence, n_mutations, term_<name>, front, …
run.site_report(best)                      # per condition and site: complex vs free-partner preference
```

From the command line, `python inference/design_switch.py` runs the example above on the shipped PD-L1 complex
and writes the ranked designs to `inference/outputs/switch_designs.tsv`. `--help` lists the flags: the two sites,
the pH values and states, the binding model, `--states propka` and `--reference-pka propka`.

**Redesign around a residue.** `designable.near` keeps only the binder positions the model couples to the
named sites — the same neighbourhood `neighbour_k` / `max_mutations` define in `run_ph_redesign`, except that
a site can now be a **target** residue (`k` caps each site's neighbours, `max_mutations` the total — one cap on
the shared set when there are several target states).

**Free-state reference.** `run.site_report(best)` → one row per condition and titratable site:

| column | meaning |
|--------|---------|
| `state_token` | the state the condition imposes (`HIS-S`, `ASP-P`, …) |
| `gap_complex` · `gap_free` | protonated − deprotonated energy of the site in the complex · in the free partners |
| `binding_gap` | their difference; **negative** = binding favours the protonated state → its apparent pKa **rises** on binding |
| `apparent_pka_free` · `_bound` · `_shift` | `linked_equilibrium` only — pKa implied by the protonated fraction at the condition's pH |

All values are in model units (`beta` is uncalibrated).

**Several target states.** `engine.run_switch_design_ensemble(structures={"human": aa_h, "mouse": aa_m},
binder_chain="A", config=config)` designs **one binder against several complexes**. The binder must be the
same in every complex (same residues, same sequence) and the sites named in the conditions must exist in each,
with the same numbering. A binder residue that is not designed keeps the token of its own complex, so a
fixed His labelled `HIS-P` in one structure and `HIS-S` in another is scored in each as labelled. A state in
which a `near` site has no binder neighbours still counts; only an empty union over states is an error.
Every term is evaluated in every state, then combined by `target_reduce`:

| `target_reduce` | a term takes the value of … |
|-----------------|-----------------------------|
| `max` (default) | the **worst** state → the design must hold up in all of them |
| `mean` | the **average** over states |

The reduction is per term, so the worst state for `potency` need not be the worst for `switch`. Each record
keeps the per-state values (`record.per_state`, `term_<name>@<state>` columns in `to_rows`), and `site_report`
covers every state.

**Seeds.** `seed=` on `run_switch_design` / `run_switch_design_ensemble`, when given, overrides `base_seed` in the
config (an explicit `0` too); the search draws from `base_seed + 1000·run + seed_index`, so the same seed gives
the same designs.

**Legacy vocabularies.** In a v3/v4-style vocabulary a state can name several tokens (two neutral His tautomers,
`HID` / `HIE`): name them explicitly in the conditions, and tell the report how to compare them with
`"report_tokens": {"HIS": {"deprotonated": ["HID", "HIE"]}}` — a list is averaged (as `dep_map` did), a single
name picks one token. Without it `site_report` stops and says the state is ambiguous.

**Protonation source (PROPKA, opt-in).** EV6 and one pKa constant per residue type are the defaults. Two
independent settings under `"protonation"` in the config let PROPKA take over two different jobs; neither is on
unless you ask, and the engine logs which source it uses.

| setting | default | `"propka"` |
|---|---|---|
| `states` | `ev6`: the geometric classifier the checkpoint was trained with, one call per residue, independent of pH | the starting state of every His, Asp and Glu that no condition names comes from PROPKA's pKa at the pH `ph` (required): protonated if pKa > pH + `margin`, neutral if pKa < pH − `margin`, else ambiguous (`-A`); `margin` defaults to 0.5. HBPLUS is not needed: the H-bond passes only feed EV6, so they are skipped |
| `reference_pka` | `constant`: one value per residue type (Asp 3.7, Glu 4.3, His 6.5) | `linked_equilibrium` only: the pKa of each site it sums over comes from PROPKA run on the free binder and the free receptor. `binding_options.pka_by_site` (`{"A:32": 5.9}`) sets single sites by hand and wins |

For example `"protonation": {"states": "propka", "ph": 7.4, "reference_pka": "propka"}`; `reference_pka` needs
`"binding_model": "linked_equilibrium"` and the config refuses it with any other model. `run_ph_redesign(...,
protonation={"states": "propka", "ph": 7.4})` takes `states` only. Residues named in a condition keep the state the
condition gives them whatever `states` is. PROPKA is an optional install (`uv pip install propka`); without it
the run stops at once with that instruction.

Logs use Python's `logging`. `INFO` lines state what was used, in plain words:

```text
Protonation sources: starting states PROPKA (pH 7.4, margin 0.5), reference pKa PROPKA on the free partners.
PROPKA states for chain(s) A,B at pH 7.40, margin 0.50: ASP 9 deprotonated, GLU 9 deprotonated, HIS 3 neutral.
Reference pKa of A:32 (HIS): 5.53 from PROPKA on the free partner; the per-type constant is 6.5.
```

`WARNING` lines are short sentences about what to be careful with, the caveats below: once per Python process
"Starting states come from PROPKA, but the checkpoint was trained on EV6 labels. Against experiment PROPKA labels
are weak …", and each time PROPKA's His values enter the model "PROPKA's reference pKa for His did not beat the
per-type constant on PKAD …". With no logging set up, Python prints WARNING records as bare lines on stderr and
drops INFO ones; `logging.basicConfig(); logging.getLogger("mpnn").setLevel(logging.INFO)` shows both with their
level (`inference/design_switch.py` does this).

Caveats, each measured by `benchmarks/propka_vs_experiment.py`:

- **`states`.** The checkpoint was trained on EV6 labels, and PROPKA is a weak labeller against experiment. On
  217 neutron structures (8071 residues) the ROC AUC of its pKa is 0.62 for His and 0.78 for Asp/Glu, and at pH 7
  its His labels are no better than calling every His neutral. EV6 does far better, but it was trained on those
  structures, so the comparison is not fair. What the setting adds is the dependence on pH. A warning says this,
  once per Python process.
- **`reference_pka`.** PROPKA beats the constant for Asp (PKAD, 221 residues: Pearson r 0.81, mean error 0.71
  against 1.04), only slightly for Glu (254: r 0.71, 0.65 against 0.70), and not for His (171: r 0.15, 1.03
  against 0.80), the residue of most pH switches; a warning appears whenever PROPKA's His values enter the model.
  The free partners are used because the pKa shift on binding is already in the Potts energies, and PROPKA on the
  complex would count it twice (the example binder His A:32: 5.5 free, 3.4 in the complex). The Potts energies are
  not calibrated (`beta`), so apparent pKa shifts are in model units.
- **Both.** PROPKA reads the structure as given and drops ligands, metals and waters. Its values are only as good
  as the backbone and side chains it is shown, which matters for designed or predicted binders.

Unknown config keys raise instead of being ignored, and so do unknown keys inside a condition (`states`, `ph`,
`label` — a typo such as `state` is an error, not an empty assignment). A switch term whose `on` and `off`
conditions (its own, when it names them) impose identical states is refused, since it would be zero for every
design; with `linked_equilibrium`, which sums over every imposed site, only a difference in pH counts.
Weights must be finite. Designed positions
take only non-titratable residues — and every returned design respects that, including any native token at a
designable position — unless `allow_bare_titratable` lists a parent (bare His/Asp/Glu mean "state
unspecified"). Tests:

```bash
cd foundry/models/mpnn && PYTHONPATH=src python -m pytest tests/ph --confcutdir=tests/ph -q
```

Most of these run on synthetic Potts tables. `tests/ph/test_real_structure.py` (needs `HBPLUS_PATH`) and
`tests/ph/test_real_structure_propka.py` (needs propka; it removes HBPLUS from its environment) run the engine on
`inference/examples/pdl1_seed_binder.pdb` with the packaged checkpoint (CPU, and CUDA when present). Each test is
skipped when atomworks, one of those, or the checkpoint is missing.

---

## Layout

| folder | what it holds |
|--------|---------------|
| [`foundry/`](foundry/) | a full verbatim copy of the `ph/foundry` monorepo. The `mpnn` package is the deliverable: the model (`model/pottsmpnn.py`), transforms, the **deployed FLAML labeller** (`transforms/ev6/`), the **design engine** (`inference_engines/potts_mpnn_ph.py`), and `prepare_potts_input` (`potts_inference.py`). `models/{rf3,rfd3,rfd3na}` come along but their multi-GB weights are not shipped. |
| [`labeller/`](labeller/) | the FLAML protonation labeller: `label_pdb.py` (example above), the train path `01…05_*.py`, `pr_curve.py` (AUPR), and the trained models. |
| [`inference/`](inference/) | `design_ph.ipynb` / `.py` (example above) + `design_placement_scan.py` (the engine's per-design placement plot) + `fold_rf3.py` (fold a design with the packaged RF3 engine; needs `RF3_CKPT` + GPU) + `design_switch.py` (switch design from the command line) + `examples/` (a PD-L1 seed binder + a real RF3 fold). |
| [`scoring/`](scoring/) | fold read-outs: `charge_clash` (geometry) + `annotate` (pH-sensitive H-bonds / salt bridges via HBPLUS+PLIP). `score_example.py` is a runnable demo. |
| [`benchmarks/`](benchmarks/) | PKAD (pKa), MegaScale/FireProt (stability), binding AP, within-backbone and `placement_by_class.py` (library-wide placement propensity). The data of all but within-backbone and `placement_by_class.py` is in-folder (see below). |
| [`training/`](training/) | PottsMPNN + H-bond-head SLURM launchers — **reference** (cluster-specific; the trainer itself is `mpnn.train`). |
| [`checkpoints/`](checkpoints/) | the v6 design checkpoint (21 MB). |

### Where the core code lives

| component | path |
|-----------|------|
| **PottsMPNN implementation** (encoder → Potts field + pairwise couplings) | [`model/pottsmpnn.py`](foundry/models/mpnn/src/mpnn/model/pottsmpnn.py) · loss: [`loss/potts_loss.py`](foundry/models/mpnn/src/mpnn/loss/potts_loss.py) · train entry: [`train.py`](foundry/models/mpnn/src/mpnn/train.py) |
| **Transformation pipeline** (structure → model input: parse, HBPLUS/SASA features, protonation-vocab encoding) | [`potts_inference.py`](foundry/models/mpnn/src/mpnn/potts_inference.py) (`prepare_potts_input`) · transforms: [`transforms/`](foundry/models/mpnn/src/mpnn/transforms/) (FLAML labeller in [`transforms/ev6/`](foundry/models/mpnn/src/mpnn/transforms/ev6/)) |
| **Inference / design engine** (placement + block-descent pH-redesign) | [`inference_engines/potts_mpnn_ph.py`](foundry/models/mpnn/src/mpnn/inference_engines/potts_mpnn_ph.py) (`PottsMPNNPHEngine`, `PHDesignCriteria`) |
| **Switch design with explicit states** (conditions on either chain, terms, binding models, Pareto, several target states) | [`ph/`](foundry/models/mpnn/src/mpnn/ph/) · entry points `run_switch_design` / `run_switch_design_ensemble` in `potts_mpnn_ph.py` · tests: [`tests/ph/`](foundry/models/mpnn/tests/ph/) |

### Everything else, one command each

| I want to… | run | out |
|------------|-----|-----|
| **label a PDB** | `HBPLUS_PATH=… python labeller/label_pdb.py` | `labeller/outputs/protonation_labels.csv` |
| **design a binder** | `python inference/design_ph.py` | placement / trajectory / Pareto PNGs + `designs*.fasta` / `.tsv` (with protonation states) |
| **switch design, explicit states** | `python inference/design_switch.py`, or `engine.run_switch_design(atom_array=aa, binder_chain="A", config=config)` (see above) | ranked designs: `run.to_rows("A")`, `run.site_report(best)` |
| **PROPKA against experiment** | `python benchmarks/propka_vs_experiment.py pkad` (or `neutron`, slower, needs HBPLUS) | reference-pKa and starting-state scores against PKAD and the neutron structures |
| **test `mpnn.ph`** | `cd foundry/models/mpnn && PYTHONPATH=src python -m pytest tests/ph --confcutdir=tests/ph -q` | pass / fail |
| **retrain the labeller** | `python labeller/05_train_one.py HIS features 1200` | `labeller/models/automl_feature_HIS/` |
| **labeller AUPR** | `python labeller/pr_curve.py` | `labeller/{his,acid}_pr.png` (AP HIS 0.70, acids 0.33) |
| **score a fold** | `HBPLUS_PATH=… python scoring/score_example.py` | pH-bond / charge-clash counts |
| **placement by class** (library-wide) | `python benchmarks/placement_by_class.py` | `benchmarks/placement_by_class.png` |
| **pKa benchmark** | `python -m mpnn.scripts.eval_pkad --ckpt_dir checkpoints/potts_v6_afdb_edge_his0.3_acid0.06 --pkad_csv benchmarks/data/PKAD/PKAD-R-v1.0_2026-04-22T16_0755.441Z.csv --pdb_dir benchmarks/data/PKAD/pdb_cache --extended_vocab v6 --output_dir DIR` | `pkad_*.csv` + scatter in `DIR` |
| **stability benchmark** | `EV6_OUT_SUBDIR=his0.3_acid0.06 python benchmarks/stability_benchmark.py` | ΔΔG CSVs (GPU recommended) |
| **3-way bar plot** | `EV6_OUT_SUBDIR=his0.3_acid0.06 python benchmarks/benchmark_barplot.py` | recovery + MegaScale + FireProt figure |
| **binding AP** | `python benchmarks/binding_ap.py` | `benchmarks/results/summary_global_ap.png` |
| **within-backbone** | `python benchmarks/within_backbone.py` | `within_backbone_inverse_outcomes.png` |

Benchmark scripts derive the package root from their location (override with `PROTON_ROOT=/abs/path`), except
`eval_pkad`, whose default paths point outside this repository: pass the flags above, including
`--extended_vocab v6` for the shipped checkpoint (the default `v4` builds a 32-token model and the weight load
fails). On the 105 usable PKAD structures it takes about 8 minutes on a CPU. The model-quality benchmarks (PKAD,
stability) compute from raw structures. Of the design benchmarks only binding AP reads a shipped parquet
(`mb_gbind_benchmark.parquet`); `within_backbone.py` needs `egfr_pdl1_*.parquet`, `optimized_sweep_scores.parquet`
and `curated_designs.parquet`, and `placement_by_class.py` needs `data/placement_scan.parquet`. None of those is in
the repo, and regenerating them needs the Boltz-2 / RF3 oracles.

---

## License

This project is released under the **MIT License** — see [`LICENSE`](LICENSE). It covers the
Proton-PottsMPNN code authored here (`labeller/`, `inference/`, `scoring/`, `benchmarks/`, `training/`,
`checkpoints/`, the docs, and the pH-design additions to the `mpnn` package). The bundled [`foundry/`](foundry/)
is a verbatim copy of IPD's rc-foundry and keeps its own **BSD 3-Clause License**
([`foundry/LICENSE.md`](foundry/LICENSE.md), © 2025 Institute for Protein Design, University of Washington).

If you use this in academic work, please cite the manuscript *"pH-sensitive binder design with
Proton-PottsMPNN"* (Jacobsen et al., 2026).
