# %% [markdown]
# THE deliverable: grouped bar plot — ProteinMPNN vs PottsMPNN (MPNN head + Potts head).
# Four benchmark groups:
#   Seq. recovery (auto_regressive)         — the two DECODERS only (their native mode); Potts head has no
#                                             auto-regressive mode, so it is not shown here.
#   Seq. recovery (conditional_minus_self)  — all three, full-context masked prediction (equal information;
#                                             the Potts head's intrinsic mode). Verified: the Potts head
#                                             energy prediction excludes the residue at the position scored.
#   MegaScale ΔΔG   (|Pearson r|)           — physics score, always conditional_minus_self.
#   FireProt  ΔΔG   (|Pearson r|)           — physics score, always conditional_minus_self.
# Manuscript style: colourblind-safe, minimal text. Bar colour = model; groups may hold 2 or 3 models.

# %%
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# EV6_OUT_SUBDIR selects WHICH benchmarked model to plot; the figure is written with a matching suffix
# so each model gets its own file. Unset = the shipped checkpoint (his0.3_acid0.06), the same default as
# stability_benchmark.py, so a plain run reads and overwrites the tracked results.
import os, sys
BASE = Path(os.environ.get("PROTON_ROOT", Path(__file__).resolve().parents[1])) / "benchmarks/results"
SUBDIR = os.environ.get("EV6_OUT_SUBDIR", "his0.3_acid0.06")
OUT = BASE / SUBDIR
FIG = BASE / (f"benchmark_3way_{SUBDIR}.png" if SUBDIR else "benchmark_3way.png")
REC, MEGA, FIRE = OUT / "val_recovery_all.csv", OUT / "stability_megascale_summary.csv", OUT / "stability_fireprot_summary.csv"

# model index -> (name, colour)
MODELS = [("ProteinMPNN", "#828080"), ("PottsMPNN — MPNN head", "#4153A4"), ("PottsMPNN — Potts head", "#189486")]


# A missing input used to leave its bars out of the figure without a word; now it stops the plot instead.
_missing = [p for p in (REC, MEGA, FIRE) if not p.exists()]
if _missing:
    sys.exit("missing benchmark results (run val_recovery_benchmark.py and stability_benchmark.py first):\n  "
             + "\n  ".join(map(str, _missing)))


def stab(path, col):
    return float(pd.read_csv(path, index_col=0).loc[col, "abs_pearson"])

d = pd.read_csv(REC)
n_rec = len(d) if d is not None else 0
# each group: (label, [(model_idx, value), ...]) in the model order they should appear
GROUPS = [
    ("Seq. recovery\nauto-regressive", [(0, d.proteinmpnn_canonical_ar.mean()), (1, d.decoder_canonical_ar.mean())] if d is not None else []),
    ("Seq. recovery\ncond. minus-self", [(0, d.proteinmpnn_canonical_cms.mean()), (1, d.decoder_canonical_cms.mean()),
                                          (2, d.potts_canonical.mean())] if d is not None else []),
    ("MegaScale ΔΔG\n|Pearson r|", [(0, stab(MEGA, "proteinmpnn")), (1, stab(MEGA, "potts_decoder")), (2, stab(MEGA, "potts_energy"))]),
    ("FireProt ΔΔG\n|Pearson r|", [(0, stab(FIRE, "proteinmpnn")), (1, stab(FIRE, "potts_decoder")), (2, stab(FIRE, "potts_energy"))]),
]

# %%
w = 0.26
fig, ax = plt.subplots(figsize=(16, 9.0), dpi=300)
seen = set()
for gi, (label, bars) in enumerate(GROUPS):
    k = len(bars)
    for pos, (mi, val) in enumerate(bars):
        xc = gi + (pos - (k - 1) / 2) * w
        name, col = MODELS[mi]
        ax.bar(xc, val, w, color=col, edgecolor="black", linewidth=0.9,
               label=name if mi not in seen else None)
        seen.add(mi)
        if not np.isnan(val):
            ax.text(xc, val + 0.010, f"{val:.2f}", ha="center", va="bottom", fontsize=23)

ax.axvline(1.5, color="0.85", lw=1.0, zorder=0)               # divider: recovery pair | physics pair
ax.set_xticks(range(len(GROUPS))); ax.set_xticklabels([g[0] for g in GROUPS], fontsize=24)
ax.tick_params(axis="y", labelsize=22)
ax.set_ylabel("Benchmark performance", fontsize=24)
allv = [v for _, bars in GROUPS for _, v in bars if not np.isnan(v)]
ax.set_ylim(0, max(0.74, max(allv) * 1.22))
# legend in fixed model order
handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black", linewidth=0.9) for _, c in MODELS]
ax.legend(handles, [n for n, _ in MODELS], frameon=False, fontsize=24, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.09))
ax.spines[["top", "right"]].set_visible(False)
ax.margins(x=0.02)
fig.tight_layout()
fig.savefig(FIG, dpi=300, bbox_inches="tight")
plt.close(fig)
print("wrote", FIG, f"(recovery n={n_rec}, model={SUBDIR or 'v6_afdb_edge'})")
for label, bars in GROUPS:
    print(f"  {label.replace(chr(10),' '):34s} " + "  ".join(f"{MODELS[mi][0].split('—')[-1].strip()}={v:.3f}" for mi, v in bars))
