"""Per-design placement scan — where the pipeline WOULD place each protonation state on ONE binder.

For every free binder position, the SELECTIVE PLACEMENT SCORE of each state (Potts field:
ef[pos, STATE] − min_dep ef[pos, DEP]; lower = preferred), plus the state-agnostic ΔE / substitution-cost
scan (higher = substitution is costly). A star marks each state's top-1 placement; the track below colours
every residue by structural class (interface / core / surface).

This is `example_placement_scores.py`, as a function the design notebook calls for its example binder.
Run standalone on the packaged example:  python placement/design_placement_scan.py
"""
import numpy as np

STATES = {"HIS-P": ["HIS-S"], "HIS-S": ["HIS-P"], "ASP-P": ["ASP-D"], "GLU-P": ["GLU-D"]}
STATE_ORDER = ["HIS-P", "HIS-S", "ASP-P", "GLU-P"]
STATE_COL = {"HIS-P": "#4153A4", "HIS-S": "#189486", "ASP-P": "#a3002e", "GLU-P": "#FF477B"}  # manuscript palette
STATE_STYLE = {"HIS-P": "-", "HIS-S": "--", "ASP-P": "-", "GLU-P": "-"}
CLASSES = ["interface", "core", "surface"]
CLS_COL = {"interface": "#a3002e", "core": "#4153A4", "surface": "#FF477B"}   # band: red / blue / lighter red
DE_COL = "0.6"
CANON = ["ALA", "ARG", "ASN", "CYS", "GLN", "GLY", "ILE", "LEU", "LYS", "MET",
         "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "HIS-S", "ASP-D", "GLU-D"]
IFACE_DIST, RASA_THR = 8.0, 0.2


def classify_binder(structure, bc):
    """{res_id: 'interface'|'core'|'surface'} for binder chain bc (biotite, table-free)."""
    import biotite.structure as struc
    from biotite.structure import CellList
    b = structure[structure.chain_id == bc]; t = structure[structure.chain_id != bc]
    bca, tca = b[b.atom_name == "CA"], t[t.atom_name == "CA"]
    iface = set()
    if tca.array_length() and bca.array_length():
        cont = CellList(tca, cell_size=IFACE_DIST).get_atoms(bca.coord, radius=IFACE_DIST)
        hit = (np.asarray(cont) != -1).reshape(bca.array_length(), -1).any(axis=1)
        iface = {int(r) for r, h in zip(bca.res_id, hit) if h}
    sasa = struc.sasa(b, point_number=200)
    starts = list(struc.get_residue_starts(b)) + [b.array_length()]
    res_sasa = np.array([np.nansum(sasa[starts[i]:starts[i + 1]]) for i in range(len(starts) - 1)], float)
    res_ids = [int(b.res_id[starts[i]]) for i in range(len(starts) - 1)]
    mx = np.nanmax(res_sasa) if np.isfinite(res_sasa).any() and np.nanmax(res_sasa) > 0 else 1.0
    rasa = res_sasa / mx
    return {rid: ("interface" if rid in iface else ("core" if ra < RASA_THR else "surface"))
            for rid, ra in zip(res_ids, rasa)}


def plot_design_placement(engine, atom_array, binder_chain, fig=None):
    """Image-#62 placement scan for ONE binder. Returns (fig, {state: (res_id, class, score)} top-1 placement)."""
    import torch
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    ctx = engine._build_context(atom_array, binder_chain)
    t2i = ctx.encoding.token_to_idx
    region = ctx.chA_free_idx
    res_ids = np.array([int(ctx.token_aa.res_id[int(p)]) for p in region.tolist()])

    # Structural class from THE ENGINE'S OWN region masks (ctx.region_masks), so the class track under
    # the scan is the same partition `placement_region=["core"|"interface"|"surface"]` actually selects.
    # The local `classify_binder` below disagrees with the engine on both criteria -- it normalises SASA
    # by the most exposed residue in THIS binder rather than by the residue type's Tien et al. maximum,
    # and uses an 8 A interface cutoff against the engine's 6 A -- so colouring the figure with it could
    # label a residue "core" that the engine would never draw from the core region. Kept as the
    # standalone fallback for when the engine's SASA classification failed (it is fail-soft and can
    # return empty masks).
    _rm = getattr(ctx, "region_masks", None) or {}
    if any(m is not None and bool(m.any()) for m in _rm.values()):
        def _cls_at(p):                       # precedence interface > core > surface, as before
            for name in ("interface", "core"):
                m = _rm.get(name)
                if m is not None and bool(m[int(p)]):
                    return name
            return "surface"
        pos_cls = np.array([_cls_at(p) for p in region.tolist()])
    else:
        cls = classify_binder(atom_array, binder_chain)
        pos_cls = np.array([cls.get(int(r), "surface") for r in res_ids])

    seq = ctx.S_native.clone()
    with torch.no_grad():
        ef = ctx.field_potts(seq)                              # [L, V]
    scores = {}
    for stt, deps in STATES.items():
        prot = ef[region, t2i[stt]]
        dep = torch.stack([ef[region, t2i[d]] for d in deps], 0).min(0).values
        scores[stt] = (prot - dep).cpu().numpy()               # base cancels in the selective score
    base = ef[region, seq[region]]
    canon_idx = [t2i[a] for a in CANON if a in t2i]
    imp = (ef[region][:, canon_idx] - base[:, None]).mean(1).cpu().numpy()   # ΔE (substitution cost)

    if fig is None:
        fig = plt.figure(figsize=(11, 5.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.045], hspace=0.06)
    ax = fig.add_subplot(gs[0]); axs = fig.add_subplot(gs[1], sharex=ax)
    xmin, xmax = res_ids.min() - 0.5, res_ids.max() + 0.5
    ax.axhline(0, color="0.8", lw=0.8, zorder=0)
    top = {}
    for stt in STATE_ORDER:
        y = scores[stt]
        ax.plot(res_ids, y, STATE_STYLE[stt], color=STATE_COL[stt], lw=2.0, zorder=3, label=stt)
        k = int(np.argmin(y))                                  # placement = minimum score
        ax.plot(res_ids[k], y[k], "*", color=STATE_COL[stt], ms=16, mec="black", mew=0.8, zorder=5)
        top[stt] = (int(res_ids[k]), str(pos_cls[k]), float(y[k]))
    ax.set_ylabel("Selective placement score\n(lower = preferred)", labelpad=2)
    ax.grid(axis="y", color="0.92", lw=0.8); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelbottom=False); ax.set_xlim(xmin, xmax)

    axr = ax.twinx()
    axr.plot(res_ids, imp, color=DE_COL, ls=":", lw=1.2, alpha=0.85, zorder=2)
    ki = int(np.argmax(imp))
    axr.plot(res_ids[ki], imp[ki], "D", color=DE_COL, ms=7.5, mec="white", mew=0.7, zorder=6)
    axr.set_ylabel("Substitution $\\Delta E$\n(higher = substitution is costly)", labelpad=2)
    axr.spines[["top"]].set_visible(False); axr.set_xlim(xmin, xmax)
    ax.set_zorder(axr.get_zorder() + 1); ax.patch.set_visible(False)

    axs.bar(res_ids, np.ones(len(res_ids)), width=1.0, color=[CLS_COL[c] for c in pos_cls], edgecolor="none")
    axs.set_yticks([]); axs.set_ylim(0, 1); axs.set_xlim(xmin, xmax); axs.set_xlabel("Binder residue")
    axs.spines[["top", "right", "left"]].set_visible(False)

    type_h = [Line2D([0], [0], color=STATE_COL[s], ls=STATE_STYLE[s], lw=2.0, label=s) for s in STATE_ORDER]
    type_h.append(Line2D([0], [0], color=DE_COL, ls=":", lw=1.2, label=r"$\Delta E$ (right axis)"))
    cls_h = [Patch(facecolor=CLS_COL[c], label=c.capitalize()) for c in CLASSES]
    ax.legend(handles=type_h, ncol=5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=1.4)
    axs.legend(handles=cls_h, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -2.2), handlelength=1.2)
    return fig, top


if __name__ == "__main__":
    import os, sys, json
    from pathlib import Path
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.pop("DEBUG", None)
    PKG = Path(__file__).resolve().parents[1]
    import matplotlib; matplotlib.use("Agg")
    from biotite.structure.io.pdb import PDBFile
    from mpnn.inference_engines.potts_mpnn_ph import PottsMPNNPHEngine

    meta = json.loads((PKG / "inference" / "examples" / "example_meta.json").read_text())
    eng = PottsMPNNPHEngine(
        checkpoint_path=str(PKG / "checkpoints" / "potts_v6_afdb_edge_his0.3_acid0.06" / "epoch-0125.ckpt"),
        extended_vocab="v6", out_directory=None, write_fasta=False, write_structures=False)
    aa = PDBFile.read(str(PKG / "inference" / "examples" / meta["pdb"])).get_structure(model=1)
    fig, top = plot_design_placement(eng, aa, meta["binder_chain"])
    out = PKG / "placement" / "design_placement_scan.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    for stt, (rid, c, sc) in top.items():
        print(f"  {stt:6s} top placement: res {rid:4d}  class={c:9s}  score={sc:.3f}")
    print(f"saved -> {out}")
