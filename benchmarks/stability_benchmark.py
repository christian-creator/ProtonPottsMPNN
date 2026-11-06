# %% [markdown]
# ONE-OFF stability benchmark (NOT part of training). For MegaScale + FireProt single-mutation ΔΔG,
# correlate three predictors' ΔΔG proxies against experimental ddG_expt:
#   Potts head   : full Potts-Hamiltonian ΔΔG = H(mut) - H(wt)  (calc_potts_eners) — the native energy score
#   PottsMPNN MPNN head : decoder masked-marginal proxy = logP(wt@pos) - logP(mut@pos)
#   ProteinMPNN  : decoder masked-marginal proxy = logP(wt@pos) - logP(mut@pos)  (classic 21-token weights)
# All decoder marginals use teacher_forcing + causality_pattern="conditional_minus_self" (each position
# scored from the full true context of every OTHER residue), the same protocol as the recovery benchmark.
# Reuses the exact featurization + mutant-building the MegaScale training callback uses.

# %%
import sys, gc
import os
from pathlib import Path
import numpy as np, pandas as pd, torch
import contextlib, os
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

os.environ.pop("DEBUG", None)                      # a stray non-boolean DEBUG breaks environs/foundry
_P = Path(os.environ.get("PROTON_ROOT", Path(__file__).resolve().parents[1]))   # mpnn/foundry from the venv
from atomworks.io.parser import STANDARD_PARSER_ARGS
from atomworks.ml.datasets.pandas_dataset import PandasDataset, StructuralDatasetWrapper
from atomworks.ml.datasets.parsers.default_metadata_row_parsers import GenericDFParser
from mpnn.pipelines.potts_mpnn import build_mpnn_transform_pipeline as build_potts_pipe
from mpnn.pipelines.mpnn import build_mpnn_transform_pipeline as build_std_pipe
from mpnn.transforms.extended_vocab import get_vocab
from mpnn.model.pottsmpnn import PottsMPNN
from mpnn.model.layers.graph_embeddings import PottsProteinFeatures
from mpnn.inference_engines.mpnn import MPNNInferenceEngine
from mpnn.callbacks.megascale_benchmark import MegaScaleEnergy

gc.disable()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXTENDED_VOCAB = os.environ.get("EV6_VOCAB", "v6")   # MUST match EV6_MODEL_DIR's checkpoint: it sizes the token encoding AND labels the features
POTTS_CKPT = Path(os.environ.get("POTTS_CKPT", _P / "checkpoints/potts_v6_afdb_edge_his0.3_acid0.06/epoch-0125.ckpt"))
PROTEINMPNN_CKPT = os.environ.get("PROTEINMPNN_CKPT", "/novo/projects/departments/rdd/cpjb/foundry/model_ckpts/proteinmpnn_v_48_020.pt")
# Results go to benchmarks/results/<EV6_OUT_SUBDIR>/, by default the shipped checkpoint's folder, so a plain
# run overwrites the tracked results. Set EV6_OUT_SUBDIR (with POTTS_CKPT) to benchmark another model.
OUT_DIR = _P / "benchmarks/results" / os.environ.get("EV6_OUT_SUBDIR", "his0.3_acid0.06")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATASETS = {
    "fireprot":  (_P / "benchmarks/data/energy_benchmark_datasets/fireprot_test_subset.csv",
                  _P / "benchmarks/data/energy_benchmark_datasets/fireprot_pdbs", None),
    "megascale": (_P / "benchmarks/data/energy_benchmark_datasets/megascale_test_subset.csv",
                  _P / "benchmarks/data/energy_benchmark_datasets/megascale_pdbs", 150),   # cap #PDBs for runtime
}

# ── models ────────────────────────────────────────────────────────────────────────
_state = torch.load(POTTS_CKPT, map_location="cpu", weights_only=False)["model"]
potts = PottsMPNN(graph_featurization_module=PottsProteinFeatures(token_encoding=get_vocab(EXTENDED_VOCAB)["token_encoding"]),
                  etab_source=PottsMPNN.infer_etab_source(_state))
potts.load_state_dict(_state, strict=True); potts.eval().to(DEVICE)
eng = MPNNInferenceEngine(model_type="protein_mpnn", checkpoint_path=PROTEINMPNN_CKPT, is_legacy_weights=True,
                          out_directory=None, write_fasta=False, write_structures=False, device=DEVICE)
pmpnn = eng.model.eval()
o2i_potts = MegaScaleEnergy.build_one_to_foundry_int(potts.token_to_idx)
o2i_pmpnn = MegaScaleEnergy.build_one_to_foundry_int(pmpnn.token_to_idx)
print(f"loaded potts {POTTS_CKPT.name} ({potts.potts_vocab_size} tok) + ProteinMPNN on {DEVICE}", flush=True)

potts_pipe = build_potts_pipe(model_type="potts_mpnn", is_inference=True, minimal_return=True,
                              train_structure_noise_default=0.0, extended_vocab=EXTENDED_VOCAB)
std_pipe = build_std_pipe(model_type="protein_mpnn", is_inference=True, minimal_return=True,
                          train_structure_noise_default=0.0)


def _mkds(pdb_ids, pdb_dir, pipe):
    df = pd.DataFrame({"example_id": pdb_ids, "path": [str(pdb_dir / f"{p}.pdb") for p in pdb_ids],
                       "assembly_id": "1"})
    return StructuralDatasetWrapper(
        dataset=PandasDataset(data=df, id_column="example_id", name="stab"),
        dataset_parser=GenericDFParser(example_id_colname="example_id", path_colname="path",
                                       assembly_id_colname="assembly_id"),
        transform=pipe,
        cif_parser_args={**STANDARD_PARSER_ARGS, "add_bond_types_from_struct_conn": (),
                         "load_from_cache": False, "save_to_cache": False, "cache_dir": None})


@torch.no_grad()
def _forward(model, ds, idx):
    with open(os.devnull, "w") as _n, contextlib.redirect_stdout(_n), contextlib.redirect_stderr(_n):
        od = ds[idx]
    inp = {k: (v.unsqueeze(0).to(DEVICE) if torch.is_tensor(v) else v) for k, v in od["input_features"].items()}
    inp["decode_type"] = "teacher_forcing"; inp["causality_pattern"] = "conditional_minus_self"
    out = model({"input_features": inp})
    lp = out["decoder_features"]["log_probs"][0]                      # [L, V]
    S = inp["S"][0]                                                   # [L]
    ctx = out.get("potts_context") if isinstance(out, dict) else getattr(out, "potts_context", None)
    return S, lp, ctx


# %%
def run_dataset(name, csv_path, pdb_dir, max_pdbs):
    out_csv = OUT_DIR / f"stability_{name}.csv"
    mut = pd.read_csv(csv_path)
    groups = {pid: g for pid, g in mut.groupby("pdb")}
    pdb_ids = list(groups.keys())
    if max_pdbs:
        pdb_ids = pdb_ids[:max_pdbs]
    ds_p = _mkds(pdb_ids, pdb_dir, potts_pipe)
    ds_m = _mkds(pdb_ids, pdb_dir, std_pipe)

    rows, done = [], set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv); rows = prev.to_dict("records")
        done = set(prev.pdb.astype(str)); print(f"[{name}] resume: {len(done)} pdbs done", flush=True)

    for i, pid in enumerate(tqdm(pdb_ids, desc=name)):
        if str(pid) in done:
            continue
        try:
            Sp, lp_p, ctx = _forward(potts, ds_p, i)
            Sm, lp_m, _ = _forward(pmpnn, ds_m, i)
        except Exception as e:
            tqdm.write(f"  [{name}] skip {pid}: {type(e).__name__}: {str(e)[:70]}"); continue
        etab, E_idx = ctx.etab_out, ctx.E_idx
        Lp, Lm = Sp.shape[0], Sm.shape[0]

        valid, mut_seqs = [], []
        for _, r in groups[pid].iterrows():
            try:
                pos, wt, mt = MegaScaleEnergy.parse_mut_type(r["mut_type"])
            except ValueError:
                continue
            wp, mp = o2i_potts.get(wt), o2i_potts.get(mt)
            wm, mm = o2i_pmpnn.get(wt), o2i_pmpnn.get(mt)
            if None in (wp, mp, wm, mm):
                continue
            if pos >= Lp or pos >= Lm:
                continue
            if int(Sp[pos]) != wp or int(Sm[pos]) != wm:            # sequence must match at the mutated pos
                continue
            valid.append((pos, wp, mp, wm, mm, float(r["ddG_expt"])))
            mut_seqs.append(MegaScaleEnergy.build_mutant_seq(Sp, pos, mt, o2i_potts))
        if not valid:
            continue

        mut_batch = torch.stack(mut_seqs)                            # [N, Lp]
        e_wt = PottsMPNN.calc_potts_eners(etab, E_idx, Sp.unsqueeze(0))     # [1]
        e_mut = PottsMPNN.calc_potts_eners(etab, E_idx, mut_batch)          # [N]
        d_potts_e = (e_mut - e_wt).cpu().numpy()
        for j, (pos, wp, mp, wm, mm, ddg) in enumerate(valid):
            rows.append(dict(
                pdb=str(pid), ddg_expt=ddg,
                potts_energy=float(d_potts_e[j]),
                potts_decoder=float(lp_p[pos, wp] - lp_p[pos, mp]),
                proteinmpnn=float(lp_m[pos, wm] - lp_m[pos, mm])))
        done.add(str(pid))
        pd.DataFrame(rows).to_csv(out_csv, index=False)             # incremental
        if len(done) % 20 == 0:
            d = pd.DataFrame(rows)
            tqdm.write(f"  [{name}] {len(done)} pdbs, {len(d)} muts | "
                       f"|r| potts_e={abs(pearsonr(d.potts_energy, d.ddg_expt)[0]):.3f} "
                       f"dec={abs(pearsonr(d.potts_decoder, d.ddg_expt)[0]):.3f} "
                       f"pmpnn={abs(pearsonr(d.proteinmpnn, d.ddg_expt)[0]):.3f}")

    D = pd.DataFrame(rows)
    summ = {}
    for col in ["potts_energy", "potts_decoder", "proteinmpnn"]:
        pr = pearsonr(D[col], D.ddg_expt); sr = spearmanr(D[col], D.ddg_expt)
        summ[col] = dict(pearson=pr[0], abs_pearson=abs(pr[0]), spearman=sr[0], abs_spearman=abs(sr[0]))
    pd.DataFrame(summ).T.to_csv(OUT_DIR / f"stability_{name}_summary.csv")
    print(f"\n=== {name}: {len(D)} mutations, {D.pdb.nunique()} pdbs ===", flush=True)
    for col in ["potts_energy", "potts_decoder", "proteinmpnn"]:
        print(f"  {col:14s} |Pearson|={summ[col]['abs_pearson']:.3f}  |Spearman|={summ[col]['abs_spearman']:.3f}")
    return D


# %%
for name, (csv_path, pdb_dir, max_pdbs) in DATASETS.items():   # fireprot first (small), then megascale
    run_dataset(name, csv_path, pdb_dir, max_pdbs)
print("\nDONE stability benchmark", flush=True)
