"""Build the precomputed-annotation cache for PottsMPNN training.

For every POST-FILTER training example_id, run the offline pipeline (parse -> cleanup -> p=1 multimer crop
-> HBPLUS/EV6 annotation) and write the resulting `data` snapshot to `--precomputed-dir`, keyed by
example_id. Training then sets MPNN_PRECOMPUTED_DIR to this dir and skips the per-epoch annotation.

The filters MUST match train.py exactly (the user wants only the observations actually trained on). They
are replicated here from mpnn/train.py -- keep the two in sync. One dataset per invocation:
  --dataset pdb   : data/mpnn_split/train_df_filtered.parquet, X-ray/EM, resolution+date gated.
  --dataset afdb  : afdb_highquality_sample.parquet, method=ALPHAFOLD (COMMON_FILTERS only).

Resumable: an example_id whose snapshot already exists is skipped, so re-running a failed shard is cheap.
Shard with --shard-id / --n-shards (strided, for load balance across structure sizes).
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

gc.disable()   # atomworks leaks cyclic garbage each iteration; the periodic sweep can stall the loop ~100s

import pandas as pd
from atomworks.io.parser import STANDARD_PARSER_ARGS
from atomworks.ml.datasets.pandas_dataset import PandasDataset, StructuralDatasetWrapper
from atomworks.ml.datasets.parsers.default_metadata_row_parsers import GenericDFParser

from mpnn.pipelines.potts_mpnn import build_precompute_pipeline
from mpnn.transforms.precomputed import snapshot_path

# ── training filters, mirrored from mpnn/train.py (batch_size=10000 for both pdb and afdb_pdb) ──────
BATCH_SIZE = 10000
COMMON_FILTERS = [
    "n_non_atomized_tokens >= 30",
    f"n_non_atomized_tokens < {BATCH_SIZE}",
    "cluster.notnull() and cluster != 'nan'",
    "n_prot >= 1",
    "assembly_id == '1'",
]
PDB_QUALITY_FILTERS = [
    "resolution < 3.5 and ~method.str.contains('NMR')",
    "method in ['X-RAY_DIFFRACTION', 'ELECTRON_MICROSCOPY']",
]
DATASETS = {
    "pdb": dict(
        parquet="data/mpnn_split/train_df_filtered.parquet",
        # MPNN_TRAIN_FILTERS = [deposition_date < cutoff] + PDB_QUALITY + COMMON
        filters=["deposition_date < '2021-08-02'"] + PDB_QUALITY_FILTERS + COMMON_FILTERS,
    ),
    "afdb": dict(
        parquet="/novo/users/cpjb/rdd/cpjb/afdb_complexes/afdb_highquality_sample.parquet",
        # AFDB rows are all method==ALPHAFOLD, so the afdb_pdb OR-clauses collapse to COMMON_FILTERS.
        filters=["method == 'ALPHAFOLD'"] + COMMON_FILTERS,
    ),
    # --- top-ups so the afdb_pdb (PDB+AFDB) training/validation is FULLY precomputed ---
    # afdb_pdb uses the LATER PDB cutoff (2022-12-16), so its PDB train includes the 2021-08..2022-12 slice
    # the "pdb" build (2021-08-02 cutoff) missed. Resumable, so this only builds the ~50k new example_ids.
    "pdb_afdbpdb_train": dict(
        parquet="data/mpnn_split/train_df_filtered.parquet",
        filters=["deposition_date < '2022-12-16'"] + PDB_QUALITY_FILTERS + COMMON_FILTERS,
    ),
    # PDB validation was never cached (val is post-cutoff / a separate split). No date filter -- the val
    # split is already made; just the quality + common gates the afdb_pdb val applies to PDB rows.
    "pdb_val": dict(
        parquet="data/mpnn_split/val_df_filtered.parquet",
        filters=PDB_QUALITY_FILTERS + COMMON_FILTERS,
    ),
}


def build_dataset(parquet: str, filters: list[str], precomputed_dir: str, extended_vocab: str):
    manifest = pd.read_parquet(parquet)
    ds = StructuralDatasetWrapper(
        dataset=PandasDataset(data=manifest, id_column="example_id", name="precompute", filters=filters),
        dataset_parser=GenericDFParser(example_id_colname="example_id", path_colname="path",
                                       assembly_id_colname="assembly_id"),
        transform=build_precompute_pipeline(extended_vocab=extended_vocab, precomputed_dir=precomputed_dir,
                                            deterministic=True, complex_max_atoms=BATCH_SIZE,
                                            complex_min_atom_contacts=10),
        cif_parser_args={**STANDARD_PARSER_ARGS, "add_bond_types_from_struct_conn": (),
                         "load_from_cache": False, "save_to_cache": False, "cache_dir": None},
    )
    return ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--precomputed-dir", required=True)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--extended-vocab", default="v6")
    ap.add_argument("--limit", type=int, default=0, help="stop after N structures (smoke test); 0 = all")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    ds = build_dataset(cfg["parquet"], cfg["filters"], args.precomputed_dir, args.extended_vocab)
    n = len(ds)
    mine = [i for i in range(n) if i % args.n_shards == args.shard_id]
    if args.limit:
        mine = mine[: args.limit]
    print(f"[{args.dataset} shard {args.shard_id}/{args.n_shards}] {n:,} filtered examples, "
          f"{len(mine):,} in this shard -> {args.precomputed_dir}", flush=True)

    ok = skip = fail = 0
    t0 = time.perf_counter()
    for k, i in enumerate(mine, 1):
        example_id = ds.idx_to_id(i)
        if snapshot_path(args.precomputed_dir, example_id).exists():
            skip += 1
        else:
            try:
                ds[i]                       # parse -> cleanup -> crop -> annotate -> SaveAnnotationSnapshot
                ok += 1
            except Exception as e:
                fail += 1
                if fail <= 20:
                    print(f"  FAIL {example_id}: {type(e).__name__}: {str(e)[:120]}", flush=True)
        if k % args.log_every == 0:
            el = time.perf_counter() - t0
            print(f"  {k}/{len(mine)}  ok={ok} skip={skip} fail={fail}  {el:.0f}s "
                  f"({el/max(ok,1):.2f}s/built)", flush=True)

    el = time.perf_counter() - t0
    print(f"[{args.dataset} shard {args.shard_id}] DONE  ok={ok} skip={skip} fail={fail}  {el:.0f}s", flush=True)


if __name__ == "__main__":
    main()
