"""Augment existing PottsMPNN annotation snapshots IN PLACE with the continuous FLAML scores.

The precompute build (scripts/build_snapshots.py) bakes only the discrete `protonation_label` token and
throws away the FLAML ensemble outputs. To make the protonation `prob_thr` a train-time knob (swept via
`ApplyProtonationThreshold`), each snapshot must additionally carry the raw per-residue scores. This script
adds them WITHOUT re-parsing CIFs or re-running HBPLUS: the snapshot already holds the cropped, H-stripped
atom_array AND `data["hbond_records"]`, which is exactly what `EV6Predictor.predict()` consumes. So it just
re-runs the (cheap, cached-input) FLAML pass and writes three new per-atom annotations:

    flaml_p     float32   mean p(protonated) across the 5 folds        (NaN on non-titratable atoms)
    flaml_sd    float32   SD of p(protonated) across the folds         (NaN on non-titratable atoms)
    flaml_metal bool      metal-adjacent (out-of-domain) flag          (False on non-titratable atoms)

Resumable: a snapshot that already has `flaml_p` is skipped, so re-running a failed shard is cheap.
Consistency check: the re-derived DEFAULT-threshold token must equal the already-baked `protonation_label`
(both are thresholds.json operating points); mismatches are counted and the first few printed.

Shard over the cache's 256 top-level hash directories with --shard-id / --n-shards. Writes back to the SAME
directory (atomic tmp+replace), so run it with EV6_HIS_PROB_THR / EV6_ACID_PROB_THR UNSET — the persisted
scores are threshold-independent and the token is pinned to defaults regardless, but keeping the env clean
avoids confusion.

    python scripts/augment_snapshots_flaml.py --precomputed-dir /novo/users/cpjb/rdd/cpjb/ev6_snapshots \
        --shard-id $SLURM_ARRAY_TASK_ID --n-shards 100
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

gc.disable()   # atomworks/biotite leak cyclic garbage; the periodic sweep can stall the loop (see build_snapshots)

import numpy as np

from mpnn.transforms.precomputed import load_snapshot, save_snapshot
from mpnn.transforms.vocab_annotation import AnnotateProtonationStates

_TITRATABLE = {"HIS", "ASP", "GLU"}


def _iter_shard_files(precomputed_dir: Path, shard_id: int, n_shards: int):
    """Yield .pkl snapshot paths in this shard. Sharded by top-level hash dir (256 of them) for balance
    and determinism — a snapshot's dir is fixed by its example_id hash, so shards never overlap."""
    top_dirs = sorted(p for p in precomputed_dir.iterdir() if p.is_dir())
    for idx, top in enumerate(top_dirs):
        if idx % n_shards != shard_id:
            continue
        for pkl in top.rglob("*.pkl"):
            yield pkl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precomputed-dir", required=True)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--extended-vocab", default="v6")
    ap.add_argument("--limit", type=int, default=0, help="stop after N snapshots (smoke test); 0 = all")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    precomputed_dir = Path(args.precomputed_dir)
    # persist_scores=True runs the DEFAULT-threshold predictor (env-independent) and adds flaml_* annotations.
    annotate = AnnotateProtonationStates(extended_vocab=args.extended_vocab, persist_scores=True)

    files = _iter_shard_files(precomputed_dir, args.shard_id, args.n_shards)
    ok = skip = fail = miss = mismatch = 0
    t0 = time.perf_counter()
    k = 0
    for pkl in files:
        k += 1
        if args.limit and ok >= args.limit:
            break
        try:
            with open(pkl, "rb") as fh:
                import pickle
                data = pickle.load(fh)
        except Exception as e:
            fail += 1
            if fail <= 20:
                print(f"  FAIL load {pkl.name}: {type(e).__name__}: {str(e)[:100]}", flush=True)
            continue

        aa = data.get("atom_array")
        if aa is None:
            miss += 1
            continue
        if "flaml_p" in aa.get_annotation_categories():
            skip += 1
            continue

        old_label = (aa.protonation_label.copy()
                     if "protonation_label" in aa.get_annotation_categories() else None)
        try:
            data = annotate.forward(data)
        except Exception as e:
            fail += 1
            if fail <= 20:
                print(f"  FAIL annotate {data.get('example_id', pkl.name)}: "
                      f"{type(e).__name__}: {str(e)[:100]}", flush=True)
            continue

        # Free consistency check: the re-derived default token must match the baked label on titratable
        # residues (both are thresholds.json). Continuous scores are what we actually need; a mismatch only
        # signals the original build ran at a non-default operating point.
        if old_label is not None:
            aa2 = data["atom_array"]
            titr = np.isin(aa2.res_name, list(_TITRATABLE))
            if np.any(titr & (aa2.protonation_label != old_label)):
                mismatch += 1
                if mismatch <= 10:
                    print(f"  MISMATCH {data.get('example_id')}: re-derived token != baked label", flush=True)

        try:
            save_snapshot(data, precomputed_dir)
            ok += 1
        except Exception as e:
            fail += 1
            if fail <= 20:
                print(f"  FAIL save {data.get('example_id', pkl.name)}: "
                      f"{type(e).__name__}: {str(e)[:100]}", flush=True)
            continue

        if k % args.log_every == 0:
            el = time.perf_counter() - t0
            print(f"  {k} seen  ok={ok} skip={skip} fail={fail} miss={miss} mismatch={mismatch}  "
                  f"{el:.0f}s ({el / max(ok, 1):.2f}s/aug)", flush=True)

    el = time.perf_counter() - t0
    print(f"[augment shard {args.shard_id}/{args.n_shards}] DONE  ok={ok} skip={skip} fail={fail} "
          f"miss={miss} mismatch={mismatch}  {el:.0f}s", flush=True)


if __name__ == "__main__":
    main()
