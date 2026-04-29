#!/usr/bin/env python3
"""
Stratified sample of N predictions rows (default 100) spread across question_type and scene_id.

Uses round-robin over (question_type, scene_id) buckets so rare (qt, scene) cells are visited
before repeating a busy cell. Optionally restrict to eval keys present in all given JSONLs
so row indices can be exported per variant for visualization.

Outputs (under --out-dir):
  predictions_sample_{n}.jsonl   — full rows from --primary (default name)
  sample_manifest.json           — list of {eval_key, row_index_<label>, ...} in sample order
  row_indices_<label>.txt        — one 0-based row index per line (-1 if key missing in that run)

Example (align bare / geo / chain):
  cd Models/reason3d && PYTHONPATH=. python scripts/sample_surprise_predictions.py \\
    --ann /nfs-stor/lan.wei/data/annotations/surprise_val.json \\
    --primary lavis/output/reason3d_surprise_zeroshot/20260427161/qualitative/predictions.jsonl \\
    --also lavis/output/reason3d_surprise_zeroshot_geo/20260427162/qualitative/predictions.jsonl \\
         lavis/output/reason3d_surprise_zeroshot_chain/20260426161/qualitative/predictions.jsonl \\
    --labels bare,geo,chain \\
    --intersect-keys \\
    --out-dir lavis/output/_samples/surprise_100 \\
    -n 100 --seed 42

Then export a val JSON for masked re-eval (see scripts/export_surprise_val_subset.py and
scripts/run_surprise_zeroshot_eval_subset_masks.sh).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

from surprise_pred_join import (
    eval_key_to_jsonable,
    first_index_by_eval_key,
    load_ann_question_types,
    load_rows,
    row_eval_key,
)


def stratified_indices(
    rows: List[Dict[str, Any]],
    ann_qt: Dict[Tuple[str, Tuple[int, ...], str], str],
    n: int,
    seed: int,
    key_allowlist: Optional[Set[Tuple[str, Tuple[int, ...], str]]] = None,
) -> List[int]:
    """Return up to ``n`` row indices into ``rows``."""
    buckets: Dict[Tuple[str, str], deque] = defaultdict(deque)
    for i, r in enumerate(rows):
        k = row_eval_key(r)
        if k is None:
            continue
        if key_allowlist is not None and k not in key_allowlist:
            continue
        qt = ann_qt.get(k, "UNMATCHED")
        sid = k[0]
        buckets[(qt, sid)].append(i)

    rng = random.Random(seed)
    cells = list(buckets.keys())
    rng.shuffle(cells)
    iters = {c: buckets[c] for c in cells}

    picked: List[int] = []
    active = [c for c in cells if iters[c]]
    safety = 0
    max_sweeps = max(len(rows), n * 200)
    while len(picked) < n and active and safety < max_sweeps:
        safety += 1
        progressed = False
        for c in list(active):
            if len(picked) >= n:
                break
            dq = iters[c]
            if not dq:
                active.remove(c)
                continue
            picked.append(dq.popleft())
            progressed = True
        if not progressed:
            break

    if len(picked) < n:
        used = set(picked)
        rest = [i for i in range(len(rows)) if i not in used and row_eval_key(rows[i]) is not None]
        if key_allowlist is not None:
            rest = [i for i in rest if row_eval_key(rows[i]) in key_allowlist]
        rng.shuffle(rest)
        for i in rest:
            if len(picked) >= n:
                break
            picked.append(i)

    return picked[:n]


def main() -> None:
    p = argparse.ArgumentParser(description="Stratified sample of Surprise predictions JSONL rows.")
    p.add_argument("--ann", type=str, default="/nfs-stor/lan.wei/data/annotations/surprise_val.json")
    p.add_argument("--primary", type=str, required=True, help="JSONL used for stratification and sample rows copy.")
    p.add_argument(
        "--also",
        nargs="*",
        default=[],
        help="Additional predictions.jsonl paths (same order as --labels after the first).",
    )
    p.add_argument(
        "--labels",
        type=str,
        default="bare",
        help="Comma-separated labels: first = primary, then one per --also (e.g. bare,geo,chain).",
    )
    p.add_argument(
        "--intersect-keys",
        action="store_true",
        help="Only sample eval keys that appear in primary and every --also file (first occurrence).",
    )
    p.add_argument("-n", type=int, default=100, help="Target number of samples.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, required=True)
    args = p.parse_args()

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    paths = [args.primary] + list(args.also)
    if len(labels) != len(paths):
        raise SystemExit(f"Need len(--labels)={len(labels)} to match 1 primary + {len(args.also)} --also = {len(paths)} jsonl paths.")

    ann_qt = load_ann_question_types(args.ann)
    all_rows = [load_rows(p) for p in paths]

    key_allowlist: Optional[Set[Tuple[str, Tuple[int, ...], str]]] = None
    if args.intersect_keys:
        sets = []
        for rows in all_rows:
            sets.append(set(first_index_by_eval_key(rows).keys()))
        key_allowlist = set.intersection(*sets) if sets else set()
        print(f"intersect_keys: {len(key_allowlist)} keys in common across {len(paths)} runs")

    primary = all_rows[0]
    picked = stratified_indices(primary, ann_qt, args.n, args.seed, key_allowlist)
    if len(picked) < args.n:
        print(f"warning: only {len(picked)} samples (requested {args.n})")

    os.makedirs(args.out_dir, exist_ok=True)
    n_out = len(picked)
    sample_path = os.path.join(args.out_dir, f"predictions_sample_{n_out}.jsonl")
    manifest: List[Dict[str, Any]] = []

    maps = [first_index_by_eval_key(rows) for rows in all_rows]

    with open(sample_path, "w", encoding="utf-8") as fout:
        for pi in picked:
            r = primary[pi]
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            k = row_eval_key(r)
            entry: Dict[str, Any] = {"eval_key": eval_key_to_jsonable(k) if k else None}
            for lab, m in zip(labels, maps):
                entry[f"row_index_{lab}"] = m.get(k, -1) if k is not None else -1
            manifest.append(entry)

    with open(os.path.join(args.out_dir, "sample_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    for lab in labels:
        lines = [str(entry[f"row_index_{lab}"]) for entry in manifest]
        with open(os.path.join(args.out_dir, f"row_indices_{lab}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    print(f"Wrote {sample_path}")
    print(f"Wrote {os.path.join(args.out_dir, 'sample_manifest.json')}")
    for lab in labels:
        print(f"Wrote {os.path.join(args.out_dir, f'row_indices_{lab}.txt')}")


if __name__ == "__main__":
    main()
