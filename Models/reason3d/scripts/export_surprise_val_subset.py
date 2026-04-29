#!/usr/bin/env python3
"""
Build a Surprise-style val JSON list (same schema as surprise_val.json) containing only
the prompts listed in sample_manifest.json from sample_surprise_predictions.py, **in manifest order**.

Point evaluate.py at this file to rerun zero-shot on exactly those rows with masks on:

  datasets.3d_refer.build_info.annotations.test.storage=<this.json>
  (or datasets.3d_refer_geo / datasets.3d_refer_chain for other CFGs)

You can also pass --sample-jsonl predictions_sample_100.jsonl instead of --manifest.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from typing import Any, Dict, List, Tuple

from surprise_pred_join import eval_key_from_jsonable, load_rows, pre_question, row_eval_key


def _ann_key(a: Dict[str, Any]) -> Tuple[str, Tuple[int, ...], str]:
    sid = str(a["scene_id"])
    raw_oid = a["object_id"]
    if isinstance(raw_oid, list):
        oids = tuple(sorted(int(x) for x in raw_oid))
    else:
        oids = (int(raw_oid),)
    d = pre_question(a.get("description", ""))
    return (sid, oids, d)


def load_ann_index(ann_path: str) -> Dict[Tuple[str, Tuple[int, ...], str], Dict[str, Any]]:
    with open(ann_path, "r", encoding="utf-8") as f:
        anns = json.load(f)
    idx: Dict[Tuple[str, Tuple[int, ...], str], Dict[str, Any]] = {}
    for a in anns:
        k = _ann_key(a)
        if k not in idx:
            idx[k] = a
    return idx


def main() -> None:
    p = argparse.ArgumentParser(description="Export subset val JSON for masked re-eval.")
    p.add_argument("--ann", type=str, required=True, help="Full surprise_val.json")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--manifest", type=str, help="sample_manifest.json from sample_surprise_predictions.py")
    g.add_argument("--sample-jsonl", type=str, dest="sample_jsonl", help="predictions_sample_N.jsonl rows")
    p.add_argument("--out", type=str, required=True, help="Output JSON path (list of ann objects)")
    args = p.parse_args()

    index = load_ann_index(args.ann)
    keys_in_order: List[Tuple[str, Tuple[int, ...], str]] = []

    if args.manifest:
        with open(args.manifest, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        for entry in manifest:
            ek = entry.get("eval_key")
            if not ek:
                raise SystemExit(f"manifest entry missing eval_key: {entry!r}")
            keys_in_order.append(eval_key_from_jsonable(ek))
    else:
        for r in load_rows(args.sample_jsonl):
            k = row_eval_key(r)
            if k is None:
                raise SystemExit(f"could not parse eval key from row: {r!r}")
            keys_in_order.append(k)

    out_list: List[Dict[str, Any]] = []
    missing = []
    for k in keys_in_order:
        a = index.get(k)
        if a is None:
            missing.append(k)
            continue
        out_list.append(copy.deepcopy(a))

    if missing:
        raise SystemExit(
            f"{len(missing)} keys not found in --ann (first: scene={missing[0][0]!r} oids={missing[0][1]!r})"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_list, f, ensure_ascii=False)

    print(f"Wrote {len(out_list)} annotations to {args.out!r}")


if __name__ == "__main__":
    main()
