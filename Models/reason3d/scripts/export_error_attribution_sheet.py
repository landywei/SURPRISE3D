#!/usr/bin/env python3
"""
Merge parallel predictions.jsonl runs (same val subset) into one CSV for per-case
error attribution: question, question_type, IoUs per variant, blank note columns,
and chain decoded text.

Rows follow the order of --order-jsonl (default: bare). Rows are matched across
files by eval key (scene_id, object_id, blip_question(description from text_input)).

Example:
  cd Models/reason3d && PYTHONPATH=scripts python scripts/export_error_attribution_sheet.py \\
    --ann /nfs-stor/lan.wei/data/annotations/surprise_val.json \\
    --order-jsonl lavis/output/reason3d_surprise_zeroshot/20260427201/qualitative/predictions.jsonl \\
    --bare lavis/output/reason3d_surprise_zeroshot/20260427201/qualitative/predictions.jsonl \\
    --geo lavis/output/reason3d_surprise_zeroshot_geo/20260427203/qualitative/predictions.jsonl \\
    --chain lavis/output/reason3d_surprise_zeroshot_chain/20260427200/qualitative/predictions.jsonl \\
    --out report/surprise100_error_attribution.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from surprise_pred_join import load_ann_question_types, load_rows, row_eval_key


def _row_by_key(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, Tuple[int, ...], str], Dict[str, Any]]:
    m: Dict[Tuple[str, Tuple[int, ...], str], Dict[str, Any]] = {}
    for r in rows:
        k = row_eval_key(r)
        if k is None:
            continue
        if k not in m:
            m[k] = r
    return m


def main() -> None:
    p = argparse.ArgumentParser(description="CSV sheet for per-instance error attribution.")
    p.add_argument("--ann", type=str, required=True, help="surprise_val.json (for question_type).")
    p.add_argument(
        "--order-jsonl",
        type=str,
        required=True,
        help="Predictions JSONL whose row order defines case_id (e.g. bare run).",
    )
    p.add_argument("--bare", type=str, required=True, help="Bare predictions.jsonl")
    p.add_argument("--geo", type=str, required=True, help="Geo predictions.jsonl")
    p.add_argument("--chain", type=str, required=True, help="Chain predictions.jsonl")
    p.add_argument("--out", type=str, required=True, help="Output .csv path")
    args = p.parse_args()

    ann_qt = load_ann_question_types(args.ann)
    order_rows = load_rows(args.order_jsonl)
    bare_m = _row_by_key(load_rows(args.bare))
    geo_m = _row_by_key(load_rows(args.geo))
    chain_m = _row_by_key(load_rows(args.chain))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    fieldnames = [
        "case_id",
        "eval_save_index",
        "scene_id",
        "object_id_json",
        "question_type",
        "text_input",
        "point_iou_bare",
        "point_iou_geo",
        "point_iou_chain",
        "superpoint_iou_bare",
        "superpoint_iou_geo",
        "superpoint_iou_chain",
        "decoded_text_chain",
        "notes_bare",
        "notes_geo",
        "notes_chain",
        "mask_npz_bare",
    ]

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        case_id = 0
        for r0 in order_rows:
            k = row_eval_key(r0)
            if k is None:
                continue
            rb = bare_m.get(k)
            rg = geo_m.get(k)
            rc = chain_m.get(k)
            if rb is None:
                continue
            case_id += 1
            qt = ann_qt.get(k, "")
            oid = rb.get("object_id")
            oid_json = json.dumps(oid, ensure_ascii=False) if oid is not None else ""

            def piou(r: Optional[Dict[str, Any]]) -> str:
                if r is None:
                    return ""
                return str(float(r.get("point_iou", 0.0)))

            def spiou(r: Optional[Dict[str, Any]]) -> str:
                if r is None:
                    return ""
                return str(float(r.get("superpoint_iou", 0.0)))

            w.writerow(
                {
                    "case_id": str(case_id),
                    "eval_save_index": str(rb.get("eval_save_index", "")),
                    "scene_id": str(rb.get("scene_id", "")),
                    "object_id_json": oid_json,
                    "question_type": qt,
                    "text_input": str(rb.get("text_input", "")),
                    "point_iou_bare": piou(rb),
                    "point_iou_geo": piou(rg),
                    "point_iou_chain": piou(rc),
                    "superpoint_iou_bare": spiou(rb),
                    "superpoint_iou_geo": spiou(rg),
                    "superpoint_iou_chain": spiou(rc),
                    "decoded_text_chain": str((rc or {}).get("decoded_text", "")),
                    "notes_bare": "",
                    "notes_geo": "",
                    "notes_chain": "",
                    "mask_npz_bare": str(rb.get("mask_npz") or ""),
                }
            )

    print(f"Wrote {case_id} rows to {args.out!r}")


if __name__ == "__main__":
    main()
