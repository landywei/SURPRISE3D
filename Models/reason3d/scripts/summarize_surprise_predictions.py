#!/usr/bin/env python3
"""
Aggregate and per-question-type metrics from qualitative/predictions.jsonl (Surprise val).

See surprise_pred_join.py for join rules.

Example:
  python scripts/summarize_surprise_predictions.py \\
    --ann /nfs-stor/lan.wei/data/annotations/surprise_val.json \\
    /path/to/qualitative/predictions.jsonl

Cross-variant markdown table (bare, geo, chain order):
  python scripts/summarize_surprise_predictions.py --markdown-cross \\
    --ann /nfs-stor/lan.wei/data/annotations/surprise_val.json \\
    bare.jsonl geo.jsonl chain.jsonl
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from surprise_pred_join import (
    load_ann_question_types,
    load_rows,
    row_eval_key,
)


def aggregate_metrics(rows: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    pi = np.array([float(r["point_iou"]) for r in rows], dtype=np.float64)
    spi = np.array([float(r.get("superpoint_iou", 0.0)) for r in rows], dtype=np.float64)
    return {
        "n": float(len(pi)),
        "miou": float(pi.mean()),
        "acc50": float((pi > 0.5).mean()),
        "acc25": float((pi > 0.25).mean()),
        "mean_sp_iou": float(spi.mean()),
    }


def rows_by_question_type(
    rows: List[Dict[str, Any]], ann_qt: Dict[Tuple[str, Tuple[int, ...], str], str]
) -> Dict[str, List[float]]:
    by_qt: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        k = row_eval_key(r)
        if k is None:
            by_qt["UNMATCHED"].append(float(r["point_iou"]))
            continue
        qt = ann_qt.get(k, "UNMATCHED")
        by_qt[qt].append(float(r["point_iou"]))
    return by_qt


def qt_metrics(pi: List[float]) -> Tuple[int, float, float, float]:
    a = np.array(pi, dtype=np.float64)
    if len(a) == 0:
        return 0, float("nan"), float("nan"), float("nan")
    return len(a), float(a.mean()), float((a > 0.5).mean()), float((a > 0.25).mean())


def print_single_jsonl(jsonl_path: str, ann_qt: Dict[Tuple[str, Tuple[int, ...], str], str]) -> None:
    rows = load_rows(jsonl_path)
    m = aggregate_metrics(rows)
    print(f"\n## {jsonl_path}")
    print(
        f"  rows={int(m['n'])}  mIoU={m['miou']:.4f}  Acc@0.50={m['acc50']:.4f}  "
        f"Acc@0.25={m['acc25']:.4f}  mean_spIoU={m['mean_sp_iou']:.4f}"
    )

    keys = [row_eval_key(r) for r in rows]
    valid = [k for k in keys if k is not None]
    dup = sum(c - 1 for c in Counter(valid).values() if c > 1)
    print(f"  unique_eval_keys={len(set(valid))}  duplicate_rows={dup}  parse_fail={sum(1 for k in keys if k is None)}")

    by_qt = rows_by_question_type(rows, ann_qt)
    print("  per question_type (point IoU):")
    for qt in sorted(by_qt.keys(), key=lambda x: (-len(by_qt[x]), x)):
        n, mi, a50, a25 = qt_metrics(by_qt[qt])
        print(f"    {qt:18s} n={n:5d}  mIoU={mi:.4f}  Acc50={a50:.4f}  Acc25={a25:.4f}")


def markdown_cross_table(
    paths: List[str],
    labels: List[str],
    ann_qt: Dict[Tuple[str, Tuple[int, ...], str], str],
) -> str:
    all_qt: set[str] = set()
    per_label_qt: Dict[str, Dict[str, List[float]]] = {}
    rows_by_label: Dict[str, List[Dict[str, Any]]] = {}
    for lab, p in zip(labels, paths):
        rows = load_rows(p)
        rows_by_label[lab] = rows
        per_label_qt[lab] = rows_by_question_type(rows, ann_qt)
        all_qt |= set(per_label_qt[lab].keys())

    def sort_key(q: str) -> Tuple[int, str]:
        if q == "UNMATCHED":
            return (2, q)
        return (0, q)

    qts = sorted(all_qt, key=sort_key)
    lines: List[str] = []
    subcols = []
    for lab in labels:
        subcols.extend([f"{lab} n", f"{lab} mIoU", f"{lab} Acc@0.50", f"{lab} Acc@0.25"])
    lines.append("| question_type | " + " | ".join(subcols) + " |")
    lines.append("| --- | " + " | ".join(["---"] * len(subcols)) + " |")
    for qt in qts:
        parts = [qt]
        for lab in labels:
            n, mi, a50, a25 = qt_metrics(per_label_qt[lab].get(qt, []))
            if n == 0:
                parts.extend(["—", "—", "—", "—"])
            else:
                parts.extend([str(n), f"{mi:.4f}", f"{a50:.4f}", f"{a25:.4f}"])
        lines.append("| " + " | ".join(parts) + " |")

    total_parts = ["**TOTAL (all rows)**"]
    for lab in labels:
        m = aggregate_metrics(rows_by_label[lab])
        total_parts.extend(
            [
                str(int(m["n"])),
                f"{m['miou']:.4f}",
                f"{m['acc50']:.4f}",
                f"{m['acc25']:.4f}",
            ]
        )
    lines.append("| " + " | ".join(total_parts) + " |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Surprise predictions.jsonl metrics (+ per question_type).")
    p.add_argument(
        "--ann",
        type=str,
        default="/nfs-stor/lan.wei/data/annotations/surprise_val.json",
        help="surprise_val.json",
    )
    p.add_argument(
        "--markdown-cross",
        action="store_true",
        help="Print one markdown table: rows=question_type, columns=metrics for each of three JSONLs.",
    )
    p.add_argument(
        "--labels",
        type=str,
        default="bare,geo,chain",
        help="Comma labels for --markdown-cross (default bare,geo,chain).",
    )
    p.add_argument("jsonl", nargs="+", help="One or more predictions.jsonl (exactly 3 with --markdown-cross).")
    args = p.parse_args()

    ann_qt = load_ann_question_types(args.ann)

    if args.markdown_cross:
        labels = [x.strip() for x in args.labels.split(",") if x.strip()]
        if len(args.jsonl) != len(labels):
            raise SystemExit(
                f"--markdown-cross: need same number of --labels ({len(labels)}) and jsonl files ({len(args.jsonl)})."
            )
        print(markdown_cross_table(args.jsonl, labels, ann_qt))
        return

    for jsonl_path in args.jsonl:
        print_single_jsonl(jsonl_path, ann_qt)


if __name__ == "__main__":
    main()
