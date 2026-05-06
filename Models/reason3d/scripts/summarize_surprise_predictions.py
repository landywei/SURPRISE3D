#!/usr/bin/env python3
"""
Aggregate and per-question-type metrics from qualitative/predictions.jsonl (Surprise val).

See surprise_pred_join.py for join rules. Question types are recovered from surprise_val.json
(JSONL ``question_type`` field is ignored — it is empty in writes from refer_seg_task[v3]).
For chainv3 JSONL (rows have ``max_per_instance_iou``/``hit_at_25``/``hit_at_50``) the
per-instance metrics ``meanMaxIoU / hit@0.25 / hit@0.50`` are auto-appended.

Plain text per-QT breakdown:
  python scripts/summarize_surprise_predictions.py \\
    --ann /nfs-stor/lan.wei/data/annotations/surprise_val.json \\
    /path/to/qualitative/predictions.jsonl

Markdown table for a single run (rows = question_type, cols = metrics):
  python scripts/summarize_surprise_predictions.py --markdown-per-qt \\
    --ann /nfs-stor/lan.wei/data/annotations/surprise_val.json \\
    /path/to/qualitative/predictions.jsonl

Add --transpose to swap rows and columns:
  --markdown-per-qt --transpose  → rows = metric, cols = question_type (+TOTAL)
  --markdown-cross --transpose   → rows = (variant, metric), cols = question_type (+TOTAL)

Add --highlight-max (cross only) to bold the winning variant per (question_type, metric):
  python scripts/summarize_surprise_predictions.py --markdown-cross --transpose --highlight-max \\
    --labels v0,v1,v2 --ann /nfs-stor/lan.wei/data/annotations/surprise_val.json \\
    a.jsonl b.jsonl c.jsonl

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


def _is_real(x: Any) -> bool:
    if x is None:
        return False
    try:
        return not np.isnan(float(x))
    except (TypeError, ValueError):
        return False


def _has_per_instance(rows: Iterable[Dict[str, Any]]) -> bool:
    for r in rows:
        if _is_real(r.get("max_per_instance_iou")):
            return True
    return False


def aggregate_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    pi = np.array([float(r["point_iou"]) for r in rows], dtype=np.float64)
    spi = np.array([float(r.get("superpoint_iou", 0.0)) for r in rows], dtype=np.float64)
    out: Dict[str, float] = {
        "n": float(len(pi)),
        "miou": float(pi.mean()) if len(pi) else float("nan"),
        "acc50": float((pi > 0.5).mean()) if len(pi) else float("nan"),
        "acc25": float((pi > 0.25).mean()) if len(pi) else float("nan"),
        "mean_sp_iou": float(spi.mean()) if len(spi) else float("nan"),
    }
    # chainv3 per-instance fields (best-of-GT-instance match); reported only when present.
    mxs = np.array(
        [float(r["max_per_instance_iou"]) for r in rows if _is_real(r.get("max_per_instance_iou"))],
        dtype=np.float64,
    )
    h25 = np.array(
        [float(r["hit_at_25"]) for r in rows if _is_real(r.get("hit_at_25"))], dtype=np.float64
    )
    h50 = np.array(
        [float(r["hit_at_50"]) for r in rows if _is_real(r.get("hit_at_50"))], dtype=np.float64
    )
    if len(mxs):
        out["n_per_instance"] = float(len(mxs))
        out["mean_max_iou"] = float(mxs.mean())
    if len(h25):
        out["hit25"] = float(h25.mean())
    if len(h50):
        out["hit50"] = float(h50.mean())
    return out


def rows_by_question_type(
    rows: List[Dict[str, Any]], ann_qt: Dict[Tuple[str, Tuple[int, ...], str], str]
) -> Dict[str, List[Dict[str, Any]]]:
    by_qt: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        k = row_eval_key(r)
        qt = ann_qt.get(k, "UNMATCHED") if k is not None else "UNMATCHED"
        by_qt[qt].append(r)
    return by_qt


def qt_metrics(rows_or_pi: Iterable[Any]) -> Dict[str, float]:
    items = list(rows_or_pi)
    if not items:
        return {"n": 0, "miou": float("nan"), "acc25": float("nan"), "acc50": float("nan")}
    if isinstance(items[0], dict):
        return aggregate_metrics(items)
    pi = np.array([float(x) for x in items], dtype=np.float64)
    return {
        "n": float(len(pi)),
        "miou": float(pi.mean()),
        "acc25": float((pi > 0.25).mean()),
        "acc50": float((pi > 0.5).mean()),
    }


def _fmt(m: Dict[str, float], key: str, width: int = 6) -> str:
    v = m.get(key)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—".ljust(width)
    return f"{v:.4f}"


def print_single_jsonl(jsonl_path: str, ann_qt: Dict[Tuple[str, Tuple[int, ...], str], str]) -> None:
    rows = load_rows(jsonl_path)
    m = aggregate_metrics(rows)
    has_inst = _has_per_instance(rows)
    print(f"\n## {jsonl_path}")
    head = (
        f"  rows={int(m['n'])}  mIoU={m['miou']:.4f}  Acc@0.50={m['acc50']:.4f}  "
        f"Acc@0.25={m['acc25']:.4f}  mean_spIoU={m['mean_sp_iou']:.4f}"
    )
    if has_inst:
        head += (
            f"  meanMaxIoU={_fmt(m, 'mean_max_iou')}  "
            f"hit@0.25={_fmt(m, 'hit25')}  hit@0.50={_fmt(m, 'hit50')}"
        )
    print(head)

    keys = [row_eval_key(r) for r in rows]
    valid = [k for k in keys if k is not None]
    dup = sum(c - 1 for c in Counter(valid).values() if c > 1)
    print(f"  unique_eval_keys={len(set(valid))}  duplicate_rows={dup}  parse_fail={sum(1 for k in keys if k is None)}")

    by_qt = rows_by_question_type(rows, ann_qt)
    if has_inst:
        print(
            "  per question_type:    n      mIoU   Acc25  Acc50  meanMaxIoU  hit@0.25  hit@0.50"
        )
        for qt in sorted(by_qt.keys(), key=lambda x: (-len(by_qt[x]), x)):
            qm = qt_metrics(by_qt[qt])
            n = int(qm["n"])
            print(
                f"    {qt:18s} {n:5d}  {_fmt(qm,'miou')}  {_fmt(qm,'acc25')}  {_fmt(qm,'acc50')}  "
                f"{_fmt(qm,'mean_max_iou'):>10s}  {_fmt(qm,'hit25'):>8s}  {_fmt(qm,'hit50'):>8s}"
            )
    else:
        print("  per question_type (point IoU):")
        for qt in sorted(by_qt.keys(), key=lambda x: (-len(by_qt[x]), x)):
            qm = qt_metrics(by_qt[qt])
            n = int(qm["n"])
            print(
                f"    {qt:18s} n={n:5d}  mIoU={_fmt(qm,'miou')}  "
                f"Acc50={_fmt(qm,'acc50')}  Acc25={_fmt(qm,'acc25')}"
            )


def _qt_sort_key(by_qt: Dict[str, List[Dict[str, Any]]], q: str) -> Tuple[int, int, str]:
    """Sort: real QTs by descending count, then alpha; UNMATCHED last."""
    if q == "UNMATCHED":
        return (2, 0, q)
    return (0, -len(by_qt.get(q, [])), q)


def _metric_cols(has_inst: bool) -> List[Tuple[str, str]]:
    """Return [(display_name, key_in_metrics_dict), ...] in canonical order."""
    cols = [("n", "n"), ("mIoU", "miou"), ("Acc@0.25", "acc25"), ("Acc@0.50", "acc50")]
    if has_inst:
        cols += [("meanMaxIoU", "mean_max_iou"), ("hit@0.25", "hit25"), ("hit@0.50", "hit50")]
    return cols


def _cell(m: Dict[str, float], key: str) -> str:
    if key == "n":
        n = int(m.get("n", 0))
        return str(n) if n > 0 else "—"
    return _fmt(m, key)


def _md_table(header: List[str], rows: List[List[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def markdown_per_qt_table(
    jsonl_path: str,
    ann_qt: Dict[Tuple[str, Tuple[int, ...], str], str],
    transpose: bool = False,
) -> str:
    """One markdown table for a single JSONL.

    Default: rows = question_type, columns = metrics.
    transpose=True: rows = metric, columns = question_type (TOTAL last).
    """
    rows = load_rows(jsonl_path)
    by_qt = rows_by_question_type(rows, ann_qt)
    has_inst = _has_per_instance(rows)
    qts = sorted(by_qt.keys(), key=lambda q: _qt_sort_key(by_qt, q))
    metric_cols = _metric_cols(has_inst)
    total = aggregate_metrics(rows)

    if not transpose:
        header = ["question_type"] + [name for name, _ in metric_cols]
        out_rows: List[List[str]] = []
        for qt in qts:
            m = qt_metrics(by_qt[qt])
            out_rows.append([qt] + [_cell(m, key) for _, key in metric_cols])
        out_rows.append(["**TOTAL**"] + [_cell(total, key) for _, key in metric_cols])
        return _md_table(header, out_rows)

    # Transposed: metric down, question_type across.
    header = ["metric"] + qts + ["**TOTAL**"]
    qt_metrics_cache = {qt: qt_metrics(by_qt[qt]) for qt in qts}
    out_rows = []
    for name, key in metric_cols:
        row = [name]
        for qt in qts:
            row.append(_cell(qt_metrics_cache[qt], key))
        row.append(_cell(total, key))
        out_rows.append(row)
    return _md_table(header, out_rows)


_TOTAL_KEY = "__TOTAL__"


def _winners(
    labels: List[str],
    qts: List[str],
    cell_metrics: Dict[Tuple[str, str], Dict[str, float]],
    label_totals: Dict[str, Dict[str, float]],
    metric_keys: Iterable[str],
) -> set:
    """Return {(label, qt_or__TOTAL__, metric_key)} where the cell ties the column max.

    Ties bold all winners. NaN / missing values are excluded. ``n`` should not be passed in.
    """
    out: set = set()
    cols: List[Tuple[str, Dict[str, Dict[str, float]]]] = [
        (qt, {lab: cell_metrics[(lab, qt)] for lab in labels}) for qt in qts
    ]
    cols.append((_TOTAL_KEY, label_totals))
    for col_key, label_to_metrics in cols:
        for mkey in metric_keys:
            best_val = float("-inf")
            for lab in labels:
                v = label_to_metrics.get(lab, {}).get(mkey)
                if v is None:
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if np.isnan(fv):
                    continue
                if fv > best_val:
                    best_val = fv
            if best_val == float("-inf"):
                continue
            for lab in labels:
                v = label_to_metrics.get(lab, {}).get(mkey)
                try:
                    fv = float(v) if v is not None else float("nan")
                except (TypeError, ValueError):
                    continue
                if not np.isnan(fv) and fv >= best_val:
                    out.add((lab, col_key, mkey))
    return out


def _wrap_bold(s: str) -> str:
    s = s.strip()
    if not s or s in {"—"}:
        return s
    return f"**{s}**"


def markdown_cross_table(
    paths: List[str],
    labels: List[str],
    ann_qt: Dict[Tuple[str, Tuple[int, ...], str], str],
    transpose: bool = False,
    highlight_max: bool = False,
) -> str:
    """Cross-table over multiple JSONLs.

    Default: rows = question_type, columns = (label × metric) blocks.
    transpose=True: rows = (label, metric), columns = question_type (+TOTAL last).
    Per-instance metric columns (meanMaxIoU/hit@0.25/hit@0.50) are included only for
    labels whose JSONL has them.
    highlight_max=True: bold the cell with the highest value across labels for each
    (question_type, metric) pair (and for TOTAL). ``n`` is never highlighted; ties bold all.
    """
    all_qt: set[str] = set()
    per_label_qt: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    rows_by_label: Dict[str, List[Dict[str, Any]]] = {}
    label_has_inst: Dict[str, bool] = {}
    for lab, p in zip(labels, paths):
        rows = load_rows(p)
        rows_by_label[lab] = rows
        per_label_qt[lab] = rows_by_question_type(rows, ann_qt)
        label_has_inst[lab] = _has_per_instance(rows)
        all_qt |= set(per_label_qt[lab].keys())

    # Stable QT order: union across labels by total count (UNMATCHED last).
    union_counts: Dict[str, int] = defaultdict(int)
    for lab in labels:
        for qt, rs in per_label_qt[lab].items():
            union_counts[qt] += len(rs)
    qts = sorted(all_qt, key=lambda q: (2, 0, q) if q == "UNMATCHED" else (0, -union_counts[q], q))

    # Per-(label, qt) aggregated metrics + per-label totals.
    cell_metrics: Dict[Tuple[str, str], Dict[str, float]] = {}
    label_totals: Dict[str, Dict[str, float]] = {}
    for lab in labels:
        label_totals[lab] = aggregate_metrics(rows_by_label[lab])
        for qt in qts:
            cell_metrics[(lab, qt)] = qt_metrics(per_label_qt[lab].get(qt, []))

    # Compute winners per (col, metric). Skip "n" (counts aren't a quality metric).
    winners: set = set()
    if highlight_max:
        all_metric_keys: List[str] = []
        seen: set = set()
        for lab in labels:
            for _, key in _metric_cols(label_has_inst[lab]):
                if key == "n" or key in seen:
                    continue
                seen.add(key)
                all_metric_keys.append(key)
        winners = _winners(labels, qts, cell_metrics, label_totals, all_metric_keys)

    def _cell_hl(m: Dict[str, float], key: str, lab: str, col_key: str) -> str:
        val = _cell(m, key)
        if highlight_max and key != "n" and (lab, col_key, key) in winners:
            return _wrap_bold(val)
        return val

    if not transpose:
        # Wide layout: rows = QT, cols = label × metric (canonical metric order).
        subcols: List[str] = []
        for lab in labels:
            for name, _ in _metric_cols(label_has_inst[lab]):
                subcols.append(f"{lab} {name}")

        header = ["question_type"] + subcols
        out_rows: List[List[str]] = []
        for qt in qts:
            row: List[str] = [qt]
            for lab in labels:
                m = cell_metrics[(lab, qt)]
                row.extend(_cell_hl(m, key, lab, qt) for _, key in _metric_cols(label_has_inst[lab]))
            out_rows.append(row)
        total_row: List[str] = ["**TOTAL (all rows)**"]
        for lab in labels:
            m = label_totals[lab]
            total_row.extend(_cell_hl(m, key, lab, _TOTAL_KEY) for _, key in _metric_cols(label_has_inst[lab]))
        out_rows.append(total_row)
        return _md_table(header, out_rows)

    # Transposed: rows = (label, metric), columns = question_type (+TOTAL last).
    header = ["variant", "metric"] + qts + ["**TOTAL**"]
    out_rows = []
    for lab in labels:
        for name, key in _metric_cols(label_has_inst[lab]):
            row = [lab, name]
            for qt in qts:
                row.append(_cell_hl(cell_metrics[(lab, qt)], key, lab, qt))
            row.append(_cell_hl(label_totals[lab], key, lab, _TOTAL_KEY))
            out_rows.append(row)
    return _md_table(header, out_rows)


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
        help="Print one markdown table: rows=question_type, columns=metrics for each of N JSONLs.",
    )
    p.add_argument(
        "--markdown-per-qt",
        action="store_true",
        help="Print a single markdown table per JSONL: rows=question_type, columns=metrics. "
             "When the JSONL contains chainv3 per-instance fields (max_per_instance_iou/hit_at_25/hit_at_50), "
             "those columns (meanMaxIoU/hit@0.25/hit@0.50) are appended automatically.",
    )
    p.add_argument(
        "--labels",
        type=str,
        default="bare,geo,chain",
        help="Comma labels for --markdown-cross (default bare,geo,chain).",
    )
    p.add_argument(
        "--transpose",
        action="store_true",
        help="Swap rows ↔ columns in --markdown-per-qt and --markdown-cross output. "
             "For --markdown-per-qt: rows=metric, cols=question_type. "
             "For --markdown-cross: rows=(variant, metric), cols=question_type.",
    )
    p.add_argument(
        "--highlight-max",
        action="store_true",
        help="In --markdown-cross output, bold the cell with the highest value across "
             "variants for each (question_type, metric) pair (and for TOTAL). "
             "n is never highlighted; ties bold all winners.",
    )
    p.add_argument("jsonl", nargs="+", help="One or more predictions.jsonl (matches --labels with --markdown-cross).")
    args = p.parse_args()

    ann_qt = load_ann_question_types(args.ann)

    if args.markdown_cross:
        labels = [x.strip() for x in args.labels.split(",") if x.strip()]
        if len(args.jsonl) != len(labels):
            raise SystemExit(
                f"--markdown-cross: need same number of --labels ({len(labels)}) and jsonl files ({len(args.jsonl)})."
            )
        print(
            markdown_cross_table(
                args.jsonl,
                labels,
                ann_qt,
                transpose=args.transpose,
                highlight_max=args.highlight_max,
            )
        )
        return

    if args.markdown_per_qt:
        for i, jsonl_path in enumerate(args.jsonl):
            if i > 0:
                print()
            print(f"### {jsonl_path}")
            print(markdown_per_qt_table(jsonl_path, ann_qt, transpose=args.transpose))
        return

    for jsonl_path in args.jsonl:
        print_single_jsonl(jsonl_path, ann_qt)


if __name__ == "__main__":
    main()
