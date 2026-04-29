#!/usr/bin/env python3
"""
Statistical summary for surprise100_error_attribution.csv.

Usage:
  cd Models/reason3d && PYTHONPATH=scripts python scripts/analyze_surprise100_attribution.py \\
    /path/to/surprise100_error_attribution.csv \\
    [--out report/surprise100_attribution_analysis.md]

Notes columns (notes_bare, notes_geo, notes_chain): any non-empty text is counted.
Optional tag split: use semicolons or pipes in a cell, e.g. "semantic;partial_hit".

Apple Numbers: File → Export To → CSV, save into the repo, then point --csv here.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple


def _f(x: str) -> float:
    return float(x) if x.strip() else float("nan")


def _split_tags(s: str) -> List[str]:
    if not s or not str(s).strip():
        return []
    parts = re.split(r"[;|]", str(s))
    return [p.strip().lower() for p in parts if p.strip()]


def pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n != len(ys) or n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    return cov / math.sqrt(vx * vy)


def summarize_column(vals: List[float]) -> str:
    v = [x for x in vals if not math.isnan(x)]
    if not v:
        return "n=0"
    n = len(v)
    mean = sum(v) / n
    var = sum((x - mean) ** 2 for x in v) / max(n - 1, 1)
    sd = math.sqrt(var)
    return f"n={n} mean={mean:.4f} sd={sd:.4f} min={min(v):.4f} max={max(v):.4f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv_path", type=str)
    p.add_argument("--out", type=str, default=None, help="Write markdown report here")
    args = p.parse_args()

    rows: List[Dict[str, str]] = []
    with open(args.csv_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({k: (v or "") for k, v in row.items()})

    lines: List[str] = []
    lines.append("# Surprise-100 attribution analysis\n\n")
    lines.append(f"Source: `{args.csv_path}`\n\n")
    lines.append(f"Rows: **{len(rows)}**\n\n")

    pb = [_f(r["point_iou_bare"]) for r in rows]
    pg = [_f(r["point_iou_geo"]) for r in rows]
    pc = [_f(r["point_iou_chain"]) for r in rows]

    lines.append("## Point IoU (per variant)\n\n")
    for name, arr in [("bare", pb), ("geo", pg), ("chain", pc)]:
        lines.append(f"- **{name}**: {summarize_column(arr)}\n")

    lines.append("\n## Head-to-head (point IoU)\n\n")
    pairs = [
        ("chain", "bare", pc, pb),
        ("geo", "bare", pg, pb),
        ("chain", "geo", pc, pg),
    ]
    for a, b, va, vb in pairs:
        wins = sum(1 for i in range(len(rows)) if not math.isnan(va[i]) and not math.isnan(vb[i]) and va[i] > vb[i])
        ties = sum(1 for i in range(len(rows)) if not math.isnan(va[i]) and not math.isnan(vb[i]) and va[i] == vb[i])
        losses = sum(1 for i in range(len(rows)) if not math.isnan(va[i]) and not math.isnan(vb[i]) and va[i] < vb[i])
        lines.append(f"- **{a} vs {b}**: wins={wins}, ties={ties}, losses={losses}\n")

    bb, cc = [], []
    for i in range(len(rows)):
        if not math.isnan(pb[i]) and not math.isnan(pc[i]):
            bb.append(pb[i])
            cc.append(pc[i])
    rp2 = pearson(bb, cc)
    lines.append(f"\n**Pearson r (bare vs chain point IoU):** {rp2 if rp2 is not None else 'n/a'}\n")

    lines.append("\n## By question_type (mean point IoU)\n\n")
    lines.append("| question_type | n | bare | geo | chain |\n|---|---|---|---|---|\n")
    by_qt: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)
    for i, r in enumerate(rows):
        qt = r.get("question_type", "").strip() or "unknown"
        by_qt[qt].append((pb[i], pg[i], pc[i]))
    for qt in sorted(by_qt.keys()):
        xs = by_qt[qt]
        n = len(xs)
        mb = sum(a for a, _, _ in xs) / n
        mg = sum(b for _, b, _ in xs) / n
        mc = sum(c for _, _, c in xs) / n
        lines.append(f"| {qt} | {n} | {mb:.4f} | {mg:.4f} | {mc:.4f} |\n")

    nb = [r.get("notes_bare", "").strip() for r in rows]
    ng = [r.get("notes_geo", "").strip() for r in rows]
    nc = [r.get("notes_chain", "").strip() for r in rows]
    nonempty_b = sum(1 for x in nb if x)
    nonempty_g = sum(1 for x in ng if x)
    nonempty_c = sum(1 for x in nc if x)

    lines.append("\n## Qualitative notes\n\n")
    lines.append(
        f"Non-empty cells: bare={nonempty_b}, geo={nonempty_g}, chain={nonempty_c}. "
        "If all are zero, export from Apple Numbers as CSV into this repo and re-run.\n\n"
    )

    tag_counter = {"bare": Counter(), "geo": Counter(), "chain": Counter()}
    for lab, arr in [("bare", nb), ("geo", ng), ("chain", nc)]:
        for cell in arr:
            for t in _split_tags(cell):
                tag_counter[lab][t] += 1

    if any(tag_counter[k] for k in tag_counter):
        lines.append("### Tag frequencies (split on `;` or `|`)\n\n")
        for lab in ("bare", "geo", "chain"):
            if not tag_counter[lab]:
                continue
            lines.append(f"**{lab}**\n\n")
            for tag, cnt in tag_counter[lab].most_common():
                lines.append(f"- `{tag}`: {cnt}\n")

    text = "".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\nWrote {args.out!r}")


if __name__ == "__main__":
    main()
