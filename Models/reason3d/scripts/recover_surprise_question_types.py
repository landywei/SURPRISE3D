#!/usr/bin/env python3
"""
Fill ``question_type`` on each predictions.jsonl row by joining to surprise_val.json
(same triple as training: scene_id, object_id, blip_question(description) from text_input).

Example:
  python scripts/recover_surprise_question_types.py \\
    --ann /nfs-stor/lan.wei/data/annotations/surprise_val.json \\
    --in lavis/output/.../qualitative/predictions.jsonl \\
    --out lavis/output/.../qualitative/predictions_with_qt.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

from surprise_pred_join import load_ann_question_types, load_rows, row_eval_key


def main() -> None:
    p = argparse.ArgumentParser(description="Recover question_type per predictions.jsonl row.")
    p.add_argument("--ann", type=str, default="/nfs-stor/lan.wei/data/annotations/surprise_val.json")
    p.add_argument("--in", dest="inp", type=str, required=True, help="Input predictions.jsonl")
    p.add_argument("--out", type=str, required=True, help="Output JSONL with question_type set")
    args = p.parse_args()

    ann_qt = load_ann_question_types(args.ann)
    rows = load_rows(args.inp)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    n_ok = 0
    n_miss = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for r in rows:
            k = row_eval_key(r)
            qt = ann_qt.get(k, "") if k is not None else ""
            out: Dict[str, Any] = dict(r)
            out["question_type"] = qt
            if qt:
                n_ok += 1
            else:
                n_miss += 1
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {args.out!r}  matched_question_type={n_ok}  empty={n_miss}")


if __name__ == "__main__":
    main()
