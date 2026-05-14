#!/usr/bin/env python3
"""
Materialize a *filtered* Surprise val JSON: drop annotations whose `object_id`
is not in `sampled_instance_anno_id` of the corresponding scene's `.pth`.

Why:
  ThreeDReferDataset's runtime filter (`filter_missing_gt_in_pth: true`) reads
  `.pth` at dataset init and drops mismatched rows. If `.pth` files are
  regenerated between runs, OR if eval auto-resume kicks in with a
  non-deterministic prior partial JSONL, the kept-row set drifts across runs
  of the same val split. Pinning a filtered JSON once makes every later eval
  see exactly the same `n` regardless of `.pth` state or resume behaviour.

How to use:
  1. Run this once against the canonical preprocessing and val JSON:

        python3 scripts/build_filtered_surprise_val.py \\
          --val-json   /nfs-stor/lan.wei/data/annotations/surprise_val.json \\
          --pts-root   /nfs-stor/lan.wei/data/scannetpp \\
          --pth-subdir processed_surprise_full_pth \\
          --out-json   /nfs-stor/lan.wei/data/annotations/surprise_val_filtered_v1.json \\
          --out-cache  /nfs-stor/lan.wei/data/annotations/surprise_inst_id_cache_v1.json

  2. In each val YAML under lavis/projects/reason3d/val/, point
     `build_info.annotations.test.storage` at `--out-json` and set
     `dataset_init.filter_missing_gt_in_pth: false` so the runtime filter is
     a no-op on already-filtered data. The optional `--out-cache` produces an
     instance-id cache compatible with `dataset_init.instance_id_cache_file`
     so future filter-on runs (e.g. ScanRefer / ScanReason future work) skip
     the per-scene `torch.load`.

Output JSON shape:
  - Same shape as the input val JSON: a list of annotation dicts. Each dict
    is preserved verbatim; only the list is filtered.

Determinism:
  - Output order matches input order. No randomness.
  - Re-running with the same inputs produces a byte-identical JSON.

Failure mode:
  - If a scene's `.pth` is missing, its annotations are KEPT (matches
    ThreeDReferDataset._ann_has_target_in_pth which returns True on missing
    .pth so the original error path -- empty GT mask -- still surfaces). Use
    `--strict-missing-pth` to drop them instead, which produces a strictly
    smaller and stricter filtered JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import os.path as osp
import sys
from typing import Dict, List, Optional, Set

import numpy as np
import torch


_INSTANCE_CACHE_VERSION = "reason3d_instance_id_cache_v1"


def _load_pth_instance_ids(pth_path: str) -> Optional[Set[int]]:
    if not osp.isfile(pth_path):
        return None
    try:
        d = torch.load(pth_path, map_location="cpu", weights_only=False)
    except TypeError:
        d = torch.load(pth_path, map_location="cpu")
    inst = np.asarray(d["sampled_instance_anno_id"]).astype(np.int64).reshape(-1)
    return {int(x) for x in np.unique(inst).tolist() if int(x) != -100}


def _ann_target_in_pth(ann: dict, ids: Optional[Set[int]]) -> bool:
    if ids is None:
        return True
    oid = ann["object_id"]
    targets = [int(x) for x in oid] if isinstance(oid, list) else [int(oid)]
    return any(t in ids for t in targets)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--val-json", required=True, help="Path to surprise_val.json (input).")
    p.add_argument(
        "--pts-root",
        default="/nfs-stor/lan.wei/data/scannetpp",
        help="Root containing <pth-subdir>/<scene>.pth (matches dataset.build_info.points.storage).",
    )
    p.add_argument(
        "--pth-subdir",
        default="processed_surprise_full_pth",
        help="Subdir under --pts-root holding per-scene .pth files (matches dataset_init.pth_rel_subdir).",
    )
    p.add_argument(
        "--out-json",
        required=True,
        help="Where to write the filtered annotations JSON (same shape as --val-json).",
    )
    p.add_argument(
        "--out-cache",
        default=None,
        help="Optional: write an instance_id_cache JSON (compatible with dataset_init.instance_id_cache_file).",
    )
    p.add_argument(
        "--strict-missing-pth",
        action="store_true",
        help="Drop annotations whose scene's .pth is missing (default keeps them, matching the dataset filter).",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true", help="Per-scene logging."
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    val_json = osp.abspath(osp.expanduser(args.val_json))
    pts_root = osp.abspath(osp.expanduser(args.pts_root))
    out_json = osp.abspath(osp.expanduser(args.out_json))
    out_cache = osp.abspath(osp.expanduser(args.out_cache)) if args.out_cache else None
    pth_dir = osp.join(pts_root, args.pth_subdir)

    if not osp.isfile(val_json):
        logging.error("--val-json not a file: %s", val_json)
        return 2
    if not osp.isdir(pth_dir):
        logging.error("--pts-root/--pth-subdir not a directory: %s", pth_dir)
        return 2

    with open(val_json, "r", encoding="utf-8") as f:
        anns = json.load(f)
    if not isinstance(anns, list):
        logging.error("Expected a top-level JSON list in %s, got %s", val_json, type(anns).__name__)
        return 2

    n_in = len(anns)
    scenes = sorted({a.get("scene_id") for a in anns if a.get("scene_id") is not None})
    logging.info(
        "Loaded %d annotations across %d unique scenes from %s",
        n_in,
        len(scenes),
        val_json,
    )

    inst_cache: Dict[str, Optional[Set[int]]] = {}
    n_missing_pth = 0
    for i, sid in enumerate(scenes, 1):
        ids = _load_pth_instance_ids(osp.join(pth_dir, f"{sid}.pth"))
        inst_cache[sid] = ids
        if ids is None:
            n_missing_pth += 1
            logging.warning("Scene %s: .pth missing under %s", sid, pth_dir)
        elif args.verbose:
            logging.debug("Scene %s: %d unique instance ids", sid, len(ids))
        if i % 50 == 0:
            logging.info("  ... loaded %d / %d scenes", i, len(scenes))
    logging.info(
        "Read instance ids for %d scenes (%d had missing .pth).",
        len(scenes) - n_missing_pth,
        n_missing_pth,
    )

    kept: List[dict] = []
    n_drop_filter = 0
    n_drop_missing = 0
    for a in anns:
        sid = a.get("scene_id")
        ids = inst_cache.get(sid)
        if ids is None:
            if args.strict_missing_pth:
                n_drop_missing += 1
                continue
            kept.append(a)
            continue
        if _ann_target_in_pth(a, ids):
            kept.append(a)
        else:
            n_drop_filter += 1

    logging.info(
        "Filter result: kept %d / %d (dropped %d for missing target id; %d for missing .pth%s).",
        len(kept),
        n_in,
        n_drop_filter,
        n_drop_missing,
        " [strict mode]" if args.strict_missing_pth else " [kept under default mode]",
    )

    os.makedirs(osp.dirname(out_json) or ".", exist_ok=True)
    tmp_json = out_json + ".tmp"
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False)
    os.replace(tmp_json, out_json)
    logging.info("Wrote %d filtered annotations to %s", len(kept), out_json)

    if out_cache:
        instance_sets: Dict[str, List[int]] = {}
        missing: List[str] = []
        for sid, ids in inst_cache.items():
            if ids is None:
                missing.append(sid)
            else:
                instance_sets[sid] = sorted(ids)
        blob = {
            "format": _INSTANCE_CACHE_VERSION,
            "pts_root": pts_root,
            "pth_rel_subdir": str(args.pth_subdir),
            "instance_sets": {k: instance_sets[k] for k in sorted(instance_sets)},
            "missing_pth": sorted(missing),
        }
        os.makedirs(osp.dirname(out_cache) or ".", exist_ok=True)
        tmp_cache = out_cache + ".tmp"
        with open(tmp_cache, "w", encoding="utf-8") as f:
            json.dump(blob, f, indent=0)
        os.replace(tmp_cache, out_cache)
        logging.info(
            "Wrote instance-id cache to %s (%d scenes with ids, %d missing .pth).",
            out_cache,
            len(instance_sets),
            len(missing),
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
