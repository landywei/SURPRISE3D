#!/usr/bin/env python3
"""Verify Surprise3D JSON object_id vs segments_anno segGroups vs .pth sampled_instance_anno_id."""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch


def arr(x):
    return x if isinstance(x, np.ndarray) else x.numpy()


def seg_group_ids(scannetpp_root: str, scene: str) -> set[int] | None:
    p = os.path.join(scannetpp_root, "data", scene, "scans", "segments_anno.json")
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        seg = json.load(f)
    return {int(g["id"]) for g in seg.get("segGroups", [])}


def pth_instance_sets(pth_path: str) -> tuple[set[int], set[int]] | None:
    if not os.path.isfile(pth_path):
        return None
    d = torch.load(pth_path, map_location="cpu", weights_only=False)
    u_ann = set(arr(d["sampled_instance_anno_id"]).astype(np.int64).tolist())
    u_lab = set(arr(d["sampled_instance_labels"]).astype(np.int64).tolist())
    u_ann.discard(-100)
    u_lab.discard(-100)
    return u_ann, u_lab


def json_id_union(train_json: str, scene: str) -> set[int]:
    with open(train_json) as f:
        rows = json.load(f)
    out: set[int] = set()
    for r in rows:
        if r.get("scene_id") != scene:
            continue
        oid = r["object_id"]
        if isinstance(oid, list):
            out.update(int(x) for x in oid)
        else:
            out.add(int(oid))
    return out


def report_scene(
    scannetpp_root: str, train_json: str, scene: str, pth_subdir: str = "processed"
) -> int:
    seg = seg_group_ids(scannetpp_root, scene)
    pth = os.path.join(scannetpp_root, pth_subdir, f"{scene}.pth")
    sets = pth_instance_sets(pth)
    j = json_id_union(train_json, scene)

    print(f"=== {scene} ===")
    if seg is None:
        print("  MISSING segments_anno.json")
        return 1
    print(f"  segGroup ids: count={len(seg)} range {min(seg)}..{max(seg)}")

    if sets is None:
        print(f"  MISSING {pth}")
        return 1
    u_ann, u_lab = sets
    print(f"  .pth anno_id unique (excl -100): count={len(u_ann)} range {min(u_ann)}..{max(u_ann)}")
    print(f"  .pth labels unique (excl -100): count={len(u_lab)} range {min(u_lab)}..{max(u_lab)}")
    print(f"  JSON union(object_id): count={len(j)} range {min(j)}..{max(j)}")
    print(f"  JSON \\ segGroup: {sorted(j - seg)[:20]}")
    miss = j - u_ann
    print(f"  JSON ids not on any sampled point (anno): {len(miss)} sample {sorted(miss)[:25]}")
    print(f"  pth anno ⊆ segGroup: {u_ann.issubset(seg)}")
    return 0


def aggregate_sample(
    scannetpp_root: str, train_json: str, n: int, seed: int, pth_subdir: str = "processed"
) -> int:
    with open(train_json) as f:
        rows = json.load(f)
    by: dict[str, set[int]] = defaultdict(set)
    for r in rows:
        sid = r["scene_id"]
        oid = r["object_id"]
        if isinstance(oid, list):
            by[sid].update(int(x) for x in oid)
        else:
            by[sid].add(int(oid))
    scenes = list(by.keys())
    random.seed(seed)
    sample = random.sample(scenes, min(n, len(scenes)))

    not_in_mesh = 0
    fracs: list[float] = []
    missing_anno = 0
    for scene in sample:
        seg = seg_group_ids(scannetpp_root, scene)
        if seg is None:
            missing_anno += 1
            continue
        J = by[scene]
        if not J <= seg:
            not_in_mesh += 1
        pth = os.path.join(scannetpp_root, pth_subdir, f"{scene}.pth")
        s = pth_instance_sets(pth)
        if s is None:
            continue
        u_ann, _ = s
        fracs.append(len(J - u_ann) / max(1, len(J)))

    print(f"Sampled {len(sample)} scenes from {train_json} (pth subdir: {pth_subdir})")
    print(f"  missing segments_anno.json: {missing_anno}")
    print(f"  scenes with JSON id not in segGroup: {not_in_mesh}")
    if fracs:
        print(
            "  fraction of JSON ids absent from pth[anno_id]: "
            f"min={min(fracs):.2f} mean={sum(fracs)/len(fracs):.2f} max={max(fracs):.2f}"
        )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scannetpp-root", default="/data/scannetpp")
    p.add_argument("--train-json", default="/data/annotations/surprise_train.json")
    p.add_argument("--pth-subdir", default="processed")
    p.add_argument("--scene", default="")
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.sample > 0:
        return aggregate_sample(
            args.scannetpp_root,
            args.train_json,
            args.sample,
            args.seed,
            args.pth_subdir,
        )
    if args.scene:
        return report_scene(
            args.scannetpp_root,
            args.train_json,
            args.scene,
            args.pth_subdir,
        )
    p.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
