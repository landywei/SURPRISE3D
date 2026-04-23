#!/usr/bin/env python3
"""
For one scene, show which segGroup ids appear on mesh vertices (segments + segments_anno)
but never appear on sampled points in a .pth (sampled_instance_anno_id).

Proves "missing in JSON vs pth" is usually sampling coverage, not wrong ids.

Example:
  python scripts/verify_mesh_vs_sampled_instances.py --scene 7739004a45 \\
    --scannetpp-root /data/scannetpp --pth-subdir processed_trial
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch


def arr(x):
    return x if isinstance(x, np.ndarray) else x.numpy()


def mesh_vertex_instance_ids(scannetpp_root: str, scene: str) -> np.ndarray:
    scans = os.path.join(scannetpp_root, "data", scene, "scans")
    with open(os.path.join(scans, "segments.json")) as f:
        seg_indices = np.array(json.load(f)["segIndices"], dtype=np.int64)
    with open(os.path.join(scans, "segments_anno.json")) as f:
        anno = json.load(f)

    seg_to_inst: dict[int, int] = {}
    for g in anno["segGroups"]:
        gid = int(g["id"])
        for s in g["segments"]:
            seg_to_inst[int(s)] = gid

    out = np.zeros(len(seg_indices), dtype=np.int64)
    for i, s in enumerate(seg_indices):
        out[i] = seg_to_inst.get(int(s), 0)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", required=True)
    p.add_argument("--scannetpp-root", default="/data/scannetpp")
    p.add_argument("--pth-subdir", default="processed_trial")
    args = p.parse_args()

    inst_v = mesh_vertex_instance_ids(args.scannetpp_root, args.scene)
    u_mesh = set(int(x) for x in np.unique(inst_v).tolist() if x != 0 and x != -100)

    pth = os.path.join(args.scannetpp_root, args.pth_subdir, f"{args.scene}.pth")
    if not os.path.isfile(pth):
        print("missing", pth)
        return 1
    d = torch.load(pth, map_location="cpu", weights_only=False)
    a = arr(d["sampled_instance_anno_id"]).astype(np.int64).reshape(-1)
    u_samp = set(int(x) for x in np.unique(a).tolist() if x != -100)

    on_mesh_not_sampled = sorted(u_mesh - u_samp)
    sampled_not_on_mesh = sorted(u_samp - u_mesh)

    print(f"scene={args.scene!r} pth={pth!r}")
    print(f"  unique instance ids on mesh vertices (seg→anno): {len(u_mesh)}")
    print(f"  unique sampled_instance_anno_id on points: {len(u_samp)}")
    print(f"  mesh \\ sampled (on mesh, never on a sampled point): {len(on_mesh_not_sampled)}")
    if on_mesh_not_sampled:
        print(f"    ids: {on_mesh_not_sampled}")
    print(f"  sampled \\ mesh (unexpected): {len(sampled_not_on_mesh)}")
    if sampled_not_on_mesh:
        print(f"    ids: {sampled_not_on_mesh}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
