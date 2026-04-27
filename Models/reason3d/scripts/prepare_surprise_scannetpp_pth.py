#!/usr/bin/env python3
"""
Build ScanNet++ .pth files for Reason3D / SURPRISE-style training without top100 / ScanNet200
instance filtering: all segGroups with non-ignore semantics get instance + objectId on vertices,
then on sampled points (prepare_training_data-compatible tensors).

Adds:
  - sampled_mesh_vertex_idx: int64 (N,) — nearest mesh vertex for each sampled point (same rule as
    label transfer in SamplePointsOnMesh). Use this to pull any per-vertex array from the mesh
    or to align with Surprise JSON / segments_anno.json.
  - superpoints: int64 (N,) — optional mesh segmentator partition propagated vertex -> point
    (same idea as Models/reason3d/update_superpoints.py but without UniDet3D .npy).

Does not modify third_party/scannetpp sources; expects that repo on PYTHONPATH (see shell wrapper).

Upstream reference: third_party/scannetpp/semantic/prep/prepare_training_data.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def _ensure_scannetpp_on_path(scannetpp_repo: str) -> None:
    root = os.path.abspath(os.path.expanduser(scannetpp_repo))
    if not os.path.isdir(os.path.join(root, "semantic")):
        raise SystemExit(f"Invalid ScanNet++ repo (no semantic/): {root}")
    if root not in sys.path:
        sys.path.insert(0, root)


def build_transform(data_cfg, use_vertex_superpoints: bool):
    from semantic.transforms.common import Compose
    from semantic.transforms.mesh import (
        AddMeshVertices,
        GetLabelsOnVertices,
        MapLabelToIndex,
        SamplePointsOnMesh,
    )

    transforms: list = []

    transforms.append(AddMeshVertices())

    if "map_label_to_index" in data_cfg.transforms:
        transforms.append(
            MapLabelToIndex(
                data_cfg.labels_path,
                data_cfg.ignore_label,
                count_thresh=data_cfg.get("count_thresh", 0),
                mapping_file=data_cfg.get("mapping_file"),
                keep_classes=data_cfg.get("keep_classes"),
            )
        )

    if "get_labels_on_vertices" in data_cfg.transforms:
        transforms.append(
            GetLabelsOnVertices(
                data_cfg.ignore_label,
                data_cfg.get("multilabel"),
                use_instances=data_cfg.use_instances,
                instance_labels_path=data_cfg.get("instance_labels_path"),
                all_instance_classes=bool(data_cfg.get("all_instance_classes", False)),
            )
        )

    transforms.append(AddMeshVertexIndex())

    if use_vertex_superpoints:
        transforms.append(ComputeVertexSuperpoints())

    if "sample_points_on_mesh" in data_cfg.transforms:
        transforms.append(SamplePointsOnMesh(data_cfg["sample_factor"]))

    return Compose(transforms)


class AddMeshVertexIndex:
    """vtx_mesh_vertex_idx[i] = i; after sampling -> sampled_mesh_vertex_idx = nearest vertex."""

    def __call__(self, sample):
        v = np.asarray(sample["o3d_mesh"].vertices)
        n = int(v.shape[0])
        sample["vtx_mesh_vertex_idx"] = np.arange(n, dtype=np.int64)
        return sample


class ComputeVertexSuperpoints:
    """Per-vertex superpoint id (segmentator); copied to sampled points by SamplePointsOnMesh."""

    def __call__(self, sample):
        try:
            import segmentator as seg
        except ImportError:
            import segmentator_pytorch as seg  # type: ignore

        import open3d as o3d

        mesh: o3d.geometry.TriangleMesh = sample["o3d_mesh"]
        vertices = torch.from_numpy(np.asarray(mesh.vertices, dtype=np.float32))
        faces = torch.from_numpy(np.asarray(mesh.triangles, dtype=np.int64))
        sp = seg.segment_mesh(vertices, faces).numpy().astype(np.int64, copy=False)
        sample["vtx_superpoints"] = sp
        return sample


_MP_DS = None
_MP_OUT: Path | None = None


def _mp_worker(i: int) -> str:
    """Process one dataset index (used with forked workers; see --num-workers)."""
    assert _MP_DS is not None and _MP_OUT is not None
    sample = _MP_DS[i]
    sid = sample["scene_id"]
    _save_sample(sample, _MP_OUT / f"{sid}.pth")
    return str(sid)


def _save_sample(sample: dict, out_path: Path) -> None:
    keep_keys = ["scene_id"] + [
        k for k in sample.keys() if k.startswith("vtx_") or k.startswith("sampled_")
    ]
    save = {k: v for k, v in sample.items() if k in keep_keys}
    if "sampled_superpoints" in save:
        save["superpoints"] = save.pop("sampled_superpoints")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save, out_path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config_file", help="YAML config (see scannetpp_surprise_full_pth.yml)")
    p.add_argument(
        "--scannetpp-repo",
        default=os.environ.get("SCANNPP_REPO", ""),
        help="Path to ScanNet++ python repo root (default: env SCANNPP_REPO)",
    )
    p.add_argument(
        "--no-vertex-superpoints",
        action="store_true",
        help="Skip segmentator mesh superpoints (add later via update_superpoints.py + UniDet3D npy).",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Parallel scene workers (Linux fork). Default 1 (sequential). "
        "Use ~CPU count for IO-heavy mesh load; avoid very large values on shared NFS.",
    )
    args = p.parse_args()

    repo = args.scannetpp_repo.strip()
    if not repo:
        raise SystemExit("Pass --scannetpp-repo or set SCANNPP_REPO to third_party/scannetpp")
    _ensure_scannetpp_on_path(repo)

    from common.file_io import load_yaml_munch, read_txt_list
    from semantic.datasets.scannetpp_release import ScannetPP_Release_Dataset

    cfg = load_yaml_munch(args.config_file)
    data_cfg = cfg.data
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = build_transform(data_cfg, use_vertex_superpoints=not args.no_vertex_superpoints)

    class_names = read_txt_list(data_cfg.labels_path)
    print("Num classes in class list:", len(class_names))
    print("all_instance_classes:", bool(data_cfg.get("all_instance_classes", False)))
    print("Saving to:", out_dir.resolve())

    ds = ScannetPP_Release_Dataset(
        data_root=data_cfg.data_root,
        list_file=data_cfg.list_path,
        transform=transform,
    )
    n = len(ds)
    print("Num samples:", n)

    nw = max(1, int(args.num_workers))
    if nw == 1:
        for i in tqdm(range(n), desc="scenes"):
            sample = ds[i]
            sid = sample["scene_id"]
            _save_sample(sample, out_dir / f"{sid}.pth")
    else:
        if sys.platform == "win32":
            raise SystemExit("--num-workers > 1 is only supported on POSIX (uses fork). Use --num-workers 1 on Windows.")
        import multiprocessing as mp

        global _MP_DS, _MP_OUT
        _MP_DS = ds
        _MP_OUT = out_dir
        print(f"Parallel mode: num_workers={nw} (fork); each worker runs dataset __getitem__ + transforms.")
        ctx = mp.get_context("fork")
        indices = list(range(n))
        with ctx.Pool(nw) as pool:
            for _ in tqdm(
                pool.imap_unordered(_mp_worker, indices, chunksize=1),
                total=n,
                desc="scenes",
            ):
                pass

    if args.no_vertex_superpoints:
        print(
            "Note: superpoints were not written. Merge UniDet3D *superpoints.npy with:\n"
            f"  python {Path(__file__).resolve().parents[1] / 'update_superpoints.py'} "
            f"--pth_dir {out_dir} --scene_dir <dir_with_per_scene_superpoints_npy> "
            f"--scannetpp_root <ScanNet++ root containing data/>"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
