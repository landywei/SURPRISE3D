"""
Merge UniDet3D (or compatible) per-vertex `*superpoints.npy` into ScanNet++ `prepare_training_data` `.pth`.

Vertex superpoints (length V) are propagated to each sampled point (length N) via nearest mesh vertex
(Open3D mesh vertices + scipy cKDTree), matching the ScanNet++ sampling layout.

CLI is backward compatible with the minimal upstream script (--pth_dir, --scene_dir only).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree


MESH_REL = ("scans", "mesh_aligned_0.05.ply")


def _load_pth(path: str) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _mesh_vertices(scannetpp_root: str, scene_id: str) -> np.ndarray:
    mesh_path = os.path.join(scannetpp_root, "data", scene_id, *MESH_REL)
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh.is_empty():
        raise RuntimeError(f"empty mesh: {mesh_path}")
    return np.asarray(mesh.vertices, dtype=np.float64)


def _find_superpoints_npy(scene_path: str) -> str | None:
    hits = [
        os.path.join(scene_path, f)
        for f in os.listdir(scene_path)
        if f.endswith("superpoints.npy")
    ]
    return hits[0] if hits else None


def merge_one(
    scene_id: str,
    pth_path: str,
    scene_path: str,
    scannetpp_root: str,
    force: bool,
) -> str:
    data = _load_pth(pth_path)
    if "sampled_coords" not in data:
        return "skip_no_sampled_coords"

    coords = np.asarray(data["sampled_coords"], dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 3:
        return "skip_bad_coords"
    pts = coords[:, :3]
    n_pts = int(pts.shape[0])

    sp_existing = data.get("superpoints")
    if (
        not force
        and sp_existing is not None
        and hasattr(sp_existing, "__len__")
        and len(sp_existing) == n_pts
    ):
        return "skip_pth_already_has_superpoints"

    npy_path = _find_superpoints_npy(scene_path)
    if not npy_path:
        return "skip_no_superpoints_npy"

    sp_vert = np.load(npy_path).reshape(-1)
    verts = _mesh_vertices(scannetpp_root, scene_id)
    if int(sp_vert.shape[0]) != int(verts.shape[0]):
        return f"error_len_sp_{sp_vert.shape[0]}_ne_v_{verts.shape[0]}"

    tree = cKDTree(verts)
    _, nn = tree.query(pts, k=1, workers=-1)
    sp_on_pts = sp_vert[nn].astype(np.int64, copy=False)
    data["superpoints"] = sp_on_pts
    torch.save(data, pth_path)
    return f"ok_propagated n_pts={n_pts} n_vert={verts.shape[0]}"


def _merge_job(
    payload: tuple[str, str, str, str, bool],
) -> tuple[str, str, str | None]:
    """Run merge_one in a worker process. Returns (scene_id, message, error_string_or_None)."""
    scene_id, pth_path, scene_path, scannetpp_root, force = payload
    try:
        msg = merge_one(scene_id, pth_path, scene_path, scannetpp_root, force)
        return scene_id, msg, None
    except Exception as e:
        return scene_id, "", str(e)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pth_dir", required=True, help="Directory with <scene_id>.pth")
    p.add_argument(
        "--scene_dir",
        required=True,
        help="Parent of per-scene folders (each with *superpoints.npy), e.g. unidet3d_prep/data",
    )
    p.add_argument(
        "--scannetpp_root",
        default=os.environ.get("SCNNETPP", "/nfs-stor/lan.wei/data/scannetpp"),
        help="ScanNet++ root with data/<scene>/scans/mesh_aligned_0.05.ply (default: $SCNNETPP or /nfs-stor/lan.wei/data/scannetpp)",
    )
    p.add_argument(
        "--only_scenes",
        default="",
        help="Comma-separated scene ids to process (default: all .pth in pth_dir)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-merge even if superpoints length already matches sampled_coords",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Parallel scene workers (separate processes). Default 1. Use 4–16 on NFS; "
        "each task already uses scipy cKDTree.query(..., workers=-1).",
    )
    args = p.parse_args()

    only = {s.strip() for s in args.only_scenes.split(",") if s.strip()}

    if not os.path.isdir(args.pth_dir):
        print(f"error: --pth_dir is not a directory: {args.pth_dir!r}", file=sys.stderr)
        return 1

    pth_files = {
        os.path.splitext(f)[0]: os.path.join(args.pth_dir, f)
        for f in os.listdir(args.pth_dir)
        if f.endswith(".pth")
    }
    if only:
        pth_files = {k: v for k, v in pth_files.items() if k in only}

    if not pth_files:
        print(
            f"No .pth files under --pth_dir={os.path.abspath(args.pth_dir)!r}. "
            "Set OUT_DIR / pth_dir to the directory that contains <scene_id>.pth (same as prepare output).",
            file=sys.stderr,
        )
        return 1

    if not os.path.isdir(args.scene_dir):
        print(f"error: --scene_dir is not a directory: {args.scene_dir!r}", file=sys.stderr)
        return 1

    scene_dirs = {
        d: os.path.join(args.scene_dir, d)
        for d in os.listdir(args.scene_dir)
        if os.path.isdir(os.path.join(args.scene_dir, d))
    }
    if not scene_dirs:
        print(
            f"No scene subfolders under --scene_dir={os.path.abspath(args.scene_dir)!r}. "
            "UniDet output is usually .../output/data/<scene_id>/ (pass .../output/data, not .../output).",
            file=sys.stderr,
        )
        return 1

    print(
        f"[init] pth_files={len(pth_files)} scene_dirs={len(scene_dirs)} "
        f"pth_dir={os.path.abspath(args.pth_dir)} scene_dir={os.path.abspath(args.scene_dir)}"
    )

    ok = skip_sp = skip_other = err = 0
    nw = max(1, int(args.num_workers))

    def _handle_msg(scene_name: str, msg: str) -> None:
        nonlocal ok, skip_sp, skip_other
        print(f"[{scene_name}] {msg}")
        if msg.startswith("ok"):
            ok += 1
        elif "already_has" in msg:
            skip_sp += 1
        else:
            skip_other += 1

    tasks: list[tuple[str, str, str, str, bool]] = []
    for scene_name, pth_path in sorted(pth_files.items()):
        if scene_name not in scene_dirs:
            print(f"[{scene_name}] skip_no_scene_dir")
            skip_other += 1
            continue
        scene_path = scene_dirs[scene_name]
        tasks.append(
            (scene_name, pth_path, scene_path, args.scannetpp_root, args.force)
        )

    if nw == 1:
        for scene_name, pth_path, scene_path, root, force in tasks:
            try:
                msg = merge_one(scene_name, pth_path, scene_path, root, force)
            except Exception as e:
                print(f"[{scene_name}] error {e}", file=sys.stderr)
                err += 1
                continue
            _handle_msg(scene_name, msg)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        try:
            from tqdm import tqdm
        except ImportError:

            def tqdm(x, **kwargs):  # type: ignore[misc,no-redef]
                return x

        with ProcessPoolExecutor(max_workers=nw) as ex:
            futures = {ex.submit(_merge_job, t): t[0] for t in tasks}
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="merge_superpoints",
            ):
                scene_name, msg, err_s = fut.result()
                if err_s is not None:
                    print(f"[{scene_name}] error {err_s}", file=sys.stderr)
                    err += 1
                    continue
                _handle_msg(scene_name, msg)

    print(
        f"[summary] merged_ok={ok} skip_already_superpoints={skip_sp} skip_other={skip_other} errors={err}"
    )
    return 0 if err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
