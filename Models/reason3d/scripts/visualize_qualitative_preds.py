#!/usr/bin/env python3
"""
Visualize Reason3D qualitative eval artifacts (predictions.jsonl + masks/*.npz).

Typical layout after run_surprise_zeroshot_eval_small.sh:
  <run_dir>/qualitative/predictions.jsonl
  <run_dir>/qualitative/masks/<scene_id>_<ann_id>.npz

Each .npz contains float16 arrays pred_pmask, gt_pmask with one value per point,
in the same order as the eval dataloader after ThreeDReferDataset.load +
transform_test (see lavis/datasets/datasets/threedrefer_datasets.py).

Important: mask files are keyed only by (scene_id, ann_id). If several JSONL rows
share the same mask_npz path, the file reflects the last write during eval, not
every text query. Use JSONL for per-prompt metrics; use NPZ for geometry for that
final (scene, ann) pair.

Examples:
  python scripts/visualize_qualitative_preds.py --qual-dir lavis/output/.../qualitative \\
    --pts-root /nfs-stor/lan.wei/data/scannetpp --pth-subdir processed --list

  python scripts/visualize_qualitative_preds.py --qual-dir .../qualitative \\
    --pts-root /nfs-stor/lan.wei/data/scannetpp --pth-subdir processed \\
    --export-row 2 --out-dir /tmp/qual_vis

  # All JSONL rows (use --stride for smaller files; full res is large per row)
  python scripts/visualize_qualitative_preds.py --qual-dir .../qualitative \\
    --pts-root .../scannetpp --pth-subdir processed --export-all --out-dir /tmp/all_vis --stride 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError as e:  # pragma: no cover
    print("This script requires PyTorch (same env as Reason3D).", file=sys.stderr)
    raise e


def _as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


@dataclass
class SceneGeometry:
    xyz_middle: np.ndarray  # [N, 3] float, same frame as coord_float at eval
    rgb_unit: np.ndarray  # [N, 3] float in [0, 1]


def load_scene_geometry(pth_path: str) -> SceneGeometry:
    """Match ThreeDReferDataset.load + transform_test (eval, no augment)."""
    data = torch.load(pth_path, weights_only=False)
    xyz = _as_numpy(data["sampled_coords"])[:, :3].astype(np.float64)
    rgb = _as_numpy(data["sampled_colors"]).astype(np.float64)

    xyz = xyz[:, :3] - xyz[:, :3].mean(0)
    rgb = rgb / 0.5 - 1.0

    xyz_middle = xyz
    # Eval uses transform_test voxel coords internally; per-point masks follow the same
    # point order as in the .pth after load() above (no subsampling in default test path).

    rgb_unit = np.clip((rgb + 1.0) * 0.5, 0.0, 1.0)
    return SceneGeometry(xyz_middle=xyz_middle.astype(np.float32), rgb_unit=rgb_unit.astype(np.float32))


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def resolve_mask_path(qual_dir: str, mask_npz: str) -> str:
    if os.path.isabs(mask_npz):
        return mask_npz
    return os.path.normpath(os.path.join(os.path.dirname(qual_dir), mask_npz))


def scene_pth_path(pts_root: str, pth_rel_subdir: str, scene_id: str) -> str:
    return os.path.join(pts_root, pth_rel_subdir, f"{scene_id}.pth")


def build_overlay_colors(
    rgb_unit: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    pred_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (colors_gt_highlight, colors_pred_highlight) each [N,3].
    Base color is rgb_unit; gt paints red, pred paints green (binary).
    """
    base = rgb_unit.copy()
    gt_m = gt >= 0.5
    pr_m = pred >= pred_threshold
    c_gt = base.copy()
    c_gt[gt_m] = np.array([1.0, 0.2, 0.2], dtype=np.float32)
    c_pr = base.copy()
    c_pr[pr_m] = np.array([0.2, 1.0, 0.2], dtype=np.float32)
    return c_gt, c_pr


def build_confidence_colors(rgb_unit: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Color points by sigmoid-like confidence: blue channel from pred probability."""
    p = np.clip(pred.astype(np.float64), 0.0, 1.0)
    c = rgb_unit.copy()
    c[:, 2] = np.clip(c[:, 2] * 0.4 + 0.6 * p, 0.0, 1.0).astype(np.float32)
    return c


def _write_ply_binary_numpy(xyz: np.ndarray, rgb_u8: np.ndarray, out_path: str) -> None:
    """Binary little-endian PLY (no extra deps). Fast for ~1M+ points."""
    n = int(xyz.shape[0])
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    xyz_f = np.ascontiguousarray(xyz.astype("<f4", copy=False))
    rgb_c = np.ascontiguousarray(rgb_u8.astype(np.uint8, copy=False))
    blob = np.empty(n, dtype=[("xyz", "<f4", (3,)), ("rgb", "u1", (3,))])
    blob["xyz"] = xyz_f
    blob["rgb"] = rgb_c
    with open(out_path, "wb") as f:
        f.write(header)
        f.write(blob.tobytes())


def write_ply(xyz: np.ndarray, rgb: np.ndarray, out_path: str) -> None:
    """Write PLY with float xyz and uint8 rgb (MeshLab / CloudCompare friendly)."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float64)
    rgb = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    rgb_u8 = (rgb * 255.0 + 0.5).astype(np.uint8)

    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
        o3d.io.write_point_cloud(out_path, pcd)
        return
    except ImportError:
        pass

    try:
        from plyfile import PlyData, PlyElement

        n = xyz.shape[0]
        verts = np.empty(
            n,
            dtype=[
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ],
        )
        verts["x"] = xyz[:, 0].astype(np.float32)
        verts["y"] = xyz[:, 1].astype(np.float32)
        verts["z"] = xyz[:, 2].astype(np.float32)
        verts["red"] = rgb_u8[:, 0]
        verts["green"] = rgb_u8[:, 1]
        verts["blue"] = rgb_u8[:, 2]
        el = PlyElement.describe(verts, "vertex")
        PlyData([el], text=True).write(out_path)
        return
    except ImportError:
        pass

    _write_ply_binary_numpy(xyz, rgb_u8, out_path)


def cmd_list(rows: List[Dict[str, Any]]) -> None:
    hdr = f"{'idx':>4}  {'scene':12}  {'pIoU':>8}  {'spIoU':>8}  text"
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(rows):
        t = r.get("text_input", "") or ""
        if len(t) > 72:
            t = t[:69] + "..."
        print(
            f"{i:4d}  {str(r.get('scene_id','')):12}  "
            f"{float(r.get('point_iou', 0.0)):8.4f}  {float(r.get('superpoint_iou', 0.0)):8.4f}  {t}"
        )


def cmd_export(
    rows: List[Dict[str, Any]],
    row_index: int,
    qual_dir: str,
    pts_root: str,
    pth_subdir: str,
    out_dir: str,
    pred_threshold: float,
    heatmap_pred: bool,
    stride: int,
    verbose: bool = True,
) -> None:
    if row_index < 0 or row_index >= len(rows):
        raise SystemExit(f"row_index {row_index} out of range [0, {len(rows)-1}]")
    row = rows[row_index]
    scene_id = str(row["scene_id"])
    pth = scene_pth_path(pts_root, pth_subdir, scene_id)
    if not os.path.isfile(pth):
        raise SystemExit(f"Scene .pth not found: {pth}")

    mask_path = resolve_mask_path(qual_dir, str(row["mask_npz"]))
    if not os.path.isfile(mask_path):
        raise SystemExit(f"Mask npz not found: {mask_path}")

    z = np.load(mask_path)
    pred = np.asarray(z["pred_pmask"], dtype=np.float32).reshape(-1)
    gt = np.asarray(z["gt_pmask"], dtype=np.float32).reshape(-1)

    geo = load_scene_geometry(pth)
    if stride > 1:
        sl = slice(None, None, stride)
        geo = SceneGeometry(xyz_middle=geo.xyz_middle[sl], rgb_unit=geo.rgb_unit[sl])
        pred = pred[sl]
        gt = gt[sl]

    if geo.xyz_middle.shape[0] != pred.shape[0]:
        raise SystemExit(
            f"Point count mismatch: pth N={geo.xyz_middle.shape[0]} vs mask N={pred.shape[0]}. "
            "Check pts_root / pth_rel_subdir and that the .pth matches the eval run."
        )

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{scene_id}_row{row_index}")

    write_ply(geo.xyz_middle, geo.rgb_unit, base + "_rgb.ply")
    c_gt, c_pr = build_overlay_colors(geo.rgb_unit, gt, pred, pred_threshold)
    write_ply(geo.xyz_middle, c_gt, base + "_gt.ply")
    write_ply(geo.xyz_middle, c_pr, base + "_pred.ply")
    if heatmap_pred:
        write_ply(geo.xyz_middle, build_confidence_colors(geo.rgb_unit, pred), base + "_pred_heat.ply")

    meta = {
        "row_index": row_index,
        "scene_id": scene_id,
        "ann_id": row.get("ann_id"),
        "object_id": row.get("object_id"),
        "point_iou": row.get("point_iou"),
        "superpoint_iou": row.get("superpoint_iou"),
        "text_input": row.get("text_input"),
        "mask_npz": row.get("mask_npz"),
        "pth_used": pth,
        "pred_threshold": pred_threshold,
        "stride": stride,
        "points_written": int(geo.xyz_middle.shape[0]),
    }
    with open(base + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    with open(base + "_caption.txt", "w", encoding="utf-8") as f:
        f.write(str(row.get("text_input", "")) + "\n")

    if verbose:
        print("Wrote:")
        for suf in ("_rgb.ply", "_gt.ply", "_pred.ply"):
            print(" ", base + suf)
        if heatmap_pred:
            print(" ", base + "_pred_heat.ply")
        print(" ", base + "_meta.json")
        print(" ", base + "_caption.txt")
    else:
        extra = " +heat" if heatmap_pred else ""
        print(f"  -> {os.path.basename(base)}_rgb/gt/pred.ply{extra} +meta +caption", flush=True)


def cmd_export_all(
    rows: List[Dict[str, Any]],
    qual_dir: str,
    pts_root: str,
    pth_subdir: str,
    out_dir: str,
    pred_threshold: float,
    heatmap_pred: bool,
    stride: int,
) -> None:
    n = len(rows)
    if n >= 8 and stride == 1:
        print(
            f"Note: exporting {n} rows at full resolution (--stride 1) can use tens of GB. "
            "Consider --stride 10 or higher for previews.",
            file=sys.stderr,
        )
    for i in range(n):
        print(f"[{i + 1}/{n}] row {i} scene={rows[i].get('scene_id')!r}", flush=True)
        cmd_export(
            rows,
            i,
            qual_dir=qual_dir,
            pts_root=pts_root,
            pth_subdir=pth_subdir,
            out_dir=out_dir,
            pred_threshold=pred_threshold,
            heatmap_pred=heatmap_pred,
            stride=stride,
            verbose=False,
        )
    print(f"Done. Outputs under {os.path.abspath(out_dir)}", flush=True)


def cmd_plot_iou(rows: List[Dict[str, Any]], out_png: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("Plotting requires matplotlib: pip install matplotlib") from e

    pi = np.array([float(r.get("point_iou", 0.0)) for r in rows], dtype=np.float64)
    spi = np.array([float(r.get("superpoint_iou", 0.0)) for r in rows], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(pi, bins=20, color="steelblue", edgecolor="white")
    axes[0].set_title("Point IoU")
    axes[0].set_xlabel("IoU")
    axes[1].hist(spi, bins=20, color="seagreen", edgecolor="white")
    axes[1].set_title("Superpoint IoU")
    axes[1].set_xlabel("IoU")
    fig.suptitle("Qualitative eval (one entry per prompt)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved histogram: {out_png}")


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize qualitative zero-shot outputs.")
    p.add_argument(
        "--qual-dir",
        type=str,
        required=True,
        help="Directory containing predictions.jsonl and masks/ (e.g. .../20260423154/qualitative)",
    )
    p.add_argument(
        "--pts-root",
        type=str,
        default=os.environ.get("REASON3D_PTS_ROOT", ""),
        help="Root of point caches (YAML datasets.3d_refer.build_info.points.storage). "
        "Default: env REASON3D_PTS_ROOT.",
    )
    p.add_argument(
        "--pth-subdir",
        type=str,
        default=os.environ.get("REASON3D_PTH_SUBDIR", "processed"),
        help="Subdir under pts-root with <scene_id>.pth (YAML pth_rel_subdir). Default: processed.",
    )
    p.add_argument("--list", action="store_true", help="Print predictions.jsonl as a compact table.")
    p.add_argument("--export-row", type=int, default=None, help="Export PLYs for this 0-based JSONL row.")
    p.add_argument(
        "--export-all",
        action="store_true",
        help="Export PLYs for every row (same files as --export-row per index). Implies heavy disk use if --stride 1.",
    )
    p.add_argument("--out-dir", type=str, default="qualitative_vis", help="Output directory for exports.")
    p.add_argument(
        "--pred-threshold",
        type=float,
        default=0.5,
        help="Binary prediction mask threshold for pred PLY coloring.",
    )
    p.add_argument(
        "--heatmap-pred",
        action="store_true",
        help="Also write *_pred_heat.ply coloring by soft prediction score.",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every k-th point for lighter PLYs (default 1 = full resolution).",
    )
    p.add_argument(
        "--plot-iou-hist",
        type=str,
        default=None,
        metavar="PNG",
        help="Write a histogram of point/superpoint IoUs to this PNG path.",
    )
    args = p.parse_args()

    qual_dir = os.path.abspath(args.qual_dir)
    jsonl_path = os.path.join(qual_dir, "predictions.jsonl")
    if not os.path.isfile(jsonl_path):
        raise SystemExit(f"Missing {jsonl_path}")

    rows = read_jsonl(jsonl_path)
    if not rows:
        raise SystemExit("Empty predictions.jsonl")

    if args.list:
        cmd_list(rows)

    if args.export_all and args.export_row is not None:
        raise SystemExit("Use either --export-all or --export-row N, not both.")

    if args.export_row is not None or args.export_all:
        if int(args.stride) < 1:
            raise SystemExit("--stride must be >= 1")
        if not str(args.pts_root).strip():
            raise SystemExit("Set --pts-root or REASON3D_PTS_ROOT to the ScanNet++/surprise points root.")
        st = max(1, int(args.stride))
        out_abs = os.path.abspath(args.out_dir)
        if args.export_all:
            cmd_export_all(
                rows,
                qual_dir=qual_dir,
                pts_root=os.path.expanduser(args.pts_root),
                pth_subdir=args.pth_subdir,
                out_dir=out_abs,
                pred_threshold=args.pred_threshold,
                heatmap_pred=args.heatmap_pred,
                stride=st,
            )
        else:
            cmd_export(
                rows,
                args.export_row,
                qual_dir=qual_dir,
                pts_root=os.path.expanduser(args.pts_root),
                pth_subdir=args.pth_subdir,
                out_dir=out_abs,
                pred_threshold=args.pred_threshold,
                heatmap_pred=args.heatmap_pred,
                stride=st,
                verbose=True,
            )

    if args.plot_iou_hist:
        cmd_plot_iou(rows, args.plot_iou_hist)

    if not (args.list or args.export_row is not None or args.export_all or args.plot_iou_hist):
        p.print_help()
        print(
            "\nNo action: pass --list, --export-row N, --export-all, and/or --plot-iou-hist out.png",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
