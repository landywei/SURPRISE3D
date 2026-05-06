#!/usr/bin/env python3
"""
Visualize Reason3D qualitative eval artifacts (predictions.jsonl + masks/*.npz).

Typical layout after run_surprise_zeroshot_eval_small.sh:
  <run_dir>/qualitative/predictions.jsonl
  <run_dir>/qualitative/masks/<scene_id>_<ann_id>.npz

Each .npz contains float16 arrays pred_pmask, gt_pmask with one value per point,
in the same order as the eval dataloader after ThreeDReferDataset.load +
transform_test (see lavis/datasets/datasets/threedrefer_datasets.py). For
chain-v3 CoT runs (``reason3d_t5_chainv3_cot``) the same .npz also contains
``pred_pmask_intermediate`` for rows where the two-pass decode actually fired
(see ``intermediate_in_npz`` in predictions.jsonl); when present, this script
also writes a ``*_pred_intermediate.ply`` overlay alongside ``*_pred.ply``
so you can eyeball the landmark mass-pool input vs. the final M_2.

Mask files are named ``<scene_id>_<ann_id>_<eval_save_index>.npz`` (see
``lavis/tasks/refer_seg_task.py``). Each JSONL line lists the matching ``mask_npz``
and ``eval_save_index`` so GT/pred align with that row's ``text_input`` / ``object_id``.

Viewers: open ``*_gt.ply`` / ``*_pred.ply`` / ``*_pred_intermediate.ply`` (not
``*_rgb.ply``) and enable vertex / point colors (e.g. CloudCompare:
``Edit > Colors > Set unique`` from RGB fields; Windows 3D Viewer often ignores
PLY vertex colors).

Examples:
  python scripts/visualize_qualitative_preds.py --qual-dir lavis/output/.../qualitative \\
    --pts-root /nfs-stor/lan.wei/data/scannetpp --pth-subdir processed --list

  python scripts/visualize_qualitative_preds.py --qual-dir .../qualitative \\
    --pts-root /nfs-stor/lan.wei/data/scannetpp --pth-subdir processed \\
    --export-row 2 --out-dir /tmp/qual_vis

  # All JSONL rows (use --stride for smaller files; full res is large per row)
  python scripts/visualize_qualitative_preds.py --qual-dir .../qualitative \\
    --pts-root .../scannetpp --pth-subdir processed --export-all --out-dir /tmp/all_vis --stride 20

  # Subset from sample_surprise_predictions.py (row_indices_bare.txt; skip lines with -1)
  python scripts/visualize_qualitative_preds.py --qual-dir .../qualitative \\
    --pts-root .../scannetpp --pth-subdir processed_surprise_full_pth \\
    --row-indices-file .../row_indices_bare.txt --out-dir /tmp/vis100 --stride 5

  # Chain-v3 CoT run: by default *_pred_intermediate.ply is written whenever
  # the npz carries it. Use --skip-intermediate to suppress, or
  # --intermediate-only to skip *_pred.ply entirely.
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


def pred_for_display(pred: np.ndarray, mode: str) -> np.ndarray:
    """
    ``auto`` matches the eval saver in refer_seg_task.py: apply sigmoid only when
    ``max > 1`` or ``min < 0`` (otherwise values are stored as-is, often already in
    ``[0, 1]``). Use ``--pred-display sigmoid`` if you know logits were saved in a
    narrow band inside ``[0, 1]`` without that condition being met.
    """
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    if mode == "raw":
        return p
    if mode == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(p, -60.0, 60.0)))
    if p.size and (float(p.max()) > 1.0 or float(p.min()) < 0.0):
        return 1.0 / (1.0 + np.exp(-np.clip(p, -60.0, 60.0)))
    return np.clip(p, 0.0, 1.0)


_HIGHLIGHT_COLORS: Dict[str, np.ndarray] = {
    # GT mask: red.
    "gt": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    # Final M_2 prediction: green (matches existing behavior).
    "pred": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    # Pass-1 intermediate M_1 prediction (chainv3-CoT only): cyan/blue, picked
    # to be unmistakably distinct from both gt-red and pred-green when both
    # PLYs are loaded side-by-side in the same viewer.
    "intermediate": np.array([0.0, 0.6, 1.0], dtype=np.float32),
}


def build_mask_overlay(
    rgb_unit: np.ndarray,
    mask: np.ndarray,
    threshold: float,
    background_scale: float,
    color_name: str,
) -> np.ndarray:
    """Return ``[N, 3]`` colors with a single mask painted in the named highlight.

    Non-foreground uses ``rgb_unit * background_scale`` so sparse foregrounds
    (often ~0.1–2% of points) stay visible in MeshLab / CloudCompare; otherwise
    bright primaries can disappear against busy RGB textures at full scale.
    """
    base = np.clip(rgb_unit.astype(np.float32, copy=False), 0.0, 1.0)
    dim = np.clip(base * float(background_scale), 0.0, 1.0)
    fg = np.asarray(mask, dtype=np.float64).reshape(-1) >= float(threshold)
    out = dim.copy()
    out[fg] = _HIGHLIGHT_COLORS[color_name]
    return out


def build_overlay_colors(
    rgb_unit: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    pred_threshold: float,
    background_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper kept for callers that still want only (gt, pred)."""
    c_gt = build_mask_overlay(rgb_unit, gt, 0.5, background_scale, "gt")
    c_pr = build_mask_overlay(rgb_unit, pred, pred_threshold, background_scale, "pred")
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
    # Add CoT columns only if any row carries them, so non-CoT runs keep their
    # compact two-IoU table.
    has_cot = any(("did_two_pass" in r) or ("intermediate_point_iou" in r) for r in rows)
    if has_cot:
        hdr = (
            f"{'idx':>4}  {'scene':12}  {'pIoU':>8}  {'spIoU':>8}  "
            f"{'2pass':>5}  {'iIoU':>8}  text"
        )
    else:
        hdr = f"{'idx':>4}  {'scene':12}  {'pIoU':>8}  {'spIoU':>8}  text"
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(rows):
        t = r.get("text_input", "") or ""
        if len(t) > 72:
            t = t[:69] + "..."
        if has_cot:
            two_pass = r.get("did_two_pass")
            two_pass_s = "Y" if two_pass else ("N" if two_pass is False else "-")
            i_iou = r.get("intermediate_point_iou")
            i_iou_s = f"{float(i_iou):8.4f}" if isinstance(i_iou, (int, float)) else " " * 8
            print(
                f"{i:4d}  {str(r.get('scene_id','')):12}  "
                f"{float(r.get('point_iou', 0.0)):8.4f}  {float(r.get('superpoint_iou', 0.0)):8.4f}  "
                f"{two_pass_s:>5}  {i_iou_s}  {t}"
            )
        else:
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
    pred_display: str,
    background_scale: float,
    skip_gt: bool = False,
    skip_rgb: bool = True,
    skip_intermediate: bool = False,
    skip_pred: bool = False,
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
    pred_raw = np.asarray(z["pred_pmask"], dtype=np.float32).reshape(-1)
    gt = np.asarray(z["gt_pmask"], dtype=np.float32).reshape(-1)
    pred = pred_for_display(pred_raw, pred_display).astype(np.float32)
    # ``pred_pmask_intermediate`` only appears for chainv3-CoT rows where the
    # two-pass decode actually fired; older runs / non-CoT models silently
    # omit this key. Treat absence as "no intermediate to render".
    inter_raw: Optional[np.ndarray] = None
    if "pred_pmask_intermediate" in z.files:
        inter_raw = np.asarray(z["pred_pmask_intermediate"], dtype=np.float32).reshape(-1)
    inter: Optional[np.ndarray] = None
    if inter_raw is not None:
        inter = pred_for_display(inter_raw, pred_display).astype(np.float32)

    geo = load_scene_geometry(pth)
    if stride > 1:
        sl = slice(None, None, stride)
        geo = SceneGeometry(xyz_middle=geo.xyz_middle[sl], rgb_unit=geo.rgb_unit[sl])
        pred = pred[sl]
        gt = gt[sl]
        if inter is not None:
            inter = inter[sl]

    if geo.xyz_middle.shape[0] != pred.shape[0]:
        raise SystemExit(
            f"Point count mismatch: pth N={geo.xyz_middle.shape[0]} vs mask N={pred.shape[0]}. "
            "Check pts_root / pth_rel_subdir and that the .pth matches the eval run."
        )
    if inter is not None and inter.shape[0] != pred.shape[0]:
        raise SystemExit(
            f"Intermediate mask N={inter.shape[0]} != pred N={pred.shape[0]} for row {row_index}; "
            "stale .npz from before pred_pmask_intermediate was added with a different stride?"
        )

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{scene_id}_row{row_index}")

    gt_frac = float((gt >= 0.5).mean()) if gt.size else 0.0
    pr_frac = float((pred >= pred_threshold).mean()) if pred.size else 0.0
    inter_frac = (
        float((inter >= pred_threshold).mean()) if (inter is not None and inter.size) else None
    )
    has_inter_emit = inter is not None and not skip_intermediate
    inter_msg = (
        f" | inter>={pred_threshold} {100.0 * (inter_frac or 0.0):.4f}%" if inter is not None else ""
    )
    print(
        f"Mask stats (row {row_index}): GT {100.0 * gt_frac:.4f}% | "
        f"pred>={pred_threshold} {100.0 * pr_frac:.4f}%{inter_msg} | pred_display={pred_display!r}",
        file=sys.stderr,
    )

    if not skip_rgb:
        write_ply(geo.xyz_middle, geo.rgb_unit, base + "_rgb.ply")
    if not skip_gt:
        c_gt = build_mask_overlay(geo.rgb_unit, gt, 0.5, background_scale, "gt")
        write_ply(geo.xyz_middle, c_gt, base + "_gt.ply")
    if not skip_pred:
        c_pr = build_mask_overlay(
            geo.rgb_unit, pred, pred_threshold, background_scale, "pred"
        )
        write_ply(geo.xyz_middle, c_pr, base + "_pred.ply")
        if heatmap_pred:
            write_ply(
                geo.xyz_middle,
                build_confidence_colors(geo.rgb_unit, pred),
                base + "_pred_heat.ply",
            )
    if has_inter_emit:
        c_inter = build_mask_overlay(
            geo.rgb_unit, inter, pred_threshold, background_scale, "intermediate"
        )
        write_ply(geo.xyz_middle, c_inter, base + "_pred_intermediate.ply")
        if heatmap_pred:
            write_ply(
                geo.xyz_middle,
                build_confidence_colors(geo.rgb_unit, inter),
                base + "_pred_intermediate_heat.ply",
            )

    meta = {
        "row_index": row_index,
        "eval_save_index": row.get("eval_save_index"),
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
        "gt_foreground_fraction": gt_frac,
        "pred_positive_fraction": pr_frac,
        "pred_display": pred_display,
        "overlay_background_scale": background_scale,
    }
    # Forward chainv3-CoT diagnostics into the per-row meta JSON when present.
    for k in (
        "did_two_pass",
        "intermediate_point_iou",
        "intermediate_in_npz",
        "decoded_text_pass1",
        "n_seg_pass1",
        "n_seg_pass2",
        "decoded_text",
        "question_type",
    ):
        if k in row:
            meta[k] = row[k]
    if inter is not None:
        meta["intermediate_in_npz_loaded"] = True
        meta["intermediate_pred_positive_fraction"] = inter_frac
        meta["intermediate_emitted_ply"] = has_inter_emit
    with open(base + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    with open(base + "_caption.txt", "w", encoding="utf-8") as f:
        f.write(str(row.get("text_input", "")) + "\n")
        # Pass-1 text is what queried the intermediate mask; surfacing it in
        # the caption makes it easy to diff against the final decoded_text.
        p1_text = row.get("decoded_text_pass1")
        if p1_text:
            f.write(f"\n[chainv3_cot pass1 decoded] {p1_text}\n")

    suffixes: List[str] = []
    if not skip_rgb:
        suffixes.append("_rgb.ply")
    if not skip_gt:
        suffixes.append("_gt.ply")
    if not skip_pred:
        suffixes.append("_pred.ply")
    if has_inter_emit:
        suffixes.append("_pred_intermediate.ply")
    if verbose:
        print("Wrote:")
        for suf in suffixes:
            print(" ", base + suf)
        if heatmap_pred and not skip_pred:
            print(" ", base + "_pred_heat.ply")
        if heatmap_pred and has_inter_emit:
            print(" ", base + "_pred_intermediate_heat.ply")
        print(" ", base + "_meta.json")
        print(" ", base + "_caption.txt")
    else:
        extras = []
        if heatmap_pred and not skip_pred:
            extras.append("heat")
        if heatmap_pred and has_inter_emit:
            extras.append("inter_heat")
        extra = (" +" + "+".join(extras)) if extras else ""
        names = ",".join(s.lstrip("_").removesuffix(".ply") for s in suffixes)
        plys = f"{{{names}}}.ply" if len(suffixes) > 1 else (suffixes[0].lstrip("_") if suffixes else "")
        print(f"  -> {os.path.basename(base)}_{plys}{extra} +meta +caption", flush=True)


def cmd_export_all(
    rows: List[Dict[str, Any]],
    qual_dir: str,
    pts_root: str,
    pth_subdir: str,
    out_dir: str,
    pred_threshold: float,
    heatmap_pred: bool,
    stride: int,
    pred_display: str,
    background_scale: float,
    skip_gt: bool = False,
    skip_rgb: bool = True,
    skip_intermediate: bool = False,
    skip_pred: bool = False,
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
            pred_display=pred_display,
            background_scale=background_scale,
            skip_gt=skip_gt,
            skip_rgb=skip_rgb,
            skip_intermediate=skip_intermediate,
            skip_pred=skip_pred,
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
    p.add_argument(
        "--row-indices-file",
        type=str,
        default=None,
        metavar="PATH",
        help="One 0-based predictions.jsonl row index per line (e.g. from sample_surprise_predictions.py). "
        "Lines with -1 are skipped. Requires mask .npz for each row (not compatible with --export-all).",
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
    gt_group = p.add_mutually_exclusive_group()
    gt_group.add_argument(
        "--skip-gt",
        dest="skip_gt",
        action="store_true",
        default=False,
        help="Do not write the *_gt.ply overlay (saves disk + time when only pred matters).",
    )
    gt_group.add_argument(
        "--with-gt",
        dest="skip_gt",
        action="store_false",
        help="Force writing *_gt.ply even when an upstream caller defaulted to --skip-gt.",
    )
    rgb_group = p.add_mutually_exclusive_group()
    rgb_group.add_argument(
        "--skip-rgb",
        dest="skip_rgb",
        action="store_true",
        default=True,
        help="Do not write the *_rgb.ply scene point cloud (default).",
    )
    rgb_group.add_argument(
        "--with-rgb",
        dest="skip_rgb",
        action="store_false",
        help="Also write *_rgb.ply (the unmasked scene RGB point cloud).",
    )
    inter_group = p.add_mutually_exclusive_group()
    inter_group.add_argument(
        "--skip-intermediate",
        dest="skip_intermediate",
        action="store_true",
        default=False,
        help="Do not write *_pred_intermediate.ply even when the .npz contains "
        "pred_pmask_intermediate (chainv3-CoT runs).",
    )
    inter_group.add_argument(
        "--with-intermediate",
        dest="skip_intermediate",
        action="store_false",
        help="Force *_pred_intermediate.ply when the .npz carries it (default behavior).",
    )
    p.add_argument(
        "--intermediate-only",
        action="store_true",
        default=False,
        help="Skip *_pred.ply (final M_2 overlay) and only render the intermediate M_1 PLY. "
        "Useful for landmark-only inspection on chainv3-CoT runs. No effect on non-CoT npz.",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every k-th point for lighter PLYs (default 1 = full resolution).",
    )
    p.add_argument(
        "--pred-display",
        type=str,
        choices=("auto", "raw", "sigmoid"),
        default="auto",
        help="How to interpret pred_pmask in npz before thresholding (see refer_seg_task save heuristic).",
    )
    p.add_argument(
        "--overlay-background-scale",
        type=float,
        default=0.35,
        metavar="S",
        help="For *_gt.ply / *_pred.ply, multiply non-highlighted RGB by S (0–1). Lower = stronger contrast. "
        "Use 1.0 to restore old behavior (highlight on full RGB).",
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
    if args.row_indices_file and (args.export_all or args.export_row is not None):
        raise SystemExit("Use --row-indices-file alone for subset export (not with --export-all / --export-row).")

    if args.export_row is not None or args.export_all or args.row_indices_file:
        if int(args.stride) < 1:
            raise SystemExit("--stride must be >= 1")
        if not str(args.pts_root).strip():
            raise SystemExit("Set --pts-root or REASON3D_PTS_ROOT to the ScanNet++/surprise points root.")
        if args.intermediate_only and args.skip_intermediate:
            raise SystemExit(
                "--intermediate-only and --skip-intermediate are mutually exclusive."
            )
        st = max(1, int(args.stride))
        out_abs = os.path.abspath(args.out_dir)
        bg = float(args.overlay_background_scale)
        if not (0.0 <= bg <= 1.0):
            raise SystemExit("--overlay-background-scale must be between 0 and 1")
        pdisp = str(args.pred_display)
        skip_pred = bool(args.intermediate_only)
        skip_intermediate = bool(args.skip_intermediate)
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
                pred_display=pdisp,
                background_scale=bg,
                skip_gt=args.skip_gt,
                skip_rgb=args.skip_rgb,
                skip_intermediate=skip_intermediate,
                skip_pred=skip_pred,
            )
        elif args.row_indices_file:
            idx_path = os.path.abspath(args.row_indices_file)
            if not os.path.isfile(idx_path):
                raise SystemExit(f"Missing --row-indices-file {idx_path}")
            indices: List[int] = []
            with open(idx_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    indices.append(int(line))
            n_ok = 0
            for line_no, idx in enumerate(indices):
                if idx < 0:
                    print(f"skip line {line_no + 1}: index {idx}", file=sys.stderr)
                    continue
                if idx >= len(rows):
                    print(f"skip line {line_no + 1}: index {idx} out of range [0,{len(rows)-1}]", file=sys.stderr)
                    continue
                print(f"[{n_ok + 1}/{len(indices)}] row {idx} scene={rows[idx].get('scene_id')!r}", flush=True)
                cmd_export(
                    rows,
                    idx,
                    qual_dir=qual_dir,
                    pts_root=os.path.expanduser(args.pts_root),
                    pth_subdir=args.pth_subdir,
                    out_dir=out_abs,
                    pred_threshold=args.pred_threshold,
                    heatmap_pred=args.heatmap_pred,
                    stride=st,
                    pred_display=pdisp,
                    background_scale=bg,
                    skip_gt=args.skip_gt,
                    skip_rgb=args.skip_rgb,
                    skip_intermediate=skip_intermediate,
                    skip_pred=skip_pred,
                    verbose=False,
                )
                n_ok += 1
            print(f"Done. Exported {n_ok} rows under {out_abs}", flush=True)
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
                pred_display=pdisp,
                background_scale=bg,
                skip_gt=args.skip_gt,
                skip_rgb=args.skip_rgb,
                skip_intermediate=skip_intermediate,
                skip_pred=skip_pred,
                verbose=True,
            )

    if args.plot_iou_hist:
        cmd_plot_iou(rows, args.plot_iou_hist)

    if not (
        args.list
        or args.export_row is not None
        or args.export_all
        or args.row_indices_file
        or args.plot_iou_hist
    ):
        p.print_help()
        print(
            "\nNo action: pass --list, --export-row N, --export-all, --row-indices-file, and/or --plot-iou-hist out.png",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
