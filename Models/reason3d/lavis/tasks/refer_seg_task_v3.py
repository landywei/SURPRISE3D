"""
Chain v3 evaluation task for Reason3D.

Adds the per-instance ``hit@tau`` / ``max_per_instance_iou`` metric on top of
the union-based ``mIoU / Acc@0.25 / Acc@0.50`` already computed by
``ThreeDReferSegTask``. The new metric requires the dataset to ship per-instance
GT lists in ``samples["gt_pmasks_per_instance"]`` (provided by
``ThreeDReferDatasetChainV3``); when those are absent, the new fields are
reported as NaN and the v3 task collapses to the base behavior.

Also persists the chain-v3 CoT intermediate (M_1) mask when the model is
``Reason3DT5ChainV3CoT`` and the two-pass decode actually fires:

- ``predict_seg`` returns ``intermediate_sp_masks`` (list aligned with the
  predict batch; B=1 here) and a ``chainv3_cot`` diagnostics dict.
- When ``run.save_eval_prediction_masks`` is on, the per-row .npz gains a
  ``pred_pmask_intermediate`` array (sigmoided point probabilities) next to
  the existing ``pred_pmask`` / ``gt_pmask``.
- ``predictions.jsonl`` rows gain ``did_two_pass``, ``decoded_text_pass1``,
  ``n_seg_pass1``, ``n_seg_pass2``, ``intermediate_point_iou`` and
  ``intermediate_in_npz``. Rows from non-CoT models leave the CoT-only fields
  unset and ``intermediate_point_iou`` is ``null``.

Registry id: ``3d_refer_seg_v3``.
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any, Dict, List

import numpy as np
import torch

from lavis.common.dist_utils import is_main_process
from lavis.common.registry import registry
from lavis.models.reason3d_models.seg_loss import get_iou
from lavis.tasks.refer_seg_task import ThreeDReferSegTask


def _scalar(x: Any) -> float:
    if torch.is_tensor(x):
        return float(x.detach().cpu())
    return float(x)


def _is_real_number(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if not isinstance(v, (int, float)):
        return False
    return not math.isnan(float(v))


@registry.register_task("3d_refer_seg_v3")
class ThreeDReferSegTaskV3(ThreeDReferSegTask):
    """``3d_refer_seg`` plus per-instance ``hit@tau`` / max-IoU on samples that ship per-instance GT."""

    # ------------------------------------------------------------------
    # valid_step: compute base metrics + per-instance metrics, write JSONL.
    # ------------------------------------------------------------------
    def valid_step(self, model, samples):
        rep_pen = float(getattr(self, "decode_repetition_penalty", 1.0))
        ngram = int(getattr(self, "decode_no_repeat_ngram_size", 0))
        result = model.predict_seg(
            samples=samples,
            answer_list=None,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
            prompt=self.prompt,
            repetition_penalty=rep_pen,
            no_repeat_ngram_size=ngram,
        )
        decoded_text = result.get("decoded_text", "")

        assert len(samples["gt_pmasks"]) == 1, "current only support batch size = 1"
        gt_pmask = samples["gt_pmasks"][0]
        gt_spmask = samples["gt_spmasks"][0]
        pred_spmask = result["masks"][-1].squeeze()
        spiou = get_iou(pred_spmask, gt_spmask, pred_confidence=model.pred_confidence)
        pred_pmask = pred_spmask[samples["superpoints"]]
        piou = get_iou(pred_pmask, gt_pmask, pred_confidence=model.pred_confidence)

        # Chain-of-thought intermediate (M_1) mask, only present for the
        # ``reason3d_t5_chainv3_cot`` model when pass-1 emitted >= 2 [SEG]s.
        # ``intermediate_sp_masks`` is a list aligned with the predict_seg batch
        # (B=1 here); each entry is a 1-D superpoint logit tensor or ``None``.
        cot_meta = result.get("chainv3_cot", {}) if isinstance(result, dict) else {}
        inter_sp = None
        inter_list = result.get("intermediate_sp_masks") if isinstance(result, dict) else None
        if inter_list:
            inter_sp = inter_list[0]
        inter_pmask = None
        inter_piou = None
        if inter_sp is not None:
            inter_pmask = inter_sp[samples["superpoints"]]
            inter_piou = get_iou(
                inter_pmask, gt_pmask, pred_confidence=model.pred_confidence
            )

        # Per-instance hit@tau (v3-only). NaN when per-instance GT is absent.
        max_inst_iou = float("nan")
        hit_25 = float("nan")
        hit_50 = float("nan")
        per_inst_ious: List[float] = []
        if samples.get("gt_pmasks_per_instance") is not None:
            inst_list = samples["gt_pmasks_per_instance"][0] or []
            for m in inst_list:
                m_dev = m.to(pred_pmask.device).float()
                iou_k = get_iou(pred_pmask, m_dev, pred_confidence=model.pred_confidence)
                per_inst_ious.append(_scalar(iou_k))
            if per_inst_ious:
                max_inst_iou = max(per_inst_ious)
                hit_25 = 1.0 if max_inst_iou >= 0.25 else 0.0
                hit_50 = 1.0 if max_inst_iou >= 0.50 else 0.0

        result_record = dict(
            scan_id=samples["scan_ids"][0],
            object_id=samples["object_ids"][0],
            ann_id=samples["ann_ids"][0],
            piou=piou,
            spiou=spiou,
            gt_pmask=gt_pmask,
            pred_pmask=pred_pmask,
            max_per_instance_iou=max_inst_iou,
            hit_at_25=hit_25,
            hit_at_50=hit_50,
            per_instance_ious=per_inst_ious,
        )

        if getattr(self, "_save_preds_ok", False) and is_main_process():
            scan_id = result_record["scan_id"]
            ann_id = result_record["ann_id"]
            ann_key = ann_id.item() if torch.is_tensor(ann_id) else int(ann_id)
            text_in = samples["text_input"]
            if isinstance(text_in, (list, tuple)):
                text_in = text_in[0]
            oid = result_record["object_id"]
            if torch.is_tensor(oid):
                oid = oid.detach().cpu().tolist()
            save_idx = self._eval_save_idx
            mask_rel = None
            has_inter_in_npz = False
            if self.save_eval_prediction_masks and self._mask_dir is not None:
                mask_name = f"{scan_id}_{ann_key}_{save_idx:06d}.npz"
                mask_path = os.path.join(self._mask_dir, mask_name)
                pred_np = pred_pmask.detach().float().cpu().numpy().reshape(-1)
                if pred_np.size and (pred_np.max() > 1.0 or pred_np.min() < 0.0):
                    pred_np = (
                        torch.sigmoid(pred_pmask).detach().float().cpu().numpy().reshape(-1)
                    )
                gt_np = gt_pmask.detach().float().cpu().numpy().reshape(-1)
                save_kwargs: Dict[str, np.ndarray] = {
                    "pred_pmask": pred_np.astype(np.float16),
                    "gt_pmask": gt_np.astype(np.float16),
                }
                # Add the chainv3-CoT intermediate (M_1) point mask alongside the
                # final pred/gt so a single .npz fully describes the row.
                if inter_pmask is not None:
                    inter_np = inter_pmask.detach().float().cpu().numpy().reshape(-1)
                    if inter_np.size and (inter_np.max() > 1.0 or inter_np.min() < 0.0):
                        inter_np = (
                            torch.sigmoid(inter_pmask)
                            .detach()
                            .float()
                            .cpu()
                            .numpy()
                            .reshape(-1)
                        )
                    save_kwargs["pred_pmask_intermediate"] = inter_np.astype(np.float16)
                    has_inter_in_npz = True
                np.savez_compressed(mask_path, **save_kwargs)
                mask_rel = os.path.join("qualitative", "masks", mask_name)
            self._eval_save_idx = save_idx + 1
            sp_fn = ""
            if samples.get("sp_filenames") is not None:
                sp_fn = samples["sp_filenames"][0]
                if isinstance(sp_fn, bytes):
                    sp_fn = sp_fn.decode("utf-8", errors="replace")
            qt = samples.get("question_types", [""])[0] or ""
            row: Dict[str, Any] = {
                "eval_save_index": save_idx,
                "scene_id": scan_id,
                "ann_id": ann_key,
                "object_id": oid,
                "question_type": qt,
                "text_input": text_in,
                "decoded_text": decoded_text,
                "point_iou": _scalar(piou),
                "superpoint_iou": _scalar(spiou),
                "max_per_instance_iou": max_inst_iou,
                "hit_at_25": hit_25,
                "hit_at_50": hit_50,
                "per_instance_ious": [float(x) for x in per_inst_ious],
                "mask_npz": mask_rel,
                "sp_filename": sp_fn,
            }
            # chainv3-CoT diagnostics: surface whether the two-pass branch
            # actually fired and how the intermediate mask scored, plus the
            # pass-1 decoded text used to query M_1.
            if cot_meta:
                row["did_two_pass"] = bool(cot_meta.get("did_two_pass", False))
                p1_text = cot_meta.get("decoded_text_pass1")
                if isinstance(p1_text, (list, tuple)):
                    p1_text = p1_text[0] if p1_text else ""
                row["decoded_text_pass1"] = p1_text
                n_seg_p1 = cot_meta.get("n_seg_pass1") or []
                n_seg_p2 = cot_meta.get("n_seg_pass2") or []
                row["n_seg_pass1"] = int(n_seg_p1[0]) if n_seg_p1 else 0
                row["n_seg_pass2"] = int(n_seg_p2[0]) if n_seg_p2 else 0
            row["intermediate_point_iou"] = (
                _scalar(inter_piou) if inter_piou is not None else None
            )
            row["intermediate_in_npz"] = has_inter_in_npz
            with open(self._pred_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        return [{"result": result_record}]

    # ------------------------------------------------------------------
    # Helpers for after_evaluation
    # ------------------------------------------------------------------
    def _collect_eval_rows(self, val_result) -> List[Dict[str, Any]]:
        """Normalize evaluation rows into a single list of dicts with keys
        ``piou`` / ``max_per_instance_iou`` / ``hit_at_25`` / ``hit_at_50``.

        Prefers the on-disk ``predictions.jsonl`` when it is available
        (matches whatever the base task already aggregates from); otherwise
        falls back to the in-memory ``val_result``.
        """
        rows: List[Dict[str, Any]] = []
        jsonl_path = getattr(self, "_pred_jsonl", None)
        jsonl_ok = (
            self.save_eval_predictions
            and jsonl_path
            and os.path.isfile(jsonl_path)
            and os.path.getsize(jsonl_path) > 0
        )
        if jsonl_ok:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
                    rows.append(
                        {
                            "piou": float(raw.get("point_iou", float("nan"))),
                            "max_per_instance_iou": raw.get("max_per_instance_iou"),
                            "hit_at_25": raw.get("hit_at_25"),
                            "hit_at_50": raw.get("hit_at_50"),
                        }
                    )
        elif val_result:
            for entry in val_result:
                r = entry.get("result", {}) if isinstance(entry, dict) else {}
                rows.append(
                    {
                        "piou": float(_scalar(r.get("piou", float("nan")))),
                        "max_per_instance_iou": r.get("max_per_instance_iou"),
                        "hit_at_25": r.get("hit_at_25"),
                        "hit_at_50": r.get("hit_at_50"),
                    }
                )
        return rows

    @staticmethod
    def _safe_mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    # ------------------------------------------------------------------
    # after_evaluation: base mIoU/Acc + per-instance hit@tau + tracker row.
    # ------------------------------------------------------------------
    def after_evaluation(self, val_result, split_name, epoch):
        # Run the base aggregation first (writes the standard mIoU/Acc log line).
        super().after_evaluation(val_result, split_name, epoch)

        if not is_main_process():
            return

        rows = self._collect_eval_rows(val_result)
        if not rows:
            logging.info(
                "ThreeDReferSegTaskV3.after_evaluation: no eval rows; skipping summary."
            )
            return

        # Union (single-mask) metrics — recomputed here so we have all 6 numbers
        # in one place to print. Matches the base task's recipe (Acc@tau is
        # computed against piou; mIoU is the mean piou).
        pious = np.asarray(
            [r["piou"] for r in rows if _is_real_number(r.get("piou"))],
            dtype=np.float64,
        )
        n_union = int(pious.size)
        miou = float(pious.mean()) if n_union else float("nan")
        acc25 = float((pious > 0.25).sum() / n_union) if n_union else float("nan")
        acc50 = float((pious > 0.50).sum() / n_union) if n_union else float("nan")

        # Per-instance metrics (NaN when the dataset is not chainv3).
        max_ious = [
            float(r["max_per_instance_iou"])
            for r in rows
            if _is_real_number(r.get("max_per_instance_iou"))
        ]
        h25 = [
            float(r["hit_at_25"]) for r in rows if _is_real_number(r.get("hit_at_25"))
        ]
        h50 = [
            float(r["hit_at_50"]) for r in rows if _is_real_number(r.get("hit_at_50"))
        ]
        m_max = self._safe_mean(max_ious)
        m_h25 = self._safe_mean(h25)
        m_h50 = self._safe_mean(h50)
        n_perinst = len(max_ious)

        # Output dir for the metrics JSON (best-effort).
        out_dir = ""
        try:
            out_dir = str(registry.get_path("output_dir"))
        except Exception:  # noqa: BLE001 - registry path optional
            out_dir = ""

        # ---- Copy-paste-friendly summary block ------------------------------
        bar = "=" * 78
        print()
        print(bar)
        print("                          chainv3 metrics")
        print(bar)
        print(
            "split={}  epoch={}  n_union={}  n_per_inst={}  output_dir={}".format(
                split_name, epoch, n_union, n_perinst, out_dir or "<unknown>"
            )
        )
        print(
            "Union (single mask):        mIoU={:.4f}  Acc@0.25={:.4f}  Acc@0.50={:.4f}".format(
                miou, acc25, acc50
            )
        )
        print(
            "Per-instance (best of GT):  meanMaxIoU={:.4f}  hit@0.25={:.4f}  hit@0.50={:.4f}".format(
                m_max, m_h25, m_h50
            )
        )
        print()
        # Backward-compatible single line some tooling may grep for.
        print(
            "Val v3 per-instance: meanMaxIoU/hit@0.25/hit@0.50 "
            f"{m_max:.4f}/{m_h25:.4f}/{m_h50:.4f} (n={n_perinst})"
        )
        # Markdown tracker row, columns match Models/reason3d/docs/chainv3_ablation_tracker.md
        # Table B (epoch | mIoU | Acc@0.25 | Acc@0.50 | meanMaxIoU | hit@0.25 | hit@0.50).
        metric_cells = "| {epoch} | {miou:.4f} | {acc25:.4f} | {acc50:.4f} | {m_max:.4f} | {h25:.4f} | {h50:.4f} |".format(
            epoch=epoch, miou=miou, acc25=acc25, acc50=acc50, m_max=m_max, h25=m_h25, h50=m_h50
        )
        print()
        print("Tracker row (chainv3_ablation_tracker.md Table B, metric cells only):")
        print("  | epoch | mIoU | Acc@0.25 | Acc@0.50 | meanMaxIoU | hit@0.25 | hit@0.50 |")
        print("  " + metric_cells)
        print()
        print("Full tracker row (fill in #/run-name/job-id/started/wall-clock/status/notes):")
        full_row = (
            "| _#_ | _run-name_ | _jobid_ | "
            f"{out_dir or '_outdir_'} | _started_ | _wall_ | {metric_cells.strip('|').strip()} "
            "| done | _notes_ |"
        )
        print("  " + full_row)
        print(bar)
        print()

        # ---- metrics_v3_<split>.json for programmatic post-processing -------
        if out_dir:
            metrics = {
                "split": split_name,
                "epoch": epoch,
                "n_union": n_union,
                "n_per_instance": n_perinst,
                "miou": miou,
                "acc25": acc25,
                "acc50": acc50,
                "mean_max_iou": m_max,
                "hit25": m_h25,
                "hit50": m_h50,
                "tracker_row_metric_cells": metric_cells,
                "output_dir": out_dir,
            }
            try:
                metrics_path = os.path.join(out_dir, f"metrics_v3_{split_name}.json")
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)
                print(f"Wrote {metrics_path}")
            except OSError as exc:
                logging.warning(
                    "ThreeDReferSegTaskV3: failed to write metrics_v3_%s.json: %s",
                    split_name,
                    exc,
                )
