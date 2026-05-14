"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import gc
import json
import logging
import os
import shutil
import traceback as _tb
from typing import Any, Dict, Set, Tuple

import numpy as np
import torch

from lavis.common.dist_utils import is_main_process
from lavis.common.registry import registry
from lavis.models.reason3d_models.seg_loss import get_iou
from lavis.tasks.base_task import BaseTask


def _is_oom_exc(exc: BaseException) -> bool:
    """True iff ``exc`` looks like a CUDA OOM.

    Covers both the modern ``torch.cuda.OutOfMemoryError`` and the older
    ``RuntimeError`` whose message contains ``out of memory`` (both the
    PyTorch CUDA caching-allocator OOM and the cuBLAS / cuDNN flavours).
    """
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    msg = str(exc) if exc is not None else ""
    return "out of memory" in msg.lower()


def _drop_exception_frames(exc: BaseException) -> None:
    """Clear locals stored in the exception's traceback frames.

    PyTorch's caching allocator only releases blocks no live tensor points
    at. When ``predict_seg`` OOMs, the resulting exception's traceback
    keeps every frame from the failed call alive -- and those frames'
    locals still hold pointers to large GPU activations. Without this
    call, ``torch.cuda.empty_cache()`` is a no-op for those blocks and
    every subsequent row sees the same near-OOM allocator state, which
    is exactly what the run on 2026-05-13 hit on scene 578511c8a9 (15+
    consecutive OOMs at row 2130+, only 972 MiB free even after our
    ``_free_cuda`` returned). Calling ``traceback.clear_frames`` on the
    exception's traceback chain breaks those references, after which
    ``empty_cache`` actually returns the cached blocks to the driver.
    """
    tb = getattr(exc, "__traceback__", None)
    if tb is None:
        return
    try:
        _tb.clear_frames(tb)
    except Exception:  # noqa: BLE001
        pass


def _free_cuda() -> None:
    """Best-effort GPU memory release after a failed forward.

    Call site contract: callers should clear the exception's traceback
    frames (see ``_drop_exception_frames``) *before* this function so
    the caching allocator can actually reclaim the failed forward's
    activations; otherwise this is largely a no-op. ``synchronize`` is
    wrapped because it itself can raise on a wedged context; the
    empty_cache / gc.collect path is the one that actually frees.
    """
    try:
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _env_int(name: str, default: int) -> int:
    """Parse an int env var with a default; never raise on bad input."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        logging.warning("Bad int for env %s=%r; falling back to %d.", name, raw, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _scalar(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu())
    return float(x)


@registry.register_task("3d_refer_seg")
class ThreeDReferSegTask(BaseTask):

    def __init__(
        self,
        num_beams,
        max_len,
        min_len,
        evaluate,
        num_ans_candidates,
        inference_method="rank",
        prompt="",
        save_eval_predictions=False,
        eval_resume_predictions=False,
        save_eval_prediction_masks=True,
        decode_repetition_penalty=1.0,
        decode_no_repeat_ngram_size=0,
    ):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.decode_repetition_penalty = float(decode_repetition_penalty)
        self.decode_no_repeat_ngram_size = int(decode_no_repeat_ngram_size)

        self.evaluate = evaluate
        self.inference_method = inference_method
        self.num_ans_candidates = num_ans_candidates
        self.prompt = prompt
        self.save_eval_predictions = save_eval_predictions
        self.eval_resume_predictions = bool(eval_resume_predictions)
        self.save_eval_prediction_masks = bool(save_eval_prediction_masks)
        self._save_preds_ok = False
        self._qual_dir = None
        self._mask_dir = None
        self._pred_jsonl = None
        self._eval_save_idx = 0
        self.num = 0
        self.answer_list = None

        self.ques_files = dict()
        self.anno_files = dict()
        self._pred_jsonl = None

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.get("num_beams", 3)
        max_len = run_cfg.get("max_len", 10)
        min_len = run_cfg.get("min_len", 1)

        evaluate = run_cfg.get("evaluate", False)

        inference_method = run_cfg.get("inference_method", "rank")
        num_ans_candidates = run_cfg.get("num_ans_candidates", 128)
        prompt = run_cfg.get("prompt", "")
        save_eval_predictions = run_cfg.get("save_eval_predictions", False)
        eval_resume_predictions = run_cfg.get("eval_resume_predictions", False)
        save_eval_prediction_masks = run_cfg.get("save_eval_prediction_masks", True)
        decode_repetition_penalty = run_cfg.get("decode_repetition_penalty", 1.0)
        decode_no_repeat_ngram_size = run_cfg.get("decode_no_repeat_ngram_size", 0)

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            num_ans_candidates=num_ans_candidates,
            inference_method=inference_method,
            prompt=prompt,
            save_eval_predictions=save_eval_predictions,
            eval_resume_predictions=eval_resume_predictions,
            save_eval_prediction_masks=save_eval_prediction_masks,
            decode_repetition_penalty=decode_repetition_penalty,
            decode_no_repeat_ngram_size=decode_no_repeat_ngram_size,
        )

    @staticmethod
    def _load_completed_keys_from_jsonl(path: str) -> Tuple[Set[Tuple[str, int]], int]:
        """Returns (set of (scene_id, ann_id), next_eval_save_index)."""
        done: Set[Tuple[str, int]] = set()
        max_idx = -1
        if not path or not os.path.isfile(path):
            return done, 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    sid = str(row.get("scene_id", ""))
                    aid = int(row.get("ann_id", -1))
                    done.add((sid, aid))
                    max_idx = max(max_idx, int(row.get("eval_save_index", -1)))
                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    logging.warning("Skipping bad jsonl line in %r: %s", path, e)
        return done, max_idx + 1 if max_idx >= 0 else 0

    def prepare_eval_dataset_resume(self, dataset, split_name):
        if not self.eval_resume_predictions or not self.save_eval_predictions:
            return
        try:
            out = registry.get_path("output_dir")
        except Exception:
            logging.warning("eval_resume_predictions: registry has no output_dir; skipping resume filter.")
            return
        jsonl_path = os.path.join(out, "qualitative", "predictions.jsonl")
        done, _ = self._load_completed_keys_from_jsonl(jsonl_path)
        if not done:
            logging.info(
                "eval_resume_predictions: no completed rows in %r; running full eval.",
                jsonl_path,
            )
            return
        applier = getattr(dataset, "apply_eval_resume_skip", None)
        if applier is None:
            logging.warning(
                "eval_resume_predictions: dataset %r has no apply_eval_resume_skip; ignoring.",
                type(dataset).__name__,
            )
            return
        applier(done)

    def before_evaluation(self, model, dataset, **kwargs):
        super().before_evaluation(model, dataset, **kwargs)
        # Reset every eval call so a re-run with a long-lived task object
        # (e.g. unit tests) does not carry counts from the previous run.
        self._oom_count = 0
        self._oom_recovered_count = 0
        # Read OOM-retry knobs once per eval so the inner loop can branch
        # without re-parsing env vars per row.
        self._oom_retry_enabled = _env_bool("REASON3D_OOM_RETRY", True)
        self._oom_retry_num_beams = max(1, _env_int("REASON3D_OOM_RETRY_NUM_BEAMS", 1))
        self._oom_retry_max_len = max(
            1,
            _env_int("REASON3D_OOM_RETRY_MAX_LEN", min(int(self.max_len), 32)),
        )
        if is_main_process():
            logging.info(
                "OOM safety net: retry=%s num_beams=%d max_len=%d (set REASON3D_OOM_RETRY=0 to disable).",
                self._oom_retry_enabled,
                self._oom_retry_num_beams,
                self._oom_retry_max_len,
            )
        self._save_preds_ok = False
        self._pred_jsonl = None
        if not self.save_eval_predictions or not is_main_process():
            return
        try:
            out = registry.get_path("output_dir")
        except Exception:
            logging.warning("save_eval_predictions: registry has no output_dir; skipping disk save.")
            return
        self._qual_dir = os.path.join(out, "qualitative")
        self._mask_dir = (
            os.path.join(self._qual_dir, "masks") if self.save_eval_prediction_masks else None
        )
        self._pred_jsonl = os.path.join(self._qual_dir, "predictions.jsonl")
        resume = self.eval_resume_predictions and os.path.isfile(self._pred_jsonl) and os.path.getsize(self._pred_jsonl) > 0
        if resume:
            if self._mask_dir is not None:
                os.makedirs(self._mask_dir, exist_ok=True)
            _, self._eval_save_idx = self._load_completed_keys_from_jsonl(self._pred_jsonl)
            self._save_preds_ok = True
            logging.info(
                "Eval resume: appending predictions to %s (next eval_save_index=%d).",
                self._qual_dir,
                self._eval_save_idx,
            )
        else:
            shutil.rmtree(self._qual_dir, ignore_errors=True)
            os.makedirs(self._qual_dir, exist_ok=True)
            if self._mask_dir is not None:
                os.makedirs(self._mask_dir, exist_ok=True)
            self._eval_save_idx = 0
            self._save_preds_ok = True
            logging.info("Saving eval predictions under %s", self._qual_dir)

    # ------------------------------------------------------------------
    # OOM safety net
    # ------------------------------------------------------------------
    # ``valid_step`` wraps ``_valid_step_body`` in a try/except that turns
    # a CUDA OOM on a single row into either:
    #
    #   (a) a *recovered real result* via a one-shot retry with reduced
    #       beam search (``num_beams=1``) and clipped ``max_len``, OR
    #   (b) a deterministic *sentinel row* in ``predictions.jsonl``
    #       (``oom: true``, NaN ious) when the retry also OOMs / is
    #       disabled.
    #
    # Three guarantees fall out:
    #
    #   1) Reproducibility: every row in the (pinned) val set produces
    #      exactly one JSONL line (real, recovered-real, or sentinel),
    #      so ``n`` is byte-identical across runs.
    #   2) Honesty: NaN ious are excluded from the headline mIoU /
    #      Acc / meanMaxIoU / hit@tau via the existing ``_is_real_number``
    #      filter; ``n_oom`` and ``n_oom_recovered`` are logged
    #      alongside ``n`` so the carve-out is auditable.
    #   3) Auto-resume safety: the (scene_id, ann_id) of every row is in
    #      the JSONL, so on a manual retry ``apply_eval_resume_skip``
    #      treats it as done -- no "row keeps OOM-ing, auto-resume keeps
    #      retrying it" loop.
    #
    # Knobs (env vars; read once at ``before_evaluation`` time):
    #
    #   REASON3D_OOM_RETRY            : "1" (default) / "0"  -- per-row retry
    #   REASON3D_OOM_RETRY_NUM_BEAMS  : int, default 1
    #   REASON3D_OOM_RETRY_MAX_LEN    : int, default min(self.max_len, 32)
    #
    # Set ``REASON3D_OOM_RETRY=0`` to revert to the previous behaviour
    # (one OOM => sentinel immediately).
    def _build_oom_row_base(self, samples, oom_msg: str) -> Dict[str, Any]:
        """Sentinel JSONL row for a row that hit CUDA OOM.

        Subclasses (``ThreeDReferSegTaskV3``) extend this with their
        extra metric fields (``max_per_instance_iou`` etc.) all set to
        NaN so the v3 row aggregator drops them from headline metrics.
        """
        save_idx = getattr(self, "_eval_save_idx", 0)
        self._eval_save_idx = save_idx + 1
        scan_id = samples["scan_ids"][0]
        ann_id = samples["ann_ids"][0]
        ann_key = ann_id.item() if torch.is_tensor(ann_id) else int(ann_id)
        text_in = samples.get("text_input", "")
        if isinstance(text_in, (list, tuple)):
            text_in = text_in[0] if text_in else ""
        oid = samples["object_ids"][0]
        if torch.is_tensor(oid):
            oid = oid.detach().cpu().tolist()
        sp_fn = ""
        if samples.get("sp_filenames") is not None:
            sp_fn = samples["sp_filenames"][0]
            if isinstance(sp_fn, bytes):
                sp_fn = sp_fn.decode("utf-8", errors="replace")
        qt = samples.get("question_types", [""])[0] or ""
        return {
            "eval_save_index": save_idx,
            "scene_id": scan_id,
            "ann_id": ann_key,
            "object_id": oid,
            "question_type": qt,
            "text_input": text_in,
            "decoded_text": "",
            "point_iou": float("nan"),
            "superpoint_iou": float("nan"),
            "mask_npz": None,
            "sp_filename": sp_fn,
            "oom": True,
            "oom_error": (oom_msg[:512] if oom_msg else ""),
        }

    def _emit_oom_row(self, samples, oom_msg: str):
        """Append the sentinel JSONL row and return a NaN result entry.

        The result entry preserves the in-memory shape ``valid_step``
        normally returns, so downstream gather/aggregate code does not
        special-case the OOM path (NaN ious are already filtered out
        by the headline aggregator). ``_free_cuda`` is invoked once
        more here so a sentinel emitted *after* a failed retry leaves
        the allocator clean for the next row.
        """
        self._oom_count = getattr(self, "_oom_count", 0) + 1
        _free_cuda()
        sentinel_result = {
            "scan_id": samples["scan_ids"][0],
            "object_id": samples["object_ids"][0],
            "ann_id": samples["ann_ids"][0],
            "piou": float("nan"),
            "spiou": float("nan"),
            "gt_pmask": None,
            "pred_pmask": None,
            "oom": True,
        }
        if not (getattr(self, "_save_preds_ok", False) and is_main_process()):
            return [{"result": sentinel_result}]
        if not getattr(self, "_pred_jsonl", None):
            return [{"result": sentinel_result}]
        row = self._build_oom_row_base(samples, oom_msg)
        with open(self._pred_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return [{"result": sentinel_result}]

    def _ann_key_str(self, samples) -> Tuple[str, int]:
        scan_id = samples["scan_ids"][0] if samples.get("scan_ids") else "?"
        ann_id = samples["ann_ids"][0] if samples.get("ann_ids") else -1
        ann_key = ann_id.item() if torch.is_tensor(ann_id) else int(ann_id)
        return str(scan_id), int(ann_key)

    def valid_step(self, model, samples):
        # Path 1: normal forward.
        try:
            return self._valid_step_body(model, samples)
        except Exception as exc:  # noqa: BLE001 - intentionally broad; only OOM is swallowed
            if not _is_oom_exc(exc):
                raise
            scan_id, ann_key = self._ann_key_str(samples)
            oom_msg = str(exc)[:512]
            # Crucial for actually reclaiming GPU memory: drop the
            # exception's traceback frames before they'd otherwise keep
            # GPU activations from the failed forward alive.
            _drop_exception_frames(exc)
            del exc
            _free_cuda()
            logging.warning(
                "valid_step: CUDA OOM on row scene=%s ann=%s; will %sattempt fallback. msg=%s",
                scan_id,
                ann_key,
                "" if getattr(self, "_oom_retry_enabled", True) else "NOT ",
                oom_msg[:200],
            )

        # Path 2: optional one-shot retry with reduced beams / max_len.
        # Most "1.14 GiB short" OOMs we've seen disappear at num_beams=1
        # because the T5 KV cache (B x num_beams x layers x heads x ...)
        # is multiplied by the beam count.
        if getattr(self, "_oom_retry_enabled", True):
            override = {
                "num_beams": int(getattr(self, "_oom_retry_num_beams", 1)),
                "max_len": int(
                    getattr(self, "_oom_retry_max_len", min(int(self.max_len), 32))
                ),
            }
            try:
                result = self._valid_step_body(
                    model, samples, override_decode=override, oom_recovered_msg=None
                )
                # ``_valid_step_body`` already wrote the JSONL row when
                # save_preds is on; we annotate the in-memory entry with
                # ``oom_recovered`` so the aggregator can tally it.
                if isinstance(result, list) and result:
                    inner = result[0].get("result", {}) if isinstance(result[0], dict) else {}
                    if isinstance(inner, dict):
                        inner["oom_recovered"] = True
                self._oom_recovered_count = getattr(self, "_oom_recovered_count", 0) + 1
                logging.warning(
                    "valid_step: CUDA OOM recovered on row scene=%s ann=%s with num_beams=%d max_len=%d.",
                    scan_id,
                    ann_key,
                    override["num_beams"],
                    override["max_len"],
                )
                return result
            except Exception as exc2:  # noqa: BLE001
                if not _is_oom_exc(exc2):
                    raise
                retry_msg = str(exc2)[:512]
                _drop_exception_frames(exc2)
                del exc2
                _free_cuda()
                logging.warning(
                    "valid_step: retry also OOM'd on row scene=%s ann=%s; falling back to sentinel.",
                    scan_id,
                    ann_key,
                )
                return self._emit_oom_row(
                    samples,
                    "primary OOM: {}\nretry OOM: {}".format(oom_msg[:240], retry_msg[:240]),
                )

        # Path 3: retry disabled -> sentinel immediately.
        return self._emit_oom_row(samples, oom_msg)

    def _valid_step_body(self, model, samples, override_decode=None, oom_recovered_msg=None):
        """Run one row through ``predict_seg`` and emit the JSONL line.

        ``override_decode`` is a dict like ``{"num_beams": 1, "max_len": 32}``
        used by the OOM retry path to ask for a smaller-memory decode;
        when ``None``, the YAML-configured values are used. ``oom_recovered_msg``
        is reserved for symmetry with future hooks; the OOM-recovered flag is
        attached to the in-memory result by the wrapper, since we don't want
        to widen the JSONL schema only to mark a successful recovery.
        """
        # getattr: tolerate partial file syncs where __init__/setup_task lack decode_* fields
        rep_pen = float(getattr(self, "decode_repetition_penalty", 1.0))
        ngram = int(getattr(self, "decode_no_repeat_ngram_size", 0))
        if override_decode:
            num_beams = int(override_decode.get("num_beams", self.num_beams))
            max_len = int(override_decode.get("max_len", self.max_len))
        else:
            num_beams = int(self.num_beams)
            max_len = int(self.max_len)
        result = model.predict_seg(
            samples=samples,
            answer_list=None,
            inference_method=self.inference_method,
            num_beams=num_beams,
            max_len=max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
            prompt=self.prompt,
            repetition_penalty=rep_pen,
            no_repeat_ngram_size=ngram,
        )
        decoded_text = result.get("decoded_text", "")
        #print(samples.keys())
        #print(samples['text_input'])
        #print(self.prompt)
        #TODO: currently only support B = 1 when predict
        assert len(samples["gt_pmasks"]) == 1, 'current only support batch size = 1'
        #print(result['masks'][-1].squeeze().shape)
        gt_pmask = samples["gt_pmasks"][0]
        gt_spmask = samples["gt_spmasks"][0]
        pred_spmask = result['masks'][-1].squeeze()
        spiou = get_iou(pred_spmask, gt_spmask, pred_confidence = model.pred_confidence)
        pred_pmask = pred_spmask[samples["superpoints"]]
        piou = get_iou(pred_pmask, gt_pmask, pred_confidence = model.pred_confidence)
        #print('saving results')
        #if 'scene0011_00' == samples["scan_ids"][0] or 'scene0011_01' == samples["scan_ids"][0]:
        #os.makedirs(f'result/{self.num}',exist_ok  = True)
        #np.save(f'result/{self.num}/pred.npy',pred_pmask.cpu().numpy(),)
        #np.save(f'result/{self.num}/gt.npy',gt_pmask.cpu().numpy(),)
        
        #with open(f'result/{self.num}/question.txt','w') as f:
        #    f.writelines(samples['text_input'])
        #    f.writelines('\n')
        #    f.writelines(samples["scan_ids"][0])
        #self.num+=1
        # NOTE: do NOT keep ``gt_pmask`` / ``pred_pmask`` in the returned dict.
        # ``base_task.evaluation`` accumulates every valid_step's return into a
        # ``results`` list that lives until the loop ends. Returning per-point
        # CUDA tensors here was leaking ~6 MB of GPU memory per row, which on
        # 10174 rows of Surprise val grew to ~14 GiB on top of the 13 GiB of
        # fp32 model weights -- and is what caused the late-eval OOMs around
        # row 2130+ on scene 578511c8a9, even with the OOM safety net active.
        # The mask tensors are already written to disk by the np.savez_compressed
        # path below when ``save_eval_prediction_masks`` is on. ``piou`` / ``spiou``
        # are CPU-ified so the result dict holds no live CUDA references.
        result = dict(
            scan_id=samples["scan_ids"][0],
            object_id=samples["object_ids"][0],
            ann_id=samples["ann_ids"][0],
            piou=_scalar(piou),
            spiou=_scalar(spiou),
        )

        if getattr(self, "_save_preds_ok", False) and is_main_process():
            scan_id = result["scan_id"]
            ann_id = result["ann_id"]
            ann_key = ann_id.item() if torch.is_tensor(ann_id) else int(ann_id)
            text_in = samples["text_input"]
            if isinstance(text_in, (list, tuple)):
                text_in = text_in[0]
            oid = result["object_id"]
            if torch.is_tensor(oid):
                oid = oid.detach().cpu().tolist()
            # One npz per valid_step so mask_gt / mask_pred match this row's caption and object_id.
            # Filename includes eval_save_index (same as JSON field); monotonic on main process.
            save_idx = self._eval_save_idx
            mask_rel = None
            if self.save_eval_prediction_masks and self._mask_dir is not None:
                mask_name = f"{scan_id}_{ann_key}_{save_idx:06d}.npz"
                mask_path = os.path.join(self._mask_dir, mask_name)
                pred_np = pred_pmask.detach().float().cpu().numpy().reshape(-1)
                if pred_np.size and (pred_np.max() > 1.0 or pred_np.min() < 0.0):
                    pred_np = torch.sigmoid(pred_pmask).detach().float().cpu().numpy().reshape(-1)
                gt_np = gt_pmask.detach().float().cpu().numpy().reshape(-1)
                np.savez_compressed(mask_path, pred_pmask=pred_np.astype(np.float16), gt_pmask=gt_np.astype(np.float16))
                mask_rel = os.path.join("qualitative", "masks", mask_name)
            self._eval_save_idx = save_idx + 1
            sp_fn = ""
            if samples.get("sp_filenames") is not None:
                sp_fn = samples["sp_filenames"][0]
                if isinstance(sp_fn, bytes):
                    sp_fn = sp_fn.decode("utf-8", errors="replace")
            qt = samples.get("question_types", [""])[0] or ""
            row = {
                "eval_save_index": save_idx,
                "scene_id": scan_id,
                "ann_id": ann_key,
                "object_id": oid,
                "question_type": qt,
                "text_input": text_in,
                "decoded_text": decoded_text,
                "point_iou": _scalar(piou),
                "superpoint_iou": _scalar(spiou),
                "mask_npz": mask_rel,
                "sp_filename": sp_fn,
            }
            if override_decode:
                # Mark rows that needed the OOM-recovery decode so a
                # post-mortem can grep for them in predictions.jsonl.
                row["oom_recovered"] = True
                row["oom_fallback_num_beams"] = int(override_decode.get("num_beams", num_beams))
                row["oom_fallback_max_len"] = int(override_decode.get("max_len", max_len))
            with open(self._pred_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        return [{"result": result}]

    
    def after_evaluation(self, val_result, split_name, epoch):
        pious_list = []
        spious_list = []
        # n_total:         every JSONL row attempted (real + recovered + sentinel)
        # n_oom:           OOM rows that produced no mask (NaN ious; sentinel)
        # n_oom_recovered: OOM rows that succeeded on the reduced-beams retry
        #                  -> a real piou is in the row, but flagged so it's
        #                     auditable (e.g. you may want to recompute the
        #                     headline excluding them).
        n_total = 0
        n_oom = 0
        n_oom_recovered = 0

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
                        row = json.loads(line)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
                    n_total += 1
                    if bool(row.get("oom", False)):
                        n_oom += 1
                        continue
                    if bool(row.get("oom_recovered", False)):
                        n_oom_recovered += 1
                        # fall through: treat as a real row
                    try:
                        piou_val = float(row["point_iou"])
                    except (TypeError, ValueError, KeyError):
                        continue
                    if not np.isfinite(piou_val):
                        continue
                    spiou_val = row.get("superpoint_iou", row.get("spiou", 0.0))
                    try:
                        spiou_val = float(spiou_val)
                    except (TypeError, ValueError):
                        spiou_val = float("nan")
                    pious_list.append(piou_val)
                    spious_list.append(spiou_val)

        if len(pious_list) == 0:
            if not val_result:
                if is_main_process():
                    logging.warning("after_evaluation: empty val_result and no jsonl metrics; skipping.")
                return
            for i, result in enumerate(val_result):
                inner = result.get("result", {}) if isinstance(result, dict) else {}
                if bool(inner.get("oom", False)):
                    n_oom += 1
                    n_total += 1
                    continue
                if bool(inner.get("oom_recovered", False)):
                    n_oom_recovered += 1
                piou = inner.get("piou", float("nan"))
                spiou = inner.get("spiou", float("nan"))
                p = float(_scalar(piou))
                s = float(_scalar(spiou))
                n_total += 1
                if not np.isfinite(p):
                    continue
                pious_list.append(p)
                spious_list.append(s)
            pious = np.asarray(pious_list, dtype=np.float64)
            spious = np.asarray(spious_list, dtype=np.float64)
            used_jsonl = False
        else:
            pious = np.asarray(pious_list, dtype=np.float64)
            spious = np.asarray(spious_list, dtype=np.float64)
            used_jsonl = jsonl_ok

        if pious.size == 0:
            if is_main_process():
                logging.warning(
                    "after_evaluation: no real-IoU rows (n_total=%d, n_oom=%d); skipping headline metrics.",
                    n_total,
                    n_oom,
                )
            return

        precision_half = (pious > 0.5).sum().astype(float) / pious.size
        precision_quarter = (pious > 0.25).sum().astype(float) / pious.size
        miou = float(pious.mean())

        # Make ``self._oom_count`` (in-memory counter) and the JSONL-derived
        # ``n_oom`` (on-disk truth) agree when both are populated; the JSONL
        # value wins because it survives auto-resume across processes.
        n_oom = max(n_oom, int(getattr(self, "_oom_count", 0)))
        n_oom_recovered = max(
            n_oom_recovered, int(getattr(self, "_oom_recovered_count", 0))
        )
        msg = "Val result: mIoU/Acc50/Acc25 {:.4f}/{:.4f}/{:.4f} (n={} n_total={} n_oom={} n_oom_recovered={})".format(
            miou,
            precision_half,
            precision_quarter,
            int(pious.size),
            n_total,
            n_oom,
            n_oom_recovered,
        )
        if used_jsonl:
            msg += " [metrics from full qualitative/predictions.jsonl]"
        if is_main_process():
            print(msg)