"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import logging
import os
import shutil
from typing import Set, Tuple

import numpy as np
import torch

from lavis.common.dist_utils import is_main_process
from lavis.common.registry import registry
from lavis.models.reason3d_models.seg_loss import get_iou
from lavis.tasks.base_task import BaseTask


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

    def valid_step(self, model, samples):
        #print(samples["gt_spmasks"][0].shape)
        # getattr: tolerate partial file syncs where __init__/setup_task lack decode_* fields
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
        result = dict(
            scan_id=samples["scan_ids"][0],
            object_id=samples["object_ids"][0],
            ann_id=samples["ann_ids"][0],
            piou=piou,
            spiou=spiou,
            gt_pmask=gt_pmask,
            pred_pmask=pred_pmask,
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
            with open(self._pred_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        return [{"result": result}]

    
    def after_evaluation(self, val_result, split_name, epoch):
        pious_list = []
        spious_list = []

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
                        pious_list.append(float(row["point_iou"]))
                        spious_list.append(float(row.get("superpoint_iou", row.get("spiou", 0.0))))
                    except (json.JSONDecodeError, TypeError, ValueError, KeyError):
                        continue

        if len(pious_list) == 0:
            if not val_result:
                if is_main_process():
                    logging.warning("after_evaluation: empty val_result and no jsonl metrics; skipping.")
                return
            for i, result in enumerate(val_result):
                piou = result["result"]["piou"]
                spiou = result["result"]["spiou"]
                pious_list.append(float(_scalar(piou)))
                spious_list.append(float(_scalar(spiou)))
            pious = np.asarray(pious_list, dtype=np.float64)
            spious = np.asarray(spious_list, dtype=np.float64)
            used_jsonl = False
        else:
            pious = np.asarray(pious_list, dtype=np.float64)
            spious = np.asarray(spious_list, dtype=np.float64)
            used_jsonl = jsonl_ok

        precision_half = (pious > 0.5).sum().astype(float) / pious.size
        precision_quarter = (pious > 0.25).sum().astype(float) / pious.size
        miou = float(pious.mean())

        msg = "Val result: mIoU/Acc50/Acc25 {:.4f}/{:.4f}/{:.4f} (n={})".format(
            miou, precision_half, precision_quarter, int(pious.size)
        )
        if used_jsonl:
            msg += " [metrics from full qualitative/predictions.jsonl]"
        if is_main_process():
            print(msg)