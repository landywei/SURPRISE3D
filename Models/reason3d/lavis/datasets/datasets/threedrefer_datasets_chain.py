"""
Surprise3D / 3D-refer variant: training targets are ``<short rationale> [SEG].`` with an oracle
object name from the annotation JSON (e.g. ``object_name``), instead of only ``[SEG]``-style replies.

Eval / ``predict_seg`` is unchanged (free-form decode then mask from ``[SEG]``); this fork mainly
shapes the supervised LM loss during finetune.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from lavis.datasets.datasets.threedrefer_datasets import ThreeDReferDataset

# Keep ``[SEG]`` literal so Reason3D tokenization / pooling stay aligned with the baseline.
CHAIN_ANSWER_LIST: Tuple[str, ...] = (
    "The target object is {name}. [SEG].",
    "It is the {name}. [SEG].",
    "Segment the {name}. [SEG].",
    "The answer is the {name}. [SEG].",
    "{name}. [SEG].",
)


def _flatten_name_tokens(raw: Any, depth: int = 0) -> List[str]:
    """Turn ``object_name``-like values (str, list, nested list) into stripped leaf strings."""
    if raw is None or depth > 8:
        return []
    if isinstance(raw, (list, tuple)):
        out: List[str] = []
        for x in raw:
            out.extend(_flatten_name_tokens(x, depth + 1))
        return out
    s = str(raw).strip().replace("\n", " ").replace("\r", " ")
    return [s] if s else []


def _format_object_name_phrase(raw: Any) -> str:
    """
    Human-readable single phrase for chain templates (no ``str(list)`` artifacts).
    Repeats collapse to one token; multiple distinct tokens join as ``a, b``.
    """
    tokens = _flatten_name_tokens(raw)
    if not tokens:
        return ""
    uniq = list(dict.fromkeys(tokens))
    return ", ".join(uniq)


class ThreeDReferDatasetChain(ThreeDReferDataset):
    def __init__(
        self,
        text_processor,
        pts_root,
        ann_paths,
        question_type=None,
        filter_missing_gt_in_pth=False,
        pth_rel_subdir="scannetpp",
        eval_scene_ids=None,
        eval_scene_allowlist_file=None,
        eval_max_samples=None,
        train_max_samples=None,
        instance_id_cache_file=None,
        instance_id_cache_write=False,
        write_filtered_annotations_to=None,
        object_name_keys: Union[Sequence[str], str, None] = None,
        max_object_name_chars: int = 96,
        chain_answer_fallback_plain: bool = True,
    ):
        if object_name_keys is None:
            keys: Tuple[str, ...] = ("object_name",)
        elif isinstance(object_name_keys, str):
            keys = (object_name_keys,)
        else:
            keys = tuple(str(k) for k in object_name_keys)
        self._object_name_keys = keys
        self._max_object_name_chars = int(max_object_name_chars)
        self._chain_answer_fallback_plain = bool(chain_answer_fallback_plain)
        self.chain_answer_list = list(CHAIN_ANSWER_LIST)
        super().__init__(
            text_processor=text_processor,
            pts_root=pts_root,
            ann_paths=ann_paths,
            question_type=question_type,
            filter_missing_gt_in_pth=filter_missing_gt_in_pth,
            pth_rel_subdir=pth_rel_subdir,
            eval_scene_ids=eval_scene_ids,
            eval_scene_allowlist_file=eval_scene_allowlist_file,
            eval_max_samples=eval_max_samples,
            train_max_samples=train_max_samples,
            instance_id_cache_file=instance_id_cache_file,
            instance_id_cache_write=instance_id_cache_write,
            write_filtered_annotations_to=write_filtered_annotations_to,
        )

    def _object_name_from_ann(self, ann: Dict[str, Any]) -> str:
        for k in self._object_name_keys:
            if k not in ann:
                continue
            v = ann.get(k)
            s = _format_object_name_phrase(v)
            if not s:
                continue
            if self._max_object_name_chars > 0:
                s = s[: self._max_object_name_chars]
            return s
        return ""

    def _build_chain_answers(self, ann: Dict[str, Any]) -> List[str]:
        name = self._object_name_from_ann(ann)
        if not name:
            if self._chain_answer_fallback_plain:
                return [random.choice(self.answer_list)]
            return ["[SEG]."]
        tpl = random.choice(self.chain_answer_list)
        return [tpl.replace("{name}", name)]

    def __getitem__(self, index: int):
        out = super().__getitem__(index)
        out["answers"] = self._build_chain_answers(self.annotation[index])
        return out
