"""
Chain v3 CoT dataset: extends ``ThreeDReferDatasetChainV3`` with multi-step
chain-of-thought answer templates for landmark-relational queries
(``relative_position``, ``abs``, ``first_view``) and falls back to the
single-``[SEG]`` chain-v2 template for non-relational queries
(``cs``, ``hi``, ``camera_view``).

Per-sample emission:

* ``answers`` may contain either
    - a **two-``[SEG]``** CoT (``P4``) template when a landmark is
      extractable from the question text via regex, e.g.

          I need to find the {landmark} first. [SEG]. Then the {target}. [SEG].

    - a **single-``[SEG]``** chain-v2 template otherwise.
* ``is_cot`` (bool), ``landmark`` (str or empty), ``cot_target_phrase``
  (str) are added so the model can branch per-sample on ``[SEG]`` count
  and for offline analysis.

Per-instance GT, ``superpoints``, ``batch_offsets`` paths inherited from
``ThreeDReferDatasetChainV3`` are left unchanged: ``CriterionV3``'s
best-of-set / scale-aware terms operate on the **final** mask
``M_2`` regardless of whether the chain is single- or two-``[SEG]``.
"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from lavis.datasets.datasets.threedrefer_datasets_chainv3 import (
    ThreeDReferDatasetChainV3,
)
from lavis.datasets.datasets.threedrefer_datasets_chain import (
    CHAIN_ANSWER_LIST,
    _format_object_name_phrase,
)


# Question types we attempt landmark regex on. Other types fall back to
# the chain-v2 single-[SEG] format unconditionally.
_LANDMARK_QTYPES: Tuple[str, ...] = (
    "relative_position",
    "abs",
    "first_view",
)


# Compiled regex bank, executed in order against the lower-cased question
# text. The first capture group of the first match is the landmark phrase.
# Patterns are intentionally conservative: missing a landmark just falls
# back to single-[SEG], which is safe; falsely matching gives the model a
# nonsense rationale, which is harmful.
_LANDMARK_PATTERNS: Tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        # "1.33m from the door", "two meters from the table"
        r"\b(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)\s*"
        r"(?:m|meter|meters|metre|metres)?\s*"
        r"(?:away\s+)?from\s+the\s+([a-z][\w\s\-]{0,40}?)(?=[.,?!]|\s+(?:and|that|which|when|where|while|then|in|on|at|by)\b|$)",
        # "closest / nearest to the door"
        r"\b(?:closest|nearest|next|adjacent)\s+to\s+the\s+([a-z][\w\s\-]{0,40}?)(?=[.,?!]|\s+(?:and|that|which|when|where|while|then|in|on|at|by)\b|$)",
        # "near / by / next-to / beside the door"
        r"\b(?:near|by|next\s+to|beside)\s+the\s+([a-z][\w\s\-]{0,40}?)(?=[.,?!]|\s+(?:and|that|which|when|where|while|then|in|on|at|by)\b|$)",
        # "after / when (you / I) enter (through) the door"
        r"\b(?:after|when|once)\s+(?:i|you)\s+enter(?:ing)?\s+(?:through\s+|into\s+|the\s+)?([a-z][\w\s\-]{0,40}?)(?=[.,?!]|\s+(?:and|that|which|when|where|while|then|in|on|at|by)\b|$)",
        # "upon entering the room"
        r"\bupon\s+entering\s+(?:the\s+)?([a-z][\w\s\-]{0,40}?)(?=[.,?!]|\s+(?:and|that|which|when|where|while|then|in|on|at|by)\b|$)",
    )
)


# Junk landmarks we refuse to chain on even if a regex matches. These are
# pronoun-like or perspective-like and should not be re-segmented.
_LANDMARK_BLOCKLIST: frozenset = frozenset(
    {
        "me",
        "you",
        "us",
        "him",
        "her",
        "them",
        "it",
        "this",
        "that",
        "these",
        "those",
        "perspective",
        "viewpoint",
        "viewer",
        "camera",
        "front",
        "back",
        "left",
        "right",
        "side",
        "edge",
    }
)


def _normalize_landmark(raw: str) -> str:
    """Trim leading articles, trailing punctuation, and collapse spaces."""
    s = re.sub(r"\s+", " ", raw).strip(" .,?!:;'\"").lower()
    s = re.sub(r"^(?:the|a|an)\s+", "", s)
    return s


def extract_landmark(question: str, question_type: Optional[str] = None) -> str:
    """Return the landmark phrase from ``question`` or empty string.

    Conservative: returns empty string when no clean match exists. ``cs``,
    ``hi``, and ``camera_view`` short-circuit to empty.
    """
    if question_type is not None and question_type not in _LANDMARK_QTYPES:
        return ""
    if not isinstance(question, str) or not question:
        return ""
    for pat in _LANDMARK_PATTERNS:
        m = pat.search(question)
        if m is None:
            continue
        cand = _normalize_landmark(m.group(1))
        if not cand or cand in _LANDMARK_BLOCKLIST:
            continue
        return cand
    return ""


# Five P4 variants. Each contains exactly two ``[SEG].`` segments and uses
# ``{landmark}`` and ``{target}`` substitution slots. We sample one per
# step (the LM target is the only difference; the architecture and
# inference graph are unchanged).
COT_ANSWER_LIST: Tuple[str, ...] = (
    "I need to find the {landmark} first. [SEG]. Then {target}. [SEG].",
    "First locate the {landmark}. [SEG]. Then segment {target}. [SEG].",
    "Step one: the {landmark}. [SEG]. Step two: {target}. [SEG].",
    "The relevant landmark is the {landmark}. [SEG]. Now {target}. [SEG].",
    "I will start with the {landmark}. [SEG]. Given that, {target}. [SEG].",
)


class ThreeDReferDatasetChainV3CoT(ThreeDReferDatasetChainV3):
    """Chain v3 CoT dataset.

    Adds three new keys to ``__getitem__`` output:

    * ``is_cot`` (bool): whether ``answers[0]`` contains two ``[SEG]``s.
    * ``landmark`` (str): regex-extracted landmark phrase, or empty.
    * ``cot_target_phrase`` (str): the substituted ``{target}`` phrase.

    All other keys are inherited.
    """

    def __init__(
        self,
        *args: Any,
        cot_template_prob: float = 1.0,
        cot_question_types: Optional[Sequence[str]] = None,
        cot_target_phrase_template: str = "the {name}",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._cot_template_prob = float(cot_template_prob)
        self._cot_question_types = (
            tuple(cot_question_types) if cot_question_types is not None
            else _LANDMARK_QTYPES
        )
        self._cot_target_phrase_template = str(cot_target_phrase_template)
        self._cot_answer_list: List[str] = list(COT_ANSWER_LIST)

    def _build_cot_answer(
        self,
        ann: Dict[str, Any],
    ) -> Tuple[Optional[str], str, str]:
        """Try to build a 2-[SEG] CoT answer; return (answer, landmark, target_phrase).

        If a landmark cannot be extracted (or cot_template_prob gates this
        sample out), returns ``(None, "", "")`` and the caller should fall
        back to the single-[SEG] chain template.
        """
        qtype = ann.get("question_type")
        if qtype is not None and qtype not in self._cot_question_types:
            return None, "", ""

        question = ann.get("description") or ann.get("question") or ann.get("text") or ""
        landmark = extract_landmark(question, question_type=qtype)
        if not landmark:
            return None, "", ""

        if self._cot_template_prob < 1.0 and random.random() >= self._cot_template_prob:
            return None, landmark, ""

        name = self._object_name_from_ann(ann)
        if not name:
            return None, landmark, ""
        target_phrase = self._cot_target_phrase_template.replace("{name}", name)

        tpl = random.choice(self._cot_answer_list)
        ans = tpl.replace("{landmark}", landmark).replace("{target}", target_phrase)
        return ans, landmark, target_phrase

    def _build_chain_answers(self, ann: Dict[str, Any]) -> List[str]:
        cot_ans, landmark, target_phrase = self._build_cot_answer(ann)
        if cot_ans is not None:
            self._last_cot_meta = {
                "is_cot": True,
                "landmark": landmark,
                "cot_target_phrase": target_phrase,
            }
            return [cot_ans]
        # Fallback: chain-v2 single-[SEG] template (unchanged behavior).
        ans = super()._build_chain_answers(ann)
        self._last_cot_meta = {
            "is_cot": False,
            "landmark": landmark,
            "cot_target_phrase": "",
        }
        return ans

    def __getitem__(self, index: int):
        # Reset; ``super().__getitem__`` triggers ``_build_chain_answers``.
        self._last_cot_meta = {"is_cot": False, "landmark": "", "cot_target_phrase": ""}
        out = super().__getitem__(index)
        out["is_cot"] = bool(self._last_cot_meta["is_cot"])
        out["landmark"] = str(self._last_cot_meta["landmark"])
        out["cot_target_phrase"] = str(self._last_cot_meta["cot_target_phrase"])
        return out

    def collater(self, batch):
        # Pop CoT scalars BEFORE super().collater so the parent's positional
        # ``list(data.values())`` unpack still matches the legacy fixed shape.
        is_cot_list: List[bool] = [bool(d.pop("is_cot", False)) for d in batch]
        landmark_list: List[str] = [str(d.pop("landmark", "")) for d in batch]
        target_phrase_list: List[str] = [str(d.pop("cot_target_phrase", "")) for d in batch]
        out = super().collater(batch)
        out["is_cot"] = is_cot_list
        out["landmark"] = landmark_list
        out["cot_target_phrase"] = target_phrase_list
        return out
