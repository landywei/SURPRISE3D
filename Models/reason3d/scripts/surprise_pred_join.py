"""
Shared helpers to join Surprise predictions.jsonl rows to surprise_val.json.

Uses the same (scene_id, object_id, blip_question(description)) triple as the dataloader.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

QUESTION_TEMPLATES_LOWER = (
    "please segment the object according to the given 3d scene and the description: ",
    "given the 3d scene, segment this object according to the description: ",
    "respond the segmentation mask of the object: ",
)


def pre_question(question: str, max_words: int = 50) -> str:
    """Mirror BlipQuestionProcessor.pre_question (default max_words=50)."""
    question = re.sub(r'([!"()*#:;~])', "", (question or "").lower())
    question = re.sub(r"(?<!\d)\.(?!\d)", "", question)
    question = question.rstrip(" ")
    words = question.split(" ")
    if len(words) > max_words:
        question = " ".join(words[: max_words])
    return question


def desc_from_text_input(text: str) -> Optional[str]:
    t = (text or "").strip().lower()
    for pref in QUESTION_TEMPLATES_LOWER:
        if t.startswith(pref):
            rest = (text or "").strip()[len(pref) :].strip()
            if rest.endswith("."):
                rest = rest[:-1].strip()
            return rest
    return None


def oids_key(row: Dict[str, Any]) -> Tuple[int, ...]:
    o = row.get("object_id")
    if isinstance(o, list):
        return tuple(sorted(int(x) for x in o))
    return (int(o),)


def row_eval_key(row: Dict[str, Any]) -> Optional[Tuple[str, Tuple[int, ...], str]]:
    sid = str(row.get("scene_id", ""))
    oids = oids_key(row)
    desc = desc_from_text_input(row.get("text_input") or "")
    if desc is None:
        return None
    return (sid, oids, pre_question(desc))


def load_ann_question_types(ann_path: str) -> Dict[Tuple[str, Tuple[int, ...], str], str]:
    with open(ann_path, "r", encoding="utf-8") as f:
        anns = json.load(f)
    out: Dict[Tuple[str, Tuple[int, ...], str], str] = {}
    for a in anns:
        sid = str(a["scene_id"])
        raw_oid = a["object_id"]
        if isinstance(raw_oid, list):
            oids = tuple(sorted(int(x) for x in raw_oid))
        else:
            oids = (int(raw_oid),)
        d = pre_question(a.get("description", ""))
        qt = str(a.get("question_type", "") or "unknown")
        out[(sid, oids, d)] = qt
    return out


def load_rows(jsonl_path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def first_index_by_eval_key(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, Tuple[int, ...], str], int]:
    """First row index for each eval key (deduplicates repeated keys in JSONL)."""
    m: Dict[Tuple[str, Tuple[int, ...], str], int] = {}
    for i, r in enumerate(rows):
        k = row_eval_key(r)
        if k is None:
            continue
        if k not in m:
            m[k] = i
    return m


def eval_key_to_jsonable(k: Tuple[str, Tuple[int, ...], str]) -> List[Any]:
    sid, oids, d = k
    return [sid, list(oids), d]


def eval_key_from_jsonable(obj: Any) -> Tuple[str, Tuple[int, ...], str]:
    sid, oids, d = obj
    return (str(sid), tuple(int(x) for x in oids), str(d))
