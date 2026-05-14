"""Per-benchmark noun-phrase extraction.

For grounding (LVIS, ODinW13) the TSV already carries a clean `category_name`
column. For counting (CountBenchQA, CountQA, FSC147, PixmoCount) the noun is
embedded in a "How many X are/is ..." question and must be regexed out.
CountQA additionally carries an `objects` JSON list whose first item is the
target — we prefer that when present.
"""

import json
import re
from dataclasses import dataclass


# All counting benchmarks share the "How many X are/is ..." template.
# Capture greedy-but-stop-at-the-first-verb to handle CountQA's complex
# subordinate clauses ("How many tiles are on the wall with the shower?").
_HOW_MANY_RE = re.compile(r"How many\s+(.+?)\s+(?:are|is)\b", re.IGNORECASE)

# Mirrors the template at mini_vlm_toolkit/dataset/scripts/prepare_odinw.py:289.
_LOCATE_RE = re.compile(r"Locate all instances of '([^']+)' in this image", re.IGNORECASE)


@dataclass
class Extraction:
    sam3_prompt: str
    method: str
    note: str = ""

    @property
    def is_warn(self) -> bool:
        return not self.sam3_prompt or self.method.startswith("warn_")


def _normalize_category_name(name: str) -> str:
    # LVIS uses underscores ("water_bottle", "soda_can"); SAM3 expects
    # natural-language phrases. Underscore -> space; collapse whitespace.
    cleaned = name.replace("_", " ")
    return " ".join(cleaned.split())


def _from_category_name(row) -> Extraction:
    name = str(row.get("category_name") or "").strip()
    if name:
        return Extraction(sam3_prompt=_normalize_category_name(name), method="category_name_field")
    # Fallback to regex on the question.
    q = str(row.get("question") or "")
    m = _LOCATE_RE.search(q)
    if m:
        return Extraction(sam3_prompt=_normalize_category_name(m.group(1).strip()),
                          method="regex_locate_question")
    return Extraction(sam3_prompt="", method="warn_no_category", note=q[:120])


def _from_how_many(row) -> Extraction:
    q = str(row.get("question") or "")
    m = _HOW_MANY_RE.search(q)
    if m:
        return Extraction(sam3_prompt=m.group(1).strip(), method="regex_how_many")
    return Extraction(sam3_prompt="", method="warn_regex_miss", note=q[:120])


def _from_count_qa(row) -> Extraction:
    # Try the question regex first — it's usually clean ("How many tiles are
    # on the wall with the shower?" → "tiles"). Fall back to objects[0].
    primary = _from_how_many(row)
    if not primary.is_warn:
        return primary
    raw = row.get("objects")
    if raw:
        try:
            objects = json.loads(raw) if isinstance(raw, str) else list(raw)
            if objects:
                return Extraction(
                    sam3_prompt=str(objects[0]).strip(),
                    method="objects_field_first",
                )
        except Exception:
            pass
    return primary


_DISPATCH = {
    # Grounding — clean category field.
    "LVIS_bbox_train": _from_category_name,
    "LVIS_bbox_val": _from_category_name,
    "LVIS_bbox_train_positive_only": _from_category_name,
    "LVIS_bbox_val_positive_only": _from_category_name,
    "LVIS_bbox_train_federated": _from_category_name,
    "LVIS_bbox_val_federated": _from_category_name,
    "ODinW13_bbox_train": _from_category_name,
    "ODinW13_bbox_val": _from_category_name,
    "ODinW13_bbox_val_full": _from_category_name,
    "ODinW13_bbox_val_full_all": _from_category_name,
    "ODinW13_bbox_val_full_n10": _from_category_name,
    "ODinW13_bbox_val_full_n50": _from_category_name,
    "ODinW13_bbox_val_full_n100": _from_category_name,
    "ODinW13_bbox_train_full_n50": _from_category_name,
    "ODinW13_bbox_train_full_n100": _from_category_name,
    # Counting.
    "CountBenchQA": _from_how_many,
    "FSC147_train": _from_how_many,
    "FSC147_val": _from_how_many,
    "FSC147_test": _from_how_many,
    "PixmoCount_train": _from_how_many,
    "PixmoCount_val": _from_how_many,
    "PixmoCount_test": _from_how_many,
    "CountQA": _from_count_qa,
}


def extract(dataset_name: str, row) -> Extraction:
    fn = _DISPATCH.get(dataset_name)
    if fn is None:
        # Fall back: try category_name then how-many.
        try_cat = _from_category_name(row)
        if not try_cat.is_warn:
            return try_cat
        try_count = _from_how_many(row)
        if not try_count.is_warn:
            return try_count
        return Extraction(sam3_prompt="", method=f"warn_unknown_dataset_{dataset_name}")
    return fn(row)


def is_grounding(dataset_name: str) -> bool:
    return dataset_name.startswith(("LVIS_bbox", "LVIS_mask", "ODinW13_bbox"))


def is_counting(dataset_name: str) -> bool:
    return dataset_name in {
        "CountBenchQA", "CountQA",
        "FSC147_train", "FSC147_val", "FSC147_test",
        "PixmoCount_train", "PixmoCount_val", "PixmoCount_test",
        "FSCD_LVIS_count_train", "FSCD_LVIS_count_val", "FSCD_LVIS_count_test",
        "FSCD_LVIS_unseen_count_train", "FSCD_LVIS_unseen_count_test",
    }
