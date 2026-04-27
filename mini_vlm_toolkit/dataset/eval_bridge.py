"""Shared bridge from TSV-style predictions to COCO-format JSON for AP evaluation.

Both LVIS (via ``lvis.LVISEval``) and ODinW-13 (via ``pycocotools.COCOeval``)
consume the same flat JSON format:

    [{"image_id": int, "category_id": int,
      "bbox": [x, y, w, h], "score": float}]

This module:
  * parses model text predictions on each TSV row (0-1000 normalized xyxy),
  * converts them to absolute pixel xywh,
  * drops invalid / zero-area boxes,
  * assigns a score (Phase 1: constant=1.0; Phase 2: sidecar lookup).

It also provides ``filter_coco_gt_to_pairs`` for category-conditioned eval:
restricting a COCO/LVIS GT dict to the (image_id, category_id) pairs the
model was actually asked about.
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .lvis import _normalise_boxes, _parse_bbox_array_from_text

logger = logging.getLogger(__name__)


def parse_image_id_from_index(index: str) -> Optional[int]:
    """Parse the numeric image_id from a TSV row index.

    LVIS  : ``val_<image_id>_<category_id>``
    ODinW : ``<domain>_<split>_<image_id>_<category_id>``  (domain may
             contain underscores, so we split from the right.)

    Returns None on parse failure.
    """
    parts = str(index).rsplit("_", 2)
    if len(parts) < 3:
        return None
    try:
        return int(parts[-2])
    except ValueError:
        return None


def boxes_norm1k_to_abs_xywh(
    boxes_xyxy_norm1k: Sequence[Sequence[float]],
    image_w: float,
    image_h: float,
) -> List[List[float]]:
    """Convert 0-1000 xyxy boxes to absolute pixel xywh.

    Clips to image bounds, fixes inverted boxes (x2<x1 or y2<y1), and
    drops zero-area boxes.
    """
    out: List[List[float]] = []
    for box in boxes_xyxy_norm1k:
        if box is None or len(box) < 4:
            continue
        x1 = box[0] / 1000.0 * image_w
        y1 = box[1] / 1000.0 * image_h
        x2 = box[2] / 1000.0 * image_w
        y2 = box[3] / 1000.0 * image_h

        x1 = max(0.0, min(x1, image_w))
        y1 = max(0.0, min(y1, image_h))
        x2 = max(0.0, min(x2, image_w))
        y2 = max(0.0, min(y2, image_h))

        x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
        y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)
        w = x_max - x_min
        h = y_max - y_min
        if w > 0 and h > 0:
            out.append([float(x_min), float(y_min), float(w), float(h)])
    return out


def build_coco_predictions(
    rows: Iterable,
    score_mode: str = "constant",
    score_lookup: Optional[Callable] = None,
) -> List[Dict]:
    """Build a flat COCO/LVIS-format predictions list from TSV-style rows.

    Parameters
    ----------
    rows : iterable of dict-like
        Each row must support ``.get()`` / ``__getitem__`` for keys:
        ``index``, ``image_width``, ``image_height``, ``category_id``,
        ``prediction``.  pandas Series rows (from ``DataFrame.iterrows``)
        work directly.
    score_mode : {"constant", "original"}
        ``"constant"`` — every prediction gets ``score=1.0``.
        ``"original"`` — use ``score_lookup`` to recover a per-box score;
        fall back to ``1.0`` if it returns None.
    score_lookup : callable(row, parsed_box_xyxy_norm1k) -> float | None
        Phase 2 hook (sidecar SAM3 scores). Ignored in Phase 1.
    """
    if score_mode not in ("constant", "original"):
        raise ValueError(f"Unknown score_mode: {score_mode!r}")

    preds: List[Dict] = []

    for row in rows:
        index = str(row["index"])
        image_id = parse_image_id_from_index(index)
        if image_id is None:
            logger.debug("Could not parse image_id from index %r, skipping.", index)
            continue

        try:
            category_id = int(row["category_id"])
        except (KeyError, TypeError, ValueError):
            logger.debug("Missing/bad category_id for index %r, skipping.", index)
            continue

        try:
            image_w = float(row.get("image_width") or 0)
            image_h = float(row.get("image_height") or 0)
        except (TypeError, ValueError):
            image_w = image_h = 0.0
        if image_w <= 0 or image_h <= 0:
            logger.debug("Invalid image dims for index %r, skipping.", index)
            continue

        pred_text = str(row.get("prediction", "") or "")
        boxes = _parse_bbox_array_from_text(pred_text)
        if not boxes:
            continue

        # Canonicalise to 0-1000 xyxy regardless of the model's emitted scale.
        boxes_norm = _normalise_boxes(boxes, image_w, image_h)

        for norm_box in boxes_norm:
            xywh_list = boxes_norm1k_to_abs_xywh([norm_box], image_w, image_h)
            if not xywh_list:
                continue
            xywh = xywh_list[0]

            if score_mode == "constant":
                score = 1.0
            else:  # "original"
                score = None
                if score_lookup is not None:
                    try:
                        score = score_lookup(row, norm_box)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(
                            "score_lookup raised %r for index %r", exc, index
                        )
                        score = None
                if score is None:
                    score = 1.0

            preds.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": xywh,
                    "score": float(score),
                }
            )

    return preds


def filter_coco_gt_to_pairs(
    gt_dict: Dict,
    pairs: Iterable[Tuple[int, int]],
    keep_all_categories: bool = True,
    remap_ann_ids: bool = False,
) -> Dict:
    """Filter a COCO/LVIS-format GT dict to (image_id, category_id) pairs.

    Used by Phase 1's category-conditioned eval: keep only annotations on
    images and categories the model was actually asked about, so that
    LVISEval/COCOeval don't penalise non-evaluated pairs.

    Parameters
    ----------
    gt_dict
        Dict with keys ``images``, ``annotations``, ``categories``.  Other
        top-level keys (``licenses``, ``info``, etc.) are preserved.
    pairs
        Iterable of ``(image_id, category_id)`` tuples.
    keep_all_categories
        If True, retain the full categories list — needed by LVIS for
        APr/APc/APf frequency bucketing.  If False, restrict categories
        to those appearing in ``pairs``.
    remap_ann_ids
        If True, reassign each kept annotation a fresh sequential ``id``
        starting at 1.  COCOeval treats GT id ``0`` as the "no-match"
        sentinel — datasets that start ann ids at 0 (notably ODinW) cause
        the first annotation per (image, cat) to be miscounted as FP if
        ids are passed through unchanged.  LVIS ids already start at 1, so
        this is a no-op there but harmless to enable.
    """
    pair_set = {(int(i), int(c)) for i, c in pairs}
    image_id_set = {i for i, _ in pair_set}

    new = dict(gt_dict)
    new["images"] = [
        img for img in gt_dict.get("images", [])
        if int(img["id"]) in image_id_set
    ]
    kept_anns = [
        ann for ann in gt_dict.get("annotations", [])
        if (int(ann["image_id"]), int(ann["category_id"])) in pair_set
    ]
    if remap_ann_ids:
        # Make a shallow copy of each kept annotation with a 1-based id.
        new["annotations"] = [
            {**ann, "id": i + 1} for i, ann in enumerate(kept_anns)
        ]
    else:
        new["annotations"] = kept_anns

    if keep_all_categories:
        new["categories"] = list(gt_dict.get("categories", []))
    else:
        cat_ids = {c for _, c in pair_set}
        new["categories"] = [
            c for c in gt_dict.get("categories", [])
            if int(c["id"]) in cat_ids
        ]
    return new


def write_json(obj, path) -> None:
    """Write ``obj`` to ``path`` as JSON.  Convenience wrapper."""
    with open(path, "w") as f:
        json.dump(obj, f)
