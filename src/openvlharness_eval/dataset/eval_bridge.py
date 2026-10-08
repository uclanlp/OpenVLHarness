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
import os
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .lvis import _normalise_boxes, _parse_bbox_array_from_text

logger = logging.getLogger(__name__)


def _parse_qwen3_multilabel_predictions(
    text: str,
    label_to_cat_id: Dict[str, int],
) -> List[Tuple[List[float], int]]:
    """Parse Qwen3-VL multi-label output into [(bbox_xyxy, category_id), ...].

    Qwen3-VL emits a JSON list like
    ``[{"bbox_2d": [x1,y1,x2,y2], "label": "..."}, ...]``. Labels are
    mapped case-insensitively to category IDs via ``label_to_cat_id``;
    entries whose label is not in the dataset vocabulary are dropped
    (this is the same "drop unknown labels" semantics as the shipped
    ``Qwen3-VL/evaluation/ODinW-13/run_odinw.py eval``).

    Boxes are returned in the **raw** coordinate scale the model emitted —
    the caller is responsible for normalisation (so the ``coords_space``
    override in ``build_coco_predictions`` still applies).
    """
    import ast
    import re

    if not isinstance(text, str) or not text.strip():
        return []

    clean = text.strip()
    md_match = re.search(r'```(?:json)?\s*(.*?)```', clean, re.DOTALL)
    if md_match:
        clean = md_match.group(1).strip()

    parsed = None
    try:
        parsed = json.loads(clean)
    except (json.JSONDecodeError, ValueError, TypeError):
        # Fallback: Qwen sometimes mixes single/double quotes — ast handles it.
        try:
            parsed = ast.literal_eval(clean)
        except (ValueError, SyntaxError):
            return []

    # Qwen3-VL does not always emit the flat ``[{"bbox_2d", "label"}, ...]``
    # list. Observed variants (Qwen3-VL-32B on ODinW-13, 2026-09-21):
    #   {"boat": [{"bbox_2d": [..]}, ...], "car": [...]}      label is the key
    #   {"boat": [[x1,y1,x2,y2], ...]}                          bare boxes under key
    #   [{"category": "boat", "bbox_2d": [[..], [..]]}]         one record per class
    # All carry the same information, so they are flattened to (box, label)
    # pairs here. The canonical list form parses exactly as before; an
    # explicit per-box label wins over the enclosing key when both resolve.
    box_keys = ("bbox_2d", "bbox", "bounding_box")
    label_keys = ("label", "category", "category_name", "name", "class")

    def _as_box(v):
        if not isinstance(v, (list, tuple)) or len(v) < 4:
            return None
        try:
            return [float(x) for x in v[:4]]
        except (TypeError, ValueError):
            return None

    # Single-category prompts ("Locate every instance of 'pistol'"): a box that carries no label
    # at all can only refer to that category. Qwen3-VL-8B answers such rows with bare
    # ``[{"bbox_2d": [...]}, ...]`` on ODinW-13 (pistols_export, pothole, ...), which the strict
    # "drop unknown labels" rule would discard wholesale. Only applies when the vocabulary has
    # exactly one entry and the box has no label of its own; multi-class rows are unchanged.
    sole_cat_id = int(next(iter(label_to_cat_id.values()))) if len(label_to_cat_id) == 1 else None

    def _resolve(explicit, inherited):
        for label in (explicit, inherited):
            if isinstance(label, str):
                cat_id = label_to_cat_id.get(label.lower())
                if cat_id is not None:
                    return int(cat_id)
        if explicit is None and sole_cat_id is not None and (inherited is None or inherited in box_keys):
            return sole_cat_id
        return None

    results: List[Tuple[List[float], int]] = []

    def _walk(node, inherited):
        if isinstance(node, dict):
            bbox = next((node[k] for k in box_keys if k in node), None)
            if bbox is not None:
                explicit = next(
                    (node[k] for k in label_keys if isinstance(node.get(k), str)), None
                )
                cat_id = _resolve(explicit, inherited)
                if cat_id is None:
                    return
                box = _as_box(bbox)
                if box is not None:
                    results.append((box, cat_id))
                elif isinstance(bbox, list):
                    for b in bbox:
                        box = _as_box(b)
                        if box is not None:
                            results.append((box, cat_id))
                return
            for key, value in node.items():
                _walk(value, key if isinstance(key, str) else inherited)
            return
        if isinstance(node, (list, tuple)):
            if not any(isinstance(x, (list, dict)) for x in node):
                box = _as_box(node)
                if box is not None and (inherited is not None or sole_cat_id is not None):
                    cat_id = _resolve(None, inherited)
                    if cat_id is not None:
                        results.append((box, cat_id))
                return
            for elem in node:
                _walk(elem, inherited)

    _walk(parsed, None)
    return results


def parse_image_id_from_index(index: str) -> Optional[int]:
    """Parse the numeric image_id from a TSV row index.

    Three layouts are supported:
      LVIS / ODinW per-class : ``val_<image_id>_<category_id>``
                               or ``<domain>_<split>_<image_id>_<category_id>``
      ODinW Qwen3-VL one-row-per-image : ``<domain>_<split>_<image_id>``
                                          (no category suffix)

    We first try the per-class layout (image_id at the second-to-last
    segment); if that segment isn't numeric, we fall back to the
    one-row-per-image layout (image_id at the last segment).

    Returns None on parse failure.
    """
    parts = str(index).rsplit("_", 2)
    if len(parts) < 2:
        return None
    try:
        return int(parts[-2])
    except ValueError:
        try:
            return int(parts[-1])
        except ValueError:
            return None


def _xyxy_pixel_to_clipped_xywh(
    box: Sequence[float], image_w: float, image_h: float,
) -> Optional[List[float]]:
    """Clip a pixel xyxy box to image bounds, fix inverted coords, drop
    zero-area, and return xywh. Returns None if the box is degenerate.
    """
    if box is None or len(box) < 4:
        return None
    x1 = max(0.0, min(float(box[0]), float(image_w)))
    y1 = max(0.0, min(float(box[1]), float(image_h)))
    x2 = max(0.0, min(float(box[2]), float(image_w)))
    y2 = max(0.0, min(float(box[3]), float(image_h)))
    x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
    y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)
    w = x_max - x_min
    h = y_max - y_min
    if w <= 0 or h <= 0:
        return None
    return [x_min, y_min, w, h]


def _log_degenerate_sidecar_box(
    *,
    output_dir: Optional[str],
    index: str,
    image_w: float,
    image_h: float,
    matched_box_xyxy: Sequence[float],
    agent_box_norm1k: Sequence[float],
) -> None:
    """Emit a very loud error when a sidecar-matched box collapses to
    zero area after clipping.

    This branch is "shouldn't ever happen" territory — a sidecar
    candidate that matched the agent's box at IoU>=0.5 but then
    clipped to zero implies either (a) SAM3 returned a zero-area or
    out-of-bounds box, or (b) the recorded image dims disagree with
    the sidecar's coordinate space. We want it to be impossible to
    miss in a results folder.
    """
    msg = (
        "[DEGENERATE_SIDECAR_BOX] index=%s image_dims=(%s,%s) "
        "sidecar_box_xyxy=%s agent_box_xyxy_norm1k=%s"
        % (index, image_w, image_h, list(matched_box_xyxy), list(agent_box_norm1k))
    )
    banner = "!" * 80
    logger.error("\n%s\n%s\n%s", banner, msg, banner)
    if not output_dir:
        return
    try:
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, "DEGENERATE_SIDECAR_BOXES.log")
        with open(log_path, "a") as f:
            f.write(msg + "\n")
    except OSError as exc:  # noqa: BLE001
        logger.error(
            "Failed to write DEGENERATE_SIDECAR_BOXES.log under %r: %r",
            output_dir, exc,
        )


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
    coords_space: str = "auto",
    mode: str = "single",
    output_dir: Optional[str] = None,
) -> List[Dict]:
    """Build a flat COCO/LVIS-format predictions list from TSV-style rows.

    Parameters
    ----------
    rows : iterable of dict-like
        Each row must support ``.get()`` / ``__getitem__`` for keys:
        ``index``, ``image_width``, ``image_height``, ``prediction``.
        In ``mode='single'`` also: ``category_id`` (one per row).
        In ``mode='qwen3_multilabel'`` also: ``dataset_classes`` (a JSON
        list of ``{"id":N, "name":"X"}`` records covering the prompted
        vocabulary, used to resolve the model's per-box ``label`` field).
        pandas Series rows work directly.
    score_mode : {"constant", "original"}
        ``"constant"`` — every prediction gets ``score=1.0``.
        ``"original"`` — use ``score_lookup`` to recover both the per-box
        score and the matched sidecar box (full SAM3 pixel precision);
        unmatched boxes fall back to score ``0.5`` and the agent's box.
    score_lookup : callable(row, parsed_box_xyxy_norm1k)
                   -> (score: float, box_xyxy_pixels: list[float]) | None
        Phase 2 hook (sidecar SAM3 scores). Ignored in Phase 1.
    coords_space : {"auto", "pixel", "norm1k"}
        Forwarded to ``_normalise_boxes``. Default ``"auto"`` uses the
        magnitude heuristic; pass ``"pixel"`` for Qwen-VL ``bbox_2d``
        output on small images where the heuristic would misclassify.
    mode : {"single", "qwen3_multilabel"}
        ``"single"`` (default): one (image, category) per row; model
        emits a list of boxes all sharing the row's ``category_id``.
        ``"qwen3_multilabel"``: one image per row; model emits
        ``[{"bbox_2d": [...], "label": "..."}, ...]`` covering multiple
        classes. Labels are resolved to category_ids via the row's
        ``dataset_classes`` column (case-insensitive); unknown labels
        are dropped (matches shipped Qwen3-VL eval).
    output_dir : str, optional
        Directory in which to drop a ``DEGENERATE_SIDECAR_BOXES.log``
        sentinel file if a sidecar-matched box collapses to zero area
        after clipping (a "shouldn't ever happen" condition we want to
        notice loudly). If ``None``, only a ``logger.error`` is emitted.
    """
    if score_mode not in ("constant", "original"):
        raise ValueError(f"Unknown score_mode: {score_mode!r}")
    if mode not in ("single", "qwen3_multilabel"):
        raise ValueError(f"Unknown mode: {mode!r}")

    preds: List[Dict] = []

    for row in rows:
        index = str(row["index"])
        image_id = parse_image_id_from_index(index)
        if image_id is None:
            logger.debug("Could not parse image_id from index %r, skipping.", index)
            continue

        try:
            image_w = float(row.get("image_width") or 0)
            image_h = float(row.get("image_height") or 0)
        except (TypeError, ValueError):
            image_w = image_h = 0.0
        if image_w <= 0 or image_h <= 0:
            logger.debug("Invalid image dims for index %r, skipping.", index)
            continue

        pv = row.get("prediction", "")
        # Empty CSV cells round-trip through pandas as NaN (a float).
        # Treat NaN / None / empty string as "no prediction".
        if pv is None or (isinstance(pv, float) and pv != pv):
            continue
        pred_text = str(pv)
        if not pred_text.strip():
            continue

        # Parse model output into (raw_box, category_id) tuples. In
        # single-class mode every box shares the row's category_id; in
        # multi-label mode labels are resolved via dataset_classes.
        if mode == "qwen3_multilabel":
            try:
                dataset_classes = json.loads(row.get("dataset_classes") or "[]")
            except (json.JSONDecodeError, TypeError):
                dataset_classes = []
            label_to_cat_id = {
                str(c["name"]).lower(): int(c["id"])
                for c in dataset_classes
                if isinstance(c, dict) and "name" in c and "id" in c
            }
            raw_pairs = _parse_qwen3_multilabel_predictions(pred_text, label_to_cat_id)
            if not raw_pairs:
                continue
            raw_boxes = [bp[0] for bp in raw_pairs]
            per_box_cat_id = [bp[1] for bp in raw_pairs]
        else:
            try:
                category_id = int(row["category_id"])
            except (KeyError, TypeError, ValueError):
                logger.debug("Missing/bad category_id for index %r, skipping.", index)
                continue
            raw_boxes = _parse_bbox_array_from_text(pred_text)
            if not raw_boxes:
                continue
            per_box_cat_id = [category_id] * len(raw_boxes)

        # Canonicalise to 0-1000 xyxy regardless of the model's emitted scale.
        boxes_norm = _normalise_boxes(raw_boxes, image_w, image_h, coords_space=coords_space)

        for norm_box, cat_id in zip(boxes_norm, per_box_cat_id):
            # Decide score, and whether the COCO bbox should come from the
            # sidecar match (full pixel precision) or the agent's echoed box.
            xywh: Optional[List[float]] = None
            if score_mode == "constant":
                score = 1.0
            else:  # "original"
                match = None
                if score_lookup is not None:
                    try:
                        match = score_lookup(row, norm_box)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(
                            "score_lookup raised %r for index %r", exc, index
                        )
                        match = None
                if match is None:
                    score = 0.5
                else:
                    score, matched_box_xyxy = match
                    xywh = _xyxy_pixel_to_clipped_xywh(
                        matched_box_xyxy, image_w, image_h,
                    )
                    if xywh is None:
                        _log_degenerate_sidecar_box(
                            output_dir=output_dir,
                            index=index,
                            image_w=image_w,
                            image_h=image_h,
                            matched_box_xyxy=matched_box_xyxy,
                            agent_box_norm1k=norm_box,
                        )
                    # If the sidecar box is degenerate after clipping, fall
                    # through to the agent's box below.

            if xywh is None:
                xywh_list = boxes_norm1k_to_abs_xywh([norm_box], image_w, image_h)
                if not xywh_list:
                    continue
                xywh = xywh_list[0]

            preds.append(
                {
                    "image_id": image_id,
                    "category_id": cat_id,
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


def filter_coco_gt_to_image_ids(
    gt_dict: Dict,
    image_ids: Iterable[int],
    remap_ann_ids: bool = False,
) -> Dict:
    """Filter a COCO/LVIS-format GT dict to a set of ``image_id``s.

    Unlike :func:`filter_coco_gt_to_pairs`, this keeps **all images** in
    the requested set, even if they have **zero annotations**.  That's
    essential for ODinW full-detection eval: the model is asked about
    every (image, category) pair in the chosen image set, including
    images with no GT annotations ("negative" images), and a hallucinated
    prediction on such an image must be counted as a false positive by
    COCOeval — which means its image_id has to exist in the GT.

    Parameters
    ----------
    gt_dict
        Source COCO/LVIS GT dict.
    image_ids
        Iterable of image_ids to keep.  Images present in ``gt_dict``
        whose ``id`` is in this set survive (with all their annotations);
        everything else is dropped.
    remap_ann_ids
        If True, renumber kept annotations 1..N (avoids the COCOeval
        ``id=0`` "no-match sentinel" bug — see ``filter_coco_gt_to_pairs``
        for context).
    """
    iid_set = {int(i) for i in image_ids}
    new = dict(gt_dict)
    new["images"] = [
        img for img in gt_dict.get("images", [])
        if int(img["id"]) in iid_set
    ]
    kept_anns = [
        ann for ann in gt_dict.get("annotations", [])
        if int(ann["image_id"]) in iid_set
    ]
    if remap_ann_ids:
        new["annotations"] = [
            {**ann, "id": i + 1} for i, ann in enumerate(kept_anns)
        ]
    else:
        new["annotations"] = kept_anns
    new["categories"] = list(gt_dict.get("categories", []))
    return new


def drop_incomplete_images(
    predictions: List[Dict],
    gt_dict: Dict,
    expected_pairs: Iterable[Tuple[int, int]],
    completed_pairs: Iterable[Tuple[int, int]],
) -> Tuple[List[Dict], Dict, Dict]:
    """Phase 1.5 partial-run safety net for full-detection evaluation.

    Drops every ``(image_id)`` group whose **full** set of expected category
    queries wasn't completed.  Removes those images from BOTH the predictions
    list and the GT dict (images + annotations) so COCOeval's recall isn't
    penalised by un-queried (image, cat) pairs.

    Parameters
    ----------
    predictions
        COCO-format predictions for one domain (each dict has ``image_id``,
        ``category_id``, ``bbox``, ``score``).
    gt_dict
        The (already category-filtered / id-remapped) per-domain GT dict.
        Modified in a fresh shallow copy; the caller's dict is untouched.
    expected_pairs
        ``(image_id, category_id)`` tuples that the source TSV asked the
        model about (i.e. that should have a prediction row).
    completed_pairs
        ``(image_id, category_id)`` tuples for which inference actually
        produced a row (the model emitted something — empty string answers
        count as missing, ``"[]"`` counts as completed).

    Returns
    -------
    filtered_preds : list[dict]
    filtered_gt : dict
    summary : dict with keys
        images_total, images_completed, images_dropped, dropped_image_ids
    """
    expected_by_image: Dict[int, set] = {}
    for iid, cid in expected_pairs:
        expected_by_image.setdefault(int(iid), set()).add(int(cid))

    completed_set = {(int(i), int(c)) for i, c in completed_pairs}

    completed_image_ids = set()
    dropped_image_ids = []
    for image_id, expected_cats in expected_by_image.items():
        if all((image_id, c) in completed_set for c in expected_cats):
            completed_image_ids.add(image_id)
        else:
            dropped_image_ids.append(image_id)

    new_gt = dict(gt_dict)
    new_gt["images"] = [
        img for img in gt_dict.get("images", [])
        if int(img["id"]) in completed_image_ids
    ]
    kept_image_ids = {int(img["id"]) for img in new_gt["images"]}
    new_gt["annotations"] = [
        ann for ann in gt_dict.get("annotations", [])
        if int(ann["image_id"]) in kept_image_ids
    ]
    if "categories" in gt_dict:
        new_gt["categories"] = list(gt_dict["categories"])
    # Defensive: filter predictions by the actual surviving GT image_ids
    # (``kept_image_ids``), not by the abstract ``completed_image_ids``.
    # If something upstream pre-filtered ``gt_dict`` and dropped images,
    # ``completed_image_ids`` may include image_ids no longer in GT —
    # passing predictions for those to COCOeval triggers an assertion
    # ``Results do not correspond to current coco set``.  Using
    # ``kept_image_ids`` guarantees ``pred_image_ids ⊆ gt_image_ids``.
    filtered_preds = [
        p for p in predictions if int(p["image_id"]) in kept_image_ids
    ]

    summary = {
        "images_total": len(expected_by_image),
        "images_completed": len(completed_image_ids),
        "images_dropped": len(dropped_image_ids),
        "dropped_image_ids": sorted(dropped_image_ids),
    }
    return filtered_preds, new_gt, summary


def write_json(obj, path) -> None:
    """Write ``obj`` to ``path`` as JSON.  Convenience wrapper."""
    with open(path, "w") as f:
        json.dump(obj, f)


def resolve_trajectory_path(eval_file: str, basename: str) -> str:
    """Resolve a per-row trajectory ``basename`` (e.g.
    ``val_294_253.json``) to its absolute path on disk.

    Preference order, taking the first that actually exists:

    1. ``<eval_dir>/raw_traces/<basename>``  — written when the
       inference run used ``--save-raw-trace`` (token-based training traces).
    2. ``<eval_dir>/full_trajectories/<basename>``  — always written
       during inference; full agent messages / reasoning / predictions.
    3. Empty string when neither exists.  This happens for LVIS
       auto-extended rows where the evaluator considered an (image,
       cat) slot the TSV never asked about, so no trajectory was
       produced.

    ``eval_file`` is the predictions xlsx path (the same arg
    ``dataset.evaluate(...)`` receives); its parent directory is the
    dataset run dir that contains ``raw_traces/`` and
    ``full_trajectories/``.
    """
    eval_dir = os.path.dirname(os.path.abspath(eval_file))
    raw = os.path.join(eval_dir, "raw_traces", basename)
    if os.path.isfile(raw):
        return raw
    full = os.path.join(eval_dir, "full_trajectories", basename)
    if os.path.isfile(full):
        return full
    return ""


def extract_per_row_diagnostics(
    evaluator,
    *,
    iou_threshold: float = 0.5,
    area_label: str = "all",
) -> List[Dict]:
    """Extract per-(image, category) eval diagnostics from a populated
    ``pycocotools.cocoeval.COCOeval`` or ``lvis.LVISEval`` instance.

    Must be called **after** ``evaluator.evaluate()`` has populated
    ``evalImgs`` / ``eval_imgs``. Both wrappers do this as part of
    ``run()`` (LVIS) or as a step inside their normal eval flow (COCO).

    Returns a list of dicts (one per non-trivial (image_id, category_id)
    pair the evaluator considered, at area=``area_label``):

    .. code-block:: python

        {
            "image_id":     int,
            "category_id":  int,
            "num_pred":     int,   # detections (excluding ignored)
            "num_gt":       int,   # GT (excluding ignored)
            "tp":           int,   # at the chosen IoU threshold
            "fp":           int,
            "fn":           int,
            "mean_match_iou": float | None,  # mean IoU of matched pairs
        }

    Pairs the evaluator returned ``None`` for (no GT and no detections)
    are skipped.
    """
    import numpy as np

    p = evaluator.params
    # Detect evaluator flavour by attribute naming convention.
    is_lvis = hasattr(p, "iou_thrs")
    if is_lvis:
        eval_imgs = evaluator.eval_imgs
        iou_thrs = p.iou_thrs
        area_rng = p.area_rng
        area_rng_lbl = p.area_rng_lbl
        K_DT_MATCHES = "dt_matches"
        K_GT_MATCHES = "gt_matches"
        K_DT_IDS = "dt_ids"
        K_GT_IDS = "gt_ids"
        K_DT_IGNORE = "dt_ignore"
        K_GT_IGNORE = "gt_ignore"
        K_ARNG = "area_rng"
    else:
        eval_imgs = evaluator.evalImgs
        iou_thrs = p.iouThrs
        area_rng = p.areaRng
        area_rng_lbl = p.areaRngLbl
        K_DT_MATCHES = "dtMatches"
        K_GT_MATCHES = "gtMatches"
        K_DT_IDS = "dtIds"
        K_GT_IDS = "gtIds"
        K_DT_IGNORE = "dtIgnore"
        K_GT_IGNORE = "gtIgnore"
        K_ARNG = "aRng"

    # Closest IoU threshold index.
    iou_arr = np.asarray(iou_thrs, dtype=float)
    iou_idx = int(np.argmin(np.abs(iou_arr - float(iou_threshold))))

    # Area-label index.
    try:
        area_idx = list(area_rng_lbl).index(area_label)
    except ValueError:
        area_idx = 0
    target_arng = list(area_rng[area_idx])

    # IoU matrix lookup: evaluator.ious is keyed by (image_id, category_id)
    # with shape (D_sorted, G).  We use it to recover per-match IoU.
    ious_map = getattr(evaluator, "ious", {}) or {}

    rows: List[Dict] = []
    for entry in eval_imgs:
        if entry is None:
            continue
        # Filter to area=all (or whatever the caller asked for).
        if list(entry.get(K_ARNG, [])) != target_arng:
            continue

        dt_ids = list(entry.get(K_DT_IDS, []))
        gt_ids = list(entry.get(K_GT_IDS, []))
        dt_matches = np.asarray(entry.get(K_DT_MATCHES, []))
        gt_matches = np.asarray(entry.get(K_GT_MATCHES, []))
        dt_ignore = np.asarray(entry.get(K_DT_IGNORE, []))
        gt_ignore = np.asarray(entry.get(K_GT_IGNORE, []))

        # GT count (non-ignored)
        if gt_ignore.size:
            num_gt = int(np.count_nonzero(~gt_ignore.astype(bool)))
        else:
            num_gt = 0

        # Pull the chosen IoU threshold's slice (or empty if dt is empty).
        def _slice(mat):
            if mat.ndim == 2 and mat.shape[0] > iou_idx:
                return mat[iou_idx]
            return np.zeros(0)

        dt_m_thr = _slice(dt_matches)
        dt_ig_thr = _slice(dt_ignore)
        gt_m_thr = _slice(gt_matches)

        # DT count (non-ignored at this IoU)
        if dt_ig_thr.size:
            num_pred = int(np.count_nonzero(~dt_ig_thr.astype(bool)))
        else:
            num_pred = len(dt_ids)

        # TP / FP / FN
        if dt_m_thr.size:
            matched = (dt_m_thr != 0) & (~dt_ig_thr.astype(bool))
            tp = int(np.count_nonzero(matched))
            fp = int(num_pred - tp)
        else:
            tp = 0
            fp = num_pred
        if gt_m_thr.size:
            gt_matched = (gt_m_thr != 0) & (~gt_ignore.astype(bool))
            fn = int(num_gt - int(np.count_nonzero(gt_matched)))
        else:
            fn = num_gt

        # Mean IoU of matched pairs (best-effort; NaN if recovery fails).
        # Emitted on a 0-100 percentage scale, rounded to 2 dp — matches the
        # AP / Prec / Rec conventions used elsewhere in the per-row CSVs.
        mean_match_iou: Optional[float] = None
        if tp > 0:
            try:
                key = (int(entry["image_id"]), int(entry["category_id"]))
                ious_mat = ious_map.get(key)
                if ious_mat is not None and hasattr(ious_mat, "shape"):
                    ious_mat = np.asarray(ious_mat)
                    if ious_mat.ndim == 2 and ious_mat.size > 0:
                        gt_id_to_idx = {int(g): i for i, g in enumerate(gt_ids)}
                        matched_ious: List[float] = []
                        for d_idx in range(len(dt_m_thr)):
                            if dt_m_thr[d_idx] == 0:
                                continue
                            if d_idx < len(dt_ig_thr) and dt_ig_thr[d_idx]:
                                continue
                            g_idx = gt_id_to_idx.get(int(dt_m_thr[d_idx]))
                            if g_idx is None:
                                continue
                            if d_idx < ious_mat.shape[0] and g_idx < ious_mat.shape[1]:
                                matched_ious.append(float(ious_mat[d_idx, g_idx]))
                        if matched_ious:
                            mean_match_iou = round(float(np.mean(matched_ious)) * 100.0, 2)
            except Exception:
                mean_match_iou = None

        rows.append({
            "image_id":       int(entry["image_id"]),
            "category_id":    int(entry["category_id"]),
            "num_pred":       num_pred,
            "num_gt":         num_gt,
            "tp":             tp,
            "fp":             fp,
            "fn":             fn,
            "mean_match_iou": mean_match_iou,
        })

    return rows


def emit_counting_per_row(
    eval_file: str,
    indices: Iterable,
    image_paths: Iterable,
    *,
    extra_cols: Optional[Dict[str, Iterable]] = None,
    sort_score: Optional[Iterable[float]] = None,
    file_suffix: str = "_per_row",
) -> str:
    """Build and dump a per-row CSV for counting-style evaluators.

    Counterpart to :func:`extract_per_row_diagnostics` for grounding tasks,
    but much simpler: there is no IoU matcher to walk — each TSV row already
    has a single (gt, pred) pair the caller has computed.  This helper just
    resolves the per-row trajectory path, lays out the columns in a
    reviewer-friendly order, and writes the CSV next to ``eval_file``.

    Parameters
    ----------
    eval_file
        The predictions xlsx path (same arg ``dataset.evaluate(...)`` gets).
        Used only to resolve the trajectory dir and the output path.
    indices
        Per-row TSV index values (e.g. ``data['index']``).  Trajectory
        basenames are ``f'{idx}.json'``.
    image_paths
        Per-row image paths from the TSV.
    extra_cols
        Ordered dict of additional columns (e.g. ``gt``, ``pred``,
        ``exact_match``, ``ra5``, ``is_focused``).  Insertion order is
        preserved in the output CSV.
    sort_score
        Optional per-row score (higher = better).  When provided, rows are
        sorted by descending score before writing; the score column itself
        is **not** included in the output (silent sort).  Original input
        order is preserved within ties via mergesort.
    file_suffix
        Suffix passed to :func:`get_intermediate_file_path` (default
        ``_per_row``, matching the grounding evaluators).

    Returns
    -------
    str
        The absolute path of the CSV written.
    """
    import pandas as pd
    from ..smp.file import get_intermediate_file_path, dump

    indices = list(indices)
    image_paths = list(image_paths)
    n = len(indices)
    assert len(image_paths) == n, (
        f"image_paths length {len(image_paths)} != indices length {n}"
    )

    rows = []
    for i in range(n):
        idx = indices[i]
        rows.append({
            "trajectory_file": resolve_trajectory_path(eval_file, f"{idx}.json"),
            "index": idx,
            "image_path": image_paths[i],
        })

    if extra_cols:
        for col_name, values in extra_cols.items():
            values = list(values)
            assert len(values) == n, (
                f"extra_cols[{col_name!r}] length {len(values)} != {n}"
            )
            for i, v in enumerate(values):
                rows[i][col_name] = v

    df = pd.DataFrame(rows)

    if sort_score is not None:
        scores = list(sort_score)
        assert len(scores) == n, (
            f"sort_score length {len(scores)} != {n}"
        )
        # Stable mergesort preserves original (TSV) order within tied scores.
        df = (
            df.assign(_sort_score=scores)
              .sort_values("_sort_score", ascending=False, kind="mergesort")
              .drop(columns="_sort_score")
              .reset_index(drop=True)
        )

    out_path = get_intermediate_file_path(eval_file, file_suffix, target_format="csv")
    dump(df, out_path)
    logger.info("[per-row] wrote %d-row CSV to %s", n, out_path)
    return out_path


def _sort_grounding_per_row(rows: List[Dict]) -> List[Dict]:
    """Sort grounding per-row records (output of
    :func:`extract_per_row_diagnostics`) by F1 desc, with mean_match_iou desc
    as tiebreaker.

    F1 = 2*tp / (2*tp + fp + fn), with safe-zero denom.  The F1 column is
    NOT added to the records — sort is silent.  Returns a new list (does not
    mutate input).
    """
    def f1_score(r):
        tp = float(r.get("tp") or 0)
        fp = float(r.get("fp") or 0)
        fn = float(r.get("fn") or 0)
        denom = 2 * tp + fp + fn
        return (2 * tp / denom) if denom > 0 else 0.0

    def iou_key(r):
        v = r.get("mean_match_iou")
        try:
            return float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    # Negative because Python sort is ascending; we want both keys descending.
    return sorted(rows, key=lambda r: (-f1_score(r), -iou_key(r)))


def _resolve_score_modes(
    tool_detections_dir: str, score_mode: Optional[str] = None,
) -> Tuple[List[str], str]:
    """Decide which score_mode(s) to run for one ``evaluate()`` call.

    Resolution order:

    1. Explicit ``score_mode`` from the judge config → run that single mode.
       Must be ``constant`` or ``original``; invalid values raise ValueError.
    2. ``$SCORE_MODE`` env var, if set to ``constant`` or ``original`` →
       run that single mode.
    3. Sidecars present (``tool_detections/`` dir with at least one entry)
       → run BOTH ``original`` (primary) and ``constant`` (alternate) so
       the score-utility ablation is produced in one shot.
    4. Else → run ``constant`` only.

    Returns ``(modes, primary)`` where ``modes`` is the ordered list of
    modes to evaluate (primary first) and ``primary`` is the mode whose
    output files are the un-suffixed ``_metrics.json`` / ``_acc.csv``;
    other modes are written as ``_metrics_<mode>.json`` / ``_acc_<mode>.csv``.
    """
    if score_mode is not None:
        if not isinstance(score_mode, str) or score_mode.strip().lower() not in ("constant", "original"):
            raise ValueError("score_mode must be 'constant' or 'original'.")
        mode = score_mode.strip().lower()
        return [mode], mode
    override = os.environ.get("SCORE_MODE", "").strip().lower()
    if override in ("constant", "original"):
        return [override], override
    has_sidecars = (
        os.path.isdir(tool_detections_dir)
        and any(os.scandir(tool_detections_dir))
    )
    if has_sidecars:
        return ["original", "constant"], "original"
    return ["constant"], "constant"


def _iou_xyxy(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """IoU of two boxes in [x1, y1, x2, y2] format. 0 for degenerate boxes."""
    ax1, ay1, ax2, ay2 = box_a[:4]
    bx1, by1, bx2, by2 = box_b[:4]
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    if union <= 0:
        return 0.0
    return inter / union


def build_sam3_score_lookup(
    tool_detections_dir: str,
    iou_threshold: float = 0.5,
):
    """Phase 2 score + box recovery for ``score_mode='original'``.

    Returns a callable
    ``score_lookup(row, parsed_box_xyxy_norm1k)
        -> (score: float, box_xyxy_pixels: list[float]) | None``
    suitable for ``build_coco_predictions``.  Uses the per-row sidecar
    JSONs written by the inference loop at
    ``tool_detections_dir/<row_index>.json`` (Phase 2 spec).

    For each row:
      1. Lazily load the sidecar (cached per row).
      2. Pool every (box_xyxy_pixel, score) candidate across all tool
         calls in the row.
      3. Convert the agent's parsed box (still in 0-1000 xyxy) to
         absolute pixel xyxy using the row's image dimensions.
      4. Match against the candidate pool by IoU.  If max IoU >= the
         threshold, return that candidate's ``(score, box_xyxy_pixels)``;
         else return None (the caller falls back to the agent's box
         and a constant score).

    The returned callable also exposes a ``counters`` dict with
    ``hits`` / ``misses`` / ``rows_with_sidecar`` / ``rows_without_sidecar``
    so the evaluator can report a hit rate in the metrics blob.
    """
    sidecar_cache: Dict[str, List[Dict]] = {}
    counters = {
        "hits": 0,
        "misses": 0,
        "rows_with_sidecar": 0,
        "rows_without_sidecar": 0,
        "rows_seen": set(),
    }

    def _load_sidecar(task_id: str) -> List[Dict]:
        if task_id in sidecar_cache:
            return sidecar_cache[task_id]
        path = os.path.join(tool_detections_dir, f"{task_id}.json")
        if not os.path.isfile(path):
            sidecar_cache[task_id] = []
            return []
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            data = []
        if not isinstance(data, list):
            data = []
        sidecar_cache[task_id] = data
        return data

    def score_lookup(row, parsed_box_xyxy_norm1k):
        task_id = str(row["index"])
        if task_id not in counters["rows_seen"]:
            counters["rows_seen"].add(task_id)
            sc = _load_sidecar(task_id)
            if sc:
                counters["rows_with_sidecar"] += 1
            else:
                counters["rows_without_sidecar"] += 1

        sidecar = _load_sidecar(task_id)
        if not sidecar:
            counters["misses"] += 1
            return None

        try:
            image_w = float(row.get("image_width") or 0)
            image_h = float(row.get("image_height") or 0)
        except (TypeError, ValueError):
            image_w = image_h = 0.0
        if image_w <= 0 or image_h <= 0:
            counters["misses"] += 1
            return None

        # Convert agent box from 0-1000 xyxy to absolute pixel xyxy.
        x1 = parsed_box_xyxy_norm1k[0] / 1000.0 * image_w
        y1 = parsed_box_xyxy_norm1k[1] / 1000.0 * image_h
        x2 = parsed_box_xyxy_norm1k[2] / 1000.0 * image_w
        y2 = parsed_box_xyxy_norm1k[3] / 1000.0 * image_h
        agent_box = [
            min(x1, x2), min(y1, y2),
            max(x1, x2), max(y1, y2),
        ]

        best_score = None
        best_box: Optional[List[float]] = None
        best_iou = -1.0
        for entry in sidecar:
            boxes = entry.get("boxes_xyxy_pixels") or []
            scores = entry.get("scores") or []
            for cand_box, cand_score in zip(boxes, scores):
                if len(cand_box) < 4:
                    continue
                iou = _iou_xyxy(agent_box, cand_box)
                if iou > best_iou:
                    best_iou = iou
                    best_score = float(cand_score)
                    best_box = [float(v) for v in cand_box[:4]]
        if (
            best_iou >= iou_threshold
            and best_score is not None
            and best_box is not None
        ):
            counters["hits"] += 1
            return best_score, best_box
        counters["misses"] += 1
        return None

    score_lookup.counters = counters  # type: ignore[attr-defined]
    return score_lookup
