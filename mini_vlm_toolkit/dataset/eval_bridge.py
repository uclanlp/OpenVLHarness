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

        pv = row.get("prediction", "")
        # Empty CSV cells round-trip through pandas as NaN (a float).
        # Treat NaN / None / empty string as "no prediction".
        if pv is None or (isinstance(pv, float) and pv != pv):
            continue
        pred_text = str(pv)
        if not pred_text.strip():
            continue
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

    1. ``<eval_dir>/raw_trajectories/<basename>``  — written when the
       inference run used ``--save-raw-trajectory`` (intended for SFT).
    2. ``<eval_dir>/full_trajectories/<basename>``  — always written
       during inference; full agent messages / reasoning / predictions.
    3. Empty string when neither exists.  This happens for LVIS
       auto-extended rows where the evaluator considered an (image,
       cat) slot the TSV never asked about, so no trajectory was
       produced.

    ``eval_file`` is the predictions xlsx path (the same arg
    ``dataset.evaluate(...)`` receives); its parent directory is the
    dataset run dir that contains ``raw_trajectories/`` and
    ``full_trajectories/``.
    """
    eval_dir = os.path.dirname(os.path.abspath(eval_file))
    raw = os.path.join(eval_dir, "raw_trajectories", basename)
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
                            mean_match_iou = float(np.mean(matched_ious))
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


def _resolve_score_modes(tool_detections_dir: str) -> Tuple[List[str], str]:
    """Decide which score_mode(s) to run for one ``evaluate()`` call.

    Resolution order:

    1. ``$SCORE_MODE`` env var, if set to ``constant`` or ``original`` →
       run that single mode.
    2. Sidecars present (``tool_detections/`` dir with at least one entry)
       → run BOTH ``original`` (primary) and ``constant`` (alternate) so
       the score-utility ablation is produced in one shot.
    3. Else → run ``constant`` only.

    Returns ``(modes, primary)`` where ``modes`` is the ordered list of
    modes to evaluate (primary first) and ``primary`` is the mode whose
    output files are the un-suffixed ``_metrics.json`` / ``_acc.csv``;
    other modes are written as ``_metrics_<mode>.json`` / ``_acc_<mode>.csv``.
    """
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
    """Phase 2 score recovery for ``score_mode='original'``.

    Returns a callable ``score_lookup(row, parsed_box_xyxy_norm1k) -> float | None``
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
         threshold, return that candidate's score; else return None
         (the caller falls back to 1.0).

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
        if best_iou >= iou_threshold and best_score is not None:
            counters["hits"] += 1
            return best_score
        counters["misses"] += 1
        return None

    score_lookup.counters = counters  # type: ignore[attr-defined]
    return score_lookup
