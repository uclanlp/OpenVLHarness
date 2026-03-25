"""LVIS v1 dataset classes for bounding-box and segmentation-mask evaluation.

Two dataset classes are provided:

  LVISBBoxDataset  — evaluates predicted bounding boxes against LVIS ground truth.
  LVISMaskDataset  — evaluates predicted segmentation masks against LVIS ground truth.
                     Accepts both polygon-coordinate and RLE-encoded predictions.

Both classes expect pre-generated TSV files produced by:
    python -m mini_vlm_toolkit.dataset.scripts.prepare_lvis

TSV naming convention:
    LVIS_bbox_train.tsv / LVIS_bbox_val.tsv   (bounding-box evaluation)
    LVIS_mask_train.tsv / LVIS_mask_val.tsv   (segmentation-mask evaluation)

Each row represents a unique (image, category) pair.  The 'answer' column
holds the ground-truth annotations in JSON format.

Usage:
    python run_benchmark.py --data LVIS_bbox_val ...
    python run_benchmark.py --data LVIS_mask_val ...
"""

import json
import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .image_base import ImageBaseDataset
from ..smp import *  # noqa: F401,F403
from ..smp.file import LMUDataRoot, get_intermediate_file_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _parse_bbox_array_from_text(text: str) -> List[List[float]]:
    """Extract bounding boxes from model output text.

    Supports multiple formats:
      1. JSON array:   [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
      2. Single box:   [0.1, 0.2, 0.3, 0.4]
      3. Loose numbers: fall back to grouping every 4 numbers as a box

    Returns a list of [x1, y1, x2, y2] lists (normalised 0-1).
    """
    if not isinstance(text, str) or not text.strip():
        return []

    # --- Try JSON parse first ---
    try:
        # Strip markdown code blocks if present
        clean = text.strip()
        md_match = re.search(r'```(?:json)?\s*(.*?)```', clean, re.DOTALL)
        if md_match:
            clean = md_match.group(1).strip()

        parsed = json.loads(clean)

        # Case: [[x1,y1,x2,y2], ...]
        if isinstance(parsed, list):
            boxes = []
            for item in parsed:
                if isinstance(item, list) and len(item) >= 4:
                    boxes.append([float(v) for v in item[:4]])
                elif isinstance(item, dict):
                    # Handle {"bbox": [x1,y1,x2,y2]} format
                    bbox = item.get('bbox', item.get('bounding_box', None))
                    if isinstance(bbox, list) and len(bbox) >= 4:
                        boxes.append([float(v) for v in bbox[:4]])
            if boxes:
                return boxes

            # Maybe it's a single flat list [x1,y1,x2,y2]
            if len(parsed) >= 4 and all(isinstance(v, (int, float)) for v in parsed[:4]):
                return [[float(v) for v in parsed[:4]]]
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # --- Regex fallback: find all bracket-enclosed 4-tuples ---
    number_pat = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
    sep = r'(?:\s*,\s*|\s+)'
    bracket_pat = re.compile(
        rf'\[\s*({number_pat}){sep}({number_pat}){sep}({number_pat}){sep}({number_pat})\s*\]'
    )
    boxes = []
    for m in bracket_pat.finditer(text):
        try:
            boxes.append([float(m.group(i)) for i in range(1, 5)])
        except ValueError:
            continue
    if boxes:
        return boxes

    # --- Last resort: extract all numbers and group by 4 ---
    nums = re.findall(number_pat, text)
    if len(nums) >= 4:
        boxes = []
        for i in range(0, len(nums) - 3, 4):
            boxes.append([float(nums[i]), float(nums[i+1]),
                          float(nums[i+2]), float(nums[i+3])])
        return boxes

    return []


def _normalise_boxes(boxes: List[List[float]], img_w: float, img_h: float) -> List[List[float]]:
    """Ensure boxes are in normalised [0,1] coordinate space.

    Heuristic: if max coordinate > 1.5, assume absolute pixel coords and
    normalise; if max > 1.5 but <= 1000 and image dims unknown, assume
    0-1000 scale.
    """
    if not boxes:
        return boxes

    max_val = max(max(abs(v) for v in box) for box in boxes)

    if max_val <= 1.5:
        # Already normalised
        return boxes
    elif max_val <= 1000 and img_w <= 1 and img_h <= 1:
        # 0-1000 scale (e.g. Qwen models)
        return [[v / 1000.0 for v in box] for box in boxes]
    else:
        # Absolute pixel coordinates
        normalised = []
        for box in boxes:
            normalised.append([
                box[0] / img_w, box[1] / img_h,
                box[2] / img_w, box[3] / img_h,
            ])
        return normalised


def _compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    inter = (x_right - x_left) * (y_bottom - y_top)
    area1 = max(box1[2] - box1[0], 0) * max(box1[3] - box1[1], 0)
    area2 = max(box2[2] - box2[0], 0) * max(box2[3] - box2[1], 0)
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def _greedy_match(pred_boxes, gt_boxes, iou_threshold=0.5):
    """Greedy-match predicted boxes to ground-truth boxes by IoU.

    For each (pred, gt) pair, compute IoU.  Greedily assign the highest-IoU
    pair first, removing both from future consideration.

    Returns
    -------
    matches : list of (pred_idx, gt_idx, iou)
        Successfully matched pairs (IoU >= threshold).
    unmatched_preds : list of int
        Indices of predictions with no matching GT.
    unmatched_gts : list of int
        Indices of GTs with no matching prediction.
    """
    if not pred_boxes or not gt_boxes:
        return (
            [],
            list(range(len(pred_boxes))),
            list(range(len(gt_boxes))),
        )

    # Compute full IoU matrix
    iou_matrix = []
    for pi, pb in enumerate(pred_boxes):
        for gi, gb in enumerate(gt_boxes):
            iou_val = _compute_iou(pb, gb)
            if iou_val >= iou_threshold:
                iou_matrix.append((iou_val, pi, gi))

    # Sort descending by IoU
    iou_matrix.sort(key=lambda x: -x[0])

    matched_preds = set()
    matched_gts = set()
    matches = []

    for iou_val, pi, gi in iou_matrix:
        if pi in matched_preds or gi in matched_gts:
            continue
        matches.append((pi, gi, iou_val))
        matched_preds.add(pi)
        matched_gts.add(gi)

    unmatched_preds = [i for i in range(len(pred_boxes)) if i not in matched_preds]
    unmatched_gts = [i for i in range(len(gt_boxes)) if i not in matched_gts]

    return matches, unmatched_preds, unmatched_gts


# ---------------------------------------------------------------------------
# LVISBBoxDataset
# ---------------------------------------------------------------------------

class LVISBBoxDataset(ImageBaseDataset):
    """LVIS v1 bounding-box evaluation dataset.

    Each row is a unique (image, category) pair.  The model is asked to locate
    all instances of a named category and return bounding boxes.

    Evaluation uses greedy IoU matching with per-frequency-group breakdowns.

    Supported dataset names:
        LVIS_bbox_train, LVIS_bbox_val
    """

    TYPE = 'GROUNDING'
    MODALITY = 'IMAGE'

    DATASET_URL = {
        'LVIS_bbox_train': '',
        'LVIS_bbox_val': '',
    }
    DATASET_MD5 = {}

    # IoU threshold for considering a detection as correct
    IOU_THRESHOLD = 0.5

    # Frequency group display order
    FREQ_ORDER = ['f', 'c', 'r']
    FREQ_LABELS = {'f': 'Frequent', 'c': 'Common', 'r': 'Rare'}

    def build_prompt(self, line):
        """Build the inference prompt: image + question text from TSV."""
        if isinstance(line, int):
            line = self.data.iloc[line]
        tgt_path = self.dump_image(line)
        msgs = []
        msgs.extend([dict(type='image', value=p) for p in tgt_path])
        msgs.append(dict(type='text', value=line['question']))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate bounding-box predictions against LVIS ground truth.

        Metrics computed:
          - Precision@50: fraction of predictions that match a GT (IoU >= 0.5)
          - Recall@50:    fraction of GTs matched by a prediction
          - F1@50:        harmonic mean of precision and recall
          - Mean IoU:     average IoU of matched pairs
          - Count Acc:    fraction of rows where predicted count == GT count
          Broken down by: overall, and per frequency group (frequent/common/rare).

        Parameters
        ----------
        eval_file : str
            Path to the prediction TSV (must have 'prediction' column).
        judge_kwargs : dict
            Optional; currently unused but kept for API compatibility.

        Returns
        -------
        summary_df : pd.DataFrame
            Summary metrics table.
        """
        data = load(eval_file)
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        if 'prediction' not in data:
            raise KeyError('Prediction file must contain a `prediction` column.')

        # Build metadata lookup from the dataset's own data
        meta = self.data.copy()
        meta['index'] = meta['index'].astype(str)
        meta = meta.set_index('index')

        # --- Per-row evaluation ---
        detail_rows = []
        # Accumulators for aggregation
        group_stats = defaultdict(lambda: {
            'tp': 0, 'fp': 0, 'fn': 0,
            'iou_sum': 0.0, 'iou_count': 0,
            'count_correct': 0, 'total': 0,
        })

        for _, row in data.iterrows():
            idx = str(row['index'])
            pred_text = str(row.get('prediction', ''))

            if idx not in meta.index:
                logger.warning(f'Prediction index {idx} not found in LVIS metadata, skipping.')
                continue

            meta_row = meta.loc[idx]
            img_w = float(meta_row.get('image_width', 1))
            img_h = float(meta_row.get('image_height', 1))
            freq = str(meta_row.get('frequency', ''))

            # Parse ground truth bboxes (normalised [x1,y1,x2,y2])
            try:
                gt_boxes = json.loads(str(meta_row['answer']))
            except (json.JSONDecodeError, KeyError):
                gt_boxes = []

            # Parse predicted bboxes from model output
            pred_boxes = _parse_bbox_array_from_text(pred_text)
            pred_boxes = _normalise_boxes(pred_boxes, img_w, img_h)

            # Greedy match
            matches, unmatched_p, unmatched_g = _greedy_match(
                pred_boxes, gt_boxes, self.IOU_THRESHOLD
            )

            tp = len(matches)
            fp = len(unmatched_p)
            fn = len(unmatched_g)
            match_ious = [m[2] for m in matches]
            mean_iou = float(np.mean(match_ious)) if match_ious else 0.0
            count_correct = int(len(pred_boxes) == len(gt_boxes))

            detail_rows.append({
                'index': idx,
                'frequency': freq,
                'gt_count': len(gt_boxes),
                'pred_count': len(pred_boxes),
                'tp': tp, 'fp': fp, 'fn': fn,
                'mean_iou': round(mean_iou, 4),
                'count_correct': count_correct,
                'prediction': pred_text[:500],  # truncate for readability
            })

            # Accumulate for 'overall' and per-frequency groups
            for group_key in ['overall', freq]:
                if not group_key:
                    continue
                s = group_stats[group_key]
                s['tp'] += tp
                s['fp'] += fp
                s['fn'] += fn
                s['iou_sum'] += sum(match_ious)
                s['iou_count'] += len(match_ious)
                s['count_correct'] += count_correct
                s['total'] += 1

        # --- Build summary ---
        summary_rows = []

        def _build_summary_row(label, stats):
            tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
            prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            m_iou = stats['iou_sum'] / stats['iou_count'] if stats['iou_count'] > 0 else 0.0
            count_acc = stats['count_correct'] / stats['total'] * 100 if stats['total'] > 0 else 0.0
            return {
                'Split': label,
                'Precision@50': round(prec, 2),
                'Recall@50': round(rec, 2),
                'F1@50': round(f1, 2),
                'Mean IoU': round(m_iou, 4),
                'Count Acc (%)': round(count_acc, 2),
                'Samples': stats['total'],
            }

        # Per-frequency rows (in order: frequent, common, rare)
        for fkey in self.FREQ_ORDER:
            if fkey in group_stats:
                label = self.FREQ_LABELS.get(fkey, fkey)
                summary_rows.append(_build_summary_row(label, group_stats[fkey]))

        # Overall row
        if 'overall' in group_stats:
            summary_rows.append(_build_summary_row('Overall', group_stats['overall']))

        summary_df = pd.DataFrame(summary_rows)

        # Save outputs
        detail_df = pd.DataFrame(detail_rows)
        detail_file = get_intermediate_file_path(eval_file, '_detail')
        dump(detail_df, detail_file)

        score_file = get_intermediate_file_path(eval_file, '_acc')
        dump(summary_df, score_file)

        logger.info(f'LVIS bbox evaluation complete. {len(detail_rows)} rows evaluated.')
        return summary_df


# ---------------------------------------------------------------------------
# LVISMaskDataset
# ---------------------------------------------------------------------------

class LVISMaskDataset(ImageBaseDataset):
    """LVIS v1 segmentation-mask evaluation dataset.

    Each row is a unique (image, category) pair.  The model is asked to segment
    all instances of a named category.

    Predictions can be in two formats:
      1. Polygon coordinates: [[[x1,y1,x2,y2,...], ...], ...]
         (list of instances, each instance is a list of polygons)
      2. RLE-encoded masks:   [{"counts": "...", "size": [h, w]}, ...]

    Ground truth is stored as polygon coordinates (matching the LVIS annotation
    format).  Both GT and predictions are converted to binary masks via
    pycocotools for IoU computation.

    Requires: pycocotools (pip install pycocotools)

    Supported dataset names:
        LVIS_mask_train, LVIS_mask_val
    """

    TYPE = 'GROUNDING'
    MODALITY = 'IMAGE'

    DATASET_URL = {
        'LVIS_mask_train': '',
        'LVIS_mask_val': '',
    }
    DATASET_MD5 = {}

    IOU_THRESHOLD = 0.5
    FREQ_ORDER = ['f', 'c', 'r']
    FREQ_LABELS = {'f': 'Frequent', 'c': 'Common', 'r': 'Rare'}

    def build_prompt(self, line):
        """Build the inference prompt: image + question text from TSV."""
        if isinstance(line, int):
            line = self.data.iloc[line]
        tgt_path = self.dump_image(line)
        msgs = []
        msgs.extend([dict(type='image', value=p) for p in tgt_path])
        msgs.append(dict(type='text', value=line['question']))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate segmentation-mask predictions against LVIS ground truth.

        Computes both mask IoU and bbox IoU metrics.  Predictions may be
        polygon coordinates or RLE-encoded masks.

        Metrics (same structure as bbox, plus mask-specific):
          - Mask Precision@50, Recall@50, F1@50, Mean Mask IoU
          - Bbox Precision@50, Recall@50, F1@50, Mean Bbox IoU  (from answer_bbox)
          - Per frequency group breakdown

        Requires pycocotools for mask operations.

        Parameters
        ----------
        eval_file : str
            Path to the prediction TSV (must have 'prediction' column).
        judge_kwargs : dict
            Optional; currently unused.

        Returns
        -------
        summary_df : pd.DataFrame
        """
        # Import pycocotools — hard requirement for mask evaluation
        try:
            import pycocotools.mask as mask_util
        except ImportError:
            raise ImportError(
                'pycocotools is required for LVIS mask evaluation. '
                'Install it with: pip install pycocotools'
            )

        data = load(eval_file)
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        if 'prediction' not in data:
            raise KeyError('Prediction file must contain a `prediction` column.')

        meta = self.data.copy()
        meta['index'] = meta['index'].astype(str)
        meta = meta.set_index('index')

        detail_rows = []
        group_stats = defaultdict(lambda: {
            'mask_tp': 0, 'mask_fp': 0, 'mask_fn': 0,
            'mask_iou_sum': 0.0, 'mask_iou_count': 0,
            'bbox_tp': 0, 'bbox_fp': 0, 'bbox_fn': 0,
            'bbox_iou_sum': 0.0, 'bbox_iou_count': 0,
            'total': 0,
        })

        for _, row in data.iterrows():
            idx = str(row['index'])
            pred_text = str(row.get('prediction', ''))

            if idx not in meta.index:
                logger.warning(f'Prediction index {idx} not found in LVIS metadata, skipping.')
                continue

            meta_row = meta.loc[idx]
            img_w = int(float(meta_row.get('image_width', 1)))
            img_h = int(float(meta_row.get('image_height', 1)))
            freq = str(meta_row.get('frequency', ''))

            # --- Parse GT masks (polygon format) ---
            try:
                gt_segmentations = json.loads(str(meta_row['answer']))
            except (json.JSONDecodeError, KeyError):
                gt_segmentations = []

            gt_masks = []
            for seg in gt_segmentations:
                mask = self._segmentation_to_mask(seg, img_h, img_w, mask_util)
                if mask is not None:
                    gt_masks.append(mask)

            # --- Parse GT bboxes (for bbox metric) ---
            gt_boxes = []
            try:
                gt_boxes = json.loads(str(meta_row.get('answer_bbox', '[]')))
            except (json.JSONDecodeError, KeyError):
                pass

            # --- Parse predicted masks ---
            pred_masks, pred_boxes_from_mask = self._parse_mask_predictions(
                pred_text, img_h, img_w, mask_util
            )

            # --- Mask evaluation: greedy match by mask IoU ---
            mask_matches, mask_unmatched_p, mask_unmatched_g = self._greedy_match_masks(
                pred_masks, gt_masks, self.IOU_THRESHOLD, mask_util
            )
            mask_tp = len(mask_matches)
            mask_fp = len(mask_unmatched_p)
            mask_fn = len(mask_unmatched_g)
            mask_match_ious = [m[2] for m in mask_matches]
            mask_mean_iou = float(np.mean(mask_match_ious)) if mask_match_ious else 0.0

            # --- Bbox evaluation (using answer_bbox column) ---
            pred_boxes = _parse_bbox_array_from_text(pred_text)
            pred_boxes = _normalise_boxes(pred_boxes, float(img_w), float(img_h))
            # If no bboxes parsed from text, try deriving from pred_masks
            if not pred_boxes and pred_boxes_from_mask:
                pred_boxes = pred_boxes_from_mask

            bbox_matches, bbox_unmatched_p, bbox_unmatched_g = _greedy_match(
                pred_boxes, gt_boxes, self.IOU_THRESHOLD
            )
            bbox_tp = len(bbox_matches)
            bbox_fp = len(bbox_unmatched_p)
            bbox_fn = len(bbox_unmatched_g)
            bbox_match_ious = [m[2] for m in bbox_matches]
            bbox_mean_iou = float(np.mean(bbox_match_ious)) if bbox_match_ious else 0.0

            detail_rows.append({
                'index': idx,
                'frequency': freq,
                'gt_mask_count': len(gt_masks),
                'pred_mask_count': len(pred_masks),
                'mask_tp': mask_tp, 'mask_fp': mask_fp, 'mask_fn': mask_fn,
                'mask_mean_iou': round(mask_mean_iou, 4),
                'gt_bbox_count': len(gt_boxes),
                'pred_bbox_count': len(pred_boxes),
                'bbox_tp': bbox_tp, 'bbox_fp': bbox_fp, 'bbox_fn': bbox_fn,
                'bbox_mean_iou': round(bbox_mean_iou, 4),
                'prediction': pred_text[:500],
            })

            for group_key in ['overall', freq]:
                if not group_key:
                    continue
                s = group_stats[group_key]
                s['mask_tp'] += mask_tp
                s['mask_fp'] += mask_fp
                s['mask_fn'] += mask_fn
                s['mask_iou_sum'] += sum(mask_match_ious)
                s['mask_iou_count'] += len(mask_match_ious)
                s['bbox_tp'] += bbox_tp
                s['bbox_fp'] += bbox_fp
                s['bbox_fn'] += bbox_fn
                s['bbox_iou_sum'] += sum(bbox_match_ious)
                s['bbox_iou_count'] += len(bbox_match_ious)
                s['total'] += 1

        # --- Build summary ---
        summary_rows = []

        def _build_summary_row(label, stats):
            # Mask metrics
            mtp, mfp, mfn = stats['mask_tp'], stats['mask_fp'], stats['mask_fn']
            m_prec = mtp / (mtp + mfp) * 100 if (mtp + mfp) > 0 else 0.0
            m_rec = mtp / (mtp + mfn) * 100 if (mtp + mfn) > 0 else 0.0
            m_f1 = 2 * m_prec * m_rec / (m_prec + m_rec) if (m_prec + m_rec) > 0 else 0.0
            m_iou = (stats['mask_iou_sum'] / stats['mask_iou_count']
                     if stats['mask_iou_count'] > 0 else 0.0)
            # Bbox metrics
            btp, bfp, bfn = stats['bbox_tp'], stats['bbox_fp'], stats['bbox_fn']
            b_prec = btp / (btp + bfp) * 100 if (btp + bfp) > 0 else 0.0
            b_rec = btp / (btp + bfn) * 100 if (btp + bfn) > 0 else 0.0
            b_f1 = 2 * b_prec * b_rec / (b_prec + b_rec) if (b_prec + b_rec) > 0 else 0.0
            b_iou = (stats['bbox_iou_sum'] / stats['bbox_iou_count']
                     if stats['bbox_iou_count'] > 0 else 0.0)
            return {
                'Split': label,
                'Mask Prec@50': round(m_prec, 2),
                'Mask Rec@50': round(m_rec, 2),
                'Mask F1@50': round(m_f1, 2),
                'Mean Mask IoU': round(m_iou, 4),
                'BBox Prec@50': round(b_prec, 2),
                'BBox Rec@50': round(b_rec, 2),
                'BBox F1@50': round(b_f1, 2),
                'Mean BBox IoU': round(b_iou, 4),
                'Samples': stats['total'],
            }

        for fkey in self.FREQ_ORDER:
            if fkey in group_stats:
                summary_rows.append(
                    _build_summary_row(self.FREQ_LABELS.get(fkey, fkey), group_stats[fkey])
                )

        if 'overall' in group_stats:
            summary_rows.append(_build_summary_row('Overall', group_stats['overall']))

        summary_df = pd.DataFrame(summary_rows)

        detail_df = pd.DataFrame(detail_rows)
        detail_file = get_intermediate_file_path(eval_file, '_detail')
        dump(detail_df, detail_file)

        score_file = get_intermediate_file_path(eval_file, '_acc')
        dump(summary_df, score_file)

        logger.info(f'LVIS mask evaluation complete. {len(detail_rows)} rows evaluated.')
        return summary_df

    # --- Mask helper methods ---

    @staticmethod
    def _segmentation_to_mask(segmentation, img_h, img_w, mask_util):
        """Convert a single instance's segmentation to a binary mask.

        Parameters
        ----------
        segmentation : list or dict
            LVIS polygon format: [[x1,y1,x2,y2,...], ...]  (list of polygons)
            OR RLE dict: {"counts": ..., "size": [h, w]}
        img_h, img_w : int
            Image dimensions.
        mask_util : module
            pycocotools.mask module.

        Returns
        -------
        rle : dict
            RLE-encoded mask, or None if conversion fails.
        """
        try:
            if isinstance(segmentation, dict):
                # Already RLE
                if 'counts' in segmentation and 'size' in segmentation:
                    # Ensure size matches
                    if isinstance(segmentation['counts'], list):
                        return mask_util.frPyObjects(segmentation, img_h, img_w)
                    return segmentation
            elif isinstance(segmentation, list):
                if len(segmentation) == 0:
                    return None
                # Check if it's a list of polygons (list of lists of numbers)
                # or a flat list of numbers
                if isinstance(segmentation[0], list):
                    # List of polygons — standard LVIS format
                    rles = mask_util.frPyObjects(segmentation, img_h, img_w)
                    return mask_util.merge(rles)
                elif isinstance(segmentation[0], (int, float)):
                    # Flat list — treat as single polygon
                    rles = mask_util.frPyObjects([segmentation], img_h, img_w)
                    return mask_util.merge(rles)
        except Exception as e:
            logger.debug(f'Failed to convert segmentation to mask: {e}')
        return None

    @staticmethod
    def _compute_mask_iou(rle1, rle2, mask_util) -> float:
        """Compute IoU between two RLE-encoded masks."""
        try:
            iou = mask_util.iou([rle1], [rle2], [0])
            return float(iou[0][0])
        except Exception:
            return 0.0

    @classmethod
    def _greedy_match_masks(cls, pred_masks, gt_masks, iou_threshold, mask_util):
        """Greedy-match predicted masks to GT masks by mask IoU.

        Same algorithm as _greedy_match for boxes but using mask IoU.

        Returns (matches, unmatched_preds, unmatched_gts).
        """
        if not pred_masks or not gt_masks:
            return (
                [],
                list(range(len(pred_masks))),
                list(range(len(gt_masks))),
            )

        # Compute IoU matrix
        iou_pairs = []
        for pi, pm in enumerate(pred_masks):
            for gi, gm in enumerate(gt_masks):
                iou_val = cls._compute_mask_iou(pm, gm, mask_util)
                if iou_val >= iou_threshold:
                    iou_pairs.append((iou_val, pi, gi))

        iou_pairs.sort(key=lambda x: -x[0])

        matched_preds = set()
        matched_gts = set()
        matches = []

        for iou_val, pi, gi in iou_pairs:
            if pi in matched_preds or gi in matched_gts:
                continue
            matches.append((pi, gi, iou_val))
            matched_preds.add(pi)
            matched_gts.add(gi)

        unmatched_preds = [i for i in range(len(pred_masks)) if i not in matched_preds]
        unmatched_gts = [i for i in range(len(gt_masks)) if i not in matched_gts]

        return matches, unmatched_preds, unmatched_gts

    @classmethod
    def _parse_mask_predictions(cls, text, img_h, img_w, mask_util):
        """Parse mask predictions from model output text.

        Handles two formats:
          1. Polygon coordinates: [[[x1,y1,...], ...], ...]
          2. RLE-encoded masks:   [{"counts": "...", "size": [h, w]}, ...]

        Also extracts bounding boxes from masks (for bbox metric fallback).

        Returns
        -------
        masks : list of RLE dicts
            Parsed and converted masks.
        bboxes_normalised : list of [x1, y1, x2, y2]
            Bounding boxes derived from masks, normalised to [0, 1].
        """
        if not isinstance(text, str) or not text.strip():
            return [], []

        masks = []
        bboxes = []

        # Strip markdown code blocks
        clean = text.strip()
        md_match = re.search(r'```(?:json)?\s*(.*?)```', clean, re.DOTALL)
        if md_match:
            clean = md_match.group(1).strip()

        # Try JSON parse
        try:
            parsed = json.loads(clean)
        except (json.JSONDecodeError, ValueError):
            parsed = None

        if parsed is not None and isinstance(parsed, list):
            for item in parsed:
                rle = None
                if isinstance(item, dict) and 'counts' in item and 'size' in item:
                    # RLE mask prediction
                    rle = cls._segmentation_to_mask(item, img_h, img_w, mask_util)
                elif isinstance(item, list):
                    # Could be:
                    #   - A single polygon: [x1,y1,x2,y2,...]
                    #   - A list of polygons: [[x1,y1,...], [x1,y1,...]]
                    #   - A list of instances, each a list of polygons
                    if len(item) > 0 and isinstance(item[0], (int, float)):
                        # Single polygon
                        rle = cls._segmentation_to_mask([item], img_h, img_w, mask_util)
                    elif len(item) > 0 and isinstance(item[0], list):
                        # List of polygons for one instance
                        rle = cls._segmentation_to_mask(item, img_h, img_w, mask_util)
                elif isinstance(item, dict) and 'segmentation' in item:
                    # {"segmentation": [...], ...} format
                    rle = cls._segmentation_to_mask(
                        item['segmentation'], img_h, img_w, mask_util
                    )

                if rle is not None:
                    masks.append(rle)
                    # Derive bbox from mask
                    try:
                        bbox = mask_util.toBbox(rle).tolist()  # [x, y, w, h]
                        x1 = bbox[0] / img_w
                        y1 = bbox[1] / img_h
                        x2 = (bbox[0] + bbox[2]) / img_w
                        y2 = (bbox[1] + bbox[3]) / img_h
                        bboxes.append([x1, y1, x2, y2])
                    except Exception:
                        pass

        return masks, bboxes
