"""LVIS v1 dataset classes for bounding-box and segmentation-mask evaluation.

Two dataset classes are provided:

  LVISBBoxDataset  — evaluates predicted bounding boxes against LVIS ground truth
                     using the **official lvis-api LVISEval** (AP, AP50, AP75,
                     APr, APc, APf with max_dets=300).
  LVISMaskDataset  — evaluates predicted segmentation masks against LVIS ground
                     truth via a custom IoU-based matcher.  (Mask AP migration
                     to LVISEval is future work.)

Both classes expect pre-generated TSV files produced by:
    python -m mini_vlm_toolkit.dataset.scripts.prepare_lvis

TSV naming convention:
    LVIS_bbox_train.tsv / LVIS_bbox_val.tsv   (bounding-box evaluation)
    LVIS_mask_train.tsv / LVIS_mask_val.tsv   (segmentation-mask evaluation)

Each row represents a unique (image, category) pair.  The 'answer' column
holds the ground-truth annotations in JSON format.

Bbox evaluation is **category-conditioned ("Restricted LVIS")**: the GT JSON
is filtered to the (image_id, category_id) pairs in our TSV before LVISEval
runs.  Paper text should describe this as category-conditioned LVIS grounding,
not full LVIS detection.

Usage:
    python run_benchmark.py --data LVIS_bbox_val ...
    python run_benchmark.py --data LVIS_mask_val ...
"""

import json
import logging
import os
import os.path as osp
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
      1. JSON array:   [[100, 200, 300, 400], [500, 600, 700, 800]]
      2. Single box:   [100, 200, 300, 400]
      3. Loose numbers: fall back to grouping every 4 numbers as a box

    Returns a list of [x1, y1, x2, y2] lists in raw model-emitted units.
    Use _normalise_boxes() to canonicalise the scale.
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
    """Ensure boxes are on a 0-1000 normalised coordinate scale (Qwen-style).

    Heuristic:
      • max <= 1.5            → legacy 0-1 normalised input; rescale to 0-1000.
      • max <= 1024           → already on the 0-1000 scale; pass through.
                                The 24-unit slop covers floating-point overshoot
                                from prepare scripts that compute v/dim*1000 and
                                land at 1000.0+epsilon.
      • otherwise             → absolute pixel coordinates; divide by image
                                dimensions and rescale to 0-1000.
    """
    if not boxes:
        return boxes

    max_val = max(max(abs(v) for v in box) for box in boxes)

    if max_val <= 1.5:
        # Legacy 0-1 normalised; rescale to 0-1000
        return [[v * 1000.0 for v in box] for box in boxes]
    elif max_val <= 1024:
        # Already on the 0-1000 scale (e.g. Qwen models)
        return [[float(v) for v in box] for box in boxes]
    else:
        # Absolute pixel coordinates → 0-1000
        normalised = []
        for box in boxes:
            normalised.append([
                box[0] / img_w * 1000.0, box[1] / img_h * 1000.0,
                box[2] / img_w * 1000.0, box[3] / img_h * 1000.0,
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

def _resolve_lvis_gt_path(dataset_name: str) -> str:
    """Locate the LVIS annotation JSON (lvis_v1_<split>.json).

    Resolution order:
      1. Env var ``LVIS_GT_PATH`` — full path to the JSON file.
      2. Env var ``LVIS_DATA_ROOT`` — directory containing
         ``lvis_v1_<split>.json``.
      3. ``${LMUData}/lvis-api/data/lvis_v1_<split>.json``.
      4. ``${LMUData}/lvis_v1_<split>.json``.

    The split is inferred from the dataset name (``LVIS_bbox_val`` → ``val``,
    ``LVIS_bbox_train`` → ``train``).
    """
    split = 'val'
    if dataset_name.endswith('_train'):
        split = 'train'
    elif dataset_name.endswith('_val'):
        split = 'val'

    env_gt = os.environ.get('LVIS_GT_PATH')
    if env_gt and osp.isfile(env_gt):
        return env_gt

    fname = f'lvis_v1_{split}.json'
    candidates = []
    env_root = os.environ.get('LVIS_DATA_ROOT')
    if env_root:
        candidates.append(osp.join(env_root, fname))
    try:
        lmu_root = LMUDataRoot()
        candidates.append(osp.join(lmu_root, 'lvis-api', 'data', fname))
        candidates.append(osp.join(lmu_root, fname))
    except EnvironmentError:
        pass

    for cand in candidates:
        if osp.isfile(cand):
            return cand

    raise FileNotFoundError(
        f'Could not locate LVIS annotation JSON ({fname}). '
        f'Set LVIS_GT_PATH or LVIS_DATA_ROOT, or place the file under '
        f'$LMUData/lvis-api/data/.  Tried: {candidates + ([env_gt] if env_gt else [])}'
    )


class LVISBBoxDataset(ImageBaseDataset):
    """LVIS v1 bounding-box evaluation dataset.

    Each TSV row is a unique (image, category) pair.  The model is asked to
    locate all instances of a named category and return bounding boxes.

    Evaluation uses the official ``lvis.LVISEval`` and reports
    ``AP, AP50, AP75, APr, APc, APf`` with ``max_dets=300``.  This is
    category-conditioned LVIS grounding: the GT JSON is filtered to the
    (image_id, category_id) pairs present in the TSV before LVISEval runs.

    Phase 1 sets every prediction's score to 1.0; Phase 2 will plumb real
    SAM3 confidence scores via a sidecar JSON and switch ``score_mode`` to
    ``original`` for the BaseAgentWithTools path.

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

    # Phase 1 score handling.  Phase 2 will introduce a config switch.
    SCORE_MODE = 'constant'      # 'constant' | 'original'
    EVAL_SCOPE = 'category_conditioned'  # 'category_conditioned' | 'full_detection'
    LVIS_MAX_DETS = 300

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
        """Evaluate bounding-box predictions with the official LVIS API.

        Reports AP, AP50, AP75, APr, APc, APf following the LVIS detection
        benchmark, restricted to the (image_id, category_id) pairs present
        in this TSV (category-conditioned LVIS grounding).

        Outputs (alongside ``eval_file``):
          * ``..._metrics.json`` — full metrics blob (all LVISEval keys + meta).
          * ``..._acc.<fmt>``    — slim summary table (one row of metrics).
          * ``..._lvis_gt.json`` — the filtered GT used by LVISEval (kept for
                                    debugging / re-running the eval offline).
          * ``..._lvis_pred.json`` — the COCO-format prediction list submitted.
        """
        # lvis-api 0.5.3 still uses np.float (removed in NumPy 1.20+); patch
        # before import. Safe because lvis only uses it as a dtype alias for
        # builtin float.
        if not hasattr(np, 'float'):
            np.float = float  # type: ignore[attr-defined]
        try:
            from lvis import LVIS, LVISEval, LVISResults
        except ImportError as exc:
            raise ImportError(
                'lvis-api is required for LVIS bbox evaluation. '
                'Install it with `pip install lvis`.'
            ) from exc

        from .eval_bridge import (
            build_coco_predictions,
            build_sam3_score_lookup,
            extract_per_row_diagnostics,
            filter_coco_gt_to_pairs,
            parse_image_id_from_index,
            resolve_trajectory_path,
            write_json,
            _resolve_score_modes,
        )

        # Phase 2: decide which score_mode(s) to run.  When SCORE_MODE is
        # NOT explicitly set and sidecars are present, we run BOTH modes
        # in a single evaluate() call and emit two sets of metrics files
        # (primary + alternate) so the score-utility ablation needs no
        # second invocation.
        tool_detections_dir = osp.join(osp.dirname(eval_file), 'tool_detections')
        modes_to_run, primary_mode = _resolve_score_modes(tool_detections_dir)

        data = load(eval_file)
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        if 'prediction' not in data:
            raise KeyError('Prediction file must contain a `prediction` column.')

        # ── Mode-independent setup ──────────────────────────────────────
        # Join predictions with TSV metadata to recover image dims and category_id.
        meta = self.data.copy()
        meta['index'] = meta['index'].astype(str)
        meta = meta.set_index('index')

        rows = []
        skipped = 0
        for _, p_row in data.iterrows():
            idx = str(p_row['index'])
            if idx not in meta.index:
                skipped += 1
                continue
            m_row = meta.loc[idx]
            rows.append({
                'index': idx,
                'image_width': m_row.get('image_width'),
                'image_height': m_row.get('image_height'),
                'category_id': m_row.get('category_id'),
                'prediction': p_row.get('prediction', ''),
            })
        if skipped:
            logger.warning(
                'Skipped %d prediction rows whose index is not in the LVIS TSV.',
                skipped,
            )

        # Compute the (image_id, category_id) pairs the model was asked about.
        kept_pairs = set()
        for r in rows:
            iid = parse_image_id_from_index(r['index'])
            try:
                cid = int(r['category_id'])
            except (TypeError, ValueError):
                continue
            if iid is not None:
                kept_pairs.add((iid, cid))

        gt_path = _resolve_lvis_gt_path(self.dataset_name)
        logger.info('Loading LVIS GT from %s ...', gt_path)
        with open(gt_path) as f:
            gt = json.load(f)

        if self.EVAL_SCOPE == 'category_conditioned':
            filtered_gt = filter_coco_gt_to_pairs(
                gt, kept_pairs, keep_all_categories=True
            )
            logger.info(
                'Restricted LVIS GT to %d images / %d annotations '
                '(was %d / %d).',
                len(filtered_gt['images']), len(filtered_gt['annotations']),
                len(gt['images']), len(gt['annotations']),
            )
        else:
            filtered_gt = gt

        filtered_gt_path = get_intermediate_file_path(
            eval_file, '_lvis_gt', target_format='json'
        )
        write_json(filtered_gt, filtered_gt_path)

        # ── Per-mode loop ───────────────────────────────────────────────
        primary_summary = None
        for score_mode in modes_to_run:
            is_primary = (score_mode == primary_mode)
            file_suffix = '' if is_primary else f'_{score_mode}'

            score_lookup = None
            if score_mode == 'original':
                score_lookup = build_sam3_score_lookup(tool_detections_dir)

            coco_preds = build_coco_predictions(
                rows,
                score_mode=score_mode,
                score_lookup=score_lookup,
            )
            logger.info(
                '[LVIS %s] Built %d predictions across %d rows.',
                score_mode, len(coco_preds), len(rows),
            )

            pred_path = get_intermediate_file_path(
                eval_file, f'_lvis_pred{file_suffix}', target_format='json'
            )
            write_json(coco_preds, pred_path)

            if not coco_preds:
                logger.warning('[LVIS %s] No valid predictions to evaluate.', score_mode)
                results_dict = {
                    'AP': 0.0, 'AP50': 0.0, 'AP75': 0.0,
                    'APr': 0.0, 'APc': 0.0, 'APf': 0.0,
                    'APs': 0.0, 'APm': 0.0, 'APl': 0.0,
                }
                evaluator_obj = None
            else:
                lvis_gt = LVIS(filtered_gt_path)
                lvis_dt = LVISResults(lvis_gt, pred_path, max_dets=self.LVIS_MAX_DETS)
                evaluator = LVISEval(lvis_gt, lvis_dt, iou_type='bbox')
                evaluator.run()
                evaluator.print_results()
                raw = evaluator.get_results()
                # LVISEval may emit NaN for empty splits; coerce to 0.0.
                def _coerce(v):
                    try:
                        fv = float(v)
                        if np.isnan(fv) or np.isinf(fv):
                            return 0.0
                        return fv
                    except (TypeError, ValueError):
                        return 0.0
                results_dict = {k: _coerce(v) for k, v in raw.items()}
                evaluator_obj = evaluator

            # Per-row instance-level diagnostics CSV.
            # Walks evaluator.eval_imgs at IoU=0.5 / area=all and dumps
            # one CSV row per (image_id, category_id) the evaluator
            # considered, with TP/FP/FN counts and mean matched IoU.
            # The per-row identity is captured as ``trajectory_file``
            # (matches the row's TSV index + ``.json``), so users can
            # cross-reference straight to ``full_trajectories/<file>``
            # or ``raw_trajectories/<file>`` without parsing columns.
            if evaluator_obj is not None:
                per_row_records = extract_per_row_diagnostics(
                    evaluator_obj, iou_threshold=0.5, area_label='all',
                )
                # Attach LVIS frequency bucket so downstream analysis can
                # group by f/c/r without re-joining against the GT JSON.
                freq_by_cat = {
                    int(c['id']): c.get('frequency')
                    for c in filtered_gt.get('categories', [])
                }
                # Collapse (image_id, category_id) into a single
                # trajectory filename, matching the prepare_lvis row
                # index format ``<split>_<image_id>_<category_id>``.
                # Note: for LVIS, rows the evaluator auto-added (positive
                # categories not in the TSV) will reference filenames
                # that do not exist on disk; that's a clear signal that
                # the model never ran that (image, cat) pair.
                lvis_split = (
                    'train' if self.dataset_name.endswith('_train')
                    else 'val'
                )
                for r in per_row_records:
                    basename = (
                        f"{lvis_split}_{r['image_id']}_{r['category_id']}.json"
                    )
                    # Absolute path (preferring raw_trajectories/ when
                    # --save-raw-trajectory was set during inference,
                    # falling back to full_trajectories/, and empty
                    # string when neither exists — the LVIS auto-
                    # extended-row case).
                    r['trajectory_file'] = resolve_trajectory_path(
                        eval_file, basename
                    )
                    r['frequency'] = freq_by_cat.get(int(r['category_id']))
                    r.pop('image_id', None)
                    r.pop('category_id', None)
                per_row_df = pd.DataFrame(per_row_records)
                preferred_cols = [
                    'trajectory_file', 'frequency',
                    'num_pred', 'num_gt', 'tp', 'fp', 'fn', 'mean_match_iou',
                ]
                per_row_df = per_row_df[
                    [c for c in preferred_cols if c in per_row_df.columns]
                    + [c for c in per_row_df.columns if c not in preferred_cols]
                ]
                per_row_path = get_intermediate_file_path(
                    eval_file, f'_per_row{file_suffix}', target_format='csv'
                )
                dump(per_row_df, per_row_path)
                logger.info(
                    '[LVIS %s] wrote %d-row per-instance CSV to %s',
                    score_mode, len(per_row_df), per_row_path,
                )

            sidecar_diag = None
            if score_lookup is not None and hasattr(score_lookup, 'counters'):
                c = score_lookup.counters
                total = c['hits'] + c['misses']
                sidecar_diag = {
                    'sidecar_hits': c['hits'],
                    'sidecar_misses': c['misses'],
                    'sidecar_hit_rate': (c['hits'] / total) if total else 0.0,
                    'rows_with_sidecar': c['rows_with_sidecar'],
                    'rows_without_sidecar': c['rows_without_sidecar'],
                }

            metrics_blob = {
                'dataset': self.dataset_name,
                'evaluator': 'lvis.LVISEval',
                'iou_type': 'bbox',
                'max_dets': self.LVIS_MAX_DETS,
                'score_mode': score_mode,
                'eval_scope': self.EVAL_SCOPE,
                'num_predictions': len(coco_preds),
                'num_rows_evaluated': len(rows),
                'num_kept_pairs': len(kept_pairs),
                'metrics': results_dict,
            }
            if sidecar_diag is not None:
                metrics_blob['sidecar'] = sidecar_diag
            metrics_file = get_intermediate_file_path(
                eval_file, f'_metrics{file_suffix}', target_format='json'
            )
            dump(metrics_blob, metrics_file)

            # Slim summary CSV (per mode).
            def _pct(v):
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    return 'N/A'
                if fv < 0:
                    return 'N/A'
                return round(fv * 100, 2)

            summary_df = pd.DataFrame([{
                'Split': 'Overall',
                'score_mode': score_mode,
                'AP': _pct(results_dict.get('AP', 0.0)),
                'AP50': _pct(results_dict.get('AP50', 0.0)),
                'AP75': _pct(results_dict.get('AP75', 0.0)),
                'APr': _pct(results_dict.get('APr', 0.0)),
                'APc': _pct(results_dict.get('APc', 0.0)),
                'APf': _pct(results_dict.get('APf', 0.0)),
                'Samples': len(rows),
            }])
            score_file = get_intermediate_file_path(eval_file, f'_acc{file_suffix}')
            dump(summary_df, score_file)

            logger.info('[LVIS %s] complete: AP=%.4f AP50=%.4f',
                        score_mode, results_dict.get('AP', 0.0),
                        results_dict.get('AP50', 0.0))

            if is_primary:
                primary_summary = summary_df

        return primary_summary


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
                    # Derive bbox from mask (on the 0-1000 scale)
                    try:
                        bbox = mask_util.toBbox(rle).tolist()  # [x, y, w, h]
                        x1 = bbox[0] / img_w * 1000.0
                        y1 = bbox[1] / img_h * 1000.0
                        x2 = (bbox[0] + bbox[2]) / img_w * 1000.0
                        y2 = (bbox[1] + bbox[3]) / img_h * 1000.0
                        bboxes.append([x1, y1, x2, y2])
                    except Exception:
                        pass

        return masks, bboxes
