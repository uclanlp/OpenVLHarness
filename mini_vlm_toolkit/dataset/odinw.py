"""ODinW (Object Detection in the Wild) dataset class for bbox evaluation.

Covers both ODinW-13 and ODinW-35 benchmarks.  ODinW has bounding-box
annotations only (no segmentation masks), so a single class suffices.

Pre-generated TSV files are produced by:
    python -m mini_vlm_toolkit.dataset.scripts.prepare_odinw

TSV naming:
    ODinW13_bbox_train.tsv / ODinW13_bbox_val.tsv
    ODinW35_bbox_train.tsv / ODinW35_bbox_val.tsv

Each row is a unique (image, category) pair within a domain.

Usage:
    python run_benchmark.py --data ODinW13_bbox_val ...
    python run_benchmark.py --data ODinW35_bbox_val ...
"""

import json
import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from .image_base import ImageBaseDataset
from ..smp import *  # noqa: F401,F403
from ..smp.file import get_intermediate_file_path

# Reuse bbox utilities from the LVIS module to avoid duplication.
from .lvis import (
    _parse_bbox_array_from_text,
    _normalise_boxes,
    _compute_iou,
    _greedy_match,
)

logger = logging.getLogger(__name__)


class ODinWBBoxDataset(ImageBaseDataset):
    """ODinW-13 / ODinW-35 bounding-box evaluation dataset.

    Each row is a (image, category) pair.  The model locates all instances
    of a named category and returns bounding boxes.

    Evaluation uses greedy IoU matching with **per-domain** breakdowns
    (instead of the per-frequency breakdown used by LVIS).

    Supported dataset names:
        ODinW13_bbox_train, ODinW13_bbox_val,
        ODinW35_bbox_train, ODinW35_bbox_val
    """

    TYPE = 'GROUNDING'
    MODALITY = 'IMAGE'

    DATASET_URL = {
        'ODinW13_bbox_train': '',
        'ODinW13_bbox_val': '',
        'ODinW35_bbox_train': '',
        'ODinW35_bbox_val': '',
    }
    DATASET_MD5 = {}

    IOU_THRESHOLD = 0.5

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
        """Evaluate bbox predictions against ODinW ground truth.

        Metrics (same as LVIS bbox):
          - Precision@50, Recall@50, F1@50, Mean IoU, Count Accuracy
          Broken down per domain + overall.

        Parameters
        ----------
        eval_file : str
            Path to prediction TSV (must have 'prediction' column).

        Returns
        -------
        summary_df : pd.DataFrame
        """
        data = load(eval_file)
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        if 'prediction' not in data:
            raise KeyError('Prediction file must contain a `prediction` column.')

        # Build metadata lookup
        meta = self.data.copy()
        meta['index'] = meta['index'].astype(str)
        meta = meta.set_index('index')

        detail_rows = []
        # Per-domain and overall accumulators
        group_stats = defaultdict(lambda: {
            'tp': 0, 'fp': 0, 'fn': 0,
            'iou_sum': 0.0, 'iou_count': 0,
            'count_correct': 0, 'total': 0,
        })

        for _, row in data.iterrows():
            idx = str(row['index'])
            pred_text = str(row.get('prediction', ''))

            if idx not in meta.index:
                logger.warning(f'Index {idx} not found in ODinW metadata, skipping.')
                continue

            meta_row = meta.loc[idx]
            img_w = float(meta_row.get('image_width', 1))
            img_h = float(meta_row.get('image_height', 1))
            domain = str(meta_row.get('domain', 'unknown'))

            # Parse GT bboxes
            try:
                gt_boxes = json.loads(str(meta_row['answer']))
            except (json.JSONDecodeError, KeyError):
                gt_boxes = []

            # Parse predicted bboxes
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
                'domain': domain,
                'gt_count': len(gt_boxes),
                'pred_count': len(pred_boxes),
                'tp': tp, 'fp': fp, 'fn': fn,
                'mean_iou': round(mean_iou, 4),
                'count_correct': count_correct,
                'prediction': pred_text[:500],
            })

            for group_key in ['overall', domain]:
                s = group_stats[group_key]
                s['tp'] += tp
                s['fp'] += fp
                s['fn'] += fn
                s['iou_sum'] += sum(match_ious)
                s['iou_count'] += len(match_ious)
                s['count_correct'] += count_correct
                s['total'] += 1

        # ── Build summary table ──────────────────────────────────────────
        summary_rows = []

        def _row(label, stats):
            tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
            prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            m_iou = stats['iou_sum'] / stats['iou_count'] if stats['iou_count'] > 0 else 0.0
            cnt_acc = stats['count_correct'] / stats['total'] * 100 if stats['total'] > 0 else 0.0
            return {
                'Domain': label,
                'Precision@50': round(prec, 2),
                'Recall@50': round(rec, 2),
                'F1@50': round(f1, 2),
                'Mean IoU': round(m_iou, 4),
                'Count Acc (%)': round(cnt_acc, 2),
                'Samples': stats['total'],
            }

        # Per-domain rows (alphabetical)
        for domain in sorted(group_stats.keys()):
            if domain == 'overall':
                continue
            summary_rows.append(_row(domain, group_stats[domain]))

        # Overall row
        if 'overall' in group_stats:
            summary_rows.append(_row('Overall', group_stats['overall']))

        summary_df = pd.DataFrame(summary_rows)

        # Save outputs
        detail_df = pd.DataFrame(detail_rows)
        dump(detail_df, get_intermediate_file_path(eval_file, '_detail'))
        dump(summary_df, get_intermediate_file_path(eval_file, '_acc'))

        logger.info(f'ODinW bbox evaluation complete. {len(detail_rows)} rows evaluated.')
        return summary_df
