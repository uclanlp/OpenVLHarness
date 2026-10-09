"""FSCD-147 and FSCD-LVIS dataset classes for bbox-grounding evaluation.

Each TSV row is a (image, class) grounding query — the model locates every
instance of a named class in one image and returns bounding boxes on a 0-1000
normalised scale. Evaluation runs a single category-conditioned
``pycocotools.COCOeval`` pass over a GT built on the fly from the TSV's
``answer`` column, plus a counting-accuracy block (MAE / RMSE / NAE / EM /
acc@5% / acc@10%) derived from the per-row predicted vs. ground-truth instance
counts.

Pre-generated TSVs:
    python -m evaluation_kit.dataset.scripts.prepare_fscd147_box
    python -m evaluation_kit.dataset.scripts.prepare_fscdlvis_box

TSV naming:
    FSCD_147_bbox_val.tsv, FSCD_147_bbox_test.tsv
    FSCD_LVIS_bbox_{train,val,test}.tsv
    FSCD_LVIS_unseen_bbox_{train,test}.tsv

Usage:
    python run_benchmark.py --data FSCD_147_bbox_val ...
    python run_benchmark.py --data FSCD_LVIS_bbox_val ...
"""

import json
import logging
import math
import os.path as osp
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .image_base import ImageBaseDataset
from .lvis import _normalise_boxes, _parse_bbox_array_from_text
from ..smp import *  # noqa: F401,F403
from ..smp.file import get_intermediate_file_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _xyxy_norm1k_to_abs_xywh(
    boxes_xyxy_norm1k, img_w: float, img_h: float
) -> List[List[float]]:
    """Convert 0-1000 xyxy boxes to absolute pixel xywh.

    Tolerates inverted boxes (x2<x1, y2<y1) and clips to image bounds.
    Drops degenerate (zero-area) results.
    """
    out: List[List[float]] = []
    for box in boxes_xyxy_norm1k:
        if box is None or len(box) < 4:
            continue
        x1 = float(box[0]) / 1000.0 * img_w
        y1 = float(box[1]) / 1000.0 * img_h
        x2 = float(box[2]) / 1000.0 * img_w
        y2 = float(box[3]) / 1000.0 * img_h
        x1 = max(0.0, min(x1, img_w))
        y1 = max(0.0, min(y1, img_h))
        x2 = max(0.0, min(x2, img_w))
        y2 = max(0.0, min(y2, img_h))
        x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
        y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)
        w = x_max - x_min
        h = y_max - y_min
        if w > 0 and h > 0:
            out.append([float(x_min), float(y_min), float(w), float(h)])
    return out


def _build_synthetic_gt_from_meta(meta_rows: List[Dict]) -> Dict:
    """Build a COCO-format GT dict from TSV rows (one row per image-class query).

    Each row contributes one image entry and one annotation per box in its
    ``answer`` column. Annotation ids start at 1 to avoid COCOeval's
    ``id=0`` "no-match sentinel" bug.
    """
    images: List[Dict] = []
    annotations: List[Dict] = []
    cat_ids = set()
    ann_id = 1
    for m in meta_rows:
        image_id = m['image_id']
        img_w = m['image_width']
        img_h = m['image_height']
        cat_id = m['category_id']
        cat_ids.add(int(cat_id))
        images.append({
            'id': int(image_id),
            'file_name': m.get('image_path', f'{image_id}.jpg'),
            'width': int(img_w),
            'height': int(img_h),
        })
        try:
            gt_boxes_norm = json.loads(m['answer'])
        except (TypeError, ValueError, json.JSONDecodeError):
            gt_boxes_norm = []
        gt_boxes_xywh = _xyxy_norm1k_to_abs_xywh(
            gt_boxes_norm, float(img_w), float(img_h)
        )
        for box_xywh in gt_boxes_xywh:
            annotations.append({
                'id':           ann_id,
                'image_id':     int(image_id),
                'category_id':  int(cat_id),
                'bbox':         box_xywh,
                'area':         float(box_xywh[2] * box_xywh[3]),
                'iscrowd':      0,
            })
            ann_id += 1

    categories = [
        {'id': cid, 'name': str(cid)} for cid in sorted(cat_ids)
    ]
    return {
        # pycocotools.COCO.loadRes() deepcopies these top-level keys, so they
        # must exist even if empty — KeyError otherwise.
        'info':        {},
        'licenses':    [],
        'images':      images,
        'annotations': annotations,
        'categories':  categories,
    }


def _coerce_float(v) -> float:
    """Coerce a metric to float, NaN/Inf → 0.0."""
    try:
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv):
            return 0.0
        return fv
    except (TypeError, ValueError):
        return 0.0


def _run_cocoeval(gt_dict: Dict, predictions: List[Dict]) -> Tuple[Dict, Optional[object]]:
    """Run ``pycocotools.COCOeval`` and return (stats_dict, evaluator).

    ``evaluator`` is None when no predictions were submitted.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import tempfile
    import json as _json
    import os as _os

    if not predictions:
        return ({k: 0.0 for k in ('AP', 'AP50', 'AP75', 'APs', 'APm', 'APl')}, None)

    # COCO() needs a path; we don't have one for the synthetic GT, so write
    # a temp file. Caller handles persisting the GT itself.
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.json', delete=False
    ) as tf:
        _json.dump(gt_dict, tf)
        gt_path = tf.name
    try:
        coco_gt = COCO(gt_path)
        coco_dt = coco_gt.loadRes(predictions)
        evaluator = COCOeval(coco_gt, coco_dt, iouType='bbox')
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        # Emit AP* on the standard 0-100 percentage scale (matches AP_pct in
        # _acc.csv and the convention used by LVIS / ODinW reporting).
        stats = {
            'AP':   round(_coerce_float(evaluator.stats[0]) * 100.0, 2),
            'AP50': round(_coerce_float(evaluator.stats[1]) * 100.0, 2),
            'AP75': round(_coerce_float(evaluator.stats[2]) * 100.0, 2),
            'APs':  round(_coerce_float(evaluator.stats[3]) * 100.0, 2),
            'APm':  round(_coerce_float(evaluator.stats[4]) * 100.0, 2),
            'APl':  round(_coerce_float(evaluator.stats[5]) * 100.0, 2),
        }
        return stats, evaluator
    finally:
        try:
            _os.unlink(gt_path)
        except OSError:
            pass


def _compute_counting_metrics(
    per_row_counts: List[Tuple[int, int]],
) -> Dict[str, float]:
    """Compute aggregate counting metrics from (pred_count, gt_count) pairs.

    Returns MAE, RMSE (raw instance-count units) and NAE, EM, acc@5%, acc@10%
    (all scaled to 0-100 to match the AP* convention used elsewhere).
    """
    n = len(per_row_counts)
    if n == 0:
        return {
            'count_MAE': 0.0, 'count_RMSE': 0.0, 'count_NAE': 0.0,
            'count_EM': 0.0, 'count_acc_5pct': 0.0, 'count_acc_10pct': 0.0,
            'count_rows': 0,
        }
    abs_errs = [abs(p - g) for p, g in per_row_counts]
    mae = sum(abs_errs) / n
    rmse = math.sqrt(sum(e * e for e in abs_errs) / n)
    norm_errs = [
        abs_errs[i] / per_row_counts[i][1]
        for i in range(n) if per_row_counts[i][1] > 0
    ]
    nae = (sum(norm_errs) / len(norm_errs)) if norm_errs else 0.0
    em = sum(1 for p, g in per_row_counts if p == g) / n
    # acc@k%: |Δ|/max(gt,1) <= k. Using max(gt,1) avoids division by zero
    # for the (rare in this benchmark) gt=0 case while keeping the threshold
    # meaningful.
    acc5 = sum(
        1 for p, g in per_row_counts
        if abs(p - g) / max(g, 1) <= 0.05
    ) / n
    acc10 = sum(
        1 for p, g in per_row_counts
        if abs(p - g) / max(g, 1) <= 0.10
    ) / n
    # MAE/RMSE stay in raw instance units. NAE/EM/acc bands scaled to 0-100
    # so they read like AP percentages in the summary CSV. All to 2 dp.
    return {
        'count_MAE':       round(mae, 2),
        'count_RMSE':      round(rmse, 2),
        'count_NAE':       round(nae * 100.0, 2),
        'count_EM':        round(em * 100.0, 2),
        'count_acc_5pct':  round(acc5 * 100.0, 2),
        'count_acc_10pct': round(acc10 * 100.0, 2),
        'count_rows':      n,
    }


# ---------------------------------------------------------------------------
# Base class — shared by FSCD-147 and FSCD-LVIS
# ---------------------------------------------------------------------------

class _FSCDBBoxBase(ImageBaseDataset):
    """Common bbox-grounding eval flow for FSCD-147 and FSCD-LVIS.

    The model is asked to locate every instance of one class in one image
    and emit boxes on a 0-1000 normalised scale (same prompt as ODinW/LVIS).

    Evaluation:
      * Build a synthetic COCO GT dict from the TSV's ``answer`` column.
      * Build COCO-format predictions via
        :func:`evaluation_kit.dataset.eval_bridge.build_coco_predictions`.
      * Category-conditioned filter to the (image_id, category_id) pairs
        actually queried, then run ``pycocotools.COCOeval`` once.
      * Extract per-(image, cat) TP/FP/FN diagnostics via
        :func:`extract_per_row_diagnostics`.
      * Compute counting metrics (MAE / RMSE / NAE / EM / acc@5% / acc@10%)
        from per-row predicted vs. ground-truth instance counts.

    Outputs alongside ``eval_file``:
      ``_metrics.json``  full metrics blob (AP* + counting + meta)
      ``_acc.<fmt>``     slim summary table
      ``_per_row.csv``   one row per (image, category) with trajectory
                          path, tp/fp/fn/mean_iou, pred_count/gt_count/abs_err
      ``_fscd_gt.json``  the synthetic GT JSON fed to COCOeval (for debug)
      ``_fscd_pred.json`` the COCO-format predictions submitted
    """

    TYPE = 'GROUNDING'
    MODALITY = 'IMAGE'

    SCORE_MODE = 'constant'  # 'constant' | 'original' (Phase 2 sidecar)
    EVAL_SCOPE = 'category_conditioned'

    def build_prompt(self, line):
        """Image + question from the TSV."""
        if isinstance(line, int):
            line = self.data.iloc[line]
        tgt_path = self.dump_image(line)
        msgs = [dict(type='image', value=p) for p in tgt_path]
        msgs.append(dict(type='text', value=line['question']))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from pycocotools.coco import COCO  # noqa: F401  (import test)
        from .eval_bridge import (
            _resolve_score_modes,
            build_coco_predictions,
            build_sam3_score_lookup,
            extract_per_row_diagnostics,
            filter_coco_gt_to_pairs,
            parse_image_id_from_index,
            resolve_trajectory_path,
            write_json,
            _sort_grounding_per_row,
        )

        # Config takes precedence over SCORE_MODE (matches LVIS/ODinW).
        tool_detections_dir = osp.join(osp.dirname(eval_file), 'tool_detections')
        modes_to_run, primary_mode = _resolve_score_modes(
            tool_detections_dir, score_mode=judge_kwargs.get('score_mode'),
        )

        data = load(eval_file)
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        if 'prediction' not in data:
            raise KeyError('Prediction file must contain a `prediction` column.')

        meta = self.data.copy()
        meta['index'] = meta['index'].astype(str)
        meta = meta.set_index('index')

        # Build the eval rows by joining predictions with TSV metadata.
        rows: List[Dict] = []
        skipped = 0
        for _, p_row in data.iterrows():
            idx = str(p_row['index'])
            if idx not in meta.index:
                skipped += 1
                continue
            m_row = meta.loc[idx]
            image_id = parse_image_id_from_index(idx)
            if image_id is None:
                skipped += 1
                continue
            try:
                cat_id = int(m_row['category_id'])
            except (TypeError, ValueError, KeyError):
                skipped += 1
                continue
            try:
                num_instances = int(m_row.get('num_instances', 0))
            except (TypeError, ValueError):
                num_instances = 0
            rows.append({
                'index':          idx,
                'image_id':       int(image_id),
                'category_id':    cat_id,
                'image_width':    m_row.get('image_width'),
                'image_height':   m_row.get('image_height'),
                'image_path':     m_row.get('image_path'),
                'answer':         m_row.get('answer'),
                'num_instances':  num_instances,
                'prediction':     p_row.get('prediction', ''),
            })
        if skipped:
            logger.warning(
                'FSCD eval: skipped %d prediction rows missing in TSV metadata.',
                skipped,
            )

        # Build the synthetic GT from the TSV — one image entry per row,
        # annotations decoded from the row's ``answer`` column.
        full_gt = _build_synthetic_gt_from_meta(rows)

        # Category-conditioned restriction: keep only the (image, cat) pairs
        # the model was actually queried about. For our TSVs this is a no-op
        # (the synthetic GT already only contains queried pairs), but going
        # through the helper preserves the ann-id-remap that COCOeval needs.
        kept_pairs = {(r['image_id'], r['category_id']) for r in rows}
        filtered_gt = filter_coco_gt_to_pairs(
            full_gt, kept_pairs,
            keep_all_categories=False,
            remap_ann_ids=True,
        )

        # Drop zero-area GT (defensive — shouldn't happen from our prep).
        n_before = len(filtered_gt['annotations'])
        filtered_gt['annotations'] = [
            a for a in filtered_gt['annotations']
            if len(a.get('bbox', [])) >= 4
            and a['bbox'][2] > 0 and a['bbox'][3] > 0
        ]
        n_dropped = n_before - len(filtered_gt['annotations'])
        if n_dropped:
            logger.info('FSCD eval: dropped %d zero-area GT annotations.', n_dropped)
            for new_id, ann in enumerate(filtered_gt['annotations'], start=1):
                ann['id'] = new_id

        gt_path = get_intermediate_file_path(
            eval_file, '_fscd_gt', target_format='json'
        )
        write_json(filtered_gt, gt_path)

        # Precompute per-row predicted bbox counts (in canonicalised
        # 0-1000-to-abs-xywh form, matching what build_coco_predictions
        # would emit). One pass, reused per score_mode.
        pred_counts_by_idx: Dict[str, int] = {}
        for r in rows:
            pv = r.get('prediction', '')
            if pv is None or (isinstance(pv, float) and pv != pv):
                pred_counts_by_idx[r['index']] = 0
                continue
            text = str(pv).strip()
            if not text:
                pred_counts_by_idx[r['index']] = 0
                continue
            parsed = _parse_bbox_array_from_text(text)
            try:
                w = float(r['image_width']); h = float(r['image_height'])
            except (TypeError, ValueError):
                pred_counts_by_idx[r['index']] = 0
                continue
            if w <= 0 or h <= 0:
                pred_counts_by_idx[r['index']] = 0
                continue
            norm = _normalise_boxes(parsed, w, h)
            valid = _xyxy_norm1k_to_abs_xywh(norm, w, h)
            pred_counts_by_idx[r['index']] = len(valid)

        # Per-mode eval loop.
        primary_summary = None
        for score_mode in modes_to_run:
            is_primary = (score_mode == primary_mode)
            file_suffix = '' if is_primary else f'_{score_mode}'

            score_lookup = None
            if score_mode == 'original':
                score_lookup = build_sam3_score_lookup(tool_detections_dir)

            preds = build_coco_predictions(
                rows, score_mode=score_mode, score_lookup=score_lookup,
                output_dir=osp.dirname(eval_file),
            )

            pred_path = get_intermediate_file_path(
                eval_file, f'_fscd_pred{file_suffix}', target_format='json'
            )
            write_json(preds, pred_path)

            logger.info(
                'FSCD [%s] mode=%s: %d rows → %d predictions; GT %d images / %d ann.',
                self.dataset_name, score_mode, len(rows), len(preds),
                len(filtered_gt['images']), len(filtered_gt['annotations']),
            )

            stats, evaluator = _run_cocoeval(filtered_gt, preds)

            # Build per-row CSV. Order: trajectory_file, tp/fp/fn columns,
            # then count diagnostics. We start from the COCOeval per-row
            # records and merge in pred_count/gt_count/count_abs_err.
            per_row_records: List[Dict] = []
            if evaluator is not None:
                per_row_records = extract_per_row_diagnostics(
                    evaluator, iou_threshold=0.5, area_label='all',
                )

            # Map (image_id, category_id) → TSV row for joining counts.
            row_by_pair: Dict[Tuple[int, int], Dict] = {
                (r['image_id'], r['category_id']): r for r in rows
            }
            split_token = ''  # used only as a fallback when no row matches
            try:
                split_token = str(self.data.iloc[0].get('split', ''))
            except (IndexError, KeyError):
                pass

            merged_per_row: List[Dict] = []
            seen_pairs = set()
            for rec in per_row_records:
                key = (int(rec['image_id']), int(rec['category_id']))
                seen_pairs.add(key)
                tsv_row = row_by_pair.get(key)
                if tsv_row is not None:
                    basename = f"{tsv_row['index']}.json"
                    pred_count = pred_counts_by_idx.get(tsv_row['index'], 0)
                    gt_count = tsv_row.get('num_instances', 0)
                else:
                    basename = f"{split_token}_{key[0]}_{key[1]}.json"
                    pred_count = 0
                    gt_count = 0
                rec['trajectory_file'] = resolve_trajectory_path(
                    eval_file, basename
                )
                rec['pred_count'] = int(pred_count)
                rec['gt_count'] = int(gt_count)
                rec['count_abs_err'] = abs(int(pred_count) - int(gt_count))
                rec.pop('image_id', None)
                rec.pop('category_id', None)
                merged_per_row.append(rec)

            # Add rows the evaluator skipped (no GT and no detection — they
            # don't show up in evalImgs) so the CSV is exhaustive.
            for key, tsv_row in row_by_pair.items():
                if key in seen_pairs:
                    continue
                basename = f"{tsv_row['index']}.json"
                gt_count = tsv_row.get('num_instances', 0)
                pred_count = pred_counts_by_idx.get(tsv_row['index'], 0)
                merged_per_row.append({
                    'trajectory_file': resolve_trajectory_path(eval_file, basename),
                    'num_pred':        int(pred_count),
                    'num_gt':          int(gt_count),
                    'tp':              0,
                    'fp':              0,
                    'fn':              int(gt_count),
                    'mean_match_iou':  None,
                    'pred_count':      int(pred_count),
                    'gt_count':        int(gt_count),
                    'count_abs_err':   abs(int(pred_count) - int(gt_count)),
                })

            merged_per_row = _sort_grounding_per_row(merged_per_row)
            per_row_df = pd.DataFrame(merged_per_row)
            preferred = [
                'trajectory_file',
                'num_pred', 'num_gt', 'tp', 'fp', 'fn', 'mean_match_iou',
                'pred_count', 'gt_count', 'count_abs_err',
            ]
            per_row_df = per_row_df[
                [c for c in preferred if c in per_row_df.columns]
                + [c for c in per_row_df.columns if c not in preferred]
            ]
            per_row_path = get_intermediate_file_path(
                eval_file, f'_per_row{file_suffix}', target_format='csv'
            )
            dump(per_row_df, per_row_path)
            logger.info(
                '[FSCD %s] wrote %d-row per-instance CSV to %s',
                score_mode, len(per_row_df), per_row_path,
            )

            # Counting metrics, computed over every TSV row (not just rows
            # the COCOeval considered) so missing predictions count as 0.
            per_row_counts: List[Tuple[int, int]] = [
                (pred_counts_by_idx.get(r['index'], 0), r['num_instances'])
                for r in rows
            ]
            counting = _compute_counting_metrics(per_row_counts)

            # Optional sidecar diagnostics.
            sidecar_diag = None
            if score_lookup is not None and hasattr(score_lookup, 'counters'):
                c = score_lookup.counters
                total = c['hits'] + c['misses']
                sidecar_diag = {
                    'sidecar_hits':         c['hits'],
                    'sidecar_misses':       c['misses'],
                    'sidecar_hit_rate':     (c['hits'] / total) if total else 0.0,
                    'rows_with_sidecar':    c['rows_with_sidecar'],
                    'rows_without_sidecar': c['rows_without_sidecar'],
                }

            metrics_blob = {
                'dataset':            self.dataset_name,
                'evaluator':          'pycocotools.COCOeval',
                'iou_type':           'bbox',
                'score_mode':         score_mode,
                'eval_scope':         self.EVAL_SCOPE,
                'num_predictions':    len(preds),
                'num_rows_evaluated': len(rows),
                'num_kept_pairs':     len(kept_pairs),
                'detection':          stats,
                'counting':           counting,
            }
            if sidecar_diag is not None:
                metrics_blob['sidecar'] = sidecar_diag
            metrics_file = get_intermediate_file_path(
                eval_file, f'_metrics{file_suffix}', target_format='json'
            )
            dump(metrics_blob, metrics_file)

            def _pct(v):
                # Inputs are already on the 0-100 percentage scale (from
                # _run_cocoeval). Just round and handle missing/N/A.
                fv = _coerce_float(v)
                return round(fv, 2) if fv >= 0 else 'N/A'

            summary_df = pd.DataFrame([{
                'Split':         'Overall',
                'score_mode':    score_mode,
                'AP':            _pct(stats['AP']),
                'AP50':          _pct(stats['AP50']),
                'AP75':          _pct(stats['AP75']),
                'APs':           _pct(stats['APs']),
                'APm':           _pct(stats['APm']),
                'APl':           _pct(stats['APl']),
                'count_MAE':     counting['count_MAE'],
                'count_RMSE':    counting['count_RMSE'],
                'count_NAE':     counting['count_NAE'],
                'count_EM':      counting['count_EM'],
                'count_acc5':    counting['count_acc_5pct'],
                'count_acc10':   counting['count_acc_10pct'],
                'Samples':       len(rows),
            }])
            score_file = get_intermediate_file_path(eval_file, f'_acc{file_suffix}')
            dump(summary_df, score_file)

            logger.info(
                '[FSCD %s] %s complete: AP=%.4f AP50=%.4f MAE=%.2f EM=%.3f',
                score_mode, self.dataset_name,
                stats['AP'], stats['AP50'],
                counting['count_MAE'], counting['count_EM'],
            )

            if is_primary:
                primary_summary = summary_df

        return primary_summary


# ---------------------------------------------------------------------------
# Concrete classes
# ---------------------------------------------------------------------------

class FSCD147BBoxDataset(_FSCDBBoxBase):
    """FSCD-147 bbox-grounding evaluation (val and test)."""

    DATASET_URL = {
        'FSCD_147_bbox_val':  '',
        'FSCD_147_bbox_test': '',
    }
    DATASET_MD5 = {}


class FSCDLVISBBoxDataset(_FSCDBBoxBase):
    """FSCD-LVIS bbox-grounding evaluation (seen train/val/test + unseen train/test)."""

    DATASET_URL = {
        'FSCD_LVIS_bbox_train':         '',
        'FSCD_LVIS_bbox_val':           '',
        'FSCD_LVIS_bbox_test':          '',
        'FSCD_LVIS_unseen_bbox_train':  '',
        'FSCD_LVIS_unseen_bbox_test':   '',
    }
    DATASET_MD5 = {}
