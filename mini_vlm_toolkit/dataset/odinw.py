"""ODinW (Object Detection in the Wild) dataset class for bbox evaluation.

Covers ODinW-13 (and ODinW-35 by the same code path).  ODinW has bbox
annotations only (no segmentation masks), so a single class suffices.

Pre-generated TSV files are produced by:
    python -m mini_vlm_toolkit.dataset.scripts.prepare_odinw

TSV naming:
    ODinW13_bbox_train.tsv / ODinW13_bbox_val.tsv
    ODinW35_bbox_train.tsv / ODinW35_bbox_val.tsv

Each row is a unique (image, category) pair within a domain.  The val TSV
contains both ``val`` and ``test`` rows (whichever splits the source
dataset provides); the per-row ``split`` column distinguishes them.

Evaluation uses **per-domain ``pycocotools.COCOeval``** with results
averaged across the 13 domains (the ODinW-13 reporting convention used by
GLIP / Grounding DINO).  Phase 1 is category-conditioned: for each domain
the GT JSON is filtered to the (image_id, category_id) pairs present in
the TSV before COCOeval runs.  Phase 1.5 will introduce a full-detection
TSV variant and skip this filtering.

Usage:
    python run_benchmark.py --data ODinW13_bbox_val ...
"""

import json
import logging
import os
import os.path as osp
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from .image_base import ImageBaseDataset
from ..smp import *  # noqa: F401,F403
from ..smp.file import LMUDataRoot, get_intermediate_file_path

# Imports from lvis.py kept for back-compat with any external callers that
# pulled them through this module (no longer used by the evaluator itself).
from .lvis import (  # noqa: F401
    _parse_bbox_array_from_text,
    _normalise_boxes,
    _compute_iou,
    _greedy_match,
)

logger = logging.getLogger(__name__)


# ── ODinW-13 domain list (must match prepare_odinw.ODINW_13_CONFIGS) ─────────
ODINW_13_CONFIGS = [
    'AerialMaritimeDrone_large',
    'Aquarium_Aquarium_Combined.v2-raw-1024.coco',
    'CottontailRabbits',
    'EgoHands_generic',
    'NorthAmericaMushrooms_North_American_Mushrooms.v1-416x416.coco',
    'Packages_Raw',
    'PascalVOC',
    'Raccoon_Raccoon.v2-raw.coco',
    'ShellfishOpenImages_raw',
    'VehiclesOpenImages_416x416',
    'pistols_export',
    'pothole',
    'thermalDogsAndPeople',
]


def _resolve_glip_root() -> str:
    """Locate the GLIP repository root (which contains ``configs/`` and
    ``DATASET/``).

    Resolution order:
      1. Env var ``GLIP_ROOT``.
      2. Env var ``ODINW_GLIP_ROOT`` (alias).
      3. ``${LMUData}/GLIP``.
    """
    for env in ('GLIP_ROOT', 'ODINW_GLIP_ROOT'):
        v = os.environ.get(env)
        if v and osp.isdir(v):
            return v
    try:
        cand = osp.join(LMUDataRoot(), 'GLIP')
        if osp.isdir(cand):
            return cand
    except EnvironmentError:
        pass
    raise FileNotFoundError(
        'Could not locate the GLIP repository. Set GLIP_ROOT to the directory '
        'containing configs/odinw_13/ and DATASET/odinw/.'
    )


def _build_odinw_gt_path_map(
    domains: List[str],
    glip_root: Optional[str] = None,
) -> Dict[Tuple[str, str], str]:
    """Build a {(domain, split): ann_file_path} map by parsing GLIP YAMLs.

    Mirrors ``prepare_odinw.parse_glip_config`` but at eval time.  The same
    YAML files that drove TSV creation drive eval-time GT lookup.
    """
    if glip_root is None:
        glip_root = _resolve_glip_root()
    cfg_dirs = [
        osp.join(glip_root, 'configs', 'odinw_13'),
        osp.join(glip_root, 'configs', 'odinw_35'),
    ]
    dataset_root = osp.join(glip_root, 'DATASET')
    out: Dict[Tuple[str, str], str] = {}

    for domain in domains:
        yaml_path = None
        for cdir in cfg_dirs:
            cand = osp.join(cdir, f'{domain}.yaml')
            if osp.isfile(cand):
                yaml_path = cand
                break
        if yaml_path is None:
            logger.warning('GLIP YAML not found for ODinW domain %r', domain)
            continue

        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        register = cfg.get('DATASETS', {}).get('REGISTER', {})

        for split_key, split_info in register.items():
            if split_key not in ('train', 'val', 'test'):
                continue
            ann_file = split_info.get('ann_file', '')
            if not ann_file:
                continue
            ann_abs = osp.join(dataset_root, ann_file)
            if not osp.isfile(ann_abs):
                logger.warning(
                    'ODinW ann_file missing for %s/%s: %s',
                    domain, split_key, ann_abs,
                )
                continue
            out[(domain, split_key)] = ann_abs

    return out


def _evaluate_coco_bbox(annotation_json: str, predictions: List[Dict]) -> Dict:
    """Run pycocotools.COCOeval for one domain.  Returns the 6 AP metrics."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(annotation_json)
    if not predictions:
        return {k: 0.0 for k in ('AP', 'AP50', 'AP75', 'APs', 'APm', 'APl')}

    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, iouType='bbox')
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    def _coerce(v):
        try:
            fv = float(v)
            if np.isnan(fv) or np.isinf(fv):
                return 0.0
            return fv
        except (TypeError, ValueError):
            return 0.0

    return {
        'AP':   _coerce(evaluator.stats[0]),
        'AP50': _coerce(evaluator.stats[1]),
        'AP75': _coerce(evaluator.stats[2]),
        'APs':  _coerce(evaluator.stats[3]),
        'APm':  _coerce(evaluator.stats[4]),
        'APl':  _coerce(evaluator.stats[5]),
    }


class ODinWBBoxDataset(ImageBaseDataset):
    """ODinW-13 / ODinW-35 bounding-box evaluation dataset.

    Each TSV row is a (image, category) pair within a domain.  The model
    locates all instances of a named category and returns bounding boxes.

    Evaluation runs ``pycocotools.COCOeval`` per domain and averages the
    six AP metrics (AP, AP50, AP75, APs, APm, APl) across the 13 domains
    — the ODinW-13 reporting convention from GLIP / Grounding DINO.

    Phase 1 is category-conditioned: for each domain the GT JSON is
    filtered to (image_id, category_id) pairs present in the TSV.

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

    SCORE_MODE = 'constant'              # 'constant' | 'original'
    EVAL_SCOPE = 'category_conditioned'  # 'category_conditioned' | 'full_detection'

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
        """Evaluate bbox predictions with ``pycocotools.COCOeval`` per domain.

        Reports AP, AP50, AP75, APs, APm, APl per domain plus a simple mean
        across domains (``ODinW13_average`` / ``ODinW35_average``).

        Outputs alongside ``eval_file``:
          * ``..._metrics.json`` — full per-domain + average metrics blob.
          * ``..._acc.<fmt>``    — slim summary table (one row per domain
                                    plus an Average row).
          * ``..._odinw_pred.json`` — combined COCO-format predictions.
        """
        try:
            from pycocotools.coco import COCO  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                'pycocotools is required for ODinW bbox evaluation. '
                'Install it with `pip install pycocotools`.'
            ) from exc

        from .eval_bridge import (
            build_coco_predictions,
            filter_coco_gt_to_pairs,
            parse_image_id_from_index,
            write_json,
        )

        data = load(eval_file)
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        if 'prediction' not in data:
            raise KeyError('Prediction file must contain a `prediction` column.')

        # Join predictions with TSV metadata.
        meta = self.data.copy()
        meta['index'] = meta['index'].astype(str)
        meta = meta.set_index('index')

        # Group rows by (domain, split); ODinW eval uses per-domain COCOeval
        # and the GT JSON differs per (domain, split).
        rows_by_domain_split: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
        skipped = 0
        for _, p_row in data.iterrows():
            idx = str(p_row['index'])
            if idx not in meta.index:
                skipped += 1
                continue
            m_row = meta.loc[idx]
            domain = str(m_row.get('domain', 'unknown'))
            split = str(m_row.get('split', 'test'))
            rows_by_domain_split[(domain, split)].append({
                'index': idx,
                'image_width': m_row.get('image_width'),
                'image_height': m_row.get('image_height'),
                'category_id': m_row.get('category_id'),
                'prediction': p_row.get('prediction', ''),
            })
        if skipped:
            logger.warning(
                'Skipped %d prediction rows whose index is not in the ODinW TSV.',
                skipped,
            )

        # Resolve GT paths once.
        domains_seen = sorted({d for d, _ in rows_by_domain_split.keys()})
        gt_path_map = _build_odinw_gt_path_map(domains_seen)

        per_domain: Dict[str, Dict] = {}
        all_preds: List[Dict] = []
        domain_splits_evaluated: List[str] = []

        for (domain, split), rows in sorted(rows_by_domain_split.items()):
            gt_path = gt_path_map.get((domain, split))
            if gt_path is None:
                logger.warning(
                    'No GT JSON for ODinW (%s, %s); skipping %d rows.',
                    domain, split, len(rows),
                )
                continue

            preds = build_coco_predictions(rows, score_mode=self.SCORE_MODE)
            all_preds.extend(preds)

            # Per-domain category-conditioned restriction.
            kept_pairs = set()
            for r in rows:
                iid = parse_image_id_from_index(r['index'])
                try:
                    cid = int(r['category_id'])
                except (TypeError, ValueError):
                    continue
                if iid is not None:
                    kept_pairs.add((iid, cid))

            with open(gt_path) as f:
                gt = json.load(f)

            if self.EVAL_SCOPE == 'category_conditioned':
                eval_pairs = kept_pairs
                keep_cats = False
            else:
                # Full-detection: keep all annotations the GT JSON has.
                eval_pairs = {
                    (int(a['image_id']), int(a['category_id']))
                    for a in gt['annotations']
                }
                keep_cats = True
            # remap_ann_ids=True is essential for ODinW because some COCO
            # JSONs (e.g. AerialMaritimeDrone) start annotation ids at 0,
            # which COCOeval interprets as the unmatched sentinel.
            filtered_gt = filter_coco_gt_to_pairs(
                gt, eval_pairs,
                keep_all_categories=keep_cats,
                remap_ann_ids=True,
            )
            # Drop GT annotations with zero/negative width or height — a
            # handful of ODinW JSONs (Aquarium, pistols, pothole) carry
            # degenerate boxes that COCOeval can't match against, dragging
            # AP down to ~0.99 even on perfect predictions.
            n_before = len(filtered_gt['annotations'])
            filtered_gt['annotations'] = [
                a for a in filtered_gt['annotations']
                if len(a.get('bbox', [])) >= 4
                and a['bbox'][2] > 0 and a['bbox'][3] > 0
            ]
            n_dropped = n_before - len(filtered_gt['annotations'])
            if n_dropped:
                logger.info(
                    'ODinW [%s/%s]: dropped %d degenerate (zero-area) GT annotations.',
                    domain, split, n_dropped,
                )
                # Re-id sequentially again after the drop.
                for new_id, ann in enumerate(filtered_gt['annotations'], start=1):
                    ann['id'] = new_id

            filtered_gt_path = get_intermediate_file_path(
                eval_file, f'_odinw_gt_{domain}_{split}', target_format='json'
            )
            write_json(filtered_gt, filtered_gt_path)

            logger.info(
                'ODinW [%s/%s]: %d rows → %d predictions, GT %d images / %d ann.',
                domain, split, len(rows), len(preds),
                len(filtered_gt['images']), len(filtered_gt['annotations']),
            )

            metrics = _evaluate_coco_bbox(filtered_gt_path, preds)
            per_domain[f'{domain}/{split}'] = {
                **metrics,
                'num_predictions': len(preds),
                'num_rows': len(rows),
                'num_kept_pairs': len(kept_pairs),
                'num_images': len(filtered_gt['images']),
                'num_annotations': len(filtered_gt['annotations']),
            }
            domain_splits_evaluated.append(f'{domain}/{split}')

        # Average across domains using the **test** split only.  This matches
        # the ODinW-13 reporting convention used by GLIP / Grounding DINO.
        # The val TSV bundles both val and test rows (per prepare_odinw.py);
        # both are reported per-domain but only test feeds the average.
        # Skip -1 ("no data") entries when averaging so APs/APm/APl don't
        # get pulled down by domains that have no small/medium/large boxes.
        metric_keys = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl']
        avg_source = (
            [d for d in per_domain if d.endswith('/test')]
            or list(per_domain.keys())
        )

        def _mean_skip_missing(vals):
            valid = [float(v) for v in vals if v is not None and float(v) >= 0]
            if not valid:
                return -1.0
            return float(np.mean(valid))

        if avg_source:
            avg = {
                k: _mean_skip_missing([per_domain[d][k] for d in avg_source])
                for k in metric_keys
            }
        else:
            avg = {k: 0.0 for k in metric_keys}
        test_keys = [d for d in per_domain if d.endswith('/test')]

        avg_label = (
            'ODinW13_average' if 'ODinW13' in self.dataset_name
            else 'ODinW35_average' if 'ODinW35' in self.dataset_name
            else 'ODinW_average'
        )

        # Save full metrics JSON.
        metrics_blob = {
            'dataset': self.dataset_name,
            'evaluator': 'pycocotools.COCOeval',
            'iou_type': 'bbox',
            'score_mode': self.SCORE_MODE,
            'eval_scope': self.EVAL_SCOPE,
            'num_predictions_total': len(all_preds),
            'domain_splits_evaluated': domain_splits_evaluated,
            'average_split': 'test' if test_keys else (
                domain_splits_evaluated[0].split('/', 1)[1] if domain_splits_evaluated else None
            ),
            'per_domain': per_domain,
            avg_label: avg,
        }
        metrics_file = get_intermediate_file_path(
            eval_file, '_metrics', target_format='json'
        )
        dump(metrics_blob, metrics_file)

        # Combined predictions JSON for offline re-runs.
        combined_pred_path = get_intermediate_file_path(
            eval_file, '_odinw_pred', target_format='json'
        )
        write_json(all_preds, combined_pred_path)

        # Slim summary DataFrame.  COCOeval emits -1 for empty area buckets
        # (e.g., a domain with only large boxes has APs=-1); render those as
        # 'N/A' rather than the misleading -100.0.  Raw -1.0 values stay in
        # the full metrics JSON for machine-readable consumers.
        def _pct(v):
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return 'N/A'
            if fv < 0:
                return 'N/A'
            return round(fv * 100, 2)

        rows_out: List[Dict] = []
        for d in sorted(per_domain.keys()):
            m = per_domain[d]
            rows_out.append({
                'Domain': d,
                'AP':   _pct(m['AP']),
                'AP50': _pct(m['AP50']),
                'AP75': _pct(m['AP75']),
                'APs':  _pct(m['APs']),
                'APm':  _pct(m['APm']),
                'APl':  _pct(m['APl']),
                'Samples': m['num_rows'],
            })
        rows_out.append({
            'Domain': avg_label,
            'AP':   _pct(avg['AP']),
            'AP50': _pct(avg['AP50']),
            'AP75': _pct(avg['AP75']),
            'APs':  _pct(avg['APs']),
            'APm':  _pct(avg['APm']),
            'APl':  _pct(avg['APl']),
            'Samples': sum(per_domain[d]['num_rows'] for d in per_domain) if per_domain else 0,
        })
        summary_df = pd.DataFrame(rows_out)
        score_file = get_intermediate_file_path(eval_file, '_acc')
        dump(summary_df, score_file)

        logger.info('ODinW bbox evaluation complete: %s = %s', avg_label, avg)
        return summary_df
