"""ODinW-13 (Object Detection in the Wild) dataset class for bbox evaluation.

ODinW has bbox annotations only (no segmentation masks), so a single
class suffices.

Pre-generated TSV files are produced by:
    python -m openvlharness_eval.dataset.scripts.prepare_odinw

TSV naming:
    ODinW13_bbox_train.tsv / ODinW13_bbox_val.tsv / ODinW13_bbox_test.tsv

Each row is a unique (image, category) pair within a domain — full
detection, so negatives (no instances) are also represented.

Evaluation uses per-domain ``pycocotools.COCOeval`` with results averaged
across the 13 domains (the ODinW-13 reporting convention used by GLIP /
Grounding DINO).  GT is NOT filtered to predicted (image, cat) pairs; a
per-image completeness filter drops incomplete images so partial runs
don't artificially deflate AP.

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

from .hf_release import ensure_hf_glip
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
      3. ``${LMUData}/GLIP``, downloaded from the HuggingFace release
         (see ``hf_release``) if absent.
    """
    for env in ('GLIP_ROOT', 'ODINW_GLIP_ROOT'):
        v = os.environ.get(env)
        if v and osp.isdir(v):
            return v
    try:
        cand = ensure_hf_glip(LMUDataRoot())
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


def _evaluate_coco_bbox(
    annotation_json: str, predictions: List[Dict],
) -> Tuple[Dict, Optional['COCOeval']]:
    """Run pycocotools.COCOeval for one domain.

    Returns a 2-tuple ``(stats_dict, evaluator)``.  ``evaluator`` is the
    populated ``COCOeval`` instance (so callers can inspect ``evalImgs``
    / ``ious`` for per-row diagnostics) — or ``None`` if no predictions
    were submitted.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(annotation_json)
    if not predictions:
        return ({k: 0.0 for k in ('AP', 'AP50', 'AP75', 'APs', 'APm', 'APl')}, None)

    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, iouType='bbox')
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    def _coerce(v):
        # COCOeval emits -1 for "no GT in this size band"; preserve that
        # sentinel so callers/the summary formatter can render it as N/A.
        # Other valid stats get scaled to the 0-100 percentage scale.
        try:
            fv = float(v)
            if np.isnan(fv) or np.isinf(fv):
                return 0.0
            if fv < 0:
                return fv
            return round(fv * 100.0, 2)
        except (TypeError, ValueError):
            return 0.0

    stats = {
        'AP':   _coerce(evaluator.stats[0]),
        'AP50': _coerce(evaluator.stats[1]),
        'AP75': _coerce(evaluator.stats[2]),
        'APs':  _coerce(evaluator.stats[3]),
        'APm':  _coerce(evaluator.stats[4]),
        'APl':  _coerce(evaluator.stats[5]),
    }
    return stats, evaluator


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
        ODinW13_bbox_train, ODinW13_bbox_val, ODinW13_bbox_test
    """

    TYPE = 'GROUNDING'
    MODALITY = 'IMAGE'

    DATASET_URL = {
        # Full-detection (one row per (image, category) including negatives).
        'ODinW13_bbox_train': '',
        'ODinW13_bbox_val': '',
        'ODinW13_bbox_test': '',
        # Qwen2.5-VL protocol: positives-only, per-image class filtering.
        # All four toggle combinations are registered so any can be loaded
        # without re-editing this file. See
        # openvlharness_eval/dataset/scripts/prepare_odinw_qwen25.py.
        'ODinW13_bbox_val_qwen25': '',
        'ODinW13_bbox_val_qwen25_raw': '',
        'ODinW13_bbox_val_qwen25_adapted': '',
        'ODinW13_bbox_val_qwen25_adapted_raw': '',
        'ODinW13_bbox_test_qwen25': '',
        'ODinW13_bbox_test_qwen25_raw': '',
        'ODinW13_bbox_test_qwen25_adapted': '',
        'ODinW13_bbox_test_qwen25_adapted_raw': '',
        # Qwen3-VL protocol: all-categories-at-once, one row per image.
        # Model emits a JSON list of {bbox_2d, label} records; labels are
        # resolved to category_ids via the row's dataset_classes column.
        # See openvlharness_eval/dataset/scripts/prepare_odinw_qwen3.py.
        'ODinW13_bbox_val_qwen3': '',
        'ODinW13_bbox_val_qwen3_mmdet': '',
        'ODinW13_bbox_test_qwen3': '',
        'ODinW13_bbox_test_qwen3_mmdet': '',
    }
    DATASET_MD5 = {}

    SCORE_MODE = 'constant'              # 'constant' | 'original'
    # Default eval scope: full-detection (the canonical Phase-1.5 path).
    # Qwen2.5 variants override to ``category_conditioned`` at evaluate()
    # time via ``_resolve_eval_scope``.
    EVAL_SCOPE = 'full_detection'

    @staticmethod
    def _resolve_eval_scope(dataset_name: str) -> str:
        """Qwen variants need positives-only / multilabel paths.

        * ``_qwen25``: positives-only, per-image class filtering ->
          category_conditioned eval (GT restricted to queried pairs).
        * ``_qwen3``: one row per image, all-classes-joined prompt; the
          response itself names the classes -> qwen3_multilabel eval
          (GT scoped by image_id; per-row prediction parsed for multiple
          labels each mapped to a category_id).
        * everything else: full_detection (one row per (image, cat) with
          negatives explicitly in scope).
        """
        if '_qwen3' in dataset_name:
            return 'qwen3_multilabel'
        if '_qwen25' in dataset_name:
            return 'category_conditioned'
        return 'full_detection'

    @staticmethod
    def _resolve_coords_space(dataset_name: str) -> str:
        """Qwen-faithful variants emit 0-1000 normalised coordinates inside
        the ``bbox_2d`` JSON (empirically verified for Qwen3-VL-8B on
        2026-05-17 — output e.g. ``[370, 367, 534, 662]`` matches GT
        ``[367.79, 350.96, 533.65, 657.45]`` on a 416x416 image). Force
        ``norm1k`` so values pass through ``_normalise_boxes`` unscaled. The
        ``auto`` heuristic would also work here in the common case, but
        ``norm1k`` is explicit and avoids surprises if a degenerate output
        produces a single number >1024 that would otherwise trigger the
        pixel-coord branch. Adapted variants and full-detection TSVs ask
        the model for 0-1000 normalised coords directly; they pass through
        ``auto`` (which detects and handles either scale).

        Historical note: Qwen2.5-VL was documented as emitting pixel-space
        ``bbox_2d``; Qwen3-VL switched to 0-1000 normalised. The legacy
        ``pixel`` mode is preserved on ``_normalise_boxes`` for future
        Qwen2.5 reproductions.
        """
        if '_qwen3' in dataset_name:
            return 'norm1k'
        if '_qwen25' in dataset_name and '_adapted' not in dataset_name:
            return 'norm1k'
        return 'auto'

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
        across domains (``ODinW13_average``).

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
            _resolve_score_modes,
            build_coco_predictions,
            build_sam3_score_lookup,
            drop_incomplete_images,
            extract_per_row_diagnostics,
            filter_coco_gt_to_image_ids,
            filter_coco_gt_to_pairs,
            parse_image_id_from_index,
            resolve_trajectory_path,
            write_json,
        )

        # Config takes precedence over SCORE_MODE. Without either override,
        # sidecars select both modes (original primary, constant alternate).
        tool_detections_dir = osp.join(osp.dirname(eval_file), 'tool_detections')
        modes_to_run, primary_mode = _resolve_score_modes(
            tool_detections_dir, score_mode=judge_kwargs.get('score_mode'),
        )

        eval_scope = self._resolve_eval_scope(self.dataset_name)
        coords_space = self._resolve_coords_space(self.dataset_name)

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
                # Qwen3 multi-label rows carry the prompted vocabulary in
                # ``dataset_classes`` (a JSON list of {id, name} records);
                # build_coco_predictions consumes it to resolve labels to
                # category_ids. Older TSVs don't have this column — pandas
                # returns NaN/None and downstream code handles it.
                'dataset_classes': m_row.get('dataset_classes'),
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

        # Phase 1.5+: build expected-pair sets from the SOURCE TSV per
        # (domain, split). These are the (image_id, category_id) pairs the
        # model was supposed to be queried about, regardless of whether
        # inference completed each one. Used by the completeness filter
        # (full_detection / qwen3_multilabel) and by GT image_id filtering.
        #
        # Single-class rows (full_detection): one pair per row from
        # ``category_id``. Multi-label rows (qwen3): expand the row to
        # (image_id, every cat_id in dataset_classes).
        expected_pairs_by_ds: Dict[Tuple[str, str], set] = defaultdict(set)
        if eval_scope in ('full_detection', 'qwen3_multilabel'):
            for idx, m_row in meta.iterrows():
                d = str(m_row.get('domain', 'unknown'))
                s = str(m_row.get('split', 'test'))
                iid = parse_image_id_from_index(str(idx))
                if iid is None:
                    continue
                try:
                    cid = int(m_row['category_id'])
                    expected_pairs_by_ds[(d, s)].add((iid, cid))
                    continue
                except (TypeError, ValueError, KeyError):
                    pass
                # Multi-label row: explode by dataset_classes.
                try:
                    dataset_classes = json.loads(m_row.get('dataset_classes') or '[]')
                except (json.JSONDecodeError, TypeError):
                    dataset_classes = []
                for c in dataset_classes:
                    if isinstance(c, dict) and 'id' in c:
                        try:
                            expected_pairs_by_ds[(d, s)].add((iid, int(c['id'])))
                        except (TypeError, ValueError):
                            continue

        # ── Per-mode loop ───────────────────────────────────────────────
        primary_summary = None
        for score_mode in modes_to_run:
            is_primary = (score_mode == primary_mode)
            file_suffix = '' if is_primary else f'_{score_mode}'

            score_lookup = None
            if score_mode == 'original':
                score_lookup = build_sam3_score_lookup(tool_detections_dir)

            # Per-row instance-level diagnostics, accumulated across all
            # (domain, split) entries for this mode.
            all_per_row: List[Dict] = []

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

                preds = build_coco_predictions(
                    rows, score_mode=score_mode, score_lookup=score_lookup,
                    coords_space=coords_space,
                    mode='qwen3_multilabel' if eval_scope == 'qwen3_multilabel' else 'single',
                    output_dir=osp.dirname(eval_file),
                )
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

                # remap_ann_ids=True is essential for ODinW because some COCO
                # JSONs (e.g. AerialMaritimeDrone) start annotation ids at 0,
                # which COCOeval interprets as the unmatched sentinel.
                if eval_scope == 'category_conditioned':
                    # Phase 1 path: restrict GT to the (image, cat) pairs
                    # the model was asked about. Negative pairs aren't part
                    # of the eval task; their absence from GT is correct.
                    filtered_gt = filter_coco_gt_to_pairs(
                        gt, kept_pairs,
                        keep_all_categories=False,
                        remap_ann_ids=True,
                    )
                else:
                    # full_detection or qwen3_multilabel: restrict GT by
                    # **image_id** (not by (image, cat) pair). This
                    # preserves images that have zero annotations for some
                    # queried classes so the model's hallucinated boxes on
                    # those classes get counted as FPs by COCOeval — the
                    # correct full-detection / open-vocabulary semantics.
                    # Filtering by pair would drop those images and trigger
                    # the ``Results do not correspond to current coco set``
                    # assertion when the model emitted any prediction there.
                    tsv_image_ids_for_ds = {
                        iid for (iid, _cid)
                        in expected_pairs_by_ds.get((domain, split), set())
                    }
                    filtered_gt = filter_coco_gt_to_image_ids(
                        gt, tsv_image_ids_for_ds, remap_ann_ids=True,
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

                # Completeness filter. Drops any (image_id) whose full set
                # of expected category queries wasn't completed (i.e.
                # inference was killed mid-image).
                #   * full_detection: one row per (image, cat); a missing
                #     row means that (image, cat) pair wasn't queried.
                #   * qwen3_multilabel: one row per image with all classes
                #     in one prompt; a non-empty prediction implies every
                #     class for that image was effectively answered (the
                #     model chose which ones to emit boxes for). An empty
                #     prediction drops the whole image.
                completion_summary: Dict = {
                    'images_total': len(filtered_gt['images']),
                    'images_completed': len(filtered_gt['images']),
                    'images_dropped': 0,
                    'dropped_image_ids': [],
                }
                if eval_scope in ('full_detection', 'qwen3_multilabel'):
                    expected_pairs = expected_pairs_by_ds.get((domain, split), set())
                    completed_pairs = set()
                    for r in rows:
                        pv = r.get('prediction', None)
                        # Treat NaN / None / empty string as "not completed".
                        # Empty CSV cells round-trip as NaN through pandas.
                        if pv is None or (isinstance(pv, float) and pd.isna(pv)):
                            continue
                        if not str(pv).strip():
                            continue
                        iid = parse_image_id_from_index(r['index'])
                        if iid is None:
                            continue
                        # Single-class row: read category_id directly.
                        try:
                            cid = int(r['category_id'])
                            completed_pairs.add((iid, cid))
                            continue
                        except (TypeError, ValueError):
                            pass
                        # Multi-label row: a non-empty prediction implies
                        # every class for this image was queried. Mark all
                        # (iid, cat_id) pairs from dataset_classes as
                        # completed so drop_incomplete_images keeps the image.
                        try:
                            dataset_classes = json.loads(r.get('dataset_classes') or '[]')
                        except (json.JSONDecodeError, TypeError):
                            dataset_classes = []
                        for c in dataset_classes:
                            if isinstance(c, dict) and 'id' in c:
                                try:
                                    completed_pairs.add((iid, int(c['id'])))
                                except (TypeError, ValueError):
                                    continue
                    preds, filtered_gt, completion_summary = drop_incomplete_images(
                        preds, filtered_gt, expected_pairs, completed_pairs,
                    )
                    if completion_summary['images_dropped']:
                        logger.warning(
                            'ODinW [%s/%s]: dropped %d/%d images for incomplete '
                            'category coverage (partial-run safety net).',
                            domain, split,
                            completion_summary['images_dropped'],
                            completion_summary['images_total'],
                        )

                # The filtered GT JSON content is mode-invariant; write
                # without a mode suffix so a single file is shared across
                # both modes' COCOeval calls.
                filtered_gt_path = get_intermediate_file_path(
                    eval_file, f'_odinw_gt_{domain}_{split}', target_format='json'
                )
                write_json(filtered_gt, filtered_gt_path)

                logger.info(
                    'ODinW [%s/%s] mode=%s: %d rows → %d predictions, GT %d images / %d ann.',
                    domain, split, score_mode, len(rows), len(preds),
                    len(filtered_gt['images']), len(filtered_gt['annotations']),
                )

                metrics, evaluator_obj = _evaluate_coco_bbox(filtered_gt_path, preds)
                per_domain[f'{domain}/{split}'] = {
                    **metrics,
                    'num_predictions': len(preds),
                    'num_rows': len(rows),
                    'num_kept_pairs': len(kept_pairs),
                    'num_images': len(filtered_gt['images']),
                    'num_annotations': len(filtered_gt['annotations']),
                    'images_total': completion_summary['images_total'],
                    'images_completed': completion_summary['images_completed'],
                    'images_dropped': completion_summary['images_dropped'],
                }
                domain_splits_evaluated.append(f'{domain}/{split}')

                # Per-row instance diagnostics for this (domain, split).
                # Collapse (domain, split, image_id, category_id) into a
                # single ``trajectory_file`` matching prepare_odinw's
                # row index format: ``<domain>_<split>_<image_id>_<category_id>.json``.
                # That string is the basename used by full_trajectories/
                # and raw_traces/, so users can cross-reference
                # straight from this CSV to the agent's per-row JSON.
                if evaluator_obj is not None:
                    domain_per_row = extract_per_row_diagnostics(
                        evaluator_obj, iou_threshold=0.5, area_label='all',
                    )
                    # Index basename layout depends on the TSV: per-class
                    # rows store one trajectory per (image, category); qwen3
                    # multi-label rows store one trajectory per image (the
                    # same trajectory is referenced by every per-class
                    # diagnostic row for that image).
                    qwen3_layout = (eval_scope == 'qwen3_multilabel')
                    for r in domain_per_row:
                        if qwen3_layout:
                            basename = f"{domain}_{split}_{r['image_id']}.json"
                        else:
                            basename = (
                                f"{domain}_{split}_{r['image_id']}_{r['category_id']}.json"
                            )
                        r['trajectory_file'] = resolve_trajectory_path(
                            eval_file, basename
                        )
                        r.pop('image_id', None)
                        r.pop('category_id', None)
                    all_per_row.extend(domain_per_row)

            # 13-domain macro-mean. Each TSV holds exactly one split, so
            # ``per_domain`` already contains exactly one entry per domain
            # (keyed ``<domain>/<split>``). Skip -1 ("no data") entries so
            # APs/APm/APl aren't pulled down by domains with no small/
            # medium/large boxes.
            metric_keys = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl']

            def _mean_skip_missing(vals):
                valid = [float(v) for v in vals if v is not None and float(v) >= 0]
                if not valid:
                    return -1.0
                return float(np.mean(valid))

            if per_domain:
                avg = {
                    k: _mean_skip_missing([d[k] for d in per_domain.values()])
                    for k in metric_keys
                }
            else:
                avg = {k: 0.0 for k in metric_keys}

            avg_label = 'ODinW13_average'

            # Phase 2: emit sidecar diagnostics if we used real SAM3 scores.
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

            # Save full metrics JSON (per mode).
            metrics_blob = {
                'dataset': self.dataset_name,
                'evaluator': 'pycocotools.COCOeval',
                'iou_type': 'bbox',
                'score_mode': score_mode,
                'eval_scope': eval_scope,
                'num_predictions_total': len(all_preds),
                'domain_splits_evaluated': domain_splits_evaluated,
                'average_split': (
                    domain_splits_evaluated[0].split('/', 1)[1]
                    if domain_splits_evaluated else None
                ),
                'per_domain': per_domain,
                avg_label: avg,
            }
            if sidecar_diag is not None:
                metrics_blob['sidecar'] = sidecar_diag
            metrics_file = get_intermediate_file_path(
                eval_file, f'_metrics{file_suffix}', target_format='json'
            )
            dump(metrics_blob, metrics_file)

            # Combined predictions JSON for offline re-runs (per mode).
            combined_pred_path = get_intermediate_file_path(
                eval_file, f'_odinw_pred{file_suffix}', target_format='json'
            )
            write_json(all_preds, combined_pred_path)

            # Combined per-row instance-level diagnostics CSV (per mode).
            # One row per (domain, split, image_id, category_id) the
            # evaluator considered, with TP/FP/FN counts at IoU=0.5.
            # The identity is encoded as ``trajectory_file`` (matches
            # the row's TSV index + ``.json``).
            if all_per_row:
                # Sort silently by F1 desc (mean_match_iou desc as tiebreak)
                # so the highest-scoring rows surface to the top of the CSV.
                from .eval_bridge import _sort_grounding_per_row
                all_per_row = _sort_grounding_per_row(all_per_row)
                per_row_df = pd.DataFrame(all_per_row)
                preferred_cols = [
                    'trajectory_file',
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
                    '[ODinW %s] wrote %d-row per-instance CSV to %s',
                    score_mode, len(per_row_df), per_row_path,
                )

            # Slim summary DataFrame (per mode).
            def _pct(v):
                # Inputs are already on the 0-100 percentage scale (from
                # _evaluate_coco_bbox above). Just round and handle N/A.
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    return 'N/A'
                if fv < 0:
                    return 'N/A'
                return round(fv, 2)

            rows_out: List[Dict] = []
            for d in sorted(per_domain.keys()):
                m = per_domain[d]
                rows_out.append({
                    'Domain': d,
                    'score_mode': score_mode,
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
                'score_mode': score_mode,
                'AP':   _pct(avg['AP']),
                'AP50': _pct(avg['AP50']),
                'AP75': _pct(avg['AP75']),
                'APs':  _pct(avg['APs']),
                'APm':  _pct(avg['APm']),
                'APl':  _pct(avg['APl']),
                'Samples': sum(per_domain[d]['num_rows'] for d in per_domain) if per_domain else 0,
            })
            summary_df = pd.DataFrame(rows_out)
            score_file = get_intermediate_file_path(eval_file, f'_acc{file_suffix}')
            dump(summary_df, score_file)

            logger.info('[ODinW %s] complete: %s = %s', score_mode, avg_label, avg)

            if is_primary:
                primary_summary = summary_df

        return primary_summary
