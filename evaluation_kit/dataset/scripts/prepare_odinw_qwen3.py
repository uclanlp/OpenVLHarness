#!/usr/bin/env python3
"""Prepare ODinW-13 TSVs that follow the Qwen3-VL evaluation protocol.

This is the sibling of ``prepare_odinw.py`` (full-detection, one row per
(image, category)) and ``prepare_odinw_qwen25.py`` (per-image class
filtering, one row per (image, present category)). The Qwen3-VL protocol
is the all-categories-at-once variant shipped with
``Qwen3-VL/evaluation/ODinW-13/`` and described in QwenLM/Qwen3-VL issue
#1904:

  * **One row per image.** All dataset classes are joined into a single
    prompt; the model emits a JSON list of ``{bbox_2d, label}`` records
    covering multiple classes in one response.

  * **Prompt template** (verbatim from the shipped ``dataset_utils.py``):

        Locate every instance that belongs to the following categories:
        '<class_a>, <class_b>, ...'. Report bbox coordinates in JSON format.

  * **No caption_prompt rewrites by default** — the shipped Qwen3-VL eval
    feeds raw MMDet ``class_name`` tuples. Use ``--class-names mmdet`` to
    apply the rewrites if you want an A/B (e.g. ``Cottontail-Rabbit ->
    rabbit``).

Output filenames:

  ODinW13_bbox_<split>_qwen3.tsv          (default: raw class names)
  ODinW13_bbox_<split>_qwen3_mmdet.tsv    (--class-names mmdet)

Schema differs from the other two scripts because each row covers
multiple classes:

  index           "<domain>_<split>_<image_id>"   (no _<cat_id> suffix)
  image_path      "<domain>/<filename>"
  question        the all-classes-joined prompt
  answer          JSON list of per-annotation records:
                  [{"category_id":N, "category_name":"X",
                    "bbox_norm1k":[x1,y1,x2,y2],
                    "annotation_id":A, "area":F}, ...]
  category_id     "" (multi-class row; not a single id)
  category_name   "" (multi-class row; not a single name)
  image_width, image_height
  num_instances   total GT box count for this image
  annotation_ids  comma-separated all COCO ann ids
  domain, split
  areas           JSON list of all areas (parallel to flattened GT)
  dataset_classes JSON list of {"id":N, "name":"X"} for ALL classes in
                  this dataset; the prompted vocabulary. Used by eval to
                  resolve model output labels back to category_ids.

The Qwen3-VL model emits 0-1000 normalised bbox_2d coordinates
(empirically verified — see explanations/server/odinw/
qwen3vl_8b_odinw13_reproduction.md §5). The eval-side coords_space
override in ODinWBBoxDataset takes care of routing.
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd

from evaluation_kit.dataset.scripts.prepare_odinw import (
    ODINW_13_CONFIGS,
    bbox_xywh_to_normalised_xyxy,
    load_all_configs,
)
from evaluation_kit.dataset.scripts.prepare_odinw_qwen25 import (
    CAPTION_PROMPT_BY_DOMAIN,
    MMDET_CLASS_NAMES,
    _apply_caption_prompt,
    _mmdet_class_name,
    _validate_against_coco,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


def build_qwen3_question(class_names_joined: str) -> str:
    """Qwen3-VL ODinW evaluation prompt (verbatim from issue #1904 /
    ``Qwen3-VL/evaluation/ODinW-13/dataset_utils.py``).
    """
    return (
        f"Locate every instance that belongs to the following categories: "
        f"'{class_names_joined}'. Report bbox coordinates in JSON format."
    )


def _resolve_display_classes(
    domain_name: str,
    coco_categories: List[dict],
    class_names_mode: str,
) -> Tuple[List[dict], str]:
    """Build the per-row dataset_classes list and the prompt's class string.

    Returns
    -------
    classes_for_eval : list of dict
        Records ``[{"id": int, "name": str}, ...]`` covering ALL dataset
        classes (sorted by id). ``name`` is the display name as fed into
        the prompt (rewritten if class_names_mode=='mmdet').
    classes_joined : str
        ``", ".join([c["name"] for c in classes_for_eval])`` — drops
        straight into the prompt template.
    """
    cap_for_domain = CAPTION_PROMPT_BY_DOMAIN.get(domain_name)
    coco_name_by_id = {int(c['id']): c['name'] for c in coco_categories}
    sorted_ids = sorted(coco_name_by_id.keys())

    classes_for_eval: List[dict] = []
    for cat_id in sorted_ids:
        raw_name = _mmdet_class_name(domain_name, cat_id, coco_name_by_id[cat_id])
        if class_names_mode == 'mmdet':
            display_name = _apply_caption_prompt(raw_name, cap_for_domain)
        elif class_names_mode == 'raw':
            display_name = raw_name
        else:
            raise ValueError(f'Unknown class_names_mode: {class_names_mode!r}')
        classes_for_eval.append({'id': cat_id, 'name': display_name})

    classes_joined = ', '.join(c['name'] for c in classes_for_eval)
    return classes_for_eval, classes_joined


def process_domain_split_qwen3(
    domain_name: str,
    split_key: str,
    split_info: dict,
    class_names_mode: str,
) -> List[dict]:
    """Emit one row per image (multi-class prompt) for one (domain, split)."""
    ann_file = split_info['ann_file']

    with open(ann_file) as f:
        coco = json.load(f)

    images_by_id = {img['id']: img for img in coco.get('images', [])}
    coco_categories = list(coco.get('categories', []))
    _validate_against_coco(domain_name, coco_categories)

    classes_for_eval, classes_joined = _resolve_display_classes(
        domain_name, coco_categories, class_names_mode,
    )
    # id -> display name (rewritten if mmdet mode) for filling answer records
    display_name_by_id = {c['id']: c['name'] for c in classes_for_eval}

    # Group annotations by image_id; within each, by category for stable order.
    anns_by_image: Dict[int, List[dict]] = defaultdict(list)
    for ann in coco.get('annotations', []):
        anns_by_image[int(ann['image_id'])].append(ann)

    dataset_classes_json = json.dumps(classes_for_eval)
    question = build_qwen3_question(classes_joined)

    rows: List[dict] = []
    for image_id in sorted(images_by_id.keys()):
        img = images_by_id[image_id]
        img_w = img['width']
        img_h = img['height']
        image_path = f'{domain_name}/{img["file_name"]}'

        anns = sorted(
            anns_by_image.get(image_id, []),
            key=lambda a: (int(a['category_id']), int(a['id'])),
        )
        records: List[dict] = []
        all_areas: List[float] = []
        all_ann_ids: List[str] = []
        for a in anns:
            cat_id = int(a['category_id'])
            cat_name = display_name_by_id.get(cat_id, f'cat_{cat_id}')
            bbox_norm = bbox_xywh_to_normalised_xyxy(a['bbox'], img_w, img_h)
            area = round(a.get('area', 0), 2)
            ann_id = a['id']
            records.append({
                'category_id': cat_id,
                'category_name': cat_name,
                'bbox_norm1k': bbox_norm,
                'annotation_id': ann_id,
                'area': area,
            })
            all_areas.append(area)
            all_ann_ids.append(str(ann_id))

        rows.append({
            'index': f'{domain_name}_{split_key}_{image_id}',
            'image_path': image_path,
            'question': question,
            'answer': json.dumps(records),
            # Multi-class row: leave the single-class columns empty.
            'category_id': '',
            'category_name': '',
            'image_width': img_w,
            'image_height': img_h,
            'num_instances': len(records),
            'annotation_ids': ','.join(all_ann_ids),
            'domain': domain_name,
            'split': split_key,
            'areas': json.dumps(all_areas),
            'dataset_classes': dataset_classes_json,
        })
    return rows


def generate_split_tsv(
    all_domains: Dict[str, dict],
    output_dir: str,
    benchmark_label: str,
    split_key: str,
    class_names_mode: str,
) -> Optional[str]:
    """Emit one Qwen3-VL TSV for one split."""
    domain_order = {name: i for i, name in enumerate(ODINW_13_CONFIGS)}
    all_rows: List[dict] = []

    for domain_name in ODINW_13_CONFIGS:
        if domain_name not in all_domains:
            log.warning(f'  [{split_key}] domain {domain_name!r} not found; skipping')
            continue
        splits_dict = all_domains[domain_name]
        if split_key not in splits_dict:
            log.warning(f'  [{split_key}] {domain_name}: split not in YAML, skipping')
            continue
        log.info(f'  Processing {domain_name}/{split_key} (one row per image)...')
        rows = process_domain_split_qwen3(
            domain_name, split_key, splits_dict[split_key],
            class_names_mode=class_names_mode,
        )
        log.info(f'    -> {len(rows)} images')
        all_rows.extend(rows)

    if not all_rows:
        log.warning(f'  [{split_key}] No rows emitted; nothing to write.')
        return None

    def _sort_key(r):
        return (
            domain_order.get(r['domain'], 9999),
            int(r['index'].rsplit('_', 1)[-1]),  # image_id
        )
    all_rows.sort(key=_sort_key)

    # Filename: _qwen3 always present; _mmdet appended when class rewrites
    # were applied (the non-default).
    suffix_parts = ['qwen3']
    if class_names_mode == 'mmdet':
        suffix_parts.append('mmdet')
    suffix = '_'.join(suffix_parts)
    out_filename = f'{benchmark_label}_bbox_{split_key}_{suffix}.tsv'
    out_path = os.path.join(output_dir, out_filename)
    df = pd.DataFrame(all_rows)
    df.to_csv(out_path, sep='\t', index=False)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    log.info(f'  Wrote {out_path} ({len(df)} rows, {size_mb:.1f} MB)')
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description='Prepare ODinW-13 TSVs under the Qwen3-VL evaluation protocol '
                    '(one row per image, all classes joined in the prompt).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--glip-dir', required=True,
                        help='Path to the GLIP repository root.')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for the TSVs (default: $LMUData).')
    parser.add_argument(
        '--splits', nargs='+', default=['val', 'test'],
        choices=['train', 'val', 'test'],
        help='COCO splits to emit (one TSV per split). Default: val test.',
    )
    parser.add_argument(
        '--class-names', dest='class_names_mode', default='raw',
        choices=['raw', 'mmdet'],
        help='Class-name treatment. "raw" keeps the raw MMDet class_name '
             'tuple (matches the shipped Qwen3-VL eval); "mmdet" applies '
             'MMDet\'s caption_prompt rewrites. Default: raw.',
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.environ.get('LMUData', None)
    if output_dir is None:
        log.error('Either --output-dir or $LMUData must be set.')
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)

    glip_dir = args.glip_dir
    if not os.path.isdir(glip_dir):
        log.error(f'GLIP directory not found: {glip_dir}')
        sys.exit(1)

    log.info('=' * 60)
    log.info('Step 1: Parsing GLIP configs (ODinW-13)')
    log.info('=' * 60)
    domains_13 = load_all_configs(glip_dir, ODINW_13_CONFIGS, 'ODinW-13')

    log.info('=' * 60)
    log.info(f'Step 2: Generating Qwen3 TSVs (class-names={args.class_names_mode})')
    log.info('=' * 60)
    written: List[str] = []
    for split_key in args.splits:
        out_path = generate_split_tsv(
            domains_13, output_dir, 'ODinW13', split_key,
            class_names_mode=args.class_names_mode,
        )
        if out_path:
            written.append(out_path)

    log.info('=' * 60)
    log.info('Done!')
    for p in written:
        log.info(f'  TSV: {p}')


if __name__ == '__main__':
    main()
