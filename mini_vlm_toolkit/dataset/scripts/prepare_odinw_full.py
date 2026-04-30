#!/usr/bin/env python3
"""Prepare the ODinW-13 *full-detection* TSV for mini-vlm-toolkit evaluation.

This is the Phase-1.5 sibling of ``prepare_odinw.py``.  Where the original
script emits one row per **positive** (image, category) annotation group,
this script emits one row per (image, category) **pair** in the domain —
including negative pairs (no annotations) which get ``answer="[]"``.

That gives a TSV that can be evaluated as standard ODinW-13 detection
(every image queried against every category in its domain), matching the
GLIP / Grounding DINO reporting convention.

Usage:

    python -m mini_vlm_toolkit.dataset.scripts.prepare_odinw_full \\
        --glip-dir /path/to/GLIP \\
        [--output-dir $LMUData] \\
        [--num-images-per-domain N] \\
        [--image-ids-file path/to/subset.json] \\
        [--splits test val]

Subsetting:
  --num-images-per-domain N   keep only the first N image_ids per domain
                              (sorted ascending, deterministic).  Mathematically
                              the right strategy for ODinW-13's macro-mean
                              over domains: each per-domain AP is unbiased,
                              so the outer mean is unbiased.
  --image-ids-file PATH       JSON: {"<domain_name>": [image_id, ...]} for
                              explicit reproducible subsets.

Row ordering:
  Rows are sorted by (domain_index_in_ODINW_13_CONFIGS, image_id, category_id)
  so all rows for one (domain, image_id) are contiguous.  This is what makes
  ``--num-examples`` slicing or interrupted runs leave whole images either
  fully complete or fully absent — atomic-image semantics.

Output:
  ${output_dir}/ODinW13_bbox_val_full.tsv

Schema (same as prepare_odinw.py output):
  index, image_path, question, answer, category_id, category_name,
  image_width, image_height, num_instances, annotation_ids, domain, split, areas

Note: ``--shuffle`` and ``--num-examples`` on ``run_benchmark.py`` are NOT
recommended on the full TSV — they break atomic-image semantics.  Use
``--num-images-per-domain`` here instead.  The eval-time completeness
filter (in ``mini_vlm_toolkit.dataset.eval_bridge``) is a safety net for
mid-run kills, not a sampler.
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional

import pandas as pd

# Reuse helpers from the original script.
from mini_vlm_toolkit.dataset.scripts.prepare_odinw import (
    ODINW_13_CONFIGS,
    bbox_xywh_to_normalised_xyxy,
    build_bbox_question,
    load_all_configs,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


def _select_image_ids(
    domain_name: str,
    coco_image_ids_sorted: List[int],
    num_images_per_domain: Optional[int],
    image_ids_file_data: Optional[Dict[str, List[int]]],
) -> List[int]:
    """Resolve which image_ids to emit for one domain (in order)."""
    if image_ids_file_data and domain_name in image_ids_file_data:
        wanted = list(image_ids_file_data[domain_name])
        # Intersect with what's actually in the JSON, preserving the
        # caller's requested order.
        valid = set(coco_image_ids_sorted)
        return [iid for iid in wanted if iid in valid]
    if num_images_per_domain is not None:
        return coco_image_ids_sorted[:num_images_per_domain]
    return coco_image_ids_sorted


def process_domain_split_full(
    domain_name: str,
    split_key: str,
    split_info: dict,
    num_images_per_domain: Optional[int],
    image_ids_file_data: Optional[Dict[str, List[int]]],
) -> List[dict]:
    """Emit full-detection rows for one (domain, split).

    Every (chosen image, category) pair becomes a row — positive if any
    annotations exist, negative otherwise.  Rows sorted by (image_id,
    category_id) within the domain.
    """
    ann_file = split_info['ann_file']
    override_cats = split_info.get('categories')

    with open(ann_file) as f:
        coco = json.load(f)

    images_by_id = {img['id']: img for img in coco.get('images', [])}

    if override_cats:
        categories = list(override_cats)
    else:
        categories = list(coco.get('categories', []))
    categories = sorted(categories, key=lambda c: int(c['id']))

    # Group annotations by (image_id, category_id) — same key as
    # prepare_odinw.process_domain_split, lets us look up positives in O(1).
    from collections import defaultdict
    groups: Dict[tuple, list] = defaultdict(list)
    for ann in coco.get('annotations', []):
        groups[(ann['image_id'], ann['category_id'])].append(ann)

    image_ids_sorted = sorted(images_by_id.keys())
    chosen_image_ids = _select_image_ids(
        domain_name, image_ids_sorted, num_images_per_domain, image_ids_file_data,
    )

    rows: List[dict] = []
    for image_id in chosen_image_ids:
        img = images_by_id[image_id]
        img_w = img['width']
        img_h = img['height']
        img_filename = img['file_name']
        image_path = f'{domain_name}/{img_filename}'

        for cat in categories:
            cat_id = int(cat['id'])
            cat_name = cat['name']
            anns = groups.get((image_id, cat_id), [])

            if anns:
                norm_bboxes = [
                    bbox_xywh_to_normalised_xyxy(a['bbox'], img_w, img_h)
                    for a in anns
                ]
                ann_ids = [a['id'] for a in anns]
                areas = [round(a.get('area', 0), 2) for a in anns]
                answer_str = json.dumps(norm_bboxes)
                ann_ids_str = ','.join(str(a) for a in ann_ids)
                areas_str = json.dumps(areas)
                num_instances = len(anns)
            else:
                answer_str = '[]'
                ann_ids_str = ''
                areas_str = '[]'
                num_instances = 0

            rows.append({
                'index': f'{domain_name}_{split_key}_{image_id}_{cat_id}',
                'image_path': image_path,
                'question': build_bbox_question(cat_name),
                'answer': answer_str,
                'category_id': cat_id,
                'category_name': cat_name,
                'image_width': img_w,
                'image_height': img_h,
                'num_instances': num_instances,
                'annotation_ids': ann_ids_str,
                'domain': domain_name,
                'split': split_key,
                'areas': areas_str,
            })

    # rows are already in (image_id, category_id) order because we iterate
    # chosen_image_ids ascending and categories sorted by id.
    return rows


def generate_full_tsv(
    all_domains: Dict[str, dict],
    output_dir: str,
    benchmark_label: str,
    splits: List[str],
    num_images_per_domain: Optional[int],
    image_ids_file_data: Optional[Dict[str, List[int]]],
) -> Optional[str]:
    """Emit one full-detection TSV per benchmark, e.g. ODinW13_bbox_val_full.tsv.

    Bundles ``splits`` rows into the val-side TSV (matching prepare_odinw.py
    convention where val and test both go into _val.tsv).  Returns the
    written path or None if nothing was emitted.
    """
    domain_order = {name: i for i, name in enumerate(ODINW_13_CONFIGS)}
    all_rows: List[dict] = []

    for domain_name in ODINW_13_CONFIGS:
        if domain_name not in all_domains:
            log.warning(f'  [{benchmark_label}-full] domain {domain_name!r} not found; skipping')
            continue
        splits_dict = all_domains[domain_name]
        for split_key in splits:
            if split_key not in splits_dict:
                log.warning(
                    f'  [{benchmark_label}-full] {domain_name}: split {split_key!r} not in YAML, skipping'
                )
                continue
            split_info = splits_dict[split_key]
            log.info(f'  Processing {domain_name}/{split_key} (full)...')
            rows = process_domain_split_full(
                domain_name, split_key, split_info,
                num_images_per_domain, image_ids_file_data,
            )
            log.info(f'    → {len(rows)} rows')
            all_rows.extend(rows)

    if not all_rows:
        log.warning(f'  [{benchmark_label}-full] No rows emitted; nothing to write.')
        return None

    # Final sort: domain_index in ODINW_13_CONFIGS, then image_id, then cat_id.
    # image_id is recovered from the index column ("<domain>_<split>_<id>_<cat>").
    def _sort_key(r):
        return (
            domain_order.get(r['domain'], 9999),
            int(r['index'].rsplit('_', 2)[-2]),
            int(r['category_id']),
        )
    all_rows.sort(key=_sort_key)

    # Output filename:
    #   <benchmark>_bbox_<split_label>_full<size_suffix>.tsv
    # split_label: 'train' if --splits == ['train'] only, else 'val' (the val
    #              TSV bundles val + test rows by Phase 1 convention).
    # size_suffix: '_n<N>' when --num-images-per-domain is set, else '_all'.
    if list(splits) == ['train']:
        split_label = 'train'
    else:
        split_label = 'val'
    size_suffix = f'_n{num_images_per_domain}' if num_images_per_domain else '_all'
    out_filename = f'{benchmark_label}_bbox_{split_label}_full{size_suffix}.tsv'
    out_path = os.path.join(output_dir, out_filename)
    df = pd.DataFrame(all_rows)
    df.to_csv(out_path, sep='\t', index=False)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    log.info(f'  Wrote {out_path} ({len(df)} rows, {size_mb:.1f} MB)')
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description='Prepare ODinW-13 full-detection TSV (one row per (image, category) per domain).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--glip-dir', required=True,
                        help='Path to the GLIP repository root.')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for the TSV (default: $LMUData).')
    parser.add_argument(
        '--num-images-per-domain', type=int, default=None,
        help='Per-domain image cap. Each domain emits min(N, |images_d|) × k_d rows. '
             'Default: no cap (all images).',
    )
    parser.add_argument(
        '--image-ids-file', type=str, default=None,
        help='Optional JSON path of explicit per-domain image_ids: '
             '{"<domain_name>": [image_id, ...]}. Overrides --num-images-per-domain.',
    )
    parser.add_argument(
        '--splits', nargs='+', default=['test'],
        help='Which COCO splits from the GLIP YAML REGISTER to include. '
             'Default: test (matches the ODinW-13 reporting convention).',
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

    image_ids_file_data: Optional[Dict[str, List[int]]] = None
    if args.image_ids_file:
        if not os.path.isfile(args.image_ids_file):
            log.error(f'--image-ids-file not found: {args.image_ids_file}')
            sys.exit(1)
        with open(args.image_ids_file) as f:
            image_ids_file_data = json.load(f)
        log.info(f'Loaded image-ids subset for {len(image_ids_file_data)} domains.')

    log.info('=' * 60)
    log.info('Step 1: Parsing GLIP configs (ODinW-13)')
    log.info('=' * 60)
    domains_13 = load_all_configs(glip_dir, ODINW_13_CONFIGS, 'ODinW-13')

    log.info('=' * 60)
    log.info('Step 2: Generating full-detection TSV')
    log.info('=' * 60)
    out_path = generate_full_tsv(
        domains_13, output_dir, 'ODinW13', args.splits,
        args.num_images_per_domain, image_ids_file_data,
    )

    log.info('=' * 60)
    log.info('Done!')
    if out_path:
        log.info(f'  TSV: {out_path}')
        log.info(f'  Run with: python run_benchmark.py --data ODinW13_bbox_val_full ...')
        log.info('  Note: do NOT use --shuffle or --num-examples; subset at prepare time.')
        log.info('  Recommend --save-html 0 (one HTML per row × many rows = many files).')


if __name__ == '__main__':
    main()
