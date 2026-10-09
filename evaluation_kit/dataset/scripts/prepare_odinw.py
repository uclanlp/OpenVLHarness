#!/usr/bin/env python3
"""Prepare ODinW-13 full-detection TSVs for mini-vlm-toolkit evaluation.

For each requested split (val / test / train), emits one row per
(image, category) pair within each ODinW-13 domain — positive if any
annotations exist, negative (``answer="[]"``) otherwise.  That matches
the GLIP / Grounding DINO reporting convention (every image queried
against every category in its domain).

Usage:

    python -m evaluation_kit.dataset.scripts.prepare_odinw \\
        --glip-dir /path/to/GLIP \\
        [--output-dir $LMUData] \\
        [--splits val test]

Row ordering:
  Rows are sorted by (domain_index, image_id, category_id) so all rows for
  one (domain, image_id) are contiguous — preserves atomic-image semantics
  for interrupted runs.

Output (one TSV per split):
  ${output_dir}/ODinW13_bbox_<split>.tsv

Schema:
  index, image_path, question, answer, category_id, category_name,
  image_width, image_height, num_instances, annotation_ids, domain, split, areas
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


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


def parse_glip_config(yaml_path: str, dataset_root: str):
    """Parse a GLIP ODinW config YAML; return (domain_name, splits_dict)."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    domain_name = os.path.splitext(os.path.basename(yaml_path))[0]
    register = cfg.get('DATASETS', {}).get('REGISTER', {})

    override_cats = None
    override_str = cfg.get('DATASETS', {}).get('OVERRIDE_CATEGORY', '')
    if override_str:
        try:
            override_cats = json.loads(override_str)
        except (json.JSONDecodeError, TypeError):
            pass

    splits: Dict[str, dict] = {}
    for split_key, split_info in register.items():
        if split_key not in ('train', 'val', 'test'):
            continue
        ann_abs = os.path.join(dataset_root, split_info.get('ann_file', ''))
        if not os.path.isfile(ann_abs):
            log.warning(f'  [{domain_name}] Annotation file not found: {ann_abs}')
            continue
        splits[split_key] = {
            'ann_file': ann_abs,
            'categories': override_cats,
        }
    return domain_name, splits


def load_all_configs(glip_dir: str, config_names: List[str], benchmark_label: str):
    """Load and parse all configs for a benchmark. Returns domain→splits dict."""
    config_dir_13 = os.path.join(glip_dir, 'configs', 'odinw_13')
    config_dir_35 = os.path.join(glip_dir, 'configs', 'odinw_35')
    dataset_root = os.path.join(glip_dir, 'DATASET')

    all_domains: Dict[str, dict] = {}
    for cfg_name in config_names:
        yaml_path = os.path.join(config_dir_35, f'{cfg_name}.yaml')
        if not os.path.isfile(yaml_path):
            yaml_path = os.path.join(config_dir_13, f'{cfg_name}.yaml')
        if not os.path.isfile(yaml_path):
            log.warning(f'Config not found for {cfg_name}, skipping.')
            continue
        domain_name, splits = parse_glip_config(yaml_path, dataset_root)
        if splits:
            all_domains[domain_name] = splits
            log.info(f'  [{benchmark_label}] {domain_name}: {list(splits.keys())}')
        else:
            log.warning(f'  [{benchmark_label}] {domain_name}: no valid splits found')
    return all_domains


def bbox_xywh_to_normalised_xyxy(bbox, img_w, img_h):
    """Convert COCO bbox [x, y, w, h] (abs pixels) → [x1, y1, x2, y2] on 0-1000."""
    x, y, w, h = bbox
    return [
        round(x / img_w * 1000.0, 2),
        round(y / img_h * 1000.0, 2),
        round((x + w) / img_w * 1000.0, 2),
        round((y + h) / img_h * 1000.0, 2),
    ]


def build_bbox_question(category_name: str) -> str:
    return (
        f"Locate all instances of '{category_name}' in this image. "
        f"For each instance, output a bounding box as [x1, y1, x2, y2] "
        f"where coordinates are normalised on a 0-1000 scale "
        f"(with (0,0) as the top-left corner and (1000,1000) as the bottom-right corner). "
        f"Your answer should be a list of lists. "
        f"If there are no instances of '{category_name}' in the image, "
        f"output an empty list [] as your answer."
    )


def process_domain_split_full(
    domain_name: str,
    split_key: str,
    split_info: dict,
) -> List[dict]:
    """Emit full-detection rows for one (domain, split): every (image, category)."""
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

    groups: Dict[tuple, list] = defaultdict(list)
    for ann in coco.get('annotations', []):
        groups[(ann['image_id'], ann['category_id'])].append(ann)

    rows: List[dict] = []
    for image_id in sorted(images_by_id.keys()):
        img = images_by_id[image_id]
        img_w = img['width']
        img_h = img['height']
        image_path = f'{domain_name}/{img["file_name"]}'

        for cat in categories:
            cat_id = int(cat['id'])
            cat_name = cat['name']
            anns = groups.get((image_id, cat_id), [])

            if anns:
                norm_bboxes = [
                    bbox_xywh_to_normalised_xyxy(a['bbox'], img_w, img_h)
                    for a in anns
                ]
                answer_str = json.dumps(norm_bboxes)
                ann_ids_str = ','.join(str(a['id']) for a in anns)
                areas_str = json.dumps([round(a.get('area', 0), 2) for a in anns])
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
    return rows


def generate_split_tsv(
    all_domains: Dict[str, dict],
    output_dir: str,
    benchmark_label: str,
    split_key: str,
) -> Optional[str]:
    """Emit a single full-detection TSV for one split.

    Filename: ``<benchmark>_bbox_<split>.tsv``
    """
    domain_order = {name: i for i, name in enumerate(ODINW_13_CONFIGS)}
    all_rows: List[dict] = []

    for domain_name in ODINW_13_CONFIGS:
        if domain_name not in all_domains:
            log.warning(f'  [{benchmark_label}/{split_key}] domain {domain_name!r} not found; skipping')
            continue
        splits_dict = all_domains[domain_name]
        if split_key not in splits_dict:
            log.warning(
                f'  [{benchmark_label}/{split_key}] {domain_name}: split not in YAML, skipping'
            )
            continue
        log.info(f'  Processing {domain_name}/{split_key}...')
        rows = process_domain_split_full(domain_name, split_key, splits_dict[split_key])
        log.info(f'    → {len(rows)} rows')
        all_rows.extend(rows)

    if not all_rows:
        log.warning(f'  [{benchmark_label}/{split_key}] No rows emitted; nothing to write.')
        return None

    def _sort_key(r):
        return (
            domain_order.get(r['domain'], 9999),
            int(r['index'].rsplit('_', 2)[-2]),
            int(r['category_id']),
        )
    all_rows.sort(key=_sort_key)

    out_filename = f'{benchmark_label}_bbox_{split_key}.tsv'
    out_path = os.path.join(output_dir, out_filename)
    df = pd.DataFrame(all_rows)
    df.to_csv(out_path, sep='\t', index=False)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    log.info(f'  Wrote {out_path} ({len(df)} rows, {size_mb:.1f} MB)')
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description='Prepare ODinW-13 full-detection TSVs (one row per (image, category) per domain).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--glip-dir', required=True,
                        help='Path to the GLIP repository root.')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for the TSVs (default: $LMUData).')
    parser.add_argument(
        '--splits', nargs='+', default=['val', 'test'],
        choices=['train', 'val', 'test'],
        help='Which COCO splits from the GLIP YAML REGISTER to emit. One TSV per split. '
             'Default: val test.',
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
    log.info('Step 2: Generating full-detection TSVs')
    log.info('=' * 60)
    written: List[str] = []
    for split_key in args.splits:
        out_path = generate_split_tsv(domains_13, output_dir, 'ODinW13', split_key)
        if out_path:
            written.append(out_path)

    log.info('=' * 60)
    log.info('Done!')
    for p in written:
        log.info(f'  TSV: {p}')


if __name__ == '__main__':
    main()
