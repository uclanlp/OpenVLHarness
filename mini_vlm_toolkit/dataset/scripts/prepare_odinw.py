#!/usr/bin/env python3
"""Prepare ODinW-13 and ODinW-35 datasets for mini-vlm-toolkit evaluation.

This script reads the GLIP config YAMLs to discover domain→variant→split→path
mappings, then converts COCO-format annotations into the TSV format expected by
mini-vlm-toolkit.

Usage (run in tmux — image moving can take a while on slow disks):

    python -m mini_vlm_toolkit.dataset.scripts.prepare_odinw \\
        --glip-dir /path/to/GLIP \\
        [--output-dir $LMUData] \\
        [--skip-image-move] \\
        [--benchmarks 13 35]

What this script does:

  1. Parses GLIP config YAMLs from configs/odinw_13/ and configs/odinw_35/
     to find the authoritative annotation file + image directory per domain+split.

  2. Moves images from each domain's source directories into
     ${LMUData}/images/ODinW/{domain_name}/.  Per-domain subdirs are used
     because filenames can collide across domains.  Images are MOVED, not
     copied, to save disk space.

  3. Generates TSV files (bbox evaluation only — ODinW has no mask annotations):
       - ODinW13_bbox_train.tsv, ODinW13_bbox_val.tsv
       - ODinW35_bbox_train.tsv, ODinW35_bbox_val.tsv

     Each row = one (image, category) pair within a domain.

TSV columns:
  index, image_path, question, answer, category_id, category_name,
  image_width, image_height, num_instances, annotation_ids, domain, split, areas

Resumability:
  Image moves are idempotent — already-existing destinations are skipped.
  TSV generation overwrites existing files.

Note: ODinW annotations are bounding-box only (the "segmentation" field is
always []).  No mask TSVs are generated.
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# ── ODinW-13 and ODinW-35 domain lists (from GLIP config filenames) ──────────
# These are the YAML config filenames (without .yaml) that define each benchmark.

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

# ODinW-35 uses all configs listed in configs/odinw_35/_all.json.
# We read that file at runtime; this constant is a fallback.
ODINW_35_CONFIGS = [
    'AerialMaritimeDrone_large',
    'AerialMaritimeDrone_tiled',
    'AmericanSignLanguageLetters_American_Sign_Language_Letters.v1-v1.coco',
    'Aquarium_Aquarium_Combined.v2-raw-1024.coco',
    'BCCD_BCCD.v3-raw.coco',
    'ChessPieces_Chess_Pieces.v23-raw.coco',
    'CottontailRabbits',
    'DroneControl_Drone_Control.v3-raw.coco',
    'EgoHands_generic',
    'EgoHands_specific',
    'HardHatWorkers_raw',
    'MaskWearing_raw',
    'MountainDewCommercial',
    'NorthAmericaMushrooms_North_American_Mushrooms.v1-416x416.coco',
    'OxfordPets_by-breed',
    'OxfordPets_by-species',
    'PKLot_640',
    'Packages_Raw',
    'PascalVOC',
    'Raccoon_Raccoon.v2-raw.coco',
    'ShellfishOpenImages_raw',
    'ThermalCheetah',
    'UnoCards_raw',
    'VehiclesOpenImages_416x416',
    'WildfireSmoke',
    'boggleBoards_416x416AutoOrient_export_',
    'brackishUnderwater_960x540',
    'dice_mediumColor_export',
    'openPoetryVision_512x512',
    'pistols_export',
    'plantdoc_416x416',
    'pothole',
    'selfdrivingCar_fixedLarge_export_',
    'thermalDogsAndPeople',
    'websiteScreenshots',
]


# ── Config parsing ───────────────────────────────────────────────────────────

def parse_glip_config(yaml_path, dataset_root):
    """Parse a GLIP ODinW config YAML and return split info.

    Parameters
    ----------
    yaml_path : str
        Path to the YAML config file.
    dataset_root : str
        Root directory where odinw/ lives (i.e. GLIP/DATASET/).

    Returns
    -------
    domain_name : str
        Config name used as the domain identifier.
    splits : dict
        Mapping split_name → {"ann_file": abs_path, "img_dir": abs_path,
                               "categories": list or None}
    """
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    domain_name = os.path.splitext(os.path.basename(yaml_path))[0]
    register = cfg.get('DATASETS', {}).get('REGISTER', {})

    # Parse optional category override from config
    override_cats = None
    override_str = cfg.get('DATASETS', {}).get('OVERRIDE_CATEGORY', '')
    if override_str:
        try:
            override_cats = json.loads(override_str)
        except (json.JSONDecodeError, TypeError):
            pass

    splits = {}
    for split_key, split_info in register.items():
        # We only want the main train/val/test splits, not fewshot variants
        if split_key not in ('train', 'val', 'test'):
            continue

        ann_file = split_info.get('ann_file', '')
        img_dir = split_info.get('img_dir', '')

        # Resolve relative paths against dataset_root
        ann_abs = os.path.join(dataset_root, ann_file)
        img_abs = os.path.join(dataset_root, img_dir)

        if not os.path.isfile(ann_abs):
            log.warning(f'  [{domain_name}] Annotation file not found: {ann_abs}')
            continue

        splits[split_key] = {
            'ann_file': ann_abs,
            'img_dir': img_abs,
            'categories': override_cats,
        }

    return domain_name, splits


def load_all_configs(glip_dir, config_names, benchmark_label):
    """Load and parse all configs for a benchmark (13 or 35).

    Returns dict: domain_name → splits dict.
    """
    config_dir_13 = os.path.join(glip_dir, 'configs', 'odinw_13')
    config_dir_35 = os.path.join(glip_dir, 'configs', 'odinw_35')
    dataset_root = os.path.join(glip_dir, 'DATASET')

    all_domains = {}
    for cfg_name in config_names:
        # Try odinw_35 dir first (superset), then odinw_13
        yaml_path = os.path.join(config_dir_35, f'{cfg_name}.yaml')
        if not os.path.isfile(yaml_path):
            yaml_path = os.path.join(config_dir_13, f'{cfg_name}.yaml')
        if not os.path.isfile(yaml_path):
            log.warning(f'Config not found for {cfg_name}, skipping.')
            continue

        domain_name, splits = parse_glip_config(yaml_path, dataset_root)
        if splits:
            all_domains[domain_name] = splits
            log.info(f'  [{benchmark_label}] {domain_name}: '
                     f'{list(splits.keys())}')
        else:
            log.warning(f'  [{benchmark_label}] {domain_name}: no valid splits found')

    return all_domains


# ── Image moving ─────────────────────────────────────────────────────────────

def move_domain_images(img_dir, dst_dir):
    """Move all image files from img_dir to dst_dir.

    Returns (moved, skipped, errors) counts.
    """
    if not os.path.isdir(img_dir):
        return 0, 0, 0

    os.makedirs(dst_dir, exist_ok=True)
    moved = skipped = errors = 0
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

    for fname in os.listdir(img_dir):
        src = os.path.join(img_dir, fname)
        if not os.path.isfile(src):
            continue
        if os.path.splitext(fname)[1].lower() not in image_exts:
            continue

        dst = os.path.join(dst_dir, fname)
        if os.path.exists(dst):
            skipped += 1
            continue

        try:
            shutil.move(src, dst)
            moved += 1
        except Exception as e:
            log.debug(f'Failed to move {src}: {e}')
            errors += 1

    return moved, skipped, errors


def move_all_images(all_domains, output_dir):
    """Move images for all domains into $LMUData/images/ODinW/{domain}/."""
    total_moved = total_skipped = total_errors = 0
    # Collect unique (domain, img_dir) pairs to avoid moving the same dir twice
    seen_dirs = set()

    for domain_name, splits in all_domains.items():
        for split_key, split_info in splits.items():
            img_dir = split_info['img_dir']
            if img_dir in seen_dirs:
                continue
            seen_dirs.add(img_dir)

            dst_dir = os.path.join(output_dir, 'images', 'ODinW', domain_name)
            m, s, e = move_domain_images(img_dir, dst_dir)
            if m > 0 or s > 0:
                log.info(f'  [{domain_name}/{split_key}] moved={m} skipped={s} errors={e}')
            total_moved += m
            total_skipped += s
            total_errors += e

    log.info(f'Image move total: moved={total_moved} skipped={total_skipped} errors={total_errors}')


# ── Annotation processing ────────────────────────────────────────────────────

def bbox_xywh_to_normalised_xyxy(bbox, img_w, img_h):
    """Convert COCO bbox [x, y, w, h] (abs pixels) → [x1, y1, x2, y2] on a 0-1000 scale."""
    x, y, w, h = bbox
    return [
        round(x / img_w * 1000.0, 2),
        round(y / img_h * 1000.0, 2),
        round((x + w) / img_w * 1000.0, 2),
        round((y + h) / img_h * 1000.0, 2),
    ]


def build_bbox_question(category_name):
    """Build the question prompt for bbox evaluation."""
    return (
        f"Locate all instances of '{category_name}' in this image. "
        f"For each instance, output a bounding box as [x1, y1, x2, y2] "
        f"where coordinates are normalised on a 0-1000 scale "
        f"(with (0,0) as the top-left corner and (1000,1000) as the bottom-right corner). "
        f"Output a JSON array of arrays."
    )


def process_domain_split(domain_name, split_key, split_info):
    """Process one domain+split and return TSV rows.

    Returns list of row dicts.
    """
    ann_file = split_info['ann_file']
    override_cats = split_info.get('categories')

    with open(ann_file) as f:
        coco = json.load(f)

    images_by_id = {img['id']: img for img in coco.get('images', [])}

    # Use override categories from GLIP config if available, else from JSON
    if override_cats:
        cats_by_id = {cat['id']: cat for cat in override_cats}
    else:
        cats_by_id = {cat['id']: cat for cat in coco.get('categories', [])}

    # Group annotations by (image_id, category_id)
    groups = defaultdict(list)
    for ann in coco.get('annotations', []):
        groups[(ann['image_id'], ann['category_id'])].append(ann)

    rows = []
    for (image_id, category_id), anns in groups.items():
        img = images_by_id.get(image_id)
        cat = cats_by_id.get(category_id)
        if img is None or cat is None:
            continue

        img_w = img['width']
        img_h = img['height']
        img_filename = img['file_name']
        cat_name = cat['name']

        norm_bboxes = []
        ann_ids = []
        areas = []
        for ann in anns:
            norm_bboxes.append(bbox_xywh_to_normalised_xyxy(ann['bbox'], img_w, img_h))
            ann_ids.append(ann['id'])
            areas.append(ann.get('area', 0))

        # image_path is relative to $LMUData/images/ODinW/
        # Format: {domain_name}/{filename}
        image_path = f'{domain_name}/{img_filename}'

        rows.append({
            'index': f'{domain_name}_{split_key}_{image_id}_{category_id}',
            'image_path': image_path,
            'question': build_bbox_question(cat_name),
            'answer': json.dumps(norm_bboxes),
            'category_id': category_id,
            'category_name': cat_name,
            'image_width': img_w,
            'image_height': img_h,
            'num_instances': len(anns),
            'annotation_ids': ','.join(str(a) for a in ann_ids),
            'domain': domain_name,
            'split': split_key,
            'areas': json.dumps([round(a, 2) for a in areas]),
        })

    return rows


def generate_tsvs(all_domains, output_dir, benchmark_label):
    """Generate train and val TSVs for a benchmark.

    Produces:
      {benchmark_label}_bbox_train.tsv  (rows from 'train' splits)
      {benchmark_label}_bbox_val.tsv    (rows from 'val' and 'test' splits)
    """
    train_rows = []
    val_rows = []

    for domain_name, splits in sorted(all_domains.items()):
        for split_key, split_info in splits.items():
            log.info(f'  Processing {domain_name}/{split_key} ...')
            rows = process_domain_split(domain_name, split_key, split_info)
            log.info(f'    → {len(rows)} rows')

            if split_key == 'train':
                train_rows.extend(rows)
            else:
                # val and test both go into the val TSV
                val_rows.extend(rows)

    # Write TSVs
    for suffix, rows in [('train', train_rows), ('val', val_rows)]:
        if not rows:
            log.warning(f'  No {suffix} rows for {benchmark_label}')
            continue
        tsv_path = os.path.join(output_dir, f'{benchmark_label}_bbox_{suffix}.tsv')
        df = pd.DataFrame(rows)
        df.to_csv(tsv_path, sep='\t', index=False)
        size_mb = os.path.getsize(tsv_path) / (1024 * 1024)
        log.info(f'  Wrote {tsv_path} ({len(df)} rows, {size_mb:.1f} MB)')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Prepare ODinW-13/35 datasets for mini-vlm-toolkit.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--glip-dir', required=True,
                        help='Path to the GLIP repository root.')
    parser.add_argument('--output-dir', default=None,
                        help='Output root directory (default: $LMUData).')
    parser.add_argument('--skip-image-move', action='store_true',
                        help='Skip moving images (useful if already moved).')
    parser.add_argument('--benchmarks', nargs='+', default=['13', '35'],
                        choices=['13', '35'],
                        help='Which benchmarks to generate (default: both).')
    args = parser.parse_args()

    output_dir = args.output_dir or os.environ.get('LMUData', None)
    if output_dir is None:
        log.error('Either --output-dir or $LMUData must be set.')
        sys.exit(1)

    glip_dir = args.glip_dir
    if not os.path.isdir(glip_dir):
        log.error(f'GLIP directory not found: {glip_dir}')
        sys.exit(1)

    # Try to read _all.json for ODinW-35 config list
    all_json = os.path.join(glip_dir, 'configs', 'odinw_35', '_all.json')
    odinw_35_configs = ODINW_35_CONFIGS
    if os.path.isfile(all_json):
        with open(all_json) as f:
            paths = json.load(f)
        # Extract config names from paths like "configs/odinw_35/PascalVOC.yaml"
        odinw_35_configs = [
            os.path.splitext(os.path.basename(p))[0] for p in paths
        ]

    # ── Step 1: Parse configs ────────────────────────────────────────────
    log.info('=' * 60)
    log.info('Step 1: Parsing GLIP configs')
    log.info('=' * 60)

    domains_13 = {}
    domains_35 = {}

    if '13' in args.benchmarks:
        log.info('--- ODinW-13 ---')
        domains_13 = load_all_configs(glip_dir, ODINW_13_CONFIGS, 'ODinW-13')

    if '35' in args.benchmarks:
        log.info('--- ODinW-35 ---')
        domains_35 = load_all_configs(glip_dir, odinw_35_configs, 'ODinW-35')

    # Merge all domains for image moving (union of both benchmarks)
    all_domains = {}
    all_domains.update(domains_13)
    all_domains.update(domains_35)

    # ── Step 2: Move images ──────────────────────────────────────────────
    if not args.skip_image_move:
        log.info('=' * 60)
        log.info('Step 2: Moving images to %s/images/ODinW/', output_dir)
        log.info('=' * 60)
        move_all_images(all_domains, output_dir)
    else:
        log.info('Skipping image move (--skip-image-move).')

    # ── Step 3: Generate TSVs ────────────────────────────────────────────
    log.info('=' * 60)
    log.info('Step 3: Generating TSV files')
    log.info('=' * 60)

    if '13' in args.benchmarks:
        log.info('--- ODinW-13 ---')
        generate_tsvs(domains_13, output_dir, 'ODinW13')

    if '35' in args.benchmarks:
        log.info('--- ODinW-35 ---')
        generate_tsvs(domains_35, output_dir, 'ODinW35')

    # ── Summary ──────────────────────────────────────────────────────────
    log.info('=' * 60)
    log.info('Done!')
    log.info('Generated TSV files:')
    for bm in args.benchmarks:
        label = f'ODinW{bm}'
        for split in ['train', 'val']:
            p = os.path.join(output_dir, f'{label}_bbox_{split}.tsv')
            if os.path.exists(p):
                size_mb = os.path.getsize(p) / (1024 * 1024)
                log.info(f'  {p}  ({size_mb:.1f} MB)')
    log.info('=' * 60)


if __name__ == '__main__':
    main()
