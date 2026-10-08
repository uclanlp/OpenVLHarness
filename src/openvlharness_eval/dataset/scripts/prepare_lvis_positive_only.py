#!/usr/bin/env python3
"""Prepare LVIS v1 dataset for mini-vlm-toolkit evaluation (positive-only).

This script converts the LVIS v1 annotation files into the TSV format expected
by mini-vlm-toolkit, and moves images into $LMUData/images/LVIS/.

It emits **positive-only** QA rows: one row per (image_id, category_id) pair
that has at least one annotation in the LVIS JSON.  Every row therefore has a
non-empty ``answer`` and ``num_instances >= 1``.  For the federated variant
(positive + negative rows where ``answer == []``), see
``prepare_lvis_federated.py``.

Usage (run in tmux — image moving can take a long time on slow disks):

    python -m openvlharness_eval.dataset.scripts.prepare_lvis_positive_only \
        --lvis-dir /path/to/lvis-api \
        [--output-dir $LMUData] \
        [--skip-test] \
        [--skip-image-move]

What this script does:
  1. Moves images from lvis-api/images/{train2017,val2017,test2017}/ into a
     flat directory ${LMUData}/images/LVIS/.  Images are MOVED (not copied)
     to save disk space.  COCO image IDs are globally unique, so there are
     no filename collisions across splits.

  2. Generates four TSV files (two evaluation modes × two annotated splits):

     Bounding-box evaluation:
       - LVIS_bbox_train_positive_only.tsv   (from lvis_v1_train.json)
       - LVIS_bbox_val_positive_only.tsv     (from lvis_v1_val.json)

     Segmentation-mask evaluation:
       - LVIS_mask_train_positive_only.tsv   (from lvis_v1_train.json)
       - LVIS_mask_val_positive_only.tsv     (from lvis_v1_val.json)

     Each row represents a unique (image, category) pair.  The "answer"
     column stores ground-truth annotations in JSON:
       • bbox TSV  →  [[x1, y1, x2, y2], ...]   (normalised on a 0-1000 scale)
       • mask TSV  →  [[[poly_x1,poly_y1,...], ...], ...]   (polygon coords,
                       one list-of-polygons per instance, absolute pixel coords)

  3. Test splits (test_dev, test_challenge) have NO annotations in LVIS v1.
     Their images are still moved (unless --skip-test), but no TSV rows are
     generated for them because there is no ground truth to evaluate against.

Notes on the generated TSV columns
-----------------------------------
  index           : unique row id  "{split}_{image_id}_{category_id}"
  image_path      : filename inside $LMUData/images/LVIS/  (e.g. 000000266409.jpg)
  question        : natural-language query asking the model to locate/segment
  answer          : ground-truth (JSON, format depends on bbox vs mask)
  category_id     : LVIS category id  (1-1203)
  category_name   : human-readable category name
  frequency       : category frequency bucket  ("r"=rare, "c"=common, "f"=frequent)
  synonyms        : pipe-separated list of synonym strings
  image_width     : image width in pixels
  image_height    : image height in pixels
  num_instances   : number of annotated instances of this category in this image
  annotation_ids  : comma-separated annotation ids (for cross-reference)
  split           : "train" or "val"
  areas           : JSON list of instance areas (in pixels²)

For mask TSVs only:
  answer_bbox     : bounding boxes on a 0-1000 scale (same format as bbox TSV)
                    Kept here so mask evaluation can also report box metrics.

Resumability
------------
  • Image moves are idempotent — if the destination already exists the source
    is skipped.
  • TSV generation overwrites any existing file of the same name.
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

from openvlharness_eval.dataset.scripts.prepare_odinw import build_bbox_question

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image moving
# ---------------------------------------------------------------------------

def move_images(src_dirs, dst_dir, skip_test=False):
    """Move images from COCO split directories into a single flat directory.

    Parameters
    ----------
    src_dirs : dict
        Mapping from split name to source directory path.
        Example: {"train2017": "/path/to/images/train2017", ...}
    dst_dir : str
        Destination directory (e.g. $LMUData/images/LVIS/).
    skip_test : bool
        If True, skip moving test2017 images (they have no LVIS annotations).
    """
    os.makedirs(dst_dir, exist_ok=True)

    splits_to_process = list(src_dirs.keys())
    if skip_test:
        splits_to_process = [s for s in splits_to_process if 'test' not in s]

    total_moved = 0
    total_skipped = 0
    total_missing = 0

    for split_name in splits_to_process:
        src_dir = src_dirs[split_name]
        if not os.path.isdir(src_dir):
            log.warning(f'Source directory not found, skipping: {src_dir}')
            continue

        files = sorted(os.listdir(src_dir))
        log.info(f'Processing {split_name}: {len(files)} files to move ...')
        split_moved = 0
        split_skipped = 0

        for i, fname in enumerate(files):
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(dst_dir, fname)

            if not os.path.isfile(src_path):
                continue

            if os.path.exists(dst_path):
                # Already moved (or duplicate) — skip
                split_skipped += 1
                continue

            try:
                shutil.move(src_path, dst_path)
                split_moved += 1
            except Exception as e:
                log.warning(f'Failed to move {src_path}: {e}')
                total_missing += 1

            # Progress every 10,000 files
            if (i + 1) % 10000 == 0:
                log.info(f'  [{split_name}] {i+1}/{len(files)} processed '
                         f'(moved={split_moved}, skipped={split_skipped})')

        log.info(f'  [{split_name}] Done: moved={split_moved}, skipped={split_skipped}')
        total_moved += split_moved
        total_skipped += split_skipped

    log.info(f'Image move summary: moved={total_moved}, '
             f'already_existed={total_skipped}, errors={total_missing}')


# ---------------------------------------------------------------------------
# Annotation processing
# ---------------------------------------------------------------------------

def load_lvis_annotations(json_path):
    """Load an LVIS v1 annotation JSON file.

    Returns (images_by_id, annotations, categories_by_id).
    """
    log.info(f'Loading annotations from {json_path} ...')
    t0 = time.time()
    with open(json_path, 'r') as f:
        data = json.load(f)
    log.info(f'  Loaded in {time.time()-t0:.1f}s')

    images_by_id = {img['id']: img for img in data['images']}
    categories_by_id = {cat['id']: cat for cat in data['categories']}

    log.info(f'  Images: {len(images_by_id)}, '
             f'Annotations: {len(data["annotations"])}, '
             f'Categories: {len(categories_by_id)}')

    return images_by_id, data['annotations'], categories_by_id


def image_id_to_filename(image_id):
    """Convert COCO/LVIS image id to the on-disk filename.

    COCO images use zero-padded 12-digit filenames: 000000266409.jpg
    """
    return f'{image_id:012d}.jpg'


def bbox_xywh_to_normalised_xyxy(bbox, img_w, img_h):
    """Convert LVIS bbox [x, y, w, h] (absolute) to [x1, y1, x2, y2] on a 0-1000 scale.

    Returns a list of four floats rounded to 2 decimal places.
    """
    x, y, w, h = bbox
    x1 = x / img_w * 1000.0
    y1 = y / img_h * 1000.0
    x2 = (x + w) / img_w * 1000.0
    y2 = (y + h) / img_h * 1000.0
    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


def group_annotations_by_image_category(annotations):
    """Group annotations by (image_id, category_id).

    Returns a dict mapping (image_id, category_id) → list of annotation dicts.
    """
    groups = defaultdict(list)
    for ann in annotations:
        key = (ann['image_id'], ann['category_id'])
        groups[key].append(ann)
    return groups


def build_mask_question(category_name):
    """Build the question prompt for segmentation-mask evaluation.

    Accepts either polygon coordinates or RLE-encoded masks in predictions.
    """
    return (
        f'Segment all instances of \'{category_name}\' in this image. '
        f'For each instance, output the segmentation mask as polygon '
        f'coordinates [[x1,y1,x2,y2,...], ...] in absolute pixel values, '
        f'or as an RLE-encoded mask {{"counts": "...", "size": [h, w]}}. '
        f'Output a JSON array with one entry per instance.'
    )


def generate_tsv_rows(grouped_anns, images_by_id, categories_by_id, split_name):
    """Generate TSV rows from grouped annotations.

    Parameters
    ----------
    grouped_anns : dict
        (image_id, category_id) → list of annotations.
    images_by_id : dict
        image_id → image metadata.
    categories_by_id : dict
        category_id → category metadata.
    split_name : str
        "train" or "val".

    Returns
    -------
    bbox_rows : list of dicts
        Rows for the bounding-box TSV.
    mask_rows : list of dicts
        Rows for the segmentation-mask TSV.
    """
    bbox_rows = []
    mask_rows = []

    # Iterate in deterministic (image_id, category_id) order so the TSV is
    # reproducible across runs regardless of dict iteration order.
    for (image_id, category_id) in sorted(grouped_anns.keys()):
        anns = grouped_anns[(image_id, category_id)]
        img = images_by_id.get(image_id)
        cat = categories_by_id.get(category_id)
        if img is None or cat is None:
            continue

        img_w = img['width']
        img_h = img['height']
        img_filename = image_id_to_filename(image_id)
        cat_name = cat['name']
        frequency = cat.get('frequency', '')
        synonyms = '|'.join(cat.get('synonyms', []))

        # Collect per-instance data
        norm_bboxes = []
        raw_segmentations = []
        ann_ids = []
        areas = []

        for ann in anns:
            # Bounding box: LVIS format [x, y, w, h] → normalised [x1, y1, x2, y2]
            norm_bboxes.append(bbox_xywh_to_normalised_xyxy(ann['bbox'], img_w, img_h))

            # Segmentation: list of polygon coordinate arrays (absolute pixel coords)
            # Each annotation's segmentation is a list of polygons (for multi-part objects)
            raw_segmentations.append(ann['segmentation'])

            ann_ids.append(ann['id'])
            areas.append(ann.get('area', 0))

        row_index = f'{split_name}_{image_id}_{category_id}'

        # --- Bounding-box TSV row ---
        bbox_rows.append({
            'index': row_index,
            'image_path': img_filename,
            'question': build_bbox_question(cat_name),
            'answer': json.dumps(norm_bboxes),
            'category_id': category_id,
            'category_name': cat_name,
            'frequency': frequency,
            'synonyms': synonyms,
            'image_width': img_w,
            'image_height': img_h,
            'num_instances': len(anns),
            'annotation_ids': ','.join(str(a) for a in ann_ids),
            'split': split_name,
            'areas': json.dumps([round(a, 2) for a in areas]),
        })

        # --- Segmentation-mask TSV row ---
        mask_rows.append({
            'index': row_index,
            'image_path': img_filename,
            'question': build_mask_question(cat_name),
            'answer': json.dumps(raw_segmentations),
            'answer_bbox': json.dumps(norm_bboxes),
            'category_id': category_id,
            'category_name': cat_name,
            'frequency': frequency,
            'synonyms': synonyms,
            'image_width': img_w,
            'image_height': img_h,
            'num_instances': len(anns),
            'annotation_ids': ','.join(str(a) for a in ann_ids),
            'split': split_name,
            'areas': json.dumps([round(a, 2) for a in areas]),
        })

    return bbox_rows, mask_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Prepare LVIS v1 dataset for mini-vlm-toolkit evaluation.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--lvis-dir', required=True,
        help='Path to the lvis-api repository root '
             '(contains data/ and images/ subdirectories).',
    )
    parser.add_argument(
        '--output-dir', default=None,
        help='Output root directory (default: $LMUData env var).',
    )
    parser.add_argument(
        '--skip-test', action='store_true',
        help='Skip moving test2017 images (they have no LVIS annotations).',
    )
    parser.add_argument(
        '--skip-image-move', action='store_true',
        help='Skip the image-moving step (useful if images were already moved).',
    )
    parser.add_argument(
        '--splits', nargs='+', default=['val', 'train'],
        choices=['val', 'train'],
        help='Which annotation splits to process (default: val train).',
    )
    args = parser.parse_args()

    # Resolve output directory
    output_dir = args.output_dir or os.environ.get('LMUData', None)
    if output_dir is None:
        log.error('Either --output-dir or $LMUData environment variable must be set.')
        sys.exit(1)

    lvis_dir = args.lvis_dir
    if not os.path.isdir(lvis_dir):
        log.error(f'LVIS directory not found: {lvis_dir}')
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 1: Move images
    # -----------------------------------------------------------------------
    if not args.skip_image_move:
        log.info('=' * 60)
        log.info('Step 1: Moving images to %s/images/LVIS/', output_dir)
        log.info('=' * 60)

        src_dirs = {
            'train2017': os.path.join(lvis_dir, 'images', 'train2017'),
            'val2017':   os.path.join(lvis_dir, 'images', 'val2017'),
            'test2017':  os.path.join(lvis_dir, 'images', 'test2017'),
        }
        dst_dir = os.path.join(output_dir, 'images', 'LVIS')
        move_images(src_dirs, dst_dir, skip_test=args.skip_test)
    else:
        log.info('Skipping image move (--skip-image-move).')

    # -----------------------------------------------------------------------
    # Step 2: Generate TSV files from annotation JSONs
    # -----------------------------------------------------------------------
    log.info('=' * 60)
    log.info('Step 2: Generating TSV files')
    log.info('=' * 60)

    # Map split names to annotation JSON files
    annotation_files = {
        'train': os.path.join(lvis_dir, 'data', 'lvis_v1_train.json'),
        'val':   os.path.join(lvis_dir, 'data', 'lvis_v1_val.json'),
    }

    # Hard-coded expected positive-only row counts.  Loud signal if the source
    # JSON ever changes — kept in sync with prepare_lvis_federated.py.
    EXPECTED_POSITIVE_ROWS = {'train': 360_952, 'val': 70_139}

    for split_name in args.splits:
        ann_file = annotation_files[split_name]
        if not os.path.isfile(ann_file):
            log.warning(f'Annotation file not found, skipping {split_name}: {ann_file}')
            continue

        log.info(f'--- Processing split: {split_name} ---')
        images_by_id, annotations, categories_by_id = load_lvis_annotations(ann_file)

        # Group annotations by (image_id, category_id)
        log.info('  Grouping annotations by (image_id, category_id) ...')
        grouped = group_annotations_by_image_category(annotations)
        positive_pairs = len(grouped)
        log.info(f'  Unique (image, category) pairs: {positive_pairs}')

        # Generate rows (sorted deterministically by (image_id, category_id))
        log.info('  Generating TSV rows ...')
        bbox_rows, mask_rows = generate_tsv_rows(
            grouped, images_by_id, categories_by_id, split_name
        )
        log.info(f'  Generated {len(bbox_rows)} bbox rows, {len(mask_rows)} mask rows')

        # Write bbox TSV
        bbox_tsv_path = os.path.join(output_dir, f'LVIS_bbox_{split_name}_positive_only.tsv')
        bbox_df = pd.DataFrame(bbox_rows)
        bbox_df.to_csv(bbox_tsv_path, sep='\t', index=False)
        log.info(f'  Wrote {bbox_tsv_path} ({len(bbox_df)} rows)')

        # Write mask TSV
        mask_tsv_path = os.path.join(output_dir, f'LVIS_mask_{split_name}_positive_only.tsv')
        mask_df = pd.DataFrame(mask_rows)
        mask_df.to_csv(mask_tsv_path, sep='\t', index=False)
        log.info(f'  Wrote {mask_tsv_path} ({len(mask_df)} rows)')

        # ------------------------------------------------------------------
        # Validation — printed and asserted
        # ------------------------------------------------------------------
        empty_answer_rows = sum(1 for r in bbox_rows if json.loads(r['answer']) == [])
        malformed_box_rows = 0
        for r in bbox_rows:
            boxes = json.loads(r['answer'])
            if (not isinstance(boxes, list)) or len(boxes) == 0:
                malformed_box_rows += 1
                continue
            for b in boxes:
                if (not isinstance(b, list)) or len(b) != 4 \
                        or not all(isinstance(x, (int, float)) for x in b):
                    malformed_box_rows += 1
                    break

        log.info(f'  [{split_name}] num_images={len(images_by_id)}  '
                 f'num_categories={len(categories_by_id)}  '
                 f'num_annotations={len(annotations)}')
        log.info(f'  [{split_name}] positive_pairs={positive_pairs}  '
                 f'output_rows={len(bbox_rows)}  '
                 f'num_positive_rows={len(bbox_rows)}  '
                 f'empty_answer_rows={empty_answer_rows}  '
                 f'malformed_box_rows={malformed_box_rows}')

        assert empty_answer_rows == 0, \
            f'[{split_name}] expected 0 empty-answer rows, got {empty_answer_rows}'
        assert malformed_box_rows == 0, \
            f'[{split_name}] expected 0 malformed boxes, got {malformed_box_rows}'
        assert len(bbox_rows) == positive_pairs, \
            f'[{split_name}] output_rows ({len(bbox_rows)}) != positive_pairs ({positive_pairs})'

        expected = EXPECTED_POSITIVE_ROWS.get(split_name)
        if expected is not None and len(bbox_rows) != expected:
            log.warning(
                f'[{split_name}] row count {len(bbox_rows)} differs from expected '
                f'{expected}.  This is non-fatal but may indicate the LVIS JSON has '
                f'changed.  Update EXPECTED_POSITIVE_ROWS if intentional.'
            )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    log.info('=' * 60)
    log.info('Done!')
    log.info('')
    log.info('NOTE: LVIS test splits (test_dev, test_challenge) have NO ground-truth')
    log.info('annotations in LVIS v1.  No TSV files were generated for them.')
    log.info('If you need test-set evaluation, submit predictions to the LVIS')
    log.info('evaluation server: https://eval.ai/web/challenges/challenge-page/675')
    log.info('')
    log.info('Generated TSV files:')
    for split_name in args.splits:
        for mode in ['bbox', 'mask']:
            p = os.path.join(output_dir, f'LVIS_{mode}_{split_name}_positive_only.tsv')
            if os.path.exists(p):
                size_mb = os.path.getsize(p) / (1024 * 1024)
                log.info(f'  {p}  ({size_mb:.1f} MB)')
    log.info('=' * 60)


if __name__ == '__main__':
    main()
