#!/usr/bin/env python3
"""Prepare FSCD-LVIS *counting* TSV files for OpenVLHarness evaluation.

Counterpart to ``prepare_fscdlvis_box.py``: same per-(image, class) row
structure, but the model is asked for a single integer count instead of
bounding boxes. Each row asks "How many {class} are there in the image?"
and the ground-truth answer is the number of GT instances of that class
in that image.

Outputs (under $LMUData):
  Seen-class split (canonical FSCD-LVIS):
    FSCD_LVIS_count_train.tsv  (4,000 rows)
    FSCD_LVIS_count_val.tsv    (1,181 rows)
    FSCD_LVIS_count_test.tsv   (1,014 rows)

  Unseen-class split (open-vocab generalization, no val):
    FSCD_LVIS_unseen_count_train.tsv  (3,953 rows)
    FSCD_LVIS_unseen_count_test.tsv   (2,242 rows)

Usage:

    python -m evaluation_kit.dataset.scripts.prepare_fscdlvis_count \
        --fscd-root "$FSCD_ROOT" \
        --lvis-taxonomy "$LVIS_TAXONOMY" \
        [--output-dir $LMUData]

Eval is handled by ``FSC147Dataset`` (the standard counting evaluator), which
registers these five names, so ``openvlharness-eval --data FSCD_LVIS_count_val``
works.

Schema:
  index           "<split>_<image_id>_<lvis_cat_id>"
  image_path      bare filename, e.g. "000000315601.jpg"
  question        "How many {category_name} are there in the image?
                   Your final answer should be a single integer."
                  ({category_name} is the raw LVIS name from the taxonomy.)
  answer          integer GT count
  category_id     LVIS taxonomy id
  category_name   LVIS taxonomy name
  image_width     pixels
  image_height    pixels
  num_instances   same as answer; kept for parity with the bbox TSV
  split           "train" | "val" | "test" | "unseen_train" | "unseen_test"
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)



# (ann_file_in_annotations/, split_name, index_prefix)
SPLITS: List[Tuple[str, str, str]] = [
    ('instances_train.json',        'train',         'train'),
    ('instances_val.json',          'val',           'val'),
    ('instances_test.json',         'test',          'test'),
    ('unseen_instances_train.json', 'unseen_train',  'unseen_train'),
    ('unseen_instances_test.json',  'unseen_test',   'unseen_test'),
]

TSV_NAMES = {
    'train':        'FSCD_LVIS_count_train.tsv',
    'val':          'FSCD_LVIS_count_val.tsv',
    'test':         'FSCD_LVIS_count_test.tsv',
    'unseen_train': 'FSCD_LVIS_unseen_count_train.tsv',
    'unseen_test':  'FSCD_LVIS_unseen_count_test.tsv',
}

EXPECTED_ROWS = {
    'train':        4000,
    'val':          1181,
    'test':         1014,
    'unseen_train': 3953,
    'unseen_test':  2242,
}


def build_count_question(category_name: str) -> str:
    """Build the counting prompt for one class."""
    return (
        f"How many {category_name} are there in the image? "
        f"Your final answer should be a single integer."
    )


def process_split(
    inst_path: str,
    split_name: str,
    index_prefix: str,
    lvis_id_to_name: Dict[int, str],
) -> List[Dict]:
    log.info('  Loading %s ...', inst_path)
    with open(inst_path) as f:
        inst = json.load(f)

    # Dedupe images list on id (FSCD-LVIS test ships dupes).
    images_by_id: Dict[int, dict] = {}
    for im in inst['images']:
        images_by_id.setdefault(im['id'], im)

    # Group annotations by image_id; capture (single) per-image category_id.
    anns_by_img: Dict[int, list] = defaultdict(list)
    cat_by_img: Dict[int, int] = {}
    for ann in inst['annotations']:
        anns_by_img[ann['image_id']].append(ann)
        cat_by_img.setdefault(ann['image_id'], ann['category_id'])

    rows: List[Dict] = []
    no_annotations = 0
    missing_class_name = 0
    for image_id in sorted(images_by_id.keys()):
        img = images_by_id[image_id]
        fn = img['file_name']
        anns = anns_by_img.get(image_id, [])
        if not anns:
            no_annotations += 1
            continue
        cat_id = cat_by_img[image_id]
        class_name = lvis_id_to_name.get(int(cat_id))
        if not class_name:
            missing_class_name += 1
            continue

        count = len(anns)
        rows.append({
            'index':         f'{index_prefix}_{image_id}_{cat_id}',
            'image_path':    fn,
            'question':      build_count_question(class_name),
            'answer':        count,
            'category_id':   int(cat_id),
            'category_name': class_name,
            'image_width':   img['width'],
            'image_height':  img['height'],
            'num_instances': count,
            'split':         split_name,
        })

    if no_annotations or missing_class_name:
        log.warning(
            '  [%s] skipped %d images with no annotations and %d with unknown '
            'LVIS category_id.', split_name, no_annotations, missing_class_name,
        )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description='Prepare FSCD-LVIS counting TSV files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--fscd-root', required=True,
        help='Path to the unpacked FSCD dataset root.',
    )
    parser.add_argument(
        '--lvis-taxonomy', required=True,
        help='Path to lvis_v1_val.json containing category names.',
    )
    parser.add_argument(
        '--output-dir', default=None,
        help='Output root (default: $LMUData).',
    )
    parser.add_argument(
        '--splits', nargs='+', default=None,
        choices=list(TSV_NAMES.keys()),
        help='Optional subset of splits to emit (default: all five).',
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.environ.get('LMUData', None)
    if output_dir is None:
        log.error('Either --output-dir or $LMUData must be set.')
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)

    ann_dir = os.path.join(args.fscd_root, 'annotations')
    if not os.path.isdir(ann_dir):
        log.error('FSCD-LVIS annotations dir not found: %s', ann_dir)
        sys.exit(1)
    if not os.path.isfile(args.lvis_taxonomy):
        log.error('LVIS taxonomy JSON not found: %s', args.lvis_taxonomy)
        sys.exit(1)

    log.info('Loading LVIS taxonomy from %s', args.lvis_taxonomy)
    with open(args.lvis_taxonomy) as f:
        tax = json.load(f)
    lvis_id_to_name = {int(c['id']): c['name'] for c in tax['categories']}
    log.info('  %d LVIS categories loaded.', len(lvis_id_to_name))

    splits_to_run = args.splits or list(TSV_NAMES.keys())

    for ann_file, split_name, idx_prefix in SPLITS:
        if split_name not in splits_to_run:
            continue
        inst_path = os.path.join(ann_dir, ann_file)
        if not os.path.isfile(inst_path):
            log.warning('Skipping %s: file missing (%s)', split_name, inst_path)
            continue

        log.info('=== Processing split: %s ===', split_name)
        rows = process_split(inst_path, split_name, idx_prefix, lvis_id_to_name)
        log.info('  Generated %d rows for %s', len(rows), split_name)

        expected = EXPECTED_ROWS.get(split_name)
        if expected is not None and len(rows) != expected:
            log.warning(
                '  Row count %d differs from expected %d for %s. '
                'Non-fatal; investigate if unexpected.',
                len(rows), expected, split_name,
            )

        df = pd.DataFrame(rows)
        out_path = os.path.join(output_dir, TSV_NAMES[split_name])
        df.to_csv(out_path, sep='\t', index=False)
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        log.info('  Wrote %s (%d rows, %.1f MB)', out_path, len(df), size_mb)

    log.info('Done.')


if __name__ == '__main__':
    main()
