#!/usr/bin/env python3
"""Prepare LVIS v1 federated bbox QA TSVs.

LVIS evaluation is *federated*: each image is annotated for only a subset of
categories.  The annotation JSON exposes this via per-image
``neg_category_ids`` (categories the annotators verified are absent) and
``not_exhaustive_category_ids`` (present but possibly under-annotated).  This
script emits one QA row per (image_id, category_id) pair in
``positive_pairs | negative_pairs``:

  positive row: at least one GT annotation, ``answer`` = list of normalised
                bboxes, ``num_instances >= 1``, ``is_positive = True``.
  negative row: image was verified to NOT contain the category, ``answer`` =
                ``[]``, ``num_instances == 0``, ``is_positive = False``.

For the positive-only variant (no negative rows), see
``prepare_lvis_positive_only.py``.

Usage (run AFTER prepare_lvis_positive_only.py has moved the images):

    python -m evaluation_kit.dataset.scripts.prepare_lvis_federated \
        --lvis-dir /path/to/lvis-api \
        [--output-dir $LMUData] \
        [--splits val train]

Outputs (in ``$LMUData`` / ``--output-dir``):

    LVIS_bbox_train_federated.tsv   (from lvis_v1_train.json)
    LVIS_bbox_val_federated.tsv     (from lvis_v1_val.json)

TSV schema
----------
Same 14 columns as the positive-only bbox TSV, plus:
    is_positive                : "True" for positive rows, "False" for negative.
    is_not_exhaustive_category : "True" if category_id is in
                                 image["not_exhaustive_category_ids"], else "False".

Negative rows fill the per-instance fields with empty values:
    answer = "[]", num_instances = 0, annotation_ids = "", areas = "[]".
"""

import argparse
import json
import logging
import os
import sys

import pandas as pd

from evaluation_kit.dataset.scripts.prepare_lvis_positive_only import (
    bbox_xywh_to_normalised_xyxy,
    build_bbox_question,
    group_annotations_by_image_category,
    image_id_to_filename,
    load_lvis_annotations,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# Columns written, in order.  First 14 match prepare_lvis_positive_only's bbox
# TSV exactly so diffs across the two files are minimal; the two federated-
# specific bool columns are appended at the end.
COLUMNS = [
    'index', 'image_path', 'question', 'answer',
    'category_id', 'category_name', 'frequency', 'synonyms',
    'image_width', 'image_height', 'num_instances', 'annotation_ids',
    'split', 'areas',
    'is_positive', 'is_not_exhaustive_category',
]

# Hard-coded expected federated row counts (positive + negative) per split.
# Loud signal if the source JSON ever changes.
EXPECTED_FEDERATED_ROWS = {'train': 1_321_251, 'val': 256_352}
EXPECTED_POSITIVE_ROWS  = {'train':   360_952, 'val':  70_139}
EXPECTED_NEGATIVE_ROWS  = {'train':   960_299, 'val': 186_213}


def build_federated_rows(images_by_id, annotations, categories_by_id, split_name):
    """Build the federated TSV rows for one split.

    Returns
    -------
    rows : list of dicts
        Sorted by (image_id, category_id), one row per pair in
        positive_pairs ∪ negative_pairs.
    stats : dict
        Diagnostic counts for printing/asserting.
    """
    # Positive pairs: (image_id, category_id) with at least one annotation.
    positive_pairs = {(ann['image_id'], ann['category_id']) for ann in annotations}

    # Negative pairs: per-image neg_category_ids verified absent by annotators.
    negative_pairs = {
        (img['id'], cat_id)
        for img in images_by_id.values()
        for cat_id in img.get('neg_category_ids', [])
    }

    overlap = positive_pairs & negative_pairs
    assert not overlap, (
        f'[{split_name}] positive_pairs and negative_pairs overlap on '
        f'{len(overlap)} pairs (e.g. {next(iter(overlap))}); the LVIS JSON '
        f'is inconsistent.'
    )

    # not_exhaustive lookup: image_id → set(category_id).
    not_exhaustive_by_image = {
        img_id: set(img.get('not_exhaustive_category_ids', []))
        for img_id, img in images_by_id.items()
    }

    grouped = group_annotations_by_image_category(annotations)

    log.info(f'  positive_pairs={len(positive_pairs)}  '
             f'negative_pairs={len(negative_pairs)}  '
             f'overlap={len(overlap)}')

    rows = []
    for (image_id, category_id) in sorted(positive_pairs | negative_pairs):
        img = images_by_id.get(image_id)
        cat = categories_by_id.get(category_id)
        if img is None or cat is None:
            # Defensive — shouldn't happen for well-formed LVIS JSON.
            continue

        img_w = img['width']
        img_h = img['height']
        img_filename = image_id_to_filename(image_id)
        cat_name = cat['name']
        frequency = cat.get('frequency', '')
        synonyms = '|'.join(cat.get('synonyms', []))
        is_not_exhaustive = category_id in not_exhaustive_by_image.get(image_id, set())

        is_positive = (image_id, category_id) in positive_pairs
        if is_positive:
            anns = grouped[(image_id, category_id)]
            norm_bboxes = [
                bbox_xywh_to_normalised_xyxy(ann['bbox'], img_w, img_h)
                for ann in anns
            ]
            ann_ids = [ann['id'] for ann in anns]
            areas = [ann.get('area', 0) for ann in anns]
            answer_str = json.dumps(norm_bboxes)
            num_instances = len(anns)
            annotation_ids_str = ','.join(str(a) for a in ann_ids)
            areas_str = json.dumps([round(a, 2) for a in areas])
        else:
            answer_str = '[]'
            num_instances = 0
            annotation_ids_str = ''
            areas_str = '[]'

        rows.append({
            'index': f'{split_name}_{image_id}_{category_id}',
            'image_path': img_filename,
            'question': build_bbox_question(cat_name),
            'answer': answer_str,
            'category_id': category_id,
            'category_name': cat_name,
            'frequency': frequency,
            'synonyms': synonyms,
            'image_width': img_w,
            'image_height': img_h,
            'num_instances': num_instances,
            'annotation_ids': annotation_ids_str,
            'split': split_name,
            'areas': areas_str,
            'is_positive': is_positive,
            'is_not_exhaustive_category': is_not_exhaustive,
        })

    stats = {
        'num_images': len(images_by_id),
        'num_categories': len(categories_by_id),
        'num_annotations': len(annotations),
        'positive_pairs': len(positive_pairs),
        'negative_pairs': len(negative_pairs),
        'positive_and_negative_overlap': len(overlap),
        'output_rows': len(rows),
        'num_positive_rows': sum(1 for r in rows if r['is_positive']),
        'num_negative_rows': sum(1 for r in rows if not r['is_positive']),
    }
    return rows, stats


def validate(rows, stats, split_name):
    """Run hard asserts and return augmented stats with empty/malformed counts."""
    empty_answer_rows = sum(1 for r in rows if json.loads(r['answer']) == [])
    malformed_box_rows = 0
    for r in rows:
        boxes = json.loads(r['answer'])
        if not isinstance(boxes, list):
            malformed_box_rows += 1
            continue
        for b in boxes:
            if (not isinstance(b, list)) or len(b) != 4 \
                    or not all(isinstance(x, (int, float)) for x in b):
                malformed_box_rows += 1
                break

    stats = {
        **stats,
        'empty_answer_rows': empty_answer_rows,
        'malformed_box_rows': malformed_box_rows,
    }

    log.info(f'  [{split_name}] ' + '  '.join(f'{k}={v}' for k, v in stats.items()))

    assert stats['positive_and_negative_overlap'] == 0, \
        f'[{split_name}] positive/negative overlap is non-zero'
    assert stats['num_positive_rows'] + stats['num_negative_rows'] == stats['output_rows'], \
        f'[{split_name}] positive+negative != output_rows'
    assert stats['num_negative_rows'] == empty_answer_rows, (
        f'[{split_name}] num_negative_rows ({stats["num_negative_rows"]}) '
        f'!= empty_answer_rows ({empty_answer_rows})'
    )
    assert malformed_box_rows == 0, \
        f'[{split_name}] malformed_box_rows={malformed_box_rows}'

    expected_total = EXPECTED_FEDERATED_ROWS.get(split_name)
    expected_pos   = EXPECTED_POSITIVE_ROWS.get(split_name)
    expected_neg   = EXPECTED_NEGATIVE_ROWS.get(split_name)
    if expected_total is not None and stats['output_rows'] != expected_total:
        log.warning(
            f'[{split_name}] output_rows={stats["output_rows"]} differs from '
            f'expected {expected_total} (positives expected {expected_pos}, '
            f'negatives expected {expected_neg}).  Non-fatal but may indicate '
            f'the LVIS JSON has changed; update EXPECTED_*_ROWS if intentional.'
        )

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Prepare LVIS v1 federated (positive + negative) bbox TSVs.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--lvis-dir', required=True,
        help='Path to the lvis-api repository root '
             '(must contain data/lvis_v1_{train,val}.json).',
    )
    parser.add_argument(
        '--output-dir', default=None,
        help='Output directory for the TSV files (default: $LMUData).',
    )
    parser.add_argument(
        '--splits', nargs='+', default=['val', 'train'],
        choices=['val', 'train'],
        help='Which annotation splits to process (default: val train).',
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.environ.get('LMUData', None)
    if output_dir is None:
        log.error('Either --output-dir or $LMUData environment variable must be set.')
        sys.exit(1)

    lvis_dir = args.lvis_dir
    if not os.path.isdir(lvis_dir):
        log.error(f'LVIS directory not found: {lvis_dir}')
        sys.exit(1)

    annotation_files = {
        'train': os.path.join(lvis_dir, 'data', 'lvis_v1_train.json'),
        'val':   os.path.join(lvis_dir, 'data', 'lvis_v1_val.json'),
    }

    log.info('=' * 60)
    log.info('Generating federated LVIS bbox TSVs')
    log.info('=' * 60)

    for split_name in args.splits:
        ann_file = annotation_files[split_name]
        if not os.path.isfile(ann_file):
            log.warning(f'Annotation file not found, skipping {split_name}: {ann_file}')
            continue

        log.info(f'--- Processing split: {split_name} ---')
        images_by_id, annotations, categories_by_id = load_lvis_annotations(ann_file)

        rows, stats = build_federated_rows(
            images_by_id, annotations, categories_by_id, split_name
        )

        out_path = os.path.join(output_dir, f'LVIS_bbox_{split_name}_federated.tsv')
        df = pd.DataFrame(rows, columns=COLUMNS)
        df.to_csv(out_path, sep='\t', index=False)
        log.info(f'  Wrote {out_path} ({len(df)} rows)')

        validate(rows, stats, split_name)

    log.info('=' * 60)
    log.info('Done!')
    log.info('Generated TSV files:')
    for split_name in args.splits:
        p = os.path.join(output_dir, f'LVIS_bbox_{split_name}_federated.tsv')
        if os.path.exists(p):
            size_mb = os.path.getsize(p) / (1024 * 1024)
            log.info(f'  {p}  ({size_mb:.1f} MB)')
    log.info('=' * 60)


if __name__ == '__main__':
    main()
