#!/usr/bin/env python3
"""Prepare FSCD-147 bbox-grounding TSV files for mini-vlm-toolkit evaluation.

This is the *detection* (FSCD-147) counterpart to ``prepare_fsc147.py`` (which
prepares the *counting* FSC-147 / FSC-147-D TSVs).  Each row asks the model to
locate every instance of one text-described class in one image — same prompt
shape used by ODinW and LVIS bbox evaluation.

Usage:

    python -m mini_vlm_toolkit.dataset.scripts.prepare_fscd147_box \
        [--fscd-root /local/<user>/tool-vision/FSCD/FSCD_147] \
        [--desc-json /data/<shared>/datasets/CounTX/FSC-147-D.json] \
        [--output-dir $LMUData]

Outputs (under $LMUData):
  FSCD_147_bbox_val.tsv   (1,286 rows)
  FSCD_147_bbox_test.tsv  (1,190 rows)

train is intentionally skipped — FSCD-147 has no ground-truth bboxes for train
(only points + 3 exemplars per image), so there is no detection GT to evaluate
against.

Images are not copied: ``lmu_data/images/FSC147/`` already mirrors
``images_384_VarV2/`` exactly (verified at prep time).

Schema (matches LVIS/ODinW bbox TSVs, plus an exemplar column):
  index           "<split>_<image_id>_1"  (category_id is the dummy 1)
  image_path      bare filename, e.g. "5905.jpg"
  question        from build_bbox_question(class_name) (ODinW/LVIS prompt)
  answer          JSON [[x1,y1,x2,y2], ...] on a 0-1000 scale
  category_id     always 1 (FSCD-147 has one target class per image)
  category_name   text label from FSC-147-D's text_description_singular
  image_width     pixels (matches on-disk file)
  image_height    pixels
  num_instances   number of GT boxes for this image
  annotation_ids  comma-separated source ann ids
  split           "val" | "test"
  areas           JSON list of per-instance areas (px²)
  exemplar_boxes  JSON [[x1,y1,x2,y2], ...] on a 0-1000 scale
                  (3 representative instance boxes; NOT included in the prompt,
                   stored for future few-shot variants)
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List

import pandas as pd

from mini_vlm_toolkit.dataset.scripts.prepare_odinw import (
    bbox_xywh_to_normalised_xyxy,
    build_bbox_question,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# Edit-friendly path constants. Override via CLI when the FSCD folder moves.
FSCD_147_ROOT_DEFAULT = '/local/<user>/tool-vision/FSCD/FSCD_147'
FSC_147_D_JSON_DEFAULT = '/data/<shared>/datasets/CounTX/FSC-147-D.json'

EXPECTED_ROWS = {'val': 1286, 'test': 1190}


def _exemplar_quads_to_norm_boxes(quads, img_w: float, img_h: float) -> List[List[float]]:
    """Convert exemplar quadrilateral corner lists to normalised xyxy 0-1000.

    Each quad is a list of 4 [x, y] points in the resized (384-height)
    coordinate space, which matches ``instances_*.json`` image dims.
    """
    out: List[List[float]] = []
    for quad in quads:
        if not quad:
            continue
        xs = [float(p[0]) for p in quad]
        ys = [float(p[1]) for p in quad]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        if x2 <= x1 or y2 <= y1:
            continue
        out.append([
            round(x1 / img_w * 1000.0, 2),
            round(y1 / img_h * 1000.0, 2),
            round(x2 / img_w * 1000.0, 2),
            round(y2 / img_h * 1000.0, 2),
        ])
    return out


def process_split(
    split_name: str,
    instances_path: str,
    desc_data: Dict,
    anno_data: Dict,
) -> List[Dict]:
    log.info('  Loading %s ...', instances_path)
    with open(instances_path) as f:
        inst = json.load(f)

    images_by_id = {im['id']: im for im in inst['images']}
    anns_by_img: Dict[int, list] = {}
    for ann in inst['annotations']:
        anns_by_img.setdefault(ann['image_id'], []).append(ann)

    rows: List[Dict] = []
    skipped_no_class = 0
    skipped_no_split = 0
    for image_id in sorted(images_by_id.keys()):
        img = images_by_id[image_id]
        fn = img['file_name']
        img_w = img['width']
        img_h = img['height']

        desc_entry = desc_data.get(fn, {})
        # Defensive: FSC-147-D's data_split should match the instances JSON's
        # split (val or test). Skip silently if it doesn't.
        ds = desc_entry.get('data_split')
        if ds and ds != split_name:
            skipped_no_split += 1
            continue
        class_name = desc_entry.get(
            'text_description_singular',
            desc_entry.get('text_description', ''),
        )
        if not class_name:
            skipped_no_class += 1
            continue

        anns = anns_by_img.get(image_id, [])
        norm_bboxes = [
            bbox_xywh_to_normalised_xyxy(a['bbox'], img_w, img_h) for a in anns
        ]
        ann_ids = [a['id'] for a in anns]
        areas = [round(a.get('area', 0), 2) for a in anns]

        exemplar_quads = anno_data.get(fn, {}).get('box_examples_coordinates', [])
        exemplar_boxes = _exemplar_quads_to_norm_boxes(exemplar_quads, img_w, img_h)

        rows.append({
            'index':          f'{split_name}_{image_id}_1',
            'image_path':     fn,
            'question':       build_bbox_question(class_name),
            'answer':         json.dumps(norm_bboxes),
            'category_id':    1,
            'category_name':  class_name,
            'image_width':    img_w,
            'image_height':   img_h,
            'num_instances':  len(anns),
            'annotation_ids': ','.join(str(a) for a in ann_ids),
            'split':          split_name,
            'areas':          json.dumps(areas),
            'exemplar_boxes': json.dumps(exemplar_boxes),
        })

    if skipped_no_class or skipped_no_split:
        log.warning(
            '  [%s] skipped %d rows (no class label) and %d rows (split mismatch).',
            split_name, skipped_no_class, skipped_no_split,
        )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description='Prepare FSCD-147 bbox-grounding TSV files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--fscd-root', default=FSCD_147_ROOT_DEFAULT,
        help=f'FSCD-147 dataset root (default: {FSCD_147_ROOT_DEFAULT}).',
    )
    parser.add_argument(
        '--desc-json', default=FSC_147_D_JSON_DEFAULT,
        help=f'Path to FSC-147-D.json (default: {FSC_147_D_JSON_DEFAULT}).',
    )
    parser.add_argument(
        '--output-dir', default=None,
        help='Output root (default: $LMUData).',
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.environ.get('LMUData', None)
    if output_dir is None:
        log.error('Either --output-dir or $LMUData must be set.')
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(args.fscd_root):
        log.error('FSCD-147 root not found: %s', args.fscd_root)
        sys.exit(1)
    if not os.path.isfile(args.desc_json):
        log.error('FSC-147-D.json not found: %s', args.desc_json)
        sys.exit(1)

    log.info('Loading FSC-147-D descriptions from %s', args.desc_json)
    with open(args.desc_json) as f:
        desc_data = json.load(f)

    anno_path = os.path.join(args.fscd_root, 'annotation_FSC147_384.json')
    log.info('Loading exemplars from %s', anno_path)
    with open(anno_path) as f:
        anno_data = json.load(f)

    split_to_inst = {
        'val':  os.path.join(args.fscd_root, 'instances_val.json'),
        'test': os.path.join(args.fscd_root, 'instances_test.json'),
    }

    for split_name, inst_path in split_to_inst.items():
        if not os.path.isfile(inst_path):
            log.warning('Skipping %s: file missing (%s)', split_name, inst_path)
            continue

        log.info('=== Processing split: %s ===', split_name)
        rows = process_split(split_name, inst_path, desc_data, anno_data)
        log.info('  Generated %d rows for %s', len(rows), split_name)

        expected = EXPECTED_ROWS.get(split_name)
        if expected is not None and len(rows) != expected:
            log.warning(
                '  Row count %d differs from expected %d for %s. '
                'Non-fatal; investigate if unexpected.',
                len(rows), expected, split_name,
            )

        df = pd.DataFrame(rows)
        out_path = os.path.join(output_dir, f'FSCD_147_bbox_{split_name}.tsv')
        df.to_csv(out_path, sep='\t', index=False)
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        log.info('  Wrote %s (%d rows, %.1f MB)', out_path, len(df), size_mb)

    log.info('Done.')


if __name__ == '__main__':
    main()
