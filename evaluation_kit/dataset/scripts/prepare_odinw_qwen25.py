#!/usr/bin/env python3
"""Prepare ODinW-13 TSVs that follow the Qwen2.5-VL evaluation protocol.

This is the sibling of ``prepare_odinw.py``. The key differences vs the
full-detection TSVs:

  * **Positives-only.** One row per (image, category) pair *that actually has
    at least one annotation* in the COCO JSON. Negative pairs are not emitted
    — matches HumanZhong's "we select the obj_names that exist in the image
    during evaluation" protocol from QwenLM/Qwen3-VL issue #839.

  * **MMDetection's caption_prompt rewrites** (default). Class names follow
    the rewritten phrases from the MMDetection Grounding-DINO ODinW-13 val
    config (e.g. ``Cottontail-Rabbit`` -> ``rabbit``, ``hand`` ->
    ``hand of a person``, ``pothole`` -> ``there are some holes on the
    road``). Use ``--class-names raw`` to keep the original MMDet
    ``class_name`` tuple verbatim.

  * **Qwen2.5-VL prompt template** (default). Single-sentence prompt
    ``Locate every <X> in the image and output the coordinates in JSON
    format.`` Use ``--prompt-format adapted`` to substitute this repo's
    existing verbose 0-1000-normalized prompt instead — useful for an A/B
    that isolates "per-image class filtering" from "prompt + output format".

Two toggles give four combinations; the filename suffix marks the deviations
from the default (mmdet classes + Qwen2.5 prompt):

  ODinW13_bbox_<split>_qwen25.tsv               (default)
  ODinW13_bbox_<split>_qwen25_raw.tsv           --class-names raw
  ODinW13_bbox_<split>_qwen25_adapted.tsv       --prompt-format adapted
  ODinW13_bbox_<split>_qwen25_adapted_raw.tsv   both

Usage:

    python -m evaluation_kit.dataset.scripts.prepare_odinw_qwen25 \\
        --glip-dir /path/to/GLIP \\
        [--output-dir $LMUData] \\
        [--splits val test] \\
        [--class-names {mmdet,raw}] \\
        [--prompt-format {qwen25,adapted}]

The Qwen2.5-faithful prompt expects model output of the form
``[{"bbox_2d": [x1, y1, x2, y2], "label": "..."}, ...]`` in *pixel*
coordinates of the original image. The downstream parser must extract
``bbox_2d`` and renormalise to 0-1000 using the TSV's image_width /
image_height columns. The ``label`` field is model-emitted (not requested
by the prompt) and should be ignored at eval time — the queried class is
already fixed by the row's index/category_id columns.
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd

from evaluation_kit.dataset.scripts.prepare_odinw import (
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


# ---------------------------------------------------------------------------
# MMDetection class names + caption_prompt rewrites
#
# Sourced verbatim from
# explanations/server/odinw/odinw.py (the MMDet Grounding-DINO ODinW-13 val
# config). The class_name tuple is the per-domain category list MMDet uses;
# its element at position ``i`` corresponds to COCO category_id ``i + 1``
# (verified for AerialMaritimeDrone / CottontailRabbits / Raccoon /
# ShellfishOpenImages / VehiclesOpenImages; ``_validate_against_coco`` below
# guards the rest at load time).
#
# caption_prompt encodes MMDet's per-class rewrite. Composition is
# ``f"{prefix or ''}{name or class_name}{suffix or ''}"``. Domains absent
# from CAPTION_PROMPT_BY_DOMAIN (or with caption_prompt = None) keep their
# raw class names.
# ---------------------------------------------------------------------------

MMDET_CLASS_NAMES: Dict[str, tuple] = {
    'AerialMaritimeDrone_large': ('boat', 'car', 'dock', 'jetski', 'lift'),
    'Aquarium_Aquarium_Combined.v2-raw-1024.coco': (
        'fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray',
    ),
    'CottontailRabbits': ('Cottontail-Rabbit',),
    'EgoHands_generic': ('hand',),
    'NorthAmericaMushrooms_North_American_Mushrooms.v1-416x416.coco': (
        'CoW', 'chanterelle',
    ),
    'Packages_Raw': ('package',),
    'PascalVOC': (
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
        'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
        'tvmonitor',
    ),
    'Raccoon_Raccoon.v2-raw.coco': ('raccoon',),
    'ShellfishOpenImages_raw': ('Crab', 'Lobster', 'Shrimp'),
    'VehiclesOpenImages_416x416': ('Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck'),
    'pistols_export': ('pistol',),
    'pothole': ('pothole',),
    'thermalDogsAndPeople': ('dog', 'person'),
}

CAPTION_PROMPT_BY_DOMAIN: Dict[str, Dict[str, Dict[str, str]]] = {
    'CottontailRabbits': {
        'Cottontail-Rabbit': {'name': 'rabbit'},
    },
    'EgoHands_generic': {
        'hand': {'suffix': ' of a person'},
    },
    'NorthAmericaMushrooms_North_American_Mushrooms.v1-416x416.coco': {
        'CoW': {'name': 'flat mushroom'},
        'chanterelle': {'name': 'yellow mushroom'},
    },
    'Packages_Raw': {
        'package': {'prefix': 'there is a ', 'suffix': ' on the porch'},
    },
    'pothole': {
        'pothole': {'prefix': 'there are some ', 'name': 'holes', 'suffix': ' on the road'},
    },
}


def _apply_caption_prompt(class_name: str, cap_for_domain: Optional[Dict[str, Dict[str, str]]]) -> str:
    """Compose MMDet's caption_prompt rewrite. Returns ``class_name`` unchanged
    if the domain has no rewrite for this class."""
    if not cap_for_domain or class_name not in cap_for_domain:
        return class_name
    rule = cap_for_domain[class_name]
    return f"{rule.get('prefix', '')}{rule.get('name', class_name)}{rule.get('suffix', '')}"


def _validate_against_coco(domain_name: str, coco_categories: List[dict]) -> None:
    """Sanity check: the MMDet tuple must line up with sorted COCO category_ids.

    Logs a warning (rather than failing) if there's a mismatch — the user can
    decide whether to proceed.
    """
    mmdet = MMDET_CLASS_NAMES.get(domain_name)
    if mmdet is None:
        log.warning(f'  [{domain_name}] No MMDet class_name tuple registered; using raw COCO names.')
        return
    cats_sorted = sorted(coco_categories, key=lambda c: int(c['id']))
    coco_names = tuple(c['name'] for c in cats_sorted)
    if len(coco_names) != len(mmdet):
        log.warning(
            f'  [{domain_name}] MMDet tuple has {len(mmdet)} entries but COCO has '
            f'{len(coco_names)} categories: MMDet={mmdet} COCO={coco_names}'
        )
        return
    if coco_names != mmdet:
        log.warning(
            f'  [{domain_name}] MMDet tuple differs from COCO category names: '
            f'MMDet={mmdet} COCO={coco_names}'
        )


def _mmdet_class_name(domain_name: str, category_id: int, fallback: str) -> str:
    """Look up the MMDet class name for a (domain, category_id). Falls back to
    ``fallback`` (the COCO/GLIP name) if the domain or index is out of range.
    Category IDs are assumed 1-based and consecutive (standard ODinW)."""
    mmdet = MMDET_CLASS_NAMES.get(domain_name)
    if mmdet is None:
        return fallback
    idx = int(category_id) - 1
    if idx < 0 or idx >= len(mmdet):
        return fallback
    return mmdet[idx]


def build_qwen25_question(class_name: str) -> str:
    """Qwen2.5-VL ODinW evaluation prompt template.

    Source: jefferyZhan's reproduction in QwenLM/Qwen3-VL issue #839, not
    contested by HumanZhong. The model is expected to emit pixel-coordinate
    bbox_2d JSON in response.
    """
    return f"Locate every {class_name} in the image and output the coordinates in JSON format."


def process_domain_split_qwen25(
    domain_name: str,
    split_key: str,
    split_info: dict,
    class_names_mode: str,
    prompt_format: str,
) -> List[dict]:
    """Emit positives-only rows for one (domain, split) under Qwen's protocol.

    Parameters
    ----------
    class_names_mode : {'mmdet', 'raw'}
        ``mmdet`` applies MMDet's caption_prompt rewrites. ``raw`` keeps
        MMDet's raw class_name tuple verbatim (no rewrites).
    prompt_format : {'qwen25', 'adapted'}
        ``qwen25`` uses the Qwen2.5 single-sentence prompt with JSON output.
        ``adapted`` uses this repo's existing 0-1000 normalized list prompt.
    """
    ann_file = split_info['ann_file']

    with open(ann_file) as f:
        coco = json.load(f)

    images_by_id = {img['id']: img for img in coco.get('images', [])}
    coco_categories = list(coco.get('categories', []))
    _validate_against_coco(domain_name, coco_categories)

    # Map category_id -> fallback (COCO/GLIP) name. We use this as the input
    # to ``_mmdet_class_name`` so a missing MMDet entry falls back gracefully.
    coco_name_by_id = {int(c['id']): c['name'] for c in coco_categories}

    # Group annotations by (image_id, category_id) for O(1) positive lookup.
    groups: Dict[tuple, list] = defaultdict(list)
    for ann in coco.get('annotations', []):
        groups[(int(ann['image_id']), int(ann['category_id']))].append(ann)

    cap_for_domain = CAPTION_PROMPT_BY_DOMAIN.get(domain_name)

    rows: List[dict] = []
    # Iterate positives-only: one row per (image, present category).
    for (image_id, cat_id), anns in sorted(groups.items()):
        img = images_by_id.get(image_id)
        if img is None:
            log.warning(f'  [{domain_name}/{split_key}] annotation references unknown image_id={image_id}; skipping')
            continue
        img_w = img['width']
        img_h = img['height']
        image_path = f'{domain_name}/{img["file_name"]}'

        # Class name resolution:
        #   1. Start from the MMDet raw tuple (or COCO name if domain unmapped).
        #   2. In mmdet mode, compose the caption_prompt rewrite on top.
        raw_class = _mmdet_class_name(
            domain_name, cat_id, coco_name_by_id.get(cat_id, f'cat_{cat_id}')
        )
        if class_names_mode == 'mmdet':
            display_class = _apply_caption_prompt(raw_class, cap_for_domain)
        elif class_names_mode == 'raw':
            display_class = raw_class
        else:
            raise ValueError(f'Unknown class_names_mode: {class_names_mode!r}')

        if prompt_format == 'qwen25':
            question = build_qwen25_question(display_class)
        elif prompt_format == 'adapted':
            question = build_bbox_question(display_class)
        else:
            raise ValueError(f'Unknown prompt_format: {prompt_format!r}')

        norm_bboxes = [
            bbox_xywh_to_normalised_xyxy(a['bbox'], img_w, img_h) for a in anns
        ]
        answer_str = json.dumps(norm_bboxes)
        ann_ids_str = ','.join(str(a['id']) for a in anns)
        areas_str = json.dumps([round(a.get('area', 0), 2) for a in anns])

        rows.append({
            'index': f'{domain_name}_{split_key}_{image_id}_{cat_id}',
            'image_path': image_path,
            'question': question,
            'answer': answer_str,
            'category_id': int(cat_id),
            'category_name': display_class,
            'image_width': img_w,
            'image_height': img_h,
            'num_instances': len(anns),
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
    class_names_mode: str,
    prompt_format: str,
) -> Optional[str]:
    """Emit one positives-only TSV for one split under Qwen's protocol."""
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
        log.info(f'  Processing {domain_name}/{split_key} (positives-only)...')
        rows = process_domain_split_qwen25(
            domain_name, split_key, splits_dict[split_key],
            class_names_mode=class_names_mode,
            prompt_format=prompt_format,
        )
        log.info(f'    -> {len(rows)} positive (image, category) pairs')
        all_rows.extend(rows)

    if not all_rows:
        log.warning(f'  [{split_key}] No rows emitted; nothing to write.')
        return None

    def _sort_key(r):
        return (
            domain_order.get(r['domain'], 9999),
            int(r['index'].rsplit('_', 2)[-2]),
            int(r['category_id']),
        )
    all_rows.sort(key=_sort_key)

    # Filename: _qwen25 always present (this is the Qwen2.5 protocol).
    # _adapted / _raw appended (in that order) to mark deviations from the
    # canonical default (mmdet classes + qwen25 prompt).
    suffix_parts = ['qwen25']
    if prompt_format == 'adapted':
        suffix_parts.append('adapted')
    if class_names_mode == 'raw':
        suffix_parts.append('raw')
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
        description='Prepare ODinW-13 positives-only TSVs under the Qwen2.5-VL evaluation protocol.',
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
        '--class-names', dest='class_names_mode', default='mmdet',
        choices=['mmdet', 'raw'],
        help='Class-name treatment. "mmdet" applies MMDet caption_prompt '
             'rewrites; "raw" keeps the raw MMDet class_name tuple. Default: mmdet.',
    )
    parser.add_argument(
        '--prompt-format', default='qwen25',
        choices=['qwen25', 'adapted'],
        help='Prompt template. "qwen25" uses Qwen2.5\'s single-sentence prompt; '
             '"adapted" uses this repo\'s verbose 0-1000-normalized prompt. '
             'Default: qwen25.',
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
    log.info(f'Step 2: Generating positives-only TSVs '
             f'(class-names={args.class_names_mode}, prompt={args.prompt_format})')
    log.info('=' * 60)
    written: List[str] = []
    for split_key in args.splits:
        out_path = generate_split_tsv(
            domains_13, output_dir, 'ODinW13', split_key,
            class_names_mode=args.class_names_mode,
            prompt_format=args.prompt_format,
        )
        if out_path:
            written.append(out_path)

    log.info('=' * 60)
    log.info('Done!')
    for p in written:
        log.info(f'  TSV: {p}')


if __name__ == '__main__':
    main()
