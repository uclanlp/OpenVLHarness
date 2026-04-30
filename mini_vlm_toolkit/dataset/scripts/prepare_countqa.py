#!/usr/bin/env python3
"""Prepare the CountQA benchmark (Jayant-Sravan/CountQA) for mini-vlm-toolkit.

CountQA is hosted on HuggingFace at ``Jayant-Sravan/CountQA`` and has a single
``test`` split.  The HF schema stores list-valued ``questions`` and
``answers`` (one HF row per image, 1–28 QA pairs each) plus per-image metadata
(``objects``, ``categories``, ``is_focused``, ``full_config``).

This script flattens the dataset to one TSV row per (image, question, answer)
triple — the format the rest of the framework expects.  ``objects``,
``categories``, and ``is_focused`` are repeated across an image's expanded
rows; ``full_config`` is dropped.

Usage:
    python -m mini_vlm_toolkit.dataset.scripts.prepare_countqa \
        [--output-dir $LMUData] \
        [--hf-cache-dir <path>]

Outputs (under ``$LMUData`` or ``--output-dir``):
    images/CountQA/0000.jpg ... 1000.jpg   (one JPG per source image)
    CountQA.tsv                            (flattened QA rows)

TSV columns
-----------
    index        unique row id, "countqa_{i:04d}_q{q_idx}"
    image_path   filename inside $LMUData/images/CountQA/  (e.g. 0042.jpg)
    question     CountQA question text + " Your final answer should be a single integer."
    answer       ground-truth count (string, e.g. "18")
    objects      JSON list of object names mentioned in the source image
    categories   JSON list of category labels
    is_focused   bool, repeated across an image's expanded rows
"""

import argparse
import json
import logging
import os
import sys

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


HF_DATASET_ID = 'Jayant-Sravan/CountQA'
HF_SPLIT = 'test'
DATASET_NAME = 'CountQA'
QUESTION_SUFFIX = ' Your final answer should be a single integer.'


def _resolve_cache_dir(arg_cache_dir, output_dir):
    if arg_cache_dir:
        return arg_cache_dir
    env = os.environ.get('HF_DATASETS_CACHE_DIR')
    if env:
        return env
    return os.path.join(output_dir, 'hf_cache', 'datasets')


def main():
    parser = argparse.ArgumentParser(
        description='Prepare CountQA TSV for mini-vlm-toolkit.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--output-dir', default=None,
        help='Output root (default: $LMUData).',
    )
    parser.add_argument(
        '--hf-cache-dir', default=None,
        help='HuggingFace datasets cache dir '
             '(default: $HF_DATASETS_CACHE_DIR or $LMUData/hf_cache/datasets).',
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.environ.get('LMUData', None)
    if output_dir is None:
        log.error('Either --output-dir or $LMUData environment variable must be set.')
        sys.exit(1)

    cache_dir = _resolve_cache_dir(args.hf_cache_dir, output_dir)
    os.makedirs(cache_dir, exist_ok=True)

    image_dir = os.path.join(output_dir, 'images', DATASET_NAME)
    os.makedirs(image_dir, exist_ok=True)

    log.info(f'Loading {HF_DATASET_ID} (split={HF_SPLIT}) ...')
    from datasets import load_dataset
    ds = load_dataset(HF_DATASET_ID, split=HF_SPLIT, cache_dir=cache_dir)
    log.info(f'  {len(ds)} HF rows')

    # ------------------------------------------------------------------
    # Save images and flatten QA pairs
    # ------------------------------------------------------------------
    rows = []
    images_saved = 0
    images_skipped = 0
    is_focused_count = 0
    qpc = []  # questions-per-image list, for summary

    for i, row in enumerate(ds):
        questions = row['questions']
        answers = row['answers']
        assert len(questions) == len(answers), (
            f'row {i}: len(questions)={len(questions)} != len(answers)={len(answers)}'
        )
        qpc.append(len(questions))

        image_filename = f'{i:04d}.jpg'
        image_path_abs = os.path.join(image_dir, image_filename)
        if os.path.exists(image_path_abs):
            images_skipped += 1
        else:
            img = row['image']
            # HF stores as PIL Image with original mode (often RGB; some PNGs
            # arrive as RGBA — convert before JPEG-encoding).
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(image_path_abs, 'JPEG', quality=95)
            images_saved += 1

        if row['is_focused']:
            is_focused_count += 1

        objects_json = json.dumps(row['objects'])
        categories_json = json.dumps(row['categories'])
        is_focused = bool(row['is_focused'])

        for q_idx, (q, a) in enumerate(zip(questions, answers)):
            q_clean = (q or '').rstrip()
            assert q_clean, f'row {i} q{q_idx}: empty question'
            rows.append({
                'index': f'countqa_{i:04d}_q{q_idx}',
                'image_path': image_filename,
                'question': q_clean + QUESTION_SUFFIX,
                'answer': str(a),
                'objects': objects_json,
                'categories': categories_json,
                'is_focused': is_focused,
            })

        if (i + 1) % 200 == 0:
            log.info(f'  processed {i+1}/{len(ds)} images ({len(rows)} QA rows so far)')

    # ------------------------------------------------------------------
    # Write TSV
    # ------------------------------------------------------------------
    tsv_path = os.path.join(output_dir, f'{DATASET_NAME}.tsv')
    df = pd.DataFrame(rows, columns=[
        'index', 'image_path', 'question', 'answer',
        'objects', 'categories', 'is_focused',
    ])
    df.to_csv(tsv_path, sep='\t', index=False)

    # ------------------------------------------------------------------
    # Validate every row's `answer` parses as an integer
    # ------------------------------------------------------------------
    from mini_vlm_toolkit.dataset.utils.fsc147 import extract_integer_from_prediction
    bad_answers = [
        (r['index'], r['answer']) for r in rows
        if extract_integer_from_prediction(r['answer']) is None
    ]
    if bad_answers:
        log.warning(f'{len(bad_answers)} rows had unparseable answers '
                    f'(first 5: {bad_answers[:5]})')

    # Confirm every referenced image exists on disk
    missing_imgs = [
        r['image_path'] for r in rows
        if not os.path.isfile(os.path.join(image_dir, r['image_path']))
    ]
    assert not missing_imgs, f'{len(missing_imgs)} rows reference missing images'

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    qpc_sorted = sorted(qpc)
    median_q = qpc_sorted[len(qpc_sorted) // 2]
    log.info('=' * 60)
    log.info('Done!')
    log.info(f'  images_saved={images_saved}  images_skipped={images_skipped}')
    log.info(f'  total_qa_rows={len(rows)}')
    log.info(f'  is_focused_true_images={is_focused_count} / {len(ds)}')
    log.info(f'  questions_per_image: min={min(qpc)} median={median_q} max={max(qpc)} '
             f'mean={sum(qpc)/len(qpc):.2f}')
    log.info(f'  bad_answer_rows={len(bad_answers)}')
    log.info(f'  Wrote {tsv_path}')
    log.info(f'  Images at {image_dir}/')
    log.info('=' * 60)


if __name__ == '__main__':
    main()
