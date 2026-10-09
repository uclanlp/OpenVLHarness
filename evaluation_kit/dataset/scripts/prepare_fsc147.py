#!/usr/bin/env python3
"""Prepare FSC-147 dataset TSV files for OpenVLHarness evaluation.

Usage:
    python -m evaluation_kit.dataset.scripts.prepare_fsc147 \
        --desc-json /path/to/FSC-147-D.json \
        --anno-json /path/to/annotation_FSC147_384.json \
        --image-dir /path/to/images_384_VarV2 \
        [--output-dir $LMUData]

The script:
  1. Copies all source images to ${LMUData}/images/FSC147/
  2. Generates three TSV files (FSC147_train.tsv, FSC147_val.tsv, FSC147_test.tsv)
     with columns: index, image_path, question, answer
"""

import argparse
import json
import os
import shutil

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Prepare FSC-147 dataset TSVs')
    parser.add_argument('--desc-json', required=True,
                        help='Path to FSC-147-D.json (split + text descriptions)')
    parser.add_argument('--anno-json', required=True,
                        help='Path to annotation_FSC147_384.json (point annotations)')
    parser.add_argument('--image-dir', required=True,
                        help='Path to source image directory (e.g. images_384_VarV2)')
    parser.add_argument('--output-dir', default=None,
                        help='Output root directory (default: $LMUData)')
    args = parser.parse_args()

    output_dir = args.output_dir or os.environ.get('LMUData', None)
    if output_dir is None:
        raise ValueError('Either --output-dir or $LMUData environment variable must be set')

    # Load JSON files
    with open(args.desc_json, 'r') as f:
        desc_data = json.load(f)
    with open(args.anno_json, 'r') as f:
        anno_data = json.load(f)

    # Copy images to ${LMUData}/images/FSC147/
    dst_image_dir = os.path.join(output_dir, 'images', 'FSC147')
    if not os.path.exists(dst_image_dir):
        print(f'Copying images from {args.image_dir} to {dst_image_dir} ...')
        shutil.copytree(args.image_dir, dst_image_dir)
        print('Done copying images.')
    else:
        print(f'Image directory {dst_image_dir} already exists, skipping copy.')

    # Build split -> list of records
    split_map = {'train': 'FSC147_train', 'val': 'FSC147_val', 'test': 'FSC147_test'}
    split_records = {s: [] for s in split_map.values()}

    for image_name, info in desc_data.items():
        split = info.get('data_split', None)
        if split not in split_map:
            print(f'Warning: unknown split "{split}" for {image_name}, skipping')
            continue

        dataset_name = split_map[split]

        # Get point annotations to compute ground-truth count
        points = anno_data.get(image_name, {}).get('points', [])
        gt_count = len(points)

        # Build question from text description
        text_desc = info.get('text_description_singular', info.get('text_description', ''))
        # Use singular form for the question
        question = (
            f'How many {text_desc} are there in the image? '
            f'Your final answer should be a single integer.'
        )

        split_records[dataset_name].append({
            'image_path': image_name,
            'question': question,
            'answer': gt_count,
        })

    # Write TSV files
    for dataset_name, records in split_records.items():
        if not records:
            print(f'Warning: no records for {dataset_name}')
            continue

        df = pd.DataFrame(records)
        df.insert(0, 'index', range(len(df)))
        tsv_path = os.path.join(output_dir, f'{dataset_name}.tsv')
        df.to_csv(tsv_path, sep='\t', index=False)
        print(f'Wrote {len(df)} records to {tsv_path}')


if __name__ == '__main__':
    main()
