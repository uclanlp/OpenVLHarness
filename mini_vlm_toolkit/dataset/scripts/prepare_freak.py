#!/usr/bin/env python3
"""Prepare FREAK dataset TSV files for VLMEvalKit evaluation.

Usage:
    python prepare_freak.py [--output-dir $LMUData]

The script:
  1. Downloads the FREAK dataset from HuggingFace (hansQAQ/FREAK, test split)
  2. Saves all images to ${LMUData}/images/FREAK/
  3. Splits examples by type ("mcq" vs "fib")
  4. Shuffles MCQ options (seed=42) so the correct answer isn't always A
  5. Generates FREAK_MCQ.tsv and FREAK_VQA.tsv with a pre-built "prompt" column
"""

import argparse
import json
import os
import random
import string

import pandas as pd
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description='Prepare FREAK dataset TSVs')
    parser.add_argument('--output-dir', default=None,
                        help='Output root directory (default: $LMUData)')
    args = parser.parse_args()

    output_dir = args.output_dir or os.environ.get('LMUData', None)
    if output_dir is None:
        raise ValueError('Either --output-dir or $LMUData environment variable must be set')

    # Create image directory
    dst_image_dir = os.path.join(output_dir, 'images', 'FREAK')
    os.makedirs(dst_image_dir, exist_ok=True)

    # Load dataset from HuggingFace
    print('Loading FREAK dataset from HuggingFace...')
    ds = load_dataset('hansQAQ/FREAK', split='test')
    print(f'Loaded {len(ds)} examples')

    rng = random.Random(42)

    mcq_records = []
    vqa_records = []
    mcq_idx = 0
    vqa_idx = 0

    for example in ds:
        example_id = example['id']
        img = example['image']
        question = example['question']
        ground_truth = example['ground_truth']
        options = example['options']
        example_type = example['type']
        category_list = example.get('category', [])

        # Save image
        img_filename = f'{example_id}.jpg'
        img_path = os.path.join(dst_image_dir, img_filename)
        if not os.path.exists(img_path):
            img.save(img_path)

        # Store categories as JSON list
        category_str = json.dumps(category_list) if category_list else '[]'

        # Verify type/options consistency
        if example_type == 'mcq' and options is None:
            print(f'Warning: id={example_id} has type="mcq" but options is None, skipping')
            continue
        if example_type == 'fib' and options is not None:
            print(f'Warning: id={example_id} has type="fib" but options is not None, treating as VQA')

        if example_type == 'mcq':
            # Shuffle options so the correct answer is not always in the same position
            shuffled = list(options)
            rng.shuffle(shuffled)

            # Find which letter the ground_truth maps to after shuffling
            answer_letter = None
            for i, opt in enumerate(shuffled):
                if opt.strip() == ground_truth.strip():
                    answer_letter = string.ascii_uppercase[i]
                    break

            if answer_letter is None:
                # Fallback: case-insensitive match
                for i, opt in enumerate(shuffled):
                    if opt.strip().lower() == ground_truth.strip().lower():
                        answer_letter = string.ascii_uppercase[i]
                        break

            if answer_letter is None:
                print(f'Warning: id={example_id} ground_truth "{ground_truth}" '
                      f'does not match any option {options}, skipping')
                continue

            # Build prompt: question + formatted options
            option_parts = [f'{string.ascii_uppercase[i]}. {opt}'
                            for i, opt in enumerate(shuffled)]
            prompt = f'{question} Choose one option from the following: {", ".join(option_parts)}'

            record = {
                'index': mcq_idx,
                'image_path': img_filename,
                'question': question,
                'prompt': prompt,
                'mcq_GT': answer_letter,
                'choices': json.dumps(shuffled),
                'text_GT': ground_truth,
                'category': category_str,
            }
            mcq_records.append(record)
            mcq_idx += 1

        elif example_type == 'fib':
            prompt = (f'{question} Provide your final answer in a clear and '
                      f'concise sentence or a few words.')
            record = {
                'index': vqa_idx,
                'image_path': img_filename,
                'question': question,
                'prompt': prompt,
                'GT': ground_truth,
                'category': category_str,
            }
            vqa_records.append(record)
            vqa_idx += 1

        else:
            print(f'Warning: id={example_id} has unknown type "{example_type}", skipping')

    # Write MCQ TSV
    if mcq_records:
        columns = ['index', 'image_path', 'question', 'prompt', 'mcq_GT',
                    'choices', 'text_GT', 'category']
        mcq_df = pd.DataFrame(mcq_records, columns=columns)
        mcq_path = os.path.join(output_dir, 'FREAK_MCQ.tsv')
        mcq_df.to_csv(mcq_path, sep='\t', index=False)
        print(f'Wrote {len(mcq_df)} MCQ records to {mcq_path}')

        # Print answer distribution to verify shuffling
        dist = mcq_df['mcq_GT'].value_counts().sort_index()
        print(f'Answer distribution: {dict(dist)}')
    else:
        print('Warning: no MCQ records found')

    # Write VQA TSV
    if vqa_records:
        columns = ['index', 'image_path', 'question', 'prompt', 'GT', 'category']
        vqa_df = pd.DataFrame(vqa_records, columns=columns)
        vqa_path = os.path.join(output_dir, 'FREAK_VQA.tsv')
        vqa_df.to_csv(vqa_path, sep='\t', index=False)
        print(f'Wrote {len(vqa_df)} VQA records to {vqa_path}')
    else:
        print('Warning: no VQA records found')

    print('Done!')


if __name__ == '__main__':
    main()
