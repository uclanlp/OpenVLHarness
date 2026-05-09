#!/usr/bin/env python3
"""Prepare the Pixmo-Count benchmark (allenai/pixmo-count) for mini-vlm-toolkit.

Pixmo-Count is hosted on HuggingFace at ``allenai/pixmo-count`` with three
splits (``train``, ``validation``, ``test``).  The HF dataset stores image
*URLs* (Flickr ``live.staticflickr.com`` links) plus ``image_sha256``,
``label``, and ``count``.  The image bytes themselves are not redistributed,
so this script downloads each image from its URL and saves it to disk under
``$LMUData/images/PixmoCount/<sha256>.jpg``.

For each split the script emits two artifacts:

  PixmoCount_{split}.tsv          unified benchmark TSV (4 cols)
  _pixmo_download_logs/
      {split}_status.tsv          per-row download outcome (7 cols)

PixmoCount_{split}.tsv columns
------------------------------
    index        upstream HF row index (preserved across fail-row drops, so
                 there can be gaps; this matches the existing on-disk TSVs)
    image_path   "<sha256>.jpg" (basename only — resolved against
                 ``$LMUData/images/PixmoCount/`` by ``img_root_map``)
    question     "How many {label} are there in the image? If the same
                 instance appears more than once, it should be counted as
                 multiple instances. Your final answer should be a single
                 integer."
    answer       integer count (string-typed when written by pandas)

{split}_status.tsv columns
--------------------------
    index, image_url, image_sha256, label, count, status, dl_size

``status`` is one of:
    ok                  download succeeded and SHA256 matched upstream
    ok_sha_mismatch     download succeeded but SHA256 differs (Flickr often
                        re-encodes; we keep these — empirically the dominant
                        outcome)
    skip_existing       file already on disk; not re-downloaded
    fail                download or decode failed (row excluded from TSV)

Usage:
    python -m mini_vlm_toolkit.dataset.scripts.prepare_pixmo_count \\
        [--output-dir $LMUData] \\
        [--hf-cache-dir <path>] \\
        [--splits train validation test] \\
        [--num-workers 16] [--timeout 30] [--retries 2]
"""

import argparse
import csv
import hashlib
import io
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


HF_DATASET_ID = 'allenai/pixmo-count'
DATASET_NAME = 'PixmoCount'
# HF split name -> output TSV split tag.  HF uses "validation"; the unified
# TSV shipped with this repo uses "val".
SPLITS = {'train': 'train', 'validation': 'val', 'test': 'test'}
QUESTION_TEMPLATE = (
    'How many {label} are there in the image? '
    'If the same instance appears more than once, it should be counted as multiple instances. '
    'Your final answer should be a single integer.'
)

# Browser-ish UA — Flickr 403s on default Python UAs.
USER_AGENT = (
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/120.0 Safari/537.36'
)


def _resolve_cache_dir(arg_cache_dir, output_dir):
    if arg_cache_dir:
        return arg_cache_dir
    env = os.environ.get('HF_DATASETS_CACHE_DIR')
    if env:
        return env
    return os.path.join(output_dir, 'hf_cache', 'datasets')


def _download_one(url, dest_path, expected_sha, timeout, retries):
    """Fetch one image. Returns (status, dl_size). Writes file on success.

    On any non-2xx, network error, or PIL-decode failure we return
    ('fail', 0) and leave no file behind. SHA mismatch is non-fatal — most
    Flickr images re-encode and never match the upstream hash.
    """
    import requests
    from PIL import Image

    if os.path.exists(dest_path):
        return 'skip_existing', os.path.getsize(dest_path)

    headers = {'User-Agent': USER_AGENT}
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout, stream=False)
            if r.status_code != 200:
                last_err = f'http_{r.status_code}'
                # 4xx is terminal (gone/forbidden); only retry 5xx and timeouts
                if 400 <= r.status_code < 500:
                    break
                time.sleep(0.5 * (attempt + 1))
                continue
            content = r.content
            sha = hashlib.sha256(content).hexdigest()
            # Decode + re-encode as RGB JPEG so downstream code can rely on a
            # canonical format (some upstream URLs serve PNGs with alpha).
            try:
                img = Image.open(io.BytesIO(content))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
            except Exception as e:
                last_err = f'decode_{type(e).__name__}'
                break
            tmp_path = dest_path + '.tmp'
            img.save(tmp_path, 'JPEG', quality=95)
            os.replace(tmp_path, dest_path)
            status = 'ok' if sha == expected_sha else 'ok_sha_mismatch'
            return status, len(content)
        except Exception as e:
            last_err = f'{type(e).__name__}'
            time.sleep(0.5 * (attempt + 1))

    return 'fail', 0


def _process_split(hf_split, out_split, output_dir, image_dir, log_dir,
                   cache_dir, num_workers, timeout, retries):
    from datasets import load_dataset

    log.info(f'Loading {HF_DATASET_ID} (split={hf_split}) ...')
    ds = load_dataset(HF_DATASET_ID, split=hf_split, cache_dir=cache_dir)
    log.info(f'  {len(ds)} HF rows')

    # Materialize the metadata we need into a list of dicts so we don't keep
    # the HF dataset object alive across the threadpool.
    rows_meta = []
    for i, row in enumerate(ds):
        rows_meta.append({
            'index': i,
            'image_url': row['image_url'],
            'image_sha256': row['image_sha256'],
            'label': row['label'],
            'count': int(row['count']),
        })

    # ------------------------------------------------------------------
    # Parallel download
    # ------------------------------------------------------------------
    statuses = [None] * len(rows_meta)
    dl_sizes = [0] * len(rows_meta)
    done = 0
    log_every = max(50, len(rows_meta) // 50)

    def _task(meta):
        dest = os.path.join(image_dir, f"{meta['image_sha256']}.jpg")
        status, size = _download_one(
            meta['image_url'], dest, meta['image_sha256'], timeout, retries
        )
        return meta['index'], status, size

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(_task, m) for m in rows_meta]
        for fut in as_completed(futures):
            idx, status, size = fut.result()
            statuses[idx] = status
            dl_sizes[idx] = size
            done += 1
            if done % log_every == 0:
                ok = sum(1 for s in statuses if s in ('ok', 'ok_sha_mismatch', 'skip_existing'))
                log.info(f'  [{out_split}] downloaded {done}/{len(rows_meta)}  ok={ok}')

    # ------------------------------------------------------------------
    # Status TSV (mirrors the on-disk _pixmo_download_logs/*.tsv schema)
    # ------------------------------------------------------------------
    status_path = os.path.join(log_dir, f'{out_split}_status.tsv')
    status_rows = [
        {
            'index': m['index'],
            'image_url': m['image_url'],
            'image_sha256': m['image_sha256'],
            'label': m['label'],
            'count': m['count'],
            'status': statuses[m['index']],
            'dl_size': dl_sizes[m['index']],
        }
        for m in rows_meta
    ]
    pd.DataFrame(status_rows).to_csv(
        status_path, sep='\t', index=False,
        # Match the existing on-disk format: no quoting unless required.
        quoting=csv.QUOTE_MINIMAL,
    )

    # ------------------------------------------------------------------
    # Unified TSV — drop fail rows, preserve upstream index (gappy)
    # ------------------------------------------------------------------
    tsv_rows = []
    for m in rows_meta:
        st = statuses[m['index']]
        if st == 'fail':
            continue
        tsv_rows.append({
            'index': m['index'],
            'image_path': f"{m['image_sha256']}.jpg",
            'question': QUESTION_TEMPLATE.format(label=m['label']),
            'answer': m['count'],
        })

    tsv_path = os.path.join(output_dir, f'{DATASET_NAME}_{out_split}.tsv')
    pd.DataFrame(tsv_rows, columns=['index', 'image_path', 'question', 'answer']).to_csv(
        tsv_path, sep='\t', index=False,
    )

    n_ok = sum(1 for s in statuses if s == 'ok')
    n_mm = sum(1 for s in statuses if s == 'ok_sha_mismatch')
    n_skip = sum(1 for s in statuses if s == 'skip_existing')
    n_fail = sum(1 for s in statuses if s == 'fail')
    log.info(f'  [{out_split}] ok={n_ok}  ok_sha_mismatch={n_mm}  '
             f'skip_existing={n_skip}  fail={n_fail}')
    log.info(f'  [{out_split}] wrote {tsv_path} ({len(tsv_rows)} rows)')
    log.info(f'  [{out_split}] wrote {status_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Prepare PixmoCount TSVs for mini-vlm-toolkit.',
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
    parser.add_argument(
        '--splits', nargs='+', default=list(SPLITS.keys()),
        choices=list(SPLITS.keys()),
        help='HF splits to process (default: all three).',
    )
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--timeout', type=int, default=30,
                        help='Per-request timeout in seconds.')
    parser.add_argument('--retries', type=int, default=2,
                        help='Retry count for 5xx and network errors.')
    args = parser.parse_args()

    output_dir = args.output_dir or os.environ.get('LMUData', None)
    if output_dir is None:
        log.error('Either --output-dir or $LMUData environment variable must be set.')
        sys.exit(1)

    cache_dir = _resolve_cache_dir(args.hf_cache_dir, output_dir)
    os.makedirs(cache_dir, exist_ok=True)

    image_dir = os.path.join(output_dir, 'images', DATASET_NAME)
    os.makedirs(image_dir, exist_ok=True)

    log_dir = os.path.join(output_dir, '_pixmo_download_logs')
    os.makedirs(log_dir, exist_ok=True)

    for hf_split in args.splits:
        out_split = SPLITS[hf_split]
        _process_split(
            hf_split=hf_split,
            out_split=out_split,
            output_dir=output_dir,
            image_dir=image_dir,
            log_dir=log_dir,
            cache_dir=cache_dir,
            num_workers=args.num_workers,
            timeout=args.timeout,
            retries=args.retries,
        )

    log.info('=' * 60)
    log.info('Done!')
    log.info(f'  Images at {image_dir}/')
    log.info(f'  Status logs at {log_dir}/')
    log.info('=' * 60)


if __name__ == '__main__':
    main()
