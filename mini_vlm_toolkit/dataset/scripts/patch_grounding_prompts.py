#!/usr/bin/env python3
"""One-shot in-place patcher: rewrite grounding `question` text in TSVs.

Updates LVIS_bbox_*.tsv, ODinW13_bbox_*.tsv, and ODinW35_bbox_*.tsv
under ``$LMUData`` (or the path passed via ``--lmu-data``) to match
the canonical prompt produced by
``mini_vlm_toolkit.dataset.scripts.prepare_odinw.build_bbox_question``.

Idempotent: matches both the legacy "JSON array of arrays" wording and
the prior `_full` variant that already had the "if empty" hint, and
re-emits the canonical text in either case.

Mask TSVs (``LVIS_mask_*.tsv``) are explicitly skipped — they use a
different output schema (polygons / RLE) handled by
``build_mask_question``.
"""

import argparse
import glob
import logging
import os
import re

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)


OLD_PROMPT_RE = re.compile(
    r"(Locate all instances of '([^']+)' in this image\. .*?bottom-right corner\)\.) "
    r"Output a JSON array of arrays\."
    r"(?: If there are no instances of '\2' in the image, output an empty array \[\]\.)?",
    re.DOTALL,
)


def _replace(match: re.Match) -> str:
    cat = match.group(2)
    return (
        f"{match.group(1)} Your answer should be a list of lists. "
        f"If there are no instances of '{cat}' in the image, "
        f"output an empty list [] as your answer."
    )


def patch_file(path: str) -> int:
    """Patch one TSV in-place. Returns the number of rows whose `question`
    column actually changed."""
    df = pd.read_csv(path, sep='\t')
    if 'question' not in df.columns:
        return 0
    before = df['question'].copy()
    df['question'] = df['question'].astype(str).apply(
        lambda q: OLD_PROMPT_RE.sub(_replace, q)
    )
    changed = int((before != df['question']).sum())
    if changed:
        df.to_csv(path, sep='\t', index=False)
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--lmu-data',
        default=os.environ.get('LMUData', '/local/<user>/tool-vision/lmu_data'),
        help='Path to the LMUData root (default: $LMUData).',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Report what would be patched without writing.',
    )
    args = parser.parse_args()

    patterns = ['LVIS_bbox_*.tsv', 'ODinW13_bbox_*.tsv', 'ODinW35_bbox_*.tsv']
    paths = []
    for p in patterns:
        paths.extend(glob.glob(os.path.join(args.lmu_data, p)))
    paths = sorted(p for p in paths if 'mask' not in os.path.basename(p))

    log.info(f'Found {len(paths)} candidate TSVs under {args.lmu_data}')
    total_changed = 0
    for path in paths:
        if args.dry_run:
            df = pd.read_csv(path, sep='\t')
            if 'question' not in df.columns:
                log.info(f'  [skip - no question col] {os.path.basename(path)}')
                continue
            new_q = df['question'].astype(str).apply(
                lambda q: OLD_PROMPT_RE.sub(_replace, q)
            )
            n = int((df['question'] != new_q).sum())
            log.info(f'  [{n:>6} rows would change] {os.path.basename(path)}')
            total_changed += n
        else:
            n = patch_file(path)
            log.info(f'  [{n:>6} rows changed]       {os.path.basename(path)}')
            total_changed += n

    verb = 'would change' if args.dry_run else 'changed'
    log.info(f'Total rows {verb}: {total_changed}')


if __name__ == '__main__':
    main()
