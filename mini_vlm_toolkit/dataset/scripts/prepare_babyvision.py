"""Pre-build the BabyVision TSV.

Source: https://huggingface.co/datasets/UnipatAI/BabyVision

This is OPTIONAL. `BabyVision` builds its TSV automatically the first time it is
evaluated on a machine that does not have $LMUData/BabyVision.tsv yet. Use this
script only to warm the cache ahead of time -- e.g. on a node without outbound
network access at eval time, or to avoid paying the ~1 min build inside a timed run.

Usage:
    LMUData=/path/to/lmu_data python -m mini_vlm_toolkit.dataset.scripts.prepare_babyvision
"""
import argparse
import os.path as osp

from ...smp import LMUDataRoot
from ..babyvision import HF_REPO, build_babyvision_tsv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=HF_REPO)
    ap.add_argument('--out', default=None, help='output TSV (default: $LMUData/BabyVision.tsv)')
    ap.add_argument('--force', action='store_true', help='rebuild even if the TSV already exists')
    args = ap.parse_args()

    out = args.out or osp.join(LMUDataRoot(), 'BabyVision.tsv')
    if osp.exists(out) and not args.force:
        print(f'{out} already exists; pass --force to rebuild')
        return

    df = build_babyvision_tsv(out, repo=args.repo)
    print('categories:', df['category'].value_counts().to_dict())
    print('tasks:', df['task'].nunique())


if __name__ == '__main__':
    main()
