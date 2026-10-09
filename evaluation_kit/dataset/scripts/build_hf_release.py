#!/usr/bin/env python3
"""Build the HuggingFace dataset repo that ``hf_release`` downloads from.

Packs the processed TSVs for the counting / grounding splits in ``hf_release.HF_SPLITS``, the
images they reference, and the ODinW-13 GT files into a staging directory::

    <out>/<split>.tsv            copied verbatim from $LMUData
    <out>/images/<folder>.zip    only the images referenced by the splits
    <out>/GLIP.zip               GLIP/configs/odinw_13/*.yaml (val/test entries)
                                 + GLIP/DATASET/<their ann_file>
    <out>/README.md              dataset card

and prints the ``HF_DATASET_MD5`` dict to paste into ``hf_release.py``.
With ``--upload`` the staging directory is pushed to ``hf_release.HF_REPO_ID``, except
README.md: the repo's card also covers splits added by other scripts, so edit it there.

Usage:
    python -m evaluation_kit.dataset.scripts.build_hf_release \\
        --glip-root $GLIP_ROOT [--lmu-data $LMUData] \\
        [--out outputs/hf_release] [--upload [--private]]
"""

import argparse
import hashlib
import json
import os
import os.path as osp
import shutil
import zipfile
from collections import defaultdict

import pandas as pd
import yaml

from evaluation_kit.dataset.hf_release import HF_REPO_ID, HF_SPLITS as _ALL_HF_SPLITS
from evaluation_kit.dataset.odinw import ODINW_13_CONFIGS
from evaluation_kit.smp.misc import toliststr

# Only the splits this script builds; the rest of HF_SPLITS comes from backup_eval_datasets.py.
HF_SPLITS = {s: f for s, f in _ALL_HF_SPLITS.items() if f in ('FSC147', 'LVIS', 'ODinW')}


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_zip(zip_path, files):
    """files: iterable of (src_abs_path, arcname). JPEGs are stored, not deflated."""
    os.makedirs(osp.dirname(zip_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
        for src, arc in sorted(files, key=lambda x: x[1]):
            zf.write(src, arc)


def build_images(lmu, out, split_rows):
    per_folder = defaultdict(set)
    for split, folder in HF_SPLITS.items():
        df = pd.read_csv(osp.join(lmu, f'{split}.tsv'), sep='\t')
        split_rows[split] = len(df)
        for x in df['image_path']:
            per_folder[folder].update(toliststr(x))
    for folder, paths in per_folder.items():
        files = []
        for p in paths:
            src = osp.join(lmu, 'images', folder, p)
            if not osp.exists(src):
                raise FileNotFoundError(src)
            files.append((src, f'{folder}/{p}'))
        write_zip(osp.join(out, 'images', f'{folder}.zip'), files)
        print(f'images/{folder}.zip: {len(files)} images')
    return {f: len(p) for f, p in per_folder.items()}


def build_glip(glip_root, out):
    files = []
    stage = osp.join(out, '_glip_configs')
    os.makedirs(stage, exist_ok=True)
    for domain in ODINW_13_CONFIGS:
        with open(osp.join(glip_root, 'configs', 'odinw_13', f'{domain}.yaml')) as f:
            cfg = yaml.safe_load(f)
        register = cfg['DATASETS']['REGISTER']
        # Only the eval splits; other entries would point at files we don't ship.
        cfg['DATASETS']['REGISTER'] = {k: register[k] for k in ('val', 'test')}
        cfg_path = osp.join(stage, f'{domain}.yaml')
        with open(cfg_path, 'w') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        files.append((cfg_path, f'GLIP/configs/odinw_13/{domain}.yaml'))
        for split in ('val', 'test'):
            ann = register[split]['ann_file']
            arc = f'GLIP/DATASET/{ann}'
            if arc not in {a for _, a in files}:
                files.append((osp.join(glip_root, 'DATASET', ann), arc))
    write_zip(osp.join(out, 'GLIP.zip'), files)
    shutil.rmtree(stage)
    print(f'GLIP.zip: {len(files)} files')


CARD = """---
license: other
pretty_name: mini-vlm-toolkit counting & grounding benchmarks
task_categories:
- visual-question-answering
- object-detection
configs:
{configs}
---

# mini-vlm-toolkit counting & grounding benchmarks

Processed evaluation splits used by
[mini-vlm-toolkit](https://github.com/tool-vision/mini-vlm-toolkit).
Each `<split>.tsv` holds the exact prompts (`question`) and annotations
(`answer` plus metadata) we evaluate on; `image_path` is relative to
`images/<folder>/` inside `images/<folder>.zip`. `GLIP.zip` holds the ODinW-13
configs and COCO-format val/test annotations used for ODinW AP evaluation.

You normally don't need to download anything by hand: running
`python run_benchmark.py --data <split> ...` fetches the TSV, the images and
(for ODinW) the GT files into `$LMUData` on first use.

| Split | Rows | Image archive |
|---|---|---|
{table}

## Sources, licenses and citations

This repo only re-packages existing public benchmarks; all credit goes to the
original authors, and use is subject to the original licenses/terms.

- **FSC147** — Ranjan et al., *Learning To Count Everything*, CVPR 2021
  ([code/data](https://github.com/cvlab-stonybrook/LearningToCountEverything)).
- **FSCD-147 / FSCD-LVIS** — Nguyen et al., *Few-shot Object Counting and
  Detection*, ECCV 2022 ([code/data](https://github.com/VinAIResearch/Counting-DETR)).
  FSCD-LVIS images are from COCO / LVIS (Gupta et al., CVPR 2019) and follow
  their terms of use.
- **ODinW-13** — Li et al., *Grounded Language-Image Pre-training (GLIP)*,
  CVPR 2022, and Li et al., *ELEVATER*, NeurIPS 2022. Domains originate from
  Roboflow public datasets and PASCAL VOC and follow their respective licenses.

If you are an author of one of these datasets and would like it removed, please
open a discussion on this repo.
"""


def write_card(out, split_rows):
    configs = '\n'.join(
        f'- config_name: {s}\n  data_files: {s}.tsv\n  sep: "\\t"' for s in HF_SPLITS)
    table = '\n'.join(
        f'| {s} | {split_rows[s]} | images/{HF_SPLITS[s]}.zip |' for s in HF_SPLITS)
    with open(osp.join(out, 'README.md'), 'w') as f:
        f.write(CARD.format(configs=configs, table=table))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--lmu-data', default=os.environ.get('LMUData'))
    ap.add_argument('--glip-root', default=os.environ.get('GLIP_ROOT'))
    ap.add_argument('--out', default='outputs/hf_release')
    ap.add_argument('--upload', action='store_true')
    ap.add_argument('--private', action='store_true')
    args = ap.parse_args()
    if not args.lmu_data or not args.glip_root:
        ap.error('--lmu-data and --glip-root (or $LMUData / $GLIP_ROOT) are required')

    os.makedirs(args.out, exist_ok=True)
    split_rows = {}
    build_images(args.lmu_data, args.out, split_rows)
    build_glip(args.glip_root, args.out)

    md5s = {}
    for split in HF_SPLITS:
        dst = osp.join(args.out, f'{split}.tsv')
        shutil.copyfile(osp.join(args.lmu_data, f'{split}.tsv'), dst)
        md5s[split] = md5(dst)
    write_card(args.out, split_rows)
    print('HF_DATASET_MD5 = ' + json.dumps(md5s, indent=4))

    if args.upload:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(HF_REPO_ID, repo_type='dataset', private=args.private, exist_ok=True)
        api.upload_folder(repo_id=HF_REPO_ID, repo_type='dataset', folder_path=args.out,
                          ignore_patterns=['README.md'],
                          commit_message='Upload processed counting/grounding splits')
        print(f'Uploaded to https://huggingface.co/datasets/{HF_REPO_ID}')


if __name__ == '__main__':
    main()
