#!/usr/bin/env python3
"""Back up general-VQA and spatial benchmarks to the shared HuggingFace dataset repo.

Adds these splits to ``HF_REPO_ID`` as one pull request (nothing else touched):

    RealWorldQA, HallusionBench, VStarBench                  TSV with embedded images
    MMSIBench_wo_circular, ERQA, SparBench(_tiny),
    EmbSpatialBench                                          TSV with embedded images
    FREAK_MCQ, FREAK_VQA                                     TSV + images/FREAK.zip
    MindCubeBench_tiny_raw_qa, MindCubeBench_raw_qa          TSV + images/MindCube.zip
    OmniSpatialBench                                         TSV + images/OmniSpatial.zip

Each TSV is taken from $LMUData when its MD5 matches the original source, otherwise
downloaded from the source; the uploaded TSV is byte-identical to what the dataset
classes downloaded before. FREAK is taken from $LMUData if present (else rebuilt from
HF). MindCube / OmniSpatial images are copied out of the original HF zips, keeping
only the images the TSVs reference. README.md rows and config entries are merged into
the current card; existing sections are left as they are.

This script does not import the dataset classes, so it keeps working after they are
switched to the backup.

Usage (`hf auth login` first; opening a PR only needs the org's contributor role,
and an org admin merges it):
    python -m openvlharness_eval.dataset.scripts.backup_eval_datasets \\
        [--lmu-data $LMUData] [--out outputs/hf_backup] [--upload]
"""

import argparse
import hashlib
import json
import os
import os.path as osp
import random
import shutil
import string
import zipfile

import pandas as pd
import requests

HF_REPO_ID = 'uclanlp/OpenVLHarness-Evaluation-Datasets'

EASI = 'https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main'
VLMEVAL = 'https://opencompass.openxlab.space/utils/VLMEval'

# split -> (original TSV URL, MD5 the dataset class checks)
TSV_SOURCES = {
    'RealWorldQA': (f'{VLMEVAL}/RealWorldQA.tsv', '4de008f55dc4fd008ca9e15321dc44b7'),
    'HallusionBench': (f'{VLMEVAL}/HallusionBench.tsv', '0c23ac0dc9ef46832d7a24504f2a0c7c'),
    'VStarBench': ('https://huggingface.co/datasets/xjtupanda/VStar_Bench/resolve/main/VStarBench.tsv',
                   'b18854d7075574be06b631cd5f7d2d6a'),
    'MMSIBench_wo_circular': (f'{EASI}/MMSIBench_wo_circular.tsv', '548c5f33f1a12948d5355d5f600749e4'),
    'ERQA': (f'{EASI}/ERQA.tsv', '40d45d4f0bb1852dbc68b28ee70d4ca5'),
    'SparBench': (f'{EASI}/SparBench.tsv', 'cb1e5f5855c454241e0a70c3b8152976'),
    'SparBench_tiny': (f'{EASI}/SparBench_tiny.tsv', 'c435e186064b795e0a9759aa60465a00'),
    'EmbSpatialBench': (f'{EASI}/EmbSpatialBench.tsv', 'a836cfd8fbe84bb42928ecef1e62ad32'),
    'MindCubeBench_tiny_raw_qa': (f'{EASI}/MindCubeBench_tiny_raw_qa.tsv', '35f69fc30d7c2d2880417ce0769f5347'),
    'MindCubeBench_raw_qa': (f'{EASI}/MindCubeBench_raw_qa.tsv', '6a53cd353bc93d8e3a87098249c806ad'),
    'OmniSpatialBench': (f'{EASI}/OmniSpatialBench.tsv', '5d0896fc57c452055b020cc309ed799b'),
}

# images/<folder>.zip -> (source HF dataset repo, zip file in it, prefix of the TSV
# paths inside that zip, splits whose image_path lists point into it)
IMAGE_SOURCES = {
    'MindCube': ('MLL-Lab/MindCube', 'data.zip', '',
                 ['MindCubeBench_tiny_raw_qa', 'MindCubeBench_raw_qa']),
    'OmniSpatial': ('qizekun/OmniSpatial', 'OmniSpatial-test.zip', 'OmniSpatial-test/',
                    ['OmniSpatialBench']),
}

FREAK_REPO = 'hansQAQ/FREAK'
FREAK_SPLITS = ['FREAK_MCQ', 'FREAK_VQA']

EMBEDDED = '— (images embedded in TSV)'
GENERAL_ROWS = [  # (split, image archive) in README order
    ('RealWorldQA', EMBEDDED),
    ('HallusionBench', EMBEDDED),
    ('VStarBench', EMBEDDED),
    ('FREAK_MCQ', 'images/FREAK.zip'),
    ('FREAK_VQA', 'images/FREAK.zip'),
]
SPATIAL_ROWS = [
    ('MMSIBench_wo_circular', EMBEDDED),
    ('ERQA', EMBEDDED),
    ('SparBench', EMBEDDED),
    ('SparBench_tiny', EMBEDDED),
    ('MindCubeBench_raw_qa', 'images/MindCube.zip'),
    ('MindCubeBench_tiny_raw_qa', 'images/MindCube.zip'),
    ('EmbSpatialBench', EMBEDDED),
    ('OmniSpatialBench', 'images/OmniSpatial.zip'),
]

SPATIAL_SECTION = """## Spatial reasoning

| Split | Rows | Image archive |
|---|---|---|
{table}

`OmniSpatialBench` is shared by the `OmniSpatialBench_{{default,zeroshot_cot,manual_cot}}`
prompt variants; `SparBench_tiny` and `MindCubeBench_tiny_raw_qa` are the smaller
subsets of `SparBench` and `MindCubeBench_raw_qa`.

"""

GENERAL_CREDITS = """- **RealWorldQA** — released by xAI with Grok-1.5 Vision, 2024
  ([xai-org/RealworldQA](https://huggingface.co/datasets/xai-org/RealworldQA), CC BY-ND 4.0).
  `RealWorldQA` is the VLMEvalKit TSV, unchanged.
- **HallusionBench** — Guan et al., *HallusionBench: An Advanced Diagnostic Suite for
  Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models*,
  CVPR 2024 (arXiv:2310.14566). `HallusionBench` is the VLMEvalKit TSV, unchanged.
- **V\\* Bench** — Wu and Xie, *V\\*: Guided Visual Search as a Core Mechanism in
  Multimodal LLMs*, CVPR 2024 (arXiv:2312.14135;
  [craigwu/vstar_bench](https://huggingface.co/datasets/craigwu/vstar_bench)).
  `VStarBench` is the TSV from
  [xjtupanda/VStar_Bench](https://huggingface.co/datasets/xjtupanda/VStar_Bench), unchanged.
- **FREAK** — [hansQAQ/FREAK](https://huggingface.co/datasets/hansQAQ/FREAK) (test split).
  `FREAK_MCQ` holds the multiple-choice questions with options shuffled
  (`random.Random(42)`) so the answer is not always A; `FREAK_VQA` holds the
  open-ended questions.
"""

SPATIAL_CREDITS = """### Spatial reasoning

All spatial TSVs are the versions from the EASI leaderboard
([lmms-lab-si/EASI-Leaderboard-Data](https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data),
Apache-2.0), unchanged. MindCube and OmniSpatial images are copied from the original
releases, keeping only the images the TSVs use.

- **MMSI-Bench** — *MMSI-Bench: A Benchmark for Multi-Image Spatial Intelligence*
  (arXiv:2505.23764; [RunsenXu/MMSI-Bench](https://huggingface.co/datasets/RunsenXu/MMSI-Bench), CC BY 4.0).
- **ERQA** — Gemini Robotics Team, *Gemini Robotics: Bringing AI into the Physical World*
  (arXiv:2503.20020).
- **SPAR-Bench** — *From Flatland to Space: Teaching Vision-Language Models to Perceive
  and Reason in 3D* (arXiv:2503.22976).
- **MindCube** — *Spatial Mental Modeling from Limited Views* (arXiv:2506.21458;
  [MLL-Lab/MindCube](https://huggingface.co/datasets/MLL-Lab/MindCube), MIT).
- **EmbSpatial-Bench** — *EmbSpatial-Bench: Benchmarking Spatial Understanding for Embodied
  Tasks with Large Vision-Language Models* (arXiv:2406.05756;
  [Phineas476/EmbSpatial-Bench](https://huggingface.co/datasets/Phineas476/EmbSpatial-Bench), CC BY 4.0).
- **OmniSpatial** — *OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for
  Vision Language Models* (arXiv:2506.03135;
  [qizekun/OmniSpatial](https://huggingface.co/datasets/qizekun/OmniSpatial), Apache-2.0).

"""


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def download(url, dst):
    tmp = dst + '.part'
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, 'wb') as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    os.replace(tmp, dst)


def stage_tsv(split, lmu, out):
    """Copy the split's original TSV into ``out``; raise if its MD5 is not the expected one."""
    url, want = TSV_SOURCES[split]
    dst = osp.join(out, f'{split}.tsv')
    if osp.exists(dst) and md5(dst) == want:
        return
    local = osp.join(lmu, f'{split}.tsv') if lmu else None
    if local and osp.exists(local) and md5(local) == want:
        print(f'{split}: copying {local}')
        shutil.copyfile(local, dst)
    else:
        print(f'{split}: downloading {url}')
        download(url, dst)
    got = md5(dst)
    if got != want:
        raise RuntimeError(f'{split}: MD5 {got} != expected {want}')


def tsv_image_paths(tsv_path):
    """All image paths referenced by a TSV's ``image_path`` column (lists are expanded)."""
    import ast
    paths = set()
    for x in pd.read_csv(tsv_path, sep='\t', usecols=['image_path'])['image_path']:
        x = str(x).strip()
        items = ast.literal_eval(x) if x.startswith('[') else [x]
        paths.update(str(p).strip().lstrip('/\\') for p in items)
    return paths


def build_image_zip(folder, out):
    """Write images/<folder>.zip holding only the images its splits reference."""
    from huggingface_hub import hf_hub_download

    repo, zip_name, prefix, splits = IMAGE_SOURCES[folder]
    wanted = set()
    for s in splits:
        wanted |= tsv_image_paths(osp.join(out, f'{s}.tsv'))
    print(f'images/{folder}.zip: {len(wanted)} images from {repo}/{zip_name}')
    src_zip = hf_hub_download(repo, zip_name, repo_type='dataset')
    dst_zip = osp.join(out, 'images', f'{folder}.zip')
    os.makedirs(osp.dirname(dst_zip), exist_ok=True)
    with zipfile.ZipFile(src_zip) as src:
        members = {osp.normpath(i.filename).lstrip('/\\'): i for i in src.infolist() if not i.is_dir()}
        missing = sorted(p for p in wanted if osp.normpath(prefix + p) not in members)
        if missing:
            raise FileNotFoundError(f'{folder}: {len(missing)} images not in {zip_name}, e.g. {missing[:3]}')
        with zipfile.ZipFile(dst_zip + '.part', 'w', zipfile.ZIP_STORED) as dst:
            for p in sorted(wanted):
                dst.writestr(f'{folder}/{osp.normpath(p)}', src.read(members[osp.normpath(prefix + p)]))
    os.replace(dst_zip + '.part', dst_zip)


def build_freak(img_dir, out):
    """Rebuild FREAK_MCQ.tsv / FREAK_VQA.tsv (same logic as the old FREAK builder)."""
    from datasets import load_dataset

    os.makedirs(img_dir, exist_ok=True)
    ds = load_dataset(FREAK_REPO, split='test')
    rng = random.Random(42)
    mcq, vqa = [], []
    for ex in ds:
        img_name = f"{ex['id']}.jpg"
        if not osp.exists(osp.join(img_dir, img_name)):
            ex['image'].save(osp.join(img_dir, img_name))
        question, gt, options = ex['question'], ex['ground_truth'], ex['options']
        category = json.dumps(ex.get('category') or [])
        if ex['type'] == 'mcq':
            if options is None:
                continue
            shuffled = list(options)
            rng.shuffle(shuffled)
            letter = None
            for norm in (str.strip, lambda s: s.strip().lower()):
                letter = next((string.ascii_uppercase[i] for i, o in enumerate(shuffled) if norm(o) == norm(gt)), None)
                if letter is not None:
                    break
            if letter is None:
                continue
            option_str = ', '.join(f'{string.ascii_uppercase[i]}. {o}' for i, o in enumerate(shuffled))
            mcq.append({'index': len(mcq), 'image_path': img_name, 'question': question,
                        'prompt': f'{question} Choose one option from the following: {option_str}',
                        'mcq_GT': letter, 'choices': json.dumps(shuffled), 'text_GT': gt, 'category': category})
        elif ex['type'] == 'fib':
            vqa.append({'index': len(vqa), 'image_path': img_name, 'question': question,
                        'prompt': f'{question} Provide your final answer in a clear and concise sentence or a few words.',
                        'GT': gt, 'category': category})
    pd.DataFrame(mcq).to_csv(osp.join(out, 'FREAK_MCQ.tsv'), sep='\t', index=False)
    pd.DataFrame(vqa).to_csv(osp.join(out, 'FREAK_VQA.tsv'), sep='\t', index=False)


def stage_freak(lmu, out):
    """Use $LMUData's FREAK TSVs + images when complete, else rebuild them from HF."""
    lmu_img = osp.join(lmu, 'images', 'FREAK') if lmu else None
    have = lmu and all(osp.exists(osp.join(lmu, f'{s}.tsv')) for s in FREAK_SPLITS)
    if have:
        paths = set().union(*(tsv_image_paths(osp.join(lmu, f'{s}.tsv')) for s in FREAK_SPLITS))
        have = all(osp.exists(osp.join(lmu_img, p)) for p in paths)
    if have:
        print(f'FREAK: using TSVs and images from {lmu}')
        for s in FREAK_SPLITS:
            shutil.copyfile(osp.join(lmu, f'{s}.tsv'), osp.join(out, f'{s}.tsv'))
        img_dir = lmu_img
    else:
        print(f'FREAK: rebuilding from {FREAK_REPO}')
        img_dir = osp.join(out, '_freak_images')
        build_freak(img_dir, out)
    paths = set().union(*(tsv_image_paths(osp.join(out, f'{s}.tsv')) for s in FREAK_SPLITS))
    dst_zip = osp.join(out, 'images', 'FREAK.zip')
    os.makedirs(osp.dirname(dst_zip), exist_ok=True)
    with zipfile.ZipFile(dst_zip, 'w', zipfile.ZIP_STORED) as zf:
        for p in sorted(paths):
            zf.write(osp.join(img_dir, p), f'FREAK/{p}')
    print(f'images/FREAK.zip: {len(paths)} images')


def count_rows(out, split):
    return len(pd.read_csv(osp.join(out, f'{split}.tsv'), sep='\t', usecols=[0]))


def table(rows, n_rows):
    return '\n'.join(f'| {s} | {n_rows[s]} | {arc} |' for s, arc in rows)


def merge_readme(text, n_rows):
    """Add config entries, table rows and credits for the new splits to the current card."""
    def insert_before(text, marker, block):
        if marker not in text:
            raise ValueError(f'README.md: marker {marker!r} not found; update merge_readme')
        return text.replace(marker, block + marker, 1)

    splits = [s for s, _ in GENERAL_ROWS + SPATIAL_ROWS]
    configs = ''.join(f'- config_name: {s}\n  data_files: {s}.tsv\n  sep: "\\t"\n'
                      for s in splits if f'config_name: {s}\n' not in text)
    front_end = text.index('\n---\n', 4) + 1  # closing '---' of the YAML front matter
    text = text[:front_end] + configs + text[front_end:]

    new_general = [(s, a) for s, a in GENERAL_ROWS if f'| {s} |' not in text]
    if new_general:
        # Append to the existing "General VQA" table (it ends at the first blank line).
        start = text.index('## General VQA\n')
        end = text.index('\n\n', text.index('|---|', start))
        text = text[:end] + '\n' + table(new_general, n_rows) + text[end:]
    if '## Spatial reasoning\n' not in text:
        text = insert_before(text, '## Sources, licenses and citations\n',
                             SPATIAL_SECTION.format(table=table(SPATIAL_ROWS, n_rows)))
    if '**RealWorldQA**' not in text:
        start = text.index('### General VQA\n')
        end = text.index('\n\n', text.index('\n- ', start))  # end of that bullet list
        text = text[:end] + '\n' + GENERAL_CREDITS.rstrip('\n') + text[end:]
    if '### Spatial reasoning\n' not in text:
        text = insert_before(text, 'If you are an author of one of these datasets', SPATIAL_CREDITS)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--lmu-data', default=os.environ.get('LMUData'))
    ap.add_argument('--out', default='outputs/hf_backup')
    ap.add_argument('--upload', action='store_true', help=f'open a pull request on {HF_REPO_ID} with the staged files')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for split in TSV_SOURCES:
        stage_tsv(split, args.lmu_data, args.out)
    for folder in IMAGE_SOURCES:
        build_image_zip(folder, args.out)
    stage_freak(args.lmu_data, args.out)

    splits = [s for s, _ in GENERAL_ROWS + SPATIAL_ROWS]
    n_rows = {s: count_rows(args.out, s) for s in splits}
    md5s = {s: md5(osp.join(args.out, f'{s}.tsv')) for s in splits}
    print('rows: ' + json.dumps(n_rows, indent=4))
    print('MD5: ' + json.dumps(md5s, indent=4))

    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    api = HfApi()
    head = api.dataset_info(HF_REPO_ID).sha
    card = hf_hub_download(HF_REPO_ID, 'README.md', repo_type='dataset', revision=head)
    with open(card) as f:
        readme = merge_readme(f.read(), n_rows)
    with open(osp.join(args.out, 'README.md'), 'w') as f:
        f.write(readme)
    print(f'README.md merged into {args.out}/README.md')

    if args.upload:
        files = [f'{s}.tsv' for s in splits] + [f'images/{f}.zip' for f in [*IMAGE_SOURCES, 'FREAK']]
        ops = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=osp.join(args.out, f)) for f in files]
        ops.append(CommitOperationAdd(path_in_repo='README.md', path_or_fileobj=osp.join(args.out, 'README.md')))
        # A PR rather than a direct commit: contributors cannot push to repos created by
        # others in the org. It is based on `head`, the revision README.md was merged against.
        info = api.create_commit(HF_REPO_ID, repo_type='dataset', operations=ops, parent_commit=head,
                                 create_pr=True,
                                 commit_message='Add general-VQA and spatial benchmarks '
                                                '(RealWorldQA, HallusionBench, V*, FREAK, MMSI, ERQA, SPAR, '
                                                'MindCube, EmbSpatial, OmniSpatial)')
        print(f'Opened {info.pr_url} -- an org admin needs to merge it')


if __name__ == '__main__':
    main()
