"""FactIP dataset (Unify-Agent paper, arXiv 2603.29620).

Knowledge-grounded image generation focused on IP identity preservation.
500 prompts across 12 fine-grained categories under 3 groups
(CHARACTER / OBJECT / SCENE). Bilingual: 287 zh + 213 en.

Each prompt is paired with 2 ground-truth "seed" reference images
(seed1, seed2) that jointly define the IP identity. The judge compares
the model-generated image against both seeds along 5 dimensions, with
75% weight on text_relevance_ip (IP identity preservation) — see
``dataset/utils/judge_factip.py::_WEIGHTS``.

Data source: HuggingFace ``csfufu/FactIP`` snapshot.
  - ``factIP.json`` — 500-entry dict keyed by index (hash) with fields:
      ``index, ip_name, ip_name_zh, ip_name_en, ip_category,
       source_category, image_prompt, language, country,
       seed1_url, seed2_url``
  - ``images.zip`` (~975 MB) — bundle of all seed reference images.

We auto-download both on first access and cache seed images under
``$LMUData/FactIP_seed_images/``. The upstream ``seed*_url`` fields point
to a Tencent COS bucket; we ONLY rely on those URLs to derive the
filename hash and map it to the local image in the extracted zip.

Prediction resolution mirrors KnowGen: walk
``<pred_root>/full_trajectories/images/<task_id>_*.png`` to find the
model's final generated image (needs ``--save-images files`` at run time).

Judge: only ``judge_factip`` (Unify-Agent protocol). Default judge model
= whatever ``judge_rui.yaml`` specifies (gpt-4o); paper uses gpt-4o
directly so this is aligned.
"""

import glob
import json
import os
import os.path as osp
import zipfile
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import pandas as pd

from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
from ..smp.file import LMUDataRoot, get_intermediate_file_path
from ..utils import track_progress_rich


_HF_REPO_ID = 'csfufu/FactIP'
_HF_METADATA = 'factIP.json'
_HF_IMAGES_ZIP = 'images.zip'


def _snapshot_download_factip_metadata() -> str:
    from huggingface_hub import hf_hub_download

    dest = osp.join(LMUDataRoot(), '_factip_source')
    return hf_hub_download(
        repo_id=_HF_REPO_ID,
        filename=_HF_METADATA,
        repo_type='dataset',
        local_dir=dest,
    )


def _snapshot_download_factip_images() -> str:
    """Download images.zip once and extract. Return extraction root."""
    from huggingface_hub import hf_hub_download

    root = osp.join(LMUDataRoot(), 'FactIP_seed_images')
    marker = osp.join(root, '.extracted_ok')
    if osp.isfile(marker):
        return root

    os.makedirs(root, exist_ok=True)
    zip_path = hf_hub_download(
        repo_id=_HF_REPO_ID,
        filename=_HF_IMAGES_ZIP,
        repo_type='dataset',
        local_dir=root,
    )
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(path=root)
    with open(marker, 'w') as f:
        f.write('extraction complete\n')
    return root


def _resolve_seed_image(seed_url: str, images_root: str) -> str:
    """Given the upstream Tencent COS seed URL, find the corresponding local file.

    URL format:
      http://prc-videoframe-pub-....myqcloud.com/hunyuan_img2img/somoszhang/
      IP-Datas/ip-data-seed-imgs/IP类目-动漫角色类/<hash>.png
    """
    if not seed_url:
        return ''
    try:
        parsed = urlparse(str(seed_url))
        path = parsed.path
    except Exception:
        return ''
    filename = osp.basename(path)  # <hash>.png
    if not filename:
        return ''

    # Search under images_root for the file (any subdir)
    for dirpath, _, files in os.walk(images_root):
        if filename in files:
            return osp.join(dirpath, filename)
    return ''


def _build_seed_index_map(images_root: str) -> Dict[str, Dict[str, str]]:
    """Map sample ``index`` -> ``{'1': seed1_path, '2': seed2_path}``.

    ``images.zip`` lays the seeds out as
    ``images/images/<category>/<index>[_<n>]/seed{1,2}.png`` — i.e. the directory
    basename is the sample INDEX (optionally suffixed ``_<n>``). The upstream
    ``seed*_url`` hash filenames do NOT match the shipped filenames (the URL bucket
    and the released zip use different naming), so index is the reliable join key.
    The ``_<n>`` suffix is inconsistent between the metadata index and the zip dir,
    so each dir is registered under BOTH its full name and its stripped hash; the
    lookup (``_convert_benchmark``) tries the full index then the stripped one.
    Built once per benchmark conversion (single walk).
    """
    def _strip(name: str) -> str:
        if '_' in name:
            head, tail = name.rsplit('_', 1)
            if tail.isdigit():
                return head
        return name

    out: Dict[str, Dict[str, str]] = {}
    for dirpath, _dirs, files in os.walk(images_root):
        base = osp.basename(dirpath)
        seeds: Dict[str, str] = {}
        for f in files:
            fl = f.lower()
            if fl.startswith('seed1.'):
                seeds['1'] = osp.join(dirpath, f)
            elif fl.startswith('seed2.'):
                seeds['2'] = osp.join(dirpath, f)
        if not seeds:
            continue
        for key in {base, _strip(base)}:
            entry = out.setdefault(key, {})
            entry.update(seeds)
    return out


def _grade_item_judge_factip(model, item):
    """Invoke the FactIP judge on one sample."""
    from .utils.judge_factip import (
        build_judge_factip_messages,
        parse_judge_factip_judgement,
    )

    prompt = str(item.get('prompt', item.get('image_prompt', '')))
    gt1 = str(item.get('seed1_local', ''))
    gt2 = str(item.get('seed2_local', ''))
    gen = str(item.get('generated_image', ''))
    sample_id = str(item.get('index', ''))

    if not gen or not osp.isfile(gen):
        return dict(
            correct=0, weighted_overall=0.0,
            clarity=0, content_quality=0, aesthetics=0,
            text_relevance_ip=0, overall_score=0,
            rationale='', reasoning=f'generated image missing: {gen!r}',
        )
    if (not gt1 or not osp.isfile(gt1)) and (not gt2 or not osp.isfile(gt2)):
        return dict(
            correct=0, weighted_overall=0.0,
            clarity=0, content_quality=0, aesthetics=0,
            text_relevance_ip=0, overall_score=0,
            rationale='', reasoning='both GT seed images missing',
        )
    # If only one GT exists, duplicate it so the judge still gets 3 images
    if not gt1 or not osp.isfile(gt1):
        gt1 = gt2
    if not gt2 or not osp.isfile(gt2):
        gt2 = gt1

    messages = build_judge_factip_messages(
        sample_id=sample_id, prompt=prompt,
        generated_image_path=gen, gt1_path=gt1, gt2_path=gt2,
    )

    retry = 3
    last = None
    while retry:
        try:
            out = model.generate(messages)
        except Exception as e:
            out = f'Failed to obtain answer via API: {e}'
        if 'Failed to obtain answer via API' in str(out):
            retry -= 1
            continue
        last = out
        parsed = parse_judge_factip_judgement(str(out))
        if parsed['verdict']:
            return dict(
                correct=parsed['correct'],
                weighted_overall=parsed['weighted_overall'],
                clarity=parsed['clarity'],
                content_quality=parsed['content_quality'],
                aesthetics=parsed['aesthetics'],
                text_relevance_ip=parsed['text_relevance_ip'],
                overall_score=parsed['overall_score'],
                rationale=parsed['rationale'],
                reasoning=parsed['reasoning'],
            )
        retry -= 1

    return dict(
        correct=0, weighted_overall=0.0,
        clarity=0, content_quality=0, aesthetics=0,
        text_relevance_ip=0, overall_score=0,
        rationale='', reasoning=f'judge_factip unparseable: {str(last)[:200]}' if last else 'judge unreachable',
    )


def _resolve_generated_image(pred_root: str, task_id: str) -> Optional[str]:
    img_dir = osp.join(pred_root, 'full_trajectories', 'images')
    if not osp.isdir(img_dir):
        return None
    matches: List[str] = []
    for ext in ('png', 'jpg', 'jpeg', 'webp'):
        matches.extend(glob.glob(osp.join(img_dir, f'{task_id}_*.{ext}')))
    if not matches:
        return None

    def _sort_key(p: str):
        stem = osp.basename(p).rsplit('.', 1)[0]
        try:
            return int(stem.rsplit('_', 1)[-1])
        except Exception:
            return 0

    matches.sort(key=_sort_key)
    return matches[-1]


class FactIPDataset(ImageBaseDataset):
    """FactIP (Unify-Agent paper) — 500-prompt IP-identity image generation."""

    TYPE = 'VQA'

    DATASET_URL = {
        'FactIP': '',
    }
    DATASET_MD5 = {}

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_URL)

    def load_data(self, dataset):
        data_root = LMUDataRoot()
        tsv_path = osp.join(data_root, f'{dataset}.tsv')
        if not osp.exists(tsv_path):
            df = self._convert_benchmark()
            os.makedirs(data_root, exist_ok=True)
            df.to_csv(tsv_path, sep='\t', index=False)
        self.data_path = tsv_path
        return load(tsv_path)

    @staticmethod
    def _convert_benchmark() -> pd.DataFrame:
        meta_path = _snapshot_download_factip_metadata()
        images_root = _snapshot_download_factip_images()

        with open(meta_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Seed images join by sample INDEX (the zip directory key), NOT by the
        # seed*_url hash — the URL bucket and the released images.zip use different
        # filename hashes, so the URL-basename lookup finds nothing (all-zero judge).
        seed_map = _build_seed_index_map(images_root)

        # data is a dict keyed by index hash → row dict
        if isinstance(data, dict):
            it = data.items()
        elif isinstance(data, list):
            it = enumerate(data)
        else:
            raise ValueError(f'Unexpected FactIP shape: {type(data)}')

        rows: List[Dict[str, Any]] = []
        for _, obj in it:
            idx = str(obj.get('index', ''))
            prompt = str(obj.get('image_prompt', ''))
            seed1_url = str(obj.get('seed1_url', ''))
            seed2_url = str(obj.get('seed2_url', ''))
            _seeds = seed_map.get(idx) or seed_map.get(idx.rsplit('_', 1)[0], {})
            seed1_local = _seeds.get('1', '') or _resolve_seed_image(seed1_url, images_root)
            seed2_local = _seeds.get('2', '') or _resolve_seed_image(seed2_url, images_root)

            language = str(obj.get('language', 'en'))
            country = str(obj.get('country', ''))
            ip_name = str(obj.get('ip_name', ''))
            ip_name_en = str(obj.get('ip_name_en', ''))
            ip_category = str(obj.get('ip_category', 'unknown') or 'unknown')
            source_cat = str(obj.get('source_category', 'unknown') or 'unknown')

            rows.append({
                'index': idx,
                'prompt': prompt,
                'question': prompt,       # framework compat alias
                'answer': ip_name_en,     # optional text ref
                'ip_name': ip_name,
                'ip_name_en': ip_name_en,
                'ip_category': ip_category,
                'source_category': source_cat,
                'seed1_local': seed1_local,
                'seed2_local': seed2_local,
                'seed1_url': seed1_url,
                'seed2_url': seed2_url,
                'language': language,
                'country': country,
                'category': source_cat,           # top-level (Chinese labels)
                'l2-category': f'{source_cat}/{ip_category}',
            })
        return pd.DataFrame(rows)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        prompt = str(line.get('prompt', line.get('question', '')))
        return [dict(type='text', value=prompt)]

    def evaluate(self, eval_file, **judge_kwargs):
        logger = get_logger('Evaluation')
        data = load(eval_file)
        assert 'prediction' in data, 'prediction column missing from eval file'
        data['prediction'] = [str(x) for x in data['prediction']]

        # Merge metadata
        meta = self.data
        idx_str = lambda v: str(v)
        for col in ('prompt', 'seed1_local', 'seed2_local', 'language',
                    'ip_category', 'source_category'):
            if col in meta.columns:
                mp = {idx_str(i): v for i, v in zip(meta['index'], meta[col])}
                data[col] = [mp.get(idx_str(i), '') for i in data['index']]

        # Resolve generated images from run dir
        pred_root = osp.dirname(osp.abspath(eval_file))
        gen_paths = []
        missing = 0
        for i in range(len(data)):
            tid = idx_str(data.iloc[i]['index'])
            p = _resolve_generated_image(pred_root, tid)
            if not p:
                missing += 1
            gen_paths.append(p or '')
        data['generated_image'] = gen_paths
        if missing:
            logger.warning(
                f'{missing}/{len(data)} tasks have no resolvable generated image under '
                f'{pred_root}/full_trajectories/images/. Judge will score them 0.'
            )

        judge_kwargs = dict(judge_kwargs)
        nproc = judge_kwargs.pop('nproc', 4)
        judge_kwargs.pop('enable_llm_judge', None)

        judge_name = (
            os.environ.get('MMDR_JUDGE_PROMPT')
            or judge_kwargs.pop('judge_prompt', None)
            or 'judge_factip'
        ).strip().lower()
        # Unconditional pop: the or-chain above short-circuits when the env
        # var is set, which would leave judge_prompt in judge_kwargs and leak
        # it into the API payload (cf. livevqa.py).
        judge_kwargs.pop('judge_prompt', None)
        if judge_name != 'judge_factip':
            raise ValueError(
                f"FactIP only supports 'judge_factip'; got {judge_name!r}."
            )

        # Full-detail image input for the judge (framework default is 'low')
        judge_kwargs.setdefault('img_detail', 'high')
        model = build_judge(**judge_kwargs)
        if model is None or not model.working():
            logger.error(
                'FactIP evaluation requires a working multimodal judge (default gpt-4o).'
            )
            logger.error(DEBUG_MESSAGE)
            return pd.DataFrame()

        grade_fn = _grade_item_judge_factip
        stem_base = '_factip_judge'
        grade_stem = f'_grade{stem_base}'
        detail_stem = f'_detail{stem_base}'
        acc_stem = f'_acc{stem_base}'

        result_file = get_intermediate_file_path(eval_file, grade_stem, 'pkl')
        result_cache = {}
        if osp.exists(result_file):
            result_cache = load(result_file)

        items_to_grade = []
        for i in range(len(data)):
            item = data.iloc[i]
            idx = str(item['index'])
            if idx not in result_cache:
                items_to_grade.append(item)

        if items_to_grade:
            tups = [dict(model=model, item=x) for x in items_to_grade]
            keys = [str(x['index']) for x in items_to_grade]
            res = track_progress_rich(
                grade_fn, tups, nproc=nproc, chunksize=nproc,
                save=result_file, keys=keys,
            )
            result_cache = load(result_file)
            for k, v in zip(keys, res):
                if k not in result_cache:
                    result_cache[k] = v

        for col, default in [
            ('correct', 0),
            ('weighted_overall', 0.0),
            ('clarity', 0),
            ('content_quality', 0),
            ('aesthetics', 0),
            ('text_relevance_ip', 0),
            ('overall_score', 0),
            ('reasoning', ''),
        ]:
            data[col] = [
                result_cache.get(str(idx), {}).get(col, default)
                for idx in data['index']
            ]

        detail_file = get_intermediate_file_path(eval_file, detail_stem)
        dump(data, detail_file)

        acc = self._compute_accuracy(data)
        score_file = get_intermediate_file_path(eval_file, acc_stem, 'csv')
        dump(acc, score_file)
        logger.info(f'FactIP [{judge_name}] accuracy saved to {score_file}')
        return acc

    @staticmethod
    def _compute_accuracy(data) -> pd.DataFrame:
        rows = []

        def _bloc(df: pd.DataFrame, name: str):
            if len(df) == 0:
                return None
            row = {'group': name, 'n': len(df)}
            for k in ('weighted_overall', 'correct', 'clarity', 'content_quality',
                      'aesthetics', 'text_relevance_ip', 'overall_score'):
                if k in df.columns:
                    vals = pd.to_numeric(df[k], errors='coerce').fillna(0.0)
                    row[k] = float(vals.mean())
            return row

        blk = _bloc(data, 'Overall')
        if blk is not None:
            rows.append(blk)

        for lang in ('zh', 'en'):
            sub = data[data.get('language', pd.Series([], dtype=str)) == lang]
            blk = _bloc(sub, f'language:{lang}')
            if blk is not None:
                rows.append(blk)

        if 'source_category' in data.columns:
            for cat, sub in data.groupby('source_category'):
                blk = _bloc(sub, f'source:{cat}')
                if blk is not None:
                    rows.append(blk)

        if 'ip_category' in data.columns:
            for cat, sub in data.groupby('ip_category'):
                blk = _bloc(sub, f'ip:{cat}')
                if blk is not None:
                    rows.append(blk)

        return pd.DataFrame(rows)
