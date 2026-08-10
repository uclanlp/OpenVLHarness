"""T2I-FactualBench dataset (arXiv 2404.19467).

Knowledge-intensive text-to-image benchmark structured in 3 hierarchical
tiers:

  - SKCM (Single Knowledge Concept Memorization): 1628 samples,
    ``task = "momerization"`` [sic] — evaluated on concept factuality only.
  - SKCI (Single Knowledge Concept Instantiation): 600 samples,
    ``task ∈ {"action", "attribute", "background variation", "variation"}``
    — evaluated on concept factuality + task completeness.
  - MKCC (Multiple Knowledge Concept Composition): 800 samples with two
    concepts, ``task ∈ {"interaction", "size", "differentiating"}`` —
    evaluated on concept factuality + task completeness + integration
    factuality.

  Total: 3028 samples across all 3 tiers.

Per-sample schema (source JSONL under
``$LMUData/T2I-FactualBench/data/prompts/<TIER>/prompt.jsonl``; override the
prompts directory via ``T2I_FACTUALBENCH_PROMPTS_DIR``):
  - ``concept``          : list[str] (len 1 for SKCM/SKCI, len 2 for MKCC)
  - ``type``             : list[str] (same length, category tag per concept)
  - ``sentence``         : str (the T2I prompt)
  - ``reference_image``  : list[str] (paths inside the released tarball —
                          rewritten to local snapshot paths on load)
  - ``addition``         : list[str] (task-specific: action verb / attribute /
                          size relation / etc.; empty for SKCM)
  - ``task``             : str (task type per tier)

Reference images live in ``Sakeoffellow001/T2i_Factualbench/
concept_image.tar.gz`` on HuggingFace; we snapshot-download and extract
on first access.

Metric (per ``dataset/utils/judge_t2i_factualbench.py::metric_for_tier``,
normalized to [0, 1]):
  - SKCM: Concept_Score / 4
  - SKCI: (Concept_Score / 4 + Task_Score) / 2
  - MKCC: (Concept_Score / 4 + Task_Score + Integration_Score) / 3

Judge: only ``judge_t2i_factualbench`` (multi-round VQA per paper).
Default judge model = whatever ``judge_rui.yaml`` specifies (gpt-4o).
Paper uses gpt-4o-0513-global via Aliyun API.
"""

import glob
import json
import os
import os.path as osp
import tarfile
from typing import Any, Dict, List, Optional

import pandas as pd

from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
from ..smp.file import LMUDataRoot, get_intermediate_file_path
from ..utils import track_progress_rich


# ---------------------------------------------------------------------------
# Source data
# ---------------------------------------------------------------------------
_HF_REPO_ID = 'Sakeoffellow001/T2i_Factualbench'
_TAR_NAME = 'concept_image.tar.gz'
_TIER_FILES = {
    'SKCM': ('SKCM', 'prompt.jsonl'),
    'SKCI': ('SKCI', 'prompt.jsonl'),
    'MKCC': ('MKCC', 'prompt.jsonl'),
}


def _default_prompts_dir() -> str:
    """Default T2I-FactualBench prompts dir, under the shared LMUData root
    (portable across checkouts; override with env T2I_FACTUALBENCH_PROMPTS_DIR)."""
    return osp.join(LMUDataRoot(), 'T2I-FactualBench', 'data', 'prompts')


def _resolve_prompts_dir() -> str:
    d = os.environ.get('T2I_FACTUALBENCH_PROMPTS_DIR') or _default_prompts_dir()
    d = osp.abspath(d)
    if not osp.isdir(d):
        raise FileNotFoundError(
            f'T2I-FactualBench prompts dir not found at {d!r}. '
            'Clone https://github.com/Sakeoffellow001/T2I-FactualBench and either put it under '
            '$LMUData/T2I-FactualBench or set env T2I_FACTUALBENCH_PROMPTS_DIR.'
        )
    return d


def _snapshot_download_concept_images() -> str:
    """Download the concept_image.tar.gz snapshot to a stable location and
    extract if not already extracted. Return the root dir containing the
    per-type subfolders (animal/, pet/, ..., landmark/, ...).
    """
    from huggingface_hub import hf_hub_download

    cache_root = osp.join(LMUDataRoot(), 'T2IFactualBench_concept_images')
    marker = osp.join(cache_root, '.extracted_ok')
    if osp.isfile(marker):
        return cache_root

    os.makedirs(cache_root, exist_ok=True)
    tar_path = hf_hub_download(
        repo_id=_HF_REPO_ID,
        filename=_TAR_NAME,
        repo_type='dataset',
        local_dir=cache_root,
    )
    with tarfile.open(tar_path, 'r:gz') as tf:
        # Validate every member before extracting: extractall() would happily
        # follow '../' entries or absolute paths and write outside cache_root.
        # Same guard as searchgen._ensure_reference_images.
        safe_root = osp.abspath(cache_root)
        for m in tf.getmembers():
            dst = osp.abspath(osp.join(cache_root, m.name))
            if dst != safe_root and not dst.startswith(safe_root + osp.sep):
                raise ValueError(f'tar member escapes {cache_root}: {m.name}')
            if m.issym() or m.islnk():
                link_dst = osp.abspath(
                    osp.join(osp.dirname(dst), m.linkname)
                )
                if not link_dst.startswith(safe_root + osp.sep):
                    raise ValueError(f'tar member links outside {cache_root}: {m.name}')
        tf.extractall(path=cache_root)
    with open(marker, 'w') as f:
        f.write('extraction complete\n')
    return cache_root


def _rewrite_ref_image_path(src: str, concept_root: str) -> str:
    """Translate a ``/mnt/workspace/.../image/<type>/<file>`` path to the local
    snapshot at ``<concept_root>/<...matching subdir structure...>``.
    """
    if not src:
        return ''
    # Grab the tail (last two components: type_/filename)
    parts = src.replace('\\', '/').split('/')
    if len(parts) >= 2:
        candidate = osp.join(concept_root, parts[-2], parts[-1])
        if osp.isfile(candidate):
            return candidate
        # Some releases nest under concept_image/
        candidate2 = osp.join(concept_root, 'concept_image', parts[-2], parts[-1])
        if osp.isfile(candidate2):
            return candidate2
    # Best-effort: return unmodified so downstream error surfaces the path
    return src


# ---------------------------------------------------------------------------
# Per-sample grader
# ---------------------------------------------------------------------------

def _join_raws(concept_raws: list, task_raw: str, integration_raw: str) -> str:
    """Keep every sub-judge's raw output, labelled by which judge produced it.

    A sample is graded by up to three judges (concept x N, task, integration),
    so returning only the first concept raw would silently drop most of the
    judge evidence needed to audit a score.
    """
    parts = []
    for i, r in enumerate(concept_raws):
        if r:
            parts.append(f'[concept {i + 1}]\n{r}')
    if task_raw:
        parts.append(f'[task]\n{task_raw}')
    if integration_raw:
        parts.append(f'[integration]\n{integration_raw}')
    return '\n\n'.join(parts)


def _grade_item_judge_t2i_factualbench(model, item):
    """Run the 1-3 judge calls appropriate for the row's tier + task.

    Returns dict with:
      - concept_score, task_score, integration_score (raw)
      - metric (normalized [0,1] via metric_for_tier)
      - correct (1 if metric >= 0.5 else 0)
      - reasoning, raw
    """
    from .utils.judge_t2i_factualbench import (
        concept_prompt,
        task_prompt_skci,
        task_prompt_mkcc,
        integration_prompt,
        parse_concept_rating,
        parse_task_score,
        parse_integration_score,
        build_concept_messages,
        build_task_messages,
        build_integration_messages,
        metric_for_tier,
    )

    tier = str(item.get('tier', ''))
    sentence = str(item.get('sentence', item.get('prompt', '')))
    concepts = list(item.get('concept', []) or [])
    types = list(item.get('type', []) or [])
    additions = list(item.get('addition', []) or [])
    ref_imgs = list(item.get('reference_image', []) or [])
    task_type = str(item.get('task_type', ''))
    gen_path = str(item.get('generated_image', ''))

    if isinstance(concepts, str):
        try:
            concepts = json.loads(concepts)
        except Exception:
            concepts = [concepts]
    if isinstance(types, str):
        try:
            types = json.loads(types)
        except Exception:
            types = [types]
    if isinstance(additions, str):
        try:
            additions = json.loads(additions)
        except Exception:
            additions = [additions]
    if isinstance(ref_imgs, str):
        try:
            ref_imgs = json.loads(ref_imgs)
        except Exception:
            ref_imgs = [ref_imgs]

    if not gen_path or not osp.isfile(gen_path):
        return dict(
            tier=tier, task_type=task_type,
            concept_score=0, task_score=0, integration_score=0,
            metric=0.0, correct=0,
            reasoning=f'generated image missing: {gen_path!r}',
            raw='',
        )

    def _generate_with_retry(messages, parser=None):
        """Call the judge, retrying on API failure AND on parse failure.

        ⛔ Retrying only on API errors is not enough: all three parsers map an
        unparseable completion to score 0, so a malformed or truncated judge
        response would be silently recorded as a genuine zero -- the exact
        failure mode this grader is supposed to surface. When ``parser`` is
        given, a response that it cannot parse also consumes a retry, and the
        last response is returned only after every attempt is used up.
        """
        retry = 3
        last = None
        while retry:
            retry -= 1
            try:
                out = model.generate(messages)
            except Exception as e:
                out = f'Failed to obtain answer via API: {e}'
            if 'Failed to obtain answer via API' in str(out):
                continue
            last = out
            if parser is None or parser(str(out)).get('parsed_ok'):
                return out
        return last or ''

    # ---- Concept judge: mean over each concept in the prompt --------
    concept_scores = []
    concept_raws = []
    for c, t, ref in zip(concepts, types, ref_imgs):
        ptext = concept_prompt(concept=str(c), type_=str(t), sentence=sentence)
        msgs = build_concept_messages(
            prompt_text=ptext,
            generated_image_path=gen_path,
            concept_image_path=str(ref),
        )
        out = _generate_with_retry(msgs, parse_concept_rating)
        parsed = parse_concept_rating(str(out))
        concept_scores.append(parsed['concept_score'])
        concept_raws.append(parsed['raw'])
    # ⛔ 保留小数:rating 2 和 3 的均值是 2.5,int(round()) 会压成 2
    concept_score = (sum(concept_scores) / len(concept_scores)) if concept_scores else 0.0

    # ---- Task judge: SKCI + MKCC ----------------------------------
    task_score = None
    task_raw = ''
    if tier == 'SKCI' and concepts:
        addition_str = additions[0] if additions else ''
        ptext = task_prompt_skci(
            task_type=task_type,
            concept=str(concepts[0]),
            type_=str(types[0]),
            addition=str(addition_str),
        )
        out = _generate_with_retry(build_task_messages(ptext, gen_path), parse_task_score)
        parsed = parse_task_score(str(out))
        task_score = parsed['task_score']
        task_raw = parsed['raw']
    elif tier == 'MKCC' and len(concepts) >= 2:
        ptext = task_prompt_mkcc(
            task_type=task_type,
            concepts=[str(c) for c in concepts[:2]],
            types=[str(t) for t in types[:2]],
            additions=[str(a) for a in additions],
        )
        out = _generate_with_retry(build_task_messages(ptext, gen_path), parse_task_score)
        parsed = parse_task_score(str(out))
        task_score = parsed['task_score']
        task_raw = parsed['raw']

    # ---- Integration judge: MKCC only --------------------------------
    integration_score = None
    integration_raw = ''
    if tier == 'MKCC' and len(concepts) >= 2:
        ptext = integration_prompt(
            concepts=[str(c) for c in concepts[:2]],
            types=[str(t) for t in types[:2]],
            sentence=sentence,
        )
        out = _generate_with_retry(build_integration_messages(ptext, gen_path), parse_integration_score)
        parsed = parse_integration_score(str(out))
        integration_score = parsed['integration_score']
        integration_raw = parsed['raw']

    metric = metric_for_tier(tier, concept_score, task_score, integration_score)
    return dict(
        tier=tier,
        task_type=task_type,
        concept_score=concept_score,
        task_score=int(task_score) if task_score is not None else -1,
        integration_score=int(integration_score) if integration_score is not None else -1,
        metric=float(metric),
        correct=int(metric >= 0.5),
        reasoning=(
            f'C={concept_score}'
            f'{f", T={task_score}" if task_score is not None else ""}'
            f'{f", I={integration_score}" if integration_score is not None else ""}'
            f' → metric={metric:.3f}'
        ),
        raw=_join_raws(concept_raws, task_raw, integration_raw),
    )


# ---------------------------------------------------------------------------
# Generated-image resolution (mirrors KnowGen / WISE)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class T2IFactualBenchDataset(ImageBaseDataset):
    """T2I-FactualBench — 3028 samples across 3 tiers (SKCM / SKCI / MKCC).

    Concept reference images are downloaded from HuggingFace on first load
    and cached under ``$LMUData/T2IFactualBench_concept_images/``.
    Reference-image paths in the released JSONL are rewritten to point
    inside this local snapshot.
    """

    TYPE = 'VQA'

    DATASET_URL = {
        'T2IFactualBench': '',
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
        prompts_dir = _resolve_prompts_dir()
        concept_root = _snapshot_download_concept_images()

        rows: List[Dict[str, Any]] = []
        for tier, (subdir, fname) in _TIER_FILES.items():
            jsonl_path = osp.join(prompts_dir, subdir, fname)
            if not osp.isfile(jsonl_path):
                raise FileNotFoundError(f'Missing tier file: {jsonl_path}')
            row_id = 0
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    concepts = obj.get('concept', []) or []
                    types = obj.get('type', []) or []
                    additions = obj.get('addition', []) or []
                    sentence = str(obj.get('sentence', ''))
                    task = str(obj.get('task', ''))
                    refs_raw = obj.get('reference_image', []) or []
                    refs_local = [_rewrite_ref_image_path(r, concept_root) for r in refs_raw]

                    row_id += 1
                    rows.append({
                        'index': f'{tier}_{row_id:04d}',
                        'tier': tier,
                        'task_type': task,
                        'concept': json.dumps(concepts, ensure_ascii=False),
                        'type': json.dumps(types, ensure_ascii=False),
                        'addition': json.dumps(additions, ensure_ascii=False),
                        'reference_image': json.dumps(refs_local, ensure_ascii=False),
                        'sentence': sentence,
                        'prompt': sentence,
                        'question': sentence,
                        'answer': '',
                        'category': tier,             # top-level = tier
                        'l2-category': f'{tier}/{task}',
                    })
        return pd.DataFrame(rows)

    # -- Prompt: text-only for T2I generation -----------------------------

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        return [dict(type='text', value=str(line.get('sentence', line.get('prompt', ''))))]

    # -- Evaluation -------------------------------------------------------

    def evaluate(self, eval_file, **judge_kwargs):
        logger = get_logger('Evaluation')
        data = load(eval_file)
        assert 'prediction' in data, 'prediction column missing from eval file'
        data['prediction'] = [str(x) for x in data['prediction']]

        # Merge metadata by index
        meta = self.data
        idx_str = lambda v: str(v)
        cols_to_merge = ['tier', 'task_type', 'concept', 'type', 'addition',
                         'reference_image', 'sentence', 'prompt']
        for col in cols_to_merge:
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

        # Parse list-fields back from JSON string (for the grader function)
        for col in ('concept', 'type', 'addition', 'reference_image'):
            data[col] = [json.loads(v) if isinstance(v, str) and v else [] for v in data[col]]

        judge_kwargs = dict(judge_kwargs)
        nproc = judge_kwargs.pop('nproc', 4)
        judge_kwargs.pop('enable_llm_judge', None)

        judge_name = (
            os.environ.get('MMDR_JUDGE_PROMPT')
            or judge_kwargs.pop('judge_prompt', None)
            or 'judge_t2i_factualbench'
        ).strip().lower()
        # Unconditional pop: the or-chain above short-circuits when the env
        # var is set, which would leave judge_prompt in judge_kwargs and leak
        # it into the API payload (cf. livevqa.py).
        judge_kwargs.pop('judge_prompt', None)
        if judge_name != 'judge_t2i_factualbench':
            raise ValueError(
                f"T2IFactualBench only supports 'judge_t2i_factualbench'; got {judge_name!r}."
            )

        # Full-detail image input for the judge (framework default is 'low')
        judge_kwargs.setdefault('img_detail', 'high')
        model = build_judge(**judge_kwargs)
        if model is None or not model.working():
            logger.error(
                'T2IFactualBench evaluation requires a working multimodal judge.'
            )
            logger.error(DEBUG_MESSAGE)
            return pd.DataFrame()

        grade_fn = _grade_item_judge_t2i_factualbench
        stem_base = '_t2i_factualbench_judge'
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

        # Populate columns
        for col, default in [
            ('concept_score', 0),
            ('task_score', -1),
            ('integration_score', -1),
            ('metric', 0.0),
            ('correct', 0),
            ('reasoning', ''),
            # 判分原文必须落盘:三轮 judge 的完整回复是复盘错分的唯一证据
            ('raw', ''),
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
        logger.info(f'T2IFactualBench [{judge_name}] accuracy saved to {score_file}')
        return acc

    @staticmethod
    def _compute_accuracy(data) -> pd.DataFrame:
        rows = []

        def _bloc(df: pd.DataFrame, name: str):
            if len(df) == 0:
                return None
            row = {'group': name, 'n': len(df)}
            for k in ('metric', 'correct', 'concept_score'):
                if k in df.columns:
                    vals = pd.to_numeric(df[k], errors='coerce').fillna(0.0)
                    row[k] = float(vals.mean())
            return row

        # Overall
        blk = _bloc(data, 'Overall')
        if blk is not None:
            rows.append(blk)

        # Per tier
        for tier in ('SKCM', 'SKCI', 'MKCC'):
            sub = data[data['tier'] == tier]
            blk = _bloc(sub, f'tier:{tier}')
            if blk is not None:
                rows.append(blk)

        # Per task type
        if 'task_type' in data.columns:
            for task, sub in data.groupby('task_type'):
                blk = _bloc(sub, f'task:{task}')
                if blk is not None:
                    rows.append(blk)

        return pd.DataFrame(rows)
