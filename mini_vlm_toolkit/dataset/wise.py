"""WISE-Verified benchmark dataset.

WISE (World-knowledge-Informed Semantic Evaluation) is a knowledge-intensive
text-to-image benchmark from Niu et al. 2025 (arXiv 2503.07265). 1000
prompts across 6 knowledge domains (CULTURE / TIME / SPACE / BIOLOGY /
PHYSICS / CHEMISTRY). Each prompt has a short natural-language
description whose visual answer must be *inferred* via world-knowledge
reasoning (not stated in the prompt surface).

Evaluation:
  - No GT reference image.
  - A judge LLM sees (PROMPT, EXPLANATION, generated_image) and outputs
    a binary Score in {0, 1} per the WISE protocol.
  - Overall score = mean of per-sample scores; per-category breakdown
    follows the paper's prompt_id ranges (see judge_wise.py).

Data source: cloned WISE repo at ``/home/<user>/code/WISE/data_verified/``.
Override via env var ``WISE_DATA_DIR`` (should contain merge.json).

Record schema (cached TSV under ``$LMUData/WISE.tsv``):
  - ``index``       : str (prompt_id as string, "1"..."1000")
  - ``prompt``      : str (original text-to-image prompt)
  - ``explanation`` : str (paper's reference interpretation)
  - ``question``    : str (alias of prompt; kept for framework compat)
  - ``answer``      : str (placeholder empty; WISE has no text GT)
  - ``category``    : str (paper category: CULTURE / TIME / ...)
  - ``l2-category`` : str (Subcategory from JSON, e.g. "Cultural knowledge/Festival")

Prediction resolution: mirrors KnowGen. The generated image lives at
``<pred_root>/full_trajectories/images/<task_id>_*.png`` (needs
``--save-images files`` set at run time).

Judge: only ``judge_wise`` (binary 0/1). The paper uses Qwen3.5-35B-A3B
as judge; we default to whatever ``judge_rui.yaml`` specifies (typically
gpt-4o). Judge-model tier gap should be noted when comparing to
paper-reported numbers (e.g., Qwen-Image-Agent Table 2).
"""

import glob
import json
import os
import os.path as osp
from typing import Any, Dict, List, Optional

import pandas as pd

from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
from ..smp.file import LMUDataRoot, get_intermediate_file_path
from ..utils import track_progress_rich


# ---------------------------------------------------------------------------
# Source data — local WISE repo (data_verified/merge.json)
# ---------------------------------------------------------------------------
_DEFAULT_WISE_DATA_DIR = '/home/<user>/code/WISE/data_verified'
_MERGE_JSON = 'merge.json'


def _resolve_wise_data_dir() -> str:
    """Return the local ``WISE/data_verified/`` directory.

    Priority: env var ``WISE_DATA_DIR`` → default local checkout path.
    Errors with clear instructions if neither exists.
    """
    d = os.environ.get('WISE_DATA_DIR', _DEFAULT_WISE_DATA_DIR)
    d = osp.abspath(d)
    merge_p = osp.join(d, _MERGE_JSON)
    if not osp.isfile(merge_p):
        raise FileNotFoundError(
            f'WISE merge.json not found at {merge_p!r}. '
            'Clone https://github.com/PKU-Alignment/WISE and either put it at '
            f'{_DEFAULT_WISE_DATA_DIR} or set env var WISE_DATA_DIR to point at '
            'its data_verified/ directory.'
        )
    return d


# ---------------------------------------------------------------------------
# Per-sample grader
# ---------------------------------------------------------------------------

def _grade_item_judge_wise(model, item):
    """Invoke the WISE binary judge on one sample.

    Requires:
      - ``item['prompt']``          : original T2I prompt
      - ``item['explanation']``     : paper's reference interpretation
      - ``item['generated_image']`` : path to model's generated image

    Returns dict with ``correct`` (0/1), ``reasoning``, ``raw``.
    """
    from .utils.judge_wise import (
        build_judge_wise_messages,
        parse_judge_wise_judgement,
    )

    prompt = str(item.get('prompt', ''))
    explanation = str(item.get('explanation', ''))
    gen_path = str(item.get('generated_image', ''))

    if not gen_path or not osp.isfile(gen_path):
        return dict(
            correct=0,
            score=0.0,
            reasoning=f'generated image missing: {gen_path!r}',
            raw='',
        )

    messages = build_judge_wise_messages(
        prompt_text=prompt,
        explanation_text=explanation,
        generated_image_path=gen_path,
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
        parsed = parse_judge_wise_judgement(out)
        if parsed.get('score') is not None:
            return dict(
                correct=parsed['correct'],
                score=float(parsed['score']),
                reasoning=parsed['reasoning'],
                raw=parsed['raw'],
            )
        retry -= 1

    return dict(
        correct=0,
        score=0.0,
        reasoning=f'judge_wise unparseable: {str(last)[:200]}' if last else 'judge unreachable',
        raw=str(last or '')[:500],
    )


# ---------------------------------------------------------------------------
# Generated-image resolution from the run directory (mirrors KnowGen)
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

class WISEDataset(ImageBaseDataset):
    """WISE-Verified (arXiv 2503.07265, Niu et al. 2025).

    1000 knowledge-intensive text-to-image prompts. No GT image; the LLM
    judge sees (prompt, explanation, generated_image) and emits binary
    Score in {0, 1}.
    """

    TYPE = 'VQA'

    DATASET_URL = {
        'WISE': '',
    }
    DATASET_MD5 = {}

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_URL)

    # -- Data loading -----------------------------------------------------

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
        """Read the WISE merge.json (1000 rows) and return a DataFrame."""
        from .utils.judge_wise import wise_category_from_prompt_id

        data_dir = _resolve_wise_data_dir()
        merge_path = osp.join(data_dir, _MERGE_JSON)
        with open(merge_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f'Expected list in WISE merge.json, got {type(data)}')

        rows: List[Dict[str, Any]] = []
        for obj in data:
            pid = obj.get('prompt_id')
            if pid is None:
                continue
            prompt = str(obj.get('Prompt', ''))
            explanation = str(obj.get('Explanation', ''))
            category_json = str(obj.get('Category', 'unknown') or 'unknown')
            subcategory = str(obj.get('Subcategory', 'unknown') or 'unknown')
            paper_cat = wise_category_from_prompt_id(pid)

            rows.append({
                'index': str(pid),
                'prompt': prompt,
                'question': prompt,          # framework compat alias
                'answer': '',                # no text GT
                'explanation': explanation,
                'category': paper_cat,       # paper aggregation category
                'l2-category': f'{category_json}/{subcategory}',
            })

        return pd.DataFrame(rows)

    # -- Prompt: text-only input for image generation ---------------------

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        prompt = str(line.get('prompt', line.get('question', '')))
        # WISE is a text-to-image GENERATION task; the prompt is an implicit,
        # knowledge-grounded description (not a question). Frame it explicitly as
        # generation so the agent produces an image rather than answering in text
        # (see personal memory gen_eval_framework_gaps: ~half of bare prompts got
        # answered textually). The judge still scores against meta['prompt'].
        instruction = (
            'Text-to-image generation task. The description below is an implicit, '
            'knowledge-grounded prompt: infer what must appear using world knowledge '
            '(search for facts or visual references when helpful), then GENERATE an '
            'image that realizes it. You MUST produce an image using the image '
            'generation tool as your final result; do not answer with text only.\n\n'
            f'Prompt to realize: {prompt}'
        )
        return [dict(type='text', value=instruction)]

    # -- Evaluation: WISE binary judge ------------------------------------

    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate WISE predictions with the paper's binary protocol.

        For each row, resolves the generated image from
        ``<pred_root>/full_trajectories/images/<task_id>_*.png``, invokes
        the LLM judge with (prompt, explanation, image), extracts a binary
        score, and aggregates per-category means.
        """
        logger = get_logger('Evaluation')
        data = load(eval_file)
        assert 'prediction' in data, 'prediction column missing from eval file'
        data['prediction'] = [str(x) for x in data['prediction']]

        # Merge fields from dataset metadata
        meta = self.data
        idx_str = lambda v: str(v)
        prompt_map = {idx_str(idx): p for idx, p in zip(meta['index'], meta['prompt'])}
        expl_map = {idx_str(idx): e for idx, e in zip(meta['index'], meta['explanation'])}

        data['prompt'] = [prompt_map.get(idx_str(i), '') for i in data['index']]
        data['explanation'] = [expl_map.get(idx_str(i), '') for i in data['index']]

        # Resolve generated images from the run directory
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
            or 'judge_wise'
        ).strip().lower()
        # Unconditional pop: the or-chain above short-circuits when the env
        # var is set, which would leave judge_prompt in judge_kwargs and leak
        # it into the API payload (cf. livevqa.py).
        judge_kwargs.pop('judge_prompt', None)
        if judge_name != 'judge_wise':
            raise ValueError(
                f"WISE only supports 'judge_wise'; got {judge_name!r}. "
                "Set MMDR_JUDGE_PROMPT=judge_wise or omit."
            )

        # Full-detail image input for the judge (framework default is 'low')
        judge_kwargs.setdefault('img_detail', 'high')
        model = build_judge(**judge_kwargs)
        if model is None or not model.working():
            logger.error(
                'WISE evaluation requires a working multimodal judge. '
                'Please configure judge_kwargs properly.'
            )
            logger.error(DEBUG_MESSAGE)
            return pd.DataFrame()

        grade_fn = _grade_item_judge_wise
        stem_base = '_wise_judge_wise'
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

        # Populate score columns
        for col, default in [
            ('correct', 0),
            ('score', 0.0),
            ('reasoning', ''),
            ('raw', ''),   # full judge response (was computed but not persisted)
        ]:
            data[col] = [
                result_cache.get(str(idx), {}).get(col, default)
                for idx in data['index']
            ]

        # Detailed dump
        detail_file = get_intermediate_file_path(eval_file, detail_stem)
        dump(data, detail_file)

        # Accuracy breakdown by paper category + subcategory
        acc = self._compute_accuracy(data)
        score_file = get_intermediate_file_path(eval_file, acc_stem, 'csv')
        dump(acc, score_file)
        logger.info(f'WISE [{judge_name}] accuracy saved to {score_file}')

        return acc

    @staticmethod
    def _compute_accuracy(data) -> pd.DataFrame:
        """Aggregate WISE scores overall + per paper category + per l2-category."""
        rows = []

        def _bloc(df: pd.DataFrame, name: str):
            if len(df) == 0:
                return None
            row = {'group': name, 'n': len(df)}
            for k in ('correct', 'score'):
                if k in df.columns:
                    vals = pd.to_numeric(df[k], errors='coerce').fillna(0.0)
                    row[k] = float(vals.mean())
            return row

        # Overall
        blk = _bloc(data, 'Overall')
        if blk is not None:
            rows.append(blk)

        # Per paper category
        if 'category' in data.columns:
            # Preserve paper order
            for cat in ('CULTURE', 'TIME', 'SPACE', 'BIOLOGY', 'PHYSICS', 'CHEMISTRY'):
                sub = data[data['category'] == cat]
                blk = _bloc(sub, f'category:{cat}')
                if blk is not None:
                    rows.append(blk)

        # Per l2 category (sub-topic)
        if 'l2-category' in data.columns:
            for l2, sub in data.groupby('l2-category'):
                blk = _bloc(sub, f'l2:{l2}')
                if blk is not None:
                    rows.append(blk)

        return pd.DataFrame(rows)
