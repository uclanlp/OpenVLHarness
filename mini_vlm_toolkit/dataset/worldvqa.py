"""WorldVQA dataset (Moonshot AI, arXiv 2602.02537v1).

"Measuring Atomic World Knowledge in Multimodal Large Language Models" —
a 3000-sample VQA benchmark measuring visual entity naming / grounding
across 8 categories and 3 difficulty levels, bilingual (zh + non-zh).

Data source: HuggingFace dataset ``moonshotai/WorldVQA`` (single TSV file
with columns ``index, category, question, answer, image, language,
difficulty``). Images are stored inline as base64 PNG in the TSV; this
dataset class passes them through as-is (framework accepts base64
strings for ``type: image`` entries).

Judge: byte-exact port of ``WorldVQA/eval/eval.py``'s Correct/Incorrect/
Unattempted classifier. Metric: accuracy = mean(correct) over all rows; a
judge error counts as incorrect, as in the other benchmarks here (upstream
drops such rows instead). Aggregation by category, difficulty, and language.
"""

import json
import os
import os.path as osp
from typing import Any, Dict, List, Optional

import pandas as pd

from .image_base import ImageBaseDataset
from .hf_release import HF_SPLITS, load_hf_split
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
from ..smp.file import LMUDataRoot, get_intermediate_file_path
from ..utils import track_progress_rich


_HF_REPO_ID = 'moonshotai/WorldVQA'
_HF_TSV_PATH = 'WorldVQA.tsv'


def _snapshot_download_worldvqa() -> str:
    from huggingface_hub import hf_hub_download

    data_root = osp.join(LMUDataRoot(), '_worldvqa_source')
    return hf_hub_download(
        repo_id=_HF_REPO_ID,
        filename=_HF_TSV_PATH,
        repo_type='dataset',
        local_dir=data_root,
    )


def _grade_item_judge_worldvqa(model, item):
    """Invoke WorldVQA judge on one sample."""
    from .utils.judge_worldvqa import (
        build_judge_worldvqa_prompt,
        parse_judge_worldvqa_judgement,
    )

    prediction = str(item.get('prediction', ''))
    # Strip <think>...</think> to match upstream (WorldVQA/eval.py:273-278)
    if '</think>' in prediction:
        prediction = prediction.split('</think>')[-1].strip()

    question = str(item.get('question', ''))
    gt_answer = str(item.get('answer', ''))

    prompt_text = build_judge_worldvqa_prompt(
        question=question,
        model_answer=prediction,
        ground_truth_answer=gt_answer,
    )

    retry = 3
    last = None
    while retry:
        try:
            out = model.generate(prompt_text)
        except Exception as e:
            out = f'Failed to obtain answer via API: {e}'
        if 'Failed to obtain answer via API' in str(out):
            retry -= 1
            continue
        last = out
        parsed = parse_judge_worldvqa_judgement(str(out))
        return dict(
            correct=parsed['correct'],
            category=parsed['category'],
            reasoning=parsed['reasoning'],
        )

    return dict(
        correct=0,
        category='error',
        reasoning=f'judge_worldvqa unreachable: {str(last)[:200]}' if last else 'judge unreachable',
    )


def _grade_item_generic(model, item, judge_name):
    """Grade one sample with a framework-shared judge (hle / judge_rs / judge_vdr).

    Mirrors the LIVEVQA / SimpleVQA dispatch. Binary judges have no
    Correct/Incorrect/Unattempted 3-way label, so ``category`` is set to
    ``'Correct'``/``'Incorrect'`` from the binary verdict (never
    ``'Unattempted'``).
    """
    prediction = str(item.get('prediction', ''))
    question = str(item.get('question', ''))
    answer = str(item.get('answer', ''))

    try:
        if judge_name == 'judge_rs':
            from .utils.judge_rs import grade_with_judge_rs
            parsed = grade_with_judge_rs(model, question, answer, prediction)
        elif judge_name == 'judge_vdr':
            from .utils.judge_vdr import grade_with_judge_vdr
            parsed = grade_with_judge_vdr(model, question, answer, prediction)
        else:  # 'hle'
            from .utils.hle_judge import grade_with_hle_judge
            parsed = grade_with_hle_judge(model, question, answer, prediction)
    except Exception as e:
        return dict(correct=0, category='error', reasoning=f'{judge_name} error: {e}')
    correct = int(parsed.get('correct', 0))
    return dict(
        correct=correct,
        category='Correct' if correct else 'Incorrect',
        reasoning=str(parsed.get('reasoning', '')),
    )


class WorldVQADataset(ImageBaseDataset):
    """WorldVQA — 3000 samples measuring atomic world knowledge.

    Two addressable sets:
      * ``WorldVQA``         all 3000 rows (upstream full set)
      * ``WorldVQA_VDR300``  the 300 rows Vision-DeepResearch reports on, hosted on the HF eval repo
    """

    TYPE = 'VQA'

    DATASET_URL = {
        'WorldVQA': '',
        'WorldVQA_VDR300': '',
    }
    DATASET_MD5 = {}

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_URL)

    def load_data(self, dataset):
        """``WorldVQA_VDR300`` is read from the HF eval repo as is; ``WorldVQA`` is the upstream TSV
        (embedded base64 images), cached under ``$LMUData/WorldVQA.tsv``."""
        if dataset in HF_SPLITS:
            return load_hf_split(self, dataset)
        data_root = LMUDataRoot()
        tsv_path = osp.join(data_root, 'WorldVQA.tsv')
        if not osp.exists(tsv_path):
            src = _snapshot_download_worldvqa()
            os.makedirs(data_root, exist_ok=True)
            # Copy through to LMUData location; the source is already TSV
            import shutil
            shutil.copyfile(src, tsv_path)
        self.data_path = tsv_path
        return load(tsv_path)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        # WorldVQA ships images inline as base64 PNG. Decode them to files via
        # the standard dump_image() and pass file paths — BaseAgent.preproc_content
        # resolves image items through parse_file (i.e. expects a file path), so a
        # raw base64 value trips its `assert item['type'] == 'text'` guard.
        msgs: List[Dict[str, Any]] = [dict(type='image', value=p) for p in self.dump_image(line)]

        question = str(line.get('question', ''))
        is_zh = str(line.get('language', 'en')).lower() == 'zh'
        if is_zh:
            prefix = '请尽可能提供详细的回答。\n'
        else:
            prefix = 'Please provide as much detail as possible in your answer.\n'
        msgs.append(dict(type='text', value=f'{prefix}{question}'))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        logger = get_logger('Evaluation')
        data = load(eval_file)
        assert 'prediction' in data, 'prediction column missing from eval file'
        data['prediction'] = [str(x) for x in data['prediction']]

        # Merge question / answer / difficulty / category / language
        meta = self.data
        idx_str = lambda v: str(v)
        for col in ('question', 'answer', 'category', 'difficulty', 'language'):
            if col in meta.columns:
                mp = {idx_str(i): v for i, v in zip(meta['index'], meta[col])}
                data[col] = [mp.get(idx_str(i), '') for i in data['index']]

        judge_kwargs = dict(judge_kwargs)
        nproc = judge_kwargs.pop('nproc', 4)
        judge_kwargs.pop('enable_llm_judge', None)

        judge_name = (
            os.environ.get('MMDR_JUDGE_PROMPT')
            or judge_kwargs.pop('judge_prompt', None)
            or 'judge_worldvqa'
        ).strip().lower()
        # Unconditional pop: the or-chain above short-circuits when the env
        # var is set, which would leave judge_prompt in judge_kwargs and leak
        # it into the API payload (cf. livevqa.py).
        judge_kwargs.pop('judge_prompt', None)
        if judge_name not in ('judge_worldvqa', 'hle', 'judge_rs', 'judge_vdr'):
            raise ValueError(
                f"Unknown judge_prompt {judge_name!r}; expected 'judge_worldvqa' "
                "(default, upstream WorldVQA judge), 'hle', 'judge_rs', or 'judge_vdr'."
            )

        model = build_judge(**judge_kwargs)
        if model is None or not model.working():
            logger.error(
                'WorldVQA evaluation requires a working judge model. '
                'Please configure judge_kwargs properly.'
            )
            logger.error(DEBUG_MESSAGE)
            return pd.DataFrame()

        # Judge-specific output stems so switching judges never overwrites
        # another judge's grade / detail / acc files.
        stem_base = '_worldvqa_judge' if judge_name == 'judge_worldvqa' else f'_{judge_name}'
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
            if judge_name == 'judge_worldvqa':
                grade_fn = _grade_item_judge_worldvqa
                tups = [dict(model=model, item=x) for x in items_to_grade]
            else:
                grade_fn = _grade_item_generic
                tups = [
                    dict(model=model, item=x, judge_name=judge_name)
                    for x in items_to_grade
                ]
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
            ('category', ''),  # answer_category (not row category)
            ('reasoning', ''),
        ]:
            data[f'judge_{col}' if col == 'category' else col] = [
                result_cache.get(str(idx), {}).get(col, default)
                for idx in data['index']
            ]
        # Rename to avoid conflict with row category column
        if 'judge_category' in data.columns:
            data['answer_category'] = data.pop('judge_category')

        detail_file = get_intermediate_file_path(eval_file, detail_stem)
        dump(data, detail_file)

        acc = self._compute_accuracy(data)
        score_file = get_intermediate_file_path(eval_file, acc_stem, 'csv')
        dump(acc, score_file)
        logger.info(f'WorldVQA [{judge_name}] accuracy saved to {score_file}')
        return acc

    @staticmethod
    def _compute_accuracy(data) -> pd.DataFrame:
        rows = []

        def _bloc(df: pd.DataFrame, name: str):
            if len(df) == 0:
                return None
            row = {'group': name, 'n': len(df)}
            if 'correct' in df.columns:
                vals = pd.to_numeric(df['correct'], errors='coerce').fillna(0.0)
                row['accuracy'] = float(vals.mean())
            return row

        blk = _bloc(data, 'Overall')
        if blk is not None:
            rows.append(blk)

        # By difficulty
        for diff in ('easy', 'medium', 'hard'):
            sub = data[data.get('difficulty', pd.Series([], dtype=str)) == diff]
            blk = _bloc(sub, f'difficulty:{diff}')
            if blk is not None:
                rows.append(blk)

        # By language
        for lang in ('zh', 'non-zh'):
            sub = data[data.get('language', pd.Series([], dtype=str)) == lang]
            blk = _bloc(sub, f'language:{lang}')
            if blk is not None:
                rows.append(blk)

        # By category (may have many)
        if 'category' in data.columns:
            for cat, sub in data.groupby('category'):
                blk = _bloc(sub, f'category:{cat}')
                if blk is not None:
                    rows.append(blk)

        # By answer_category (correct / incorrect / unattempted breakdown)
        if 'answer_category' in data.columns:
            ac_counts = data['answer_category'].value_counts().to_dict()
            for ac, count in ac_counts.items():
                rows.append({'group': f'answer_category:{ac}', 'n': int(count)})

        return pd.DataFrame(rows)
