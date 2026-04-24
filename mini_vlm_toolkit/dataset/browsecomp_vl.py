"""BrowseComp-VL benchmark dataset.

A visual web-browsing comprehension benchmark with two difficulty levels
(L1 and L2) across 16 domains.  The model receives an image and a question,
and must produce a free-form text answer.

Evaluation follows the original BrowseComp-VL protocol
(``evaluate_hle_official.py``):
  - Judge extracts the final answer from the model's response, then compares
    against the ground truth.
  - Binary grading: correct (yes) or incorrect (no).
  - Reports accuracy per level (L1/L2), per domain, and overall.

TSV files are produced by ``scripts/convert_browsecomp_vl.py`` and placed in
``$LMUData/``.
"""

import json
import re

import pandas as pd

from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
from ..smp.file import get_intermediate_file_path
from ..utils import track_progress_rich

# ---------------------------------------------------------------------------
# Judge prompt — shared HLE-style protocol, imported from utils/hle_judge.py
# (byte-identical to what the BrowseComp-VL upstream ``evaluate_hle_official.py``
# uses, and the same prompt now applied to hle / mmsearch / mmsearch_plus /
# simplevqa_en / vdrbench for cross-benchmark consistency).
# ---------------------------------------------------------------------------
from .utils.hle_judge import HLE_JUDGE_PROMPT as JUDGE_PROMPT, parse_hle_judgement


# ---------------------------------------------------------------------------
# Per-item grading helper
# ---------------------------------------------------------------------------

def _grade_item(model, item):
    """Grade a single prediction using the BrowseComp-VL JUDGE_PROMPT.

    Returns a dict with keys: correct (1 or 0), extracted_answer, reasoning.
    """
    prediction = str(item.get('prediction', ''))
    gt_list = item.get('ground_truth_list', '')
    question = item.get('question', '')

    # Parse ground_truth_list from TSV string repr
    if isinstance(gt_list, str):
        try:
            gt_list = json.loads(gt_list)
        except Exception:
            gt_list = [gt_list]
    if not gt_list:
        return dict(correct=0, extracted_answer='', reasoning='Empty ground truth')

    # Use the first answer as the correct_answer (matches original code line 76)
    correct_answer = gt_list[0]

    prompt = JUDGE_PROMPT.format(
        question=question,
        correct_answer=correct_answer,
        response=prediction,
    )

    retry = 3
    while retry:
        ans = model.generate(prompt)
        if 'Failed to obtain answer via API' in ans:
            retry -= 1
            continue
        parsed = parse_hle_judgement(ans)
        # parse_hle_judgement signals "couldn't parse" via this reasoning
        # string; on that case, retry. Any other verdict (correct=0 or 1)
        # is a legitimate judge answer.
        if parsed.get('reasoning') != 'judge failed to parse':
            return dict(
                correct=parsed['correct'],
                extracted_answer=parsed.get('extracted_answer', ''),
                reasoning=parsed.get('reasoning', ''),
            )
        retry -= 1

    return dict(correct=0, extracted_answer='', reasoning='Judge failed')


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class BrowseCompVLDataset(ImageBaseDataset):

    TYPE = 'VQA'

    DATASET_URL = {
        'BrowseCompVL': '',
    }
    DATASET_MD5 = {}

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_URL)

    # -- Prompt: open-ended VQA (question only) -----------------------------

    def build_prompt(self, line):
        """Build a prompt with image + question.

        The model receives the image and the question directly, matching the
        original BrowseComp-VL inference setup.
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = toliststr(line['image_path'])

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]

        msgs.append(dict(type='text', value=line['question']))
        return msgs

    # -- Evaluation: binary judge grading (yes/no) --------------------------

    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate predictions using the original BrowseComp-VL grading protocol."""
        logger = get_logger('Evaluation')
        data = load(eval_file)
        assert 'prediction' in data, 'prediction column missing from eval file'
        data['prediction'] = [str(x) for x in data['prediction']]

        # Merge ground_truth_list from dataset metadata
        meta = self.data
        gt_map = {
            str(idx): gt for idx, gt in zip(meta['index'], meta['ground_truth_list'])
        }
        data['ground_truth_list'] = [
            gt_map.get(str(idx), '') for idx in data['index']
        ]

        # Build judge model
        nproc = judge_kwargs.pop('nproc', 4)
        model = build_judge(**judge_kwargs)
        if model is None or not model.working():
            logger.error(
                'BrowseCompVL evaluation requires a working judge model. '
                'Please configure judge_kwargs properly.'
            )
            logger.error(DEBUG_MESSAGE)
            return pd.DataFrame()

        # Run grading
        result_file = get_intermediate_file_path(eval_file, '_bcvl_grade', 'pkl')
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
                _grade_item, tups, nproc=nproc, chunksize=nproc,
                save=result_file, keys=keys,
            )
            result_cache = load(result_file)
            for k, v in zip(keys, res):
                if k not in result_cache:
                    result_cache[k] = v

        # Map results back to data
        data['correct'] = [
            result_cache.get(str(idx), {}).get('correct', 0)
            for idx in data['index']
        ]
        data['extracted_answer'] = [
            result_cache.get(str(idx), {}).get('extracted_answer', '')
            for idx in data['index']
        ]
        data['reasoning'] = [
            result_cache.get(str(idx), {}).get('reasoning', '')
            for idx in data['index']
        ]

        # Save detailed results
        detail_file = get_intermediate_file_path(eval_file, '_bcvl_detail')
        dump(data, detail_file)

        # Compute accuracy
        acc = self._compute_accuracy(data)
        score_file = get_intermediate_file_path(eval_file, '_bcvl_acc', 'csv')
        dump(acc, score_file)
        logger.info(f'BrowseComp-VL accuracy results saved to {score_file}')
        return acc

    @staticmethod
    def _compute_accuracy(data):
        """Compute accuracy breakdown per level and per domain.

        Returns a DataFrame with rows for Overall, L1, L2, and per-domain,
        with columns: total, correct, accuracy.
        """
        rows = []

        def _acc_row(name, subset):
            total = len(subset)
            correct = int(subset['correct'].sum())
            return {
                'split': name,
                'total': total,
                'correct': correct,
                'accuracy': round(correct / total * 100, 2) if total > 0 else 0.0,
            }

        rows.append(_acc_row('Overall', data))

        # Per level (category column: L1 / L2)
        if 'category' in data:
            for level in sorted(data['category'].unique()):
                rows.append(_acc_row(level, data[data['category'] == level]))

        # Per domain (l2-category column)
        if 'l2-category' in data:
            for domain in sorted(data['l2-category'].unique()):
                rows.append(_acc_row(domain, data[data['l2-category'] == domain]))

        return pd.DataFrame(rows)
