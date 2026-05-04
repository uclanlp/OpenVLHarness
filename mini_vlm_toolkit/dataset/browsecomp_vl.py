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
from .utils.judge_rs import (
    JUDGE_RS_PROMPT, JUDGE_RS_SYSTEM_PROMPT, JUDGE_RS_USER_PROMPT,
    JUDGE_RS_VERSION, parse_judge_rs_judgement,
)


# ---------------------------------------------------------------------------
# Auxiliary VDR-style judge prompt (opt-in placeholder)
#
# Adapted from Vision-DeepResearch's ``rllm/rewards/deepresearch_reward.py``
# judging_prompt — a lenient contains-based binary judge. This is *not*
# the BrowseComp-VL paper protocol; it exists to enable cross-benchmark
# consistency studies (VDR / LIVEVQA / VisBrowseBench use the same style)
# and is only invoked when the judge config sets ``enable_vdr_aux: true``.
# Output is written to a separate ``_bcvl_vdr.json`` file and does NOT
# override the primary original-protocol scores.
# ---------------------------------------------------------------------------
_VDR_JUDGE_PROMPT = """You are an impartial judge evaluating whether an AI assistant's answer contains the correct answer.

[Question]
{question}

[Correct Answer]
{correct_answer}

[Assistant's Answer]
{response}

Task: Determine if the assistant's answer contains the correct answer anywhere in its content.

Instructions:
1. Read through the entire answer carefully.
2. Look for the correct answer anywhere in the response (it may be embedded in paragraphs, reasoning, or explanations).
3. Check if the information in the response is consistent with the correct answer.
4. The answer does NOT need to be in a specific format or labeled as "final answer".
5. Provide your reasoning.
6. Answer "yes" if the response contains the correct answer (semantically equivalent is fine), "no" if it doesn't or contradicts it.

Respond **strictly in a valid JSON object** matching this schema:
{{
  "reasoning": string,
  "correct": "yes" | "no"
}}"""


# ---------------------------------------------------------------------------
# Per-item grading helpers
# ---------------------------------------------------------------------------

def _grade_item(model, item, judge_name: str = 'hle'):
    """Grade a single prediction using the selected judge protocol.

    ``judge_name`` selects the judge prompt + parser:
      - ``'hle'`` (default): paper-protocol HLE_JUDGE_PROMPT (extract +
        binary correct yes/no JSON). Reported numbers match the
        BrowseComp-VL paper.
      - ``'judge_rs'``: Rui's iterative judge protocol (more lenient, MM-DR
        11-rule pair). For exploratory comparison.

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

    if judge_name == 'judge_rs':
        # Judge_RS schema: ground_truth_answer + candidate_answers + model_response.
        # Send system + user as separate roles via ``generate_inner`` (skips
        # ``model.generate``'s ``preprocess_message_with_role`` which would
        # mutate ``self.system_prompt`` on the wrapper instance).
        rest = [str(x) for x in gt_list[1:] if str(x).strip()]
        candidates = ' | '.join(rest) if rest else '(none)'
        msgs_role_split = [
            {"role": "system", "content": [{"type": "text", "value": JUDGE_RS_SYSTEM_PROMPT}]},
            {"role": "user",   "content": [{"type": "text", "value": JUDGE_RS_USER_PROMPT.format(
                question=str(question),
                ground_truth_answer=str(correct_answer),
                candidate_answers=candidates,
                model_response=prediction,
            )}]},
        ]
        parser = parse_judge_rs_judgement
        unparseable_sentinel = 'judge_rs failed to parse'
        use_role_split = True
    else:
        prompt = JUDGE_PROMPT.format(
            question=question,
            correct_answer=correct_answer,
            response=prediction,
        )
        parser = parse_hle_judgement
        unparseable_sentinel = 'judge failed to parse'
        use_role_split = False

    retry = 3
    while retry:
        if use_role_split:
            try:
                code, ans, _log = model.generate_inner(msgs_role_split)
            except Exception:
                retry -= 1
                continue
            if code != 0 or not ans:
                retry -= 1
                continue
        else:
            ans = model.generate(prompt)
            if 'Failed to obtain answer via API' in ans:
                retry -= 1
                continue
        parsed = parser(ans)
        # parser signals "couldn't parse" via this reasoning string; on
        # that case, retry. Any other verdict (correct=0 or 1) is a
        # legitimate judge answer.
        if parsed.get('reasoning') != unparseable_sentinel:
            return dict(
                correct=parsed['correct'],
                extracted_answer=parsed.get('extracted_answer', ''),
                reasoning=parsed.get('reasoning', ''),
            )
        retry -= 1

    return dict(correct=0, extracted_answer='', reasoning='Judge failed')


def _grade_item_vdr(model, item):
    """Auxiliary VDR-style contains-based binary judge.

    Returns dict with ``verdict`` ('correct' / 'incorrect' / 'error') and
    ``reasoning``. Invoked only when ``enable_vdr_aux`` is set.
    """
    prediction = str(item.get('prediction', ''))
    gt_list = item.get('ground_truth_list', '')
    question = item.get('question', '')

    if isinstance(gt_list, str):
        try:
            gt_list = json.loads(gt_list)
        except Exception:
            gt_list = [gt_list]
    if not gt_list:
        return dict(verdict='error', reasoning='empty ground_truth_list')

    correct_answer = gt_list[0]
    prompt = _VDR_JUDGE_PROMPT.format(
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

        m = re.search(r'\{[\s\S]*\}', ans)
        if m:
            try:
                obj = json.loads(m.group(0))
                verdict = str(obj.get('correct', '')).strip().lower()
                if verdict in ('yes', 'y', 'true', 'positive'):
                    return dict(verdict='correct',
                                reasoning=str(obj.get('reasoning', '')))
                if verdict in ('no', 'n', 'false', 'negative'):
                    return dict(verdict='incorrect',
                                reasoning=str(obj.get('reasoning', '')))
            except (json.JSONDecodeError, ValueError):
                pass

        raw = ans.strip().lower()
        if 'correct: yes' in raw or raw in ('yes', 'y'):
            return dict(verdict='correct', reasoning='')
        if 'correct: no' in raw or raw in ('no', 'n'):
            return dict(verdict='incorrect', reasoning='')

        retry -= 1

    return dict(verdict='error', reasoning='judge unreachable after retries')


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
        # ``enable_vdr_aux`` is a framework-level flag (not an OpenAI
        # parameter). Pop it before passing kwargs to build_judge() so it
        # doesn't leak into the OpenAI request payload and trigger 400.
        enable_vdr_aux = bool(judge_kwargs.pop('enable_vdr_aux', False))
        # Optional alternative judge protocol. Default = paper-protocol HLE.
        # Override via judge_kwargs['judge_prompt'] = 'judge_rs' OR env
        # MVT_JUDGE_PROMPT=judge_rs. Filename suffix '_judge_rs' is added
        # to scoring artifacts so HLE + Judge_RS results coexist in the
        # same run dir without overwriting each other.
        judge_name = (
            os.environ.get('MVT_JUDGE_PROMPT')
            or judge_kwargs.pop('judge_prompt', None)
            or 'hle'
        ).lower()
        if judge_name not in ('hle', 'judge_rs'):
            raise ValueError(
                f"Unknown judge_prompt {judge_name!r}; supported: 'hle' (default), 'judge_rs'"
            )
        judge_suffix = '' if judge_name == 'hle' else f'_{judge_name}'
        model = build_judge(**judge_kwargs)
        if model is None or not model.working():
            logger.error(
                'BrowseCompVL evaluation requires a working judge model. '
                'Please configure judge_kwargs properly.'
            )
            logger.error(DEBUG_MESSAGE)
            return pd.DataFrame()

        # Run grading
        result_file = get_intermediate_file_path(eval_file, f'_bcvl_grade{judge_suffix}', 'pkl')
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
            tups = [dict(model=model, item=x, judge_name=judge_name) for x in items_to_grade]
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

        # Save detailed results (filename includes judge_suffix so HLE +
        # Judge_RS coexist in the same run dir without overwriting).
        detail_file = get_intermediate_file_path(eval_file, f'_bcvl_detail{judge_suffix}')
        dump(data, detail_file)

        # Compute accuracy
        acc = self._compute_accuracy(data)
        score_file = get_intermediate_file_path(eval_file, f'_bcvl_acc{judge_suffix}', 'csv')
        dump(acc, score_file)
        logger.info(f'BrowseComp-VL accuracy results saved to {score_file}')

        # -- Auxiliary VDR judge pass (opt-in via enable_vdr_aux) ----------
        # Does NOT override the primary original-protocol scores above.
        # Produces a separate ``_bcvl_vdr.json`` side-output for
        # cross-benchmark studies that want VDR-style contains-based grading.
        if enable_vdr_aux:
            vdr_cache_file = get_intermediate_file_path(
                eval_file, '_bcvl_vdr_grade', 'pkl')
            vdr_cache = load(vdr_cache_file) if osp.exists(vdr_cache_file) else {}
            vdr_items = [
                data.iloc[i] for i in range(len(data))
                if str(data.iloc[i]['index']) not in vdr_cache
            ]
            if vdr_items:
                vdr_tups = [dict(model=model, item=x) for x in vdr_items]
                vdr_keys = [str(x['index']) for x in vdr_items]
                vdr_res = track_progress_rich(
                    _grade_item_vdr, vdr_tups, nproc=nproc, chunksize=nproc,
                    save=vdr_cache_file, keys=vdr_keys,
                )
                vdr_cache = load(vdr_cache_file)
                for k, v in zip(vdr_keys, vdr_res):
                    if k not in vdr_cache:
                        vdr_cache[k] = v

            n_total = len(data)
            n_correct = sum(
                1 for idx in data['index']
                if vdr_cache.get(str(idx), {}).get('verdict') == 'correct'
            )
            n_incorrect = sum(
                1 for idx in data['index']
                if vdr_cache.get(str(idx), {}).get('verdict') == 'incorrect'
            )
            vdr_summary = {
                'total': n_total,
                'correct': n_correct,
                'incorrect': n_incorrect,
                'error': n_total - n_correct - n_incorrect,
                'accuracy': round(100 * n_correct / n_total, 2) if n_total else 0.0,
            }
            vdr_score_file = get_intermediate_file_path(
                eval_file, '_bcvl_vdr', 'json')
            dump(vdr_summary, vdr_score_file)
            logger.info(
                f'BrowseComp-VL VDR auxiliary judge results saved to {vdr_score_file}'
            )

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
