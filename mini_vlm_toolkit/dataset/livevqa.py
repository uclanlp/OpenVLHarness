"""LIVEVQA benchmark dataset (News / Paper / Video).

Each sub-benchmark contains level1 (L1) and level2 (L2) open-ended VQA
questions.  The model receives only the image and question (no MCQ options)
and produces a free-form text answer.

Evaluation follows the original LIVEVQA protocol:
  1. Compare the model's prediction against Ground_Truth_List using an AI
     judge (GRADER_TEMPLATE) that assigns one of three grades:
       A = CORRECT, B = INCORRECT, C = NOT_ATTEMPTED
  2. Report accuracy per level (L1 / L2), per question_type, and overall.
     Two accuracy metrics are produced:
       - strict:  A / (A + B + C)
       - lenient: A / (A + B)   (excluding NOT_ATTEMPTED)

TSV files are produced by ``scripts/convert_livevqa.py`` and placed in
``$LMUData/``.
"""

import string
from collections import defaultdict
from functools import partial

import pandas as pd

from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
from ..smp.file import get_intermediate_file_path
from ..utils import track_progress_rich

# ---------------------------------------------------------------------------
# Grader prompt – faithfully ported from the original LIVEVQA
# ``unified_benchmark.py`` GRADER_TEMPLATE.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# QA prompt – matches the original LIVEVQA ANSWER_WITH_CONFIDENCE_PROMPT_TEMPLATE.
# The model is asked to return a JSON object with "answer" and
# "confidence_score" fields, making the output easy to parse downstream.
# ---------------------------------------------------------------------------
ANSWER_WITH_CONFIDENCE_PROMPT = """\
Here is the question:
{question}
Please provide your best guess and a confidence score between \
0% to 100% in the following JSON format:
{{
"answer": "Your answer here",
"confidence_score": number
}}"""


GRADER_TEMPLATE = """\
Your job is to look at a question, a gold target, and a predicted answer, and then assign a \
grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.

The following are examples of CORRECT predicted answers.
'''
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to \
double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha \
Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born \
on July 4, 1998, and Sasha was born on June 10, 2001.
'''
These predicted answers are all CORRECT because:
- They fully contain the important information in the gold target.
- They do not contain any information that contradicts the gold target.
- Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
- Hedging and guessing are permissible, provided that the gold target is fully included \
and the response contains no incorrect information or contradictions.

The following are examples of INCORRECT predicted answers.
'''
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or \
it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama \
has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify \
further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's \
recommended to confirm the most accurate and updated information since this could change \
over time. This model may not always reflect the most current information.
'''
These predicted answers are all INCORRECT because:
- A factual statement in the answer contradicts the gold target. Incorrect statements \
that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") \
are also considered incorrect.

The following are examples of NOT_ATTEMPTED predicted answers.
'''
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I \
can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm \
not sure about the other one.
'''
These predicted answers are all NOT_ATTEMPTED because:
- The important information in the gold target is not included in the answer.
- No statements in the answer contradict the gold target.

Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be \
correct to the last significant figure in the gold answer. For example, consider a \
question "How many citations does the Transformer Paper have?" with gold target "120k".
- Predicted answers "120k", "124k", and "115k" are all CORRECT.
- Predicted answers "100k" and "113k" are INCORRECT.
- Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED \
because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the \
predicted answer only needs to contain the information that is in the question.
- For example, consider the question "What episode did Derek and Meredith get legally \
married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". \
Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred \
from the question.
- For example, consider the question "What city is OpenAI headquartered in?" and the gold \
target "San Francisco, California". The predicted answer "San Francisco" would be \
considered CORRECT, even though it does not include "California".
- Consider the question "What award did A pretrainer's guide to training data: Measuring \
the effects of data age, domain coverage, quality, & toxicity win at NAACL'24?", the \
gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" \
would be considered CORRECT, because "award" is presumed in the question.
- For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 \
m". The predicted answer "1.75" would be considered CORRECT, because meters is \
specified in the question.
- For the question "What is the name of Barack Obama's wife?", the gold target is \
"Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because \
the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name.
- For example, if the gold target is "Hyung Won Chung", you can consider the following \
predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".

Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT_ATTEMPTED. Don't \
apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
'''
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
'''
Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED
Just return the letters "A", "B", or "C", with no text around it."""


# ---------------------------------------------------------------------------
# Per-item grading helper
# ---------------------------------------------------------------------------

def _extract_answer_text(prediction):
    """Extract the answer text from a model prediction.

    The model is prompted to return JSON ``{"answer": "...", "confidence_score": N}``.
    This helper tries to parse that JSON and extract the ``answer`` field.
    Falls back to using the raw prediction string if parsing fails.
    """
    import json as _json
    import re as _re

    pred = str(prediction).strip()
    # Try direct JSON parse
    try:
        obj = _json.loads(pred)
        if isinstance(obj, dict) and 'answer' in obj:
            return str(obj['answer'])
    except (_json.JSONDecodeError, ValueError):
        pass

    # Try to extract a JSON substring from the response
    m = _re.search(r'\{[\s\S]*\}', pred)
    if m:
        try:
            obj = _json.loads(m.group(0))
            if isinstance(obj, dict) and 'answer' in obj:
                return str(obj['answer'])
        except (_json.JSONDecodeError, ValueError):
            pass

    # Fallback: use the raw prediction
    return pred


def _grade_item(model, item):
    """Grade a single prediction using the LIVEVQA GRADER_TEMPLATE.

    Returns a dict with keys: grade (A/B/C or None), status (CORRECT /
    INCORRECT / NOT_ATTEMPTED / Error).
    """
    prediction = _extract_answer_text(item.get('prediction', ''))
    gt_list = item.get('ground_truth_list', '')
    question = item.get('question', '')

    # Format ground truth list the same way as the original benchmark
    if isinstance(gt_list, str):
        try:
            gt_list = eval(gt_list)  # TSV stores lists as string repr
        except Exception:
            gt_list = [gt_list]
    if not gt_list:
        return dict(grade=None, status='Error')

    formatted_target = ' OR '.join(f'"{gt}"' for gt in gt_list)
    prompt = GRADER_TEMPLATE.format(
        question=question,
        target=formatted_target,
        predicted_answer=prediction,
    )

    retry = 3
    while retry:
        ans = model.generate(prompt)
        if 'Failed to obtain answer via API' in ans:
            retry -= 1
            continue
        raw = ans.strip().upper()
        if raw in ('A', 'B', 'C'):
            status_map = {'A': 'CORRECT', 'B': 'INCORRECT', 'C': 'NOT_ATTEMPTED'}
            return dict(grade=raw, status=status_map[raw])
        # Sometimes the judge returns full words
        for letter, word in [('A', 'CORRECT'), ('B', 'INCORRECT'), ('C', 'NOT_ATTEMPTED')]:
            if word in raw:
                return dict(grade=letter, status=word)
        retry -= 1

    return dict(grade=None, status='Error')


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class LIVEVQADataset(ImageBaseDataset):

    TYPE = 'VQA'

    # Empty URL strings trigger local-TSV loading via ImageBaseDataset.load_data
    DATASET_URL = {
        'LIVEVQA_News': '',
        'LIVEVQA_Paper': '',
        'LIVEVQA_Video': '',
    }
    DATASET_MD5 = {}

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_URL)

    # -- Prompt: open-ended VQA (no MCQ options) ----------------------------

    def build_prompt(self, line):
        """Build a prompt with image + question in the original LIVEVQA format.

        The model receives the image and is asked to return a JSON object with
        ``"answer"`` and ``"confidence_score"`` fields, matching the original
        LIVEVQA ``ANSWER_WITH_CONFIDENCE_PROMPT_TEMPLATE``.
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = toliststr(line['image_path'])

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]

        prompt_text = ANSWER_WITH_CONFIDENCE_PROMPT.format(question=line['question'])
        msgs.append(dict(type='text', value=prompt_text))
        return msgs

    # -- Evaluation: AI judge grading (A/B/C) -------------------------------

    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate predictions using the original LIVEVQA grading protocol.

        Requires a judge model (e.g. gpt-4o) configured via judge_kwargs.
        """
        logger = get_logger('Evaluation')
        data = load(eval_file)
        assert 'prediction' in data, 'prediction column missing from eval file'
        data['prediction'] = [str(x) for x in data['prediction']]

        # Merge ground_truth_list from the dataset metadata
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
                'LIVEVQA evaluation requires a working judge model. '
                'Please configure judge_kwargs properly.'
            )
            logger.error(DEBUG_MESSAGE)
            return pd.DataFrame()

        # Run grading
        result_file = get_intermediate_file_path(eval_file, '_livevqa_grade', 'pkl')
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
        data['grade'] = [
            result_cache.get(str(idx), {}).get('grade', None)
            for idx in data['index']
        ]
        data['status'] = [
            result_cache.get(str(idx), {}).get('status', 'Error')
            for idx in data['index']
        ]

        # Save detailed results
        detail_file = get_intermediate_file_path(eval_file, '_livevqa_detail')
        dump(data, detail_file)

        # Compute accuracy statistics
        acc = self._compute_accuracy(data)
        score_file = get_intermediate_file_path(eval_file, '_livevqa_acc', 'csv')
        dump(acc, score_file)
        logger.info(f'LIVEVQA accuracy results saved to {score_file}')
        return acc

    @staticmethod
    def _compute_accuracy(data):
        """Compute accuracy breakdown matching the original LIVEVQA statistics.

        Returns a DataFrame with rows for Overall, L1, L2, and per-question_type,
        with columns: total, A_correct, B_incorrect, C_not_attempted, errors,
        accuracy_strict (A/(A+B+C)), accuracy_lenient (A/(A+B)).
        """
        rows = []

        def _acc_row(name, subset):
            total = len(subset)
            a = (subset['status'] == 'CORRECT').sum()
            b = (subset['status'] == 'INCORRECT').sum()
            c = (subset['status'] == 'NOT_ATTEMPTED').sum()
            err = (subset['status'] == 'Error').sum()
            ab = a + b
            abc = a + b + c
            return {
                'split': name,
                'total': total,
                'A_correct': int(a),
                'B_incorrect': int(b),
                'C_not_attempted': int(c),
                'errors': int(err),
                'accuracy_strict': round(a / abc * 100, 2) if abc > 0 else 0.0,
                'accuracy_lenient': round(a / ab * 100, 2) if ab > 0 else 0.0,
            }

        rows.append(_acc_row('Overall', data))

        # Per level (category column: L1 / L2)
        if 'category' in data:
            for level in sorted(data['category'].unique()):
                rows.append(_acc_row(level, data[data['category'] == level]))

        # Per question type (l2-category column)
        if 'l2-category' in data:
            for qtype in sorted(data['l2-category'].unique()):
                rows.append(_acc_row(qtype, data[data['l2-category'] == qtype]))

        return pd.DataFrame(rows)
