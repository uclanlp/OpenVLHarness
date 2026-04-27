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

Data is auto-fetched from the LIVEVQA upstream GitHub repo on first use
(sparse-checkout into ``$LMUData/LIVEVQA/``, ~340MB). Benchmark JSONs and
images are converted to a flat TSV per sub-dataset on demand.  The
``LIVEVQA_ROOT`` environment variable overrides the download location
when you already have a local clone.
"""

import json as _json
import os
import os.path as osp
import string
import subprocess
import urllib.error
import urllib.request
from collections import defaultdict
from functools import partial

import pandas as pd

from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
from ..smp.file import LMUDataRoot, get_intermediate_file_path
from ..utils import track_progress_rich

# ---------------------------------------------------------------------------
# Source-data layout and auto-download
# ---------------------------------------------------------------------------
_LIVEVQA_GITHUB_URL = 'https://github.com/fumingyang2004/LIVEVQA.git'


def _ensure_livevqa_data():
    """Ensure LIVEVQA source data is available locally; auto-clone on first use.

    Returns the path to ``liveVQA_benchmarks/`` containing News/Paper/Video
    sub-directories. Honours the ``LIVEVQA_ROOT`` env var as an override for
    users who already have a local clone (e.g. offline environments).
    """
    override = os.environ.get('LIVEVQA_ROOT')
    if override:
        if not osp.isdir(override):
            raise RuntimeError(
                f"LIVEVQA_ROOT={override!r} is set but is not a directory."
            )
        return override

    root = osp.join(LMUDataRoot(), 'LIVEVQA')
    data_dir = osp.join(root, 'liveVQA_benchmarks')
    if osp.exists(data_dir):
        return data_dir

    os.makedirs(root, exist_ok=True)
    print(
        f"[LIVEVQA] first-run download (~340MB) from {_LIVEVQA_GITHUB_URL} "
        f"into {root}"
    )
    subprocess.run(
        [
            'git', 'clone', '--depth', '1',
            '--filter=blob:none', '--sparse',
            _LIVEVQA_GITHUB_URL, root,
        ],
        check=True,
    )
    subprocess.run(
        ['git', '-C', root, 'sparse-checkout', 'set', 'liveVQA_benchmarks'],
        check=True,
    )
    if not osp.exists(data_dir):
        raise RuntimeError(
            f"LIVEVQA auto-download completed but {data_dir} is missing. "
            f"Set LIVEVQA_ROOT manually."
        )
    return data_dir


# ---------------------------------------------------------------------------
# LIVEVQA_v2 (WebWatcher-aligned) — pinned to pre-trim snapshot
# ---------------------------------------------------------------------------
# WebWatcher's ``infer/vl_search_r1/eval_data/livevqa.jsonl`` subset (300
# samples) was cloned BEFORE LIVEVQA upstream's 2025-10-11 "update the
# benchmark" commit, which dropped ~308 samples from the dataset (1500 → 1192
# items). 73 of WebWatcher's 300 samples live only in the pre-trim snapshot.
# Pinning LIVEVQA_v2 to commit 4274316 (2025-08-12, last pre-trim state with
# 1500 items × 2 QA levels = 3000 QAs) recovers all 300 WebWatcher samples
# without needing WebWatcher's OSS bucket.
_LIVEVQA_V2_PIN_COMMIT = '4274316a5cfa23be1eefd8f85e91635e93a3980d'  # 2025-08-12 "update video code" — last pre-trim state

# WebWatcher's eval_data jsonl files (Apache 2.0, hosted in
# Alibaba-NLP/DeepResearch). Used to specify which 300 LIVEVQA samples
# constitute the LIVEVQA_v2 (WebWatcher-aligned) subset. Auto-downloaded
# into $LMUData/webwatcher_eval_data/ on first build.
_WEBWATCHER_EVAL_DATA_BASE = (
    'https://raw.githubusercontent.com/Alibaba-NLP/DeepResearch/main/'
    'WebAgent/WebWatcher/infer/vl_search_r1/eval_data'
)


def _download_webwatcher_eval_jsonl(name: str) -> str:
    """Download a WebWatcher eval_data jsonl into the LMUData cache.

    ``name`` should be one of ``hle | mmsearch | simplevqa | livevqa``
    (without the ``.jsonl`` suffix). Returns the local cache path. If the
    file is already cached, returns the existing path without
    re-downloading.

    Used by LIVEVQA_v2 first-build to (q, a)-match against the upstream
    spec. Cache survives across runs.
    """
    cache_dir = osp.join(LMUDataRoot(), 'webwatcher_eval_data')
    target = osp.join(cache_dir, f'{name}.jsonl')
    if osp.isfile(target):
        return target
    os.makedirs(cache_dir, exist_ok=True)
    url = f'{_WEBWATCHER_EVAL_DATA_BASE}/{name}.jsonl'
    print(f"[LIVEVQA_v2] downloading WebWatcher eval_data/{name}.jsonl from {url}")
    try:
        urllib.request.urlretrieve(url, target)
    except urllib.error.URLError as e:
        # Clean up partial file before re-raising.
        if osp.exists(target):
            try:
                os.remove(target)
            except OSError:
                pass
        raise RuntimeError(
            f"Failed to download WebWatcher {name}.jsonl from {url}: {e}. "
            f"Verify network access, or set LIVEVQA_V2_WW_JSONL to point at "
            f"a local copy of the file."
        ) from e
    return target


def _ensure_livevqa_v2_data():
    """Ensure the pinned pre-trim LIVEVQA snapshot is available locally.

    Stores under ``$LMUData/images/LIVEVQA_v2/`` following the
    repo-wide convention (other benchmarks' images live under
    ``$LMUData/images/<benchmark>/``). The full git repo is cloned
    there and checked out to the pinned commit; the benchmark.json
    files + image dirs live together in ``liveVQA_benchmarks/``.

    Honours the ``LIVEVQA_V2_ROOT`` env var for offline overrides
    (point it at an existing ``liveVQA_benchmarks/`` directory).

    Returns the path to ``liveVQA_benchmarks/`` containing News/Paper/
    Video sub-directories with benchmark.json + image files.
    """
    override = os.environ.get('LIVEVQA_V2_ROOT')
    if override:
        if not osp.isdir(override):
            raise RuntimeError(
                f"LIVEVQA_V2_ROOT={override!r} is set but is not a directory."
            )
        return override

    clone_root = osp.join(LMUDataRoot(), 'images', 'LIVEVQA_v2')
    data_dir = osp.join(clone_root, 'liveVQA_benchmarks')
    if osp.exists(data_dir):
        return data_dir

    os.makedirs(osp.dirname(clone_root), exist_ok=True)
    print(
        f"[LIVEVQA_v2] first-run download (~320MB) from {_LIVEVQA_GITHUB_URL} "
        f"@ commit {_LIVEVQA_V2_PIN_COMMIT[:10]} into {clone_root}"
    )
    subprocess.run(
        ['git', 'clone', _LIVEVQA_GITHUB_URL, clone_root],
        check=True,
    )
    subprocess.run(
        ['git', '-C', clone_root, 'checkout', _LIVEVQA_V2_PIN_COMMIT],
        check=True,
    )
    if not osp.exists(data_dir):
        raise RuntimeError(
            f"LIVEVQA_v2 auto-download completed but {data_dir} is missing. "
            f"Set LIVEVQA_V2_ROOT manually."
        )
    return data_dir


# ---------------------------------------------------------------------------
# Sub-benchmark variants
# ---------------------------------------------------------------------------
_BENCHMARK_CFG = {
    #  dataset_name            → (sub-benchmark, levels_to_include, mcq_mode)
    # --- open-ended (default LIVEVQA protocol, used by unified_benchmark.py) ---
    'LIVEVQA_News':         ('News',  ('level1_qa', 'level2_qa'), False),
    'LIVEVQA_Paper':        ('Paper', ('level1_qa', 'level2_qa'), False),
    'LIVEVQA_Video':        ('Video', ('level1_qa', 'level2_qa'), False),
    'LIVEVQA_News_L1':      ('News',  ('level1_qa',),             False),
    'LIVEVQA_Paper_L1':     ('Paper', ('level1_qa',),             False),
    'LIVEVQA_Video_L1':     ('Video', ('level1_qa',),             False),
    # --- MCQ (options appended to question; aligns with VDR reporting) ---
    'LIVEVQA_News_MCQ':     ('News',  ('level1_qa', 'level2_qa'), True),
    'LIVEVQA_Paper_MCQ':    ('Paper', ('level1_qa', 'level2_qa'), True),
    'LIVEVQA_Video_MCQ':    ('Video', ('level1_qa', 'level2_qa'), True),
    'LIVEVQA_News_L1_MCQ':  ('News',  ('level1_qa',),             True),
    'LIVEVQA_Paper_L1_MCQ': ('Paper', ('level1_qa',),             True),
    'LIVEVQA_Video_L1_MCQ': ('Video', ('level1_qa',),             True),
}

# Combined pool: News + Paper + Video × (L1+L2). 2384 samples total.
# Use ``LIVEVQA_All`` (open-ended) / ``LIVEVQA_All_MCQ`` (MCQ) with
# ``--num-examples N --shuffle --seed 42`` for a reproducible random subset.
_COMBINED_CFG = {
    'LIVEVQA_All':     (('News', 'Paper', 'Video'), ('level1_qa', 'level2_qa'), False),
    'LIVEVQA_All_MCQ': (('News', 'Paper', 'Video'), ('level1_qa', 'level2_qa'), True),
    'LIVEVQA_All_L1':  (('News', 'Paper', 'Video'), ('level1_qa',),             False),
}

_IMG_DIR = {
    'News':  'News/News_imgs',
    'Paper': 'Paper/Paper_images',
    'Video': 'Video/Video_images',
}

_JSON_PATH = {
    'News':  'News/benchmark.json',
    'Paper': 'Paper/benchmark.json',
    'Video': 'Video/benchmark.json',
}

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
}}
For multiple-choice questions, write the full text of the selected option as your answer (not just the letter)."""


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
# Auxiliary VDR-style grader (opt-in placeholder)
#
# Adapted from Vision-DeepResearch's ``rllm/rewards/deepresearch_reward.py``
# judging_prompt — a contains-based binary judge (yes/no). This is *not*
# the LIVEVQA paper protocol; it exists to enable cross-benchmark
# consistency studies (VDR / BCVL / RealX use the same lenient style) and
# is only invoked when the judge config sets ``enable_vdr_aux: true``.
# Output is written to a separate ``_livevqa_vdr.json`` file and does NOT
# override the primary original-protocol scores.
#
# This is a placeholder — exact prompt wording / routing may evolve.
# ---------------------------------------------------------------------------
_VDR_GRADER_TEMPLATE = """You are an impartial judge evaluating whether an AI assistant's answer contains the correct answer.

[Question]
{question}

[Correct Answer]
{target}

[Assistant's Answer]
{predicted_answer}

Task: Determine if the assistant's answer contains the correct answer anywhere in its content.

Instructions:
1. Read through the entire answer carefully.
2. Look for the correct answer anywhere in the response (it may be embedded in paragraphs, reasoning, or explanations).
3. Check if the information in the response is consistent with the correct answer.
4. The answer does NOT need to be in a specific format or labeled as "final answer".
5. The [Correct Answer] field may contain multiple acceptable variants joined by "OR" — matching any ONE of them counts as correct.
6. Provide your reasoning.
7. Answer "yes" if the response contains the correct answer (semantically equivalent is fine), "no" if it doesn't or contradicts it.

Respond **strictly in a valid JSON object** matching this schema:
{{
  "reasoning": string,
  "correct": "yes" | "no"
}}"""


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


def _grade_item_vdr(model, item):
    """Auxiliary VDR-style contains-based binary judge.

    Returns dict with ``verdict`` ('correct' / 'incorrect' / 'error') and
    ``reasoning``. Invoked only when ``enable_vdr_aux`` is set.
    """
    import re as _re

    prediction = str(item.get('prediction', ''))  # pass raw, like VDR
    gt_list = item.get('ground_truth_list', '')
    question = item.get('question', '')

    if isinstance(gt_list, str):
        try:
            gt_list = _json.loads(gt_list)
        except Exception:
            try:
                gt_list = eval(gt_list)
            except Exception:
                gt_list = [gt_list]
    if not gt_list:
        return dict(verdict='error', reasoning='empty ground_truth_list')

    formatted_target = ' OR '.join(f'"{gt}"' for gt in gt_list)
    prompt = _VDR_GRADER_TEMPLATE.format(
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

        m = _re.search(r'\{[\s\S]*\}', ans)
        if m:
            try:
                obj = _json.loads(m.group(0))
                verdict = str(obj.get('correct', '')).strip().lower()
                if verdict in ('yes', 'y', 'true', 'positive'):
                    return dict(verdict='correct',
                                reasoning=str(obj.get('reasoning', '')))
                if verdict in ('no', 'n', 'false', 'negative'):
                    return dict(verdict='incorrect',
                                reasoning=str(obj.get('reasoning', '')))
            except (_json.JSONDecodeError, ValueError):
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

class LIVEVQADataset(ImageBaseDataset):

    TYPE = 'VQA'

    # Empty URL strings: no external TSV download. load_data() below produces
    # the TSV on first use by converting from the upstream benchmark JSONs.
    DATASET_URL = {k: '' for k in list(_BENCHMARK_CFG) + list(_COMBINED_CFG)}
    DATASET_MD5 = {}

    @classmethod
    def supported_datasets(cls):
        return list(_BENCHMARK_CFG) + list(_COMBINED_CFG)

    # -- Data loading: auto-convert from benchmark JSON --------------------

    def load_data(self, dataset):
        """Load LIVEVQA data, auto-converting from benchmark JSON on first use.

        The converted TSV is cached in ``$LMUData/`` so subsequent runs skip
        conversion. ``LIVEVQA_All*`` keys produce a flat pool by concatenating
        News + Paper + Video with a fresh contiguous index.
        """
        data_root = LMUDataRoot()
        tsv_path = osp.join(data_root, f'{dataset}.tsv')

        if not osp.exists(tsv_path):
            if dataset in _COMBINED_CFG:
                subs, levels, mcq = _COMBINED_CFG[dataset]
                dfs = [self._convert_benchmark(s, levels, mcq=mcq) for s in subs]
                df = pd.concat(dfs, ignore_index=True)
                df['index'] = list(range(len(df)))
            else:
                cfg = _BENCHMARK_CFG.get(dataset)
                if cfg is None:
                    raise ValueError(f'Unknown LIVEVQA dataset: {dataset}')
                benchmark, levels, mcq = cfg
                df = self._convert_benchmark(benchmark, levels, mcq=mcq)
            os.makedirs(data_root, exist_ok=True)
            df.to_csv(tsv_path, sep='\t', index=False)

        self.data_path = tsv_path
        return load(tsv_path)

    @staticmethod
    def _convert_benchmark(benchmark, levels, mcq=False):
        """Read a LIVEVQA benchmark JSON and return a flat DataFrame.

        When ``mcq=True``, the multiple-choice ``options`` are appended to
        the ``question`` field in the format::

            <raw question>

            Options:
            A. ...
            B. ...

        This aligns the presentation with Vision-DeepResearch (VDR), which
        presents LIVEVQA items as MCQ. When ``mcq=False`` the question is
        left unchanged, matching the original ``unified_benchmark.py`` /
        open-ended protocol.
        """
        root = _ensure_livevqa_data()
        json_path = osp.join(root, _JSON_PATH[benchmark])
        img_base = osp.join(root, _IMG_DIR[benchmark])

        with open(json_path, 'r', encoding='utf-8') as f:
            data = _json.load(f)

        level_map = {'level1_qa': 'L1', 'level2_qa': 'L2'}
        rows, idx = [], 0
        for item in data:
            for level_key in levels:
                qa = item.get(level_key)
                if qa is None:
                    continue
                img_rel = qa.get('img_path') or item.get('img_path', '')
                img_abs = osp.join(img_base, img_rel)
                qtype = qa.get('question_type', 'unspecified') or 'unspecified'
                gt_list = qa.get('Ground_Truth_List', [])

                question_text = qa['question']
                if mcq:
                    options = qa.get('options') or []
                    if options:
                        question_text = (
                            question_text.rstrip()
                            + "\n\nOptions:\n"
                            + "\n".join(str(o) for o in options)
                        )

                rows.append({
                    'index': idx,
                    'question': question_text,
                    'answer': qa.get('Ground_Truth', ''),
                    'ground_truth_list': _json.dumps(gt_list),
                    'image_path': img_abs,
                    'category': level_map[level_key],
                    'l2-category': qtype,
                })
                idx += 1

        return pd.DataFrame(rows)

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
        # ``enable_vdr_aux`` is a framework-level flag (not an OpenAI
        # parameter). Pop it before passing kwargs to build_judge() so it
        # doesn't leak into the OpenAI request payload and trigger 400.
        enable_vdr_aux = bool(judge_kwargs.pop('enable_vdr_aux', False))
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

        # -- Auxiliary VDR judge pass (opt-in via enable_vdr_aux) ----------
        # Does NOT override the primary original-protocol scores above.
        # Produces a separate ``_livevqa_vdr.json`` side-output for
        # cross-benchmark studies that want VDR-style contains-based grading.
        if enable_vdr_aux:
            vdr_cache_file = get_intermediate_file_path(
                eval_file, '_livevqa_vdr_grade', 'pkl')
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
                eval_file, '_livevqa_vdr', 'json')
            dump(vdr_summary, vdr_score_file)
            logger.info(
                f'LIVEVQA VDR auxiliary judge results saved to {vdr_score_file}'
            )

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


# ===========================================================================
# LIVEVQA_v2 — WebWatcher-aligned subset (300 samples, MCQ, paper-comparable)
# ===========================================================================

class LIVEVQA_v2(ImageBaseDataset):
    """WebWatcher-aligned LIVEVQA evaluation subset (300 samples, MCQ format).

    Data acquisition:
      1. Clone upstream LIVEVQA at commit 4274316 (2025-08-12 pre-trim state,
         1500 items × 2 QA levels = 3000 total QAs). Later upstream commits
         removed ~308 items that WebWatcher still uses, so we pin this commit.
      2. Match WebWatcher's ``infer/vl_search_r1/eval_data/livevqa.jsonl``
         (300 MCQ prompts with options + multi-variant answers) against the
         pinned snapshot by normalized question-stem — recovers 300/300.
      3. Build a flat TSV where ``question`` = WebWatcher's original prompt
         BYTE-EXACT (keeps the MCQ format "...\\nYou can choose among these
         Choices:\\nA. ..."), ``image_path`` = local path inside the pinned
         snapshot, ``ground_truth_list`` = WebWatcher's ``answers`` list.

    Scoring:
      Uses the shared HLE-style judge (``utils/hle_judge``) — same protocol
      as our other WebWatcher-subset benchmarks (hle / MMSearch /
      SimpleVQA_EN / MMSearchPlus / VDRBench / BrowseCompVL).
    """

    TYPE = 'VQA'

    DATASET_URL = {'LIVEVQA_v2': ''}
    DATASET_MD5 = {}

    @classmethod
    def supported_datasets(cls):
        return ['LIVEVQA_v2']

    def load_data(self, dataset):
        """Build-or-load the 300-sample LIVEVQA_v2 TSV.

        Cached at ``$LMUData/LIVEVQA_v2.tsv`` after first build.
        """
        data_root = LMUDataRoot()
        tsv_path = osp.join(data_root, f'{dataset}.tsv')
        if osp.exists(tsv_path):
            self.data_path = tsv_path
            return load(tsv_path)

        # First-build path: ensure upstream data + WebWatcher spec are both
        # available, then match and flatten. WebWatcher's livevqa.jsonl is
        # auto-downloaded from the public Alibaba-NLP/DeepResearch repo
        # (Apache 2.0). Set LIVEVQA_V2_WW_JSONL to override with a local file.
        snapshot_root = _ensure_livevqa_v2_data()
        ww_jsonl_path = os.environ.get('LIVEVQA_V2_WW_JSONL')
        if not ww_jsonl_path:
            ww_jsonl_path = _download_webwatcher_eval_jsonl('livevqa')
        if not osp.isfile(ww_jsonl_path):
            raise RuntimeError(
                f"LIVEVQA_v2: WebWatcher subset spec not found at "
                f"{ww_jsonl_path}. Set LIVEVQA_V2_WW_JSONL to point at the "
                f"WebWatcher livevqa.jsonl, or clone the WebWatcher repo."
            )

        print(f'[LIVEVQA_v2] building TSV from snapshot={snapshot_root} '
              f'+ WebWatcher subset={ww_jsonl_path}')

        # Normalized question-stem → list of (sub, item_idx, level_key, qa_dict)
        import re as _re
        def _strip_options(p):
            return _re.split(r'\nYou can choose|\nOptions:', p, maxsplit=1)[0].strip()
        def _normalize(s):
            return _re.sub(r'\s+', ' ', str(s).strip().lower())

        stem_map = {}
        for sub in ('News', 'Paper', 'Video'):
            json_path = osp.join(snapshot_root, _JSON_PATH[sub])
            img_base = osp.join(snapshot_root, _IMG_DIR[sub])
            with open(json_path, 'r', encoding='utf-8') as f:
                items = _json.load(f)
            for i, item in enumerate(items):
                for level_key in ('level1_qa', 'level2_qa'):
                    qa = item.get(level_key)
                    if qa is None:
                        continue
                    stem = _normalize(qa.get('question', ''))
                    img_rel = qa.get('img_path') or item.get('img_path', '')
                    img_abs = osp.join(img_base, img_rel) if img_rel else ''
                    stem_map.setdefault(stem, []).append(
                        (sub, i, level_key, qa, img_abs)
                    )

        level_map = {'level1_qa': 'L1', 'level2_qa': 'L2'}
        rows = []
        misses = []
        used = set()  # (sub, item_idx, level_key) — greedy claim-once
        with open(ww_jsonl_path) as f:
            for line in f:
                r = _json.loads(line)
                ww_prompt = r['prompt']  # byte-exact preserved as question
                ww_answers = r.get('answers', [])
                if not isinstance(ww_answers, list):
                    ww_answers = [ww_answers]
                stem = _normalize(_strip_options(ww_prompt))
                candidates = stem_map.get(stem, [])
                chosen = None
                for cand in candidates:
                    sub, i, lk, qa, img = cand
                    if (sub, i, lk) in used:
                        continue
                    chosen = cand
                    break
                if chosen is None:
                    misses.append(r.get('question_id'))
                    continue
                sub, i, lk, qa, img_abs = chosen
                used.add((sub, i, lk))
                rows.append({
                    'index': len(rows),
                    'question': ww_prompt,   # BYTE-EXACT MCQ from WebWatcher
                    'answer': str(ww_answers[0]) if ww_answers else '',
                    'ground_truth_list': _json.dumps([str(a) for a in ww_answers]),
                    'image_path': img_abs,
                    'category': level_map[lk],
                    'l2-category': qa.get('question_type', 'unspecified') or 'unspecified',
                    'webwatcher_qid': r.get('question_id', ''),
                    'subset': sub,
                })

        if misses:
            print(f'[LIVEVQA_v2] WARNING: {len(misses)} WebWatcher samples could '
                  f'not be matched to local snapshot (question_ids: {misses[:10]}...)')
        else:
            print(f'[LIVEVQA_v2] matched 300/300 WebWatcher samples.')

        df = pd.DataFrame(rows)
        os.makedirs(data_root, exist_ok=True)
        df.to_csv(tsv_path, sep='\t', index=False)
        self.data_path = tsv_path
        return load(tsv_path)

    # -- Prompt: byte-exact WebWatcher MCQ ----------------------------------

    def build_prompt(self, line):
        """Feed the model the WebWatcher prompt verbatim (no extra template)."""
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

    # -- Evaluation: HLE-style judge + simple binary accuracy ---------------

    def evaluate(self, eval_file, **judge_kwargs):
        """Score predictions using the shared HLE-style judge.

        Per-sample verdicts cached in ``*_livevqa_v2_grade.pkl`` for
        resumable re-eval. Aggregated accuracy saved to ``*_acc.csv``.
        """
        logger = get_logger('Evaluation')
        from .utils.hle_judge import HLE_JUDGE_PROMPT, parse_hle_judgement

        data = load(eval_file)
        assert 'prediction' in data, 'prediction column missing from eval file'
        data['prediction'] = [str(x) for x in data['prediction']]

        # Merge ground_truth_list from metadata
        meta = self.data
        gt_map = {
            str(idx): gt for idx, gt in zip(meta['index'], meta['ground_truth_list'])
        }
        data['ground_truth_list'] = [
            gt_map.get(str(idx), '') for idx in data['index']
        ]

        judge_kwargs = dict(judge_kwargs)
        nproc = judge_kwargs.pop('nproc', 4)
        judge_kwargs.pop('enable_llm_judge', None)
        judge_kwargs.pop('enable_vdr_aux', None)
        max_tokens = judge_kwargs.pop('judge_max_tokens', 1024)
        model = build_judge(max_tokens=max_tokens, **judge_kwargs)
        if model is None or not model.working():
            logger.error(
                'LIVEVQA_v2 evaluation requires a working judge model. '
                'Please configure judge_kwargs properly.'
            )
            logger.error(DEBUG_MESSAGE)
            return pd.DataFrame()

        # Per-sample grading helper (HLE-style binary).
        def _grade(item):
            pred = str(item.get('prediction', ''))
            gt_list = item.get('ground_truth_list', '')
            question = str(item.get('question', ''))
            if isinstance(gt_list, str):
                try:
                    gt_list = _json.loads(gt_list)
                except Exception:
                    gt_list = [gt_list]
            if not gt_list:
                return dict(hit=0, log='empty ground truth')
            # Use first GT variant as correct_answer (HLE convention).
            ca = str(gt_list[0])
            prompt = HLE_JUDGE_PROMPT.format(
                question=question, correct_answer=ca, response=pred,
            )
            retry = 3
            last = None
            while retry > 0:
                ans = model.generate(prompt)
                last = ans
                if 'Failed to obtain answer via API' in (ans or ''):
                    retry -= 1
                    continue
                parsed = parse_hle_judgement(ans)
                if parsed.get('reasoning') != 'judge failed to parse':
                    log = (f"verdict={parsed['correct']} "
                           f"extracted={parsed.get('extracted_answer', '')[:120]} "
                           f"reasoning={parsed.get('reasoning', '')[:200]}")
                    return dict(hit=int(parsed['correct']), log=log)
                retry -= 1
            return dict(hit=0, log=f'judge unparseable: {str(last)[:200]}' if last else 'judge unreachable')

        result_file = get_intermediate_file_path(eval_file, '_livevqa_v2_grade', 'pkl')
        result_cache = load(result_file) if osp.exists(result_file) else {}
        items_to_grade = []
        for i in range(len(data)):
            item = data.iloc[i]
            idx = str(item['index'])
            if idx not in result_cache:
                items_to_grade.append(item)
        if items_to_grade:
            tups = [dict(item=x) for x in items_to_grade]
            keys = [str(x['index']) for x in items_to_grade]
            res = track_progress_rich(
                _grade, items_to_grade, nproc=nproc, chunksize=nproc,
                save=result_file, keys=keys,
            )
            result_cache = load(result_file)
            for k, v in zip(keys, res):
                if k not in result_cache:
                    result_cache[k] = v

        data['hit'] = [result_cache.get(str(i), {}).get('hit', 0) for i in data['index']]
        data['log'] = [result_cache.get(str(i), {}).get('log', '') for i in data['index']]

        # Aggregate
        rows = []
        def _row(name, subset):
            total = len(subset)
            correct = int(subset['hit'].sum())
            return {
                'split': name,
                'total': total,
                'correct': correct,
                'accuracy': round(100 * correct / total, 2) if total else 0.0,
            }
        rows.append(_row('Overall', data))
        if 'category' in data:
            for level in sorted(data['category'].unique()):
                rows.append(_row(level, data[data['category'] == level]))
        if 'subset' in data:
            for sub in sorted(data['subset'].unique()):
                rows.append(_row(sub, data[data['subset'] == sub]))

        acc = pd.DataFrame(rows)
        score_file = get_intermediate_file_path(eval_file, '_livevqa_v2_acc', 'csv')
        dump(acc, score_file)
        logger.info(f'LIVEVQA_v2 accuracy saved to {score_file}')
        return acc
