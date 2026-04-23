"""VisBrowse-Bench benchmark dataset.

Visual-native web browsing benchmark (169 VQA instances across 7 domains and
24 sub-domains) introduced in arxiv 2603.16289. Each item requires visual
reasoning + web search to answer; dataset is distributed in an encrypted
canary format and must be decrypted on load.

Source layout (cloned from https://github.com/ZhengboZhang/VisBrowse-Bench):
  $VISBROWSE_ROOT/
    data/VisBrowse-Bench.jsonl        — encrypted JSONL, 169 records
    VisBrowse_Bench_images/           — local copy of images (1-3 per item)

Record schema (encrypted):
  {
    "id": int,
    "question": <canary-encrypted str>,
    "image": [url, ...],
    "answer": [<canary-encrypted str>, ...],
    "domain": "Media"|"Life"|"Art"|"Geography"|"Technology"|"Sport"|"Finance",
    "sub_domain": str,
    "canary": str,
  }

Decryption: per-item XOR with SHA-256-derived key using ``canary`` as the
password (matches ``decrypt_data.py`` in the upstream repo).

Evaluation: VDR contains-based LLM judge (aligned with our BrowseCompVL /
LiveVQA ports). Official VisBrowse-Bench judge is kept commented below for
reference.
"""

import base64
import hashlib
import json
import os
import os.path as osp
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import pandas as pd

from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
from ..smp.file import LMUDataRoot, get_intermediate_file_path
from ..utils import track_progress_rich


# ---------------------------------------------------------------------------
# Source-data layout
# ---------------------------------------------------------------------------
_VISBROWSE_ROOT_DEFAULT = '/home/<user>/code/VisBrowse-Bench'
_JSONL_REL = 'data/VisBrowse-Bench.jsonl'
_IMG_DIR_REL = 'VisBrowse_Bench_images'


# ---------------------------------------------------------------------------
# Canary decryption (mirrors upstream ``decrypt_data.py``)
# ---------------------------------------------------------------------------

def _derive_key(password: str, length: int) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def _decrypt(ciphertext_b64: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext_b64)
    key = _derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode()


# ---------------------------------------------------------------------------
# Judge prompt — VDR contains-based (same style as browsecomp_vl.py /
# livevqa.py). Kept consistent so cross-bench numbers are comparable.
# ---------------------------------------------------------------------------
JUDGE_PROMPT = """You are an impartial judge evaluating whether an AI assistant's answer contains the correct answer.

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
5. The [Correct Answer] field may contain multiple acceptable variants joined by "OR" — matching any ONE of them counts as correct.
6. Provide your reasoning.
7. Answer "yes" if the response contains the correct answer (semantically equivalent is fine), "no" if it doesn't or contradicts it.

Respond **strictly in a valid JSON object** matching this schema:
{{
  "reasoning": string,
  "correct": "yes" | "no"
}}"""

# ---------------------------------------------------------------------------
# Original VisBrowse-Bench judge (simpler semantic-equivalence yes/no, from
# upstream ``eval.py``). Kept here commented for reproducibility of the
# paper's reported numbers.
# ---------------------------------------------------------------------------
# JUDGE_PROMPT = """
# You are a rigorous question-and-answer quality assessment expert. Given a question, your task is to compare the "groud truth" and the "model answer", determine whether they are semantically equivalent, and output "yes" or "no".
# question: {question}
# ground truth: {correct_answer}
# model answer: {response}
# """


# ---------------------------------------------------------------------------
# Per-item grading helper
# ---------------------------------------------------------------------------

def _grade_item(model, item):
    """Grade a single prediction using the VDR contains-based judge.

    Returns a dict with keys: correct (1 or 0), extracted_answer, reasoning.
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
        return dict(correct=0, extracted_answer='', reasoning='Empty ground truth')

    # Fold multi-variant ground truths into an OR-joined target so the
    # contains-judge treats any variant as acceptable.
    formatted_target = ' OR '.join(f'"{str(gt)}"' for gt in gt_list)
    prompt = JUDGE_PROMPT.format(
        question=question,
        correct_answer=formatted_target,
        response=prediction,
    )

    retry = 3
    while retry:
        ans = model.generate(prompt)
        if 'Failed to obtain answer via API' in ans:
            retry -= 1
            continue

        # Prefer JSON parse (VDR-style)
        try:
            m = re.search(r'\{[\s\S]*\}', ans)
            if m:
                obj = json.loads(m.group(0))
                verdict = str(obj.get('correct', 'no')).strip().lower()
                acc = 1 if verdict in ('y', 'yes', 'true', 'positive') else 0
                return dict(
                    correct=acc,
                    extracted_answer=obj.get('extracted_final_answer', ''),
                    reasoning=obj.get('reasoning', ''),
                )
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: plain yes/no
        raw = ans.strip().lower()
        if 'correct: yes' in raw or raw in ('yes', 'y'):
            return dict(correct=1, extracted_answer='', reasoning=ans.strip()[:500])
        if 'correct: no' in raw or raw in ('no', 'n'):
            return dict(correct=0, extracted_answer='', reasoning=ans.strip()[:500])

        retry -= 1

    return dict(correct=0, extracted_answer='', reasoning='Judge failed')


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class VisBrowseBenchDataset(ImageBaseDataset):
    """VisBrowse-Bench (arxiv 2603.16289).

    169 visual-native search VQA instances across 7 domains x 24 sub-domains.
    Data ships encrypted via canary XOR; this class decrypts on first load
    and caches a plaintext TSV under ``$LMUData``.
    """

    TYPE = 'VQA'

    DATASET_URL = {
        'VisBrowseBench': '',
    }
    DATASET_MD5 = {}

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_URL)

    # -- Data loading: auto-convert from encrypted JSONL on first use ----

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
    def _convert_benchmark():
        """Read the encrypted JSONL, decrypt per-item, and return a flat DataFrame.

        image_path is stored as:
          - a single absolute path (str) when the item has exactly one image
          - a JSON-encoded list when the item has multiple images

        Local images under ``$VISBROWSE_ROOT/VisBrowse_Bench_images/`` are
        preferred; if a file is missing, the original URL is stored instead.
        """
        root = os.environ.get('VISBROWSE_ROOT', _VISBROWSE_ROOT_DEFAULT)
        jsonl_path = osp.join(root, _JSONL_REL)
        img_base = osp.join(root, _IMG_DIR_REL)

        rows = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                canary = obj.get('canary', '')

                # Decrypt question + answers
                q_cipher = obj.get('question', '')
                answers_cipher = obj.get('answer', []) or []
                try:
                    question = _decrypt(q_cipher, canary) if canary else q_cipher
                except Exception:
                    question = q_cipher  # keep as-is if decryption fails
                answers = []
                for a in answers_cipher:
                    try:
                        answers.append(_decrypt(a, canary) if canary else a)
                    except Exception:
                        answers.append(a)
                if not answers:
                    answers = ['']

                # Resolve image paths (prefer local)
                img_urls = obj.get('image', []) or []
                if isinstance(img_urls, str):
                    img_urls = [img_urls]
                resolved = []
                for u in img_urls:
                    u = str(u)
                    parsed = urlparse(u)
                    basename = os.path.basename(parsed.path) if parsed.scheme else os.path.basename(u)
                    local = osp.join(img_base, basename) if basename else ''
                    if local and osp.isfile(local):
                        resolved.append(local)
                    else:
                        resolved.append(u)

                if len(resolved) == 1:
                    image_path_field = resolved[0]
                else:
                    image_path_field = json.dumps(resolved)

                rows.append({
                    'index': idx,
                    'question': question,
                    'answer': answers[0],
                    'ground_truth_list': json.dumps(answers),
                    'image_path': image_path_field,
                    'category': obj.get('domain', 'unknown') or 'unknown',
                    'l2-category': obj.get('sub_domain', 'unknown') or 'unknown',
                })

        return pd.DataFrame(rows)

    # -- Prompt: open-ended VQA (question only, as in the original infer) -

    def build_prompt(self, line):
        """Build a prompt with image(s) + question. Matches the original
        VisBrowse-Bench inference setup (question only, no answer format).
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = line['image_path']

        # Handle multi-image items (TSV may serialize list as JSON string)
        if isinstance(tgt_path, str) and tgt_path.startswith('['):
            try:
                tgt_path = json.loads(tgt_path)
            except Exception:
                pass
        if not isinstance(tgt_path, list):
            tgt_path = [tgt_path]

        msgs = [dict(type='image', value=str(p)) for p in tgt_path if p]
        msgs.append(dict(type='text', value=str(line['question'])))
        return msgs

    # -- Evaluation: binary judge grading (yes/no contains-based) --------

    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate predictions with the VDR contains-based LLM judge."""
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

        # Build judge model. Pop MMSearch-specific keys that would otherwise
        # be forwarded to OpenAIWrapper.generate_inner and into the request
        # body, where OpenAI returns 400 on unrecognized fields.
        judge_kwargs = dict(judge_kwargs)
        nproc = judge_kwargs.pop('nproc', 4)
        judge_kwargs.pop('enable_llm_judge', None)
        max_tokens = judge_kwargs.pop('judge_max_tokens', 1024)
        model = build_judge(max_tokens=max_tokens, **judge_kwargs)
        if model is None or not model.working():
            logger.error(
                'VisBrowseBench evaluation requires a working judge model. '
                'Please configure judge_kwargs properly.'
            )
            logger.error(DEBUG_MESSAGE)
            return pd.DataFrame()

        # Run grading with file cache for resumability
        result_file = get_intermediate_file_path(eval_file, '_vbb_grade', 'pkl')
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
        detail_file = get_intermediate_file_path(eval_file, '_vbb_detail')
        dump(data, detail_file)

        # Compute accuracy breakdown
        acc = self._compute_accuracy(data)
        score_file = get_intermediate_file_path(eval_file, '_vbb_acc', 'csv')
        dump(acc, score_file)
        logger.info(f'VisBrowseBench accuracy results saved to {score_file}')
        return acc

    @staticmethod
    def _compute_accuracy(data):
        """Accuracy breakdown: Overall, per-domain (category), per-sub_domain
        (l2-category).
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

        if 'category' in data:
            for domain in sorted(data['category'].dropna().unique()):
                rows.append(_acc_row(domain, data[data['category'] == domain]))

        if 'l2-category' in data:
            for sub in sorted(data['l2-category'].dropna().unique()):
                rows.append(_acc_row(sub, data[data['l2-category'] == sub]))

        return pd.DataFrame(rows)
