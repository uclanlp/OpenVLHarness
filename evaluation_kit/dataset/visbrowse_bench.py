"""VisBrowse-Bench benchmark dataset.

Visual-native web browsing benchmark (169 VQA instances across 7 domains and
24 sub-domains) introduced in arxiv 2603.16289. Each item requires visual
reasoning + web search to answer; dataset is distributed in an encrypted
canary format and must be decrypted on load.

Source: HuggingFace dataset ``Zhengbo-Zhang/VisBrowse-Bench`` (169 rows
of encrypted metadata + image URL pointers to GitHub raw). Images are
fetched on first build into ``$LMUData/images/VisBrowseBench/<basename>``
and cached for reuse.

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

Evaluation: the VDR contains-based judge (``judge_vdr``) by default; the official
VisBrowse-Bench LLM judge (upstream ``eval.py``, simple semantic-yes/no) is selected
with ``judge_prompt: vbb``. ``enable_vdr_aux: true`` adds the VDR judge as a side
output (``_vbb_vdr.json``) next to the ``vbb`` score.
"""

import base64
import hashlib
import json
import os
import os.path as osp
import re
import urllib.error
import urllib.request
from urllib.parse import urlparse

import pandas as pd

from .image_base import ImageBaseDataset
from .hf_release import HF_SPLITS, load_hf_split
from .utils import build_judge, DEBUG_MESSAGE
from .utils.judge_cache import guard_cache, require_judge
from ..smp import *
from ..smp.file import LMUDataRoot, get_intermediate_file_path
from ..utils import track_progress_rich


# ---------------------------------------------------------------------------
# Source-data: HuggingFace dataset (jsonl) + GitHub raw image fetch
# ---------------------------------------------------------------------------
_HF_DATASET_ID = 'Zhengbo-Zhang/VisBrowse-Bench'


def _load_hf_dataset():
    """Load VisBrowse-Bench from HuggingFace, with a cache fallback.

    Mirrors the MMSearch loader's defensive pattern — if the user's HF
    datasets cache is in an incompatible state, retry with a fresh
    cache_dir under ``LMUDataRoot()``.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The `datasets` library is required to use VisBrowse-Bench.\n"
            "Install via `pip install datasets`."
        ) from exc

    env_cache_dir = os.environ.get('HF_DATASETS_CACHE_DIR')
    if env_cache_dir:
        os.makedirs(env_cache_dir, exist_ok=True)
        return load_dataset(_HF_DATASET_ID, split='train', cache_dir=env_cache_dir)

    try:
        return load_dataset(_HF_DATASET_ID, split='train')
    except TypeError as e:
        if 'must be called with a dataclass type or instance' not in str(e):
            raise
        fallback_cache_dir = osp.join(LMUDataRoot(), 'hf_cache', 'datasets')
        os.makedirs(fallback_cache_dir, exist_ok=True)
        return load_dataset(_HF_DATASET_ID, split='train', cache_dir=fallback_cache_dir)


def _download_image(url: str, target: str) -> None:
    """Download a single image URL to target path; idempotent (skips cached)."""
    if osp.isfile(target):
        return
    os.makedirs(osp.dirname(target), exist_ok=True)
    try:
        urllib.request.urlretrieve(url, target)
    except urllib.error.URLError as e:
        if osp.exists(target):
            try:
                os.remove(target)
            except OSError:
                pass
        raise RuntimeError(
            f"Failed to download VisBrowse-Bench image from {url}: {e}"
        ) from e


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
# Original VisBrowse-Bench protocol (yes/no semantic equivalence), selected
# with judge_prompt 'vbb'; the default judge is judge_vdr.
# Prompt + parser live in utils/visbrowse_bench_judge.py.
# ---------------------------------------------------------------------------
from .utils.visbrowse_bench_judge import (
    VISBROWSE_BENCH_JUDGE_PROMPT,
    parse_visbrowse_bench_verdict,
)


# ---------------------------------------------------------------------------
# Auxiliary VDR-style judge prompt (opt-in placeholder)
#
# Adapted from Vision-DeepResearch's ``rllm/rewards/deepresearch_reward.py``
# judging_prompt — a lenient contains-based binary judge. Only invoked when
# the judge config sets ``enable_vdr_aux: true``. Output goes to a separate
# ``_vbb_vdr.json`` file and does NOT override the primary paper-protocol
# scores. Exists for cross-benchmark consistency studies (VDR / LIVEVQA /
# BCVL use the same style).
# ---------------------------------------------------------------------------
# NOTE: This is the OLD paraphrased VDR judge prompt — kept as ``_BK`` for
# reference. The paper-exact VDR prompt (with ``[Deep Research Report]``
# field and ``correct: yes/no\\nreasoning: ...`` plain-text output) lives
# in ``dataset/utils/judge_vdr.py`` and is reached via
# ``MMDR_JUDGE_PROMPT=judge_vdr``. This inline copy still drives the legacy
# ``enable_vdr_aux: true`` aux-judge path; do not remove without first
# migrating that path to ``utils/judge_vdr.py``.
_VDR_JUDGE_PROMPT_BK = """You are an impartial judge evaluating whether an AI assistant's answer contains the correct answer.

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
# Per-item grading helper
# ---------------------------------------------------------------------------

def _grade_item(model, item):
    """Grade a single prediction using the ORIGINAL VisBrowse-Bench judge.

    Follows the upstream ``eval.py`` protocol: the judge outputs a bare
    ``yes`` or ``no`` indicating semantic equivalence between model
    answer and ground truth. Multi-variant ground truths are checked
    individually; if the judge says ``yes`` to any one variant, the
    sample is counted correct.

    Returns a dict with keys: correct (1 or 0), extracted_answer (''),
    reasoning (trimmed judge response).
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

    for gt in gt_list:
        prompt = VISBROWSE_BENCH_JUDGE_PROMPT.format(
            question=question,
            correct_answer=str(gt),
            response=prediction,
        )
        retry = 3
        while retry:
            ans = model.generate(prompt)
            if 'Failed to obtain answer via API' in ans:
                retry -= 1
                continue
            verdict = parse_visbrowse_bench_verdict(ans)
            if verdict == 'yes':
                return dict(correct=1, extracted_answer='', reasoning=ans.strip())
            if verdict == 'no':
                # 'no' for this variant — try next GT variant
                break
            retry -= 1
        else:
            # Exhausted retries without a parseable answer for this variant
            continue

    return dict(correct=0, extracted_answer='', reasoning='No matching GT variant')


def _grade_item_judge_rs(model, item):
    """Judge_RS variant of ``_grade_item`` for VisBrowse-Bench.

    Sends ``JUDGE_RS_SYSTEM_PROMPT`` + ``JUDGE_RS_USER_PROMPT`` as
    separate roles via ``grade_with_judge_rs`` helper. All ground-truth
    variants are passed as ``Candidate Answers`` (Rule 10 — match any →
    correct). Opt-in via env ``MMDR_JUDGE_PROMPT=judge_rs``.
    """
    from .utils.judge_rs import grade_with_judge_rs

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

    try:
        parsed = grade_with_judge_rs(model, question, gt_list, prediction)
    except Exception as e:
        return dict(correct=0, extracted_answer='', reasoning=f'judge_rs error: {e}')
    return dict(
        correct=int(parsed['correct']),
        extracted_answer='',
        reasoning=str(parsed.get('reasoning', '')),
    )


def _grade_item_judge_vdr(model, item):
    """Judge_VDR variant of ``_grade_item`` (VDR paper contains-based judge,
    plain-text output). All ground-truth variants are joined into a single
    ``[Correct Answer]`` field. The default judge.
    """
    from .utils.judge_vdr import grade_with_judge_vdr

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

    try:
        parsed = grade_with_judge_vdr(model, question, gt_list, prediction)
    except Exception as e:
        return dict(correct=0, extracted_answer='', reasoning=f'judge_vdr error: {e}')
    return dict(
        correct=int(parsed['correct']),
        extracted_answer='',
        reasoning=str(parsed.get('reasoning', '')),
    )


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

    formatted_target = ' OR '.join(f'"{str(gt)}"' for gt in gt_list)
    prompt = _VDR_JUDGE_PROMPT_BK.format(
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
        if dataset in HF_SPLITS:
            return load_hf_split(self, dataset)
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
        """Load HF rows, decrypt per-item, fetch images, return a DataFrame.

        Images are downloaded from each row's ``image`` URL list into
        ``$LMUData/images/VisBrowseBench/<basename>`` and cached for reuse.
        image_path is stored as:
          - a single absolute path (str) when the item has exactly one image
          - a JSON-encoded list when the item has multiple images
        """
        img_cache_root = osp.join(LMUDataRoot(), 'images', 'VisBrowseBench')
        hf_ds = _load_hf_dataset()

        rows = []
        for idx, obj in enumerate(hf_ds):
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

            # Download images (cached) and resolve to local paths.
            img_urls = obj.get('image', []) or []
            if isinstance(img_urls, str):
                img_urls = [img_urls]
            resolved = []
            for u in img_urls:
                u = str(u)
                parsed = urlparse(u)
                basename = os.path.basename(parsed.path) if parsed.scheme else os.path.basename(u)
                if not basename:
                    resolved.append(u)
                    continue
                local = osp.join(img_cache_root, basename)
                _download_image(u, local)
                resolved.append(local)

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

    # Needs an LLM judge (or exact_matching); checked before inference.
    REQUIRES_LLM_JUDGE = True

    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate predictions with the selected judge (judge_vdr by default).

        ``judge_prompt: vbb`` selects the upstream ``eval.py`` semantic-yes/no
        judge. Setting ``enable_vdr_aux:
        true`` in the judge config additionally runs a VDR-style
        contains-based grader and saves a separate ``_vbb_vdr.json``
        side-output; the primary accuracy is not overridden.
        """
        require_judge(self.dataset_name, judge_kwargs)
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

        # Build judge model. Pop framework-level keys so they don't leak
        # into the OpenAI request payload (would trigger 400 on unknown
        # fields). ``enable_vdr_aux`` gates the optional VDR aux grader
        # below; other two are historical keys kept for back-compat.
        judge_kwargs = dict(judge_kwargs)
        nproc = judge_kwargs.pop('nproc', 4)
        judge_kwargs.pop('enable_llm_judge', None)
        # ``judge_max_tokens`` is consumed centrally in build_judge() (yaml is
        # the single source of truth; no per-evaluator default).
        enable_vdr_aux = bool(judge_kwargs.pop('enable_vdr_aux', False))

        # Judge selection: env var > judge_kwargs > 'judge_vdr' default
        judge_name = (
            os.environ.get('MMDR_JUDGE_PROMPT')
            or judge_kwargs.pop('judge_prompt', None)
            or 'judge_vdr'
        ).strip().lower()
        if judge_name not in ('vbb', 'judge_rs', 'judge_vdr'):
            raise ValueError(
                f"Unknown judge_prompt {judge_name!r}; expected 'vbb', 'judge_rs', or 'judge_vdr'"
            )
        judge_kwargs.pop('judge_prompt', None)

        judge_model = judge_kwargs.get('model')
        model = build_judge(**judge_kwargs)
        if model is None or not model.working():
            logger.error(
                'VisBrowseBench evaluation requires a working judge model. '
                'Please configure judge_kwargs properly.'
            )
            logger.error(DEBUG_MESSAGE)
            return pd.DataFrame()

        if judge_name == 'judge_rs':
            grade_fn = _grade_item_judge_rs
            grade_stem = '_vbb_grade_judge_rs'
            detail_stem = '_vbb_detail_judge_rs'
            acc_stem = '_vbb_acc_judge_rs'
        elif judge_name == 'judge_vdr':
            grade_fn = _grade_item_judge_vdr
            grade_stem = '_vbb_grade_judge_vdr'
            detail_stem = '_vbb_detail_judge_vdr'
            acc_stem = '_vbb_acc_judge_vdr'
        else:
            grade_fn = _grade_item
            grade_stem = '_vbb_grade'
            detail_stem = '_vbb_detail'
            acc_stem = '_vbb_acc'

        # Run grading with file cache for resumability
        result_file = get_intermediate_file_path(eval_file, grade_stem, 'pkl')
        guard_cache(result_file, data, judge_model, prompt=judge_name)
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
        detail_file = get_intermediate_file_path(eval_file, detail_stem)
        dump(data, detail_file)

        # Compute accuracy breakdown
        acc = self._compute_accuracy(data)
        score_file = get_intermediate_file_path(eval_file, acc_stem, 'csv')
        dump(acc, score_file)
        logger.info(f'VisBrowseBench accuracy results saved to {score_file}')

        # -- Auxiliary VDR judge pass (opt-in via enable_vdr_aux) ----------
        # Does NOT override the primary original-protocol scores above.
        # Produces a separate ``_vbb_vdr.json`` side-output for
        # cross-benchmark studies that want VDR-style contains-based grading.
        if enable_vdr_aux:
            vdr_cache_file = get_intermediate_file_path(
                eval_file, '_vbb_vdr_grade', 'pkl')
            guard_cache(vdr_cache_file, data, judge_model)
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
                eval_file, '_vbb_vdr', 'json')
            dump(vdr_summary, vdr_score_file)
            logger.info(
                f'VisBrowseBench VDR auxiliary judge results saved to {vdr_score_file}'
            )

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
