"""BrowseComp-V3 benchmark dataset.

Visual, vertical, verifiable web-browsing benchmark — 300 multi-step,
visually-grounded, web-searchable questions across diverse domains
introduced in BrowseComp-V3 (project page: halcyon-zhang.github.io/
BrowseComp-V3). Each task pairs one or more images with a question
that requires multi-hop search depth from image evidence to entities,
pages, dates, objects, and counts.

Source: HuggingFace dataset ``Halcyon-Zhang/BrowseComp-V3`` (encrypted
JSONL at ``data/train.jsonl`` plus image assets under ``data/images/``).
Questions and answers are encrypted with AES-256-GCM; we decrypt on
first load using the public passphrase, then cache a plaintext TSV
under ``$LMUData``.

Record schema (after decryption):
  - ``id``: str (e.g., ``001_Culture_Art_L3_p2_t2_m0_r2_w0_c0_g5``)
  - ``question``: str
  - ``answer``: str
  - ``image_paths``: list[str] (relative paths inside the HF snapshot)
  - ``sub_goals``: list of ``{sg_id, description, key_info, ability,
    dependence}`` annotations used by ``judge_bcv3``
  - ``category``: str (top-level domain, e.g. ``Culture``)
  - ``sub_category``: str (sub-domain, e.g. ``Art``)
  - ``metadata``: dict with ``level``, ``difficulty``, ``domain``,
    ``trajectory``, ``vis_inputs``, etc.

Evaluation: four judges are supported via ``MMDR_JUDGE_PROMPT`` env or
``judge_kwargs['judge_prompt']``:

  - ``judge_vdr`` (DEFAULT): contains-based VDR paper judge. Single
    binary correct/incorrect, no process_score.
  - ``judge_rs``: MMSearch-R1 paper rule-based judge.
  - ``hle``: HLE-style binary judge with confidence.
  - ``judge_bcv3``: VERBATIM port of the BrowseComp-V3 paper grader —
    two-dimensional scoring (``answer_correct`` + ``process_score``)
    using the ground-truth sub-goal annotations. This is the only judge
    that consumes ``sub_goals`` and produces a non-zero
    ``process_score``; the other three ignore sub-goals and only score
    the final answer.
"""

import base64
import hashlib
import json
import os
import os.path as osp
import re
import shutil
from typing import Any, Dict, List, Optional

import pandas as pd

from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
from ..smp.file import LMUDataRoot, get_intermediate_file_path
from ..utils import track_progress_rich


# ---------------------------------------------------------------------------
# Source data — HuggingFace + AES-GCM decryption
# ---------------------------------------------------------------------------
_HF_REPO_ID = 'Halcyon-Zhang/BrowseComp-V3'
_HF_JSONL_PATH = 'data/train.jsonl'

# Public passphrase for the official release. Documented in the upstream
# README under "Quick Start" (key.txt). NOT a secret — included verbatim
# so first-time users don't need any external setup.
_BCV3_PASSPHRASE = 'A_Visual_Vertical_Verifiable_Benchmark_for_Multimodal_Browsing_Agents'


def _derive_key(passphrase: str) -> bytes:
    """SHA-256 of the passphrase → 32-byte AES key (matches upstream)."""
    return hashlib.sha256(passphrase.encode('utf-8')).digest()


def _decrypt_field(enc: Any, key: bytes) -> str:
    """Decrypt one ``{iv, ciphertext, tag}`` blob (base64) → utf-8 string.

    Accepts either a dict or a JSON-encoded string (the upstream
    ``encrypted_question`` / ``encrypted_answer`` fields are
    JSON-encoded strings inside the outer JSONL line).
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if isinstance(enc, str):
        try:
            enc = json.loads(enc)
        except Exception:
            return ''
    if not isinstance(enc, dict):
        return ''
    iv = base64.b64decode(enc.get('iv', ''))
    ciphertext = base64.b64decode(enc.get('ciphertext', ''))
    tag = base64.b64decode(enc.get('tag', ''))
    aes = AESGCM(key)
    plaintext = aes.decrypt(iv, ciphertext + tag, None)
    return plaintext.decode('utf-8')


def _snapshot_download_bcv3() -> str:
    """Download the BCV3 HF snapshot (jsonl + images) and return its
    local root. Cached by HF in the user's HF cache; first call is
    slow (a few hundred MB for images), subsequent calls are
    near-instant.
    """
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=_HF_REPO_ID, repo_type='dataset')


# ---------------------------------------------------------------------------
# Default VDR-style binary judge prompt (BC-VL / VBB style, used as the
# auxiliary VDR judge when ``enable_vdr_aux: true`` is set in the judge
# config — kept here as a self-contained backup for cross-protocol checks).
# ---------------------------------------------------------------------------
_VDR_JUDGE_PROMPT_BK = """You are an impartial judge evaluating whether an AI assistant's answer contains the correct answer.

[Question] {question}
[Correct Answer] {correct_answer}
[Assistant Answer] {response}

Task: Determine if the assistant's answer contains the correct answer.

Output exactly one JSON object on a single line:
{{"correct": "yes" | "no", "reasoning": "<your one-sentence reasoning>"}}
"""


# ---------------------------------------------------------------------------
# Per-judge grade item helpers — same signature as the VBB pattern:
# (model, item) → dict with at least ``correct``, ``reasoning``.
# For ``judge_bcv3`` we additionally include ``process_score`` + raw
# ``verdict`` so the downstream aggregator can surface them in acc.csv.
# ---------------------------------------------------------------------------

def _coerce_gt_list(item) -> List[str]:
    gt_list = item.get('ground_truth_list', '')
    if isinstance(gt_list, str):
        try:
            gt_list = json.loads(gt_list)
        except Exception:
            gt_list = [gt_list]
    if not isinstance(gt_list, list):
        gt_list = [str(gt_list)] if gt_list else []
    return [str(g) for g in gt_list if g]


def _coerce_sub_goals(item) -> List[Dict[str, Any]]:
    sg = item.get('sub_goals', '')
    if isinstance(sg, str):
        try:
            sg = json.loads(sg)
        except Exception:
            sg = []
    return sg if isinstance(sg, list) else []


def _grade_item_judge_vdr(model, item):
    """Default judge for BCV3. VDR paper contains-based judge — same
    helper as the rest of the search-bench family."""
    from .utils.judge_vdr import grade_with_judge_vdr

    prediction = str(item.get('prediction', ''))
    question = item.get('question', '')
    gt_list = _coerce_gt_list(item)
    if not gt_list:
        return dict(correct=0, reasoning='Empty ground truth')
    try:
        parsed = grade_with_judge_vdr(model, question, gt_list, prediction)
    except Exception as e:
        return dict(correct=0, reasoning=f'judge_vdr error: {e}')
    return dict(
        correct=int(parsed['correct']),
        reasoning=str(parsed.get('reasoning', '')),
    )


def _grade_item_judge_rs(model, item):
    """MMSearch-R1 paper rule-based judge."""
    from .utils.judge_rs import grade_with_judge_rs

    prediction = str(item.get('prediction', ''))
    question = item.get('question', '')
    gt_list = _coerce_gt_list(item)
    if not gt_list:
        return dict(correct=0, reasoning='Empty ground truth')
    try:
        parsed = grade_with_judge_rs(model, question, gt_list, prediction)
    except Exception as e:
        return dict(correct=0, reasoning=f'judge_rs error: {e}')
    return dict(
        correct=int(parsed['correct']),
        reasoning=str(parsed.get('reasoning', '')),
    )


def _grade_item_hle(model, item):
    """HLE-style binary judge."""
    from .utils.hle_judge import HLE_JUDGE_PROMPT, parse_hle_judgement

    prediction = str(item.get('prediction', ''))
    question = item.get('question', '')
    gt_list = _coerce_gt_list(item)
    if not gt_list:
        return dict(correct=0, reasoning='Empty ground truth')

    last = None
    for gt in gt_list:
        prompt = HLE_JUDGE_PROMPT.format(
            question=question,
            correct_answer=str(gt),
            response=prediction,
        )
        retry = 3
        while retry:
            out = model.generate(prompt)
            if 'Failed to obtain answer via API' in str(out):
                retry -= 1
                continue
            last = out
            parsed = parse_hle_judgement(out)
            if parsed.get('reasoning') != 'judge failed to parse':
                if parsed['correct']:
                    return dict(correct=1, reasoning=str(parsed.get('reasoning', '')))
                break  # parsed 'no' for this gt; try next gt
            retry -= 1
    return dict(correct=0, reasoning=str(last)[:500] if last else 'judge unreachable')


def _grade_item_judge_bcv3(model, item):
    """Verbatim port of the BrowseComp-V3 paper grader.

    Returns dict with ``correct`` (binary, derived from
    ``answer_correct``), ``process_score`` (float [0,1]), ``reasoning``
    (one-line summary), plus the full structured ``verdict`` for
    downstream auditing.

    Our framework only stores a flat prediction string per item, so we
    pass that string as both ``pred_answer`` AND as a single-element
    ``pred_subgoals`` list. The judge can still compute
    ``process_score`` because it checks whether each GT sub-goal's
    ``key_info`` appears within the prediction text (semantic match).
    """
    from .utils.judge_bcv3 import (
        build_judge_bcv3_prompt,
        parse_judge_bcv3_judgement,
    )

    prediction = str(item.get('prediction', ''))
    question = item.get('question', '')
    gt_answer = str(item.get('answer', '')) or (
        _coerce_gt_list(item)[0] if _coerce_gt_list(item) else ''
    )
    gt_subgoals = _coerce_sub_goals(item)
    pred_subgoals = [prediction] if prediction else []

    prompt = build_judge_bcv3_prompt(
        question=question,
        gt_answer=gt_answer,
        gt_subgoals=gt_subgoals,
        pred_answer=prediction,
        pred_subgoals=pred_subgoals,
    )

    retry = 3
    last = None
    while retry:
        out = model.generate(prompt)
        if 'Failed to obtain answer via API' in str(out):
            retry -= 1
            continue
        last = out
        parsed = parse_judge_bcv3_judgement(out)
        if parsed['verdict']:  # got a parseable JSON object
            return dict(
                correct=int(parsed['correct']),
                process_score=parsed['process_score'],
                reasoning=str(parsed['verdict'].get('judge_reasoning', '')),
                anomaly_flag=parsed.get('anomaly_flag', False),
                anomaly_reason=str(parsed.get('anomaly_reason', '')),
                confidence=parsed.get('confidence', ''),
                verdict=parsed['verdict'],
            )
        retry -= 1

    return dict(
        correct=0,
        process_score=0.0,
        reasoning=f'judge_bcv3 unparseable: {str(last)[:200]}' if last else 'judge unreachable',
        anomaly_flag=False,
        anomaly_reason='',
        confidence='',
        verdict={},
    )


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class BrowseCompV3Dataset(ImageBaseDataset):
    """BrowseComp-V3 (paper: project page halcyon-zhang.github.io/BrowseComp-V3).

    300 hand-crafted visually grounded multi-hop web search questions.
    Data ships AES-256-GCM encrypted; this class decrypts on first load
    and caches a plaintext TSV under ``$LMUData``. Images live inside
    the HF snapshot and are referenced by their absolute paths in the
    cached TSV.
    """

    TYPE = 'VQA'

    DATASET_URL = {
        'BrowseCompV3': '',
    }
    DATASET_MD5 = {}

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_URL)

    # -- Data loading: auto-download from HF + decrypt + cache TSV --------

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
        """Download HF snapshot, decrypt each row, return a DataFrame
        ready to TSV-cache.

        Image paths are kept as absolute paths inside the HF snapshot
        (no re-copying). If the user clears the HF cache the dataset
        will re-download on next access.
        """
        snap_root = _snapshot_download_bcv3()
        jsonl_path = osp.join(snap_root, _HF_JSONL_PATH)
        if not osp.isfile(jsonl_path):
            raise FileNotFoundError(
                f'BCV3 train.jsonl not found in HF snapshot: {jsonl_path}'
            )

        key = _derive_key(_BCV3_PASSPHRASE)
        rows: List[Dict[str, Any]] = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for raw_idx, line in enumerate(f):
                if not line.strip():
                    continue
                obj = json.loads(line)

                # Two encrypted-payload shapes are documented upstream:
                #   {encrypted_data: {question: ..., answer: ...}}
                #   {encrypted_question: ..., encrypted_answer: ...}
                # Handle both for compat.
                if 'encrypted_data' in obj:
                    enc = obj.get('encrypted_data', {})
                    question = _decrypt_field(enc.get('question', {}), key)
                    answer = _decrypt_field(enc.get('answer', {}), key)
                elif 'encrypted_question' in obj:
                    question = _decrypt_field(obj.get('encrypted_question', ''), key)
                    answer = _decrypt_field(obj.get('encrypted_answer', ''), key)
                else:
                    # Plaintext fallback (e.g., dev datasets)
                    question = str(obj.get('question', ''))
                    answer = str(obj.get('answer', ''))

                # image_paths is a JSON-encoded list in the raw file
                img_paths_raw = obj.get('image_paths', obj.get('image', []))
                if isinstance(img_paths_raw, str):
                    try:
                        img_paths_raw = json.loads(img_paths_raw)
                    except Exception:
                        img_paths_raw = [img_paths_raw]
                if not isinstance(img_paths_raw, list):
                    img_paths_raw = [img_paths_raw] if img_paths_raw else []

                # Resolve to absolute paths inside the HF snapshot
                resolved: List[str] = []
                for rel in img_paths_raw:
                    rel = str(rel)
                    if not rel:
                        continue
                    candidate = osp.join(snap_root, rel)
                    if not osp.isfile(candidate) and rel.startswith('data/'):
                        # Snapshot may store under a different prefix; try without 'data/'
                        candidate = osp.join(snap_root, rel[len('data/'):])
                    if osp.isfile(candidate):
                        resolved.append(candidate)
                    else:
                        # Keep the upstream relative path for debugging; the
                        # framework's image resolver will error visibly if the
                        # file isn't found at inference time.
                        resolved.append(rel)

                if len(resolved) == 1:
                    image_path_field = resolved[0]
                else:
                    image_path_field = json.dumps(resolved)

                # sub_goals is a JSON string in the raw file
                sub_goals_raw = obj.get('sub_goals', '[]')
                if isinstance(sub_goals_raw, str):
                    try:
                        # Validate it parses; re-serialize for clean TSV cell
                        sub_goals_list = json.loads(sub_goals_raw)
                    except Exception:
                        sub_goals_list = []
                elif isinstance(sub_goals_raw, list):
                    sub_goals_list = sub_goals_raw
                else:
                    sub_goals_list = []

                category = str(obj.get('category', 'unknown') or 'unknown')
                sub_category = str(obj.get('sub_category', 'unknown') or 'unknown')

                rows.append({
                    'index': obj.get('id', raw_idx),
                    'question': question,
                    'answer': answer,
                    'ground_truth_list': json.dumps([answer], ensure_ascii=False),
                    'sub_goals': json.dumps(sub_goals_list, ensure_ascii=False),
                    'image_path': image_path_field,
                    'category': category,
                    'l2-category': sub_category,
                })

        return pd.DataFrame(rows)

    # -- Prompt: image(s) + question, open-ended QA -----------------------

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = line['image_path']
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

    # -- Evaluation: 4 judges (vdr default / rs / hle / bcv3) ------------

    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate predictions with a configurable LLM judge.

        Judges (selected via ``MMDR_JUDGE_PROMPT`` env var or
        ``judge_kwargs['judge_prompt']``; default ``judge_vdr``):

          - ``judge_vdr`` (default): contains-based VDR paper judge
          - ``judge_rs``: MMSearch-R1 paper rule-based judge
          - ``hle``: HLE-style yes/no with confidence
          - ``judge_bcv3``: VERBATIM port of the BrowseComp-V3 paper
            grader. Produces ``answer_correct`` (binary) AND
            ``process_score`` (continuous [0,1] from sub-goal coverage).
            This is the only judge that consumes ``sub_goals``.
        """
        logger = get_logger('Evaluation')
        data = load(eval_file)
        assert 'prediction' in data, 'prediction column missing from eval file'
        data['prediction'] = [str(x) for x in data['prediction']]

        # Merge ``answer`` / ``ground_truth_list`` / ``sub_goals`` /
        # ``question`` from dataset metadata into the eval frame keyed
        # on ``index``.
        meta = self.data
        idx_str = lambda v: str(v)
        gt_map = {idx_str(idx): gt for idx, gt in zip(meta['index'], meta['ground_truth_list'])}
        ans_map = {idx_str(idx): a for idx, a in zip(meta['index'], meta['answer'])}
        sg_map = {idx_str(idx): sg for idx, sg in zip(meta['index'], meta['sub_goals'])}
        q_map = {idx_str(idx): q for idx, q in zip(meta['index'], meta['question'])}

        data['ground_truth_list'] = [gt_map.get(idx_str(i), '') for i in data['index']]
        data['answer'] = [ans_map.get(idx_str(i), '') for i in data['index']]
        data['sub_goals'] = [sg_map.get(idx_str(i), '[]') for i in data['index']]
        if 'question' not in data.columns:
            data['question'] = [q_map.get(idx_str(i), '') for i in data['index']]

        # Pop framework-level kwargs so they don't leak into the OpenAI
        # request payload.
        judge_kwargs = dict(judge_kwargs)
        nproc = judge_kwargs.pop('nproc', 4)
        judge_kwargs.pop('enable_llm_judge', None)

        judge_name = (
            os.environ.get('MMDR_JUDGE_PROMPT')
            or judge_kwargs.pop('judge_prompt', None)
            or 'judge_vdr'
        ).strip().lower()
        if judge_name not in ('judge_vdr', 'judge_rs', 'hle', 'judge_bcv3'):
            raise ValueError(
                f"Unknown judge_prompt {judge_name!r}; expected one of "
                f"'judge_vdr', 'judge_rs', 'hle', 'judge_bcv3'"
            )

        model = build_judge(**judge_kwargs)
        if model is None or not model.working():
            logger.error(
                'BrowseCompV3 evaluation requires a working judge model. '
                'Please configure judge_kwargs properly.'
            )
            logger.error(DEBUG_MESSAGE)
            return pd.DataFrame()

        if judge_name == 'judge_rs':
            grade_fn = _grade_item_judge_rs
            stem_base = '_bcv3_judge_rs'
        elif judge_name == 'hle':
            grade_fn = _grade_item_hle
            stem_base = '_bcv3_hle'
        elif judge_name == 'judge_bcv3':
            grade_fn = _grade_item_judge_bcv3
            stem_base = '_bcv3_judge_bcv3'
        else:
            grade_fn = _grade_item_judge_vdr
            stem_base = '_bcv3_judge_vdr'

        grade_stem = f'_grade{stem_base}'
        detail_stem = f'_detail{stem_base}'
        acc_stem = f'_acc{stem_base}'

        # Run grading with file cache for resumability
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

        # Map results back to data
        data['correct'] = [
            result_cache.get(str(idx), {}).get('correct', 0)
            for idx in data['index']
        ]
        data['reasoning'] = [
            result_cache.get(str(idx), {}).get('reasoning', '')
            for idx in data['index']
        ]
        # ``process_score`` and BCV3-only fields surface only when the
        # judge actually produced them (``judge_bcv3`` mode).
        if judge_name == 'judge_bcv3':
            data['process_score'] = [
                float(result_cache.get(str(idx), {}).get('process_score', 0.0))
                for idx in data['index']
            ]
            data['anomaly_flag'] = [
                bool(result_cache.get(str(idx), {}).get('anomaly_flag', False))
                for idx in data['index']
            ]
            data['confidence'] = [
                str(result_cache.get(str(idx), {}).get('confidence', ''))
                for idx in data['index']
            ]

        # Save detailed results
        detail_file = get_intermediate_file_path(eval_file, detail_stem)
        dump(data, detail_file)

        # Compute accuracy breakdown
        acc = self._compute_accuracy(data, with_process_score=(judge_name == 'judge_bcv3'))
        score_file = get_intermediate_file_path(eval_file, acc_stem, 'csv')
        dump(acc, score_file)
        logger.info(f'BrowseCompV3 [{judge_name}] accuracy saved to {score_file}')

        return acc

    @staticmethod
    def _compute_accuracy(data, with_process_score: bool = False):
        """Accuracy breakdown: Overall, per category (top-level domain),
        per ``l2-category`` (sub-domain). When ``with_process_score`` is
        True (i.e., we ran ``judge_bcv3``), add a ``process_score``
        column (mean of per-item process_score within each split).
        """
        rows = []

        def _row(name, subset):
            total = len(subset)
            correct = int(subset['correct'].sum())
            r = {
                'split': name,
                'total': total,
                'correct': correct,
                'accuracy': round(correct / total * 100, 2) if total > 0 else 0.0,
            }
            if with_process_score and 'process_score' in subset:
                r['process_score'] = round(float(subset['process_score'].mean()) * 100, 2) if total > 0 else 0.0
            return r

        rows.append(_row('Overall', data))
        if 'category' in data:
            for domain in sorted(data['category'].dropna().unique()):
                rows.append(_row(domain, data[data['category'] == domain]))
        if 'l2-category' in data:
            for sub in sorted(data['l2-category'].dropna().unique()):
                rows.append(_row(sub, data[data['l2-category'] == sub]))
        return pd.DataFrame(rows)
