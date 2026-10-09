"""MM-BrowseComp dataset for visual grounding & reasoning with web browsing.

Data format: JSONL with fields id, question, images (list of URLs), answer.
Compatible with BaseAgent and BaseAgentWithTools for inference.
"""

import json
import mimetypes
import os
import os.path as osp

import pandas as pd
import validators

from .image_base import ImageBaseDataset
from .hf_release import HF_SPLITS, load_hf_split
from .utils.judge_cache import require_judge
from .utils.vqa_eval import process_line, hit_calculate
from ..smp import (
    LMUDataRoot,
    decode_base64_to_image_file,
    d2df,
    download_file,
    dump,
    load,
    md5,
    mp,
    np,
    toliststr,
)
from ..smp.file import get_intermediate_file_path


def _resolve_to_local_path(s):
    """Resolve image reference to a local cache path.

    Supports:
    - Existing absolute/relative local paths (returned as-is or rooted under LMUData).
    - data:image/... base64 URLs (decoded once into LMUData/files).
    - HTTP/HTTPS URLs (downloaded once into LMUData/files).

    This keeps the same semantics as smp.file.parse_file so base/general code
    does not need MMBrowseComp-specific logic.
    """
    # 1) Existing path as-is (absolute or relative to CWD).
    if osp.exists(s) and s != '.' and osp.isfile(s):
        return s

    # 2) Try resolving relative paths under LMUData/images/MMBrowseComp.
    root = LMUDataRoot()
    img_root = osp.join(root, 'images', 'MMBrowseComp')

    # Support a few common relative patterns:
    # - "images/MMBrowseComp/xxx.png"
    # - "MMBrowseComp/xxx.png"
    # - "xxx.png" (assumed under images/MMBrowseComp)
    rel_candidates = [s]
    if not s.startswith('images/'):
        rel_candidates.append(osp.join('images', s))
    if not s.startswith('MMBrowseComp/'):
        rel_candidates.append(osp.join('MMBrowseComp', s))
    if not s.startswith('images/MMBrowseComp/'):
        rel_candidates.append(osp.join('images', 'MMBrowseComp', s))

    for rel in rel_candidates:
        cand = osp.join(root, rel)
        if osp.exists(cand) and osp.isfile(cand):
            return cand

    # 3) data:image/... base64 → decode to LMUData/files cache.
    if s.startswith('data:image/'):
        content = s[11:]
        content = ';'.join(content.split(';')[1:])
        if not content.startswith('base64,'):
            return s
        b64 = content[7:]
        os.makedirs(img_root, exist_ok=True)
        tgt = osp.join(img_root, md5(b64) + '.png')
        if not osp.exists(tgt):
            decode_base64_to_image_file(b64, tgt)
        return tgt

    # 4) HTTP/HTTPS URL → download to LMUData/files cache.
    if validators.url(s):
        suffix = osp.splitext(s)[1].lower()
        if suffix == '.webp' or suffix in mimetypes.types_map:
            os.makedirs(img_root, exist_ok=True)
            tgt = osp.join(img_root, md5(s) + suffix)
            if not osp.exists(tgt):
                download_file(s, tgt)
            return tgt
    return s


class MMBrowseCompDataset(ImageBaseDataset):
    """MM-BrowseComp: Multimodal browse-and-compare benchmark (VQA-style)."""

    TYPE = 'VQA'
    MODALITY = 'IMAGE'

    DATASET_URL = {"MMBrowseComp": "https://huggingface.co/datasets/mmbrowsecomp/MMBrowseComp"}
    DATASET_MD5 = {"MMBrowseComp": ""}

    def __init__(self, dataset='MMBrowseComp', skip_noimg=True, jsonl_path=None):
        self._jsonl_path = jsonl_path
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)

    def load_data(self, dataset):
        """Load MMBrowseComp data.

        A JSONL named explicitly via ``MMBROWSECOMP_JSONL`` or the ``jsonl_path`` argument wins;
        otherwise the hosted split on the OpenVLHarness eval repo is used (exactly the 243
        questions with images). JSONLs in the default ``$LMUData`` locations and the upstream
        ``mmbrowsecomp/MMBrowseComp`` dataset are only consulted if MMBrowseComp is removed from
        ``hf_release.HF_SPLITS``.
        """
        root = LMUDataRoot()
        mmbc_dir = osp.join(root, 'MMBrowseComp')

        explicit = [p for p in (os.environ.get('MMBROWSECOMP_JSONL'), self._jsonl_path) if p]
        missing = [p for p in explicit if not osp.isfile(p)]
        if missing:
            raise FileNotFoundError(f'MMBrowseComp JSONL not found: {missing[0]}')
        discovered = [
            osp.join(mmbc_dir, 'MMBrowseComp_400_decrypted.jsonl'),
            osp.join(mmbc_dir, 'MMBrowseComp_400.jsonl'),
            osp.join(root, 'MMBrowseComp_400_decrypted.jsonl'),
            osp.join(root, 'MMBrowseComp_400.jsonl'),
        ]
        discovered = [p for p in discovered if osp.isfile(p)]

        if explicit:
            df = self._load_from_jsonl(osp.abspath(explicit[0]))
            self._images_from_hf = False
        elif dataset in HF_SPLITS:
            df = load_hf_split(self, dataset)
        elif discovered:
            df = self._load_from_jsonl(osp.abspath(discovered[0]))
        else:
            df = self._load_from_hf(mmbc_dir)

        # Ensure 'level' column exists (defaults to 0 if not present)
        if 'level' not in df.columns:
            df['level'] = 0

        # Optional: limit number of samples via environment variable.
        # Set MMBROWSECOMP_MAX_SAMPLES to a positive integer to only keep N
        # samples; by default (unset or invalid) all samples are used.
        # Sampling strategy:
        # - default: keep the first N rows
        # - if MMBROWSECOMP_SAMPLE_MODE in {"random", "rand"}: randomly sample N rows
        max_samples = os.environ.get('MMBROWSECOMP_MAX_SAMPLES')
        if max_samples is not None:
            try:
                n = int(max_samples)
                if n > 0:
                    sample_mode = os.environ.get('MMBROWSECOMP_SAMPLE_MODE', '').strip().lower()
                    if sample_mode in {'random', 'rand'} and len(df) > n:
                        # Optional seed for reproducibility; default 42 if set but invalid.
                        seed_env = os.environ.get('MMBROWSECOMP_SAMPLE_SEED')
                        random_state = None
                        if seed_env is not None:
                            try:
                                random_state = int(seed_env)
                            except ValueError:
                                random_state = 42
                        df = df.sample(n=n, random_state=random_state)
                    else:
                        df = df.iloc[:n].copy()
            except ValueError:
                pass

        return df

    def _load_from_jsonl(self, jsonl_path: str) -> pd.DataFrame:
        """Load from local JSONL (legacy path)."""
        rows = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                item_id = item.get('id', len(rows))
                question = item.get('question', '')
                images = item.get('images', [])
                answer = item.get('answer', '')
                if not question:
                    continue
                if isinstance(images, str):
                    images = [images]
                image_path = [_resolve_to_local_path(im) for im in images]
                row = {
                    'index': item_id,
                    'question': question,
                    'answer': answer,
                    'image_path': image_path,
                }
                if 'category' in item:
                    row['category'] = item['category']
                if 'subtask' in item:
                    row['subtask'] = item['subtask']
                if 'level' in item and item['level'] is not None:
                    try:
                        row['level'] = int(item['level'])
                    except (ValueError, TypeError):
                        pass
                if 'checklist' in item:
                    row['checklist'] = item['checklist'] if isinstance(item['checklist'], list) else [item['checklist']]
                else:
                    row['checklist'] = []
                rows.append(row)
        return pd.DataFrame(rows)

    def _load_from_hf(self, cache_dir: str) -> pd.DataFrame:
        """Load from HuggingFace Hub (mmbrowsecomp/MMBrowseComp).

        NOTE: The HF dataset is **encrypted**. This method checks the first
        sample's question field — if it looks like base64 ciphertext rather
        than natural language, it raises an error directing the user to
        provide a pre-decrypted local JSONL file instead.
        """
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "The `datasets` library is required to load MMBrowseComp from HuggingFace.\n"
                "Install via `pip install datasets`. Or provide a local JSONL file."
            ) from exc

        env_cache = os.environ.get("HF_DATASETS_CACHE_DIR")
        hf_kwargs = {"split": "train"}
        if env_cache:
            hf_kwargs["cache_dir"] = env_cache
        ds = load_dataset("mmbrowsecomp/MMBrowseComp", **hf_kwargs)

        # Detect encrypted data: try to decode the first question as base64.
        # If it decodes successfully and the result is not readable UTF-8 text,
        # the data is still encrypted.
        if len(ds) > 0:
            sample_q = str(ds[0].get("question", ""))
            _is_encrypted = False
            if sample_q and " " not in sample_q:
                try:
                    import base64
                    base64.b64decode(sample_q, validate=True)
                    _is_encrypted = True  # valid base64, not natural language
                except Exception:
                    pass
            if _is_encrypted:
                raise RuntimeError(
                    "MMBrowseComp HuggingFace dataset appears to be encrypted. "
                    "This loader does not support automatic decryption. "
                    "Please provide a pre-decrypted JSONL file instead:\n"
                    "  1. Decrypt the dataset using the official MMBrowseComp tools\n"
                    "  2. Save as MMBrowseComp_400_decrypted.jsonl\n"
                    "  3. Place it under $LMUData/MMBrowseComp/ or set MMBROWSECOMP_JSONL env var"
                )

        os.makedirs(cache_dir, exist_ok=True)

        rows = []
        for i, item in enumerate(ds):
            item_id = item.get('id', i)
            question = item.get('question', '')
            if not question:
                continue
            answer = item.get('answer', '')
            images = item.get('images', [])
            if isinstance(images, str):
                images = [images]
            image_path = [_resolve_to_local_path(im) for im in images]
            row = {
                'index': item_id,
                'question': question,
                'answer': answer,
                'image_path': image_path,
            }
            if 'category' in item:
                row['category'] = item['category']
            if 'subtask' in item:
                row['subtask'] = item['subtask']
            if 'level' in item and item['level'] is not None:
                row['level'] = int(item['level'])
            if 'source' in item:
                row['source'] = item['source']
            if 'checklist' in item:
                row['checklist'] = item['checklist'] if isinstance(item['checklist'], list) else [item['checklist']]
            else:
                row['checklist'] = []
            rows.append(row)
        return pd.DataFrame(rows)

    def build_prompt(self, line):
        """Build prompt with image URLs. Order matches gen_answer.py: question first, then images."""
        if isinstance(line, int):
            line = self.data.iloc[line]
        tgt_path = toliststr(line['image_path'])
        question = line['question']
        msgs = [dict(type='text', value=question)]
        for p in tgt_path:
            msgs.append(dict(type='image', value=p))
        return msgs

    @classmethod
    def supported_datasets(cls):
        return ['MMBrowseComp']

    def evaluate(self, eval_file, **judge_kwargs):
        """MMBrowseComp uses LLM-as-a-judge by default. Set use_exact_match=True for quick exact-match only.

        Judge selection (via env var ``MMDR_JUDGE_PROMPT`` or
        ``judge_kwargs['judge_prompt']``):
          - default: paper-protocol ``mmbrowsecomp`` checklist judge
          - ``'judge_rs'``: multimodal-search-r1 binary judge (no
            checklist; only overall correctness). Score files are
            suffixed with ``_judge_rs`` to coexist with the default.
        """
        if judge_kwargs.get('use_exact_match', False):
            return self._evaluate_exact_match(eval_file, **judge_kwargs)
        require_judge(self.dataset_name, judge_kwargs)

        judge_name = (
            os.environ.get('MMDR_JUDGE_PROMPT')
            or judge_kwargs.get('judge_prompt', None)
            or 'mmbrowsecomp'
        ).strip().lower()
        if judge_name not in ('mmbrowsecomp', 'judge_rs', 'judge_vdr'):
            raise ValueError(
                f"Unknown judge_prompt {judge_name!r}; expected 'mmbrowsecomp', 'judge_rs', or 'judge_vdr'"
            )

        if judge_name == 'judge_rs':
            return self._evaluate_judge_rs(eval_file, **judge_kwargs)
        if judge_name == 'judge_vdr':
            return self._evaluate_judge_vdr(eval_file, **judge_kwargs)
        return self._evaluate_judge(eval_file, **judge_kwargs)

    def _evaluate_exact_match(self, eval_file, **judge_kwargs):
        """Exact match fallback for quick testing."""
        from functools import partial

        data = load(eval_file)
        if isinstance(data, list):
            data = pd.DataFrame(data)
        dataset = self.dataset_name
        assert 'answer' in data and 'prediction' in data
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]
        lt = len(data)
        pool = mp.Pool(16)
        lines = [data.iloc[i] for i in range(lt)]
        res = pool.map(partial(process_line, method='accuracy'), lines)

        data['eval_gt'] = [r['gt'] for r in res]
        data['eval_pred'] = [r['pred'] for r in res]
        data['eval_match'] = [r['match'] for r in res]
        data['eval_score'] = [np.mean(r['match']) for r in res]

        detailed_result_file = get_intermediate_file_path(eval_file, '_results')
        dump(data, detailed_result_file)

        hit = hit_calculate(res, dataset)
        ret = {'Overall': np.mean(hit) * 100}
        if 'category' in data:
            cates = sorted(set(data['category']))
            for c in cates:
                sub = [r for line, r in zip(lines, res) if line.get('category') == c]
                if sub:
                    hit_sub = hit_calculate(sub, dataset)
                    ret[c] = np.mean(hit_sub) * 100
        ret = d2df(ret)
        ret = ret.round(2)
        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(ret, score_file)
        return ret

    def _evaluate_judge(self, eval_file, **judge_kwargs):
        """LLM-as-a-judge evaluation (from MM-BrowseComp eval.py)."""
        from openai import OpenAI
        from tqdm import tqdm

        from .utils.mmbrowsecomp_judge import (
            call_judge_api,
            construct_judge_prompt,
            parse_judge_response,
        )

        data = load(eval_file)
        if isinstance(data, list):
            data = pd.DataFrame(data)
        assert 'answer' in data and 'prediction' in data
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]
        if 'checklist' not in data:
            data['checklist'] = [[]] * len(data)

        from .utils.judge_util import resolve_judge_max_tokens

        judge_model = judge_kwargs.get('model')
        judge_key = judge_kwargs.get('key', 'empty')
        judge_base = judge_kwargs.get('api_base')
        # Custom OpenAI client path (not via build_judge); resolve from yaml
        # → JUDGE_MAX_TOKENS_DEFAULT centrally.
        judge_max_tokens = resolve_judge_max_tokens(judge_kwargs)

        if not judge_model:
            raise ValueError(
                'MMBrowseComp judge evaluation requires judge model. '
                'Set model_name in judge config (configs/judge.yaml) or pass model in judge_kwargs.'
            )

        api_key = judge_key if judge_key and str(judge_key).lower() != 'empty' else os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                'Judge API key required. Set key in judge config or export OPENAI_API_KEY.'
            )

        client_params = {'api_key': api_key}
        if judge_base:
            client_params['base_url'] = judge_base
        client = OpenAI(**client_params)

        lines = [data.iloc[i] for i in range(len(data))]
        overall_correct_list = []
        checklist_scores = []
        strict_correct_list = []

        for line in tqdm(lines, desc='MMBrowseComp judge'):
            question_full = line['question']
            question = question_full.split('Question: ')[-1] if 'Question: ' in question_full else question_full
            ref_answer = line['answer']
            pred = line['prediction']
            checklist = line.get('checklist', [])
            if not isinstance(checklist, list):
                if checklist is None or (isinstance(checklist, str) and str(checklist).lower() in ('nan', '')):
                    checklist = []
                elif isinstance(checklist, str):
                    try:
                        import ast
                        checklist = ast.literal_eval(checklist)
                    except (ValueError, SyntaxError):
                        checklist = []
                else:
                    checklist = [checklist] if checklist else []

            prompt = construct_judge_prompt(question, pred, ref_answer, checklist)
            response = call_judge_api(client, judge_model, [{'role': 'user', 'content': prompt}], judge_max_tokens)
            parsed = parse_judge_response(response)

            overall_yes = parsed.get('overall_correctness') == 'YES'
            overall_correct_list.append(1.0 if overall_yes else 0.0)

            cl_correct = parsed.get('checklist_correct_count')
            cl_total = parsed.get('checklist_total_count')
            if isinstance(cl_correct, int) and isinstance(cl_total, int) and cl_total > 0:
                checklist_scores.append(cl_correct / cl_total)
            else:
                checklist_scores.append(None)

            strict = (
                overall_yes
                and isinstance(cl_correct, int)
                and isinstance(cl_total, int)
                and cl_total > 0
                and cl_correct == cl_total
            )
            strict_correct_list.append(1.0 if strict else 0.0)

        data['judge_overall_correct'] = overall_correct_list
        data['judge_checklist_score'] = checklist_scores
        data['judge_strict_correct'] = strict_correct_list

        detailed_result_file = get_intermediate_file_path(eval_file, '_judge_results')
        dump(data, detailed_result_file)

        overall_acc = np.mean(overall_correct_list) * 100
        valid_cl = [s for s in checklist_scores if s is not None]
        avg_checklist = (np.mean(valid_cl) * 100) if valid_cl else None
        strict_acc = np.mean(strict_correct_list) * 100

        ret = {
            'Overall': overall_acc,
            'Strict': strict_acc,
        }
        if avg_checklist is not None:
            ret['Avg_Checklist'] = avg_checklist

        if 'category' in data:
            cates = sorted((c for c in set(data['category']) if pd.notna(c) and c != ''))
            for c in cates:
                mask = data['category'] == c
                sub_overall = [overall_correct_list[i] for i in range(len(data)) if mask.iloc[i]]
                if sub_overall:
                    ret[c] = np.mean(sub_overall) * 100

        ret = d2df(ret)
        ret = ret.round(2)
        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(ret, score_file)
        return ret

    def _evaluate_judge_rs(self, eval_file, **judge_kwargs):
        """LLM-as-a-judge using Judge_RS (multimodal-search-r1 binary
        protocol). Produces ONLY overall correctness — no checklist
        breakdown (Judge_RS prompt doesn't have a checklist concept).
        Output files use the ``_judge_rs`` suffix to keep the default
        mmbrowsecomp checklist judge results intact.
        """
        from tqdm import tqdm

        from .utils import build_judge
        from .utils.judge_rs import grade_with_judge_rs

        data = load(eval_file)
        if isinstance(data, list):
            data = pd.DataFrame(data)
        assert 'answer' in data and 'prediction' in data
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]

        judge_kwargs_local = dict(judge_kwargs)
        judge_kwargs_local.pop('use_exact_match', None)
        judge_kwargs_local.pop('judge_prompt', None)
        judge_kwargs_local.pop('enable_llm_judge', None)
        # ``judge_max_tokens`` is consumed centrally in build_judge() (yaml is
        # the single source of truth; no per-evaluator default).
        judge_kwargs_local.pop('nproc', None)

        model = build_judge(**judge_kwargs_local)
        if model is None or not model.working():
            raise RuntimeError(
                'MMBrowseComp Judge_RS evaluation requires a working judge model. '
                'Please configure judge_kwargs properly.'
            )

        lines = [data.iloc[i] for i in range(len(data))]
        overall_correct_list = []

        for line in tqdm(lines, desc='MMBrowseComp judge_rs'):
            question_full = line['question']
            question = (
                question_full.split('Question: ')[-1]
                if 'Question: ' in question_full else question_full
            )
            ref_answer = line['answer']
            pred = line['prediction']

            try:
                parsed = grade_with_judge_rs(model, question, ref_answer, pred)
                overall_correct_list.append(1.0 if parsed.get('correct') == 1 else 0.0)
            except Exception:
                overall_correct_list.append(0.0)

        data['judge_overall_correct'] = overall_correct_list

        detailed_result_file = get_intermediate_file_path(eval_file, '_judge_results_judge_rs')
        dump(data, detailed_result_file)

        overall_acc = float(np.mean(overall_correct_list)) * 100
        ret = {'Overall': overall_acc}

        if 'category' in data:
            cates = sorted((c for c in set(data['category']) if pd.notna(c) and c != ''))
            for c in cates:
                mask = data['category'] == c
                sub = [overall_correct_list[i] for i in range(len(data)) if mask.iloc[i]]
                if sub:
                    ret[c] = float(np.mean(sub)) * 100

        ret = d2df(ret)
        ret = ret.round(2)
        score_file = get_intermediate_file_path(eval_file, '_acc_judge_rs', 'csv')
        dump(ret, score_file)
        return ret

    def _evaluate_judge_vdr(self, eval_file, **judge_kwargs):
        """LLM-as-a-judge using Judge_VDR (VDR paper contains-based protocol).
        Produces ONLY overall correctness — no checklist breakdown. Output
        files use the ``_judge_vdr`` suffix to keep results separate from the
        default mmbrowsecomp checklist judge and from Judge_RS.
        """
        from tqdm import tqdm

        from .utils import build_judge
        from .utils.judge_vdr import grade_with_judge_vdr

        data = load(eval_file)
        if isinstance(data, list):
            data = pd.DataFrame(data)
        assert 'answer' in data and 'prediction' in data
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]

        judge_kwargs_local = dict(judge_kwargs)
        judge_kwargs_local.pop('use_exact_match', None)
        judge_kwargs_local.pop('judge_prompt', None)
        judge_kwargs_local.pop('enable_llm_judge', None)
        judge_kwargs_local.pop('nproc', None)

        model = build_judge(**judge_kwargs_local)
        if model is None or not model.working():
            raise RuntimeError(
                'MMBrowseComp Judge_VDR evaluation requires a working judge model. '
                'Please configure judge_kwargs properly.'
            )

        lines = [data.iloc[i] for i in range(len(data))]
        overall_correct_list = []

        for line in tqdm(lines, desc='MMBrowseComp judge_vdr'):
            question_full = line['question']
            question = (
                question_full.split('Question: ')[-1]
                if 'Question: ' in question_full else question_full
            )
            ref_answer = line['answer']
            pred = line['prediction']

            try:
                parsed = grade_with_judge_vdr(model, question, ref_answer, pred)
                overall_correct_list.append(1.0 if parsed.get('correct') == 1 else 0.0)
            except Exception:
                overall_correct_list.append(0.0)

        data['judge_overall_correct'] = overall_correct_list

        detailed_result_file = get_intermediate_file_path(eval_file, '_judge_results_judge_vdr')
        dump(data, detailed_result_file)

        overall_acc = float(np.mean(overall_correct_list)) * 100
        ret = {'Overall': overall_acc}

        if 'category' in data:
            cates = sorted((c for c in set(data['category']) if pd.notna(c) and c != ''))
            for c in cates:
                mask = data['category'] == c
                sub = [overall_correct_list[i] for i in range(len(data)) if mask.iloc[i]]
                if sub:
                    ret[c] = float(np.mean(sub)) * 100

        ret = d2df(ret)
        ret = ret.round(2)
        score_file = get_intermediate_file_path(eval_file, '_acc_judge_vdr', 'csv')
        dump(ret, score_file)
        return ret
