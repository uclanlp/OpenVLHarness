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

    DATASET_URL = {}  # Not using URL; loading from local JSONL
    DATASET_MD5 = {}

    def __init__(self, dataset='MMBrowseComp', skip_noimg=True, jsonl_path=None):
        self._jsonl_path = jsonl_path
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)

    def load_data(self, dataset):
        """Load from JSONL. Expects LMUData/MMBrowseComp_decrypted.jsonl or MMBrowseComp.jsonl."""
        root = LMUDataRoot()
        mmbc_dir = osp.join(root, 'MMBrowseComp')
        candidates = [
            os.environ.get('MMBROWSECOMP_JSONL'),
            self._jsonl_path,
            # Prefer the 400-sample decrypted file; fall back to non-decrypted variant if present.
            osp.join(mmbc_dir, 'MMBrowseComp_400_decrypted.jsonl'),
            osp.join(mmbc_dir, 'MMBrowseComp_400.jsonl'),
            osp.join(root, 'MMBrowseComp_400_decrypted.jsonl'),
            osp.join(root, 'MMBrowseComp_400.jsonl'),
        ]
        candidates = [osp.abspath(p) for p in candidates if p and osp.isfile(p)]

        jsonl_path = candidates[0] if candidates else None
        if jsonl_path is None:
            raise FileNotFoundError(
                f'MMBrowseComp JSONL not found. Please place MMBrowseComp_decrypted.jsonl '
                f'or MMBrowseComp.jsonl under {root}/MMBrowseComp/, or pass jsonl_path to build_dataset.'
            )

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
                # Resolve data URLs / HTTP URLs to local cache paths here (only decode/download if not exists),
                # so base class and smp.file need no MMBrowseComp-specific handling.
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
                if 'checklist' in item:
                    row['checklist'] = item['checklist'] if isinstance(item['checklist'], list) else [item['checklist']]
                else:
                    row['checklist'] = []
                rows.append(row)
        df = pd.DataFrame(rows)

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
        """MMBrowseComp uses LLM-as-a-judge by default. Set use_exact_match=True for quick exact-match only."""
        if judge_kwargs.get('use_exact_match', False):
            return self._evaluate_exact_match(eval_file, **judge_kwargs)
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

        judge_model = judge_kwargs.get('model')
        judge_key = judge_kwargs.get('key', 'empty')
        judge_base = judge_kwargs.get('api_base')
        judge_max_tokens = judge_kwargs.get('judge_max_tokens', 5120)

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
