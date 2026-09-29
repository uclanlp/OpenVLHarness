"""FREAK (hansQAQ/FREAK): counter-commonsense image questions, MCQ + open-ended.

Both TSVs are built from the HuggingFace test split on first use: images go to
$LMUData/images/FREAK/, MCQ options are shuffled (seed=42) so the correct answer
is not always A, and each row carries a ready-made ``prompt`` column.
"""
import json
import os
import os.path as osp
import random
import string
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from .image_mcq import ImageMCQDataset
from .image_vqa import ImageVQADataset
from .utils import build_judge
from ..smp import *
from ..smp.file import LMUDataRoot, get_intermediate_file_path

HF_REPO = 'hansQAQ/FREAK'


def build_freak_tsvs(data_root, repo=HF_REPO):
    """Write FREAK_MCQ.tsv and FREAK_VQA.tsv under ``data_root`` from the HF test split."""
    from datasets import load_dataset

    img_dir = osp.join(data_root, 'images', 'FREAK')
    os.makedirs(img_dir, exist_ok=True)
    ds = load_dataset(repo, split='test')
    print(f'[FREAK] building TSVs from {repo} ({len(ds)} examples)')

    rng = random.Random(42)
    mcq_records, vqa_records = [], []
    for ex in ds:
        img_name = f"{ex['id']}.jpg"
        img_path = osp.join(img_dir, img_name)
        if not osp.exists(img_path):
            ex['image'].save(img_path)

        question, gt, options = ex['question'], ex['ground_truth'], ex['options']
        category = json.dumps(ex.get('category') or [])

        if ex['type'] == 'mcq':
            if options is None:
                print(f"[FREAK] id={ex['id']}: type=mcq but no options, skipping")
                continue
            shuffled = list(options)
            rng.shuffle(shuffled)
            answer_letter = None
            for norm in (str.strip, lambda s: s.strip().lower()):
                for i, opt in enumerate(shuffled):
                    if norm(opt) == norm(gt):
                        answer_letter = string.ascii_uppercase[i]
                        break
                if answer_letter is not None:
                    break
            if answer_letter is None:
                print(f"[FREAK] id={ex['id']}: ground truth {gt!r} not in options, skipping")
                continue
            option_str = ', '.join(f'{string.ascii_uppercase[i]}. {opt}' for i, opt in enumerate(shuffled))
            mcq_records.append({
                'index': len(mcq_records),
                'image_path': img_name,
                'question': question,
                'prompt': f'{question} Choose one option from the following: {option_str}',
                'mcq_GT': answer_letter,
                'choices': json.dumps(shuffled),
                'text_GT': gt,
                'category': category,
            })
        elif ex['type'] == 'fib':
            vqa_records.append({
                'index': len(vqa_records),
                'image_path': img_name,
                'question': question,
                'prompt': f'{question} Provide your final answer in a clear and concise sentence or a few words.',
                'GT': gt,
                'category': category,
            })
        else:
            print(f"[FREAK] id={ex['id']}: unknown type {ex['type']!r}, skipping")

    pd.DataFrame(mcq_records).to_csv(osp.join(data_root, 'FREAK_MCQ.tsv'), sep='\t', index=False)
    pd.DataFrame(vqa_records).to_csv(osp.join(data_root, 'FREAK_VQA.tsv'), sep='\t', index=False)
    print(f'[FREAK] wrote {len(mcq_records)} MCQ and {len(vqa_records)} VQA rows to {data_root}')


def _ensure_freak_tsv(dataset):
    if not osp.exists(osp.join(LMUDataRoot(), f'{dataset}.tsv')):
        build_freak_tsvs(LMUDataRoot())


class FREAKMCQDataset(ImageMCQDataset):
    TYPE = 'MCQ'
    DATASET_URL = {'FREAK_MCQ': ''}
    DATASET_MD5 = {}

    def load_data(self, dataset):
        _ensure_freak_tsv(dataset)
        return super().load_data(dataset)


    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line)

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=line['prompt']))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        import json as json_mod
        from .utils.multiple_choice import extract_answer_from_item

        data = load(eval_file)
        data['prediction'] = [str(x) for x in data['prediction']]
        data['mcq_GT'] = [str(x) for x in data['mcq_GT']]

        # Expand the JSON 'choices' column into A, B, C, D, ... columns
        # so that extract_answer_from_item / build_choices can find them.
        if 'choices' in data.columns:
            for i in range(len(data)):
                try:
                    choices_list = json_mod.loads(data['choices'].iloc[i])
                except Exception:
                    choices_list = []
                for j, opt in enumerate(choices_list):
                    col = string.ascii_uppercase[j]
                    if col not in data.columns:
                        data[col] = pd.Series([None] * len(data), dtype=object)
                    data.at[data.index[i], col] = opt

        # Add a 'question' column if missing (extract_answer_from_item needs it)
        if 'question' not in data.columns and 'prompt' in data.columns:
            data['question'] = data['prompt']

        # Build judge model for LLM fallback
        model = judge_kwargs.get('model', 'exact_matching')
        if model == 'exact_matching':
            model = None
        else:
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                model = None

        extracted = []
        for i in tqdm(range(len(data)), desc='FREAK_MCQ eval'):
            item = data.iloc[i]
            res = extract_answer_from_item(model, item, dataset_name='FREAK_MCQ')
            extracted.append(res['opt'])

        data['extracted'] = extracted
        data['hit'] = [1 if e == a else 0 for e, a in zip(extracted, data['mcq_GT'])]

        # Derive name_str from judge model for result file naming
        model_name = judge_kwargs.get('model', 'exact_matching')
        name_str_map = {"chatgpt-0125": "openai", "gpt-4-0125": "gpt4"}
        name_str = name_str_map.get(model_name, model_name)
        if "/" in name_str:
            name_str = name_str.split("/")[-1]
        result_file = get_intermediate_file_path(eval_file, f'_{name_str}_result')
        dump(data, result_file)

        correct = sum(1 for e, a in zip(extracted, data['mcq_GT']) if e == a)
        total = len(data)

        ret = {'Overall': round(correct / total * 100, 2) if total > 0 else 0}
        if 'category' in data.columns:
            for c in sorted(set(str(x) for x in data['category'] if pd.notna(x))):
                mask = data['category'] == c
                sub_ext = [e for e, m in zip(extracted, mask) if m]
                sub_ans = data.loc[mask, 'mcq_GT'].tolist()
                sub_correct = sum(1 for e, a in zip(sub_ext, sub_ans) if e == a)
                ret[c] = round(sub_correct / len(sub_ext) * 100, 2) if sub_ext else 0

        ret = d2df(ret)
        result_file = get_intermediate_file_path(eval_file, '_acc')
        dump(ret, result_file)
        return ret


class FREAKVQADataset(ImageVQADataset):
    TYPE = 'VQA'
    DATASET_URL = {'FREAK_VQA': ''}
    DATASET_MD5 = {}

    def load_data(self, dataset):
        _ensure_freak_tsv(dataset)
        return super().load_data(dataset)


    def _get_eval_method(self):
        return 'accuracy'

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        tgt_path = self.dump_image(line)
        msgs = []
        msgs.extend([dict(type='image', value=p) for p in tgt_path])
        msgs.append(dict(type='text', value=line['prompt']))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        import re
        from .utils.verifier import QUESTION_QUALITY_PROMPT_EN_NO_COT

        data = load(eval_file)
        if 'GT' in data.columns and 'answer' not in data.columns:
            data.rename(columns={'GT': 'answer'}, inplace=True)
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]

        model = build_judge(**judge_kwargs)

        def _parse_judgment(text):
            """Parse A/B/C grade from judge response."""
            boxed = re.findall(r"boxed{([A-C])}", text)
            if boxed:
                return boxed[-1]
            text = text.strip().upper()
            if text in ('A', 'B', 'C'):
                return text
            text = text.split("FINAL JUDGMENT:")[-1]
            matches = re.findall(r"\(([A-C])\)", text)
            if matches:
                return matches[-1]
            matches = re.findall(r"([A-C])", text)
            if matches:
                return matches[-1]
            return 'C'

        scores = []
        for i in tqdm(range(len(data)), desc='FREAK_VQA eval'):
            row = data.iloc[i]
            prompt = QUESTION_QUALITY_PROMPT_EN_NO_COT.format(
                question=row['question'],
                gold_answer=row['answer'],
                llm_response=row['prediction'],
            )
            ans = model.generate(prompt)
            grade = _parse_judgment(ans)
            scores.append(1 if grade == 'A' else 0)

        data['hit'] = scores

        # Derive name_str from judge model for result file naming
        model_name = judge_kwargs.get('model', 'exact_matching')
        name_str_map = {"chatgpt-0125": "openai", "gpt-4-0125": "gpt4"}
        name_str = name_str_map.get(model_name, model_name)
        if "/" in name_str:
            name_str = name_str.split("/")[-1]
        result_file = get_intermediate_file_path(eval_file, f'_{name_str}_result')
        dump(data, result_file)

        ret = {'Overall': round(np.mean(data['hit']) * 100, 2)}
        if 'category' in data.columns:
            for c in sorted(set(str(x) for x in data['category'] if pd.notna(x))):
                mask = data['category'] == c
                ret[c] = round(np.mean(data.loc[mask, 'hit']) * 100, 2)

        ret = d2df(ret)
        result_file = get_intermediate_file_path(eval_file, '_acc')
        dump(ret, result_file)
        return ret
