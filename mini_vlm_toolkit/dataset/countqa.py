"""CountQA (Jayant-Sravan/CountQA): object counting in cluttered real photos.

The TSV is built from the HuggingFace test split on first use. HF stores one row
per image with lists of questions/answers; we flatten to one row per question.
Images go to $LMUData/images/CountQA/.
"""
import json
import os
import os.path as osp

import pandas as pd

from .image_vqa import ImageVQADataset
from ..smp import *
from ..smp.file import LMUDataRoot, get_intermediate_file_path

HF_REPO = 'Jayant-Sravan/CountQA'
QUESTION_SUFFIX = ' Your final answer should be a single integer.'


def build_countqa_tsv(tsv_path, repo=HF_REPO):
    """Write the flattened CountQA TSV to ``tsv_path`` (images next to it under images/CountQA/)."""
    from datasets import load_dataset

    img_dir = osp.join(osp.dirname(tsv_path), 'images', 'CountQA')
    os.makedirs(img_dir, exist_ok=True)
    ds = load_dataset(repo, split='test')
    print(f'[CountQA] building TSV from {repo} ({len(ds)} images)')

    rows = []
    for i, ex in enumerate(ds):
        questions, answers = ex['questions'], ex['answers']
        assert len(questions) == len(answers), f'row {i}: {len(questions)} questions vs {len(answers)} answers'
        img_name = f'{i:04d}.jpg'
        img_path = osp.join(img_dir, img_name)
        if not osp.exists(img_path):
            img = ex['image']
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(img_path, 'JPEG', quality=95)
        for q_idx, (q, a) in enumerate(zip(questions, answers)):
            q = (q or '').rstrip()
            assert q, f'row {i} q{q_idx}: empty question'
            rows.append({
                'index': f'countqa_{i:04d}_q{q_idx}',
                'image_path': img_name,
                'question': q + QUESTION_SUFFIX,
                'answer': str(a),
                'objects': json.dumps(ex['objects']),
                'categories': json.dumps(ex['categories']),
                'is_focused': bool(ex['is_focused']),
            })
    pd.DataFrame(rows).to_csv(tsv_path, sep='\t', index=False)
    print(f'[CountQA] wrote {len(rows)} rows to {tsv_path}')


class CountQA(ImageVQADataset):
    TYPE = 'VQA'
    DATASET_URL = {'CountQA': ''}
    DATASET_MD5 = {}

    def load_data(self, dataset):
        tsv_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')
        if not osp.exists(tsv_path):
            build_countqa_tsv(tsv_path)
        return super().load_data(dataset)


    def _get_eval_method(self):
        return 'accuracy'

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        tgt_path = self.dump_image(line)
        msgs = []
        msgs.extend([dict(type='image', value=p) for p in tgt_path])
        msgs.append(dict(type='text', value=line['question']))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        import numpy as np
        from .utils.fsc147 import extract_integer_from_prediction

        data = load(eval_file)
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]

        model_name = judge_kwargs.get('model', None)
        if model_name:
            method = self._get_eval_method()
            data, _ = self._extract_with_judge(eval_file, data, method, **judge_kwargs)
            predictions = data['extracted_answer'].apply(
                lambda x: extract_integer_from_prediction(x)
                          if extract_integer_from_prediction(x) is not None else 0
            )
        else:
            predictions = data['prediction'].apply(
                lambda x: extract_integer_from_prediction(x)
                          if extract_integer_from_prediction(x) is not None else 0
            )

        answers = data['answer'].apply(
            lambda x: extract_integer_from_prediction(x)
                      if extract_integer_from_prediction(x) is not None else 0
        ).astype(int)
        predictions = predictions.astype(int)

        diff = (predictions - answers).abs()
        acc_per_row = (predictions == answers).astype(int)
        ra5_per_row = (diff <= 0.05 * answers).astype(int)
        ra10_per_row = (diff <= 0.10 * answers).astype(int)
        acc = acc_per_row.mean() * 100
        ra5 = ra5_per_row.mean() * 100
        ra10 = ra10_per_row.mean() * 100

        result = {
            'acc': round(float(acc), 2),
            'RA@5': round(float(ra5), 2),
            'RA@10': round(float(ra10), 2),
        }

        result_file = get_intermediate_file_path(eval_file, '_acc')
        dump(d2df(result), result_file)

        from .eval_bridge import emit_counting_per_row
        # CountQA carries an is_focused flag the user may want to slice on.
        is_focused = (
            data['is_focused'].tolist()
            if 'is_focused' in data.columns
            else [None] * len(data)
        )
        # Score: average of the three boolean metrics, in {0, 1/3, 2/3, 1}.
        score_per_row = (
            (acc_per_row + ra5_per_row + ra10_per_row) / 3.0
        ).tolist()
        emit_counting_per_row(
            eval_file, data['index'], data['image_path'],
            extra_cols={
                'gt': answers.tolist(),
                'pred': predictions.tolist(),
                'exact_match': acc_per_row.tolist(),
                'ra5': ra5_per_row.tolist(),
                'ra10': ra10_per_row.tolist(),
                'is_focused': is_focused,
            },
            sort_score=score_per_row,
        )
        return result
