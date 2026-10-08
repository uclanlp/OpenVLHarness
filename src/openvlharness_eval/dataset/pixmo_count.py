"""PixmoCount (allenai/pixmo-count): object counting, train / val / test.

The HF dataset only ships Flickr image URLs, so on first use of a split we download
the images to $LMUData/images/PixmoCount/<sha256>.jpg and build the TSV. Images that
can no longer be fetched are dropped from the TSV (the upstream row index is kept,
so indices may have gaps).
"""
import io
import os
import os.path as osp
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .image_vqa import ImageVQADataset
from ..smp import *
from ..smp.file import LMUDataRoot, get_intermediate_file_path

HF_REPO = 'allenai/pixmo-count'
# Our split tag -> HF split name.
HF_SPLITS = {'train': 'train', 'val': 'validation', 'test': 'test'}
QUESTION_TEMPLATE = (
    'How many {label} are there in the image? '
    'If the same instance appears more than once, it should be counted as multiple instances. '
    'Your final answer should be a single integer.'
)
# Flickr returns 403 for the default python-requests user agent.
USER_AGENT = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36'


def _download_image(url, dest, timeout=30, retries=2):
    """Fetch one image and save it as RGB JPEG. Returns True on success.

    4xx responses and undecodable files fail immediately; 5xx and network errors are retried.
    """
    import requests
    from PIL import Image

    if osp.exists(dest):
        return True
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers={'User-Agent': USER_AGENT}, timeout=timeout)
        except Exception:
            time.sleep(0.5 * (attempt + 1))
            continue
        if 400 <= r.status_code < 500:
            return False
        if r.status_code != 200:
            time.sleep(0.5 * (attempt + 1))
            continue
        try:
            img = Image.open(io.BytesIO(r.content)).convert('RGB')
        except Exception:
            return False
        tmp = dest + '.tmp'
        img.save(tmp, 'JPEG', quality=95)
        os.replace(tmp, dest)
        return True
    return False


def build_pixmo_count_tsv(tsv_path, split, repo=HF_REPO, num_workers=16):
    """Download one PixmoCount split's images and write its TSV to ``tsv_path``."""
    from datasets import load_dataset

    img_dir = osp.join(osp.dirname(tsv_path), 'images', 'PixmoCount')
    os.makedirs(img_dir, exist_ok=True)
    ds = load_dataset(repo, split=HF_SPLITS[split])
    print(f'[PixmoCount] downloading {len(ds)} {split} images from their source URLs')

    meta = [(i, ex['image_url'], ex['image_sha256'], ex['label'], int(ex['count'])) for i, ex in enumerate(ds)]
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        ok = list(pool.map(lambda m: _download_image(m[1], osp.join(img_dir, f'{m[2]}.jpg')), meta))

    rows = [
        {'index': i, 'image_path': f'{sha}.jpg', 'question': QUESTION_TEMPLATE.format(label=label), 'answer': count}
        for (i, _, sha, label, count), good in zip(meta, ok) if good
    ]
    pd.DataFrame(rows, columns=['index', 'image_path', 'question', 'answer']).to_csv(tsv_path, sep='\t', index=False)
    print(f'[PixmoCount] wrote {len(rows)} rows to {tsv_path} ({len(meta) - len(rows)} images unavailable)')


class PixmoCount(ImageVQADataset):
    TYPE = 'VQA'
    DATASET_URL = {'PixmoCount_train': '', 'PixmoCount_val': '', 'PixmoCount_test': ''}
    DATASET_MD5 = {}

    def load_data(self, dataset):
        tsv_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')
        if not osp.exists(tsv_path):
            build_pixmo_count_tsv(tsv_path, split=dataset.rsplit('_', 1)[1])
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
        if model_name and model_name != 'exact_matching':
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

        answers = data['answer'].astype(int)

        diff = (predictions - answers)
        mae = diff.abs().mean()
        rmse = np.sqrt((diff ** 2).mean())
        acc = (predictions == answers).mean() * 100

        result = {'mae': round(float(mae), 2), 'rmse': round(float(rmse), 2), 'acc': round(float(acc), 2)}

        result_file = get_intermediate_file_path(eval_file, '_acc')
        dump(d2df(result), result_file)

        from .eval_bridge import emit_counting_per_row
        gt_safe = np.maximum(answers.to_numpy(), 1)
        graded = 1.0 - diff.abs().to_numpy() / gt_safe
        score_per_row = np.where(
            (predictions == answers).to_numpy(), 1.0, np.clip(graded, 0.0, None)
        )
        emit_counting_per_row(
            eval_file, data['index'], data['image_path'],
            extra_cols={
                'gt': answers.astype(int).tolist(),
                'pred': predictions.astype(int).tolist(),
                'exact_match': (predictions == answers).astype(int).tolist(),
                'abs_error': diff.abs().astype(int).tolist(),
                'sq_error': (diff ** 2).astype(int).tolist(),
            },
            sort_score=score_per_row.tolist(),
        )
        return result
