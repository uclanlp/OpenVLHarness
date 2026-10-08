"""ActiveVision: visual reasoning that needs iterative scanning, tracing and comparison.

Reference:
  https://huggingface.co/datasets/activevisionai/ActiveVision  (CC-BY-4.0)

85 items over 17 tasks in 3 families (distributed_scanning, sequential_traversal,
visual_attribute_transfer). Free-form short answers -- integers, letters, or ordered
sequences -- wrapped by the model in <answer>...</answer>. Not multiple choice, so the
guessing baseline is ~0.

Scoring is exact match after normalisation (case, whitespace and separators ignored, so
"C, A, G" == "CAG"; integers compared numerically, so "07" == 7). No LLM judge is used --
the answer space is closed and a judge would only add noise and cost.

The TSV builds itself: the first run on a machine without $LMUData/ActiveVision.tsv
pulls the dataset from HuggingFace and writes it, so no manual prep step is needed.
`scripts/prepare_activevision.py` is a thin CLI over the same function for anyone who
wants to build it ahead of time.
"""
import os.path as osp
import re

import pandas as pd

from ..smp import LMUDataRoot, load, dump, encode_image_to_base64
from .image_base import ImageBaseDataset

HF_REPO = 'activevisionai/ActiveVision'


def build_activevision_tsv(out_path, repo=HF_REPO, verbose=True):
    """Materialise the ActiveVision TSV from the HuggingFace dataset.

    Images are stored as base64 PNG. Lossless matters here: several tasks turn on thin
    strokes and exact pixel adjacency, so JPEG artifacts could change the correct answer.

    The `image_path` column is load-bearing, not decorative --
    ImageBaseDataset.dump_image() writes `<index>.jpg` unless the row supplies an explicit
    image_path, and PIL picks its encoder from the suffix. Without the `.png` names the
    lossless TSV would be re-encoded to JPEG on the way to the model.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on the user's env
        raise ImportError(
            "ActiveVision builds its TSV from HuggingFace on first use and needs the "
            "`datasets` package: pip install datasets"
        ) from exc

    ds = load_dataset(repo)['train']
    if verbose:
        print(f'[ActiveVision] building {out_path} from {repo} ({len(ds)} items)...')

    rows = []
    for i, rec in enumerate(ds):
        rows.append({
            'index': i,
            'id': rec['id'],
            'task': rec['task'],
            'category': rec['category'],
            'question': rec['question'],
            'answer': str(rec['answer']),
            'image': encode_image_to_base64(rec['image'], fmt='PNG'),
            'image_path': f'{i}.png',
        })

    df = pd.DataFrame(rows)
    osp_dir = osp.dirname(out_path)
    if osp_dir:
        import os
        os.makedirs(osp_dir, exist_ok=True)
    # Write to a temp file then rename, so a crash mid-write cannot leave a
    # half-written TSV that later runs would silently treat as complete.
    tmp_path = out_path + '.tmp'
    df.to_csv(tmp_path, sep='\t', index=False)
    import os
    os.replace(tmp_path, out_path)
    if verbose:
        print(f'[ActiveVision] wrote {out_path} ({len(df)} rows)')
    return df


class ActiveVision(ImageBaseDataset):

    TYPE = 'VQA'

    # Test-only benchmark; the upstream `train` split is the whole thing.
    # Built locally by scripts/prepare_activevision.py rather than downloaded as a TSV.
    DATASET_URL = {
        'ActiveVision': '',
    }
    DATASET_MD5 = {}

    CATEGORIES = [
        'distributed_scanning',
        'sequential_traversal',
        'visual_attribute_transfer',
    ]

    # ------------------------------------------------------------ data load

    def load_data(self, dataset):
        """Build the TSV from HuggingFace on first use, then load it normally.

        The base implementation would call download_file() with the bare filename
        'ActiveVision.tsv' (there is no hosted TSV for this benchmark) and fail, so the
        file is materialised here first. Once it exists this is a no-op and the usual
        LMUData path is used.
        """
        tsv_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')
        if not osp.exists(tsv_path):
            build_activevision_tsv(tsv_path)
        return super().load_data(dataset)

    # ---------------------------------------------------------------- prompt

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line)

        # The upstream `question` already states the task and demands the
        # <answer>...</answer> wrapper, so it is passed through verbatim.
        msgs = [dict(type='image', value=p) for p in tgt_path]
        msgs.append(dict(type='text', value=line['question']))
        return msgs

    # ------------------------------------------------------------- scoring

    @staticmethod
    def extract_answer(text):
        """Pull the last <answer>...</answer> payload out of a model response.

        Last rather than first: agents often restate the tag while reasoning, and the
        final one is the committed answer. Falls back to the raw text when the model
        omitted the tag, so a correct bare answer still scores.
        """
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return ''
        text = str(text)
        matches = re.findall(r'<answer>(.*?)</answer>', text, flags=re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[-1].strip()
        # Tolerate an unclosed tag, e.g. output truncated at max_tokens.
        m = re.search(r'<answer>(.*)', text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return text.strip()

    @staticmethod
    def normalize(ans):
        """Canonical form for exact match: lowercase, strip separators/whitespace.

        "C, A, G" -> "cag";  "07" -> "07" (numeric equality handled separately).
        """
        if ans is None:
            return ''
        s = str(ans).strip().lower()
        # Drop wrapping punctuation/markup the model may add around the payload.
        s = s.strip('.\'"`*[]{}() \t\n')
        # Separators carry no meaning in this answer space.
        s = re.sub(r'[\s,;:/\\|\-_>+]', '', s)
        return s

    @classmethod
    def is_correct(cls, pred, gt):
        p, g = cls.normalize(pred), cls.normalize(gt)
        if p == '' and g != '':
            return False
        if p == g:
            return True
        # Integer answers compare numerically ("07" == "7").
        try:
            return int(p) == int(g)
        except (ValueError, TypeError):
            return False

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        assert 'prediction' in data and 'answer' in data

        preds = [self.extract_answer(x) for x in data['prediction']]
        hits = [
            self.is_correct(p, g)
            for p, g in zip(preds, data['answer'])
        ]

        data = data.copy()
        data['extracted'] = preds
        data['hit'] = [int(h) for h in hits]

        # Per-row detail next to the predictions, for error analysis.
        detail_file = eval_file.replace('.xlsx', '_per_row.csv')
        try:
            dump(data[['index', 'id', 'task', 'category', 'answer', 'extracted', 'hit']], detail_file)
        except Exception:
            pass

        overall = 100.0 * sum(hits) / max(len(hits), 1)
        res = {'overall': round(overall, 2)}

        # Family-level breakdown, in a stable order.
        if 'category' in data:
            for cat in self.CATEGORIES:
                sub = data[data['category'] == cat]
                if len(sub):
                    res[f'{cat}_accuracy'] = round(100.0 * sub['hit'].sum() / len(sub), 2)

        # Task-level breakdown (17 tasks, ~5 items each -- noisy individually,
        # but useful for spotting a task the model cannot do at all).
        if 'task' in data:
            for task in sorted(data['task'].unique()):
                sub = data[data['task'] == task]
                res[f'task.{task}_accuracy'] = round(100.0 * sub['hit'].sum() / len(sub), 2)

        score_df = pd.DataFrame([res])
        score_file = eval_file.replace('.xlsx', '_acc.csv')
        dump(score_df, score_file)
        return score_df
