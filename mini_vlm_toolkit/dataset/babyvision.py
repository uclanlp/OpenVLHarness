"""BabyVision: fine-grained visual discrimination, tracking and spatial perception.

Reference:
  https://huggingface.co/datasets/UnipatAI/BabyVision

388 items across 4 types and 22 subtypes. Two answer modes live in one dataset:
  * ansType == 'choice' (135): pick one of 2-6 options. Upstream `choiceAns` is a
    0-based index into `options`; it is converted to a letter (A/B/C/...) here because
    the questions themselves refer to the options as "A, B, C, D".
  * ansType == 'blank'  (253): free-form short answer -- coordinates, pair mappings,
    number lists, single counts.

The TSV builds itself on first use (see load_data), so no manual prep step is needed.

Scoring is rule-based exact match after normalisation; no LLM judge. The answer space is
closed and mechanical, so a judge would add cost and noise. The normalisation is the
interesting part -- see `normalize_answer` and `is_correct`.
"""
import json
import os
import os.path as osp
import re

import pandas as pd

from ..smp import LMUDataRoot, load, dump, encode_image_to_base64
from .image_base import ImageBaseDataset

HF_REPO = 'UnipatAI/BabyVision'

LETTERS = 'ABCDEFGHIJ'


def build_babyvision_tsv(out_path, repo=HF_REPO, verbose=True):
    """Materialise the BabyVision TSV from HuggingFace.

    `image_path` is load-bearing: ImageBaseDataset.dump_image() writes `<index>.jpg`
    unless the row supplies one, and PIL picks its encoder from the suffix. These are
    fine-grained discrimination tasks -- JPEG ringing around thin strokes is exactly the
    kind of artifact that could flip an answer -- so images are kept as lossless PNG.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on the user's env
        raise ImportError(
            'BabyVision builds its TSV from HuggingFace on first use and needs the '
            '`datasets` package: pip install datasets'
        ) from exc

    ds = load_dataset(repo)['train']
    if verbose:
        print(f'[BabyVision] building {out_path} from {repo} ({len(ds)} items)...')

    rows = []
    for i, rec in enumerate(ds):
        options = list(rec.get('options') or [])
        ans_type = rec.get('ansType')

        if ans_type == 'choice':
            idx = rec.get('choiceAns')
            # Upstream choiceAns is 0-based; questions refer to options as A/B/C/D.
            answer = LETTERS[idx] if isinstance(idx, int) and 0 <= idx < len(LETTERS) else ''
        else:
            answer = '' if rec.get('blankAns') is None else str(rec['blankAns'])

        rows.append({
            'index': i,
            'taskId': rec.get('taskId'),
            'type': rec.get('type'),
            'subtype': rec.get('subtype'),
            'ansType': ans_type,
            'question': rec.get('question'),
            'options': json.dumps(options, ensure_ascii=False),
            'answer': answer,
            'cot': rec.get('coT') or '',
            'image': encode_image_to_base64(rec['image'], fmt='PNG'),
            'image_path': f'{i}.png',
        })

    df = pd.DataFrame(rows)
    d = osp.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    # Write-then-rename: a crash mid-write must not leave a truncated TSV that later
    # runs would silently treat as complete.
    tmp = out_path + '.tmp'
    df.to_csv(tmp, sep='\t', index=False)
    os.replace(tmp, out_path)
    if verbose:
        print(f'[BabyVision] wrote {out_path} ({len(df)} rows)')
    return df


class BabyVision(ImageBaseDataset):

    TYPE = 'VQA'

    DATASET_URL = {
        'BabyVision': '',
    }
    DATASET_MD5 = {}

    TYPES = [
        'Fine-grained Discrimination',
        'Visual Tracking',
        'Spatial Perception',
        'Visual Pattern Recognition',
    ]

    # ------------------------------------------------------------ data load

    def load_data(self, dataset):
        """Build the TSV from HuggingFace on first use, then load normally.

        The base implementation would call download_file() with the bare filename
        'BabyVision.tsv' (there is no hosted TSV) and fail, so it is materialised here.
        Once the file exists this is a no-op.
        """
        tsv_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')
        if not osp.exists(tsv_path):
            build_babyvision_tsv(tsv_path)
        return super().load_data(dataset)

    # ---------------------------------------------------------------- prompt

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line)
        question = str(line['question'])

        try:
            options = json.loads(line['options']) if line.get('options') else []
        except (TypeError, ValueError):
            options = []

        parts = [question]
        if str(line.get('ansType')) == 'choice' and options:
            # Questions say "the options ... are A, B, C, D", but upstream `options`
            # is a list of descriptions, so the letters are attached explicitly here.
            parts.append('')
            parts.extend(f'{LETTERS[i]}. {opt}' for i, opt in enumerate(options))
            parts.append('\nAnswer with the option letter only.')
        else:
            parts.append('\nGive only the final answer, with no explanation.')

        msgs = [dict(type='image', value=p) for p in tgt_path]
        msgs.append(dict(type='text', value='\n'.join(parts)))
        return msgs

    # --------------------------------------------------------------- scoring

    @staticmethod
    def _strip_wrapper(s):
        s = s.strip()
        s = re.sub(r'^(the\s+)?(final\s+)?answer\s*(is|:)?\s*', '', s, flags=re.IGNORECASE)
        return s.strip().strip('.\'"`*(){}[] \t\n')

    @classmethod
    def normalize_answer(cls, ans):
        """Canonicalise an answer string into comparable tokens.

        Handles the formats actually present upstream:
          "(4,7)" / "10,5" / "Row 2, Column 3"   -> ('4','7') / ('10','5') / ('2','3')
          "2,3,5" / "6, 7, 9"                    -> ('2','3','5')
          "3 and 11"                             -> ('3','11')
          "1-7,2-9,3-10"                         -> pair set
        Returns (kind, value) where kind is 'pairs' (order-insensitive) or 'seq'.
        """
        if ans is None or (isinstance(ans, float) and pd.isna(ans)):
            return ('seq', ())
        s = str(ans).lower()
        s = cls._strip_wrapper(s)
        # "row 2, column 3" / "row 4 column 17" -> "2,3"
        m = re.findall(r'row\s*(\d+)\s*,?\s*column\s*(\d+)', s)
        if m:
            s = ','.join(f'{a},{b}' for a, b in m)
        s = s.replace(' and ', ',')
        # Coordinate groups like "(1,3)-(3,1)" keep their dash; drop the parens.
        s = s.replace('(', '').replace(')', '')
        # Split on commas / semicolons / whitespace runs.
        toks = [t.strip() for t in re.split(r'[,;\s]+', s) if t.strip()]
        toks = [t.strip('.') for t in toks]
        toks = [t for t in toks if t]
        if not toks:
            return ('seq', ())

        # A pure set of pairs ("1-7", "a-2") is a matching; the order the model lists
        # them in is arbitrary, so compare order-insensitively. Anything else (counts,
        # ordered sequences like Lines Observation) stays order-sensitive.
        if len(toks) > 1 and all(re.fullmatch(r'[^-]+-[^-]+', t) for t in toks):
            pairs = []
            for t in toks:
                a, b = t.split('-', 1)
                # "1-b" and "b-1" denote the same association.
                pairs.append(tuple(sorted((a.strip(), b.strip()))))
            return ('pairs', frozenset(pairs))

        return ('seq', tuple(cls._norm_num(t) for t in toks))

    @staticmethod
    def _norm_num(tok):
        """'07' -> '7' so numeric answers compare by value, not spelling."""
        try:
            return str(int(tok))
        except (ValueError, TypeError):
            return tok

    @classmethod
    def is_correct(cls, pred, gt):
        gt_kind, gt_val = cls.normalize_answer(gt)
        if not gt_val:
            return False
        pred_kind, pred_val = cls.normalize_answer(pred)
        if pred_kind == gt_kind and pred_val == gt_val:
            return True
        # A single-token gt may be quoted back inside a longer sentence; accept only an
        # exact token match, never a substring (avoids "1" matching "11").
        if gt_kind == 'seq' and len(gt_val) == 1 and pred_kind == 'seq':
            return len(pred_val) == 1 and pred_val[0] == gt_val[0]
        return False

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        assert 'prediction' in data and 'answer' in data

        data = data.copy()
        data['hit'] = [
            int(self.is_correct(p, g)) for p, g in zip(data['prediction'], data['answer'])
        ]

        detail_file = eval_file.replace('.xlsx', '_per_row.csv')
        try:
            cols = [c for c in ['index', 'taskId', 'type', 'subtype', 'ansType',
                                'answer', 'prediction', 'hit'] if c in data]
            dump(data[cols], detail_file)
        except Exception:
            pass

        res = {'overall': round(100.0 * data['hit'].mean(), 2) if len(data) else 0.0}

        # choice vs blank: guessing baseline differs sharply between them (1/3-1/6 vs ~0),
        # so a single overall number hides which half the model is actually solving.
        if 'ansType' in data:
            for at in ['choice', 'blank']:
                sub = data[data['ansType'] == at]
                if len(sub):
                    res[f'{at}_accuracy'] = round(100.0 * sub['hit'].mean(), 2)

        if 'type' in data:
            for t in self.TYPES:
                sub = data[data['type'] == t]
                if len(sub):
                    res[f'{t}_accuracy'] = round(100.0 * sub['hit'].mean(), 2)

        if 'subtype' in data:
            for st in sorted(data['subtype'].dropna().unique()):
                sub = data[data['subtype'] == st]
                res[f'subtype.{st}_accuracy'] = round(100.0 * sub['hit'].mean(), 2)

        score_df = pd.DataFrame([res])
        dump(score_df, eval_file.replace('.xlsx', '_acc.csv'))
        return score_df
