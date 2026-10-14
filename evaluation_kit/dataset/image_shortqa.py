# Adapted from VLMEvalKit (https://github.com/open-compass/VLMEvalKit), Apache License 2.0; modified by the OpenVLHarness authors.
import ast

from evaluation_kit import *
from ..smp import *
from .image_base import ImageBaseDataset
from .hf_release import HF_SPLITS, load_hf_split
from .utils import build_judge
from .utils.judge_cache import guard_cache, require_judge
from .utils.hle_judge import HLE_JUDGE_PROMPT, parse_hle_judgement
from .utils.multiple_choice import report_acc
from ..utils import track_progress_rich
from ..smp.file import get_intermediate_file_path


def _gt_with_alternatives(item):
    """Build a list `[primary_gt, *alternatives]` from a row.

    Handles three TSV conventions:
      - ``answer`` only (HLE, MMSearch, BCVL, ...): single string OR
        already a "[a, b, c]" JSON list string.
      - ``answer`` + ``alternative_gt_answers`` (FVQA): primary + JSON
        list of synonyms / variants. Both fields merged into one
        candidate list so Judge_RS / Judge_VDR / HLE can match any.
      - ``answer`` + ``answer_eval`` (Infoseek_GDM): same idea, separate
        column name. Merge if present.

    Returns a non-empty list of strings.
    """
    answer = item.get('answer', '')
    if isinstance(answer, str) and answer.strip().startswith('[') and answer.strip().endswith(']'):
        try:
            answer = ast.literal_eval(answer)
        except Exception:
            answer = [answer]
    if not isinstance(answer, list):
        answer = [answer]
    # FVQA-style alternatives column
    alts_raw = item.get('alternative_gt_answers', '') or item.get('answer_eval', '')
    if isinstance(alts_raw, str) and alts_raw.strip():
        try:
            alt_list = json.loads(alts_raw) if alts_raw.strip().startswith('[') else []
            if isinstance(alt_list, list) and alt_list:
                answer = list(answer) + [str(a) for a in alt_list if str(a).strip()]
        except Exception:
            pass
    # Deduplicate while preserving order
    seen = set()
    out = []
    for a in answer:
        s = str(a).strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out or ['']


def HLE_style_auxeval(model, item):
    """HLE-style binary judge for short-answer datasets (``judge_prompt: hle``).

    Returns ``{hit: 0|1, log: str}``
    but uses the shared HLE judge prompt + JSON-extracting parser, so
    the resulting accuracy is directly comparable to WebWatcher / BCVL /
    other HLE-protocol scorers.

    For datasets with alternative answers (FVQA, Infoseek_GDM), uses
    ``_gt_with_alternatives`` to merge candidates; HLE judge only sees
    the primary GT but the secondary candidates are appended into the
    prompt's correct_answer field as ``primary (also accepted: x, y)``.
    """
    item = item.iloc[0] if isinstance(item, pd.DataFrame) else item
    question = str(item.get('question', ''))
    answer = _gt_with_alternatives(item)
    if len(answer) > 1:
        correct_answer = f"{answer[0]} (also accepted: {', '.join(answer[1:])})"
    else:
        correct_answer = answer[0]
    response = str(item.get('prediction', ''))

    prompt = HLE_JUDGE_PROMPT.format(
        question=question,
        correct_answer=correct_answer,
        response=response,
    )
    retry = 3
    last = None
    for _ in range(retry):
        out = model.generate(prompt)
        if 'Failed to obtain answer via API' in (out or ''):
            continue
        last = out
        parsed = parse_hle_judgement(out)
        if parsed.get('reasoning') != 'judge failed to parse':
            log = f"verdict={parsed['correct']} extracted={parsed.get('extracted_answer','')[:120]} reasoning={parsed.get('reasoning','')[:200]}"
            return dict(hit=int(parsed['correct']), log=log)
    return dict(hit=0, log=f'judge unparseable: {str(last)[:200]}' if last else 'judge unreachable')


def Judge_RS_auxeval(model, item):
    """Judge_RS variant of ``HLE_style_auxeval``.

    Sends ``JUDGE_RS_SYSTEM_PROMPT`` + ``JUDGE_RS_USER_PROMPT`` as
    separate roles via ``grade_with_judge_rs`` helper. Returns the
    same ``{hit, log}`` shape so the downstream aggregation is
    drop-in compatible. Opt-in via env ``MMDR_JUDGE_PROMPT=judge_rs``.
    """
    from .utils.judge_rs import grade_with_judge_rs

    item = item.iloc[0] if isinstance(item, pd.DataFrame) else item
    question = str(item.get('question', ''))
    answer = _gt_with_alternatives(item)
    response = str(item.get('prediction', ''))

    try:
        parsed = grade_with_judge_rs(model, question, answer, response)
    except Exception as e:
        return dict(hit=0, log=f'judge_rs error: {e}')

    log = f"verdict={parsed['correct']} reasoning={parsed.get('reasoning','')[:200]}"
    return dict(hit=int(parsed['correct']), log=log)


def Judge_VDR_auxeval(model, item):
    """Judge_VDR variant of ``HLE_style_auxeval`` (VDR paper contains-based
    judge, plain-text output). Returns the same ``{hit, log}`` shape so
    downstream aggregation is drop-in compatible. The default judge of the
    evaluation splits.
    """
    from .utils.judge_vdr import grade_with_judge_vdr

    item = item.iloc[0] if isinstance(item, pd.DataFrame) else item
    question = str(item.get('question', ''))
    answer = _gt_with_alternatives(item)
    response = str(item.get('prediction', ''))

    try:
        parsed = grade_with_judge_vdr(model, question, answer, response)
    except Exception as e:
        return dict(hit=0, log=f'judge_vdr error: {e}')

    log = f"verdict={parsed['correct']} reasoning={parsed.get('reasoning','')[:200]}"
    return dict(hit=int(parsed['correct']), log=log)


def _restore_literal_na_answers(df, source, cols=('answer',)):
    """Restore answer strings that pandas' default NA parsing turned into NaN.

    Some ground-truth answers are literally 'None', 'NA' or 'null' (e.g. three
    InfoSeek test questions), and ``pd.read_csv`` / ``pd.read_excel`` read them
    as NaN by default. ``source`` is either the dataset TSV path or a DataFrame
    that already holds the literal strings; rows are matched on ``index`` and
    only cells that are NaN in ``df`` and non-empty in ``source`` are changed.
    """
    import pandas as pd
    if not any(col in df and df[col].isna().any() for col in cols):
        return df
    if isinstance(source, str):
        if not source or not osp.exists(source):
            return df
        source = pd.read_csv(source, sep='\t', keep_default_na=False, dtype=str)
    if source is None or 'index' not in df or 'index' not in source:
        return df
    for col in cols:
        if col not in df or col not in source:
            continue
        valid = source[col].notna() & (source[col].astype(str) != '')
        lookup = dict(zip(source.loc[valid, 'index'].astype(str),
                          source.loc[valid, col].astype(str)))
        missing = df[col].isna()
        if not missing.any():
            continue
        restored = df.loc[missing, 'index'].astype(str).map(lookup)
        restored = restored[restored.notna()]
        if len(restored):
            df[col] = df[col].astype(object)
            df.loc[restored.index, col] = restored
    return df


class ImageShortQADataset(ImageBaseDataset):
    TYPE = 'Short'

    # Every split is loaded from the OpenVLHarness evaluation-datasets repo
    # (hf_release.HF_SPLITS); the hle URL is the VLMEvalKit fallback.
    DATASET_URL = {
        'hle': 'https://opencompass.openxlab.space/utils/VLMEval/hle.tsv',
        'FVQA_test': '',
        'Infoseek_GDM_test': '',
        'EncyclopedicVQA_test': '',
        # Fixed 300-question subsets of the three test splits above, for a quick
        # estimate of the full-split score.
        'FVQA_test_300': '',
        'Infoseek_GDM_test_300': '',
        'EncyclopedicVQA_test_300': '',
    }

    DATASET_MD5 = {
        'hle': 'a83cbdbea89f27c2aa5b8f34a8894b72',
    }

    def load_data(self, dataset):
        df = load_hf_split(self, dataset) if dataset in HF_SPLITS else super().load_data(dataset)
        return _restore_literal_na_answers(df, getattr(self, 'data_path', ''))

    # It returns a DataFrame
    # Needs an LLM judge (there is no exact-matching scorer); checked before inference.
    REQUIRES_LLM_JUDGE = 'strict'

    def evaluate(self, eval_file, **judge_kwargs):
        require_judge(self.dataset_name, judge_kwargs, allow_exact_matching=False)
        data = load(eval_file)
        _ = self.dataset_name
        assert 'answer' in data and 'prediction' in data
        data = _restore_literal_na_answers(data, getattr(self, 'data', None))
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]

        # Pop MMSearch-specific keys that would otherwise be forwarded to
        # OpenAIWrapper.generate_inner and into the request body, where OpenAI
        # returns 400 on unrecognized fields. Same bug as browsecomp_vl.py
        # and livevqa.py had; fix mirrors those.
        judge_kwargs = dict(judge_kwargs)
        nproc = judge_kwargs.pop('nproc', 4)
        judge_kwargs.pop('enable_llm_judge', None)
        # ``judge_max_tokens`` is consumed centrally in build_judge() (yaml is
        # the single source of truth; no per-evaluator default).

        # Judge selection: env var > judge_kwargs > judge_vdr (the search default);
        # 'hle' and 'judge_rs' (the HLE and MMSearch-R1 protocols) remain selectable.
        judge_name = (
            os.environ.get('MMDR_JUDGE_PROMPT')
            or judge_kwargs.pop('judge_prompt', None)
            or 'judge_vdr'
        ).strip().lower()
        if judge_name not in ('hle', 'judge_rs', 'judge_vdr'):
            raise ValueError(
                f"Unknown judge_prompt {judge_name!r}; expected 'hle', 'judge_rs', or 'judge_vdr'"
            )
        judge_kwargs.pop('judge_prompt', None)

        if judge_name == 'judge_rs':
            storage_stem, tmp_stem, acc_stem = '_judge_judge_rs', '_tmp_judge_rs', '_acc_judge_rs'
        elif judge_name == 'judge_vdr':
            storage_stem, tmp_stem, acc_stem = '_judge_judge_vdr', '_tmp_judge_vdr', '_acc_judge_vdr'
        else:
            storage_stem, tmp_stem, acc_stem = '_judge', '_tmp', '_acc'
        storage = get_intermediate_file_path(eval_file, storage_stem)
        tmp_file = get_intermediate_file_path(eval_file, tmp_stem, 'pkl')
        judge_model = judge_kwargs.get('model', 'gpt-4o-mini')
        guard_cache(storage, data, judge_model, prompt=judge_name, keyed=False)
        # With g_index the tmp file is keyed by question group, not by row.
        guard_cache(tmp_file, data, judge_model, prompt=judge_name, keyed='g_index' not in data)

        if not osp.exists(storage):
            ans_map = {} if not osp.exists(tmp_file) else load(tmp_file)

            model_name = judge_kwargs.pop('model', 'gpt-4o-mini')
            model = build_judge(model=model_name, **judge_kwargs)
            if not model.working():
                raise RuntimeError(
                    f'The LLM judge {model_name} at {judge_kwargs.get("api_base") or "the default endpoint"} '
                    f'did not answer a test request; {self.dataset_name} needs a working judge. '
                    'Check the judge key and api_base.'
                )

            if model is not None:
                if 'g_index' not in data:
                    lines = [data.iloc[i] for i in range(len(data))]
                    indices = [x['index'] for x in lines if x['index'] not in ans_map]
                    lines = [x for x in lines if x['index'] not in ans_map]
                    tups = [(model, line) for line in lines]
                else:
                    main_data = data[[x == y for x, y in zip(data['index'], data['g_index'])]]
                    lines = [data[data['g_index'] == x] for x in main_data['index']]
                    indices = [x.iloc[0]['g_index'] for x in lines if x.iloc[0]['g_index'] not in ans_map]
                    lines = [x for x in lines if x.iloc[0]['g_index'] not in ans_map]
                    tups = [(model, x) for x in lines]
                    data = main_data

                if len(lines):
                    if judge_name == 'judge_rs':
                        auxeval_fn = Judge_RS_auxeval
                    elif judge_name == 'judge_vdr':
                        auxeval_fn = Judge_VDR_auxeval
                    else:
                        auxeval_fn = HLE_style_auxeval
                    res = track_progress_rich(
                        auxeval_fn, tups, nproc=nproc, chunksize=nproc, keys=indices, save=tmp_file)
                    for k, v in zip(indices, res):
                        ans_map[k] = v

            judge_results = [ans_map[x] for x in data['index']]
            data['hit'] = [x['hit'] for x in judge_results]
            data['log'] = [x['log'] for x in judge_results]
            dump(data, storage)

        data = load(storage)
        acc = report_acc(data)

        score_file = get_intermediate_file_path(eval_file, acc_stem, 'csv')
        dump(acc, score_file)
        return acc
