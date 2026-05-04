from mini_vlm_toolkit import *
from ..smp import *
from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from .utils.hle_judge import HLE_JUDGE_PROMPT, parse_hle_judgement
from .utils.multiple_choice import report_acc, eval_vanilla, eval_circular_group
from .utils.shortqa import ShortQA_prompt
from .utils.webwatcher_subset import apply_webwatcher_subset
from ..utils import track_progress_rich
from ..smp.file import get_intermediate_file_path


def ShortQA_auxeval(model, line):
    def proc_str(s):
        chs = set(s)
        chs = [x for x in chs if x not in string.ascii_letters + ': ']
        for ch in chs:
            s = s.replace(ch, ' ')
        return s

    def extraction(resp):
        correct, reason = None, None
        correct_st, correct_ed = '[Begin Correctness]', '[End Correctness]'
        reason_st, reason_ed = '[Begin Reason]', '[End Reason]'
        if correct_st in resp and correct_ed in resp:
            correct = resp.split(correct_st)[1].split(correct_ed)[0].strip().lower()
            if ('yes' in correct) ^ ('no' in correct):
                correct = 1 if 'yes' in correct else 0
                if reason_st in resp and reason_ed in resp:
                    reason = resp.split(reason_st)[1].split(reason_ed)[0].strip()
                return correct, reason
            else:
                return None, None
        else:
            return None, None

    prompt = ShortQA_prompt(line)
    retry = 3
    for i in range(retry):
        output = model.generate(prompt, temperature=0.5 * i)
        ans = extraction(output)
        # print(output, ans)
        if ans[0] in [0, 1]:
            return dict(hit=ans[0], log=ans[1])

    return dict(hit=0, log='Fail to Judge')


def Comprehensive_auxeval(model, data):
    def valid(record, key_name):
        return key_name in record and (not pd.isna(record[key_name])) and record[key_name] != ''

    if isinstance(data, pd.DataFrame) and len(data) > 1:
        # Should Adopt CircularEval
        assert valid(data.iloc[0], 'A')
        data['GT'] = data['answer']
        return eval_circular_group(model, data)
    else:
        item = data.iloc[0] if isinstance(data, pd.DataFrame) else data
        if valid(item, 'A') and len(item['answer']) == 1:
            item['GT'] = item['answer']
            return eval_vanilla(model, item)
        else:
            return ShortQA_auxeval(model, item)


def HLE_style_auxeval(model, item):
    """HLE-style binary judge for short-answer datasets (default for hle).

    Mirrors ShortQA_auxeval's interface (returns ``{hit: 0|1, log: str}``)
    but uses the shared HLE judge prompt + JSON-extracting parser, so
    the resulting accuracy is directly comparable to WebWatcher / BCVL /
    other HLE-protocol scorers.
    """
    item = item.iloc[0] if isinstance(item, pd.DataFrame) else item
    question = str(item.get('question', ''))
    answer = item.get('answer', '')
    if isinstance(answer, str) and answer.strip().startswith('[') and answer.strip().endswith(']'):
        try:
            answer = eval(answer)
        except Exception:
            answer = [answer]
    if not isinstance(answer, list):
        answer = [answer]
    correct_answer = str(answer[0]) if answer else ''
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
    drop-in compatible. Opt-in via env ``MVT_JUDGE_PROMPT=judge_rs``.
    """
    from .utils.judge_rs import grade_with_judge_rs

    item = item.iloc[0] if isinstance(item, pd.DataFrame) else item
    question = str(item.get('question', ''))
    answer = item.get('answer', '')
    if isinstance(answer, str) and answer.strip().startswith('[') and answer.strip().endswith(']'):
        try:
            answer = eval(answer)
        except Exception:
            answer = [answer]
    if not isinstance(answer, list):
        answer = [answer]
    response = str(item.get('prediction', ''))

    try:
        parsed = grade_with_judge_rs(model, question, answer, response)
    except Exception as e:
        return dict(hit=0, log=f'judge_rs error: {e}')

    log = f"verdict={parsed['correct']} reasoning={parsed.get('reasoning','')[:200]}"
    return dict(hit=int(parsed['correct']), log=log)


class ImageShortQADataset(ImageBaseDataset):
    TYPE = 'Short'

    DATASET_URL = {
        'LiveMMBench_Infographic': '',
        'LiveMMBench_Perception': '',
        'LiveMMBench_Reasoning': '',
        'LiveMMBench_Reasoning_circular': '',
        'hle':'https://opencompass.openxlab.space/utils/VLMEval/hle.tsv',
        # SFT distillation training source — TSV produced by
        # `sft/data/download_train_datasets.py --action to_tsv --dataset infoseek`.
        # Empty URL = no auto-download; local TSV is authoritative.
        'Infoseek_GDM_train': '',
        # FVQA training split (== MMSearch-R1 fvqa_train). TSV produced by
        # `sft/data/download_train_datasets.py --action to_tsv --dataset fvqa`.
        'FVQA_train': '',
    }

    DATASET_MD5 = {
        'hle': 'a83cbdbea89f27c2aa5b8f34a8894b72',
    }

    def load_data(self, dataset):
        # Default ImageBaseDataset loader; then auto-filter to the
        # WebWatcher subset where applicable (currently only ``hle``).
        # No-op for datasets without a subset entry.
        # Set MVT_DISABLE_WEBWATCHER_SUBSET=1 to bypass.
        df = super().load_data(dataset)
        return apply_webwatcher_subset(df, dataset)

    # It returns a DataFrame
    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        _ = self.dataset_name
        # Subset-aware re-eval: if the dataset has a WebWatcher subset
        # entry, filter the predictions to ONLY those indices before
        # judging. Without this, `--mode eval` on a full prediction xlsx
        # would re-judge every row (waste). Applies to ``hle``; no-op for
        # datasets without a subset entry.
        data = apply_webwatcher_subset(data, self.dataset_name)
        assert 'answer' in data and 'prediction' in data
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]

        # Pop MMSearch-specific keys that would otherwise be forwarded to
        # OpenAIWrapper.generate_inner and into the request body, where OpenAI
        # returns 400 on unrecognized fields. Same bug as browsecomp_vl.py
        # and livevqa.py had; fix mirrors those.
        judge_kwargs = dict(judge_kwargs)
        nproc = judge_kwargs.pop('nproc', 4)
        judge_kwargs.pop('enable_llm_judge', None)
        max_tokens = judge_kwargs.pop('judge_max_tokens', 1024)

        # Judge selection: env var > judge_kwargs > 'hle' default
        judge_name = (
            os.environ.get('MVT_JUDGE_PROMPT')
            or judge_kwargs.pop('judge_prompt', None)
            or 'hle'
        ).strip().lower()
        if judge_name not in ('hle', 'judge_rs'):
            raise ValueError(f"Unknown judge_prompt {judge_name!r}; expected 'hle' or 'judge_rs'")
        judge_kwargs.pop('judge_prompt', None)

        storage_stem = '_judge_judge_rs' if judge_name == 'judge_rs' else '_judge'
        tmp_stem = '_tmp_judge_rs' if judge_name == 'judge_rs' else '_tmp'
        acc_stem = '_acc_judge_rs' if judge_name == 'judge_rs' else '_acc'
        storage = get_intermediate_file_path(eval_file, storage_stem)
        tmp_file = get_intermediate_file_path(eval_file, tmp_stem, 'pkl')

        if not osp.exists(storage):
            ans_map = {} if not osp.exists(tmp_file) else load(tmp_file)

            model = judge_kwargs.pop('model', 'gpt-4o-mini')
            if model == 'exact_matching':
                model = None
            else:
                model = build_judge(model=model, max_tokens=max_tokens, **judge_kwargs)
                if not model.working():
                    warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                    warnings.warn(DEBUG_MESSAGE)
                    model = None

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
                    # For ``hle``: use HLE-style binary judge (matches
                    # WebWatcher / BCVL / paper protocol). When
                    # ``MVT_JUDGE_PROMPT=judge_rs`` is set, swap to the
                    # multimodal-search-r1 Judge_RS prompt with role-split
                    # system+user. All other ImageShortQA datasets
                    # (LiveMMBench_*, PathVQA, Infoseek_GDM_train,
                    # FVQA_train, ...) keep the existing
                    # Comprehensive_auxeval path which handles MCQ vs
                    # ShortQA dispatch (Judge_RS only wired for hle).
                    if self.dataset_name == 'hle':
                        auxeval_fn = (
                            Judge_RS_auxeval if judge_name == 'judge_rs'
                            else HLE_style_auxeval
                        )
                    else:
                        auxeval_fn = Comprehensive_auxeval
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


class PathVQA_VAL(ImageShortQADataset):
    DATASET_URL = {
        'PathVQA_VAL': 'https://huggingface.co/datasets/Pfei111/PathVQA/resolve/main/PathVQA_VAL.tsv',
    }

    DATASET_MD5 = {
        'PathVQA_VAL': None,
    }


class PathVQA_TEST(ImageShortQADataset):
    DATASET_URL = {
        'PathVQA_TEST': 'https://huggingface.co/datasets/Pfei111/PathVQA/resolve/main/PathVQA_TEST.tsv',
    }

    DATASET_MD5 = {
        'PathVQA_TEST': None,
    }
