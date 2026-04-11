from .....smp.file import get_intermediate_file_path

_NAME_STR_MAP = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4'}


def _judge_tag_from_mode_and_model(judge_mode: str | None, judge_model: str | None) -> str:
    """
    Map (judge_mode, judge_model) to a judge_tag string used in filenames.

    Follows the same naming convention as MCQ / VQA benchmarks:
        - 'exact_matching'          (rule-based)
        - 'gpt-4o' / 'openai'      (LLM-based, no 'llm_' prefix)
    """
    if judge_mode == 'llm':
        name = _NAME_STR_MAP.get(judge_model, judge_model) if judge_model else None
        if name and "/" in name:
            return name.split("/")[-1]
        return name if name else 'exact_matching'
    return 'exact_matching'


def get_judge_tag_from_score_fn(score_fn) -> str:
    """
    Infer judge_tag from attributes attached to score_fn.

    This relies on _build_score_fn setting:
        score_fn.judge_mode
        score_fn.judge_model
    """
    judge_mode = getattr(score_fn, 'judge_mode', 'rule')
    judge_model = getattr(score_fn, 'judge_model', None)
    return _judge_tag_from_mode_and_model(judge_mode, judge_model)


def build_eval_paths(eval_file: str, judge_tag: str):
    """
    Build unified evaluation-related file paths from eval_file and judge_tag.

    Follows MCQ / VQA naming convention. It returns:
        - result_file: *_{judge_tag}_result.pkl
        - xlsx_path  : *_{judge_tag}_result.xlsx
        - acc_path   : *_acc.{EVAL_FORMAT or default csv}
    """
    result_file = get_intermediate_file_path(
        eval_file,
        suffix=f'_{judge_tag}_result',
        target_format='pkl'
    )

    xlsx_path = get_intermediate_file_path(
        eval_file,
        suffix=f'_{judge_tag}_result',
        target_format='xlsx'
    )

    acc_path = get_intermediate_file_path(
        eval_file,
        suffix='_acc'
    )

    return result_file, xlsx_path, acc_path
