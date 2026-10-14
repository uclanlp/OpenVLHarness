# Adapted from VLMEvalKit (https://github.com/open-compass/VLMEvalKit), Apache License 2.0; modified by the OpenVLHarness authors.

# Single source of truth for the judge's per-call output token cap.
# - Yaml `judge_max_tokens` overrides this for ALL benchmarks at once.
# - When the yaml does not set it, every LLM-judged evaluator falls
#   back to this value. Do NOT introduce per-evaluator defaults.
# - 2048 is sized to comfortably contain the HLE-style judge's
#   ``{extracted_final_answer, reasoning, correct, confidence}`` JSON
#   payload even on long-reasoning tasks. The previous scattered
#   defaults of 128 / 1024 / 5120 caused JSON-truncation bugs that
#   surfaced as ``verdict=error`` on 20%+ of MMSearch / MMSearchPlus /
#   VDRBench / RealXBench samples.
JUDGE_MAX_TOKENS_DEFAULT = 2048


def resolve_judge_max_tokens(judge_kwargs) -> int:
    """Return the per-call max_tokens cap for a judge.

    Yaml ``judge_max_tokens`` (loaded into ``judge_kwargs`` from the
    judge config) wins. If absent, falls back to
    ``JUDGE_MAX_TOKENS_DEFAULT``. Used by evaluators that talk to
    OpenAI directly (custom client) instead of via ``build_judge``.
    """
    if not isinstance(judge_kwargs, dict):
        return JUDGE_MAX_TOKENS_DEFAULT
    v = judge_kwargs.get('judge_max_tokens')
    if v is None:
        return JUDGE_MAX_TOKENS_DEFAULT
    try:
        return int(v)
    except (TypeError, ValueError):
        return JUDGE_MAX_TOKENS_DEFAULT


def build_judge(**kwargs):
    from ..api import OpenAIWrapper
    model = kwargs.pop('model', None)
    kwargs.pop('nproc', None)
    # ``judge_max_tokens`` is the YAML-facing knob for the judge's per-call
    # output cap. Translate to ``max_tokens`` (the OpenAIWrapper kwarg) here
    # so per-evaluator pop logic isn't required, and so a too-small default
    # in one evaluator can't truncate the judge JSON mid-output (the
    # ``parse_hle_judgement`` regex requires both `{` and `}` — a clipped
    # response leaves the parser with reasoning="judge failed to parse"
    # and verdict=error). If an evaluator already popped ``judge_max_tokens``
    # and explicitly set ``max_tokens``, do not override.
    yaml_judge_mt = kwargs.pop('judge_max_tokens', None)
    if 'max_tokens' not in kwargs:
        kwargs['max_tokens'] = (
            int(yaml_judge_mt) if yaml_judge_mt is not None
            else JUDGE_MAX_TOKENS_DEFAULT
        )
    model_map = {
        'gpt-4-turbo': 'gpt-4-1106-preview',
        'gpt-4-0613': 'gpt-4-0613',
        'gpt-4-0125': 'gpt-4-0125-preview',
        'gpt-4-0409': 'gpt-4-turbo-2024-04-09',
        'chatgpt-1106': 'gpt-3.5-turbo-1106',
        'chatgpt-0125': 'gpt-3.5-turbo-0125',
        'gpt-4o': 'gpt-4o-2024-05-13',
        'gpt-4o-0806': 'gpt-4o-2024-08-06',
        'gpt-4o-1120': 'gpt-4o-2024-11-20',
        'gpt-4o-mini': 'gpt-4o-mini-2024-07-18',
        'qwen-7b': 'Qwen/Qwen2.5-7B-Instruct',
        'qwen-72b': 'Qwen/Qwen2.5-72B-Instruct',
        'deepseek': 'deepseek-ai/DeepSeek-V3',
        'llama31-8b': 'meta-llama/Llama-3.1-8B-Instruct',
    }
    model_version = model_map[model] if model in model_map else model

    if model == 'llama31-8b':
        from ..api import HFChatModel
        model = HFChatModel(model_version, **kwargs)
    else:
        model = OpenAIWrapper(model_version, **kwargs)
    return model


DEBUG_MESSAGE = """
To debug the OpenAI API, you can try the following scripts in python:
```python
from evaluation_kit.dataset.api import OpenAIWrapper
model = OpenAIWrapper('gpt-4o', verbose=True)
msgs = [dict(type='text', value='Hello!')]
code, answer, resp = model.generate_inner(msgs)
print(code, answer, resp)
```
You can see the specific error if the API call fails.
"""
