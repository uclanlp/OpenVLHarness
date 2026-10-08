from ...smp import *
from ...utils import track_progress_rich
from .vqa_eval import process_line, hit_calculate


def build_vqa_extract_prompt(question, prediction, answer):
    """Prompt to extract the final short answer from a VQA prediction."""
    tmpl = (
        'You are an AI assistant who will help me to extract the final answer from a model response to a visual question. '
        'You are provided with a question, the ground truth answer, and the model\'s response. '
        'Your task is to extract ONLY the final short answer from the response, '
        'matching the format of the ground truth (e.g. if the ground truth is a number, extract a number; '
        'if it is a single word, extract a single word). '
        'Do NOT evaluate whether the answer is correct. Just extract it.\n'
        'If the response contains reasoning or explanation, ignore it and only return the final answer.\n'
        'If the response is already a short answer, return it as-is.\n'
        'Example 1:\n'
        'Question: What color is the car?\n'
        'Ground truth: red\n'
        'Response: The image shows a parking lot with several vehicles. Looking at the main car in the center, '
        'it appears to be painted in a bright red color.\n'
        'Extracted answer: red\n'
        'Example 2:\n'
        'Question: How many people are in the image?\n'
        'Ground truth: 3\n'
        'Response: Let me count the people: there are two adults and one child, so 3 people total.\n'
        'Extracted answer: 3\n'
        'Example 3:\n'
        'Question: What is the brand name shown?\n'
        'Ground truth: Nike\n'
        'Response: Nike\n'
        'Extracted answer: Nike\n'
        'Example 4:\n'
        'Question: {}\nGround truth: {}\nResponse: {}\nExtracted answer:'
    )
    return tmpl.format(question, answer, prediction)


def extract_answer_from_vqa_item(model, item):
    """Extract the short answer from a VQA prediction using an LLM judge.

    Mirrors extract_answer_from_item() in multiple_choice.py.
    """
    logger = get_logger('Evaluation')

    if model is None:
        return dict(extracted=item['prediction'], log='No judge model, using raw prediction.')

    prompt = build_vqa_extract_prompt(item['question'], item['prediction'], item['answer'])
    retry = 3

    while retry:
        ans = model.generate(prompt)
        if 'Failed to obtain answer via API' in ans:
            logger.warning('Judge API failed to answer.')
        else:
            extracted = ans.strip()
            if extracted:
                return dict(extracted=extracted, log=ans)
        retry -= 1

    # Fallback to raw prediction if extraction fails
    logger.warning(f'Failed to extract answer after retries, using raw prediction.')
    return dict(extracted=item['prediction'], log='Extraction failed, using raw prediction.')


def eval_vqa_item(model, item, method='vqa_score'):
    """Per-item VQA evaluation: extract answer then apply metric.

    Analogous to eval_vanilla() in multiple_choice.py.
    """
    res = extract_answer_from_vqa_item(model, item)
    extracted = res['extracted']
    log = res['log']

    # Build a line dict with the extracted answer as 'prediction' for process_line
    line = dict(answer=item['answer'], prediction=extracted)
    metric_res = process_line(line, method=method)

    return dict(
        gt=metric_res['gt'],
        pred=metric_res['pred'],
        match=metric_res['match'],
        extracted=extracted,
        log=log,
    )


def vqa_vanilla_eval(model, data, nproc, result_file, dataset_name, method='vqa_score'):
    """Batch VQA evaluation with LLM judge answer extraction.

    Analogous to mcq_vanilla_eval() in multiple_choice.py.
    """
    result = {}
    if osp.exists(result_file):
        result = load(result_file)

    items = []
    for i in range(len(data)):
        item = data.iloc[i]
        idx = item['index'] if 'index' in item else i
        if idx not in result:
            items.append(item)

    tups = [dict(model=model, item=x, method=method) for x in items]
    keys = [x['index'] if 'index' in x else i for i, x in enumerate(items)]

    if len(tups):
        res = track_progress_rich(
            eval_vqa_item, tups, nproc=nproc, chunksize=nproc,
            save=result_file, keys=keys
        )
        result = load(result_file)
        for k, v in zip(keys, res):
            if k not in result:
                result[k] = v

    # Map results back to data
    all_indices = [data.iloc[i]['index'] if 'index' in data.iloc[i] else i for i in range(len(data))]
    data['extracted_answer'] = [result[idx]['extracted'] for idx in all_indices]
    data['log'] = [result[idx]['log'] for idx in all_indices]
    data['eval_match'] = [result[idx]['match'] for idx in all_indices]
    data['eval_gt'] = [result[idx]['gt'] for idx in all_indices]
    data['eval_pred'] = [result[idx]['pred'] for idx in all_indices]
    data['eval_score'] = [np.mean(result[idx]['match']) for idx in all_indices]

    return data
