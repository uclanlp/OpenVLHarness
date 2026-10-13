"""LLM-as-a-judge evaluation for MMBrowseComp.

Adapted from MM-BrowseComp/src/eval.py.
"""

import re
import time

from openai import APIError

JUDGE_API_MAX_RETRIES = 5
JUDGE_API_INITIAL_DELAY = 5
JUDGE_API_BACKOFF_FACTOR = 2


def call_judge_api(client, model_name, messages, max_tokens=5120):
    current_delay = JUDGE_API_INITIAL_DELAY
    for attempt in range(JUDGE_API_MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return completion.choices[0].message.content
        except APIError as e:
            if attempt < JUDGE_API_MAX_RETRIES - 1:
                retry_after_s = 0
                if e.response and e.response.headers:
                    if e.response.headers.get("retry-after-ms"):
                        try:
                            retry_after_s = int(e.response.headers.get("retry-after-ms")) / 1000.0 + 0.1
                        except (ValueError, TypeError):
                            pass
                    elif e.response.headers.get("Retry-After"):
                        try:
                            retry_after_s = int(e.response.headers.get("Retry-After")) + 0.1
                        except (ValueError, TypeError):
                            pass
                actual_sleep = max(current_delay, retry_after_s)
                time.sleep(actual_sleep)
                current_delay *= JUDGE_API_BACKOFF_FACTOR
            else:
                return None
        except Exception:
            if attempt < JUDGE_API_MAX_RETRIES - 1:
                time.sleep(current_delay)
                current_delay *= JUDGE_API_BACKOFF_FACTOR
            else:
                return None
    return None


def construct_judge_prompt(question, generated_answer_to_eval, reference_answer, reference_checklist=None):
    """Construct judge prompt. reference_checklist can be None or empty list."""
    reference_checklist = reference_checklist or []
    generated_answer_to_eval = (generated_answer_to_eval or '')[-25000:]

    intro = (
        "You are an AI evaluator. Your task is to evaluate the quality of an answer. "
        "I will provide you with the user's question, the reference answer (ground truth), "
        "a checklist, and the answer to be evaluated."
    )

    question_section = f"--- USER QUESTION ---\n{question}"
    reference_answer_section = (
        f"\n\n--- REFERENCE ANSWER (Ground Truth) ---\n{reference_answer}\n"
        "This reference answer is considered the correct and ideal response content-wise."
    )

    total_checklist_items = len(reference_checklist)
    if total_checklist_items > 0:
        checklist_items_formatted = "\n".join([f"{i+1}. {item}" for i, item in enumerate(reference_checklist)])
        checklist_section = f"\n\n--- REFERENCE CHECKLIST ---\n{checklist_items_formatted}"
        checklist_evaluation_instruction = (
            f"1. Checklist Score: First, determine how many of the {total_checklist_items} items in the 'REFERENCE CHECKLIST' "
            f"have been correctly and completely addressed by the 'MODEL'S GENERATED ANSWER TO EVALUATE'.\n"
            f"Please remember that for any item in the checklist, the model's generated answer to evaluate must fully comply "
            f"in order for that item to be considered complete.\n"
            f"   State this as 'CHECKLIST_SCORE: [correct_items]/{total_checklist_items}' (e.g., CHECKLIST_SCORE: "
            f"{max(0, total_checklist_items - 1)}/{total_checklist_items}).\n"
            f"2. Checklist Result Vector: Next, please provide a 0-1 vector to indicate whether each checklist item passed. "
            f"Output the vector in the order of the items in the checklist, for example, [1,0,1]. "
            f"'1' means the item is 'fully satisfied,' and '0' means 'not fully satisfied.' "
            f"Output in the format 'CHECKLIST_RESULT: ...' (e.g., CHECKLIST_RESULT: [1,0,1])."
        )
    else:
        checklist_section = "\n\n--- REFERENCE CHECKLIST ---\nN/A (no checklist for this question)"
        checklist_evaluation_instruction = (
            "1. Checklist Score: There is no checklist for this question. "
            "State 'CHECKLIST_SCORE: N/A' and 'CHECKLIST_RESULT: N/A'."
        )

    generated_answer_section = f"\n\n--- MODEL'S GENERATED ANSWER TO EVALUATE ---\n{generated_answer_to_eval}"
    overall_correctness_instruction = (
        "2. Overall Correctness: Next, you need to judge whether the 'MODEL'S GENERATED ANSWER TO EVALUATE' "
        "is consistent with the 'REFERENCE ANSWER (Ground Truth)' in terms of its core content and information.\n"
        "   - Content consistency is key. Differences in formatting or minor wording variations are acceptable "
        "as long as the essential information and meaning conveyed by the generated answer align with the reference answer.\n"
        "   - If the generated answer accurately reflects the information in the reference answer, it should be considered correct.\n"
        "   State your judgment as 'OVERALL_CORRECTNESS: [YES/NO]' (e.g., OVERALL_CORRECTNESS: YES)."
    )

    prompt = f"""{intro}
{question_section}{reference_answer_section}{checklist_section}{generated_answer_section}

--- EVALUATION INSTRUCTIONS ---
Please provide your evaluation strictly in the following format on separate lines:
{checklist_evaluation_instruction}
{overall_correctness_instruction}

Example 1 (Checklist provided, generated answer consistent with reference, some checklist items missed):
CHECKLIST_SCORE: 1/3
CHECKLIST_RESULT: [1,0,1]
OVERALL_CORRECTNESS: YES

Example 2 (Checklist provided, generated answer NOT consistent with reference, even if checklist is met):
CHECKLIST_SCORE: 4/4
CHECKLIST_RESULT: [1,1,1,1]
OVERALL_CORRECTNESS: NO

Provide only these formatted lines (CHECKLIST_SCORE, CHECKLIST_RESULT, OVERALL_CORRECTNESS) as your response.
"""
    return prompt


def parse_judge_response(response_text):
    if not response_text:
        return {
            "checklist_correct_count": None,
            "checklist_total_count": None,
            "overall_correctness": None,
            "error": "Empty response from judge.",
        }

    parsed = {
        "checklist_correct_count": None,
        "checklist_total_count": None,
        "checklist_result_vector": None,
        "overall_correctness": None,
    }

    checklist_match = re.search(r"CHECKLIST_SCORE:\s*(\d+)\s*/\s*(\d+)", response_text, re.IGNORECASE)
    if checklist_match:
        parsed["checklist_correct_count"] = int(checklist_match.group(1))
        parsed["checklist_total_count"] = int(checklist_match.group(2))
    elif re.search(r"CHECKLIST_SCORE:\s*N/A", response_text, re.IGNORECASE):
        parsed["checklist_correct_count"] = "N/A"
        parsed["checklist_total_count"] = "N/A"

    correctness_match = re.search(r"OVERALL_CORRECTNESS:\s*(YES|NO)", response_text, re.IGNORECASE)
    if correctness_match:
        parsed["overall_correctness"] = correctness_match.group(1).upper()

    if parsed["overall_correctness"] is None:
        parsed["error"] = "Failed to parse key fields (e.g., OVERALL_CORRECTNESS) from judge response."

    vector_match = re.search(r"CHECKLIST_RESULT:\s*\[([01,\s]+)\]", response_text, re.IGNORECASE)
    if vector_match:
        try:
            parsed["checklist_result_vector"] = [int(x) for x in re.findall(r"[01]", vector_match.group(1))]
        except ValueError:
            parsed["checklist_result_vector"] = None
    elif re.search(r"CHECKLIST_RESULT:\s*N/A", response_text, re.IGNORECASE):
        parsed["checklist_result_vector"] = "N/A"

    return parsed
