import math
import re


_FAILURE_MARKERS = (
    'Failed to obtain answer',
    'Unable to get response from the server',
    'Failed with all possible conversation turns',
)


def parse_count(pred):
    """Read a predicted count with the strict rule used to score counting benchmarks.

    Takes the last number in the response, after keeping only the last
    ``\\boxed{...}`` content and removing thousands separators; a whole-number
    decimal such as ``5.0`` counts as 5. Returns None, scored as wrong, for empty
    responses, agent failure messages, number words and non-integer numbers.
    No language model is involved.
    """
    if pred is None:
        return None
    if isinstance(pred, float):
        return int(pred) if math.isfinite(pred) and pred.is_integer() else None
    if isinstance(pred, int):
        return pred
    text = str(pred).strip()
    if not text or text.lower() in ('nan', 'none') or any(m in text for m in _FAILURE_MARKERS):
        return None
    boxed = re.findall(r'\\boxed\{([^{}]*)\}', text)
    if boxed:
        text = boxed[-1]
    text = re.sub(r'(?<=\d),(?=\d{3}(?!\d))', '', text)
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    if not numbers:
        return None
    value = float(numbers[-1])
    return int(value) if value.is_integer() else None


def extract_integer_from_prediction(pred):
    """Extract the first integer from a model prediction string.

    Handles:
      - Direct int/float values
      - Regex extraction of integers from text
      - Common number words (zero through twenty)

    No upper-bound cap — FSC-147 counts can reach ~3700.
    Returns None if extraction fails.
    """
    if pred is None:
        return None

    # Handle numeric types directly
    if isinstance(pred, (int, float)):
        if isinstance(pred, float) and not math.isfinite(pred):  # NaN or infinity
            return None
        return int(pred)

    pred = str(pred).strip()

    # Try direct parse
    try:
        return int(pred)
    except ValueError:
        pass
    try:
        return int(float(pred))
    except ValueError:
        pass
    except OverflowError:  # 'inf', 'Infinity', '1e999'
        return None

    # Regex: find all integers in the string, return the first one
    matches = re.findall(r'\b(\d+)\b', pred)
    if matches:
        return int(matches[0])

    # Word-to-number fallback for common words
    number_words = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
        'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
        'eighteen': 18, 'nineteen': 19, 'twenty': 20,
    }
    pred_lower = pred.lower()
    for word, val in number_words.items():
        if re.search(r'\b' + word + r'\b', pred_lower):
            return val

    return None
