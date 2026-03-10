import re


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
        if isinstance(pred, float) and (pred != pred):  # NaN check
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
