"""OCR-specific output post-processing for agent-visible messages.

Strips operational/diagnostic fields (`gpu`, `languages`) from the OCR
server's response before the result is rendered into the agent prompt.
The OCR server returns {texts, results, languages, gpu}; only `texts`
(extracted strings) and `results` (per-detection box / text / confidence)
are useful for downstream reasoning.
"""

import copy as cp

PROCESSOR_MODE = "custom"
_DROPPED_RESULT_KEYS = ("gpu", "languages")


def process_tool_output(payload, prepared, tool_metadata=None):
    """Return None to delegate to the orchestrator's default tool-return
    message construction. Tools needing a custom message build a dict
    ``{"user_message": ..., "processor_info": ...}``.
    """
    return None
