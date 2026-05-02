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


def process_tool_output(payload, memory, tool_metadata=None):
    cleaned = payload
    if isinstance(payload, dict) and isinstance(payload.get("result"), dict):
        cleaned = cp.deepcopy(payload)
        result = cleaned["result"]
        for key in _DROPPED_RESULT_KEYS:
            result.pop(key, None)

    return memory.process_tool_output_default(
        payload=cleaned,
        tool_metadata=tool_metadata,
        processor_info={"mode": PROCESSOR_MODE, "dropped_result_keys": list(_DROPPED_RESULT_KEYS)},
    )
