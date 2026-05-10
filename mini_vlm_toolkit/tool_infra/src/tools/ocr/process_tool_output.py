"""OCR-specific output post-processing for agent-visible messages.

Emits a one-line prose intro followed by the transcribed text from the
PaddleOCR-VL server. Operational metadata (`model_id`, `device`) is kept
in the raw result for debugging but stripped from the model-facing
message.
"""

PROCESSOR_MODE = "custom"


def process_tool_output(payload, prepared, tool_metadata=None):
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        return {
            "processor_info": {
                "mode": PROCESSOR_MODE,
                "status": "fallback",
                "fallback_reason": "non_success_payload",
            }
        }

    result = payload.get("result")
    if not isinstance(result, dict):
        return {
            "processor_info": {
                "mode": PROCESSOR_MODE,
                "status": "fallback",
                "fallback_reason": "result_not_dict",
            }
        }

    text = (result.get("text") or "").strip()
    tool_name = str(payload.get("tool_name") or "OCR_Tool").strip()

    if not text:
        message = f"{tool_name} did not transcribe any readable text."
    else:
        message = (
            f"{tool_name} transcribed the following text from the image:\n\n{text}"
        )

    return {
        "user_message": {"role": "user", "content": message},
        "processor_info": {"mode": PROCESSOR_MODE, "status": "ok"},
    }
