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


from ._base import BaseCapability

PUBLIC_SCHEMA = {'type': 'function',
 'function': {'name': 'OCR_Tool',
              'description': 'Extract readable text from an image. Returns the transcribed text as '
                             'a single string. Use this tool when the task is about words, '
                             'numbers, or printed labels in the image. Do not use it for general '
                             'object localization or segmentation.',
              'parameters': {'type': 'object',
                             'properties': {'image_id': {'type': 'string',
                                                         'description': 'A runtime image ID from '
                                                                        'Environment Update '
                                                                        'messages, e.g. '
                                                                        'input_image_1 or '
                                                                        'tool_generated_image_1.'}},
                             'required': []}}}

from typing import TypedDict


class Output(TypedDict):
    text: str
    model_id: str | None


class Capability(BaseCapability):
    name = 'OCR_Tool'
    backends = {'ocr': 'OCR_Tool'}
    schema = PUBLIC_SCHEMA
    process_output = staticmethod(process_tool_output)

    def store_result(self, call, result, memory):
        if not isinstance(result, dict) or not isinstance(result.get("text"), str):
            return []
        return [memory.store_binding(prefix="ocr", query="", value={"text": result["text"]},
            source_tool=self.name, source_args=call.arguments,
            descriptions={"text": "transcribed text"})]

    def normalize_result(self, call, result, prepared, binding_refs) -> Output:
        result = result if isinstance(result, dict) else {}
        return {"text": result.get("text"), "model_id": result.get("model_id")}
