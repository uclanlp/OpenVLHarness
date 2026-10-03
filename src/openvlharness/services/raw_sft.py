"""Qwen ChatML reconstruction adapted from the old _build_raw_sft_from_tokens.

Only provider-returned tokens and images from their exact request are used.
This module has no dependency on the benchmark package or its report writer.
"""
import base64
import io
import re

from PIL import Image


class RawSFTExportError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def _request_images(messages, task_id):
    images = []
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(task_id if task_id is not None else "sample"))
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image_url = part.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else None
            if not isinstance(url, str) or not url.startswith("data:image/") or ";base64," not in url:
                raise RawSFTExportError("image_unavailable", "Raw SFT export requires inline image bytes from the exact request.")
            try:
                data = base64.b64decode(url.split(";base64,", 1)[1], validate=True)
                with Image.open(io.BytesIO(data)) as image:
                    extension = ".jpg" if image.format == "JPEG" else "." + image.format.lower()
                    image.verify()
            except (ValueError, OSError, TypeError) as exc:
                raise RawSFTExportError("invalid_image", "Raw SFT export encountered invalid request image bytes.") from exc
            images.append({"filename": f"{name}_{len(images)}{extension}",
                           "bytes_b64": base64.b64encode(data).decode("ascii")})
    return images


def build_raw_sft_from_tokens(response, messages, tokenizer, task_id):
    """Reconstruct messages and image files, failing explicitly on missing evidence."""
    if tokenizer is None:
        raise RawSFTExportError("tokenizer_unavailable", "Raw SFT export requires the matching Qwen tokenizer.")
    if not isinstance(response, dict) or not isinstance(messages, list):
        raise RawSFTExportError("missing_response", "Raw SFT export has no terminating response and matching request.")
    prompt_ids = response.get("prompt_token_ids")
    choices = response.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    completion_ids = choice.get("token_ids")
    for name, ids in (("prompt_token_ids", prompt_ids), ("choices[0].token_ids", completion_ids)):
        if not isinstance(ids, list) or not ids or any(type(token) is not int or token < 0 for token in ids):
            raise RawSFTExportError("missing_token_ids", f"Raw SFT export requires nonempty integer {name}; enable return_token_ids on the Qwen endpoint.")

    try:
        for token in ("<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>"):
            token_id = tokenizer.convert_tokens_to_ids(token)
            if type(token_id) is not int or token_id < 0 or tokenizer.decode([token_id], skip_special_tokens=False) != token:
                raise ValueError(f"Tokenizer does not recognize {token}")
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        full_ids = list(prompt_ids) + list(completion_ids)
        # vLLM normally includes the stopping token; do not duplicate it.
        if full_ids[-1] != im_end_id:
            full_ids.append(im_end_id)
        decoded = tokenizer.decode(full_ids, skip_special_tokens=False)
    except Exception as exc:
        raise RawSFTExportError("tokenizer_mismatch", f"Cannot decode raw SFT tokens with the matching Qwen tokenizer: {exc}") from exc

    images = _request_images(messages, task_id)
    vision_re = re.compile(r"<\|vision_start\|>.*?<\|vision_end\|>", re.DOTALL)
    blocks = len(vision_re.findall(decoded))
    if blocks != len(images):
        raise RawSFTExportError("image_alignment", f"Raw SFT image alignment failed for task {task_id}: {blocks} vision blocks vs {len(images)} request images.")

    # Qwen's template renders native tool results as user <tool_response> text.
    message_re = re.compile(r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>", re.DOTALL)
    output = []
    position = image_index = 0
    for match in message_re.finditer(decoded):
        if decoded[position:match.start()].strip():
            raise RawSFTExportError("invalid_chatml", "Unrecognized role or content outside Qwen ChatML messages.")
        body = match.group(2)
        content = []
        cursor = 0
        for vision in vision_re.finditer(body):
            if vision.start() > cursor:
                content.append({"type": "text", "text": body[cursor:vision.start()]})
            content.append({"type": "image", "image": f"images/{images[image_index]['filename']}"})
            image_index += 1
            cursor = vision.end()
        if cursor < len(body) or not content:
            content.append({"type": "text", "text": body[cursor:]})
        output.append({"role": match.group(1), "content": content})
        position = match.end()
    if not output or output[-1]["role"] != "assistant" or decoded[position:].strip() or image_index != len(images):
        raise RawSFTExportError("invalid_chatml", "Raw SFT tokens do not form a complete Qwen conversation ending with an assistant response.")
    return {"messages": output, "images": images}
