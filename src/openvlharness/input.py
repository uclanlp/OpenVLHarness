"""Normalize public image/text inputs without importing benchmark datasets."""
import base64
import copy
import io
import mimetypes
import os
from PIL import Image
from .media import encode_image_to_base64


def normalize_input(message, system_prompt):
    if isinstance(message, str):
        message = [{"type": "text", "value": message}]
    elif isinstance(message, dict):
        message = [message]
    elif isinstance(message, list) and all(isinstance(m, str) for m in message):
        message = [{"type": "image" if os.path.isfile(m) and
                    (mimetypes.guess_type(m)[0] or "").startswith("image/") else "text", "value": m}
                   for m in message]
    message = copy.deepcopy(message)
    if not isinstance(message, list) or not message:
        raise ValueError("message must contain image/text inputs")
    question = []
    extra_system = ""
    for item in message:
        if not isinstance(item, dict) or item.get("type") not in {"text", "image"} or not isinstance(item.get("value"), str):
            raise ValueError("Each input must have type=image|text and a string value")
        if item.pop("role", "user") == "system":
            extra_system += item["value"] + "\n"
        else:
            question.append(item)
    if extra_system and extra_system not in system_prompt:
        system_prompt += "\n" + extra_system
    return question, system_prompt


def input_messages(question, system_prompt, user_prefix):
    parts = []
    if any(item["type"] == "image" for item in question):
        for item in question:
            if item["type"] == "text":
                parts.append({"type": "text", "text": item["value"]})
            else:
                value = item["value"]
                source = io.BytesIO(base64.b64decode(value.split(",", 1)[1])) if value.startswith("data:image/") else value
                with Image.open(source) as img:
                    encoded = encode_image_to_base64(img)
                options = {key: v for key, v in item.items() if key not in {"type", "value"}}
                parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}", **options}})
    else:
        parts = [{"type": "text", "text": "\n".join(item["value"] for item in question)}]
    for part in parts:
        if part["type"] == "text":
            part["text"] = user_prefix.format(user_query=part["text"])
            break
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": parts}]
