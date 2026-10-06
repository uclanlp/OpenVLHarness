"""Normalize public image/text inputs without importing benchmark datasets."""
import base64
import copy
import io
import mimetypes
import os
from urllib.parse import urlsplit

import requests
from PIL import Image
from .media import encode_image_to_base64


def _is_image_reference(value):
    """Recognize image data URLs or image-like local/HTTP paths by MIME suffix."""
    if value.startswith("data:image/"):
        return True
    remote = value.startswith(("http://", "https://"))
    path = urlsplit(value).path if remote else value
    return (remote or os.path.isfile(value)) and (mimetypes.guess_type(path)[0] or "").startswith("image/")


def _normalize_image(value):
    """Fetch URL inputs once, sharing original image bytes with messages and memory."""
    if not value.startswith(("http://", "https://")):
        return value
    with requests.get(value, timeout=30) as response:
        response.raise_for_status()
        content = response.content
    # Storage URLs may have no extension or an application/octet-stream header.
    with Image.open(io.BytesIO(content)) as image:
        mime = Image.MIME[image.format]
    return f"data:{mime};base64,{base64.b64encode(content).decode('ascii')}"


def normalize_input(message, system_prompt):
    """Copy public inputs into typed items and combine system instructions.

    Args:
        message: Text, one type/value mapping, or a list of strings or mappings.
            Image values may be local paths, data URLs or HTTP(S) URLs.
        system_prompt: Base system instruction text.

    Returns:
        A pair of user input items and the combined system prompt. Remote user
        images are downloaded into data URLs shared by messages and memory.

    Raises:
        ValueError: Inputs are empty, malformed or have unsupported types.
        requests.RequestException: Downloading a remote image fails.
        OSError: Downloaded bytes cannot be decoded as an image.
    """
    if isinstance(message, str):
        message = [{"type": "text", "value": message}]
    elif isinstance(message, dict):
        message = [message]
    elif isinstance(message, list) and all(isinstance(m, str) for m in message):
        message = [{"type": "image" if _is_image_reference(m) else "text", "value": m}
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
            if item["type"] == "image":
                item["value"] = _normalize_image(item["value"])
            question.append(item)
    if extra_system and extra_system not in system_prompt:
        system_prompt += "\n" + extra_system
    return question, system_prompt


def input_messages(question, system_prompt, user_prefix):
    """Build system and user messages from normalized image/text items.

    Args:
        question: User items returned by normalize_input; remote images must
            already be represented as data URLs.
        system_prompt: Combined system instructions.
        user_prefix: Format string applied to the first text part via user_query.

    Returns:
        Two Chat Completions-style messages, with images encoded as JPEG data URLs.
        Text-only inputs are joined into one text part.

    Raises:
        OSError: A local image cannot be opened or encoded.
    """
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
