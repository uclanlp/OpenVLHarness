import base64
import copy
import html
import io
import json
import os

from PIL import Image

from ..smp import dump


def _is_structured_record(value):
    return isinstance(value, dict) and "prediction" in value and "extra_records" in value


def _looks_like_base64(text):
    if not isinstance(text, str):
        return False
    if len(text) < 200:
        return False
    for ch in text[:200]:
        if not (ch.isalnum() or ch in "+/=_-"):
            return False
    return True


ImageLike = Image.Image | str


def _to_base64(img: ImageLike) -> str:
    if isinstance(img, str):
        return img
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _ensure_image_url(value):
    if not value:
        return ""
    if isinstance(value, str) and value.startswith("data:image/"):
        return value
    if isinstance(value, str) and os.path.exists(value):
        try:
            img = Image.open(value)
            b64 = _to_base64(img)
            return f"data:image/png;base64,{b64}"
        except Exception:
            print(f"Warning: Failed to load image from path '{value}'. Using the path as-is.")
            return value
    if isinstance(value, str) and _looks_like_base64(value):
        return f"data:image/png;base64,{value}"
    return value


def _scrub_image_urls(obj, key_hint=None):
    if isinstance(obj, str):
        if obj.startswith("data:image/"):
            return "<image_url>"
        if key_hint and ("image" in key_hint.lower() or "base64" in key_hint.lower()):
            if len(obj) > 100:
                return "<image_base64>"
        if _looks_like_base64(obj):
            return "<image_base64>"
        return obj
    if isinstance(obj, dict):
        return {k: _scrub_image_urls(v, key_hint=str(k)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub_image_urls(v, key_hint=key_hint) for v in obj]
    return obj


def build_trajectory_records(
    indices,
    data_all,
    *,
    dataset_name,
    model_name,
    agent_class,
    scrub_images=True,
    strip_assistant_content=False,
):
    records = []
    for idx in indices:
        entry = data_all.get(idx)
        if _is_structured_record(entry):
            prediction = entry.get("prediction")
            extra = entry.get("extra_records") or {}
            extra = copy.deepcopy(extra) if isinstance(extra, dict) else {"extra_records": extra}
        else:
            prediction = None if entry is None else str(entry)
            extra = {}

        trajectory = extra.pop("trajectory", None)
        if scrub_images:
            trajectory = _scrub_image_urls(trajectory)
        record = {
            "task_id": idx,
            "dataset": dataset_name,
            "model_name": model_name,
            "agent_class": agent_class,
            "prediction": prediction,
        }
        if trajectory is not None:
            if strip_assistant_content:
                trajectory = _strip_assistant_content(trajectory)
            record["trajectory"] = trajectory
        if extra:
            record["extra_records"] = extra
        records.append(record)
    return records


def write_trajectory_jsonl(records, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dump(records, output_path)


def select_records_for_report_limit(records, limit):
    if limit == 0:
        return []
    if limit < 0:
        limit = len(records)
    sorted_records = sorted(
        records,
        key=lambda r: len((r.get("trajectory") or {}).get("chat_history") or []),
        reverse=True,
    )
    return sorted_records[: min(limit, len(sorted_records))]


def write_input_snapshot_json(indices, data_all, output_path):
    """Write the first available per-sample input snapshot for run-level debugging."""
    for idx in indices:
        entry = data_all.get(idx)
        if not _is_structured_record(entry):
            continue
        extra = entry.get("extra_records")
        if not isinstance(extra, dict):
            continue
        trajectory = extra.get("trajectory")
        if not isinstance(trajectory, dict):
            continue
        snapshot = trajectory.get("input_snapshot")
        if snapshot is None:
            snapshot = trajectory.get("_input_snapshot")
        if snapshot is None:
            continue
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dump(copy.deepcopy(snapshot), output_path)
        return True
    return False


def _strip_assistant_content(trajectory):
    if not isinstance(trajectory, dict):
        return trajectory
    chat_history = trajectory.get("chat_history")
    if isinstance(chat_history, list):
        cleaned = []
        for turn in chat_history:
            if not isinstance(turn, dict):
                cleaned.append(turn)
                continue
            turn_copy = turn.copy()
            assistant = turn_copy.get("assistant")
            if isinstance(assistant, dict) and "content" in assistant:
                assistant = assistant.copy()
                assistant.pop("content", None)
                turn_copy["assistant"] = assistant
            cleaned.append(turn_copy)
        trajectory = trajectory.copy()
        trajectory["chat_history"] = cleaned
    return trajectory


def _assistant_message_matches(candidate, final_message):
    if not isinstance(candidate, dict):
        return False
    if candidate.get("role") != "assistant":
        return False
    if candidate.get("content") != final_message.get("content"):
        return False
    if "tool_calls" in final_message:
        return candidate.get("tool_calls") == final_message.get("tool_calls")
    return True


def build_complete_messages_from_trajectory(trajectory):
    """Return final-round input chain plus final assistant output when available."""
    if not isinstance(trajectory, dict):
        return None
    messages = trajectory.get("messages")
    if not isinstance(messages, list):
        return None

    complete_messages = copy.deepcopy(messages)
    chat_history = trajectory.get("chat_history")
    if not isinstance(chat_history, list) or not chat_history:
        return complete_messages

    last_turn = chat_history[-1]
    if not isinstance(last_turn, dict):
        return complete_messages
    assistant = last_turn.get("assistant")
    if not isinstance(assistant, dict):
        return complete_messages

    final_content = assistant.get("content")
    if final_content is None:
        return complete_messages
    final_message = {"role": "assistant", "content": final_content}
    tool_calls = assistant.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        final_message["tool_calls"] = copy.deepcopy(tool_calls)

    if complete_messages and _assistant_message_matches(complete_messages[-1], final_message):
        return complete_messages
    complete_messages.append(final_message)
    return complete_messages


def _escape(text):
    return html.escape(text or "")


def _shorten(text, max_chars):
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... (truncated)"


def _render_text_block(label, text, max_chars):
    if text is None:
        return ""
    safe_text = _escape(text)
    if len(text) <= max_chars:
        return f"<div class='block'><div class='label'>{_escape(label)}</div><pre>{safe_text}</pre></div>"
    short = _escape(_shorten(text, max_chars))
    return (
        "<div class='block'>"
        f"<div class='label'>{_escape(label)}</div>"
        f"<pre>{short}</pre>"
        "<details><summary>Show full</summary>"
        f"<pre>{safe_text}</pre>"
        "</details></div>"
    )


def _extract_image_urls(obj, urls, key_hint=None, allow_paths=False):
    if isinstance(obj, str):
        if obj.startswith("data:image/"):
            urls.append(obj)
            return
        if key_hint and ("image" in key_hint.lower() or "base64" in key_hint.lower()):
            if _looks_like_base64(obj):
                urls.append(_ensure_image_url(obj))
                return
            if allow_paths and os.path.exists(obj):
                urls.append(_ensure_image_url(obj))
                return
        if _looks_like_base64(obj):
            urls.append(_ensure_image_url(obj))
        if allow_paths and os.path.exists(obj):
            urls.append(_ensure_image_url(obj))
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            _extract_image_urls(value, urls, key_hint=str(key), allow_paths=allow_paths)
        return
    if isinstance(obj, list):
        for item in obj:
            _extract_image_urls(item, urls, key_hint=key_hint, allow_paths=allow_paths)


def _render_images(urls):
    if not urls:
        return ""
    img_tags = []
    for url in urls:
        img_tags.append(f"<img src='{_escape(_ensure_image_url(url))}' />")
    return "<div class='images'>" + "".join(img_tags) + "</div>"


def _normalize_messages_for_render(messages):
    if not messages:
        return []
    if isinstance(messages, dict):
        messages = [messages]
    if not isinstance(messages, list):
        return []
    if messages and isinstance(messages[0], dict) and "role" not in messages[0] and "type" in messages[0]:
        content = []
        for item in messages:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                content.append({"type": "text", "text": item.get("value", "")})
            elif item.get("type") == "image":
                content.append({"type": "image", "value": item.get("value")})
            else:
                content.append({"type": "text", "text": json.dumps(item, ensure_ascii=False)})
        return [{"role": "user", "content": content}]
    return messages


def _render_messages(messages, max_chars):
    messages = _normalize_messages_for_render(messages)
    if not messages:
        return ""
    blocks = []
    for msg in messages:
        role = msg.get("role", "message")
        blocks.append(f"<div class='message-role'>{_escape(role)}</div>")
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    blocks.append(_render_text_block("text", part.get("text", ""), max_chars))
                elif isinstance(part, dict) and part.get("type") in ("image_url", "image"):
                    url = part.get("image_url", {}).get("url", "")
                    if not url and part.get("type") == "image":
                        url = part.get("value", "")
                    blocks.append(_render_images([_ensure_image_url(url)]))
                else:
                    blocks.append(_render_text_block("content", json.dumps(part, ensure_ascii=False), max_chars))
        else:
            blocks.append(_render_text_block("content", str(content), max_chars))
        blocks.append("<div class='divider'></div>")
    return "<div class='messages'>" + "".join(blocks) + "</div>"


def _render_turns(chat_history, max_chars):
    if not chat_history:
        return ""
    sections = []
    for turn in chat_history:
        turn_id = turn.get("turn")
        assistant = turn.get("assistant", {})
        tool_calls = assistant.get("tool_calls", [])
        tool_responses = turn.get("tool_responses", [])
        sections.append(f"<div class='turn'><div class='turn-title'>Turn {turn_id}</div>")
        sections.append(_render_text_block("assistant.content", assistant.get("content", ""), max_chars))
        if tool_calls:
            scrubbed_calls = _scrub_image_urls(tool_calls)
            try:
                calls_text = json.dumps(scrubbed_calls, ensure_ascii=False, indent=2)
            except Exception:
                calls_text = str(scrubbed_calls)
            sections.append(_render_text_block("tool_calls", calls_text, max_chars))
        if tool_responses:
            scrubbed_resp = _scrub_image_urls(tool_responses)
            try:
                resp_text = json.dumps(scrubbed_resp, ensure_ascii=False, indent=2)
            except Exception:
                resp_text = str(scrubbed_resp)
            sections.append(_render_text_block("tool_responses", resp_text, max_chars))
            urls = []
            # Avoid rendering mutable local file paths from tool payloads.
            # Prefer embedded image payloads (data URLs/base64) to keep per-task determinism.
            _extract_image_urls(tool_responses, urls, allow_paths=False)
            sections.append(_render_images(urls))
        if "final_answer" in turn:
            sections.append(_render_text_block("final_answer", str(turn.get("final_answer")), max_chars))
        sections.append("</div>")
    return "<div class='turns'>" + "".join(sections) + "</div>"


def render_html_report(record, max_chars=2000):
    trajectory = record.get("trajectory") or {}
    input_messages = trajectory.get("input_messages")
    chat_history = trajectory.get("chat_history")
    final_answer = trajectory.get("final_answer")
    messages = trajectory.get("messages")

    header = (
        "<div class='header'>"
        f"<div class='title'>Task {record.get('task_id')}</div>"
        f"<div class='meta'>Dataset: {_escape(str(record.get('dataset')))} | "
        f"Model: {_escape(str(record.get('model_name')))} | "
        f"Agent: {_escape(str(record.get('agent_class')))}</div>"
        "</div>"
    )

    body = []
    if input_messages:
        body.append("<h2>Input Messages</h2>")
        body.append(_render_messages(input_messages, max_chars))
    if chat_history:
        body.append("<h2>Tool Loop</h2>")
        body.append(_render_turns(chat_history, max_chars))
    if final_answer is not None:
        body.append("<h2>Final Answer</h2>")
        body.append(_render_text_block("final_answer", str(final_answer), max_chars))
    if messages:
        body.append("<h2>Final Messages</h2>")
        body.append(_render_messages(messages, max_chars))

    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Trajectory Report</title>
  <style>
    body {{
      font-family: "IBM Plex Sans", "Source Sans 3", sans-serif;
      background: #f6f4f0;
      color: #1f1f1f;
      margin: 0;
      padding: 24px;
    }}
    .header {{
      background: #1b1a17;
      color: #f6f4f0;
      padding: 20px 24px;
      border-radius: 12px;
      margin-bottom: 20px;
    }}
    .title {{
      font-size: 22px;
      font-weight: 700;
      margin-bottom: 6px;
    }}
    .meta {{
      font-size: 14px;
      opacity: 0.8;
    }}
    h2 {{
      font-size: 18px;
      margin: 20px 0 8px;
    }}
    .block {{
      background: #ffffff;
      border-radius: 10px;
      padding: 12px 14px;
      margin-bottom: 12px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.08);
    }}
    .label {{
      font-weight: 600;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: #5a544f;
      margin-bottom: 6px;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      margin: 0;
      font-size: 13px;
      line-height: 1.4;
      font-family: "IBM Plex Mono", "Source Code Pro", monospace;
    }}
    details summary {{
      cursor: pointer;
      margin: 8px 0;
    }}
    .messages {{
      background: #ffffff;
      border-radius: 10px;
      padding: 12px 14px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.08);
    }}
    .message-role {{
      font-weight: 700;
      font-size: 12px;
      text-transform: uppercase;
      color: #a14d27;
      margin-top: 12px;
    }}
    .divider {{
      height: 1px;
      background: #e6e1da;
      margin: 10px 0;
    }}
    .turn {{
      background: #fff8f2;
      border-radius: 12px;
      padding: 12px 14px;
      margin-bottom: 14px;
      border: 1px solid #f0e0d4;
    }}
    .turn-title {{
      font-weight: 700;
      margin-bottom: 8px;
    }}
    .images {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-top: 10px;
    }}
    .images img {{
      width: 100%;
      max-height: 240px;
      object-fit: contain;
      border-radius: 8px;
      border: 1px solid #e0d7cc;
    }}
  </style>
</head>
<body>
  {header}
  {''.join(body)}
</body>
</html>
"""


def write_html_reports(records, out_dir, limit, max_chars=2000):
    if limit == 0:
        return
    os.makedirs(out_dir, exist_ok=True)
    for record in select_records_for_report_limit(records, limit):
        task_id = record.get("task_id")
        filename = f"{task_id}.html"
        output_path = os.path.join(out_dir, filename)
        html_text = render_html_report(record, max_chars=max_chars)
        with open(output_path, "w", encoding="utf8") as f:
            f.write(html_text)


def write_full_trajectory_jsons(records, out_dir, limit):
    """Write per-sample full trajectory JSONs aligned to the HTML selection policy."""
    if limit == 0:
        return
    os.makedirs(out_dir, exist_ok=True)
    for record in select_records_for_report_limit(records, limit):
        trajectory = record.get("trajectory")
        complete_messages = build_complete_messages_from_trajectory(trajectory)
        payload = {
            "task_id": copy.deepcopy(record.get("task_id")),
            "dataset": copy.deepcopy(record.get("dataset")),
            "model_name": copy.deepcopy(record.get("model_name")),
            "agent_class": copy.deepcopy(record.get("agent_class")),
            "prediction": copy.deepcopy(record.get("prediction")),
        }
        if complete_messages is not None:
            payload["messages"] = complete_messages
        elif isinstance(trajectory, dict) and isinstance(trajectory.get("messages"), list):
            payload["messages"] = copy.deepcopy(trajectory["messages"])

        task_id = payload.get("task_id")
        output_path = os.path.join(out_dir, f"{task_id}.json")
        dump(payload, output_path)
