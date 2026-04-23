import argparse
import base64
import html
import json
import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_HTML_HEIGHT = 900


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--trajectory-dir",
        type=str,
        # default="/Users/<user>/Downloads/agent_v1_0215_judge_v1_0215_20260401_004439_default/RefSpatial/full_trajectories",
        default="/Users/<user>/Downloads/agent_spatial/MindCubeBench_tiny_raw_qa/full_trajectories",
        help="Path to a full_trajectories directory containing per-task JSON files.",
    )
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


def _numeric_sort_key(path: Path) -> tuple[int, Any]:
    stem = path.stem
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


@st.cache_data(show_spinner=False)
def list_trajectory_files(trajectory_dir: str) -> list[str]:
    base = Path(trajectory_dir).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"Trajectory directory not found: {base}")
    if not base.is_dir():
        raise NotADirectoryError(f"Trajectory path is not a directory: {base}")

    files = sorted((path for path in base.iterdir() if path.suffix == ".json"), key=_numeric_sort_key)
    return [str(path) for path in files]


@st.cache_data(show_spinner=False)
def load_record(json_path: str) -> dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {json_path}")
    return payload


@st.cache_data(show_spinner=False)
def extract_tools_used(json_path: str) -> list[str]:
    """Return sorted unique tool names used in a trajectory."""
    record = load_record(json_path)
    tools = set()
    for msg in record.get("messages", []):
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            name = fn.get("name", "")
            if name:
                tools.add(name)
    return sorted(tools)


@st.cache_data(show_spinner=False)
def build_tool_index(trajectory_dir: str) -> tuple[list[str], dict[str, list[str]]]:
    """Scan all trajectories and return (all_tools, tool_to_files)."""
    files = list_trajectory_files(trajectory_dir)
    all_tools: set[str] = set()
    tool_to_files: dict[str, list[str]] = {}
    for path in files:
        tools = extract_tools_used(path)
        all_tools.update(tools)
        for tool in tools:
            tool_to_files.setdefault(tool, []).append(path)
    return sorted(all_tools), tool_to_files


def build_record_label(index: int, record: dict[str, Any], path: Path) -> str:
    task_id = record.get("task_id")
    if task_id not in (None, ""):
        return f"{index + 1}. {path.name} | task_id={task_id}"
    return f"{index + 1}. {path.name}"


def _escape(text: str) -> str:
    return html.escape(text or "")


def _shorten(text: str, max_chars: int) -> str:
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... (truncated)"


def _looks_like_base64(value: str) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if len(stripped) < 32 or len(stripped) % 4 != 0:
        return False
    try:
        base64.b64decode(stripped, validate=True)
        return True
    except Exception:
        return False


def _ensure_image_url(value: str) -> str:
    if not value:
        return ""
    if value.startswith("data:image/"):
        return value
    if os.path.exists(value):
        suffix = Path(value).suffix.lower().lstrip(".") or "png"
        with open(value, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/{suffix};base64,{encoded}"
    if _looks_like_base64(value):
        return f"data:image/png;base64,{value}"
    return value


def _render_text_block(label: str, text: str, max_chars: int) -> str:
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


def _render_images(urls: list[str]) -> str:
    clean_urls = [_ensure_image_url(url) for url in urls if url]
    if not clean_urls:
        return ""
    img_tags = [f"<img src='{_escape(url)}' />" for url in clean_urls]
    return "<div class='images'>" + "".join(img_tags) + "</div>"


def _normalize_messages_for_render(messages: Any) -> list[dict[str, Any]]:
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


def _render_messages(messages: Any, max_chars: int) -> str:
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
                    url = ""
                    if part.get("type") == "image_url":
                        image_url = part.get("image_url")
                        if isinstance(image_url, dict):
                            url = image_url.get("url", "")
                        elif isinstance(image_url, str):
                            url = image_url
                    else:
                        url = part.get("value", "")
                    blocks.append(_render_images([url]))
                else:
                    blocks.append(_render_text_block("content", json.dumps(part, ensure_ascii=False), max_chars))
        else:
            blocks.append(_render_text_block("content", str(content), max_chars))
        blocks.append("<div class='divider'></div>")
    return "<div class='messages'>" + "".join(blocks) + "</div>"


def build_html(record: dict[str, Any], max_chars: int = 2000) -> str:
    header = (
        "<div class='header'>"
        f"<div class='title'>Task {_escape(str(record.get('task_id', 'Unknown')))}</div>"
        f"<div class='meta'>Dataset: {_escape(str(record.get('dataset', 'Unknown')))} | "
        f"Model: {_escape(str(record.get('model_name', 'Unknown')))} | "
        f"Agent: {_escape(str(record.get('agent_class', 'Unknown')))}</div>"
        "</div>"
    )

    body = []
    ground_truth = record.get("ground_truth")
    if ground_truth is not None:
        body.append("<h2>Ground Truth</h2>")
        body.append(_render_text_block("ground_truth", str(ground_truth), max_chars))

    prediction = record.get("prediction")
    if prediction is not None:
        body.append("<h2>Prediction</h2>")
        body.append(_render_text_block("prediction", str(prediction), max_chars))

    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        body.append("<h2>Messages</h2>")
        body.append(_render_messages(messages, max_chars))
    else:
        body.append("<h2>Raw JSON</h2>")
        body.append(_render_text_block("record", json.dumps(record, ensure_ascii=False, indent=2), max_chars))

    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Trajectory Viewer</title>
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


def sync_selected_index() -> None:
    st.session_state.selected_index = st.session_state.selected_option


def move_selection(delta: int, total: int) -> None:
    next_index = st.session_state.selected_index + delta
    st.session_state.selected_index = max(0, min(total - 1, next_index))
    st.session_state.selected_option = st.session_state.selected_index


def main() -> None:
    st.set_page_config(page_title="Trajectory Viewer", layout="wide")
    st.title("Trajectory Viewer")

    args = parse_args()
    if not args.trajectory_dir:
        st.error(
            "No trajectory directory configured. Set the default value of `--trajectory-dir` "
            "in `parse_args()` or pass `--trajectory-dir` when launching Streamlit."
        )
        return
    trajectory_dir = str(Path(args.trajectory_dir).expanduser())
    st.caption(f"Trajectory directory: `{trajectory_dir}`")

    try:
        files = list_trajectory_files(trajectory_dir)
    except Exception as exc:
        st.error(str(exc))
        return

    if not files:
        st.warning("No JSON files found in the trajectory directory.")
        return

    all_tools, tool_to_files = build_tool_index(trajectory_dir)

    # Tool filter
    tool_options = ["All tools"] + all_tools
    if "selected_tool" not in st.session_state:
        st.session_state.selected_tool = "All tools"

    def on_tool_change():
        st.session_state.selected_index = 0
        st.session_state.selected_option = 0

    st.selectbox(
        "Filter by tool",
        options=tool_options,
        key="selected_tool",
        on_change=on_tool_change,
    )

    # Filter files based on selected tool
    if st.session_state.selected_tool == "All tools":
        filtered_files = files
    else:
        filtered_set = set(tool_to_files.get(st.session_state.selected_tool, []))
        filtered_files = [f for f in files if f in filtered_set]

    if not filtered_files:
        st.info("No trajectories found for the selected tool.")
        return

    if "selected_index" not in st.session_state:
        st.session_state.selected_index = 0
    st.session_state.selected_index = max(0, min(st.session_state.selected_index, len(filtered_files) - 1))
    st.session_state.selected_option = st.session_state.selected_index

    left, middle, right = st.columns([1, 1, 3])
    with left:
        st.button(
            "Previous",
            use_container_width=True,
            disabled=st.session_state.selected_index == 0,
            on_click=move_selection,
            args=(-1, len(filtered_files)),
        )
    with middle:
        st.button(
            "Next",
            use_container_width=True,
            disabled=st.session_state.selected_index >= len(filtered_files) - 1,
            on_click=move_selection,
            args=(1, len(filtered_files)),
        )
    with right:
        st.selectbox(
            "Select trajectory",
            options=list(range(len(filtered_files))),
            key="selected_option",
            format_func=lambda idx: build_record_label(idx, load_record(filtered_files[idx]), Path(filtered_files[idx])),
            on_change=sync_selected_index,
        )

    selected_path = Path(filtered_files[st.session_state.selected_index])
    try:
        record = load_record(str(selected_path))
    except Exception as exc:
        st.error(f"Failed to load `{selected_path.name}`: {exc}")
        return

    meta_left, meta_right = st.columns([2, 1])
    with meta_left:
        st.write(
            f"Showing `{selected_path.name}` "
            f"({st.session_state.selected_index + 1} / {len(filtered_files)})"
        )
    with meta_right:
        st.write(f"Messages: `{len(record.get('messages') or [])}`")

    html_text = build_html(record)
    components.html(html_text, height=DEFAULT_HTML_HEIGHT, scrolling=True)


if __name__ == "__main__":
    main()
