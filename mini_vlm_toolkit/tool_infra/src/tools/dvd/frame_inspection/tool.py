import os

import numpy as np

from tools.base import BaseTool
from tools.api_utils import attach_images, chat_completion_request

TOOL_NAME = "FrameInspection_Tool"

# Env-driven configuration
ENV_VLM_MODEL = "DVD_FI_VLM_MODEL"
ENV_VLM_BASE_URL = "DVD_FI_VLM_BASE_URL"
ENV_VLM_API_KEY = "DVD_FI_VLM_API_KEY"
ENV_MAX_FRAMES = "DVD_FI_MAX_FRAMES"
ENV_VIDEO_FPS = "DVD_FI_VIDEO_FPS"
ENV_VIDEO_DB_ROOT = "DVD_FI_VIDEO_DB_ROOT"

DEFAULT_PROMPT = (
    "Carefully watch the video frames. Pay attention to the cause and sequence "
    "of events, the detail and movement of objects and the action and pose of "
    "persons.\n\nBased on your observations, if you find content that can answer "
    "the question. If no relevant content is found within the given time range, "
    "return: `Error: Cannot find corresponding result in the given time range.`. \n"
    "Question: {question}\n"
)
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "memory": {
            "type": "object",
            "description": "NanoVectorDB object (runtime-injected)",
        },
        "database": {
            "type": "object",
            "description": "Alias of memory for compatibility",
        },
        "question": {"type": "string", "description": "Question to answer"},
        "time_ranges_hhmmss": {
            "type": "array",
            "description": "Time ranges as [(start, end), ...] in HH:MM:SS or MM:SS",
            "items": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {"type": "string"},
            },
        },
    },
    "required": ["question", "time_ranges_hhmmss"],
    "anyOf": [
        {"required": ["memory"]},
        {"required": ["database"]},
    ],
    "additionalProperties": False,
}


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _convert_hhmmss_to_seconds(hhmmss: str) -> int:
    hhmmss = hhmmss.split(".")[0]
    parts = hhmmss.split(":")
    if len(parts) < 2:
        raise ValueError("Invalid time format. Expected HH:MM:SS.")
    if len(parts) == 2:
        parts = ["00"] + parts
    hours, minutes, seconds = map(int, parts)
    return hours * 3600 + minutes * 60 + seconds


class FrameInspection_Tool(BaseTool):
    def __init__(self):
        self.vlm_model = os.getenv(ENV_VLM_MODEL)
        self.vlm_base_url = os.getenv(ENV_VLM_BASE_URL)
        self.vlm_api_key = os.getenv(ENV_VLM_API_KEY, "EMPTY")
        self.default_max_frames = int(os.getenv(ENV_MAX_FRAMES, "50") or 50)
        self.default_fps = float(
            os.getenv(ENV_VIDEO_FPS, "2") or 2
        )

        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Inspect frames within given time ranges from a NanoVectorDB's video "
                "metadata and answer a detailed question."
            ),
            tool_version="0.2.0",
            input_types={
                "database": "NanoVectorDB - preloaded vector database object",
                "question": "str - question to answer",
                "time_ranges_hhmmss": "list[tuple[str, str]] - ranges (HH:MM:SS or MM:SS)",
            },
            input_schema=INPUT_SCHEMA,
            output_type="str - model response",
            demo_commands=[
                {
                    "command": (
                        "FrameInspection_Tool(database=db_obj, "
                        "question=\"What is the person holding?\", "
                        "time_ranges_hhmmss=[(\"00:10\", \"00:20\")])"
                    ),
                    "description": "Inspect a short time range.",
                }
            ],
            user_metadata={
                "env": {
                    ENV_VLM_MODEL: "required",
                    ENV_VLM_BASE_URL: "required",
                    ENV_VLM_API_KEY: "optional",
                    ENV_MAX_FRAMES: "optional",
                    ENV_VIDEO_FPS: "optional",
                    ENV_VIDEO_DB_ROOT: "optional (fallback for frames path)",
                },
                "status": "implemented",
            },
        )

    def sanity_check(self):
        missing = []
        for name, value in [
            (ENV_VLM_MODEL, self.vlm_model),
            (ENV_VLM_BASE_URL, self.vlm_base_url),
        ]:
            if not value or not str(value).strip():
                missing.append(f"{name} is required")
        return missing

    def execute(
        self,
        memory=None,
        question=None,
        time_ranges_hhmmss=None,
        database=None,
    ):
        if memory is None:
            memory = database
        if memory is None:
            raise ValueError("memory is required")
        database = memory
        if not question:
            raise ValueError("question is required")
        if not time_ranges_hhmmss:
            raise ValueError("time_ranges_hhmmss is required")

        video_meta = database.get_additional_data() or {}
        video_length = video_meta.get("video_length")
        if not video_length:
            raise ValueError("video_length is missing in database additional data")

        video_length_secs = _convert_hhmmss_to_seconds(str(video_length))
        fps = float(video_meta.get("fps") or self.default_fps)

        # Resolve frames directory
        resolved_frames_dir = None
        video_file_root = video_meta.get("video_file_root")
        if video_file_root:
            resolved_frames_dir = os.path.join(video_file_root, "frames")
        if not resolved_frames_dir:
            video_db_root = os.getenv(ENV_VIDEO_DB_ROOT)
            if video_db_root:
                try:
                    storage = database.storage_file
                    video_id = os.path.basename(os.path.dirname(storage))
                    resolved_frames_dir = os.path.join(video_db_root, video_id, "frames")
                except Exception:
                    resolved_frames_dir = None
        if not resolved_frames_dir or not os.path.isdir(resolved_frames_dir):
            raise FileNotFoundError("frames directory cannot be resolved")

        # Time ranges
        time_ranges_secs = []
        for time_range in time_ranges_hhmmss:
            start_secs = _convert_hhmmss_to_seconds(str(time_range[0]))
            end_secs = _convert_hhmmss_to_seconds(str(time_range[1]))
            if start_secs > video_length_secs:
                fixed_start_time = "00:" + str(time_range[0]).split(":")[0] + ":" + str(
                    time_range[0]
                ).split(":")[1]
                fixed_end_time = "00:" + str(time_range[1]).split(":")[0] + ":" + str(
                    time_range[1]
                ).split(":")[1]
                start_secs = _convert_hhmmss_to_seconds(fixed_start_time)
                end_secs = _convert_hhmmss_to_seconds(fixed_end_time)
                if start_secs > video_length_secs:
                    raise ValueError(
                        f"Start time {time_range[0]} exceeds video length {video_length_secs}"
                    )
            end_secs = min(end_secs, video_length_secs)
            time_ranges_secs.append((start_secs, end_secs))

        time_ranges_secs.sort(key=lambda x: x[0])
        total_time = sum(end - start for start, end in time_ranges_secs)
        if total_time <= 0:
            total_time = 1

        max_frames = _get_env_int(ENV_MAX_FRAMES, self.default_max_frames)
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")

        offsets = np.linspace(0, total_time, num=max_frames, endpoint=False, dtype=float)
        prefix_len = []
        acc = 0
        for start, end in time_ranges_secs:
            seg_len = end - start
            prefix_len.append((acc, start, seg_len))
            acc += seg_len

        timepoints = []
        for off in offsets:
            for base, seg_start, seg_len in prefix_len:
                if off < base + seg_len:
                    timepoints.append(seg_start + (off - base))
                    break

        max_frame_idx = int(video_length_secs * fps) - 1
        frame_fns = os.listdir(resolved_frames_dir)
        frame_indices = []
        for fn in frame_fns:
            if not fn.startswith("frames_n") or not fn.endswith(".jpg"):
                continue
            try:
                frame_indices.append(int(fn.split("frames_n")[-1].split(".jpg")[0]))
            except ValueError:
                continue
        if not frame_indices:
            raise ValueError(f"No frame files found in {resolved_frames_dir}")
        max_frame_fn = max(frame_indices)
        if max_frame_idx > max_frame_fn:
            max_frame_idx = max_frame_fn - 1

        framepoints = [
            min(max(int(round(ts * fps)), 0), max_frame_idx) for ts in timepoints
        ]
        framepoints = sorted(set(framepoints))[:max_frames]
        files = [
            os.path.join(resolved_frames_dir, f"frames_n{fn+1:06d}.jpg")
            for fn in framepoints
        ]
        files = [path for path in files if os.path.isfile(path)]
        if not files:
            raise ValueError("No frames found for the given time ranges")

        input_msgs = [
            {
                "role": "system",
                "content": "You are a helpful assistant to answer questions.",
            },
            {
                "role": "user",
                "content": DEFAULT_PROMPT.format(question=question),
            },
        ]
        response = chat_completion_request(
            model=self.vlm_model,
            base_url=self.vlm_base_url,
            api_key=self.vlm_api_key,
            messages=input_msgs,
            temperature=0.0,
            max_tokens=512,
            timeout_seconds=120,
            image_paths=files,
        )
        message = response.get("choices", [{}])[0].get("message", {})
        content = message.get("content")
        if content is None:
            raise ValueError("No response from the model")
        return content
