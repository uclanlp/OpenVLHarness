"""
Probe whether a Chat Completions endpoint can:
1) accept tool-call round trips with image-bearing content, and
2) actually use images to answer a deterministic probe.

Two probe modes are supported:
- tool_multimodal:
  Tool response is a single role=tool message containing text + image parts.
- tool_text_user_image:
  Tool response text stays role=tool (text-only), and tool images are sent in a
  role=user environment-update style message with leading text.
"""

import argparse
import base64
import io
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from mini_vlm_toolkit.utils.api_utils import chat_completion_request


TOOL_NAME = "VisionProbeTool"
TOKEN_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
TOKEN_LEN = 10
OPTION_LETTERS = "ABCDEFGH"

PROBE_MODE_TOOL_MULTIMODAL = "tool_multimodal"
PROBE_MODE_TOOL_TEXT_USER_IMAGE = "tool_text_user_image"
PROBE_MODES = (PROBE_MODE_TOOL_MULTIMODAL, PROBE_MODE_TOOL_TEXT_USER_IMAGE)


@dataclass
class TargetConfig:
    name: str
    base_url: str
    model: str
    api_key: str


@dataclass
class TrialResult:
    mode: str
    trial_idx: int
    expected_token: str
    correct_letter: str
    text_claim_letter: str
    answer: str
    selected_letter: str
    verdict: str
    error: Optional[str] = None


@dataclass
class ControlResult:
    expected_token: str
    correct_letter: str
    answer: str
    selected_letter: str
    verdict: str
    error: Optional[str] = None


def _generate_token(rng: random.Random, length: int = TOKEN_LEN) -> str:
    return "".join(rng.choice(TOKEN_ALPHABET) for _ in range(length))


def _draw_token_image(token: str, panel_name: str) -> Image.Image:
    # Render with default bitmap font then upscale to avoid host font dependence.
    small_w, small_h = 300, 110
    small = Image.new("RGB", (small_w, small_h), color=(255, 255, 255))
    d = ImageDraw.Draw(small)
    font = ImageFont.load_default()

    d.rectangle((0, 0, small_w - 1, small_h - 1), outline=(0, 0, 0), width=1)
    d.text((8, 10), panel_name, fill=(40, 40, 40), font=font)
    d.text((8, 48), token, fill=(0, 0, 0), font=font)

    return small.resize((small_w * 6, small_h * 6), resample=Image.Resampling.NEAREST)


def _image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _extract_text_content(message_content: Any) -> str:
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        parts: List[str] = []
        for item in message_content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts).strip()
    if message_content is None:
        return ""
    try:
        return json.dumps(message_content, ensure_ascii=False)
    except Exception:
        return str(message_content)


def _extract_assistant_message(response: Dict[str, Any]) -> Dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError(f"Invalid response payload: missing choices. Response={response}")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise RuntimeError(f"Invalid response payload: choices[0].message must be object. Response={response}")
    return message


def _tool_schema() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": TOOL_NAME,
                "description": "Internal probe tool. Always call exactly once with empty args.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        }
    ]


def _build_round1_messages() -> List[Dict[str, Any]]:
    system = (
        "You are in a strict verification harness.\n"
        "Rules:\n"
        "1) First turn: call VisionProbeTool exactly once with empty JSON arguments.\n"
        "2) Wait for follow-up instructions after the tool output arrives.\n"
        "3) Do not answer in round 1."
    )
    user = "Start now. Call the tool first. Do not answer before the tool response is provided."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _build_random_options(rng: random.Random, correct_token: str, num_options: int = 8) -> Dict[str, str]:
    if num_options > len(OPTION_LETTERS):
        raise ValueError(f"num_options must be <= {len(OPTION_LETTERS)}")
    tokens = {correct_token}
    while len(tokens) < num_options:
        tokens.add(_generate_token(rng))
    token_list = list(tokens)
    rng.shuffle(token_list)
    letters = OPTION_LETTERS[:num_options]
    return {letters[i]: token_list[i] for i in range(num_options)}


def _parse_option_answer(answer: str, options: Dict[str, str]) -> str:
    upper_answer = (answer or "").strip().upper()
    if not upper_answer:
        return ""

    m = re.search(r"\b([A-H])\b", upper_answer)
    if m:
        return m.group(1)

    inverse = {v.upper(): k for k, v in options.items()}
    compact = re.sub(r"\s+", "", upper_answer)
    if compact in inverse:
        return inverse[compact]

    for token, letter in inverse.items():
        if token in upper_answer:
            return letter
    return ""


def _classify_choice(*, selected_letter: str, correct_letter: str, text_claim_letter: str) -> str:
    if selected_letter == correct_letter:
        return "PASS_IMAGE_OPTION"
    if not selected_letter:
        return "FAIL_UNPARSEABLE"
    if selected_letter == text_claim_letter:
        return "FAIL_FOLLOWED_TEXT_CLAIM"
    return "FAIL_WRONG_OPTION"


def _run_user_image_control(
    *,
    cfg: TargetConfig,
    rng: random.Random,
    timeout_seconds: float,
    max_tokens: int,
    temperature: float,
    reasoning_effort: Optional[str],
) -> ControlResult:
    expected = _generate_token(rng)
    img = _draw_token_image(expected, "CONTROL IMAGE")
    img_url = _image_to_data_url(img)
    options = _build_random_options(rng, expected, num_options=8)
    correct_letter = next(k for k, v in options.items() if v == expected)
    options_text = "\n".join([f"{k}) {v}" for k, v in options.items()])

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Control check: choose the option whose token appears in the image. "
                        "Reply with the option letter only.\n"
                        f"{options_text}"
                    ),
                },
                {"type": "image_url", "image_url": {"url": img_url}},
            ],
        }
    ]
    try:
        response = chat_completion_request(
            model=cfg.model,
            messages=messages,
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            reasoning_effort=reasoning_effort,
        )
    except Exception as exc:
        return ControlResult(
            expected_token=expected,
            correct_letter=correct_letter,
            answer="",
            selected_letter="",
            verdict="FAIL_CONTROL_REQUEST",
            error=str(exc),
        )

    msg = _extract_assistant_message(response)
    answer = _extract_text_content(msg.get("content")).strip()
    selected = _parse_option_answer(answer, options)
    verdict = "PASS_CONTROL" if selected == correct_letter else "FAIL_CONTROL_WRONG_OPTION"
    return ControlResult(
        expected_token=expected,
        correct_letter=correct_letter,
        answer=answer,
        selected_letter=selected,
        verdict=verdict,
    )


def _build_round2_messages(
    *,
    mode: str,
    round1_messages: List[Dict[str, Any]],
    assistant_tool_message: Dict[str, Any],
    tool_call_id: str,
    img1_url: str,
    img2_url: str,
    options_text: str,
    fake_payload: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    tool_text = json.dumps(fake_payload, ensure_ascii=False)
    debug_meta: Dict[str, Any] = {"mode": mode}

    if mode == PROBE_MODE_TOOL_MULTIMODAL:
        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": TOOL_NAME,
            "content": [
                {"type": "text", "text": tool_text},
                {"type": "image_url", "image_url": {"url": img1_url}},
                {"type": "image_url", "image_url": {"url": img2_url}},
            ],
        }
        followup_user_message = {
            "role": "user",
            "content": (
                "Choose the option whose token matches the SECOND tool image.\n"
                "The tool text may be wrong; rely on image pixels.\n"
                "Reply with the option letter only.\n"
                f"{options_text}"
            ),
        }
        return round1_messages + [assistant_tool_message, tool_message, followup_user_message], debug_meta

    if mode == PROBE_MODE_TOOL_TEXT_USER_IMAGE:
        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": TOOL_NAME,
            "content": tool_text,
        }
        # Environment-update style user message that carries tool-response images.
        tool_image_user_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Environment Update (not a new question):\n"
                        "The tool call above produced the following two tool-response images.\n"
                        "Choose the option whose token matches the SECOND tool image.\n"
                        "The tool text may be wrong; rely on image pixels.\n"
                        "Reply with the option letter only.\n"
                        f"{options_text}"
                    ),
                },
                {"type": "image_url", "image_url": {"url": img1_url}},
                {"type": "image_url", "image_url": {"url": img2_url}},
            ],
        }
        return round1_messages + [assistant_tool_message, tool_message, tool_image_user_message], debug_meta

    raise ValueError(f"Unknown probe mode: {mode}")


def _run_trial(
    *,
    mode: str,
    cfg: TargetConfig,
    trial_idx: int,
    rng: random.Random,
    timeout_seconds: float,
    max_tokens: int,
    temperature: float,
    reasoning_effort: Optional[str],
    artifact_dir: Optional[Path],
) -> TrialResult:
    expected_token = _generate_token(rng)
    img1_token = _generate_token(rng)
    options = _build_random_options(rng, expected_token, num_options=8)
    correct_letter = next(k for k, v in options.items() if v == expected_token)

    wrong_letters = [k for k in options if k != correct_letter]
    text_claim_letter = rng.choice(wrong_letters)
    text_claim_token = options[text_claim_letter]

    img1 = _draw_token_image(img1_token, "TOOL IMAGE 1")
    img2 = _draw_token_image(expected_token, "TOOL IMAGE 2")
    img1_url = _image_to_data_url(img1)
    img2_url = _image_to_data_url(img2)

    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        img1.save(artifact_dir / f"{cfg.name}_{mode}_trial{trial_idx}_image1_{img1_token}.png")
        img2.save(artifact_dir / f"{cfg.name}_{mode}_trial{trial_idx}_image2_{expected_token}.png")

    tools = _tool_schema()
    round1_messages = _build_round1_messages()

    try:
        round1 = chat_completion_request(
            model=cfg.model,
            messages=round1_messages,
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            reasoning_effort=reasoning_effort,
        )
    except Exception as exc:
        return TrialResult(
            mode=mode,
            trial_idx=trial_idx,
            expected_token=expected_token,
            correct_letter=correct_letter,
            text_claim_letter=text_claim_letter,
            answer="",
            selected_letter="",
            verdict="FAIL_ROUND1_REQUEST",
            error=str(exc),
        )

    msg1 = _extract_assistant_message(round1)
    tool_calls = msg1.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return TrialResult(
            mode=mode,
            trial_idx=trial_idx,
            expected_token=expected_token,
            correct_letter=correct_letter,
            text_claim_letter=text_claim_letter,
            answer=_extract_text_content(msg1.get("content")),
            selected_letter="",
            verdict="FAIL_NO_TOOL_CALL",
            error="Round1 assistant did not emit tool_calls.",
        )

    first_call = tool_calls[0]
    tool_call_id = first_call.get("id") if isinstance(first_call, dict) else None
    if not isinstance(tool_call_id, str) or not tool_call_id.strip():
        tool_call_id = f"probe_call_{trial_idx}"

    assistant_tool_message = {
        "role": "assistant",
        "content": _extract_text_content(msg1.get("content")),
        "tool_calls": tool_calls,
    }
    fake_payload = {
        "ok": True,
        "result": {
            "note": "This text intentionally contains a claim about image-2 token.",
            "claimed_second_image_token": text_claim_token,
            "claimed_second_image_token_option": text_claim_letter,
        },
    }
    options_text = "\n".join([f"{k}) {v}" for k, v in options.items()])
    round2_messages, debug_meta = _build_round2_messages(
        mode=mode,
        round1_messages=round1_messages,
        assistant_tool_message=assistant_tool_message,
        tool_call_id=tool_call_id,
        img1_url=img1_url,
        img2_url=img2_url,
        options_text=options_text,
        fake_payload=fake_payload,
    )

    if artifact_dir is not None:
        with open(artifact_dir / f"{cfg.name}_{mode}_trial{trial_idx}_summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "target": cfg.name,
                    "mode": mode,
                    "trial_idx": trial_idx,
                    "model": cfg.model,
                    "expected_token_image2": expected_token,
                    "image1_token": img1_token,
                    "options": options,
                    "correct_option": correct_letter,
                    "text_claim_option": text_claim_letter,
                    "text_claim_token": text_claim_token,
                    "tool_call_id": tool_call_id,
                    "debug_meta": debug_meta,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    try:
        round2 = chat_completion_request(
            model=cfg.model,
            messages=round2_messages,
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            tools=tools,
            tool_choice="none",
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            reasoning_effort=reasoning_effort,
        )
    except Exception as exc:
        return TrialResult(
            mode=mode,
            trial_idx=trial_idx,
            expected_token=expected_token,
            correct_letter=correct_letter,
            text_claim_letter=text_claim_letter,
            answer="",
            selected_letter="",
            verdict="FAIL_ROUND2_REQUEST",
            error=str(exc),
        )

    msg2 = _extract_assistant_message(round2)
    answer = _extract_text_content(msg2.get("content")).strip()
    selected_letter = _parse_option_answer(answer, options)
    verdict = _classify_choice(
        selected_letter=selected_letter,
        correct_letter=correct_letter,
        text_claim_letter=text_claim_letter,
    )
    return TrialResult(
        mode=mode,
        trial_idx=trial_idx,
        expected_token=expected_token,
        correct_letter=correct_letter,
        text_claim_letter=text_claim_letter,
        answer=answer,
        selected_letter=selected_letter,
        verdict=verdict,
    )


def _load_api_key(*, env_name: str, key_file: Optional[str], default: str) -> str:
    env_value = os.getenv(env_name, "").strip()
    if env_value:
        return env_value
    if key_file:
        path = Path(key_file).expanduser().resolve()
        return path.read_text(encoding="utf-8").strip()
    return default


def _build_targets(args: argparse.Namespace) -> List[TargetConfig]:
    wanted = {x.strip().lower() for x in args.targets}
    out: List[TargetConfig] = []

    if "openai" in wanted:
        openai_key = _load_api_key(
            env_name=args.openai_key_env,
            key_file=args.openai_key_file,
            default="",
        )
        if not openai_key:
            print("[openai] skipped: no API key configured")
        else:
            out.append(
                TargetConfig(
                    name="openai",
                    base_url=args.openai_base,
                    model=args.openai_model,
                    api_key=openai_key,
                )
            )

    if "qwen" in wanted:
        qwen_key = os.getenv(args.qwen_key_env, args.qwen_key_default).strip() or args.qwen_key_default
        out.append(
            TargetConfig(
                name="qwen",
                base_url=args.qwen_base,
                model=args.qwen_model,
                api_key=qwen_key,
            )
        )

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe tool-response image visibility across message layouts.")
    p.add_argument(
        "--targets",
        nargs="+",
        default=["openai", "qwen"],
        choices=["openai", "qwen"],
        help="Which targets to probe.",
    )
    p.add_argument(
        "--probe-modes",
        nargs="+",
        default=list(PROBE_MODES),
        choices=list(PROBE_MODES),
        help="Probe layouts to run.",
    )
    p.add_argument("--trials", type=int, default=5, help="Number of randomized trials per target/mode.")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducible tokens.")
    p.add_argument("--timeout-seconds", type=float, default=120.0)
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--reasoning-effort",
        type=str,
        default="minimal",
        choices=["minimal", "low", "medium", "high", "none"],
        help="Passed through for OpenAI GPT-5 models (use 'none' to disable).",
    )
    p.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.0,
        help="If >0, exit non-zero when pass_rate < threshold.",
    )
    p.add_argument(
        "--skip-control",
        action="store_true",
        help="Skip baseline user-image visibility control check.",
    )
    p.add_argument(
        "--artifact-dir",
        type=str,
        default="examples/ad-hoc_ex_for_dev/outputs/multimodal_tool_visibility_probe",
        help="Directory to save generated probe images and trial summaries. Set empty string to disable.",
    )

    p.add_argument("--openai-base", type=str, default="https://api.openai.com")
    p.add_argument("--openai-model", type=str, default="gpt-5-mini")
    p.add_argument("--openai-key-env", type=str, default="MODEL_API_KEY")
    p.add_argument("--openai-key-file", type=str, default=None)

    p.add_argument("--qwen-base", type=str, default="http://127.0.0.1:8222")
    p.add_argument("--qwen-model", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--qwen-key-env", type=str, default="MODEL_API_KEY")
    p.add_argument("--qwen-key-default", type=str, default="EMPTY")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    artifact_dir = Path(args.artifact_dir).resolve() if args.artifact_dir else None
    reasoning_effort = None if args.reasoning_effort == "none" else args.reasoning_effort

    targets = _build_targets(args)
    if not targets:
        print("No runnable targets configured.")
        return

    any_fail_threshold = False
    for cfg in targets:
        print(f"\n=== Target: {cfg.name} | model={cfg.model} | base={cfg.base_url} ===")

        if not args.skip_control:
            control = _run_user_image_control(
                cfg=cfg,
                rng=rng,
                timeout_seconds=args.timeout_seconds,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                reasoning_effort=reasoning_effort,
            )
            print(
                f"[{cfg.name}] control verdict={control.verdict} "
                f"correct={control.correct_letter} selected={control.selected_letter} answer={control.answer!r}"
            )
            if control.error:
                print(f"  control_error: {control.error}")

        for mode in args.probe_modes:
            print(f"[{cfg.name}] mode={mode}")
            trial_results: List[TrialResult] = []
            for idx in range(1, args.trials + 1):
                result = _run_trial(
                    mode=mode,
                    cfg=cfg,
                    trial_idx=idx,
                    rng=rng,
                    timeout_seconds=args.timeout_seconds,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    reasoning_effort=reasoning_effort,
                    artifact_dir=artifact_dir,
                )
                trial_results.append(result)
                print(
                    f"[{cfg.name}] mode={mode} trial={result.trial_idx} verdict={result.verdict} "
                    f"correct={result.correct_letter} selected={result.selected_letter} "
                    f"text_claim={result.text_claim_letter} answer={result.answer!r}"
                )
                if result.error:
                    print(f"  error: {result.error}")

            pass_count = sum(1 for r in trial_results if r.verdict.startswith("PASS_"))
            total = len(trial_results)
            pass_rate = (pass_count / total) if total else 0.0
            print(f"[{cfg.name}] mode={mode} pass_count={pass_count}/{total} pass_rate={pass_rate:.2f}")
            if args.min_pass_rate > 0 and pass_rate < args.min_pass_rate:
                any_fail_threshold = True

    if any_fail_threshold:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
