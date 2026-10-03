"""Registered model profiles, native API encoders and context policy."""
import copy as cp
import json
import logging
import re
from importlib.resources import files
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
import requests
import yaml
from .http import retry_with_exponential_backoff, raise_for_status_with_detail


def failure_text(payload):
    error = payload.get("error")
    if isinstance(error, dict):
        code = str(error["code"]).strip() if error.get("code") is not None else ""
        message = str(error["message"]).strip() if error.get("message") is not None else ""
        message = message or "Tool call failed."
        return f"{code}: {message}" if code else message
    return str(error).strip() if error not in (None, "", {}) and str(error).strip() else "Tool call failed."


def load_profile(alias):
    profiles = yaml.safe_load(files("openvlharness.services").joinpath("model_registry.yaml").read_text())["profiles"]
    if alias not in profiles:
        raise ValueError(f"Unknown model alias {alias!r}. Register its exact served ID, prompt and API profile in model_registry.yaml.")
    return cp.deepcopy(profiles[alias])


class AuxiliaryModel:
    """Configured search/extraction model with the paper's task-specific options."""
    def __init__(self, config):
        self.adapter = ModelAdapter({"retry": 2, **config})
        self.model = self.adapter.model
        self.base_url = self.adapter.base
        self.api_key = self.adapter.key
        # The old summarizer used the OpenAI SDK's two-retry transport policy.
        # Keep that policy while sharing the registered protocol/payload adapter.
        from openai import OpenAI
        from httpx import Timeout
        base = self.base_url.rstrip("/")
        for suffix in ("/chat/completions", "/responses"):
            if base.endswith(suffix):
                base = base[:-len(suffix)]
        if not base.endswith("/v1"):
            base += "/v1"
        self.client = OpenAI(base_url=base, api_key=self.api_key or "empty",
                             timeout=Timeout(self.adapter.timeout, connect=5), max_retries=self.adapter.retry)

    def generate(self, prompt_or_messages, system_prompt=None, **generation_kwargs):
        messages = ([{"role": "user", "content": prompt_or_messages}]
                    if isinstance(prompt_or_messages, str) else cp.deepcopy(prompt_or_messages))
        if system_prompt and not any(m.get("role") == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": system_prompt})
        budget = generation_kwargs.pop("max_tokens", 8192)
        temperature = generation_kwargs.pop("temperature", 0.0)
        top_p = generation_kwargs.pop("top_p", 1.0)
        if self.adapter.protocol == "responses":
            options = {"max_tokens": budget, "reasoning_effort": "medium"}
            messages = [self.adapter.encode_message(m) for m in messages]
        else:
            options = {"temperature": temperature,
                       "top_p": top_p, "max_tokens": budget,
                       "chat_template_kwargs": {"enable_thinking": False}}
        if generation_kwargs:
            raise ValueError(f"Unsupported auxiliary generation options: {sorted(generation_kwargs)}")
        payload = self.adapter.payload(messages, options=options)
        if self.adapter.protocol == "responses":
            response = self.client.responses.create(**payload).model_dump()
        else:
            extra = {"chat_template_kwargs": payload.pop("chat_template_kwargs")}
            response = self.client.chat.completions.create(**payload, extra_body=extra).model_dump()
        return self.adapter.parse(response, 0)[0]

    generate_messages = generate


class ModelAdapter:
    def __init__(self, config, *, max_context_length=None, requester=None):
        self.profile = load_profile(config.get("alias"))
        connection = {"alias", "base", "key", "timeout", "retry"}
        unknown = set(config) - connection - set(self.profile["allowed_options"])
        if unknown:
            raise ValueError(f"Unsupported options for {config.get('alias')}: {sorted(unknown)}")
        self.model = self.profile["model_id"]
        self.protocol = self.profile["adapter"]
        self.base = config.get("base", "")
        parsed = urlparse(self.base)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("Model base must be an HTTP(S) URL")
        trimmed = self.base.rstrip("/")
        suffix = "/chat/completions" if self.protocol == "chat_completions" else "/responses"
        if trimmed.endswith(suffix):
            self.endpoint = trimmed
        elif trimmed.endswith(("/responses", "/chat/completions")):
            raise ValueError(f"Model endpoint does not match profile protocol {self.protocol}")
        else:
            self.endpoint = trimmed + ("" if trimmed.endswith("/v1") else "/v1") + suffix
        self.key = config.get("key", "")
        self.timeout = config.get("timeout", 600)
        self.retry = config.get("retry", 8)
        if type(self.retry) is not int or self.retry < 0 or type(self.timeout) not in (int, float) or self.timeout <= 0:
            raise ValueError("Model retry must be nonnegative and timeout must be positive")
        self.options = {key: value for key, value in config.items() if key not in connection}
        for key in ("max_tokens", "top_k"):
            if key in self.options and (type(self.options[key]) is not int or self.options[key] <= 0):
                raise ValueError(f"{key} must be a positive integer")
        if self.options.get("reasoning_effort", "medium") not in {"none", "low", "medium", "high", "xhigh"}:
            raise ValueError("Unsupported reasoning_effort")
        if self.options.get("return_token_ids", False) is not False:
            raise ValueError("This release profile requires return_token_ids=false")
        for key, low, high in (("temperature", 0, 2), ("top_p", 0, 1),
                               ("presence_penalty", -2, 2), ("repetition_penalty", 0, None)):
            if key in self.options:
                value = self.options[key]
                if (type(value) not in (int, float) or value < low or
                        (high is not None and value > high) or
                        (key in {"top_p", "repetition_penalty"} and value == 0)):
                    raise ValueError(f"Invalid {key}")
        if max_context_length is not None and (type(max_context_length) is not int or max_context_length <= 0):
            raise ValueError("max_context_length must be a positive integer or null")
        self.max_tokens = self.options.get("max_tokens", 4096)
        self.max_context_length = max_context_length
        self.logger = logging.getLogger("openvlharness.models")
        self._tokenizer = None
        self.requester = requester
        self.json_schema = []
        self.counting_available = False

    def load_tokenizer(self):
        if self.profile["context_policy"] == "qwen_tokenizer" and self.max_context_length:
            try:
                from transformers import AutoTokenizer
                try:
                    self._tokenizer = AutoTokenizer.from_pretrained(self.model, local_files_only=True)
                except OSError:
                    self._tokenizer = AutoTokenizer.from_pretrained(self.model)
                self.counting_available = True
            except (ImportError, OSError, ValueError) as exc:
                self.logger.warning("Tokenizer unavailable; provider enforces context limit: %s", exc)

    def encode_message(self, message):
        if self.protocol == "chat_completions":
            return cp.deepcopy(message)
        return {"type": "message", "role": message["role"],
                "content": self._content_to_responses_parts(message.get("content"), message["role"])}

    def payload(self, messages, tools=None, tool_choice="auto", *, options=None, seed=None):
        options = cp.deepcopy(self.options if options is None else options)
        if self.protocol == "chat_completions":
            payload = {"model": self.model, "messages": cp.deepcopy(messages), **options}
            if seed is not None:
                payload["seed"] = int(seed)
            if tools:
                payload.update(tools=cp.deepcopy(tools), tool_choice=tool_choice)
        else:
            payload = {"model": self.model, "input": cp.deepcopy(messages), "store": False}
            if "max_tokens" in options:
                payload["max_output_tokens"] = options["max_tokens"]
            if options.get("reasoning_effort"):
                payload["reasoning"] = {"effort": options["reasoning_effort"]}
            if tools:
                payload["tools"] = [{"type": "function", **cp.deepcopy(t["function"])} for t in tools]
                for tool in payload["tools"]:
                    parameters = tool.get("parameters")
                    if isinstance(parameters, dict):
                        for key in ("oneOf", "anyOf", "allOf", "enum", "not"):
                            parameters.pop(key, None)
                        parameters.setdefault("type", "object")
                payload["tool_choice"] = tool_choice
        return payload

    def request(self, messages, tools=None, tool_choice="auto", *, options=None, seed=None):
        payload = self.payload(messages, tools, tool_choice, options=options, seed=seed)
        if self.requester:
            return self.requester(payload)
        def once():
            headers = {"Content-Type": "application/json"}
            if self.key and str(self.key).lower() not in {"empty", "none"}:
                headers["Authorization"] = f"Bearer {self.key}"
            response = requests.post(self.endpoint, json=payload, headers=headers, timeout=self.timeout)
            raise_for_status_with_detail(response, prefix=f"Model request failed at endpoint {self.endpoint}")
            return response.json()
        return retry_with_exponential_backoff(once, max_retries=self.retry)()

    def parse(self, response, turn_idx, *, phase="main"):
        if not isinstance(response, dict):
            raise RuntimeError("Model response must be an object")
        if self.protocol == "chat_completions":
            choices = response.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise RuntimeError("Invalid chat response payload: missing choices")
            message = choices[0].get("message")
            if not isinstance(message, dict):
                raise RuntimeError("Invalid chat response payload: choices[0].message must be an object")
            text = message.get("content") if isinstance(message.get("content"), str) else ""
            calls = self._enforce_single_tool_call_per_message(
                self._normalize_tool_calls(message.get("tool_calls"), turn_idx), turn_idx=turn_idx, phase=phase)
            return text, calls, message
        output = response.get("output") or []
        if not isinstance(output, list):
            raise RuntimeError("Invalid responses payload: output must be a list")
        calls = self._normalize_responses_tool_calls(self._extract_function_calls(output), turn_idx)
        if len(calls) > 1:
            for call in calls:
                call["parse_error"] = {"code": "multi_tool_call_rejected",
                    "message": "Only one tool call is allowed per turn. Please re-issue exactly one."}
        return self._extract_assistant_text(output), calls, None

    def append_assistant_calls(self, messages, text, calls, original):
        if self.protocol == "chat_completions":
            message = {"role": "assistant", "content": original.get("content") or "",
                       "tool_calls": [call["assistant_tool_call"] for call in calls]}
            # Future registered Chat profiles may require native reasoning fields.
            if original.get("reasoning_content") is not None:
                message["reasoning_content"] = original["reasoning_content"]
            messages.append(message)
        else:
            messages.extend(call["function_call_item"] for call in calls)
            if text and len(calls) == 1:
                messages.append(self.encode_message({"role": "assistant", "content": text}))

    def result_messages(self, call, result):
        evidence = result.evidence
        content = evidence[0]["text"] if len(evidence) == 1 and evidence[0].get("type") == "text" else evidence
        deferred = {"role": "user", "content": content} if result.ok else None
        if self.protocol == "responses":
            return [self._build_function_call_output(call_id=call["id"], normalized_payload=result.payload,
                        deferred_message=deferred, run_memory=None)], []
        ack = {"role": "tool", "tool_call_id": call["id"], "name": call["name"],
               "content": "ok" if result.ok else failure_text(result.payload)}
        return [ack], [deferred] if deferred else []

    def final_answer(self, text):
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        tagged = match.group(1).strip() if match else ""
        return tagged or text if self.profile["final_answer"] == "answer_tag_or_text" else tagged

    def generate_text(self, messages, *, max_tokens=4096):
        options = {"max_tokens": max_tokens}
        if self.protocol == "responses":
            options["reasoning_effort"] = "medium"
            messages = [self.encode_message(m) for m in messages]
        else:
            options["temperature"] = 0
        response = self.request(messages, options=options)
        text, _, _ = self.parse(response, 0)
        return text

    def _normalize_tool_calls(self, tool_calls, turn_idx: int) -> List[Dict[str, Any]]:
        """Normalize native OpenAI `tool_calls` into a stable internal format.

        Parsing failures are preserved as structured `parse_error` payloads so
        the model can receive them as tool errors rather than hard failures.
        """
        normalized = []
        if not isinstance(tool_calls, list):
            return normalized

        for idx, raw_call in enumerate(tool_calls):
            if not isinstance(raw_call, dict):
                continue
            fn = raw_call.get("function") or {}
            tool_name = fn.get("name")
            if not tool_name:
                continue

            call_id = raw_call.get("id") or f"call_{turn_idx + 1}_{idx + 1}"
            raw_args = fn.get("arguments", "{}")
            args_text = "{}"
            args_dict = {}
            parse_error = None

            if isinstance(raw_args, dict):
                args_dict = raw_args
                args_text = json.dumps(raw_args, ensure_ascii=False)
            elif isinstance(raw_args, str):
                args_text = raw_args
                if raw_args.strip():
                    try:
                        parsed_args = json.loads(raw_args)
                    except Exception as exc:
                        parsed_args = {}
                        parse_error = {
                            "code": "invalid_tool_arguments_json",
                            "message": "Tool call arguments are not valid JSON.",
                            "details": {
                                "raw_arguments": raw_args,
                                "exception": str(exc),
                            },
                        }
                    if isinstance(parsed_args, dict):
                        args_dict = parsed_args
                    elif parse_error is None:
                        parse_error = {
                            "code": "invalid_tool_arguments_type",
                            "message": "Tool call arguments JSON must decode to an object.",
                            "details": {"decoded_type": type(parsed_args).__name__},
                        }
            else:
                args_text = "{}"
                args_dict = {}
                parse_error = {
                    "code": "invalid_tool_arguments_type",
                    "message": "Tool call arguments must be a JSON object or a JSON-encoded string.",
                    "details": {"raw_type": type(raw_args).__name__},
                }

            normalized.append(
                {
                    "id": call_id,
                    "name": tool_name,
                    "raw_arguments": args_text,
                    "args": args_dict,
                    "assistant_tool_call": {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": tool_name, "arguments": args_text},
                    },
                    "parse_error": parse_error,
                }
            )
        return normalized

    def _enforce_single_tool_call_per_message(
        self,
        tool_calls: List[Dict[str, Any]],
        *,
        turn_idx: int,
        phase: str,
    ) -> List[Dict[str, Any]]:
        """Reject multi-call assistant messages by converting them into tool errors.

        We preserve all emitted `tool_call_id`s so the model receives protocol-
        consistent tool-result feedback messages, but no actual tool execution
        happens.
        """
        if not isinstance(tool_calls, list) or len(tool_calls) <= 1:
            return tool_calls

        tool_names = [str((call or {}).get("name") or "unknown") for call in tool_calls]
        self.logger.warning(
            "Policy violation: multiple tool calls in one assistant message are not allowed. "
            "turn=%s phase=%s tool_call_count=%s tool_names=%s",
            turn_idx + 1,
            phase,
            len(tool_calls),
            tool_names,
        )

        total_calls = len(tool_calls)
        for idx, call in enumerate(tool_calls, 1):
            if not isinstance(call, dict):
                continue
            original_error = call.get("parse_error")
            details = {
                "phase": phase,
                "turn": turn_idx + 1,
                "tool_call_count": total_calls,
                "tool_call_index": idx,
                "tool_names": tool_names,
            }
            if isinstance(original_error, dict):
                details["suppressed_error"] = cp.deepcopy(original_error)
            call["parse_error"] = {
                "code": "multiple_tool_calls_not_allowed",
                "message": (
                    "Multiple tool calls in one assistant message are not allowed. "
                    "Call at most one tool per turn."
                ),
                "details": details,
            }
        return tool_calls

    def _count_tokens(self, msgs, tools=None):
        """Return token count for *msgs* using the cached tokenizer, or None."""
        try:
            tokenizer = getattr(self, "_tokenizer", None)
            if tokenizer is None:
                return None
            sanitized = self._sanitize_msgs_for_template(msgs)
            tmpl_kwargs = dict(tokenize=True, add_generation_prompt=True)
            if tools:
                tmpl_kwargs["tools"] = tools
            tokens = tokenizer.apply_chat_template(sanitized, **tmpl_kwargs)
            return len(tokens)
        except Exception:
            return None

    @staticmethod
    def _is_environment_update(msg):
        """Check if a message is an Environment Update from persistent state memory."""
        content = msg.get("content", "")
        if isinstance(content, str):
            return content.startswith("Environment Update")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    if part.get("text", "").startswith("Environment Update"):
                        return True
        return False

    def _compact_history(self, msgs, token_budget, tools=None):
        """Truncate older conversation messages in-place to fit *token_budget*.

        Walks from oldest to newest and replaces tool/assistant/user content
        with a short placeholder.  Preserves:
        - The system message and initial user message block
        - The most recent assistant+tool exchange
        - The latest Environment Update message (needed for image ID tracking)

        Returns True if the messages now fit within *token_budget*.
        """
        # Find the index of the first assistant message (start of tool turns)
        first_assistant = None
        for idx, m in enumerate(msgs):
            if m.get("role") == "assistant":
                first_assistant = idx
                break
        if first_assistant is None:
            return False  # nothing to compact

        # Find the last assistant message index — preserve the most recent turn
        last_assistant = None
        for idx in range(len(msgs) - 1, -1, -1):
            if msgs[idx].get("role") == "assistant":
                last_assistant = idx
                break

        # Find the latest Environment Update — always preserve it
        last_env_update = None
        for idx in range(len(msgs) - 1, -1, -1):
            if msgs[idx].get("role") == "user" and self._is_environment_update(msgs[idx]):
                last_env_update = idx
                break

        # Compact from oldest turns forward, skipping the latest turn and env update
        for idx in range(first_assistant, last_assistant or len(msgs)):
            if idx == last_env_update:
                continue  # preserve latest environment update
            role = msgs[idx].get("role")
            if role in ("tool", "assistant", "user"):
                content = msgs[idx].get("content", "")
                if isinstance(content, str) and len(content) > 100:
                    msgs[idx]["content"] = "[Truncated to save context]"
                elif isinstance(content, list):
                    msgs[idx]["content"] = "[Truncated to save context]"
                # Also strip tool_calls from compacted assistant messages
                if role == "assistant":
                    msgs[idx].pop("tool_calls", None)
                count = self._count_tokens(msgs, tools)
                if count is not None and count <= token_budget:
                    return True

        count = self._count_tokens(msgs, tools)
        return count is not None and count <= token_budget

    def _build_vanilla_fallback_msgs(self, msgs):
        """Build a minimal user-only message list for base-agent-style fallback.

        Used when context length is exceeded — drops all tool conversation
        and retries with just the original input and a direct-answer prompt.
        """
        fallback = []
        # Do not keep the tool-agent system prompt; fallback should behave like BaseAgent, we only keep the original user question.
        for m in msgs:
            if m.get("role") == "user":
                fallback.append(m)
                break
        return fallback

    def _ensure_context_fits(self, msgs, model_tools, tool_choice):
        """Check context length and compact/force final answer if needed.

        Returns (model_tools, tool_choice) — may be modified if the model is
        forced to answer directly.
        """
        max_ctx = getattr(self, "max_context_length", None)
        if not max_ctx:
            return model_tools, tool_choice
        tokenizer = getattr(self, "_tokenizer", None)
        if tokenizer is None:
            return model_tools, tool_choice

        token_budget = max_ctx - self.max_tokens - 512
        current_tokens = self._count_tokens(msgs, model_tools)
        if current_tokens is None or current_tokens <= token_budget:
            return model_tools, tool_choice

        self.logger.warning(
            f"[ToolLoop] Token count ({current_tokens}) exceeds budget "
            f"({token_budget}), compacting history..."
        )
        if self._compact_history(msgs, token_budget, model_tools):
            return model_tools, tool_choice

        # Still over limit — fall back to vanilla QA
        self.logger.warning(
            "[ToolLoop] Still over limit after compaction, falling back to vanilla QA"
        )
        fallback_msgs = self._build_vanilla_fallback_msgs(msgs)
        msgs.clear()
        msgs.extend(fallback_msgs)
        return None, "none"

    @staticmethod
    def _sanitize_msgs_for_template(msgs):
        """Replace image_url content items with [image] placeholder for chat template."""
        sanitized = cp.deepcopy(msgs)
        for msg in sanitized:
            content = msg.get("content")
            if isinstance(content, list):
                msg["content"] = [
                    {"type": "text", "text": "[image]"}
                    if (isinstance(item, dict) and item.get("type") == "image_url")
                    else item
                    for item in content
                ]
        return sanitized

    def _to_responses_tools(self) -> List[Dict[str, Any]]:
        """Flatten Chat-API tool schema to Responses-API shape."""
        out: List[Dict[str, Any]] = []
        for tool in self.json_schema or []:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function")
            if isinstance(fn, dict):
                out.append({
                    "type": "function",
                    "name": fn.get("name"),
                    "description": fn.get("description"),
                    "parameters": cp.deepcopy(fn.get("parameters") or {"type": "object"}),
                })
            else:
                out.append(cp.deepcopy(tool))
        return out

    @staticmethod
    def _content_to_responses_parts(content: Any, role: str) -> List[Dict[str, Any]]:
        """Translate Chat-API content (`{type:"text"|"image_url",...}`) to
        Responses-API content parts (`input_text` / `input_image` for system/user,
        `output_text` for assistant).
        """
        text_type = "output_text" if role == "assistant" else "input_text"
        if isinstance(content, str):
            return [{"type": text_type, "text": content}]
        if not isinstance(content, list):
            return [{"type": text_type, "text": str(content)}]
        parts: List[Dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            if t == "text":
                parts.append({"type": text_type, "text": item.get("text", "")})
            elif t == "image_url":
                url = item.get("image_url")
                if isinstance(url, dict):
                    url = url.get("url")
                if url:
                    parts.append({"type": "input_image", "image_url": url})
            elif t in {"input_text", "output_text", "input_image", "input_file"}:
                parts.append(cp.deepcopy(item))
        return parts

    @staticmethod
    def _extract_assistant_text(output_items: List[Dict[str, Any]]) -> str:
        """Concatenate output_text parts from assistant message items."""
        chunks: List[str] = []
        for item in output_items:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message" or item.get("role") != "assistant":
                continue
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") in {"output_text", "text"}:
                    text = part.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
        return "".join(chunks).strip()

    @staticmethod
    def _extract_function_calls(output_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        calls: List[Dict[str, Any]] = []
        for item in output_items:
            if isinstance(item, dict) and item.get("type") == "function_call":
                calls.append(item)
        return calls

    def _normalize_responses_tool_calls(
        self,
        function_call_items: List[Dict[str, Any]],
        turn_idx: int,
    ) -> List[Dict[str, Any]]:
        """Decode `arguments` JSON and produce the internal call envelope used
        by the rest of the agent. Mirrors `_normalize_tool_calls` but for the
        Responses-API item shape.
        """
        normalized: List[Dict[str, Any]] = []
        for idx, item in enumerate(function_call_items):
            call_id = str(item.get("call_id") or item.get("id") or f"call_{turn_idx}_{idx}")
            name = item.get("name") or ""
            raw_args = item.get("arguments")
            parse_error: Optional[Dict[str, Any]] = None
            args: Dict[str, Any] = {}
            if isinstance(raw_args, str):
                if raw_args.strip():
                    try:
                        decoded = json.loads(raw_args)
                        if isinstance(decoded, dict):
                            args = decoded
                        else:
                            parse_error = {
                                "code": "invalid_arguments_type",
                                "message": "function.arguments must decode to a JSON object.",
                            }
                    except Exception as exc:
                        parse_error = {
                            "code": "invalid_arguments_json",
                            "message": f"Failed to parse function.arguments JSON: {exc}",
                        }
            elif isinstance(raw_args, dict):
                args = raw_args
                raw_args = json.dumps(raw_args, ensure_ascii=False)
            normalized.append({
                "id": call_id,
                "name": name,
                "raw_arguments": raw_args if isinstance(raw_args, str) else json.dumps(args, ensure_ascii=False),
                "args": args,
                "parse_error": parse_error,
                # echo of the original function_call item, for re-emission as input
                "function_call_item": {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": name,
                    "arguments": raw_args if isinstance(raw_args, str) else json.dumps(args, ensure_ascii=False),
                },
            })
        return normalized

    def _build_function_call_output(
        self,
        *,
        call_id: str,
        normalized_payload: Dict[str, Any],
        deferred_message: Optional[Dict[str, Any]],
        run_memory: Any,
    ) -> Dict[str, Any]:
        """Build the `function_call_output` input item to append after a tool runs.

        The success path reuses the same per-tool `process_tool_output` pipeline
        the Chat-Completions path uses: `consume_tool_payload` already produced
        the fully-processed model-facing user message in
        `memory_result["deferred_messages"][0]`, which we translate from
        Chat-API content shape to Responses-API content shape.

        Shape:
        - Failure: `output` is the same error string `BaseAgentWithTools`'
          memory layer builds for `role="tool"` failure acks
          (`PersistentStateMemory._build_failure_tool_content`).
        - Success, text-only: `output` is the plain processed text.
        - Success with images: `output` is a mixed-content array
          `[{type:"input_text", text:..}, {type:"input_image", image_url:"data:..."}, ...]`.
        """
        # Failure path.
        if not normalized_payload.get("ok"):
            return {
                "type": "function_call_output",
                "call_id": call_id,
                "output": failure_text(normalized_payload),
            }

        # Success path: derive output from the per-tool processed message.
        content = deferred_message.get("content") if isinstance(deferred_message, dict) else None

        if isinstance(content, str):
            return {
                "type": "function_call_output",
                "call_id": call_id,
                "output": content,
            }

        if isinstance(content, list):
            text_chunks: List[str] = []
            image_urls: List[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    text = part.get("text")
                    if isinstance(text, str):
                        text_chunks.append(text)
                elif ptype == "image_url":
                    url = part.get("image_url")
                    if isinstance(url, dict):
                        url = url.get("url")
                    if isinstance(url, str) and url:
                        image_urls.append(url)
                elif ptype == "input_text":  # already Responses-shape, accept passthrough
                    text = part.get("text")
                    if isinstance(text, str):
                        text_chunks.append(text)
                elif ptype == "input_image":
                    url = part.get("image_url")
                    if isinstance(url, str) and url:
                        image_urls.append(url)

            joined_text = "\n".join(text_chunks).strip()

            if not image_urls:
                return {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": joined_text,
                }

            output_parts: List[Dict[str, Any]] = []
            if joined_text:
                output_parts.append({"type": "input_text", "text": joined_text})
            for url in image_urls:
                output_parts.append({"type": "input_image", "image_url": url})
            return {
                "type": "function_call_output",
                "call_id": call_id,
                "output": output_parts,
            }

        # Defensive fallback: deferred message missing or malformed — should not
        # happen on success but degrade gracefully to the raw result text.
        raw = normalized_payload.get("result")
        if not isinstance(raw, str):
            try:
                raw = json.dumps(raw, ensure_ascii=False, default=str)
            except Exception:
                raw = str(raw)
        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": raw,
        }
