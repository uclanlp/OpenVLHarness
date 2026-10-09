"""Registered model profiles, native API encoders and context policy."""
import copy as cp
import difflib
import json
import logging
import re
from importlib.resources import files
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
import requests
import yaml
from .http import retry_with_exponential_backoff, raise_for_status_with_detail
from .raw_trace import RawTraceExportError, build_raw_trace_from_tokens


_CONTEXT_TOKEN_MARGIN = 512


def failure_text(payload):
    """Render an execute envelope's error code and message as model-visible text."""
    error = payload.get("error")
    if isinstance(error, dict):
        code = str(error["code"]).strip() if error.get("code") is not None else ""
        message = str(error["message"]).strip() if error.get("message") is not None else ""
        message = message or "Tool call failed."
        return f"{code}: {message}" if code else message
    return str(error).strip() if error not in (None, "", {}) and str(error).strip() else "Tool call failed."


def load_profile(alias):
    """Return an independent copy of a registered model profile.

    Args:
        alias: Exact alias in the packaged model_registry.yaml.

    Raises:
        ValueError: The alias is not registered.
    """
    profiles = yaml.safe_load(files("openvlharness.services").joinpath("model_registry.yaml").read_text())["profiles"]
    if alias not in profiles:
        message = f"Unknown model alias {alias!r}"
        served = [name for name, profile in profiles.items() if profile.get("model_id") == alias]
        close = difflib.get_close_matches(alias, profiles, n=3) if isinstance(alias, str) else []
        if served:
            message += f": this is a served model ID; use alias {served[0]!r}."
        elif close:
            message += f". Did you mean: {', '.join(map(repr, close))}?"
        else:
            message += "."
        raise ValueError(f"{message} See docs/models.md#other-registered-models. Register a new model with its "
                         "exact served ID, prompt and API profile in model_registry.yaml.")
    return cp.deepcopy(profiles[alias])


def require_raw_trace_profile(alias):
    """Return the supported Qwen profile after checking its export requirements.

    Raises:
        ValueError: The alias is unsupported or its model, protocol, format or
            pinned tokenizer revision does not meet raw trace requirements.
    """
    message = "Raw trace export is supported only for qwen3-vl-8b-instruct (Qwen/Qwen3-VL-8B-Instruct); GPT, Kimi and other profiles are unsupported."
    try:
        profile = load_profile(alias)
    except ValueError as exc:
        raise ValueError(message) from exc
    if (alias != "qwen3-vl-8b-instruct" or profile.get("model_id") != "Qwen/Qwen3-VL-8B-Instruct"
            or profile.get("adapter") != "chat_completions" or profile.get("raw_trace_format") != "qwen_chatml"):
        raise ValueError(message)
    if not isinstance(profile.get("raw_trace_tokenizer_revision"), str) or not profile["raw_trace_tokenizer_revision"]:
        raise ValueError("Raw trace export requires a registered matching Qwen tokenizer revision.")
    return profile


class AuxiliaryModel:
    """Run search summarization and evidence extraction through a configured model.

    Uses the model profile's native protocol with task-specific generation
    defaults. The OpenAI SDK manages HTTP connections and retry behavior.
    """
    def __init__(self, config):
        """Validate model settings and create an SDK client with two retries by default."""
        self.adapter = ModelAdapter({"retry": 2, **config})
        self.model = self.adapter.model
        self.base_url = self.adapter.base
        self.api_key = self.adapter.key
        # The auxiliary SDK client defaults to two retries for transient failures.
        # Protocol encoding and payload options come from the registered adapter.
        from openai import OpenAI, Timeout
        base = self.base_url.rstrip("/")
        for suffix in ("/chat/completions", "/responses"):
            if base.endswith(suffix):
                base = base[:-len(suffix)]
        if not base.endswith("/v1"):
            base += "/v1"
        self.client = OpenAI(base_url=base, api_key=self.api_key or "empty",
                             timeout=Timeout(self.adapter.timeout, connect=5), max_retries=self.adapter.retry)

    def generate(self, prompt_or_messages, system_prompt=None, **generation_kwargs):
        """Generate auxiliary text from a prompt or chat-style messages.

        Args:
            prompt_or_messages: Prompt string or list of message dictionaries.
            system_prompt: System instruction inserted only if none is present.
            **generation_kwargs: max_tokens, temperature and top_p. Defaults are
                8192, 0.0 and 1.0. Responses uses medium reasoning; Chat Completions
                applies sampling options and disables template thinking.

        Returns:
            Assistant text extracted from the provider response.

        Raises:
            ValueError: Unsupported generation options are supplied.
            RuntimeError: The response has an invalid protocol structure.
        """
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
    """Translate model requests and results using a registered native API profile.

    The adapter owns transport options, tool-call parsing, context handling and
    Qwen raw trace reconstruction. Profile configuration determines whether the
    provider uses Chat Completions or Responses.
    """
    def __init__(self, config, *, max_context_length=None, requester=None):
        """Validate model configuration and resolve its endpoint without sending requests.

        Args:
            config: Mapping with alias, base, optional key/timeout/retry and profile-
                allowed generation options. max_tokens defaults to 4096.
            max_context_length: Optional local context budget for tokenizer-based
                profiles, including reserved output tokens. Must exceed max_tokens
                plus the 512-token margin to leave room for input tokens.
            requester: Optional callable accepting a native payload and returning
                decoded provider data; bypasses the built-in HTTP retry transport.

        Raises:
            ValueError: The profile, endpoint, option values, context budget or raw
                export request is invalid or unsupported.
        """
        self.profile = (require_raw_trace_profile(config.get("alias")) if config.get("return_token_ids") is True
                        else load_profile(config.get("alias")))
        connection = {"alias", "base", "key", "timeout", "retry"}
        unknown = set(config) - connection - set(self.profile["allowed_options"])
        if unknown:
            raise ValueError(f"Unsupported options for {config.get('alias')}: {sorted(unknown)}")
        self.model = self.profile["model_id"]
        self.protocol = self.profile["adapter"]
        self.base = config.get("base", "")
        parsed = urlparse(self.base)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment or parsed.username is not None:
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
        self.options.setdefault("max_tokens", 4096)
        for key in ("max_tokens", "top_k"):
            if key in self.options and (type(self.options[key]) is not int or self.options[key] <= 0):
                raise ValueError(f"{key} must be a positive integer")
        # Union of provider effort levels; each model accepts a subset and rejects the rest.
        if self.options.get("reasoning_effort", "medium") not in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("Unsupported reasoning_effort")
        if type(self.options.get("return_token_ids", False)) is not bool:
            raise ValueError("return_token_ids must be boolean")
        self.raw_trace_enabled = self.options.get("return_token_ids", False)
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
        self.max_tokens = self.options["max_tokens"]
        if max_context_length is not None and max_context_length <= self.max_tokens + _CONTEXT_TOKEN_MARGIN:
            raise ValueError(
                f"max_context_length ({max_context_length}) must be greater than "
                f"max_tokens ({self.max_tokens}) + {_CONTEXT_TOKEN_MARGIN} to leave room for input tokens; "
                "increase max_context_length or lower max_tokens"
            )
        self.max_context_length = max_context_length
        self.logger = logging.getLogger("openvlharness.models")
        self._tokenizer = None
        self.requester = requester
        self.json_schema = []
        self.counting_available = False

    def load_tokenizer(self):
        """Load the profile's tokenizer when context counting or raw export needs it.

        Checks the local cache before allowing a download. Raw export pins the
        registered revision and raises RawTraceExportError if loading fails. Without
        export, a loading failure is logged and context enforcement stays with the
        provider. If the chat template cannot be rendered, for example without
        jinja2, counting is unavailable and a warning is logged; the tokenizer stays
        loaded because raw export does not use the template.
        """
        counting = self.profile["context_policy"] == "qwen_tokenizer" and self.max_context_length
        if self.raw_trace_enabled or counting:
            try:
                from transformers import AutoTokenizer
                options = ({"revision": self.profile["raw_trace_tokenizer_revision"]} if self.raw_trace_enabled else {})
                try:
                    self._tokenizer = AutoTokenizer.from_pretrained(self.model, local_files_only=True, **options)
                except OSError:
                    self._tokenizer = AutoTokenizer.from_pretrained(self.model, **options)
            except (ImportError, OSError, ValueError) as exc:
                if self.raw_trace_enabled:
                    raise RawTraceExportError("tokenizer_unavailable", f"Raw trace export requires the matching tokenizer for {self.model}; install openvlharness[qwen] and cache the registered revision.") from exc
                self.logger.warning("Tokenizer unavailable; provider enforces context limit: %s", exc)
                return
            self.counting_available = self._count_tokens([{"role": "user", "content": "x"}]) is not None
            if counting and not self.counting_available:
                self.logger.warning("Qwen chat template could not be rendered (is jinja2 installed?); "
                                    "provider enforces context limit")

    def export_raw_trace(self, capture, task_id):
        """Reconstruct a Qwen trace from a captured response and its exact request.

        Args:
            capture: Mapping containing response and messages from request capture.
            task_id: Sample identifier used to name image attachments.

        Returns:
            A mapping of reconstructed messages and base64 image attachments.

        Raises:
            RawTraceExportError: Token IDs, tokenizer, ChatML or image alignment are invalid.
        """
        return build_raw_trace_from_tokens(capture.get("response"), capture.get("messages"), self._tokenizer, task_id)

    def encode_message(self, message):
        """Copy a chat-style message into the configured provider's native representation."""
        if self.protocol == "chat_completions":
            return cp.deepcopy(message)
        return {"type": "message", "role": message["role"],
                "content": self._content_to_responses_parts(message.get("content"), message["role"])}

    def payload(self, messages, tools=None, tool_choice="auto", *, options=None, seed=None):
        """Build a native request payload without mutating messages or tool schemas.

        Args:
            messages: Conversation items already encoded for this adapter's protocol.
            tools: Optional function-tool schemas in Chat Completions form.
            tool_choice: Tool-selection mode when schemas are supplied.
            options: Per-request generation options, replacing configured options
                except that an omitted max_tokens inherits the adapter's budget.
            seed: Optional sampling seed, used only by Chat Completions.

        Returns:
            A provider-native JSON mapping. Responses requests disable storage and
            translate output limits, reasoning options and function schemas.
        """
        options = cp.deepcopy(self.options if options is None else options)
        options.setdefault("max_tokens", self.max_tokens)
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

    def request(self, messages, tools=None, tool_choice="auto", *, options=None, seed=None, capture=None):
        """Send a native request and optionally capture its exact messages and response.

        Args:
            messages: Provider-native conversation items.
            tools: Optional Chat Completions-style function schemas.
            tool_choice: Provider tool-selection mode.
            options: Generation overrides accepted by payload.
            seed: Optional Chat Completions sampling seed.
            capture: Mutable mapping cleared before the attempt and populated on
                success with deep copies of response and request messages. Intended
                for Qwen raw trace export.

        Returns:
            Decoded provider data; parse performs protocol validation separately.

        Raises:
            RuntimeError: The provider returns an HTTP error after applicable retries.
            requests.RequestException: A transport failure exhausts retries.
        """
        if capture is not None:
            capture.clear()
        payload = self.payload(messages, tools, tool_choice, options=options, seed=seed)
        if self.requester:
            response = self.requester(payload)
        else:
            def once():
                headers = {"Content-Type": "application/json"}
                if self.key and str(self.key).lower() not in {"empty", "none"}:
                    headers["Authorization"] = f"Bearer {self.key}"
                result = requests.post(self.endpoint, json=payload, headers=headers, timeout=self.timeout)
                raise_for_status_with_detail(result, prefix=f"Model request failed at endpoint {self.endpoint}")
                return result.json()
            response = retry_with_exponential_backoff(once, max_retries=self.retry)()
        if capture is not None:
            capture.update(response=cp.deepcopy(response), messages=cp.deepcopy(payload.get("messages")))
        return response

    def parse(self, response, turn_idx, *, phase="main"):
        """Validate provider data and normalize assistant text and tool calls.

        Args:
            response: Decoded Chat Completions or Responses payload.
            turn_idx: Zero-based turn index for generated call IDs and diagnostics.
            phase: Request phase label used in multi-call diagnostics.

        Returns:
            A (text, calls, original_message) tuple. original_message is present for
            Chat Completions only. Invalid tool arguments and multiple calls become
            per-call parse_error records so the agent can return structured feedback.

        Raises:
            RuntimeError: The response is not an object or lacks required protocol fields.
        """
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
        output = response.get("output")
        if not isinstance(output, list):
            raise RuntimeError("Invalid responses payload: output must be a list")
        calls = self._normalize_responses_tool_calls(self._extract_function_calls(output), turn_idx)
        if len(calls) > 1:
            for call in calls:
                call["parse_error"] = {"code": "multi_tool_call_rejected",
                    "message": "Only one tool call is allowed per turn. Please re-issue exactly one."}
        return self._extract_assistant_text(output), calls, None

    def append_assistant_calls(self, messages, text, calls, original):
        """Append native assistant tool-call items to history, preserving call IDs."""
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
        """Render a capability outcome into immediate and deferred message lists.

        Chat Completions receives a tool acknowledgment followed by deferred user
        evidence on success. Responses receives a function_call_output containing
        the result evidence or structured failure.
        """
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
        """Extract answer-tag content, falling back to full text only when the profile allows."""
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        tagged = match.group(1).strip() if match else ""
        return tagged or text if self.profile["final_answer"] == "answer_tag_or_text" else tagged

    def generate_text(self, messages, *, max_tokens=4096):
        """Generate text without tools from chat-style messages.

        Uses temperature zero for Chat Completions and medium reasoning for
        Responses. max_tokens limits the output; provider failures propagate.
        """
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
            # transformers 5 returns a BatchEncoding unless return_dict=False.
            tmpl_kwargs = dict(tokenize=True, add_generation_prompt=True, return_dict=False)
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
        - Tool-call declarations and result IDs, keeping each exchange valid

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
                # Retain call declarations so following tool results keep their pairing.
                count = self._count_tokens(msgs, tools)
                if count is not None and count <= token_budget:
                    return True

        count = self._count_tokens(msgs, tools)
        return count is not None and count <= token_budget

    def _build_vanilla_fallback_msgs(self, msgs):
        """Return the first user message for a direct-answer context fallback.

        Drops system instructions and tool conversation. The returned list references
        the selected message from msgs and contains no added prompt text.
        """
        fallback = []
        # Use only the initial user question for direct-answer fallback.
        for m in msgs:
            if m.get("role") == "user":
                fallback.append(m)
                break
        return fallback

    def _ensure_context_fits(self, msgs, model_tools, tool_choice):
        """Compact history in place and return the effective tools and tool choice.

        Reserves max_tokens plus 512 tokens. If compaction cannot meet the budget,
        replaces history with the first user message and returns (None, "none").
        Missing tokenizer counts leave provider-side context enforcement in effect.
        """
        max_ctx = getattr(self, "max_context_length", None)
        if not max_ctx:
            return model_tools, tool_choice
        tokenizer = getattr(self, "_tokenizer", None)
        if tokenizer is None:
            return model_tools, tool_choice

        token_budget = max_ctx - self.max_tokens - _CONTEXT_TOKEN_MARGIN
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
        """Select function_call items from a Responses output list."""
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
        """Build a Responses function_call_output from prepared capability evidence.

        Failure output is rendered error text. Successful text-only evidence remains
        text; image evidence becomes input_text/input_image parts. A missing deferred
        message falls back to serializing the raw result. run_memory is accepted but
        unused; artifacts are already registered by the capability.
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
