"""OpenAI Responses-API tool-calling orchestrator.

`APIAgentWithTools` inherits tool registry, memory bookkeeping, image-ref
resolution, and local Python execution from `BaseAgentWithToolsAndCode`, and
overrides only the conversation transport: it uses `POST /v1/responses` with
`function_call` / `function_call_output` items in place of Chat Completions'
`assistant.tool_calls` + `role="tool"` messages.

Reasoning items returned by gpt-5 are not carried across turns in this first
cut (stateless mode). Switch to stateful via `previous_response_id` + the
`store=True` flag if multi-step reasoning quality becomes a concern.
"""

from __future__ import annotations

import copy as cp
import json
from typing import Any, Dict, List, Optional, Tuple

from .base_with_code import BaseAgentWithToolsAndCode
from ..utils.api_utils import (
    responses_create_request,
    tool_execute_request,
    filter_kwargs_for_endpoint,
)


class APIAgentWithTools(BaseAgentWithToolsAndCode):
    """OpenAI Responses-API tool-calling agent."""

    # ------------------------------------------------------------------ #
    # Tool schema + input/content adaptation
    # ------------------------------------------------------------------ #
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

    def _build_initial_input(self, question: Any, system_prompt: Optional[str]) -> List[Dict[str, Any]]:
        """Convert a question into Responses-API input items."""
        chat_msgs = self.prepare_inputs(question, system_prompt=system_prompt)
        chat_msgs = self._add_user_prompt_prefix(chat_msgs)
        items: List[Dict[str, Any]] = []
        for msg in chat_msgs:
            role = msg.get("role")
            if role not in {"system", "user", "assistant"}:
                continue
            content = msg.get("content")
            parts = self._content_to_responses_parts(content, role)
            if not parts:
                continue
            items.append({"type": "message", "role": role, "content": parts})
        return items

    # ------------------------------------------------------------------ #
    # Response parsing
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Tool result envelope
    # ------------------------------------------------------------------ #
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
                "output": run_memory._build_failure_tool_content(normalized_payload),
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

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def _run(self, question, return_messages: bool = False, **kwargs) -> tuple:
        """Responses-API tool-calling loop.

        Flow per turn: POST /v1/responses → walk `response.output` →
        for each function_call: execute via tool server, append function_call
        item + function_call_output item to `input` →
        when assistant message has content and no function calls, finalize.
        """
        debug_tool_loop = bool(kwargs.pop("debug_tool_loop", False))
        debug_char_limit = int(kwargs.pop("debug_char_limit", 280))
        timeout_seconds = kwargs.pop("timeout_seconds", self.timeout)
        tool_choice = kwargs.pop("tool_choice", self.tool_choice)
        active_system_prompt = kwargs.pop("system_prompt", self.system_prompt)
        task_id = kwargs.pop("task_id", None)
        # Strip Chat-only / vLLM-only kwargs.
        kwargs.pop("dataset", None)
        kwargs.pop("repetition_penalty", None)
        kwargs.pop("return_token_ids", None)
        kwargs.pop("stop", None)
        max_tokens = kwargs.pop("max_tokens", getattr(self, "max_tokens", None))
        reasoning_effort = kwargs.pop("reasoning_effort", getattr(self, "reasoning_effort", None))
        temperature = kwargs.pop("temperature", None)  # noqa: F841 — Responses API ignores for gpt-5

        reasoning = {"effort": reasoning_effort} if reasoning_effort else None

        run_memory = self._new_persistent_state_memory()
        run_memory.reset()
        if hasattr(run_memory, "reset_for_row"):
            run_memory.reset_for_row(str(task_id) if task_id is not None else None)

        input_items: List[Dict[str, Any]] = self._build_initial_input(question, active_system_prompt)

        # Run start hook: any module-emitted messages are appended as plain
        # Chat-API style; transform them too.
        start_msgs = run_memory.on_run_start(agent=self, question=question, messages=input_items)
        if isinstance(start_msgs, list):
            for msg in start_msgs:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role")
                if role not in {"system", "user", "assistant"}:
                    continue
                parts = self._content_to_responses_parts(msg.get("content"), role)
                if parts:
                    input_items.append({"type": "message", "role": role, "content": parts})

        responses_tools = self._to_responses_tools()
        chat_history: List[Dict[str, Any]] = []
        final_answer: Optional[str] = None
        terminating_response: Optional[Dict[str, Any]] = None

        for i in range(self.max_iterations):
            if debug_tool_loop:
                self.logger.info(f"[ToolLoop] turn={i + 1}/{self.max_iterations}, items={len(input_items)}")

            if i == self.max_iterations - 1:
                input_items.append({
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Please provide your best final answer now if enough information is available."}],
                })

            response = responses_create_request(
                model=self.model,
                input=input_items,
                base_url=self.base,
                api_key=self.key,
                tools=responses_tools,
                tool_choice=tool_choice,
                max_output_tokens=max_tokens,
                reasoning=reasoning,
                store=False,
                timeout_seconds=timeout_seconds,
                extra_body=kwargs.get("extra_body"),
            )

            if not isinstance(response, dict):
                raise RuntimeError(f"Invalid responses payload type: {type(response).__name__}")
            output_items = response.get("output") or []
            if not isinstance(output_items, list):
                raise RuntimeError(f"Invalid responses payload: 'output' must be a list. {self._truncate_for_debug(response)}")

            assistant_text = self._extract_assistant_text(output_items)
            function_calls_raw = self._extract_function_calls(output_items)
            normalized_calls = self._normalize_responses_tool_calls(function_calls_raw, turn_idx=i)

            turn_info: Dict[str, Any] = {
                "turn": i + 1,
                "assistant": {
                    "content": assistant_text,
                    "tool_calls": [
                        {
                            "id": c["id"],
                            "name": c["name"],
                            "raw_arguments": c["raw_arguments"],
                            "args": c["args"],
                            "parse_error": c.get("parse_error"),
                        }
                        for c in normalized_calls
                    ],
                },
                "tool_responses": [],
                "memory_outputs": [],
            }

            if normalized_calls:
                # Multi-call policy: respond to every call_id with an error, then continue.
                if len(normalized_calls) > 1:
                    if debug_tool_loop:
                        self.logger.info(f"[ToolLoop] enforcing single-tool-per-turn: rejecting {len(normalized_calls)} calls")
                    for call in normalized_calls:
                        input_items.append(call["function_call_item"])
                    err_payload_template = {
                        "ok": False,
                        "tool_name": None,
                        "result": None,
                        "error": {
                            "code": "multi_tool_call_rejected",
                            "message": "Only one tool call is allowed per turn. Please re-issue exactly one.",
                        },
                        "tool_call_id": None,
                        "meta": {"source": "agent.single_tool_policy"},
                    }
                    for call in normalized_calls:
                        err_payload = cp.deepcopy(err_payload_template)
                        err_payload["tool_name"] = call["name"]
                        err_payload["tool_call_id"] = call["id"]
                        turn_info["tool_responses"].append(err_payload)
                        input_items.append(self._build_function_call_output(
                            call_id=call["id"], normalized_payload=err_payload, deferred_message=None,
                            run_memory=run_memory,
                        ))
                    chat_history.append(turn_info)
                    continue

                # Single call path.
                call = normalized_calls[0]
                # Re-emit the assistant's function_call as input item for the next turn.
                input_items.append(call["function_call_item"])
                # Optionally re-emit the assistant message text (rare but possible).
                if assistant_text:
                    input_items.append({
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": assistant_text}],
                    })

                parse_error = call.get("parse_error")
                if parse_error:
                    normalized_payload = {
                        "ok": False,
                        "tool_name": call["name"],
                        "result": None,
                        "error": parse_error,
                        "tool_call_id": call["id"],
                        "meta": {"source": "agent.tool_call_parser"},
                    }
                    deferred_message = None
                else:
                    resolved_args, image_resolution = run_memory.resolve_tool_call_args(
                        agent=self, args=call["args"], tool_name=call.get("name", ""),
                    )
                    if debug_tool_loop:
                        self.logger.info(
                            f"[ToolLoop] function_call call_id={call['id']} name={call['name']} "
                            f"args={self._truncate_for_debug(resolved_args, max_chars=debug_char_limit)}"
                        )
                    if image_resolution.get("errors"):
                        normalized_payload = {
                            "ok": False,
                            "tool_name": call["name"],
                            "result": None,
                            "error": {
                                "code": "unresolved_image_reference",
                                "message": "Failed to map one or more image_id arguments to runtime images.",
                                "details": image_resolution.get("errors", []),
                            },
                            "tool_call_id": call["id"],
                            "meta": {"image_resolution": image_resolution},
                        }
                    else:
                        try:
                            payload = tool_execute_request(
                                base_url=self.tool_base,
                                tool_call={
                                    "tool_name": call["name"],
                                    "args": resolved_args,
                                    "tool_call_id": call["id"],
                                },
                                timeout=self.timeout,
                            )
                            normalized_payload = self._normalize_execute_payload(
                                payload, tool_name=call["name"], tool_call_id=call["id"],
                            )
                        except Exception as exc:
                            normalized_payload = {
                                "ok": False,
                                "tool_name": call["name"],
                                "result": None,
                                "error": {"code": "tool_execution_error", "message": str(exc)},
                                "tool_call_id": call["id"],
                                "meta": {},
                            }
                        if image_resolution.get("rewrites"):
                            meta = normalized_payload.get("meta") if isinstance(normalized_payload.get("meta"), dict) else {}
                            meta["image_resolution"] = image_resolution
                            normalized_payload["meta"] = meta
                        normalized_payload = self._post_process_tool_payload(
                            normalized_payload, call, run_memory,
                        )

                    memory_result = run_memory.consume_tool_payload(
                        agent=self, turn_idx=i, tool_call=call, payload=normalized_payload,
                    )
                    deferred_messages = memory_result.get("deferred_messages") if isinstance(memory_result, dict) else None
                    deferred_message = (
                        deferred_messages[0]
                        if isinstance(deferred_messages, list) and deferred_messages
                        else None
                    )

                fco = self._build_function_call_output(
                    call_id=call["id"], normalized_payload=normalized_payload, deferred_message=deferred_message,
                    run_memory=run_memory,
                )
                if debug_tool_loop:
                    out = fco.get("output")
                    image_part_count = (
                        sum(1 for p in out if isinstance(p, dict) and p.get("type") == "input_image")
                        if isinstance(out, list) else 0
                    )
                    self.logger.info(
                        f"[ToolLoop] function_call_output call_id={call['id']} ok={normalized_payload.get('ok')} "
                        f"image_parts={image_part_count} "
                        f"error={self._truncate_for_debug(normalized_payload.get('error'), max_chars=debug_char_limit)}"
                    )

                turn_info["tool_responses"].append(normalized_payload)
                input_items.append(fco)

                # Append env-update user messages from memory modules, transformed.
                env_msgs = run_memory.emit_after_tool_turn(
                    agent=self, turn_idx=i, tool_calls=normalized_calls,
                    tool_payloads=turn_info["tool_responses"], messages=input_items,
                )
                if isinstance(env_msgs, list):
                    for msg in env_msgs:
                        if not isinstance(msg, dict):
                            continue
                        role = msg.get("role")
                        if role not in {"system", "user", "assistant"}:
                            continue
                        parts = self._content_to_responses_parts(msg.get("content"), role)
                        if parts:
                            input_items.append({"type": "message", "role": role, "content": parts})

                chat_history.append(turn_info)
                continue

            # No function calls — model produced a final assistant message.
            final_answer = self._extract_final_answer_from_content(assistant_text) or assistant_text
            turn_info["final_answer"] = final_answer
            chat_history.append(turn_info)
            if debug_tool_loop:
                self.logger.info(
                    f"[ToolLoop] final_answer={self._truncate_for_debug(final_answer, max_chars=debug_char_limit)}"
                )
            if assistant_text:
                input_items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": assistant_text}],
                })
            terminating_response = response
            self._last_tool_detections = list(getattr(run_memory, "tool_detections", []) or [])
            if return_messages:
                return final_answer or self.fail_msg, chat_history, input_items, None
            return final_answer or self.fail_msg, chat_history

        # Out of iterations.
        self._last_tool_detections = list(getattr(run_memory, "tool_detections", []) or [])
        if return_messages:
            return self.fail_msg, chat_history, input_items, None
        return self.fail_msg, chat_history
