"""Two-step RAG agent: forced image search → forced text search → final answer.

This agent enforces a rigid retrieval pipeline:
  Step 1: Image search (e.g. Google Lens) — model sees image + question,
          tool is forced, results are appended to context.
  Step 2: Text search (e.g. Gemini Search) — model sees image + question +
          image search results, generates a text query, tool is forced,
          results are appended to context.
  Step 3: Final answer — model generates answer from all accumulated context,
          no tools available.

The image and text search tools are configurable via constructor args
(or YAML config), making it easy to swap in different search backends.
"""

import copy as cp
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseAgentWithTools, get_logger
from .constant import DEFAULT_RAG_SYSTEM_PROMPT, DEFAULT_RAG_USER_PROMPT_PREFIX
from ..utils.api_utils import chat_completion_request, tool_execute_request


class BaseAgentWithRAG(BaseAgentWithTools):

    def __init__(
        self,
        image_search_tool: str = "SerpApi_Google_Lens_Tool",
        text_search_tool: str = "Gemini_Google_Search_Tool",
        **kwargs,
    ):
        # Ensure tool_list includes both RAG tools so they get loaded.
        tool_list = kwargs.get("tool_list")
        if tool_list is not None:
            for tool_name in [image_search_tool, text_search_tool]:
                if tool_name not in tool_list:
                    tool_list.append(tool_name)
            kwargs["tool_list"] = tool_list

        super().__init__(**kwargs)
        self.image_search_tool = image_search_tool
        self.text_search_tool = text_search_tool
        self.logger = get_logger("BaseAgentWithRAG")
        self.logger.info(
            f"RAG pipeline: step1={self.image_search_tool}, step2={self.text_search_tool}"
        )

        # Build per-tool schema subsets for forced tool_choice.
        self._image_search_schema = self._filter_schema(image_search_tool)
        self._text_search_schema = self._filter_schema(text_search_tool)

        # Override system prompt with RAG-specific version.
        self.system_prompt = self._construct_system_message()

    def _get_prompt_templates(self) -> tuple:
        return DEFAULT_RAG_SYSTEM_PROMPT, DEFAULT_RAG_USER_PROMPT_PREFIX

    def _construct_system_message(self) -> str:
        system_prompt = DEFAULT_RAG_SYSTEM_PROMPT
        if getattr(self, "experience_file", None) and os.path.exists(self.experience_file):
            from ..experience.reflector import load_experience
            experience_text = load_experience(self.experience_file)
            if experience_text.strip():
                system_prompt += (
                    "\n\nAdditional learned tool-usage principles from prior runs:\n"
                    "- Treat these as reusable heuristics when choosing tools, arguments, and retries.\n"
                    "- Follow them when relevant, but do not violate tool protocols above.\n"
                    "<learned_principles>\n"
                    f"{experience_text}\n"
                    "</learned_principles>"
                )
                self.logger.info(f"Loaded experience from {self.experience_file}")
        return system_prompt

    def _filter_schema(self, tool_name: str) -> List[Dict[str, Any]]:
        """Return the json_schema entry for a single tool."""
        for entry in self.json_schema:
            if (
                isinstance(entry, dict)
                and isinstance(entry.get("function"), dict)
                and entry["function"].get("name") == tool_name
            ):
                return [entry]
        self.logger.warning(f"Tool schema not found for '{tool_name}'")
        return []

    @staticmethod
    def _force_tool_choice(tool_name: str) -> Dict[str, Any]:
        """Build an OpenAI-style tool_choice that forces a specific tool."""
        return {"type": "function", "function": {"name": tool_name}}

    @staticmethod
    def _find_image_param(tool_schema: List[Dict[str, Any]]) -> Optional[str]:
        """Find the image input parameter name from a tool schema (e.g. 'image_path')."""
        if not tool_schema:
            return None
        props = tool_schema[0].get("function", {}).get("parameters", {}).get("properties", {})
        for name in props:
            if "image" in name.lower() and "path" in name.lower():
                return name
        return None

    def _format_tool_schema_for_prompt(self, tool_schema: List[Dict[str, Any]]) -> str:
        """Format a tool's JSON schema into a human-readable prompt section.

        Fully generic — derived from the schema, no hardcoded param names.
        """
        if not tool_schema:
            return ""
        entry = tool_schema[0]
        fn = entry.get("function", {})
        name = fn.get("name", "unknown")
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        properties = params.get("properties", {})
        required = params.get("required", [])

        lines = [f"Tool: {name}", f"Description: {desc}", "Parameters:"]
        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "any")
            param_desc = param_info.get("description", "")
            req_tag = " (required)" if param_name in required else ""
            lines.append(f"  - {param_name} ({param_type}{req_tag}): {param_desc}")

        one_of = params.get("oneOf")
        if one_of:
            options = [" or ".join(opt.get("required", [])) for opt in one_of if opt.get("required")]
            if options:
                lines.append(f"  Provide exactly one of: {', '.join(options)}")

        # Auto-generate example from schema.
        lines.append("")
        lines.append("Arguments must be a JSON object. Example:")
        example = {}
        if one_of:
            # Use the first option's required field as the example key.
            first_required = one_of[0].get("required", [])
            if first_required:
                key = first_required[0]
                example[key] = f"<{key}>"
        if not example and required:
            for key in required:
                example[key] = f"<{key}>"
        if not example and properties:
            key = next(iter(properties))
            example[key] = f"<{key}>"
        lines.append(f"  {json.dumps(example)}")

        return "\n".join(lines)

    def _build_step_instruction(
        self,
        step_name: str,
        step_number: int,
        tool_name: str,
        tool_schema: List[Dict[str, Any]],
        has_image: bool = True,
    ) -> str:
        """Build user-message instruction for a RAG step with tool schema."""
        schema_text = self._format_tool_schema_for_prompt(tool_schema)
        header = f"[RAG Step {step_number}: {tool_name}]"

        if step_name == "image_search":
            # Determine the image path param name from schema.
            image_param = self._find_image_param(tool_schema)
            image_id_hint = ""
            if image_param:
                image_id_hint = (
                    f'\nThe input image is available as a runtime image ID. '
                    f'Pass it via the "{image_param}" parameter '
                    f'(e.g. {{"{image_param}": "input_image_1"}}).'
                )
            return (
                f"{header}\n"
                f"Call the tool below to identify what is depicted in the input image.{image_id_hint}\n"
                f"You MUST call this tool now. Do NOT output text — only produce a tool call.\n\n"
                f"{schema_text}"
            )
        elif step_name == "text_search":
            if has_image:
                context_hint = "Based on the image and any search results above, formulate"
            else:
                context_hint = "Based on the question, formulate"
            return (
                f"{header}\n"
                f"{context_hint} a specific text search query "
                f"to find information that answers the original question.\n"
                f"You MUST call this tool now. Do NOT output text — only produce a tool call.\n\n"
                f"{schema_text}"
            )
        return f"{header}\nCall this tool now.\n\n{schema_text}"

    def _execute_tool_call(
        self,
        call: Dict[str, Any],
        run_memory: Any,
        msgs: List[Dict[str, Any]],
        turn_idx: int,
        debug: bool = False,
        debug_char_limit: int = 280,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Execute a single tool call and append results to msgs.

        Returns (normalized_payload, memory_output).
        """
        image_resolution = {"rewrites": [], "errors": []}
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
        else:
            resolved_args, image_resolution = run_memory.resolve_tool_call_args(
                agent=self, args=call["args"],
            )
            if debug:
                self.logger.info(
                    f"[RAG] execute {call['name']} id={call['id']} "
                    f"args={self._truncate_for_debug(resolved_args, max_chars=debug_char_limit)}"
                )

            execute_request = {
                "tool_name": call["name"],
                "args": resolved_args,
                "tool_call_id": call["id"],
            }

            if image_resolution.get("errors"):
                normalized_payload = {
                    "ok": False,
                    "tool_name": call["name"],
                    "result": None,
                    "error": {
                        "code": "unresolved_image_reference",
                        "message": "Failed to map one or more image_path arguments to runtime images.",
                        "details": image_resolution.get("errors", []),
                    },
                    "tool_call_id": call["id"],
                    "meta": {"image_resolution": image_resolution},
                }
            else:
                try:
                    payload = tool_execute_request(
                        base_url=self.tool_base,
                        tool_call=execute_request,
                        timeout=self.timeout,
                    )
                    normalized_payload = self._normalize_execute_payload(
                        payload, tool_name=call["name"], tool_call_id=call["id"],
                    )
                except Exception as e:
                    normalized_payload = {
                        "ok": False,
                        "tool_name": call["name"],
                        "result": None,
                        "error": {"code": "tool_execution_error", "message": str(e)},
                        "tool_call_id": call["id"],
                        "meta": {},
                    }
                if image_resolution.get("rewrites"):
                    meta = normalized_payload.get("meta")
                    if not isinstance(meta, dict):
                        meta = {}
                    meta["image_resolution"] = image_resolution
                    normalized_payload["meta"] = meta

        if debug:
            self.logger.info(
                f"[RAG] result ok={normalized_payload.get('ok')} id={call['id']}"
            )

        # Let memory process the payload and emit messages.
        memory_result = run_memory.consume_tool_payload(
            agent=self, turn_idx=turn_idx, tool_call=call, payload=normalized_payload,
        )
        immediate_messages = memory_result.get("immediate_messages", [])
        deferred_messages = memory_result.get("deferred_messages", [])

        immediate_indices = self._append_messages_with_indices(msgs, immediate_messages)
        deferred_indices = self._append_messages_with_indices(msgs, deferred_messages)

        memory_output = {
            "tool_call_id": call["id"],
            "tool_name": call["name"],
            "memory_name": run_memory.memory_name,
            "immediate_message_indices": immediate_indices,
            "deferred_message_indices": deferred_indices,
            "message_summaries": (
                [self._summarize_message_for_trace(msgs[idx]) for idx in immediate_indices]
                + [self._summarize_message_for_trace(msgs[idx]) for idx in deferred_indices]
            ),
        }

        return normalized_payload, memory_output

    def _rag_step(
        self,
        *,
        step_name: str,
        step_number: int,
        tool_name: str,
        tool_schema: List[Dict[str, Any]],
        msgs: List[Dict[str, Any]],
        run_memory: Any,
        turn_idx: int,
        chat_history: List[Dict[str, Any]],
        has_image: bool = True,
        debug: bool = False,
        debug_char_limit: int = 280,
        chat_timeout_seconds: float = 120,
        **kwargs,
    ) -> None:
        """Run one forced-tool step: inject schema instruction → call LLM → extract tool call → execute → append results."""
        # Inject step instruction with tool schema so the model knows how to call it.
        step_instruction = self._build_step_instruction(step_name, step_number, tool_name, tool_schema, has_image)
        msgs.append({"role": "user", "content": step_instruction})

        if debug:
            self.logger.info(f"[RAG] {step_name}: forcing {tool_name}, messages={len(msgs)}")

        response = chat_completion_request(
            model=self.model,
            messages=msgs,
            base_url=self.api_base,
            api_key=self.key,
            tools=tool_schema,
            tool_choice=self._force_tool_choice(tool_name),
            timeout_seconds=chat_timeout_seconds,
            **kwargs,
        )

        if not isinstance(response, dict):
            raise RuntimeError(f"[RAG] {step_name}: invalid response type: {type(response).__name__}")
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"[RAG] {step_name}: missing choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise RuntimeError(f"[RAG] {step_name}: invalid message")

        raw_response = message.get("content") or ""
        native_tool_calls = self._normalize_tool_calls(
            message.get("tool_calls"), turn_idx=turn_idx,
        )

        # Build turn info for trajectory.
        parsed = self.parse_response(raw_response)
        turn_info = {
            "turn": turn_idx + 1,
            "step": step_name,
            "forced_tool": tool_name,
            "assistant": {
                "content": raw_response,
                "perception": parsed["perception"],
                "thinking": parsed["thinking"],
                "tool_calls": [
                    {
                        "id": c["id"],
                        "name": c["name"],
                        "raw_arguments": c["raw_arguments"],
                        "args": c["args"],
                        "parse_error": c.get("parse_error"),
                    }
                    for c in native_tool_calls
                ],
            },
            "tool_responses": [],
            "memory_outputs": [],
        }

        if native_tool_calls:
            # Append assistant message with tool_calls.
            msgs.append({
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": [c["assistant_tool_call"] for c in native_tool_calls],
            })

            # Execute each tool call (usually just one since we forced it).
            for call in native_tool_calls:
                normalized_payload, memory_output = self._execute_tool_call(
                    call, run_memory, msgs, turn_idx, debug, debug_char_limit,
                )
                turn_info["tool_responses"].append(normalized_payload)
                turn_info["memory_outputs"].append(memory_output)

            # Emit turn-level memory updates (e.g. Environment Update).
            self._append_messages_with_indices(
                msgs,
                run_memory.emit_after_tool_turn(
                    agent=self,
                    turn_idx=turn_idx,
                    tool_calls=native_tool_calls,
                    tool_payloads=turn_info["tool_responses"],
                    messages=msgs,
                ),
            )
        else:
            # Model didn't produce a tool call despite being forced.
            # Append the text response so context isn't lost.
            msgs.append({"role": "assistant", "content": raw_response})
            if debug:
                self.logger.warning(f"[RAG] {step_name}: model did not produce a tool call")

        chat_history.append(turn_info)

    @staticmethod
    def _input_has_image(question) -> bool:
        """Check if the input question contains an image."""
        if isinstance(question, list):
            for item in question:
                if isinstance(item, dict):
                    if item.get("type") == "image":
                        return True
                    if item.get("type") == "image_url":
                        return True
        return False

    def _run(
        self,
        question,
        return_messages: bool = False,
        return_snapshot: bool = False,
        **kwargs,
    ) -> tuple:
        """RAG pipeline: [image search →] text search → final answer.

        If the input contains an image: image search → text search → answer (2-step).
        If the input is text-only: text search → answer (1-step).
        """
        debug = bool(kwargs.pop("debug_tool_loop", False))
        debug_char_limit = int(kwargs.pop("debug_char_limit", 280))
        chat_timeout_seconds = kwargs.pop("timeout_seconds", self.timeout)
        _ = kwargs.pop("tool_choice", None)  # ignored; we control tool_choice per step
        active_system_prompt = kwargs.pop("system_prompt", self.system_prompt)

        # Strip OpenAI-incompatible kwargs.
        endpoint_hint = f"{getattr(self, 'base', '')} {getattr(self, 'api_base', '')}".lower()
        is_openai_official = "api.openai.com" in endpoint_hint
        model_name = getattr(self, "model", "") or ""
        is_openai_native = is_openai_official or model_name.strip().lower().startswith("gpt-5")
        if is_openai_native:
            _ = kwargs.pop("dataset", None)
        repetition_penalty = kwargs.pop("repetition_penalty", getattr(self, "repetition_penalty", 1.0))
        if not is_openai_native:
            kwargs["repetition_penalty"] = repetition_penalty
        _ = kwargs.pop("tools", None)  # we manage tools per step

        # Initialize memory & messages.
        run_memory = self._new_persistent_state_memory()
        run_memory.reset()
        msgs = self.prepare_inputs(question, system_prompt=active_system_prompt)
        msgs = self._add_user_prompt_prefix(msgs)
        self._append_messages_with_indices(
            msgs,
            run_memory.on_run_start(agent=self, question=question, messages=msgs),
        )

        chat_history: List[Dict[str, Any]] = []
        has_image = self._input_has_image(question)

        # Capture input snapshot.
        all_tools = (self._image_search_schema if has_image else []) + self._text_search_schema
        input_snapshot = {
            "system_prompt": next((m["content"] for m in msgs if m["role"] == "system"), None),
            "user_messages": [m for m in msgs if m["role"] == "user"],
            "tool_definitions": cp.deepcopy(all_tools),
            "rendered_prompt": None,
            "rag_pipeline": {
                "has_image": has_image,
                "image_search_tool": self.image_search_tool if has_image else None,
                "text_search_tool": self.text_search_tool,
            },
        }

        shared_kwargs = dict(
            debug=debug,
            debug_char_limit=debug_char_limit,
            chat_timeout_seconds=chat_timeout_seconds,
            **kwargs,
        )

        turn_idx = 0
        step_number = 1

        # ---- Image search (only if input has image) ----
        if has_image:
            if debug:
                self.logger.info("[RAG] Input has image — running image search step")
            self._rag_step(
                step_name="image_search",
                step_number=step_number,
                tool_name=self.image_search_tool,
                tool_schema=self._image_search_schema,
                msgs=msgs,
                run_memory=run_memory,
                turn_idx=turn_idx,
                chat_history=chat_history,
                has_image=has_image,
                **shared_kwargs,
            )
            turn_idx += 1
            step_number += 1
        else:
            if debug:
                self.logger.info("[RAG] No image in input — skipping image search step")

        # ---- Text search ----
        self._rag_step(
            step_name="text_search",
            step_number=step_number,
            tool_name=self.text_search_tool,
            tool_schema=self._text_search_schema,
            msgs=msgs,
            run_memory=run_memory,
            turn_idx=turn_idx,
            has_image=has_image,
            chat_history=chat_history,
            **shared_kwargs,
        )

        turn_idx += 1

        # ---- Final step: answer (no tools) ----
        if debug:
            self.logger.info(f"[RAG] final_answer: requesting answer, messages={len(msgs)}")

        msgs.append({
            "role": "user",
            "content": "Based on all the information gathered above, please provide your final answer now.",
        })

        response = chat_completion_request(
            model=self.model,
            messages=msgs,
            base_url=self.api_base,
            api_key=self.key,
            tools=None,
            tool_choice="none",
            timeout_seconds=chat_timeout_seconds,
            **kwargs,
        )

        if not isinstance(response, dict):
            raise RuntimeError(f"[RAG] final_answer: invalid response type: {type(response).__name__}")
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("[RAG] final_answer: missing choices")
        final_message = choices[0].get("message")
        if not isinstance(final_message, dict):
            raise RuntimeError("[RAG] final_answer: invalid message")

        final_raw = final_message.get("content") or ""
        final_answer = self._extract_final_answer_from_content(final_raw)
        msgs.append({"role": "assistant", "content": final_raw})

        final_parsed = self.parse_response(final_raw)
        chat_history.append({
            "turn": turn_idx + 1,
            "step": "final_answer",
            "assistant": {
                "content": final_raw,
                "perception": final_parsed["perception"],
                "thinking": final_parsed["thinking"],
                "tool_calls": [],
            },
            "tool_responses": [],
            "final_answer": final_answer,
        })

        if debug:
            self.logger.info(
                f"[RAG] done: final_answer={self._truncate_for_debug(final_answer, max_chars=debug_char_limit)}"
            )

        if not final_answer:
            final_answer = self.fail_msg

        if return_messages:
            if return_snapshot:
                return final_answer, chat_history, msgs, cp.deepcopy(input_snapshot)
            return final_answer, chat_history, msgs
        if return_snapshot:
            return final_answer, chat_history, cp.deepcopy(input_snapshot)
        return final_answer, chat_history
