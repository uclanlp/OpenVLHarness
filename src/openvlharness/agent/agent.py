"""One agent loop; capabilities own tool meaning and adapters own API syntax."""
import copy
import hashlib
import os
import re

from ..contracts import VirtualCall
from ..capabilities._base import failure
from ..input import normalize_input, input_messages
from ..services.raw_sft import RawSFTExportError


def _accumulate(into, values):
    for key, value in values.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            into[key] = into.get(key, 0) + value
        elif isinstance(value, dict):
            _accumulate(into.setdefault(key, {}), value)


class MainAgent:
    fail_msg = "Error: Unable to get response from the server."

    def __init__(self, model_adapter, capabilities, memory_factory, system_prompt, user_prefix,
                 max_iterations=15, tool_choice="auto", paired_sampling_seed=False):
        self.adapter = model_adapter
        self.capabilities = dict(capabilities)
        self.memory_factory = memory_factory
        self.system_prompt = system_prompt
        self.user_prefix = user_prefix
        self.max_iterations = max_iterations
        self.tool_choice = tool_choice
        self.paired_sampling_seed = paired_sampling_seed
        self.model = model_adapter.model
        self.json_schema = [c.public_schema() for c in self.capabilities.values()]
        self.adapter.json_schema = self.json_schema

    def generate(self, message, *, dataset=None, task_id=None):
        del dataset
        original = copy.deepcopy(message)
        question, system = normalize_input(message, self.system_prompt)
        memory = self.memory_factory()
        history = [self.adapter.encode_message(m) for m in input_messages(question, system, self.user_prefix)]
        initial = memory.start(question)
        if initial:
            history.append(self.adapter.encode_message(initial))
        turns, detections, usage = [], [], {}
        final = self.fail_msg
        error = None
        capture = {} if self.adapter.raw_sft_enabled else None
        try:
            final = self._run(history, memory, turns, detections, usage, task_id, capture=capture)
        except (RuntimeError, OSError) as exc:
            error = str(exc)
        trajectory = {"input_messages": original, "chat_history": turns,
                      "messages": history, "final_answer": final}
        if error:
            trajectory["error"] = error
        if capture is not None:
            try:
                if error:
                    raise RawSFTExportError("inference_failed", f"Raw SFT export failed because inference did not complete: {error}")
                trajectory["raw_sft"] = self.adapter.export_raw_sft(capture, task_id)
            except RawSFTExportError as exc:
                trajectory["raw_sft_error"] = {"code": exc.code, "message": str(exc)}
                self.adapter.logger.error("Raw SFT export failed for task %s: %s", task_id, exc)
        extra = {"trajectory": trajectory, "tool_detections": detections}
        if usage:
            extra["usage"] = {"total": usage}
        return {"prediction": final, "extra_records": extra}

    def _run(self, history, memory, turns, detections, usage, task_id, *, capture=None):
        adapter = self.adapter
        policy = adapter.profile
        tools, choice = self.json_schema, self.tool_choice
        seed_base = None
        if self.paired_sampling_seed and task_id is not None:
            base = int(hashlib.md5(str(task_id).encode()).hexdigest()[:8], 16) % 1_000_000_000
            try:
                salt = int(os.environ.get("HARNESS_SEARCH_SAMPLE_INDEX", "0") or "0")
            except ValueError:
                salt = 0
            seed_base = (base + salt * 1_000_003) % 2_000_000_000
        for index in range(self.max_iterations):
            if index == self.max_iterations - 1:
                history.append(adapter.encode_message({"role": "user",
                    "content": "Please provide your best final answer now if enough information is available."}))
            if policy.get("context_policy") == "qwen_tokenizer":
                tools, choice = adapter._ensure_context_fits(history, tools, choice)
            seed = seed_base + index if seed_base is not None else None
            try:
                response = adapter.request(history, tools, choice, seed=seed, capture=capture)
            except RuntimeError as exc:
                context_error = any(text in str(exc).lower() for text in ("context length", "maximum input length"))
                if policy.get("context_error_fallback") and context_error:
                    response = adapter.request(adapter._build_vanilla_fallback_msgs(history), None, "none", seed=seed, capture=capture)
                else:
                    raise
            if isinstance(response, dict) and isinstance(response.get("usage"), dict):
                _accumulate(usage, response["usage"])
            text, calls, original = adapter.parse(response, index)
            turn = self._turn(index, text, calls)
            if calls:
                adapter.append_assistant_calls(history, text, calls, original)
                deferred = []
                for call in calls:
                    virtual = VirtualCall(call["id"], call["name"], call["args"])
                    if call.get("parse_error"):
                        result = failure(virtual, call["parse_error"]["code"], call["parse_error"]["message"],
                                         details=call["parse_error"].get("details"))
                    elif call["name"] not in self.capabilities:
                        result = failure(virtual, "tool_not_found", f"Tool not found: {call['name']}")
                    else:
                        result = self.capabilities[call["name"]].invoke(virtual, memory)
                    if result.detection:
                        detection = copy.deepcopy(result.detection)
                        detection["tool_call_idx"] = len(detections)
                        detection["boxes_xyxy_pixels"] = [list(map(float, box[:4])) for box in detection["boxes_xyxy_pixels"]]
                        detection["scores"] = [float(s) for s in detection["scores"]]
                        detections.append(detection)
                    turn["tool_responses"].append(result.payload)
                    immediate, later = adapter.result_messages(call, result)
                    start = len(history)
                    history.extend(immediate)
                    record = {"tool_call_id": call["id"], "tool_name": call["name"],
                              "immediate_message_indices": list(range(start, len(history))),
                              "deferred_message_indices": [], "image_refs": result.prepared.get("image_refs", []),
                              "tool_output_processor": result.processor_info,
                              "canonical_result": result.data}
                    turn["memory_outputs"].append(record)
                    deferred.append((record, later))
                for record, messages in deferred:
                    start = len(history)
                    history.extend(messages)
                    record["deferred_message_indices"] = list(range(start, len(history)))
                update = memory.build_environment_update(successful=any(p.get("ok") is True for p in turn["tool_responses"]))
                if update:
                    turn["memory_turn_message_indices"] = [len(history)]
                    history.append(adapter.encode_message(update))
                turns.append(turn)
                if index == self.max_iterations - 1 and policy.get("extra_final_request"):
                    final_tools, final_choice = adapter._ensure_context_fits(history, tools, "none")
                    response = adapter.request(history, final_tools, final_choice, seed=seed, capture=capture)
                    text, calls, original = adapter.parse(response, index + 1, phase="final_text_only")
                    if isinstance(response.get("usage"), dict):
                        _accumulate(usage, response["usage"])
                    if calls:
                        adapter.append_assistant_calls(history, text, calls, original)
                    else:
                        history.append(adapter.encode_message({"role": "assistant", "content": text}))
                    final_turn = self._turn(index + 1, text, calls)
                    final_turn["final_answer"] = adapter.final_answer(text)
                    turns.append(final_turn)
                    return final_turn["final_answer"] or self.fail_msg
                continue
            answer = adapter.final_answer(text)
            turn["final_answer"] = answer
            turns.append(turn)
            if policy.get("append_final_message") and text:
                history.append(adapter.encode_message({"role": "assistant", "content": text}))
            if answer or not policy.get("continue_empty_answer"):
                return answer or self.fail_msg
            history.append(adapter.encode_message({"role": "assistant", "content": text}))
        return self.fail_msg

    @staticmethod
    def _turn(index, text, calls):
        def tag(name):
            match = re.search(rf"<{name}>(.*?)</{name}>", text, re.DOTALL | re.IGNORECASE)
            return match.group(1).strip() if match else ""
        return {"turn": index + 1, "assistant": {"content": text,
                    "perception": tag("perception"), "thinking": tag("thinking"),
                    "tool_calls": [{key: call.get(key) for key in
                                   ("id", "name", "raw_arguments", "args", "parse_error")} for call in calls]},
                "tool_responses": [], "memory_outputs": [], "memory_turn_message_indices": []}
