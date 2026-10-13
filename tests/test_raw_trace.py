"""Raw export boundaries with deterministic Qwen-style token replies; no servers."""
import copy
import json
import re
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from openvlharness.bootstrap import load_agent_config
from openvlharness.contracts import CapabilityResult
from openvlharness.services.models import ModelAdapter
from openvlharness.services.raw_trace import RawTraceExportError
from test_agent import IMAGE, ProbeCapability, make_agent, model_response
from test_bootstrap import config_for


class Tokenizer:
    specials = {name: index for index, name in enumerate(
        ("<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>"), 1)}

    def convert_tokens_to_ids(self, token):
        return self.specials.get(token)

    def encode(self, text):
        pieces = re.split("(" + "|".join(re.escape(s) for s in self.specials) + ")", text)
        return [token for piece in pieces for token in
                ([self.specials[piece]] if piece in self.specials else [ord(c) + 1000 for c in piece])]

    def decode(self, ids, **kwargs):
        reverse = {value: key for key, value in self.specials.items()}
        return "".join(reverse[i] if i in reverse else chr(i - 1000) for i in ids)

    def prompt(self, messages):
        result = ""
        for message in messages:
            role, content = message["role"], message.get("content", "")
            if isinstance(content, list):
                content = "".join("<|vision_start|><|image_pad|><|vision_end|>" if part["type"] == "image_url"
                                  else part.get("text", "") for part in content)
            if message.get("tool_calls"):
                content += "<tool_call>" + json.dumps(message["tool_calls"]) + "</tool_call>"
            if role == "tool":
                role, content = "user", "<tool_response>" + content + "</tool_response>"
            result += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        return result + "<|im_start|>assistant\n"


class TokenReplies:
    def __init__(self, replies, tokenizer, mutate=None):
        self.replies, self.tokenizer, self.mutate = list(replies), tokenizer, mutate
        self.requests = []

    def __call__(self, payload):
        self.requests.append(copy.deepcopy(payload))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        result = copy.deepcopy(reply)
        result["prompt_token_ids"] = self.tokenizer.encode(self.tokenizer.prompt(payload["messages"]))
        message = result["choices"][0]["message"]
        completion = message["content"] or "<tool_call>" + json.dumps(message["tool_calls"]) + "</tool_call>"
        result["choices"][0]["token_ids"] = self.tokenizer.encode(completion + "<|im_end|>")
        if self.mutate:
            self.mutate(result)
        return result


def agent_with_tokens(replies, capability=None, mutate=None, **kwargs):
    agent, _, capability = make_agent("chat_completions", [], capability, **kwargs)
    tokenizer = Tokenizer()
    agent.adapter.options["return_token_ids"] = True
    agent.adapter.raw_trace_enabled = True
    agent.adapter._tokenizer = tokenizer
    source = TokenReplies(replies, tokenizer, mutate)
    agent.adapter.requester = source
    return agent, source, capability


def final(text="42"):
    return model_response("chat_completions", f"<answer>{text}</answer>")


def tool_call():
    return model_response("chat_completions", calls=[("c1", "Probe_Tool", '{"query":"objects"}')])


def image_capability():
    def result(call, memory):
        return CapabilityResult(ok=True, payload={"ok": True, "result": "image"},
                                evidence=[{"type": "image_url", "image_url": {"url": IMAGE}}])
    return ProbeCapability(result)


QUESTION = [{"type": "image", "value": IMAGE}, {"type": "text", "value": "Count."}]


@pytest.mark.parametrize("alias", ["gpt-6-luna", "kimi-k2.5", "qwen-unregistered"])
def test_unsupported_models_fail_before_inference(alias):
    config = config_for()
    config["services"]["models"]["orchestrator"]["alias"] = alias
    with patch("requests.post") as request:
        with pytest.raises(ValueError, match="Raw trace export is supported only"):
            load_agent_config(config, save_raw_trace=True)
        with pytest.raises(ValueError, match="Raw trace export is supported only"):
            ModelAdapter({"alias": alias, "base": "https://fixture.invalid", "return_token_ids": True})
    request.assert_not_called()


def test_explicit_export_changes_effective_config_without_mutating_preset():
    config = config_for()
    model = config["services"]["models"]["orchestrator"]
    model.update(alias="qwen3-vl-8b-instruct", return_token_ids=False)
    normal = load_agent_config(config)
    enabled = load_agent_config(config, save_raw_trace=True)
    assert not config["services"]["models"]["orchestrator"]["return_token_ids"]
    assert not normal["services"]["models"]["orchestrator"]["return_token_ids"]
    assert enabled["services"]["models"]["orchestrator"]["return_token_ids"] is True
    assert ModelAdapter(enabled["services"]["models"]["orchestrator"]).payload([])["return_token_ids"] is True


def test_raw_export_loads_pinned_tokenizer_without_context_counting():
    factory = Mock(return_value=Tokenizer())
    adapter = ModelAdapter({"alias": "qwen3-vl-8b-instruct", "base": "https://fixture.invalid", "return_token_ids": True})
    with patch.dict(sys.modules, {"transformers": SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=factory))}):
        adapter.load_tokenizer()
    factory.assert_called_once_with(adapter.model, local_files_only=True, revision=adapter.profile["raw_trace_tokenizer_revision"])
    assert adapter._tokenizer is not None


def test_missing_required_tokenizer_is_a_startup_error():
    factory = Mock(side_effect=OSError("not cached"))
    adapter = ModelAdapter({"alias": "qwen3-vl-8b-instruct", "base": "https://fixture.invalid", "return_token_ids": True})
    with patch.dict(sys.modules, {"transformers": SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=factory))}), patch("requests.post") as request:
        with pytest.raises(RawTraceExportError, match="requires the matching tokenizer"):
            adapter.load_tokenizer()
    request.assert_not_called()


def test_text_export_and_ordinary_trajectory_are_independent():
    agent, source, _ = agent_with_tokens([final()])
    exported = agent.generate("Compute six times seven.", task_id=9)
    raw = exported["extra_records"]["trajectory"].pop("raw_trace")
    assert raw["images"] == []
    assert raw["messages"][-1] == {"role": "assistant", "content": [{"type": "text", "text": "<answer>42</answer>"}]}
    ordinary, _, _ = make_agent("chat_completions", [final()])
    assert exported == ordinary.generate("Compute six times seven.", task_id=9)
    assert source.requests[0]["return_token_ids"] is True


@pytest.mark.parametrize("max_iterations", [1, 3])
def test_multiturn_image_tool_and_extra_final_export(max_iterations):
    agent, source, _ = agent_with_tokens([tool_call(), final("2")], image_capability(), max_iterations=max_iterations)
    with patch.object(agent.adapter, "export_raw_trace", wraps=agent.adapter.export_raw_trace) as export:
        result = agent.generate(QUESTION, task_id=17)
    capture, _ = export.call_args.args
    assert capture["messages"] == source.requests[-1]["messages"]
    assert capture["messages"][-1]["role"] != "assistant"
    raw = result["extra_records"]["trajectory"]["raw_trace"]
    assert result["prediction"] == "2"
    assert len(raw["images"]) == 2
    assert "<tool_call>" in json.dumps(raw["messages"])
    assert "<tool_response>" in json.dumps(raw["messages"])
    urls = [part["image_url"]["url"] for m in capture["messages"] if isinstance(m.get("content"), list)
            for part in m["content"] if part["type"] == "image_url"]
    assert [entry["bytes_b64"] for entry in raw["images"]] == [url.split(",", 1)[1] for url in urls]
    assert raw["messages"][-1]["content"] == [{"type": "text", "text": "<answer>2</answer>"}]
    assert len(source.requests) == 2


@pytest.mark.parametrize("field", ["prompt_token_ids", "token_ids"])
def test_missing_token_ids_report_export_failure_without_losing_prediction(field):
    def omit(reply):
        (reply if field == "prompt_token_ids" else reply["choices"][0]).pop(field)
    agent, _, _ = agent_with_tokens([final()], mutate=omit)
    result = agent.generate("Question")
    trajectory = result["extra_records"]["trajectory"]
    assert result["prediction"] == "42"
    assert "raw_trace" not in trajectory
    assert trajectory["raw_trace_error"]["code"] == "missing_token_ids"


def test_alignment_failure_is_explicit():
    agent, _, _ = agent_with_tokens([final()])
    tokenizer = agent.adapter._tokenizer
    def remove_vision(reply):
        text = tokenizer.decode(reply["prompt_token_ids"])
        reply["prompt_token_ids"] = tokenizer.encode(re.sub(r"<\|vision_start\|>.*?<\|vision_end\|>", "", text))
    agent.adapter.requester.mutate = remove_vision
    result = agent.generate(QUESTION)
    assert result["prediction"] == "42"
    assert result["extra_records"]["trajectory"]["raw_trace_error"]["code"] == "image_alignment"


@pytest.mark.parametrize("tokenizer,code", [(None, "tokenizer_unavailable"), (Mock(), "tokenizer_mismatch")])
def test_missing_or_wrong_tokenizer_at_export_is_explicit(tokenizer, code):
    agent, _, _ = agent_with_tokens([final()])
    agent.adapter._tokenizer = tokenizer
    result = agent.generate("Question")
    assert result["prediction"] == "42"
    assert result["extra_records"]["trajectory"]["raw_trace_error"]["code"] == code


def test_provider_context_fallback_exports_only_its_actual_request_images():
    agent, source, _ = agent_with_tokens([tool_call(), RuntimeError("maximum input length exceeded"), final()], image_capability())
    with patch.object(agent.adapter, "export_raw_trace", wraps=agent.adapter.export_raw_trace) as export:
        result = agent.generate(QUESTION)
    assert len(source.requests) == 3
    assert len(source.requests[-1]["messages"]) == 1
    assert export.call_args.args[0]["messages"] == source.requests[-1]["messages"]
    raw = result["extra_records"]["trajectory"]["raw_trace"]
    assert len(raw["images"]) == 1
    assert [m["role"] for m in raw["messages"]] == ["user", "assistant"]
    assert len(result["extra_records"]["trajectory"]["messages"]) > 2


def test_local_context_fallback_exports_the_reduced_request():
    agent, source, _ = agent_with_tokens([final()])
    agent.adapter.max_context_length = 5000
    with patch.object(agent.adapter, "_count_tokens", return_value=10000):
        result = agent.generate(QUESTION)
    assert len(source.requests[0]["messages"]) == 1
    raw = result["extra_records"]["trajectory"]["raw_trace"]
    assert len(raw["images"]) == 1
    assert [m["role"] for m in raw["messages"]] == ["user", "assistant"]


def test_context_compaction_removes_old_tool_image_from_export():
    agent, source, _ = agent_with_tokens([tool_call(), tool_call(), final()], image_capability(), max_iterations=3)
    agent.adapter.max_context_length = 5000
    def count(messages, tools):
        if sum(m["role"] == "assistant" for m in messages) < 2:
            return 0
        # Force compaction until the oldest tool image's content is dropped.
        return 10000 if isinstance(messages[5].get("content"), list) else 0
    with patch.object(agent.adapter, "_count_tokens", side_effect=count), \
         patch.object(agent.adapter, "export_raw_trace", wraps=agent.adapter.export_raw_trace) as export:
        result = agent.generate(QUESTION)
    assert export.call_args.args[0]["messages"] == source.requests[-1]["messages"]
    raw = result["extra_records"]["trajectory"]["raw_trace"]
    assert "[Truncated to save context]" in json.dumps(raw["messages"])
    assert len(raw["images"]) == 2  # original input + latest tool image


def test_failed_final_request_does_not_export_previous_successful_response():
    agent, _, _ = agent_with_tokens([tool_call(), RuntimeError("provider unavailable")], image_capability(), max_iterations=1)
    result = agent.generate(QUESTION)
    trajectory = result["extra_records"]["trajectory"]
    assert "raw_trace" not in trajectory
    assert trajectory["raw_trace_error"]["code"] == "inference_failed"
