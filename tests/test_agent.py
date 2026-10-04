"""Fixed-response acceptance tests for the shared loop's provider policies.

No servers, old runtime, dataset packages, or model credentials are used. Real
model encoders and per-question memory are exercised with an in-process model
response queue and capabilities that record exactly which calls were executed.
"""
import copy
import base64
import hashlib
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import unittest
from unittest.mock import Mock, patch

from PIL import Image, UnidentifiedImageError
from requests import HTTPError, Response

from openvlharness.agent.agent import MainAgent
from openvlharness.contracts import CapabilityResult
from openvlharness.memory.store import MemoryStore
from openvlharness.services.models import ModelAdapter


FIXTURE = json.loads((Path(__file__).parent / "fixtures/model_protocol/old_release.json").read_text())
IMAGE = FIXTURE["image_data_url"]
ALIASES = {"chat_completions": "qwen3-vl-8b-instruct", "responses": "gpt-6-luna"}
NUDGE = "Please provide your best final answer now if enough information is available."


def model_response(protocol, text="", calls=(), usage=None):
    if protocol == "chat_completions":
        message = {"role": "assistant", "content": text}
        if calls:
            message["tool_calls"] = [{"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
                                     for call_id, name, arguments in calls]
        result = {"choices": [{"message": message}]}
    else:
        result = {"output": ([{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}] if text else []) +
                  [{"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments}
                   for call_id, name, arguments in calls]}
    if usage is not None:
        result["usage"] = copy.deepcopy(usage)
    return result


class FixedModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, payload):
        self.requests.append(copy.deepcopy(payload))
        if not self.responses:
            raise AssertionError("Agent made an unexpected extra model request")
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return copy.deepcopy(result)


class ProbeCapability:
    name = "Probe_Tool"

    def __init__(self, handler=None):
        self.invocations = []
        self.handler = handler

    def public_schema(self):
        return {"type": "function", "function": {"name": self.name, "description": "Offline perception probe",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}

    def invoke(self, call, memory):
        self.invocations.append((copy.deepcopy(call), memory))
        if self.handler:
            return self.handler(call, memory)
        return CapabilityResult(ok=True, data={"answer": 2},
                                evidence=[{"type": "text", "text": "There are 2 objects."}],
                                payload={"ok": True, "tool_name": call.name, "tool_call_id": call.call_id,
                                         "result": {"answer": 2}, "error": None})


def make_agent(protocol, responses, capability=None, **kwargs):
    fixed = FixedModel(responses)
    model = ModelAdapter({"alias": ALIASES[protocol], "base": "http://fixture.invalid/v1", "max_tokens": 4096}, requester=fixed)
    capability = capability or ProbeCapability()
    agent = MainAgent(model, {capability.name: capability}, MemoryStore, "fixture system", "Question: {user_query}", **kwargs)
    return agent, fixed, capability


def wire_messages(protocol, request):
    return request["messages" if protocol == "chat_completions" else "input"]


def contents(message):
    value = message.get("content", "")
    return value if isinstance(value, str) else "".join(part.get("text", "") for part in value if isinstance(part, dict))


class AgentLoopTests(unittest.TestCase):
    def test_url_images_reach_model_and_memory_from_one_download(self):
        png = io.BytesIO()
        Image.new("RGB", (3, 2), "blue").save(png, format="PNG")
        data_urls = [IMAGE, "data:image/png;base64," + base64.b64encode(png.getvalue()).decode("ascii")]
        image_bytes = [base64.b64decode(url.split(",", 1)[1]) for url in data_urls]
        downloads = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                downloads.append(self.path)
                body = image_bytes[0 if self.path == "/first.png" else 1]
                self.send_response(200)
                # Signed/storage URLs need not advertise an image content type.
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                for protocol in ALIASES:
                    for input_form in ("typed", "strings"):
                        with self.subTest(protocol=protocol, input_form=input_form):
                            downloads.clear()
                            paths = ["/first.png", "/second?download=1" if input_form == "typed" else "/second.png?download=1"]
                            urls = [f"http://127.0.0.1:{server.server_port}{path}" for path in paths]
                            if input_form == "typed":
                                # M4Bench's URL-only branch emits two image dicts and text.
                                question = [{"type": "image", "value": url, "detail": "high"} for url in urls]
                                question.append({"type": "text", "value": "Compare these images."})
                            else:
                                question = urls + ["Compare these images."]
                            original = copy.deepcopy(question)
                            responses = [model_response(protocol, "<answer>different</answer>")]
                            agent, fixed, _ = make_agent(protocol, responses)
                            memory = MemoryStore()
                            agent.memory_factory = lambda: memory
                            result = agent.generate(question)
                            self.assertEqual(result["prediction"], "different")
                            self.assertEqual(len(fixed.requests), 1)
                            self.assertEqual(downloads, paths)
                            for index, data_url in enumerate(data_urls, 1):
                                self.assertEqual(memory.resolve_image(f"input_image_{index}")[0], data_url)
                                self.assertEqual(memory.resolve_image(f"image_{index}")[0], data_url)
                            self.assertEqual(list(memory.state.images.registry), ["input_image_1", "input_image_2"])
                            messages = wire_messages(protocol, fixed.requests[0])
                            self.assertIn("input_image_1", contents(messages[2]))
                            self.assertIn("input_image_2", contents(messages[2]))
                            # Existing data-URL inputs give the same image encoding and history.
                            local = [{"type": "image", "value": value,
                                      **({"detail": "high"} if input_form == "typed" else {})} for value in data_urls]
                            local.append({"type": "text", "value": "Compare these images."})
                            expected_agent, expected, _ = make_agent(protocol, responses)
                            expected_agent.generate(local)
                            self.assertEqual(fixed.requests, expected.requests)
                            self.assertEqual(question, original)
                            self.assertEqual(result["extra_records"]["trajectory"]["input_messages"], original)
            finally:
                server.shutdown()
                thread.join(timeout=5)

    def test_url_image_download_errors_stop_before_model_requests(self):
        for protocol in ALIASES:
            for status, content, error in ((404, b"missing", HTTPError),
                                           (200, b"not an image", UnidentifiedImageError)):
                with self.subTest(protocol=protocol, status=status):
                    response = Response()
                    response.status_code = status
                    response.url = "https://fixture.invalid/image"
                    response._content = content
                    response._content_consumed = True
                    agent, fixed, _ = make_agent(protocol, [])
                    question = [{"type": "image", "value": response.url}]
                    with patch("openvlharness.input.requests.get", return_value=response) as download:
                        with self.assertRaises(error):
                            agent.generate(question)
                    download.assert_called_once_with(response.url, timeout=30)
                    self.assertEqual(fixed.requests, [])
                    self.assertEqual(question, [{"type": "image", "value": response.url}])

    def test_tool_roundtrip_retains_ids_decodes_arguments_and_orders_evidence(self):
        def image_result(call, memory):
            ref = memory.register_image(IMAGE, provenance="tool_output:Probe_Tool")
            return CapabilityResult(ok=True, data={"image_ref": ref},
                evidence=[{"type": "text", "text": f"Inspect {ref}."}, {"type": "image_url", "image_url": {"url": IMAGE}}],
                prepared={"image_refs": [ref]}, processor_info={"status": "ok"},
                payload={"ok": True, "tool_name": call.name, "tool_call_id": call.call_id, "result": {"count": 2}, "error": None})
        for protocol in ALIASES:
            with self.subTest(protocol=protocol):
                probe = ProbeCapability(image_result)
                agent, fixed, _ = make_agent(protocol, [
                    model_response(protocol, "examining", [("call-retained", "Probe_Tool", '{"query":"count objects"}')]),
                    model_response(protocol, "<answer>2</answer>")], probe)
                result = agent.generate("How many?")
                self.assertEqual(result["prediction"], "2")
                self.assertEqual(len(probe.invocations), 1)
                call, memory = probe.invocations[0]
                self.assertEqual((call.call_id, call.name, call.arguments), ("call-retained", "Probe_Tool", {"query": "count objects"}))
                self.assertEqual(memory.resolve_image("tool_generated_image_1")[0], IMAGE)
                first = wire_messages(protocol, fixed.requests[0])
                second = wire_messages(protocol, fixed.requests[1])
                self.assertEqual(second[:len(first)], first)
                added = second[len(first):]
                if protocol == "chat_completions":
                    self.assertEqual([item["role"] for item in added], ["assistant", "tool", "user", "user"])
                    self.assertEqual(added[0]["tool_calls"][0]["id"], "call-retained")
                    self.assertEqual(added[1], {"role": "tool", "tool_call_id": "call-retained", "name": "Probe_Tool", "content": "ok"})
                    self.assertEqual(added[2]["content"][1], {"type": "image_url", "image_url": {"url": IMAGE}})
                else:
                    self.assertEqual([item["type"] for item in added], ["function_call", "message", "function_call_output", "message"])
                    self.assertEqual(added[0]["call_id"], "call-retained")
                    self.assertEqual(added[2], {"type": "function_call_output", "call_id": "call-retained", "output": [
                        {"type": "input_text", "text": "Inspect tool_generated_image_1."},
                        {"type": "input_image", "image_url": IMAGE}]})
                self.assertTrue(contents(added[-1]).startswith("Environment Update"))
                self.assertIn("tool_generated_image_1", contents(added[-1]))
                record = result["extra_records"]["trajectory"]["chat_history"][0]["memory_outputs"][0]
                self.assertEqual(record["image_refs"], ["tool_generated_image_1"])
                self.assertEqual(record["canonical_result"], {"image_ref": "tool_generated_image_1"})
                self.assertEqual(len(record["immediate_message_indices"]), 1)
                self.assertEqual(len(record["deferred_message_indices"]), 1 if protocol == "chat_completions" else 0)

    def test_argument_error_unknown_tool_and_failed_tool_are_model_visible_without_environment_update(self):
        for protocol in ALIASES:
            for mode in ["invalid_json", "unknown_tool", "backend_failure"]:
                with self.subTest(protocol=protocol, mode=mode):
                    code = "backend_failed"
                    def fail(call, memory):
                        return CapabilityResult(ok=False, error={"code": code, "message": "fixture unavailable"},
                                                payload={"ok": False, "tool_name": call.name, "tool_call_id": call.call_id,
                                                         "result": None, "error": {"code": code, "message": "fixture unavailable"}})
                    probe = ProbeCapability(fail)
                    name = "Missing_Tool" if mode == "unknown_tool" else "Probe_Tool"
                    args = "{bad" if mode == "invalid_json" else '{"query":"count"}'
                    agent, fixed, _ = make_agent(protocol, [model_response(protocol, calls=[("retained", name, args)]), model_response(protocol, "<answer>unknown</answer>")], probe)
                    result = agent.generate("Question")
                    self.assertEqual(result["prediction"], "unknown")
                    self.assertEqual(len(probe.invocations), int(mode == "backend_failure"))
                    messages = wire_messages(protocol, fixed.requests[1])
                    returns = [m for m in messages if m.get("role") == "tool" or m.get("type") == "function_call_output"]
                    self.assertEqual(len(returns), 1)
                    output = returns[0].get("output", returns[0].get("content"))
                    expected = ("invalid_tool_arguments_json" if protocol == "chat_completions" else "invalid_arguments_json") if mode == "invalid_json" else "tool_not_found" if mode == "unknown_tool" else code
                    self.assertTrue(output.startswith(expected + ":"))
                    self.assertEqual(returns[0].get("tool_call_id", returns[0].get("call_id")), "retained")
                    initial = wire_messages(protocol, fixed.requests[0])
                    self.assertFalse(any(contents(m).startswith("Environment Update") for m in messages[len(initial):]))

    def test_all_multicalls_are_rejected_with_matching_results_and_no_execution(self):
        for protocol in ALIASES:
            with self.subTest(protocol=protocol):
                agent, fixed, probe = make_agent(protocol, [
                    model_response(protocol, calls=[("a", "Probe_Tool", '{"query":"a"}'), ("b", "Probe_Tool", '{"query":"b"}')]),
                    model_response(protocol, "<answer>recovered</answer>")])
                result = agent.generate("Question")
                self.assertEqual(result["prediction"], "recovered")
                self.assertEqual(probe.invocations, [])
                messages = wire_messages(protocol, fixed.requests[1])
                outputs = [m for m in messages if m.get("role") == "tool" or m.get("type") == "function_call_output"]
                self.assertEqual([m.get("tool_call_id", m.get("call_id")) for m in outputs], ["a", "b"])
                expected = "multiple_tool_calls_not_allowed" if protocol == "chat_completions" else "multi_tool_call_rejected"
                self.assertTrue(all(m.get("output", m.get("content")).startswith(expected) for m in outputs))

    def test_chat_requires_answer_tag_while_responses_accepts_plain_answer(self):
        for protocol in ALIASES:
            with self.subTest(protocol=protocol):
                queue = [model_response(protocol, "plain answer")]
                if protocol == "chat_completions":
                    queue.append(model_response(protocol, "<answer>tagged answer</answer>"))
                agent, fixed, _ = make_agent(protocol, queue)
                result = agent.generate("Question")
                self.assertEqual(result["prediction"], "tagged answer" if protocol == "chat_completions" else "plain answer")
                self.assertEqual(len(fixed.requests), 2 if protocol == "chat_completions" else 1)
                history = result["extra_records"]["trajectory"]["messages"]
                assistant = [contents(m) for m in history if m.get("role") == "assistant"]
                self.assertEqual(assistant, ["plain answer"])

    def test_empty_or_reasoning_only_responses_terminates_with_failure_without_new_request(self):
        for output in [[], [{"type": "reasoning", "id": "reasoning-id", "encrypted_content": "opaque"}]]:
            with self.subTest(output=output):
                agent, fixed, _ = make_agent("responses", [{"output": output}])
                result = agent.generate("Question")
                self.assertEqual(result["prediction"], MainAgent.fail_msg)
                self.assertEqual(len(fixed.requests), 1)
                self.assertNotIn("opaque", json.dumps(result))

    def test_last_turn_nudge_and_extra_final_call_follow_profile_budget(self):
        for protocol in ALIASES:
            with self.subTest(protocol=protocol):
                queue = [model_response(protocol, calls=[("last", "Probe_Tool", '{"query":"count"}')])]
                if protocol == "chat_completions":
                    queue.append(model_response(protocol, "<answer>2</answer>"))
                agent, fixed, probe = make_agent(protocol, queue, max_iterations=1)
                result = agent.generate("Question")
                self.assertEqual(contents(wire_messages(protocol, fixed.requests[0])[-1]), NUDGE)
                self.assertEqual(len(probe.invocations), 1)
                if protocol == "chat_completions":
                    self.assertEqual(result["prediction"], "2")
                    self.assertEqual(len(fixed.requests), 2)
                    self.assertEqual(fixed.requests[-1]["tool_choice"], "none")
                    self.assertEqual(len(result["extra_records"]["trajectory"]["chat_history"]), 2)
                else:
                    self.assertEqual(result["prediction"], MainAgent.fail_msg)
                    self.assertEqual(len(fixed.requests), 1)

    def test_extra_final_call_cannot_execute_more_tools(self):
        agent, fixed, probe = make_agent("chat_completions", [
            model_response("chat_completions", calls=[("first", "Probe_Tool", '{"query":"a"}')]),
            model_response("chat_completions", calls=[("forbidden-extra", "Probe_Tool", '{"query":"b"}')])], max_iterations=1)
        result = agent.generate("Question")
        self.assertEqual(result["prediction"], MainAgent.fail_msg)
        self.assertEqual(len(fixed.requests), 2)
        self.assertEqual([call.call_id for call, _ in probe.invocations], ["first"])

    def test_non_object_extra_final_response_preserves_partial_trace_and_export_failure(self):
        for invalid in (None, [], "invalid response", 7, False):
            for raw_export in (False, True):
                with self.subTest(response=invalid, raw_export=raw_export):
                    agent, fixed, probe = make_agent("chat_completions", [
                        model_response("chat_completions", calls=[("first", "Probe_Tool", '{"query":"count"}')],
                                       usage={"total_tokens": 13}),
                        invalid], max_iterations=1)
                    agent.adapter.raw_trace_enabled = raw_export
                    agent.adapter.options["return_token_ids"] = raw_export
                    result = agent.generate("Count objects.", task_id=17)
                    extra = result["extra_records"]
                    trajectory = extra["trajectory"]
                    self.assertEqual(result["prediction"], MainAgent.fail_msg)
                    self.assertEqual(trajectory["error"], "Model response must be an object")
                    self.assertEqual(extra["usage"], {"total": {"total_tokens": 13}})
                    self.assertEqual(len(fixed.requests), 2)
                    self.assertEqual(fixed.requests[-1]["tool_choice"], "none")
                    self.assertEqual([call.call_id for call, _ in probe.invocations], ["first"])
                    self.assertEqual(len(trajectory["chat_history"]), 1)
                    self.assertTrue(trajectory["chat_history"][0]["tool_responses"][0]["ok"])
                    self.assertEqual(trajectory["messages"], fixed.requests[-1]["messages"])
                    self.assertNotIn("raw_trace", trajectory)
                    if raw_export:
                        self.assertEqual(trajectory["raw_trace_error"]["code"], "inference_failed")
                        self.assertIn(trajectory["error"], trajectory["raw_trace_error"]["message"])
                    else:
                        self.assertNotIn("raw_trace_error", trajectory)

    def test_usage_accumulates_nested_counts_and_ignores_boolean_and_text_fields(self):
        for protocol in ALIASES:
            with self.subTest(protocol=protocol):
                agent, _, _ = make_agent(protocol, [
                    model_response(protocol, calls=[("first", "Probe_Tool", '{"query":"a"}')], usage={"total_tokens": 13, "details": {"reasoning_tokens": 3}, "ignored": "text", "flag": True}),
                    model_response(protocol, "<answer>2</answer>", usage={"total_tokens": 19, "details": {"reasoning_tokens": 4}})])
                self.assertEqual(agent.generate("Question")["extra_records"]["usage"], {"total": {"total_tokens": 32, "details": {"reasoning_tokens": 7}}})

    def test_provider_runtime_error_is_recorded_without_becoming_an_answer(self):
        for protocol in ALIASES:
            with self.subTest(protocol=protocol):
                agent, fixed, probe = make_agent(protocol, [RuntimeError("401 fixture invalid credential")])
                result = agent.generate("Question")
                self.assertEqual(result["prediction"], MainAgent.fail_msg)
                self.assertEqual(result["extra_records"]["trajectory"]["error"], "401 fixture invalid credential")
                self.assertEqual(probe.invocations, [])
                self.assertEqual(len(fixed.requests), 1)

    def test_invalid_responses_output_records_protocol_error_and_preserves_partial_turns(self):
        for body in ({}, {"output": None}, {"output": False}, {"output": 0}, {"output": ""}, {"output": {}}):
            for after_tool in (False, True):
                with self.subTest(body=body, after_tool=after_tool):
                    queue = []
                    if after_tool:
                        queue.append(model_response("responses", calls=[("kept", "Probe_Tool", '{"query":"count"}')],
                                                    usage={"total_tokens": 13}))
                    queue.append(body)
                    agent, fixed, probe = make_agent("responses", queue)
                    result = agent.generate("Question")
                    trajectory = result["extra_records"]["trajectory"]
                    self.assertEqual(result["prediction"], MainAgent.fail_msg)
                    self.assertEqual(trajectory["error"], "Invalid responses payload: output must be a list")
                    self.assertEqual(len(fixed.requests), 2 if after_tool else 1)
                    self.assertEqual(len(probe.invocations), int(after_tool))
                    self.assertEqual(len(trajectory["chat_history"]), int(after_tool))
                    self.assertEqual(trajectory["messages"], fixed.requests[-1]["input"])
                    if after_tool:
                        self.assertEqual(trajectory["chat_history"][0]["tool_responses"][0]["tool_call_id"], "kept")
                        self.assertEqual(result["extra_records"]["usage"]["total"]["total_tokens"], 13)

    def test_explicit_empty_responses_output_finishes_without_protocol_error(self):
        agent, fixed, probe = make_agent("responses", [{"output": []}])
        result = agent.generate("Question")
        trajectory = result["extra_records"]["trajectory"]
        self.assertEqual(result["prediction"], MainAgent.fail_msg)
        self.assertNotIn("error", trajectory)
        self.assertEqual(len(trajectory["chat_history"]), 1)
        self.assertEqual(len(fixed.requests), 1)
        self.assertEqual(probe.invocations, [])

    def test_runtime_context_fallback_is_chat_only_and_uses_first_user_without_tools(self):
        for protocol in ALIASES:
            with self.subTest(protocol=protocol):
                queue = [RuntimeError("400 maximum input length exceeded")]
                if protocol == "chat_completions":
                    queue.append(model_response(protocol, "<answer>fallback</answer>"))
                agent, fixed, probe = make_agent(protocol, queue)
                result = agent.generate("Question")
                self.assertEqual(probe.invocations, [])
                if protocol == "chat_completions":
                    self.assertEqual(result["prediction"], "fallback")
                    self.assertNotIn("tools", fixed.requests[1])
                    self.assertNotIn("tool_choice", fixed.requests[1])
                    self.assertEqual(fixed.requests[1]["messages"], [fixed.requests[0]["messages"][1]])
                else:
                    self.assertEqual(result["prediction"], MainAgent.fail_msg)
                    self.assertEqual(len(fixed.requests), 1)

    def test_paired_sampling_seed_matches_old_task_turn_and_sample_salt_rule(self):
        for protocol in ALIASES:
            with self.subTest(protocol=protocol):
                agent, fixed, _ = make_agent(protocol, [
                    model_response(protocol, calls=[("a", "Probe_Tool", '{"query":"a"}')]),
                    model_response(protocol, "<answer>2</answer>")], paired_sampling_seed=True)
                with patch.dict("os.environ", {"HARNESS_SEARCH_SAMPLE_INDEX": "2"}):
                    agent.generate("Question", task_id="fsc147-17")
                expected = (int(hashlib.md5(b"fsc147-17").hexdigest()[:8], 16) % 1_000_000_000 + 2 * 1_000_003) % 2_000_000_000
                if protocol == "chat_completions":
                    self.assertEqual([body["seed"] for body in fixed.requests], [expected, expected + 1])
                else:
                    self.assertTrue(all("seed" not in body for body in fixed.requests))

    def test_memory_and_detection_sidecars_are_fresh_for_each_question(self):
        def detect(call, memory):
            memory.register_image(IMAGE, provenance="tool_output:Probe_Tool")
            memory.register_url("https://fixture.invalid/page", provenance="Probe_Tool")
            return CapabilityResult(ok=True, data={"count": 1}, evidence=[{"type": "text", "text": "one"}],
                payload={"ok": True, "result": {"count": 1}},
                detection={"boxes_xyxy_pixels": [[0, 1, 2, 3]], "scores": [1], "tool_name": "Probe_Tool"})
        for protocol in ALIASES:
            with self.subTest(protocol=protocol):
                probe = ProbeCapability(detect)
                queue = []
                for turn in ["first", "second"]:
                    queue += [model_response(protocol, calls=[(turn, "Probe_Tool", '{"query":"a"}')]), model_response(protocol, "<answer>1</answer>")]
                queue += [model_response(protocol, "<answer>no tools</answer>")]
                agent, fixed, _ = make_agent(protocol, queue, probe)
                left = agent.generate("First", task_id="row-one")
                right = agent.generate("Second", task_id="row-two")
                empty = agent.generate("Third", task_id="row-three")
                self.assertIsNot(probe.invocations[0][1], probe.invocations[1][1])
                for result in [left, right]:
                    detections = result["extra_records"]["tool_detections"]
                    self.assertEqual(detections, [{"boxes_xyxy_pixels": [[0.0, 1.0, 2.0, 3.0]], "scores": [1.0], "tool_name": "Probe_Tool", "tool_call_idx": 0}])
                self.assertEqual(empty["extra_records"]["tool_detections"], [])
                for request in [fixed.requests[2], fixed.requests[4]]:
                    self.assertNotIn("tool_generated_image_1", json.dumps(request))
                    self.assertNotIn("webpage_url_1", json.dumps(request))

    def test_grounding_without_overlay_reaches_model_evidence_and_detection_sidecar(self):
        from openvlharness.capabilities.grounding import Capability

        for protocol in ALIASES:
            with self.subTest(protocol=protocol):
                client = Mock()
                client.execute.return_value = {"ok": True, "result": {"prompt": "shoe",
                    "pred_boxes": [[10, 5, 40, 20]], "pred_scores": [0.8], "orig_img_h": 100, "orig_img_w": 200}}
                capability = Capability(backend="sam3_segment", client_config={}, client=client)
                agent, fixed, _ = make_agent(protocol, [
                    model_response(protocol, calls=[("ground", capability.name, '{"query":"shoe","image_id":"input_image_1"}')]),
                    model_response(protocol, '<answer>[{"bbox_2d": [50, 50, 200, 200]}]</answer>')], capability)
                result = agent.generate([{"type": "image", "value": IMAGE}, {"type": "text", "value": "Find shoes"}])
                detections = result["extra_records"]["tool_detections"]
                self.assertEqual(len(detections), 1)
                self.assertEqual(detections[0]["boxes_xyxy_pixels"], [[10, 5, 40, 20]])
                self.assertEqual(detections[0]["scores"], [0.8])
                self.assertIn("bbox_2d", json.dumps(fixed.requests[1]))
                self.assertNotIn("tool_generated_image_", json.dumps(wire_messages(protocol, fixed.requests[1])))

    def test_caller_input_and_system_prompt_are_not_mutated_between_runs(self):
        for protocol in ALIASES:
            with self.subTest(protocol=protocol):
                agent, fixed, _ = make_agent(protocol, [model_response(protocol, "<answer>1</answer>"), model_response(protocol, "<answer>2</answer>")])
                original = [{"type": "text", "value": "private rule", "role": "system"}, {"type": "text", "value": "Question"}]
                before = copy.deepcopy(original)
                agent.generate(original)
                agent.generate("Next")
                self.assertEqual(original, before)
                self.assertEqual(agent.system_prompt, "fixture system")
                self.assertIn("private rule", contents(wire_messages(protocol, fixed.requests[0])[0]))
                self.assertNotIn("private rule", contents(wire_messages(protocol, fixed.requests[1])[0]))


if __name__ == "__main__":
    unittest.main()
