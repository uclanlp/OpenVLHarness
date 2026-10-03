"""Offline protocol acceptance tests; goldens come from the pinned old release."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import requests

from openvlharness.contracts import CapabilityResult
from openvlharness.services.models import AuxiliaryModel, ModelAdapter, failure_text, load_profile


GOLDEN = json.loads((Path(__file__).parent / "fixtures/model_protocol/old_release.json").read_text())
IMAGE = GOLDEN["image_data_url"]
ALIASES = {"chat_completions": "qwen3-vl-8b-instruct", "responses": "gpt-6-luna"}


def adapter(protocol="chat_completions", **kwargs):
    return ModelAdapter({"alias": ALIASES[protocol], "base": "http://fixture.invalid:8222/v1", **kwargs})


def response(status=200, body=None):
    return SimpleNamespace(ok=status < 400, status_code=status, reason="fixture error",
                           text=json.dumps(body), json=lambda: copy.deepcopy(body))


class ModelProfileTests(unittest.TestCase):
    def test_exact_alias_and_explicit_option_allowlist_fail_before_http(self):
        invalid = [
            {"alias": "qwen-prefix-guess"},
            {"alias": "qwen3-vl-8b-instruct", "debug_tool_loop": True},
            {"alias": "qwen3-vl-8b-instruct", "reasoning_effort": "medium"},
            {"alias": "gpt-6-luna", "temperature": 0.7},
            {"alias": "gpt-6-luna", "previous_response_id": "resp-secret"},
            {"alias": "gpt-6-luna", "extra_body": {"unknown": 1}},
            {"alias": "gpt-6-luna", "reasoning_effort": "invented"},
            {"alias": "gpt-6-luna", "max_tokens": True},
            {"alias": "qwen3-vl-8b-instruct", "return_token_ids": "true"},
            {"alias": "gpt-6-luna", "return_token_ids": True},
            {"alias": "qwen3-vl-8b-instruct", "retry": -1},
            {"alias": "qwen3-vl-8b-instruct", "timeout": 0},
        ]
        with patch("openvlharness.services.models.requests.post") as post:
            for config in invalid:
                with self.subTest(config=config), self.assertRaises(ValueError):
                    ModelAdapter({"base": "http://fixture.invalid", **config})
            post.assert_not_called()

    def test_root_v1_full_endpoint_and_protocol_mismatch(self):
        for protocol, suffix in [("chat_completions", "chat/completions"), ("responses", "responses")]:
            for path in ["", "/v1", "/v1/", f"/v1/{suffix}"]:
                with self.subTest(protocol=protocol, path=path):
                    value = adapter(protocol, base="https://fixture.invalid" + path)
                    self.assertEqual(value.endpoint, f"https://fixture.invalid/v1/{suffix}")
            wrong = "responses" if protocol == "chat_completions" else "chat/completions"
            with self.assertRaises(ValueError):
                adapter(protocol, base=f"https://fixture.invalid/v1/{wrong}")
        for base in ["file:///tmp/model", "fixture.invalid", "https://fixture.invalid?key=x", "https://fixture.invalid#x"]:
            with self.subTest(base=base), self.assertRaises(ValueError):
                adapter(base=base)

    def test_profile_load_is_independent(self):
        first = load_profile("qwen3-vl-8b-instruct")
        first["allowed_options"].append("unexpected")
        self.assertNotIn("unexpected", load_profile("qwen3-vl-8b-instruct")["allowed_options"])


class ModelWireTests(unittest.TestCase):
    def test_output_budget_reaches_provider_with_default_and_overrides(self):
        for protocol in ALIASES:
            for config in ({}, {"max_tokens": 16000}):
                for options in (None, {}, {"max_tokens": 73}):
                    with self.subTest(protocol=protocol, config=config, options=options):
                        model = adapter(protocol, **config)
                        reserved = config.get("max_tokens", 4096)
                        expected = reserved if not options else options["max_tokens"]
                        with patch("openvlharness.services.models.requests.post", return_value=response(body={})) as post:
                            model.request([], options=options)
                        wire = post.call_args.kwargs["json"]
                        field = "max_tokens" if protocol == "chat_completions" else "max_output_tokens"
                        self.assertEqual(wire[field], expected)
                        self.assertEqual(model.max_tokens, reserved)
                        self.assertEqual(model.options["max_tokens"], reserved)

    def test_native_request_bodies_match_old_and_leave_inputs_unchanged(self):
        for case in GOLDEN["wire"]:
            with self.subTest(protocol=case["protocol"]):
                model = adapter(case["protocol"], **case["config"])
                messages, tools = copy.deepcopy(case["input"]), copy.deepcopy(case["tools"])
                actual = model.payload(messages, tools, seed=case["seed"])
                self.assertEqual(actual, case["expected"])
                self.assertEqual(messages, case["input"])
                self.assertEqual(tools, case["tools"])
                actual["tools"][0]["type"] = "mutated"
                self.assertEqual(tools, case["tools"])

    def test_responses_encodes_role_text_images_and_file_passthrough_like_old(self):
        model = adapter("responses")
        for case in GOLDEN["content_encoding"]:
            with self.subTest(role=case["role"], content=case["input"]):
                message = {"role": case["role"], "content": copy.deepcopy(case["input"])}
                actual = model.encode_message(message)
                self.assertEqual(actual, {"type": "message", "role": case["role"], "content": case["expected"]})
                self.assertEqual(message["content"], case["input"])

    def test_chat_encode_copies_original_image_details(self):
        message = {"role": "user", "content": [{"type": "image_url", "image_url": {"url": IMAGE, "detail": "high"}}]}
        actual = adapter().encode_message(message)
        self.assertEqual(actual, message)
        actual["content"][0]["image_url"]["detail"] = "low"
        self.assertEqual(message["content"][0]["image_url"]["detail"], "high")

    def test_no_tool_payload_omits_tool_fields(self):
        for protocol in ALIASES:
            with self.subTest(protocol=protocol):
                body = adapter(protocol).payload([], [], "none", seed=4)
                self.assertNotIn("tools", body)
                self.assertNotIn("tool_choice", body)
                if protocol == "responses":
                    self.assertFalse(body["store"])
                    self.assertNotIn("seed", body)
                    self.assertNotIn("previous_response_id", body)

    def test_tool_result_formats_keep_call_id_and_processed_images(self):
        call = {"id": "retained-call", "name": "Probe_Tool"}
        evidence = [{"type": "image_url", "image_url": {"url": IMAGE}},
                    {"type": "text", "text": " first "}, {"type": "text", "text": "second "}]
        result = CapabilityResult(ok=True, evidence=evidence, payload={"ok": True, "result": "raw"})
        immediate, deferred = adapter().result_messages(call, result)
        self.assertEqual(immediate, [{"role": "tool", "tool_call_id": "retained-call", "name": "Probe_Tool", "content": "ok"}])
        self.assertEqual(deferred, [{"role": "user", "content": evidence}])
        immediate, deferred = adapter("responses").result_messages(call, result)
        self.assertEqual(deferred, [])
        self.assertEqual(immediate, [{"type": "function_call_output", "call_id": "retained-call", "output": [
            {"type": "input_text", "text": "first \nsecond"}, {"type": "input_image", "image_url": IMAGE}]}])

    def test_function_outputs_and_failure_text_match_old(self):
        model = adapter("responses")
        for case in GOLDEN["function_outputs"]:
            with self.subTest(payload=case["payload"], deferred=case["deferred"]):
                actual = model._build_function_call_output(call_id="retained-call", normalized_payload=case["payload"],
                                                           deferred_message=case["deferred"], run_memory=None)
                self.assertEqual(actual, case["expected"])
                if not case["payload"]["ok"]:
                    self.assertEqual(failure_text(case["payload"]), case["expected"]["output"])

    def test_generate_text_uses_separate_service_budget_and_protocol_defaults(self):
        for protocol in ALIASES:
            with self.subTest(protocol=protocol):
                captured = []
                body = {"choices": [{"message": {"content": "service answer"}}]} if protocol == "chat_completions" else {
                    "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "service answer"}]}]}
                model = adapter(protocol, max_tokens=32768)
                model.requester = lambda payload: (captured.append(copy.deepcopy(payload)) or body)
                self.assertEqual(model.generate_text([{"role": "user", "content": "summarize"}]), "service answer")
                wire = captured[0]
                self.assertNotIn("tools", wire)
                if protocol == "chat_completions":
                    self.assertEqual(wire["max_tokens"], 4096)
                    self.assertEqual(wire["temperature"], 0)
                else:
                    self.assertEqual(wire["max_output_tokens"], 4096)
                    self.assertEqual(wire["reasoning"], {"effort": "medium"})
                    self.assertEqual(wire["input"][0]["content"], [{"type": "input_text", "text": "summarize"}])
                self.assertEqual(model.options["max_tokens"], 32768)


class ModelParsingTests(unittest.TestCase):
    def test_argument_decoding_and_missing_id_name_match_old(self):
        for protocol, cases in GOLDEN["normalization"].items():
            model = adapter(protocol)
            for case in cases:
                with self.subTest(protocol=protocol, case=case["name"]):
                    # Normalize before the independent single-call policy.
                    method = model._normalize_tool_calls if protocol == "chat_completions" else model._normalize_responses_tool_calls
                    self.assertEqual(method(copy.deepcopy(case["input"]), case["turn_idx"]), case["expected"])

    def test_malformed_top_level_response_fails_loudly(self):
        for protocol, bodies in [("chat_completions", [None, [], {}, {"choices": []}, {"choices": [None]}, {"choices": [{"message": "text"}]}]),
                                  ("responses", [None, [], {}, {"output": None}, {"output": False},
                                                 {"output": 0}, {"output": ""}, {"output": {}},
                                                 {"output": "not a list"}])]:
            for body in bodies:
                with self.subTest(protocol=protocol, body=body), self.assertRaises(RuntimeError):
                    adapter(protocol).parse(body, 0)

    def test_explicit_empty_responses_output_is_valid(self):
        self.assertEqual(adapter("responses").parse({"output": []}, 0), ("", [], None))

    def test_multi_call_policy_preserves_every_id_and_suppressed_parse_error(self):
        chat_calls = [{"id": "first", "function": {"name": "A", "arguments": "{bad"}},
                      {"id": "second", "function": {"name": "B", "arguments": "{}"}}]
        _, calls, _ = adapter().parse({"choices": [{"message": {"tool_calls": chat_calls}}]}, 2, phase="final_text_only")
        self.assertEqual([call["id"] for call in calls], ["first", "second"])
        for index, call in enumerate(calls, 1):
            error = call["parse_error"]
            self.assertEqual(error["code"], "multiple_tool_calls_not_allowed")
            self.assertEqual(error["details"]["tool_call_index"], index)
            self.assertEqual(error["details"]["turn"], 3)
            self.assertEqual(error["details"]["phase"], "final_text_only")
        self.assertEqual(calls[0]["parse_error"]["details"]["suppressed_error"]["code"], "invalid_tool_arguments_json")
        resp_calls = [{"type": "function_call", "call_id": call["id"], **call["function"]} for call in chat_calls]
        _, calls, _ = adapter("responses").parse({"output": resp_calls}, 2)
        self.assertEqual([call["id"] for call in calls], ["first", "second"])
        self.assertEqual([call["parse_error"]["code"] for call in calls], ["multi_tool_call_rejected"] * 2)

    def test_responses_text_concatenation_drops_reasoning_and_nonassistant_items(self):
        output = [{"type": "reasoning", "id": "reasoning-id", "encrypted_content": "opaque"},
                  {"type": "message", "role": "user", "content": [{"type": "output_text", "text": "ignored"}]},
                  {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": " first"}, {"type": "text", "text": "second "}]}]
        text, calls, original = adapter("responses").parse({"output": output}, 0)
        self.assertEqual((text, calls, original), ("firstsecond", [], None))

    def test_assistant_call_order_differs_by_protocol(self):
        for protocol in ALIASES:
            raw = {"id": "keep-id", "function": {"name": "Probe_Tool", "arguments": "{}"}}
            body = {"choices": [{"message": {"content": "explanation", "tool_calls": [raw]}}]} if protocol == "chat_completions" else {
                "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "explanation"}]},
                           {"type": "function_call", "call_id": "keep-id", **raw["function"]}]}
            model = adapter(protocol)
            text, calls, original = model.parse(body, 0)
            history = []
            model.append_assistant_calls(history, text, calls, original)
            if protocol == "chat_completions":
                self.assertEqual(len(history), 1)
                self.assertEqual(history[0]["tool_calls"][0]["id"], "keep-id")
                self.assertEqual(history[0]["content"], "explanation")
            else:
                self.assertEqual([item["type"] for item in history], ["function_call", "message"])
                self.assertEqual(history[0]["call_id"], "keep-id")

    def test_final_answer_policy_is_not_interchangeable(self):
        for protocol in ALIASES:
            model = adapter(protocol)
            self.assertEqual(model.final_answer("<thinking>x</thinking><ANSWER> 7 </ANSWER>"), "7")
            self.assertEqual(model.final_answer("plain answer"), "plain answer" if protocol == "responses" else "")


class ModelTransportTests(unittest.TestCase):
    def test_headers_timeout_are_transport_only_and_body_matches_payload(self):
        model = adapter("responses", key="fixture-key", timeout=17, retry=0, max_tokens=32)
        with patch("openvlharness.services.models.requests.post", return_value=response(body={"output": []})) as post:
            self.assertEqual(model.request([]), {"output": []})
        self.assertEqual(post.call_args.args, ("http://fixture.invalid:8222/v1/responses",))
        kwargs = post.call_args.kwargs
        self.assertEqual(kwargs["headers"], {"Content-Type": "application/json", "Authorization": "Bearer fixture-key"})
        self.assertEqual(kwargs["timeout"], 17)
        self.assertEqual(kwargs["json"], model.payload([]))
        self.assertNotIn("key", kwargs["json"])

    def test_default_native_retry_budget_is_eight_retries_nine_attempts(self):
        model = adapter()
        with patch("openvlharness.services.models.requests.post", side_effect=requests.Timeout("fixture timeout")) as post, \
             patch("openvlharness.services.http.time.sleep") as sleep, \
             patch("openvlharness.services.http.random.random", return_value=0):
            with self.assertRaises(requests.Timeout):
                model.request([])
        self.assertEqual(post.call_count, 9)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 4, 8, 16, 30, 30, 30, 30])

    def test_transient_http_retry_reuses_exact_body_and_fatal_400_does_not_retry(self):
        model = adapter("responses", retry=2)
        sequence = [response(429, {"error": {"message": "slow down"}}), response(503, {"detail": "unavailable"}), response(200, {"output": []})]
        with patch("openvlharness.services.models.requests.post", side_effect=sequence) as post, \
             patch("openvlharness.services.http.time.sleep"):
            self.assertEqual(model.request([]), {"output": []})
        self.assertEqual(post.call_count, 3)
        self.assertTrue(all(call.kwargs["json"] == post.call_args.kwargs["json"] for call in post.call_args_list))
        with patch("openvlharness.services.models.requests.post", return_value=response(400, {"error": {"message": "invalid schema"}})) as post, \
             patch("openvlharness.services.http.time.sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "400 invalid schema"):
                model.request([])
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()


class ModelContextTests(unittest.TestCase):
    def test_context_window_must_leave_room_after_output_and_margin(self):
        for protocol, alias in ALIASES.items():
            for options in ({}, {"max_tokens": 100}):
                budget = options.get("max_tokens", 4096)
                for window in (1, budget, budget + 511, budget + 512):
                    with self.subTest(protocol=protocol, options=options, window=window):
                        with self.assertRaisesRegex(ValueError, r"max_context_length.*max_tokens.*512"):
                            ModelAdapter({"alias": alias, "base": "http://fixture.invalid", **options},
                                         max_context_length=window)

    def test_context_window_accepts_positive_input_budget_or_provider_limit(self):
        for protocol, alias in ALIASES.items():
            for options in ({}, {"max_tokens": 100}):
                budget = options.get("max_tokens", 4096)
                for window in (None, budget + 513, 168000):
                    with self.subTest(protocol=protocol, options=options, window=window):
                        model = ModelAdapter({"alias": alias, "base": "http://fixture.invalid", **options},
                                             max_context_length=window)
                        self.assertEqual(model.max_context_length, window)
                        self.assertEqual(model.max_tokens, budget)
                        field = "max_tokens" if protocol == "chat_completions" else "max_output_tokens"
                        self.assertEqual(model.payload([])[field], budget)

    def test_token_counter_replaces_image_only_in_copy_and_includes_tools(self):
        model = adapter()
        model._tokenizer = Mock()
        model._tokenizer.apply_chat_template.return_value = [1, 2, 3]
        messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": IMAGE}}, {"type": "text", "text": "question"}]}]
        tools = GOLDEN["wire"][0]["tools"]
        self.assertEqual(model._count_tokens(messages, tools), 3)
        call = model._tokenizer.apply_chat_template.call_args
        self.assertEqual(call.args[0][0]["content"][0], {"type": "text", "text": "[image]"})
        self.assertEqual(call.kwargs, {"tokenize": True, "add_generation_prompt": True, "tools": tools})
        self.assertEqual(messages[0]["content"][0]["image_url"]["url"], IMAGE)
        model._tokenizer.apply_chat_template.side_effect = ValueError("unsupported template")
        self.assertIsNone(model._count_tokens(messages, tools))

    def test_compaction_protects_initial_block_latest_exchange_and_latest_inventory(self):
        messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "x" * 120, "tool_calls": [{"id": "old"}]},
                    {"role": "tool", "content": "x" * 120},
                    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": IMAGE}}]},
                    {"role": "user", "content": "Environment Update: " + "z" * 120},
                    {"role": "assistant", "content": "latest", "tool_calls": [{"id": "new"}]},
                    {"role": "tool", "content": "latest result"}]
        before = copy.deepcopy(messages)
        model = adapter()
        model._count_tokens = lambda msgs, tools=None: sum(len(str(m.get("content", ""))) for m in msgs)
        self.assertTrue(model._compact_history(messages, 280))
        for index in [0, 1, 5, 6, 7]:
            self.assertEqual(messages[index], before[index])
        for index in [2, 3, 4]:
            self.assertEqual(messages[index]["content"], "[Truncated to save context]")
        self.assertEqual(messages[2]["tool_calls"], before[2]["tool_calls"])

    def test_compacted_request_preserves_tool_call_declarations_and_results(self):
        tools = GOLDEN["wire"][0]["tools"]
        for call_count in (1, 2):
            calls = [{"id": f"old-{i}", "type": "function",
                      "function": {"name": "Probe_Tool", "arguments": '{"image_id": "image-0"}'}}
                     for i in range(call_count)]
            history = [{"role": "system", "content": "system"},
                       {"role": "user", "content": "question"},
                       {"role": "assistant", "content": "a" * 1000, "tool_calls": calls},
                       *[{"role": "tool", "tool_call_id": call["id"], "content": "r" * 1000}
                         for call in calls],
                       {"role": "user", "content": "Environment Update: image-0"},
                       {"role": "assistant", "content": "latest", "tool_calls": [
                           {"id": "latest", "type": "function",
                            "function": {"name": "Probe_Tool", "arguments": "{}"}}]},
                       {"role": "tool", "tool_call_id": "latest", "content": "latest result"}]
            # Exercise early returns after the assistant and after each tool result.
            for stop_index in range(2, 3 + call_count):
                with self.subTest(call_count=call_count, stop_index=stop_index):
                    messages, expected = copy.deepcopy(history), copy.deepcopy(history)
                    for index in range(2, stop_index + 1):
                        expected[index]["content"] = "[Truncated to save context]"
                    model = adapter(max_tokens=100)
                    model._tokenizer = object()
                    model._count_tokens = lambda msgs, tools=None: len(json.dumps(msgs))
                    model.max_context_length = 100 + 512 + model._count_tokens(expected)
                    effective_tools, choice = model._ensure_context_fits(messages, tools, "auto")
                    self.assertEqual((effective_tools, choice), (tools, "auto"))
                    model.requester = Mock(return_value={})
                    model.request(messages, effective_tools, choice)
                    wire = model.requester.call_args.args[0]["messages"]
                    self.assertEqual(wire, expected)
                    pending = set()
                    for message in wire:
                        if message["role"] == "tool":
                            self.assertIn(message["tool_call_id"], pending)
                            pending.remove(message["tool_call_id"])
                        else:
                            self.assertFalse(pending)
                            pending.update(call["id"] for call in message.get("tool_calls", []))
                    self.assertFalse(pending)

    def test_large_call_arguments_use_fallback_when_content_compaction_cannot_fit(self):
        model = adapter(max_tokens=100)
        model.max_context_length = 1000
        model._tokenizer = object()
        model._count_tokens = lambda msgs, tools=None: len(json.dumps(msgs))
        messages = [{"role": "user", "content": "question"},
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"id": "large", "type": "function",
                         "function": {"name": "Probe_Tool", "arguments": json.dumps({"text": "x" * 1000})}}]},
                    {"role": "tool", "tool_call_id": "large", "content": "ok"},
                    {"role": "assistant", "content": "latest"}]
        self.assertEqual(model._ensure_context_fits(messages, GOLDEN["wire"][0]["tools"], "auto"), (None, "none"))
        self.assertEqual(messages, [{"role": "user", "content": "question"}])

    def test_context_budget_includes_output_and_margin_then_falls_back_to_first_user(self):
        model = adapter(max_tokens=100)
        model.max_context_length = 1000
        model._tokenizer = object()
        tools = GOLDEN["wire"][0]["tools"]
        messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "first"}, {"role": "user", "content": "second"}]
        with patch.object(model, "_count_tokens", return_value=388):
            self.assertEqual(model._ensure_context_fits(messages, tools, "auto"), (tools, "auto"))
        with patch.object(model, "_count_tokens", return_value=389):
            self.assertEqual(model._ensure_context_fits(messages, tools, "auto"), (None, "none"))
        self.assertEqual(messages, [{"role": "user", "content": "first"}])

    def test_missing_tokenizer_leaves_provider_validation_in_charge(self):
        model = adapter(max_tokens=100)
        model.max_context_length = 1000
        messages = [{"role": "user", "content": "x" * 5000}]
        self.assertEqual(model._ensure_context_fits(messages, ["tool"], "auto"), (["tool"], "auto"))
        self.assertEqual(len(messages[0]["content"]), 5000)


class AuxiliaryModelTests(unittest.TestCase):
    def setUp(self):
        # Core tests can exercise the optional SDK boundary without installing
        # OpenAI or allowing an accidental request to a real endpoint.
        self.constructor = Mock()
        self.sdk = self.constructor.return_value
        self.module_patch = patch.dict("sys.modules", {"openai": SimpleNamespace(OpenAI=self.constructor)})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def engine(self, **options):
        return AuxiliaryModel({"alias": "qwen3-vl-8b-instruct", "base": "http://fixture.invalid/v1",
                               "key": "fixture-key", **options})

    def set_response(self, text):
        body = {"choices": [{"message": {"content": text}}]}
        self.sdk.chat.completions.create.return_value = SimpleNamespace(model_dump=lambda: copy.deepcopy(body))

    def test_sdk_bodies_match_old_multimodal_greedy_generation_and_system_handling(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/model_protocol/auxiliary_old_release.json").read_text())
        self.set_response("fixture response")
        engine = self.engine()
        for case in fixture["cases"]:
            with self.subTest(budget=case["budget"], system=case["system"]):
                original = copy.deepcopy(case["input"])
                result = engine.generate(original, system_prompt=case["system"], max_tokens=case["budget"], temperature=0.0, top_p=1.0)
                self.assertEqual(result, "fixture response")
                self.assertEqual(self.sdk.chat.completions.create.call_args.kwargs, case["expected"])
                self.assertEqual(original, case["input"])
        self.sdk.responses.create.assert_not_called()

    def test_auxiliary_defaults_preserve_sdk_two_retries_and_five_second_connect_timeout(self):
        self.engine()
        config = self.constructor.call_args.kwargs
        self.assertEqual(config["max_retries"], 2)
        self.assertEqual(config["timeout"].connect, 5)
        self.assertEqual((config["timeout"].read, config["timeout"].write, config["timeout"].pool), (600, 600, 600))

    def test_explicit_auxiliary_transport_options_and_endpoint_are_respected(self):
        self.engine(base="http://fixture.invalid/v1/chat/completions", timeout=17, retry=4)
        config = self.constructor.call_args.kwargs
        self.assertEqual(config["base_url"], "http://fixture.invalid/v1")
        self.assertEqual(config["max_retries"], 4)
        self.assertEqual(config["timeout"].connect, 5)
        self.assertEqual(config["timeout"].read, 17)

    def test_search_tools_keep_multimodal_bytes_order_and_greedy_4096_budget(self):
        from openvlharness.services.tools.serper_text_search.tool import Text_Search_Tool
        from openvlharness.services.tools.serper_image_search.tool import Image_Search_Tool
        blue = json.loads((Path(__file__).parent / "fixtures/memory_old_release.json").read_text())["data_urls"]["blue"]
        raw = ("1. Red object\n   Source: https://fixture.invalid/red\n   Image: https://fixture.invalid/red.png <img>\n"
               "2. Blue object\n   Source: https://fixture.invalid/blue\n   Image: https://fixture.invalid/blue.png <img>")
        self.set_response(json.dumps({"kept_indices": [2, 1], "summary": "Two distinct colors."}))
        engine = self.engine()
        for kind in ["text_images", "reverse_image"]:
            with self.subTest(kind=kind):
                if kind == "text_images":
                    tool = Text_Search_Tool(llm_engine=engine)
                    summary, images = tool._summarize_with_llm("colored objects", "images", raw, keep_k=3,
                                                              image_data_urls_by_rank={1: IMAGE, 2: blue})
                else:
                    tool = Image_Search_Tool(llm_engine=engine)
                    summary, images = tool._summarize_with_llm(raw, query_image_data_url=blue,
                                                              image_data_urls_by_rank={1: IMAGE, 2: blue})
                wire = self.sdk.chat.completions.create.call_args.kwargs
                self.assertEqual(wire["max_tokens"], 4096)
                self.assertEqual((wire["temperature"], wire["top_p"]), (0.0, 1.0))
                self.assertEqual(wire["extra_body"], {"chat_template_kwargs": {"enable_thinking": False}})
                self.assertEqual([m["role"] for m in wire["messages"]], ["system", "user"])
                parts = wire["messages"][1]["content"]
                actual_images = [p["image_url"]["url"] for p in parts if p["type"] == "image_url"]
                self.assertEqual(actual_images, [IMAGE, blue] if kind == "text_images" else [blue, IMAGE, blue])
                self.assertEqual(images, [blue, IMAGE])
                self.assertIn("[Summary]", summary)
                self.assertIn("Two distinct colors.", summary)
                self.assertIn("https://fixture.invalid/blue", summary)
                self.assertNotEqual(summary, raw)

    def test_webpage_extractor_uses_16384_greedy_budget_and_parsed_evidence(self):
        from openvlharness.services.tools.serper_webpage_visit.tool import Webpage_Visit_Tool
        self.set_response('{"evidence":"The tower opened in 1889.","summary":"It opened in 1889."}')
        tool = Webpage_Visit_Tool(llm_engine=self.engine())
        result = tool._extract_content("The tower opened in 1889. Tickets cost extra.", "Find opening year", "A tower")
        self.assertEqual(result, {"evidence": "The tower opened in 1889.", "summary": "It opened in 1889."})
        wire = self.sdk.chat.completions.create.call_args.kwargs
        self.assertEqual((wire["max_tokens"], wire["temperature"], wire["top_p"]), (16384, 0.0, 1.0))
        self.assertEqual(wire["extra_body"], {"chat_template_kwargs": {"enable_thinking": False}})
        self.assertIn("The tower opened in 1889. Tickets cost extra.", wire["messages"][1]["content"])
        self.assertIn("Find opening year", wire["messages"][1]["content"])

    def test_alternate_registered_responses_summarizer_preserves_multimodal_input(self):
        body = {"output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "summary"}]}]}
        self.sdk.responses.create.return_value = SimpleNamespace(model_dump=lambda: copy.deepcopy(body))
        engine = self.engine(alias="gpt-6-luna")
        result = engine.generate([{"role": "user", "content": [{"type": "text", "text": "identify"},
                           {"type": "image_url", "image_url": {"url": IMAGE}}]}], system_prompt="summarize", max_tokens=4096)
        self.assertEqual(result, "summary")
        wire = self.sdk.responses.create.call_args.kwargs
        self.assertEqual(wire["input"][1]["content"], [{"type": "input_text", "text": "identify"}, {"type": "input_image", "image_url": IMAGE}])
        self.assertEqual((wire["max_output_tokens"], wire["reasoning"], wire["store"]), (4096, {"effort": "medium"}, False))
        self.assertNotIn("temperature", wire)
        self.assertNotIn("chat_template_kwargs", wire)
        self.sdk.chat.completions.create.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("fastapi"), "server extra is required for router wiring test")
    def test_router_coder_uses_separate_120_second_timeout_and_4096_budget(self):
        from openvlharness.services.server import create_app
        for protocol, alias in ALIASES.items():
            with self.subTest(protocol=protocol):
                config = {"agent_name": "MainAgent", "agent": {"model_service": "orchestrator"},
                    "memory": {"modules": [{"name": "image"}, {"name": "tool_context"}, {"name": "url"}]},
                    "capabilities": {"registries": ["paper"], "tools": {"coding": {"generator": "python_coding_agent", "executor": "python_executor_local"}}},
                    "services": {"client": {"base_url": "http://fixture.invalid"}, "server": {},
                        "tools": {"python_coding_agent": {"model_service": "orchestrator", "context": "fresh"}, "python_executor_local": {"transport": "local"}},
                        "models": {"orchestrator": {"alias": alias, "base": "http://fixture.invalid/v1", "timeout": 600, "max_tokens": 16000, "retry": 8}}}}
                app = create_app(config)
                coder = app.state.registry["Python_Coding_Agent_Tool"][0]
                native = ({"choices": [{"message": {"content": "```python\nprint(7)\n```"}}]} if protocol == "chat_completions" else
                          {"output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "```python\nprint(7)\n```"}]}]})
                with patch("openvlharness.services.models.requests.post", return_value=response(body=native)) as post:
                    self.assertEqual(coder.execute("Print seven"), {"generated_code": "print(7)"})
                self.assertEqual(post.call_args.kwargs["timeout"], 120)
                self.assertEqual(coder.model_adapter.retry, 8)
                wire = post.call_args.kwargs["json"]
                self.assertEqual(wire["max_tokens" if protocol == "chat_completions" else "max_output_tokens"], 4096)
                if protocol == "chat_completions":
                    self.assertEqual(wire["temperature"], 0)
                else:
                    self.assertEqual(wire["reasoning"], {"effort": "medium"})
                self.assertNotIn("tools", wire)


if __name__ == "__main__":
    unittest.main()
