"""Capability contracts replay fixed replies; no old package or servers needed.

Old evidence, binding descriptors, arrays, references and schemas are frozen in
fixtures/capabilities. OCR storage and the DINO adapter are intentional additions.
"""
import concurrent.futures
import base64
import copy
import dataclasses
import json
from pathlib import Path
import signal
import unittest
from unittest.mock import Mock, patch

import numpy as np

from openvlharness.capabilities import (
    camera_pose, coding, depth, grounding, image_search, ocr, text_search, web_visit, zoom,
)
from openvlharness.contracts import VirtualCall
from openvlharness.memory.store import MemoryStore
from openvlharness.services.client import ToolClient


FIXTURES = Path(__file__).parent / "fixtures/capabilities"
OLD = json.loads((FIXTURES / "old_release.json").read_text())
SCHEMAS = json.loads((FIXTURES / "public_schemas.json").read_text())
RED, BLUE, GREEN = (OLD["data_urls"][key] for key in ("red", "blue", "green"))
CAPABILITIES = [
    (grounding, "sam3_segment"), (ocr, "ocr"), (depth, "depth_estimation"),
    (camera_pose, "camera_trajectory"), (zoom, "zoom_in_image_by_bbox"),
    (text_search, "serper_text_search"), (image_search, "serper_image_search"),
    (web_visit, "serper_webpage_visit"), (coding, "python_coding_agent"),
]
BY_NAME = {module.Capability.name: (module, backend) for module, backend in CAPABILITIES}


def corrected_depth_descriptors(descriptors):
    """Account for the explicit dimension-description correction in frozen fixtures."""
    expected = copy.deepcopy(descriptors)
    for descriptor in expected.values():
        if descriptor["source_tool"] == "Depth_Estimation_Tool" and "size" in descriptor["fields"]:
            descriptor["fields"]["size"]["description"] = "[height, width] of source image"
    return expected


def plain(value):
    if dataclasses.is_dataclass(value):
        return plain(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return {"values": value.tolist(), "dtype": str(value.dtype), "shape": list(value.shape)}
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, set):
        return sorted(value)
    return value


def envelope(name, response, *, ok=True, error=None):
    return {"ok": ok, "tool_name": name, "tool_call_id": "call_fixture",
            "result": copy.deepcopy(response), "error": error, "meta": {}}


def new_memory(count=2):
    memory = MemoryStore()
    memory.start([{"type": "image", "value": value} for value in (RED, BLUE)[:count]])
    return memory


def make_capability(name, response, *, backend=None, service_config=None, ok=True, error=None):
    module, default_backend = BY_NAME[name]
    client = Mock(spec=ToolClient)
    client.execute.return_value = envelope(name, response, ok=ok, error=error)
    capability = module.Capability(backend=backend or default_backend, client_config={},
                                   service_config=service_config, client=client)
    return capability, client


def replay(case, memory=None, *, response=None, service_config=None):
    memory = memory if memory is not None else new_memory()
    capability, client = make_capability(case["name"], case["response"] if response is None else response,
                                         service_config=service_config)
    call = VirtualCall("call_fixture", case["name"], copy.deepcopy(case["arguments"]))
    result = capability.invoke(call, memory)
    return result, memory, client


class CapabilityBoundaryTests(unittest.TestCase):
    def setUp(self):
        # Reference fixtures must never turn into downloads or actual HTTP calls.
        for target in ("openvlharness.memory.images._fetch_image_data_url", "requests.sessions.Session.request"):
            guard = patch(target, side_effect=AssertionError("unexpected network access in capability replay"))
            guard.start()
            self.addCleanup(guard.stop)

    def assert_old_replay(self, result, memory, case):
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.evidence, case["evidence"])
        # Backend/service metadata is new; all old model-visible prepared fields
        # must remain equal rather than forbidding extra diagnostic fields.
        self.assertEqual({key: result.prepared[key] for key in case["prepared"]}, case["prepared"])
        self.assertEqual(plain(memory.state.bindings.bindings), case["bindings"])
        self.assertEqual(memory.describe_bindings(), corrected_depth_descriptors(case["descriptors"]))
        actual_images = plain(memory.state.images)
        actual_images.pop("last_snapshot", None)
        self.assertEqual(actual_images, case["images"])
        actual_urls = plain(memory.state.urls)
        actual_urls.pop("last_webpage_counter", None)
        self.assertEqual(actual_urls, case["urls"])
        message = memory.build_environment_update()
        self.assertEqual([] if message is None else [message], case["environment"])

    def test_g01_sam3_binding_rle_overlay_evidence_and_detection_match_old(self):
        case = OLD["cases"]["sam3"]
        result, memory, client = replay(case)
        self.assert_old_replay(result, memory, case)
        client.execute.assert_called_once_with(case["name"], case["resolved_arguments"], tool_call_id="call_fixture")
        expected_detection = {key: value for key, value in case["detections"][0].items() if key != "tool_call_idx"}
        self.assertEqual(result.detection, expected_detection)
        self.assertEqual(result.data["boxes"], case["response"]["pred_boxes"])
        self.assertEqual(result.data["boxes_normalized"], [[1, 5, 402, 252], [501, 301, 1000, 998]])
        self.assertEqual(result.data["masks"], case["response"]["pred_masks"])
        self.assertEqual(result.data["num_masks"], 2)
        self.assertEqual(result.data["rendered_image"], "tool_generated_image_1")
        self.assertEqual(plain(result.artifact_refs), [{"kind": "image", "id": "tool_generated_image_1"},
                                                     {"kind": "binding", "id": "segment_red_shoe"}])

    def test_sam3_detections_and_evidence_do_not_require_an_overlay(self):
        raw = {"prompt": "shoe", "pred_boxes": [[10.25, 5.5, 40.75, 20.0]],
               "pred_scores": [0.81234], "orig_img_h": 100, "orig_img_w": 200}
        for scores in (True, False):
            with self.subTest(scores=scores):
                response = copy.deepcopy(raw)
                if not scores:
                    del response["pred_scores"]
                capability, _ = make_capability("Visual_Grounding_Tool", response)
                result = capability.invoke(VirtualCall("without_overlay", capability.name,
                    {"query": "shoe", "image_id": "input_image_1"}), new_memory())
                self.assertTrue(result.ok, result.error)
                self.assertEqual(result.data["boxes"], raw["pred_boxes"])
                self.assertIsNone(result.data["rendered_image"])
                text = result.evidence[0]["text"]
                self.assertIn('"bbox_2d": [51, 55, 204, 200]', text)
                self.assertNotIn("labeled masks rendered", text)
                self.assertNotIn("did not successfully ground", text)
                if scores:
                    self.assertEqual(result.detection, {"tool_name": capability.name, "category_query": "shoe",
                        "boxes_xyxy_pixels": raw["pred_boxes"], "scores": raw["pred_scores"],
                        "image_width": 200, "image_height": 100})
                else:
                    self.assertIsNone(result.detection)
    def test_g01_empty_sam3_is_success_with_old_evidence_and_no_detection(self):
        case = OLD["cases"]["sam3_empty"]
        result, memory, _ = replay(case)
        self.assert_old_replay(result, memory, case)
        self.assertIsNone(result.detection)
        self.assertEqual(result.data["boxes"], [])
        self.assertEqual(memory.state.bindings.bindings["segment_red_shoe"]["num_masks"], 0)

    def test_d01_depth_decodes_real_float32_array_with_exact_shape_and_values(self):
        case = OLD["cases"]["depth"]
        result, memory, _ = replay(case)
        self.assert_old_replay(result, memory, case)
        binding = memory.state.bindings.bindings["depth_result"]
        self.assertIsInstance(binding["depth_meters"], np.ndarray)
        self.assertEqual(binding["depth_meters"].dtype, np.dtype("float32"))
        self.assertEqual(binding["depth_meters"].shape, (2, 2))
        np.testing.assert_array_equal(binding["depth_meters"], [[0, 1.25], [2.5, 4]])
        self.assertNotIn("depth_meters_b64", binding)
        self.assertEqual(result.data["binding"], "depth_result")
        self.assertEqual(result.data["units"], "meters")

    def test_depth_rejects_malformed_arrays_before_registering_any_artifact(self):
        valid = OLD["cases"]["depth"]["response"]
        changes = [
            {"depth_meters_b64": "not base64!"}, {"depth_meters_b64": None},
            {"depth_meters_b64": ""}, {"depth_meters_b64": "AAAA"},
            {"depth_shape": [3, 2]}, {"depth_shape": [2, -1]}, {"depth_shape": [0, 2]},
            {"depth_shape": [True, 4]}, {"depth_shape": [2.0, 2]}, {"depth_shape": [4]},
            {"depth_meters_b64": base64.b64encode(np.array([1, 2, 3, np.nan], dtype=np.float32).tobytes()).decode()},
            {"size": [1, 0]},
        ]
        for updates in changes + [{"depth_shape": None}]:
            with self.subTest(updates=updates):
                response = dict(valid, **updates)
                if updates == {"depth_shape": None}:
                    del response["depth_shape"]
                capability, _ = make_capability("Depth_Estimation_Tool", response)
                memory = new_memory()
                before = copy.deepcopy(memory.state.images.registry)
                result = capability.invoke(VirtualCall("bad_depth", capability.name,
                    {"image_id": "input_image_1"}), memory)
                self.assertFalse(result.ok)
                self.assertEqual(result.error["code"], "invalid_tool_result")
                self.assertEqual(result.payload["tool_call_id"], "bad_depth")
                self.assertEqual(memory.state.images.registry, before)
                self.assertEqual(memory.state.bindings.bindings, {})
                self.assertEqual(result.evidence, [])

    def test_depth_rectangular_size_is_height_width_and_array_is_usable(self):
        array = np.arange(6, dtype=np.float32).reshape(2, 3)
        capability, _ = make_capability("Depth_Estimation_Tool", {"size": [2, 3], "depth_shape": [2, 3],
            "depth_meters_b64": base64.b64encode(array.tobytes()).decode()})
        memory = new_memory()
        result = capability.invoke(VirtualCall("rectangular_depth", capability.name,
            {"image_id": "input_image_1"}), memory)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.data["size"], [2, 3])
        binding = result.data["binding"]
        np.testing.assert_array_equal(memory.state.bindings.bindings[binding]["depth_meters"], array)
        self.assertEqual(memory.describe_bindings()[binding]["fields"]["size"]["description"],
                         "[height, width] of source image")

    def test_d01_camera_arrays_preserve_order_reference_frame_and_float32(self):
        case = OLD["cases"]["camera"]
        result, memory, client = replay(case)
        self.assert_old_replay(result, memory, case)
        client.execute.assert_called_once_with(case["name"], case["resolved_arguments"], tool_call_id="call_fixture")
        binding = memory.state.bindings.bindings["camera_trajectory_result"]
        for field, shape in (("intrinsics", (2, 3, 3)), ("extrinsics", (2, 3, 4))):
            self.assertIsInstance(binding[field], np.ndarray)
            self.assertEqual(binding[field].dtype, np.dtype("float32"))
            self.assertEqual(binding[field].shape, shape)
            np.testing.assert_array_equal(binding[field], np.asarray(case["response"][field], dtype=np.float32))
        self.assertEqual(binding["image_ids"], ["input_image_2", "input_image_1"])
        self.assertEqual(binding["reference_image_id"], "input_image_2")
        self.assertEqual(result.data["coordinate_system"], case["response"]["coordinate_system"])

    def test_zoom_keeps_old_overlay_reference_and_evidence(self):
        case = OLD["cases"]["zoom"]
        result, memory, _ = replay(case)
        self.assert_old_replay(result, memory, case)
        self.assertEqual(result.data["image"], "tool_generated_image_1")

    def test_search_preserves_old_url_registration_and_multimodal_image_order(self):
        for name in ("text_search", "text_search_images", "image_search"):
            with self.subTest(case=name):
                case = OLD["cases"][name]
                result, memory, client = replay(case)
                self.assert_old_replay(result, memory, case)
                client.execute.assert_called_once_with(case["name"], case["resolved_arguments"], tool_call_id="call_fixture")
                raw_text = case["response"] if isinstance(case["response"], str) else case["response"]["text"]
                self.assertEqual(result.data["text"], raw_text)
                if name != "text_search":
                    self.assertEqual(result.data["images"], ["tool_generated_image_1", "input_image_1"])
                    self.assertEqual([part["image_url"]["url"] for part in result.evidence if part["type"] == "image_url"], [GREEN, RED])
                    self.assertNotIn("__MULTIMODAL_IMG_REF_PLACEHOLDER__", result.evidence[0]["text"])

    def test_web_visit_resolves_search_reference_without_changing_old_evidence(self):
        memory = new_memory()
        memory.register_url("https://example.com/catalog", provenance="Text_Search_Tool")
        case = OLD["cases"]["web_visit"]
        result, memory, client = replay(case, memory)
        self.assert_old_replay(result, memory, case)
        client.execute.assert_called_once_with(case["name"], case["resolved_arguments"], tool_call_id="call_fixture")
        self.assertEqual(case["resolved_arguments"]["urls"], ["https://example.com/catalog"])

    def test_m03_missing_and_plural_resolution_metadata_match_old(self):
        for key, case in OLD["resolution"].items():
            with self.subTest(case=key):
                memory = new_memory(case["image_count"])
                capability, client = make_capability(case["name"], {})
                arguments = copy.deepcopy(case["arguments"])
                result = capability.invoke(VirtualCall("call_fixture", case["name"], arguments), memory)
                self.assertEqual(arguments, case["arguments"])
                if case["resolution"]["errors"]:
                    self.assertFalse(result.ok)
                    client.execute.assert_not_called()
                    self.assertEqual(result.error["code"], "unresolved_image_reference")
                    self.assertEqual(result.error["details"], case["resolution"]["errors"])
                    self.assertEqual(result.payload["meta"]["image_resolution"], case["resolution"])
                    self.assertEqual(memory.state.bindings.bindings, {})
                else:
                    self.assertTrue(result.ok, result.error)
                    client.execute.assert_called_once_with(case["name"], case["resolved_arguments"], tool_call_id="call_fixture")
                    self.assertEqual(result.payload["meta"].get("image_resolution", {"rewrites": [], "errors": []}), case["resolution"])

    def test_s01_exact_nine_public_schemas_are_independent_copies(self):
        actual = []
        for module, backend in CAPABILITIES:
            capability = module.Capability(backend=backend, client_config={}, client=Mock(spec=ToolClient))
            schema = capability.public_schema()
            actual.append(schema)
            modified = capability.public_schema()
            modified["function"]["parameters"]["properties"].clear()
            self.assertEqual(capability.public_schema(), schema)
        self.assertEqual(actual, SCHEMAS)
        self.assertEqual(len({schema["function"]["name"] for schema in actual}), 9)

    def test_s01_hidden_arguments_never_reach_service_or_memory(self):
        valid = {case["name"]: case["arguments"] for case in OLD["cases"].values()}
        valid["Python_Coding_Agent_Tool"] = {"objective": "Count"}
        for module, backend in CAPABILITIES:
            for hidden in ("image_base64", "_image_ids", "binding_descriptors", "backend", "summary_max_tokens"):
                with self.subTest(tool=module.Capability.name, argument=hidden):
                    memory = new_memory()
                    capability, client = make_capability(module.Capability.name, {})
                    args = {**valid[module.Capability.name], hidden: "injected"}
                    before = [plain(getattr(memory.state, key)) for key in ("images", "bindings", "urls")]
                    result = capability.invoke(VirtualCall("call_fixture", capability.name, args), memory)
                    self.assertFalse(result.ok)
                    self.assertEqual(result.error["code"], "invalid_tool_arguments")
                    client.execute.assert_not_called()
                    self.assertEqual([plain(getattr(memory.state, key)) for key in ("images", "bindings", "urls")], before)

    def test_o01_ocr_text_binding_is_intentional_correction_with_old_evidence(self):
        for key in ("ocr", "ocr_empty"):
            with self.subTest(case=key):
                case = OLD["cases"][key]
                result, memory, _ = replay(case)
                self.assertTrue(result.ok)
                self.assertEqual(result.evidence, case["evidence"])
                self.assertEqual(case["bindings"], {})  # The old omission is documented.
                self.assertEqual(memory.state.bindings.bindings, {"ocr_result": {"text": case["response"]["text"]}})
                descriptor = memory.describe_bindings()["ocr_result"]
                self.assertEqual(descriptor["source_tool"], "OCR_Tool")
                self.assertEqual(set(descriptor["fields"]), {"text"})
                self.assertIn("ocr_result", str(memory.build_environment_update()))
                self.assertEqual(plain(result.artifact_refs), [{"kind": "binding", "id": "ocr_result"}])
                self.assertIsNone(result.detection)

    def test_o01_failed_ocr_has_no_text_binding_or_evidence(self):
        memory = new_memory()
        error = {"code": "tool_execution_error", "message": "fixture OCR failure"}
        capability, _ = make_capability("OCR_Tool", None, ok=False, error=error)
        result = capability.invoke(VirtualCall("call_fixture", "OCR_Tool", {"image_id": "input_image_1"}), memory)
        self.assertFalse(result.ok)
        self.assertEqual(result.error, error)
        self.assertEqual(result.evidence, [])
        self.assertEqual(memory.state.bindings.bindings, {})

    def test_c01_c03_copied_execution_matches_old_and_never_persists_new_variables(self):
        case = OLD["coding"]["copied"]
        _, memory, _ = replay(OLD["cases"]["depth"])
        before = plain(memory.state.bindings.bindings)
        result, memory, client = replay(case, memory, response=case["generator_response"],
                                        service_config={"execution_timeout_seconds": 1})
        self.assert_old_replay(result, memory, case)
        expected_args = copy.deepcopy(case["resolved_arguments"])
        expected_args["binding_descriptors"] = corrected_depth_descriptors(expected_args["binding_descriptors"])
        client.execute.assert_called_once_with(case["name"], expected_args, tool_call_id="call_fixture")
        self.assertEqual(result.data["printed_output"], "99.0")
        self.assertIsInstance(result.data["variables"]["new_values"], str)
        self.assertEqual(plain(memory.state.bindings.bindings), before)
        self.assertNotIn("new_values", memory.describe_bindings())

    def test_c03_execution_error_and_empty_stdout_match_old(self):
        for key in ("failure", "empty_output"):
            with self.subTest(case=key):
                case = OLD["coding"][key]
                _, memory, _ = replay(OLD["cases"]["depth"])
                result, memory, _ = replay(case, memory, response=case["generator_response"],
                                            service_config={"execution_timeout_seconds": 1})
                self.assert_old_replay(result, memory, case)
                self.assertTrue(result.ok)  # Generation succeeded; execution errors remain in the result.
                self.assertEqual(result.data["execution_error"], case["response"].get("execution_error"))

    def test_c02_infinite_loop_times_out_in_main_and_worker_with_old_visible_error(self):
        case = OLD["coding"]["timeout"]
        previous_handler = signal.getsignal(signal.SIGALRM) if hasattr(signal, "SIGALRM") else None
        try:
            for worker in (False, True):
                with self.subTest(worker=worker):
                    _, memory, _ = replay(OLD["cases"]["depth"])
                    def invoke():
                        return replay(case, memory, response=case["generator_response"],
                                      service_config={"execution_timeout_seconds": 1})
                    if worker:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                            result, memory, _ = pool.submit(invoke).result(timeout=4)
                    elif hasattr(signal, "SIGALRM"):
                        result, memory, _ = invoke()
                    else:
                        continue
                    self.assert_old_replay(result, memory, case)
                    self.assertEqual(result.data["execution_error"], "Code execution timed out after 1s")
        finally:
            if previous_handler is not None:
                signal.signal(signal.SIGALRM, previous_handler)

    def test_c03_generation_failure_is_distinct_from_execution_failure(self):
        memory = new_memory()
        error = {"code": "tool_execution_error_payload", "message": "LLM returned empty code"}
        capability, _ = make_capability("Python_Coding_Agent_Tool", None, ok=False, error=error)
        result = capability.invoke(VirtualCall("call_fixture", capability.name, {"objective": "Compute"}), memory)
        self.assertFalse(result.ok)
        self.assertEqual(result.error, error)
        self.assertEqual(result.evidence, [])
        self.assertIsNone(result.data)
        self.assertEqual(memory.state.bindings.bindings, {})

    def test_returned_generation_failure_preserves_structured_cause_and_never_executes(self):
        for error in ("LLM request failed: unavailable", {"code": "quota", "message": "quota exhausted"}):
            with self.subTest(error=error):
                capability, _ = make_capability("Python_Coding_Agent_Tool",
                    {"error": error, "generated_code": "print('should not run')"})
                with patch("openvlharness.services.tools.python_executor_local.executor.execute_python_code") as execute:
                    result = capability.invoke(VirtualCall("failed_generation", capability.name,
                        {"objective": "Compute"}), new_memory())
                execute.assert_not_called()
                self.assertEqual(result.data["generation_error"], error)
                self.assertIsNone(result.data["execution_error"])
                self.assertIn(str(error), result.evidence[0]["text"])
                self.assertNotIn("executed", result.evidence[0]["text"])

    def test_g02_dino_adapter_uses_virtual_name_and_null_segmentation_fields(self):
        raw = {"boxes": [[10.25, 5.5, 40.75, 20.0]], "boxes_normalized": [[51.25, 55, 203.75, 200]],
               "scores": [0.81234], "labels": ["shoe"], "size": [100, 200],
               "model": "dino-fixture"}
        capability, client = make_capability("Visual_Grounding_Tool", raw, backend="grounding_dino")
        call = VirtualCall("call_fixture", capability.name, {"query": "red shoe", "image_id": "input_image_1"})
        memory = new_memory()
        result = capability.invoke(call, memory)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(capability.public_schema(), SCHEMAS[0])
        self.assertEqual(client.execute.call_args.args[0], "GroundingDINO_Tool")
        self.assertEqual(result.payload["tool_name"], "Visual_Grounding_Tool")
        self.assertEqual(result.data["boxes"], raw["boxes"])
        self.assertEqual(result.data["labels"], ["shoe"])
        self.assertIsNone(result.data["masks"])
        self.assertIsNone(result.data["num_masks"])
        self.assertEqual(memory.state.bindings.bindings["segment_red_shoe"], {
            "pred_boxes": raw["boxes"], "pred_scores": raw["scores"], "pred_masks": None,
            "num_masks": None, "orig_img_h": 100, "orig_img_w": 200})
        self.assertEqual(memory.describe_bindings()["segment_red_shoe"]["source_tool"], "Visual_Grounding_Tool")
        text = "\n".join(part["text"] for part in result.evidence if part["type"] == "text")
        self.assertNotIn("labeled masks", text)
        self.assertIn("bounding boxes", text)
        self.assertIn(json.dumps(raw["boxes_normalized"]), text)
        self.assertIsNone(result.data["rendered_image"])
        self.assertEqual(result.detection["tool_name"], "Visual_Grounding_Tool")
        self.assertEqual(result.detection["boxes_xyxy_pixels"], raw["boxes"])


if __name__ == "__main__":
    unittest.main()
