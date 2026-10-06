"""In-process HTTP router/client contracts, with fake implementations only."""
import asyncio
import copy
import importlib
import json
import subprocess
import sys
import tempfile
import textwrap
import threading
import types
import unittest
from unittest.mock import patch

import httpx
import requests
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse, PlainTextResponse

from openvlharness.bootstrap import PAPER_BACKENDS, build_capabilities
from openvlharness.contracts import ServiceResult, VirtualCall
from openvlharness.memory.store import MemoryStore
from openvlharness.services.client import ToolClient
from openvlharness.services.server import create_app


def config_for(*names):
    selected, tools = {}, {}
    for name in names:
        backend = PAPER_BACKENDS[name]
        selected[name] = ({"generator": backend, "executor": "python_executor_local"}
                          if name == "coding" else {"backend": backend})
        tools[backend] = ({"server_url": "http://raw.invalid:8080"}
                          if name in {"grounding", "depth", "camera_pose", "ocr"} else
                          {"model_service": "orchestrator", "context": "fresh"} if name == "coding" else
                          {"model_service": "auxiliary"} if name in {"text_search", "image_search", "web_visit"} else {})
    return {"agent_name": "MainAgent", "agent_config_version": "paper_v1",
            "agent": {"model_service": "orchestrator", "max_iterations": 3},
            "memory": {"modules": [{"name": "image"}, {"name": "tool_context"}, {"name": "url"}]},
            "capabilities": {"registries": ["paper"], "tools": selected},
            "services": {"client": {"base_url": "http://router.invalid", "timeout_seconds": 7.5, "retry": 0},
                         "server": {"host": "127.0.0.1", "port": 8101, "registration": "configured_only"},
                         "tools": tools,
                         "models": {"orchestrator": {"alias": "gpt-6-luna", "base": "https://model.invalid/v1", "key": "fixture-private-key"},
                                    "auxiliary": {"alias": "qwen3-vl-8b-instruct", "base": "http://aux.invalid/v1", "key": "fixture-aux-key"}}}}


class FakeTool:
    def __init__(self, value=None, error=None):
        self.value, self.error, self.calls = value, error, []

    def execute(self, **args):
        self.calls.append(copy.deepcopy(args))
        if self.error:
            raise self.error
        return copy.deepcopy(self.value)


def wire_post(client):
    """Use actual ASGI HTTP serialization while observing requests' timeout."""
    def post(url, *, json, timeout):
        response = client.post("/execute", json=json)
        result = requests.Response()
        result.status_code = response.status_code
        result._content = response.content
        result.headers.update(response.headers)
        result.reason = response.reason_phrase
        result.encoding = "utf-8"
        result.url = str(response.url)
        return result
    return post


class RouterContractTests(unittest.TestCase):
    def test_t01_call_id_timeout_endpoint_and_concrete_name_survive_actual_http(self):
        tool = FakeTool({"text": "fixture words", "model_id": "fixture-ocr"})
        app = create_app(config_for("ocr"), implementations={"ocr": tool})
        with TestClient(app) as http, patch("openvlharness.services.http.requests.post", side_effect=wire_post(http)) as post:
            client = ToolClient("http://router.invalid/", timeout_seconds=7.5, retry=0)
            result = client.execute("OCR_Tool", {"image_base64": "fixture-bytes"}, tool_call_id="call_specific_17")
        self.assertEqual(result["tool_call_id"], "call_specific_17")
        self.assertEqual(result["result"]["text"], "fixture words")
        self.assertEqual(result["meta"]["http_status"], 200)
        post.assert_called_once_with("http://router.invalid/execute", json={"tool_name": "OCR_Tool",
            "args": {"image_base64": "fixture-bytes"}, "tool_call_id": "call_specific_17"}, timeout=7.5)
        self.assertEqual(tool.calls, [{"image_base64": "fixture-bytes", "server_url": "http://raw.invalid:8080"}])

    def test_t01_non2xx_envelopes_preserve_details_without_retrying_service_errors(self):
        tool = FakeTool(error=ValueError("fixture backend rejection"))
        app = create_app(config_for("ocr"), implementations={"ocr": tool})
        with TestClient(app) as http, patch("openvlharness.services.http.requests.post", side_effect=wire_post(http)) as post:
            client = ToolClient("http://router.invalid", timeout_seconds=9, retry=3)
            result = client.execute("OCR_Tool", {}, tool_call_id="call_error")
            missing = client.execute("Missing_Tool", {}, tool_call_id="call_missing")
        self.assertFalse(result["ok"])
        self.assertEqual(result["tool_call_id"], "call_error")
        self.assertEqual(result["meta"]["http_status"], 500)
        self.assertEqual(result["error"], {"code": "tool_execution_error", "message": "fixture backend rejection",
                                          "detail": {"exception_type": "ValueError"}})
        self.assertEqual(missing["error"]["code"], "tool_not_found")
        self.assertEqual(missing["tool_call_id"], "call_missing")
        self.assertEqual(missing["meta"]["http_status"], 404)
        self.assertEqual(post.call_count, 2)

    def test_t01_transport_timeout_retries_are_bounded_and_retain_same_request(self):
        app = create_app(config_for("ocr"), implementations={"ocr": FakeTool({"text": "ok"})})
        with TestClient(app) as http:
            dispatch = wire_post(http)
            attempts = []
            def flaky(url, **kwargs):
                attempts.append(copy.deepcopy(kwargs))
                if len(attempts) < 3:
                    raise requests.Timeout("fixture timeout")
                return dispatch(url, **kwargs)
            with patch("openvlharness.services.http.requests.post", side_effect=flaky), patch("openvlharness.services.http.time.sleep") as sleep:
                result = ToolClient("http://router.invalid", timeout_seconds=2.75, retry=2).execute("OCR_Tool", {}, tool_call_id="call_retry")
            self.assertTrue(result["ok"])
            self.assertEqual(len(attempts), 3)
            self.assertTrue(all(value == attempts[0] for value in attempts))
            self.assertEqual(attempts[0]["timeout"], 2.75)
            self.assertEqual(sleep.call_count, 2)
            with patch("openvlharness.services.http.requests.post", side_effect=requests.Timeout("fixture timeout")) as post, patch("openvlharness.services.http.time.sleep") as sleep:
                with self.assertRaises(requests.Timeout):
                    ToolClient("http://router.invalid", retry=1).execute("OCR_Tool", {}, tool_call_id="call_exhausted")
                self.assertEqual(post.call_count, 2)
                self.assertEqual(sleep.call_count, 1)

    def test_t01_non_json_and_protocol_mismatch_are_structured(self):
        for status, value, expected in ((200, "not-json", "execute_non_json_response"),
                                        (503, {"ok": True, "result": {}}, "execute_protocol_error")):
            with self.subTest(status=status):
                app = create_app(config_for(), implementations={})
                # Register a dedicated wire fixture and route it through the same client.
                @app.post("/wire-fixture")
                async def fixture():
                    return PlainTextResponse(value, status_code=status) if isinstance(value, str) else JSONResponse(value, status_code=status)
                with TestClient(app) as http:
                    def post(url, **kwargs):
                        response = http.post("/wire-fixture")
                        result = requests.Response(); result.status_code = response.status_code
                        result._content = response.content; result.headers.update(response.headers); result.encoding = "utf-8"
                        return result
                    with patch("openvlharness.services.http.requests.post", side_effect=post):
                        result = ToolClient("http://router.invalid", retry=0).execute("OCR_Tool", {}, tool_call_id="call_wire")
                self.assertFalse(result["ok"])
                self.assertEqual(result["error"]["code"], expected)
                self.assertEqual(result["tool_call_id"], "call_wire")
                self.assertEqual(result["meta"]["http_status"], status)

    def test_t01_unstructured_http_failure_retries_only_transient_status(self):
        for status, attempts in ((503, 3), (400, 1)):
            with self.subTest(status=status):
                app = create_app(config_for(), implementations={})
                @app.post("/wire-fixture")
                async def fixture():
                    return JSONResponse({"detail": "fixture HTTP rejection"}, status_code=status)
                with TestClient(app) as http:
                    def post(url, **kwargs):
                        response = http.post("/wire-fixture")
                        result = requests.Response(); result.status_code = response.status_code
                        result._content = response.content; result.reason = response.reason_phrase; result.encoding = "utf-8"
                        return result
                    with patch("openvlharness.services.http.requests.post", side_effect=post) as request, patch("openvlharness.services.http.time.sleep"):
                        with self.assertRaisesRegex(RuntimeError, str(status)+".*fixture HTTP rejection"):
                            ToolClient("http://router.invalid", retry=2).execute("OCR_Tool", {}, tool_call_id="call_http")
                    self.assertEqual(request.call_count, attempts)

    def test_t02_only_configured_implementation_is_imported(self):
        real_import = importlib.import_module
        imported = []
        tool = FakeTool({"text": "ok"})
        def selected_import(name, *args, **kwargs):
            if name.startswith("openvlharness.services.tools."):
                imported.append(name)
                self.assertEqual(name, "openvlharness.services.tools.ocr.tool")
                return types.SimpleNamespace(OCR_Tool=lambda: tool)
            return real_import(name, *args, **kwargs)
        with patch("openvlharness.services.server.importlib.import_module", side_effect=selected_import):
            app = create_app(config_for("ocr"))
        self.assertEqual(imported, ["openvlharness.services.tools.ocr.tool"])
        with TestClient(app) as client:
            self.assertEqual(client.get("/health").json(), {"status": "ok", "tools": 1})
            self.assertEqual(set(client.get("/tools").json()), {"OCR_Tool"})
            schemas = client.get("/llm-schema").json()
            self.assertEqual([s["function"]["name"] for s in schemas], ["OCR_Tool"])

    def test_t02_env_redacts_keys_and_preserves_effective_settings(self):
        app = create_app(config_for("ocr"), implementations={"ocr": FakeTool()})
        with TestClient(app) as client:
            environment = client.get("/env").json()
        self.assertNotIn("fixture-private-key", json.dumps(environment))
        self.assertNotIn("fixture-aux-key", json.dumps(environment))
        self.assertEqual(environment["config"]["models"]["orchestrator"]["key"], "<redacted>")
        self.assertEqual(environment["config"]["client"]["timeout_seconds"], 7.5)
        self.assertEqual(environment["search"]["keep_k"], 3)
        self.assertTrue(environment["search"]["multimodal"])

    def test_t02_real_selected_wrapper_import_has_no_gpu_or_generic_subagent_dependencies(self):
        script = textwrap.dedent('''
            import importlib.abc, json, sys
            blocked = {'torch', 'transformers', 'cv2', 'sam3', 'depth_anything_3',
                       'paddle', 'mini_vlm_toolkit'}
            class BlockHeavy(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] in blocked or 'subagent' in fullname:
                        raise ModuleNotFoundError('Heavy/generic package intentionally unavailable: '+fullname)
            sys.meta_path.insert(0, BlockHeavy())
            from openvlharness.services.server import create_app
        ''') + "\nconfig = " + repr(config_for("ocr")) + textwrap.dedent('''

            app = create_app(config)
            assert set(app.state.registry) == {'OCR_Tool'}
            implementations = sorted(name for name in sys.modules
                if name.startswith('openvlharness.services.tools.') and name.endswith('.tool'))
            assert implementations == ['openvlharness.services.tools.ocr.tool'], implementations
            assert not any(name.split('.')[0] in blocked for name in sys.modules)
            print(json.dumps({'implementations': implementations}))
        ''')
        with tempfile.TemporaryDirectory() as cwd:
            proc = subprocess.run([sys.executable, "-I", "-c", script], cwd=cwd,
                                  capture_output=True, text=True, timeout=20)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["implementations"], ["openvlharness.services.tools.ocr.tool"])

    def test_runtime_endpoints_and_tuning_cannot_override_service_configuration(self):
        tool = FakeTool({"text": "ok"})
        app = create_app(config_for("ocr"), implementations={"ocr": tool})
        with TestClient(app) as client:
            for key in ("server_url", "timeout_seconds", "llm_api_key", "box_threshold", "num_results"):
                response = client.post("/execute", json={"tool_name": "OCR_Tool", "tool_call_id": "call_hidden", "args": {key: "override"}})
                self.assertFalse(response.json()["ok"])
                self.assertIn("cannot be supplied", response.json()["error"]["message"])
            invalid = client.post("/execute", json={"tool_name": "OCR_Tool", "args": {}})
            self.assertEqual(invalid.status_code, 422)
        self.assertEqual(tool.calls, [])

    def test_s03_null_empty_and_structured_match_fields_survive_http_and_capability(self):
        config = config_for("image_search")
        for raw, expected in ((None, None), ({"text": "No structured records"}, None),
                              ({"matches": []}, []), ({"matches": [{"title": "shoe", "link": "https://example.com/item", "width": 42}]},
                              [{"title": "shoe", "image_url": None, "source_url": "https://example.com/item", "snippet": None,
                                "source": None, "thumbnail": None, "position": None, "width": 42, "height": None}])):
            with self.subTest(raw=raw):
                app = create_app(config, implementations={"serper_image_search": FakeTool(raw)})
                capability = build_capabilities(config)["Image_Search_Tool"]
                memory = MemoryStore(modules=())
                with TestClient(app) as http, patch("openvlharness.services.http.requests.post", side_effect=wire_post(http)):
                    result = capability.invoke(VirtualCall("call_fixture", capability.name, {"image_id": "input_image_1"}), memory)
                self.assertTrue(result.ok)
                self.assertEqual(result.data["matches"], expected)
                self.assertIsNone(result.data["images"])

    def test_s03_webpage_partial_failure_retains_successful_siblings_and_original_text(self):
        pages = [{"url": "https://example.com/ok", "text": "source body", "evidence": "quoted evidence", "summary": "useful"},
                 {"url": "https://example.com/blocked", "text": "", "error": "HTTP403 blocked"}]
        rendered = "First page: useful.\nSecond page: HTTP403 blocked."
        config = config_for("web_visit")
        app = create_app(config, implementations={"serper_webpage_visit": FakeTool(ServiceResult(rendered, {"pages": pages}))})
        capability = build_capabilities(config)["Webpage_Visit_Tool"]
        with TestClient(app) as http, patch("openvlharness.services.http.requests.post", side_effect=wire_post(http)):
            result = capability.invoke(VirtualCall("call_partial", capability.name,
                {"urls": [page["url"] for page in pages], "goal": "Find details"}), MemoryStore())
        self.assertTrue(result.ok)
        self.assertEqual(result.payload["result"], rendered)
        self.assertEqual(result.data["pages"], [dict(pages[0], error=None), dict(pages[1], evidence=None, summary=None)])
        self.assertIn("HTTP403 blocked", result.evidence[0]["text"])
        self.assertIsNone(result.error)
        self.assertEqual(result.artifact_refs, [])

    def test_generation_returned_error_and_thrown_failure_preserve_causes(self):
        config = config_for("coding")
        for tool, expected_ok in ((FakeTool({"error": "LLM returned empty code", "raw_response": ""}), True),
                                  (FakeTool(error=RuntimeError("generation transport unavailable")), False)):
            with self.subTest(expected_ok=expected_ok):
                app = create_app(config, implementations={"python_coding_agent": tool})
                capability = build_capabilities(config)["Python_Coding_Agent_Tool"]
                with TestClient(app) as http, patch("openvlharness.services.http.requests.post", side_effect=wire_post(http)):
                    result = capability.invoke(VirtualCall("call_generator", capability.name, {"objective": "Compute"}), MemoryStore())
                self.assertEqual(result.ok, expected_ok)
                if expected_ok:
                    self.assertEqual(result.payload["result"]["error"], "LLM returned empty code")
                    self.assertEqual(result.data["generation_error"], "LLM returned empty code")
                    self.assertIsNone(result.data["execution_error"])
                    self.assertEqual(result.evidence, [{"type": "text", "text": "Generation error: LLM returned empty code"}])
                else:
                    self.assertEqual(result.error["code"], "tool_execution_error")
                    self.assertIn("generation transport unavailable", result.error["message"])

    def test_s03_malformed_webpage_record_returns_observable_error(self):
        config = config_for("web_visit")
        malformed = ServiceResult("Malformed page output", {"pages": [{"summary": "missing page URL"}]})
        app = create_app(config, implementations={"serper_webpage_visit": FakeTool(malformed)})
        capability = build_capabilities(config)["Webpage_Visit_Tool"]
        with TestClient(app) as http, patch("openvlharness.services.http.requests.post", side_effect=wire_post(http)):
            result = capability.invoke(VirtualCall("call_malformed_page", capability.name,
                {"urls": ["https://example.com/page"], "goal": "Find details"}), MemoryStore())
        self.assertFalse(result.ok)
        self.assertTrue(result.error.get("message"))
        self.assertEqual(result.payload["tool_call_id"], "call_malformed_page")
        self.assertEqual(result.artifact_refs, [])


class CapacityTests(unittest.IsolatedAsyncioTestCase):
    async def check_shared_capacity(self, fail_first):
        entered, release, camera_entered = threading.Event(), threading.Event(), threading.Event()
        order, active = [], {"current": 0, "maximum": 0}
        lock = threading.Lock()
        class BlockedDepth:
            def execute(self, **kwargs):
                with lock:
                    active["current"] += 1; active["maximum"] = max(active["maximum"], active["current"])
                    order.append("depth_start")
                entered.set()
                try:
                    if not release.wait(timeout=4):
                        raise TimeoutError("test did not release depth")
                    if fail_first:
                        raise RuntimeError("fixture depth failed")
                    return {"depth_shape": [1, 1]}
                finally:
                    with lock:
                        active["current"] -= 1; order.append("depth_end")
        class Camera:
            def execute(self, **kwargs):
                with lock:
                    active["current"] += 1; active["maximum"] = max(active["maximum"], active["current"])
                    order.append("camera_start")
                camera_entered.set()
                with lock:
                    active["current"] -= 1; order.append("camera_end")
                return {"num_images": 2}
        app = create_app(config_for("depth", "camera_pose"), implementations={"depth_estimation": BlockedDepth(), "camera_trajectory": Camera()})
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://router.invalid") as client:
            first = asyncio.create_task(client.post("/execute", json={"tool_name": "Depth_Estimation_Tool", "tool_call_id": "call_depth", "args": {}}))
            second = None
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                second = asyncio.create_task(client.post("/execute", json={"tool_name": "Camera_Trajectory_Tool", "tool_call_id": "call_camera", "args": {}}))
                await asyncio.sleep(0)
                health, status = await asyncio.wait_for(asyncio.gather(client.get("/health"), client.get("/backend-status")), timeout=1)
                self.assertEqual(health.status_code, 200)
                self.assertEqual(status.json()["depth"], {"limit": 1, "in_flight": 1})
                self.assertFalse(camera_entered.is_set())
            finally:
                release.set()
                tasks = [task for task in (first, second) if task is not None]
                responses = await asyncio.wait_for(asyncio.gather(*tasks), timeout=3)
            self.assertEqual(responses[0].status_code, 500 if fail_first else 200)
            self.assertEqual(responses[1].status_code, 200)
            self.assertEqual(responses[1].json()["tool_call_id"], "call_camera")
            self.assertEqual((await client.get("/backend-status")).json()["depth"]["in_flight"], 0)
        self.assertEqual(active["maximum"], 1)
        self.assertEqual(order, ["depth_start", "depth_end", "camera_start", "camera_end"])

    async def test_t03_depth_camera_share_gate_and_health_stays_responsive(self):
        await self.check_shared_capacity(False)

    async def test_t03_backend_exception_releases_capacity_for_queued_camera(self):
        await self.check_shared_capacity(True)


if __name__ == "__main__":
    unittest.main()
