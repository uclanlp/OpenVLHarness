"""ToolSmith catalog loading, generated-tool calls, per-query retrieval and router wiring, offline."""
import copy
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from openvlharness.agent.agent import MainAgent
from openvlharness.bootstrap import agent_class_name, load_agent_config
from openvlharness.contracts import VirtualCall
from openvlharness.memory.store import MemoryStore
from openvlharness.services.models import ModelAdapter
from openvlharness.toolsmith import GeneratedTool, ToolCallError
from openvlharness.toolsmith.agent import ToolSmithAgent
from openvlharness.toolsmith.capability import Capability
from openvlharness.toolsmith.catalog import load_catalog
from openvlharness.toolsmith.isolation import available

ISOLATION = available()[0]  # router tests confine the worker where this host allows it

IMAGE = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

TOOL = '''
from openvlharness.toolsmith import GeneratedTool


class {cls}(GeneratedTool):
    def __init__(self):
        super().__init__(tool_name="{name}", tool_description="{description}", input_schema={{
            "type": "object", "properties": {{"image_id": {{"type": "string"}}}}, "required": ["image_id"]}})

    def execute(self, image_base64):
        matches = self.call_tool("Image_Search_Tool", image_base64=image_base64)
        return {{"top": matches["text"].split()[0]}}
'''


PID_TOOL = '''
import os

from openvlharness.toolsmith import GeneratedTool


class WorkerPid(GeneratedTool):
    def __init__(self):
        super().__init__(tool_name="worker_pid", tool_description="pid", input_schema={"type": "object"})

    def execute(self):
        return os.getpid()
'''


SLEEPY_TOOL = '''
import time

from openvlharness.toolsmith import GeneratedTool


class Sleepy(GeneratedTool):
    def __init__(self):
        super().__init__(tool_name="sleepy", tool_description="sleeps", input_schema={"type": "object"})

    def execute(self):
        time.sleep(30)
        return "awake"
'''


def write_tool(directory, name, description, cls=None):
    cls = cls or "".join(part.title() for part in name.split("_"))
    Path(directory, f"{name}.py").write_text(TOOL.format(cls=cls, name=name, description=description))


def config(**toolsmith):
    return {"agent_name": "MainAgent", "agent_config_version": "paper_v1",
            "agent": {"model_service": "orchestrator", "max_iterations": 3},
            "memory": {"modules": [{"name": "image"}, {"name": "tool_context"}, {"name": "url"}]},
            "capabilities": {"registries": ["paper"], "tools": {"image_search": {"backend": "serper_image_search"}}},
            "services": {"client": {"base_url": "http://router.invalid", "timeout_seconds": 5, "retry": 0},
                         "server": {"host": "127.0.0.1", "port": 8101, "registration": "configured_only"},
                         "tools": {"serper_image_search": {"model_service": "auxiliary"}},
                         "models": {"orchestrator": {"alias": "qwen3-vl-8b-instruct", "base": "http://model.invalid/v1"},
                                    "auxiliary": {"alias": "qwen3-vl-8b-instruct", "base": "http://aux.invalid/v1"}}},
            "toolsmith": toolsmith}


class FakeClient:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, []

    def execute(self, tool_name, args, *, tool_call_id):
        self.calls.append((tool_name, copy.deepcopy(args), tool_call_id))
        if self.error:
            return {"ok": False, "tool_name": tool_name, "tool_call_id": tool_call_id, "result": None,
                    "error": {"code": "tool_execution_error", "message": self.error}, "meta": {}}
        return {"ok": True, "tool_name": tool_name, "tool_call_id": tool_call_id,
                "result": copy.deepcopy(self.result), "error": None, "meta": {}}


class WordEncoder:
    """Bag-of-words embeddings, so retrieval order is predictable without model weights."""
    vocabulary = ("bird", "egg", "river", "length", "painting", "artist")

    def __init__(self, name):
        self.name = name

    def encode(self, texts, normalize_embeddings, convert_to_numpy):
        vectors = np.array([[text.lower().count(word) for word in self.vocabulary] for text in texts], dtype=float)
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)


def fake_encoder():
    return patch.dict(sys.modules, {"sentence_transformers": types.SimpleNamespace(SentenceTransformer=WordEncoder)})


class CatalogTests(unittest.TestCase):
    def test_loads_one_tool_per_module_skips_private_files_and_attaches_the_client(self):
        with tempfile.TemporaryDirectory() as directory:
            write_tool(directory, "bird_egg_lookup", "Bird egg facts")
            write_tool(directory, "river_length_lookup", "River length")
            Path(directory, "_helpers.py").write_text("raise RuntimeError('never imported')\n")
            client = FakeClient()
            tools = load_catalog(directory, client=client)
        self.assertEqual(list(tools), ["bird_egg_lookup", "river_length_lookup"])
        self.assertTrue(all(isinstance(tool, GeneratedTool) and tool.client is client for tool in tools.values()))
        self.assertEqual(tools["bird_egg_lookup"].input_schema["required"], ["image_id"])

    def test_rejects_missing_directories_modules_without_one_tool_and_duplicate_names(self):
        with self.assertRaisesRegex(ValueError, "not a directory"):
            load_catalog("/nonexistent/toolsmith/catalog")
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "empty.py").write_text("VALUE = 1\n")
            with self.assertRaisesRegex(ValueError, "exactly one GeneratedTool"):
                load_catalog(directory)
        with tempfile.TemporaryDirectory() as directory:
            write_tool(directory, "a_lookup", "first", cls="Lookup")
            Path(directory, "b_lookup.py").write_text(Path(directory, "a_lookup.py").read_text())
            with self.assertRaisesRegex(ValueError, "Duplicate generated tool name 'a_lookup'"):
                load_catalog(directory)

    def test_packaged_catalog_loads(self):
        tools = load_catalog()
        self.assertTrue(all(isinstance(tool, GeneratedTool) for tool in tools.values()))


class GeneratedToolTests(unittest.TestCase):
    def make_tool(self, client):
        with tempfile.TemporaryDirectory() as directory:
            write_tool(directory, "bird_egg_lookup", "Bird egg facts")
            return load_catalog(directory, client=client)["bird_egg_lookup"]

    def test_call_tool_returns_router_results_and_raises_on_failures(self):
        client = FakeClient({"text": "Robin visual matches"})
        self.assertEqual(self.make_tool(client).execute(image_base64=IMAGE), {"top": "Robin"})
        name, args, call_id = client.calls[0]
        self.assertEqual((name, args), ("Image_Search_Tool", {"image_base64": IMAGE}))
        self.assertTrue(call_id.startswith("bird_egg_lookup-"))
        with self.assertRaisesRegex(ToolCallError, "Image_Search_Tool: lens down"):
            self.make_tool(FakeClient(error="lens down")).execute(image_base64=IMAGE)
        with self.assertRaisesRegex(RuntimeError, "no router client"):
            self.make_tool(None).execute(image_base64=IMAGE)

    def test_capability_advertises_the_tool_schema_and_resolves_image_ids(self):
        tool = self.make_tool(None)
        client = FakeClient({"answer": "14 days"})
        capability = Capability(tool, client_config={"base_url": "http://router.invalid"}, client=client)
        self.assertEqual(capability.public_schema()["function"],
                         {"name": "bird_egg_lookup", "description": "Bird egg facts", "parameters": tool.input_schema})
        memory = MemoryStore()
        memory.start([{"type": "image", "value": IMAGE}, {"type": "text", "value": "How long?"}])
        result = capability.invoke(VirtualCall("call-1", "bird_egg_lookup", {"image_id": "input_image_1"}), memory)
        self.assertTrue(result.ok)
        self.assertEqual(result.data, {"result": {"answer": "14 days"}, "images": []})
        name, args, call_id = client.calls[0]
        self.assertEqual((name, list(args), call_id), ("bird_egg_lookup", ["image_base64"], "call-1"))
        self.assertTrue(args["image_base64"].startswith("data:image/"))
        self.assertIn('"answer": "14 days"', result.evidence[0]["text"])
        rejected = capability.invoke(VirtualCall("call-2", "bird_egg_lookup", {"query": "robin"}), memory)
        self.assertEqual(rejected.error["code"], "invalid_tool_arguments")


class PackagedToolBehaviourTests(unittest.TestCase):
    """Packaged helpers with their web requests stubbed."""

    tools = load_catalog()

    def test_wikidata_confidence_reaches_one_only_for_exact_hint_tokens(self):
        match = type(self.tools["wikidata_query"]).match
        claims = {"architectural style": ["Gothic architecture"], "architect": ["Pierre de Montreuil"]}
        self.assertEqual((match(claims, "architect")["property"], match(claims, "architect")["confidence"]),
                         ("architect", 1.0))
        partial = match({"architectural style": ["Gothic architecture"]}, "architect")
        self.assertEqual((partial["property"], partial["confidence"]), ("architectural style", 0.5))
        self.assertEqual(match({"place of death": ["Paris"]}, "place of birth")["confidence"], 0.5)
        self.assertEqual(match({"place of death": ["Paris"], "place of birth": ["Lyon"]}, "place of birth"),
                         {"property": "place of birth", "values": ["Lyon"], "score": 2, "hint_token_count": 2,
                          "confidence": 1.0})

    def test_media_catalog_errors_do_not_reveal_the_tmdb_key(self):
        import requests

        tool = self.tools["media_catalog"]

        def unauthorized(url, params=None, **kwargs):
            response = requests.Response()
            response.status_code, response.reason = 401, "Unauthorized"
            response.url = requests.Request("GET", url, params=params).prepare().url
            return response
        stub = types.SimpleNamespace(get=unauthorized, RequestException=requests.RequestException)
        with patch.dict(os.environ, {"TMDB_API_KEY": "tmdb-secret-123"}), \
                patch.dict(tool.execute.__globals__, {"requests": stub}):
            result = tool.execute(title="Heat", media_type="film")
        self.assertEqual(result, {"found": False, "error": "film lookup failed: HTTP 401"})
        with patch.dict(tool.execute.__globals__, {"get_json": lambda url, params: (_ for _ in ()).throw(
                requests.ConnectionError(f"Max retries exceeded with url: {url}?api_key=tmdb-secret-123"))}):
            self.assertNotIn("tmdb-secret-123", json.dumps(tool.execute(title="Heat", media_type="music")))

    def test_media_catalog_prefers_an_exact_release_to_a_fuzzy_artist(self):
        tool = self.tools["media_catalog"]
        searches = {"artist": [{"name": "Thriller Kids", "id": "a1"}], "release": [{"title": "Thriller", "id": "r1"}]}
        records = {"a1": {"id": "a1", "name": "Thriller Kids", "type": "Group"},
                   "r1": {"id": "r1", "title": "Thriller", "date": "1982-11-30",
                          "artist-credit": [{"name": "Michael Jackson"}]}}
        requested = []

        def get_json(url, params):
            requested.append(url.rsplit("/ws/2/", 1)[1])
            entity = url.rsplit("/", 2)
            return records[entity[-1]] if entity[-1] in records else {f"{entity[-1]}s": searches[entity[-1]]}
        with patch.dict(tool.execute.__globals__, {"get_json": get_json}):
            release = tool.execute(title="thriller", media_type="music")
            self.assertEqual((release["canonical_title"], release["artist"]), ("Thriller", "Michael Jackson"))
            requested.clear()
            self.assertEqual(tool.execute(title="Thriller Kids", media_type="music")["canonical_title"], "Thriller Kids")
            self.assertEqual(requested, ["artist", "artist/a1"])
            searches["release"] = [{"title": "Thriller (Remastered)", "id": "r1"}]
            self.assertEqual(tool.execute(title="Thriller", media_type="music")["id"], "a1")


class RetrievalAndAgentTests(unittest.TestCase):
    def make_agent(self, directory, responses, top_k=1):
        requests = []

        def requester(payload):
            requests.append(copy.deepcopy(payload))
            return responses.pop(0)
        model = ModelAdapter({"alias": "qwen3-vl-8b-instruct", "base": "http://fixture.invalid/v1", "max_tokens": 64},
                             requester=requester)
        with fake_encoder():
            agent = ToolSmithAgent(model, {}, MemoryStore, "fixture system", "Question: {user_query}",
                                   toolsmith={"catalog": directory, "top_k": top_k},
                                   client_config={"base_url": "http://router.invalid", "retry": 0})
        return agent, requests

    def test_retrieves_the_most_similar_tools_per_query(self):
        with fake_encoder():
            from openvlharness.toolsmith.retrieval import ToolRetriever
            retriever = ToolRetriever({"egg_days": "bird egg incubation", "river": "river length", "art": "painting artist"},
                                      top_k=2)
        self.assertEqual(retriever.retrieve("How long is this river?"), ["river", "egg_days"])
        self.assertEqual(retriever.retrieve("Who is the artist of this painting?")[0], "art")

    def test_agent_offers_retrieved_tools_and_executes_their_calls(self):
        answer = {"choices": [{"message": {"role": "assistant", "content": "<answer>14 days</answer>"}}]}
        call = {"choices": [{"message": {"role": "assistant", "content": "checking", "tool_calls": [
            {"id": "call-1", "type": "function", "function": {"name": "bird_egg_lookup", "arguments": '{"image_id": "input_image_1"}'}}]}}]}
        with tempfile.TemporaryDirectory() as directory:
            write_tool(directory, "bird_egg_lookup", "bird egg incubation days")
            write_tool(directory, "river_length_lookup", "river length")
            agent, requests = self.make_agent(directory, [call, answer])
        client = FakeClient({"answer": "14 days"})
        agent.generated["bird_egg_lookup"].client = client
        result = agent.generate([{"type": "image", "value": IMAGE},
                                 {"type": "text", "value": "How many days does this bird egg take?"}])
        self.assertEqual(result["prediction"], "14 days")
        self.assertEqual(type(agent).__name__, agent_class_name({"toolsmith": {}}))
        self.assertEqual([tool["function"]["name"] for tool in requests[0]["tools"]], ["bird_egg_lookup"])
        self.assertEqual(client.calls[0][0], "bird_egg_lookup")
        self.assertNotIn("river_length_lookup", agent.tools_for([{"type": "text", "value": "bird egg"}]))

    def test_main_agent_offers_every_configured_tool(self):
        model = ModelAdapter({"alias": "qwen3-vl-8b-instruct", "base": "http://fixture.invalid/v1"})
        agent = MainAgent(model, {}, MemoryStore, "fixture system", "Question: {user_query}")
        self.assertIs(agent.tools_for([{"type": "text", "value": "How many?"}]), agent.capabilities)

    def test_generated_tools_cannot_shadow_base_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            write_tool(directory, "Image_Search_Tool", "shadow", cls="Shadow")
            model = ModelAdapter({"alias": "qwen3-vl-8b-instruct", "base": "http://fixture.invalid/v1"})
            base = Capability(load_catalog(directory)["Image_Search_Tool"], client_config={"base_url": "http://router.invalid"})
            with fake_encoder(), self.assertRaisesRegex(ValueError, "shadow base tools"):
                ToolSmithAgent(model, {"Image_Search_Tool": base}, MemoryStore, "s", "{user_query}",
                               toolsmith={"catalog": directory}, client_config={"base_url": "http://router.invalid"})


class ConfigurationTests(unittest.TestCase):
    def test_toolsmith_section_is_optional_and_validated(self):
        self.assertNotIn("toolsmith", load_agent_config({k: v for k, v in config().items() if k != "toolsmith"}))
        with tempfile.TemporaryDirectory() as directory:
            loaded = load_agent_config(config(catalog=directory, top_k=10, encoder="BAAI/bge-m3"))
            self.assertEqual(loaded["toolsmith"]["top_k"], 10)
        with tempfile.TemporaryDirectory() as directory:
            worker = {"environment": ["SERPER_API_KEY"], "timeout_seconds": 30, "memory_gib": 16, "write_files": True,
                      "isolation": True, "network": False}
            self.assertEqual(load_agent_config(config(catalog=directory, worker=worker))["toolsmith"]["worker"], worker)
        for bad, message in (({"top_k": 0}, "positive integer"), ({"top_k": "10"}, "positive integer"),
                             ({"router": "tfidf"}, "optional catalog, top_k, encoder and worker"),
                             ({"catalog": "/nonexistent/catalog"}, "not a directory"),
                             ({"worker": {"sandbox": True}}, "optional environment, timeout_seconds"),
                             ({"worker": {"environment": "SERPER_API_KEY"}}, "list environment variable names"),
                             ({"worker": {"environment": ["SERPER API KEY"]}}, "list environment variable names"),
                             ({"worker": {"timeout_seconds": 0}}, "timeout_seconds must be a positive number"),
                             ({"worker": {"memory_gib": float("inf")}}, "memory_gib must be a positive number"),
                             ({"worker": {"write_files": "yes"}}, "write_files must be true or false"),
                             ({"worker": {"isolation": "on"}}, "isolation must be true or false"),
                             ({"worker": {"network": None}}, "network must be true or false")):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, message):
                load_agent_config(config(**bad))
        broken = config()
        broken["toolsmith"] = None
        with self.assertRaisesRegex(ValueError, "must be a mapping"):
            load_agent_config(broken)

    def test_the_baseline_preset_is_the_toolsmith_preset_without_generated_tools(self):
        presets = Path(__file__).resolve().parents[1] / "configs" / "agent"
        toolsmith, baseline = (yaml.safe_load((presets / f"qwen3_8b_toolsmith{suffix}.yaml").read_text())
                               for suffix in ("", "_baseline"))
        section = toolsmith.pop("toolsmith")
        self.assertEqual(section.pop("worker")["write_files"], True)
        self.assertEqual(section, {"top_k": 10, "encoder": "BAAI/bge-m3"})
        self.assertEqual(toolsmith, baseline)
        self.assertEqual(baseline["agent"]["prompt_bundle"], "toolsmith")


class RouterTests(unittest.TestCase):
    def test_router_registers_catalog_tools_and_serves_nested_calls(self):
        pytest.importorskip("fastapi", reason="router tests need the server extra")
        from fastapi.testclient import TestClient

        from openvlharness.services.server import create_app

        class Lens:
            def execute(self, image_base64):
                return {"text": "Robin visual matches"}

        with tempfile.TemporaryDirectory() as directory:
            write_tool(directory, "bird_egg_lookup", "Bird egg facts")
            Path(directory, "worker_pid.py").write_text(PID_TOOL)
            app = create_app(config(catalog=directory, worker={"isolation": ISOLATION}),
                             implementations={"serper_image_search": Lens()})
        http = TestClient(app)
        self.addCleanup(app.state.registry["bird_egg_lookup"][0].close)

        class InProcessClient:
            def execute(self, tool_name, args, *, tool_call_id):
                return http.post("/execute", json={"tool_name": tool_name, "args": args, "tool_call_id": tool_call_id}).json()

        app.state.registry["bird_egg_lookup"][0].client = InProcessClient()
        self.assertEqual(http.get("/tools").json()["bird_egg_lookup"], {"tool_name": "bird_egg_lookup", "backend": "toolsmith"})
        response = http.post("/execute", json={"tool_name": "bird_egg_lookup", "tool_call_id": "call-1",
                                               "args": {"image_base64": IMAGE}}).json()
        self.assertEqual((response["ok"], response["result"]), (True, {"top": "Robin"}))
        rejected = http.post("/execute", json={"tool_name": "bird_egg_lookup", "tool_call_id": "call-2",
                                               "args": {"image_path": "/etc/passwd"}})
        self.assertEqual(rejected.status_code, 400)
        pid = http.post("/execute", json={"tool_name": "worker_pid", "tool_call_id": "call-3"}).json()["result"]
        self.assertNotEqual(pid, os.getpid())

    def test_waiting_generated_tools_leave_the_thread_pool_to_their_nested_calls(self):
        pytest.importorskip("fastapi", reason="router tests need the server extra")
        import time
        from concurrent.futures import ThreadPoolExecutor

        import anyio
        from fastapi.testclient import TestClient

        from openvlharness.services.server import create_app

        class Lens:
            def execute(self, image_base64):
                time.sleep(0.2)
                return {"text": "Robin visual matches"}

        with tempfile.TemporaryDirectory() as directory:
            write_tool(directory, "bird_egg_lookup", "Bird egg facts")
            app = create_app(config(catalog=directory, worker={"timeout_seconds": 20, "isolation": ISOLATION}),
                             implementations={"serper_image_search": Lens()})
        worker = app.state.registry["bird_egg_lookup"][0]
        self.addCleanup(worker.close)
        with TestClient(app) as http:
            # One thread-pool worker: each generated call's nested Image_Search_Tool call needs it.
            http.portal.call(lambda: setattr(anyio.to_thread.current_default_thread_limiter(), "total_tokens", 1))

            class InProcessClient:
                def execute(self, tool_name, args, *, tool_call_id):
                    return http.post("/execute", json={"tool_name": tool_name, "args": args,
                                                       "tool_call_id": tool_call_id}).json()
            worker.client = InProcessClient()
            started = time.monotonic()
            with ThreadPoolExecutor(4) as pool:
                responses = list(pool.map(lambda n: http.post("/execute", json={
                    "tool_name": "bird_egg_lookup", "tool_call_id": f"call-{n}", "args": {"image_base64": IMAGE}}).json(),
                    range(4)))
        self.assertEqual([(r["ok"], r["result"]) for r in responses], [(True, {"top": "Robin"})] * 4)
        self.assertLess(time.monotonic() - started, 15)

    def test_a_worker_timeout_reaches_the_http_client_before_its_own_deadline(self):
        uvicorn = pytest.importorskip("uvicorn", reason="router tests need the server extra")
        import socket
        import threading
        import time

        from openvlharness.services.client import ToolClient
        from openvlharness.services.server import create_app

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "sleepy.py").write_text(SLEEPY_TOOL)
            settings = config(catalog=directory, worker={"isolation": ISOLATION})
            settings["services"]["client"] = {"base_url": f"http://127.0.0.1:{port}", "timeout_seconds": 2, "retry": 8}
            app = create_app(settings, implementations={"serper_image_search": object()})
        worker = app.state.registry["sleepy"][0]
        self.addCleanup(worker.close)
        dispatched, execute = [], worker.execute_async

        async def counted(*args, **kwargs):
            dispatched.append(args[0])
            return await execute(*args, **kwargs)
        worker.execute_async = counted
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 10)
        self.addCleanup(setattr, server, "should_exit", True)
        deadline = time.monotonic() + 30
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        started = time.monotonic()
        payload = ToolClient(f"http://127.0.0.1:{port}", timeout_seconds=2, retry=8).execute(
            "sleepy", {}, tool_call_id="call-1")
        self.assertEqual((payload["ok"], payload["error"]["detail"]["exception_type"]), (False, "TimeoutError"))
        self.assertLess(time.monotonic() - started, 4)
        self.assertEqual(dispatched, ["sleepy"])

    def test_the_router_reports_the_digest_of_the_catalog_it_serves(self):
        pytest.importorskip("fastapi", reason="router tests need the server extra")
        from fastapi.testclient import TestClient

        from openvlharness.services.server import create_app
        from openvlharness.toolsmith.catalog import catalog_fingerprint

        with tempfile.TemporaryDirectory() as directory:
            write_tool(directory, "bird_egg_lookup", "Bird egg facts")
            started = catalog_fingerprint(directory)
            app = create_app(config(catalog=directory, worker={"isolation": ISOLATION}),
                             implementations={"serper_image_search": object()})
            self.addCleanup(app.state.registry["bird_egg_lookup"][0].close)
            write_tool(directory, "bird_egg_lookup", "Bird egg facts, edited on disk")
            self.assertNotEqual(catalog_fingerprint(directory), started)
            reported = TestClient(app).get("/env").json()["toolsmith"]
        self.assertEqual(reported, {"catalog_sha256": started})

    def test_generated_tools_cannot_shadow_router_tools(self):
        pytest.importorskip("fastapi", reason="router tests need the server extra")
        from openvlharness.services.server import create_app

        with tempfile.TemporaryDirectory() as directory:
            write_tool(directory, "Image_Search_Tool", "shadow", cls="Shadow")
            with self.assertRaisesRegex(ValueError, "shadow base tools"):
                create_app(config(catalog=directory, worker={"isolation": ISOLATION}),
                           implementations={"serper_image_search": object()})
