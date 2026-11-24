"""ToolSmith catalog loading, generated-tool calls, per-query retrieval and router wiring, offline."""
import copy
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
from openvlharness.bootstrap import load_agent_config
from openvlharness.contracts import VirtualCall
from openvlharness.memory.store import MemoryStore
from openvlharness.services.models import ModelAdapter
from openvlharness.toolsmith import GeneratedTool, ToolCallError
from openvlharness.toolsmith.agent import ToolSmithAgent
from openvlharness.toolsmith.capability import Capability
from openvlharness.toolsmith.catalog import load_catalog

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
            worker = {"environment": ["SERPER_API_KEY"], "timeout_seconds": 30, "memory_gib": 16, "write_files": True}
            self.assertEqual(load_agent_config(config(catalog=directory, worker=worker))["toolsmith"]["worker"], worker)
        for bad, message in (({"top_k": 0}, "positive integer"), ({"top_k": "10"}, "positive integer"),
                             ({"router": "tfidf"}, "optional catalog, top_k, encoder and worker"),
                             ({"catalog": "/nonexistent/catalog"}, "not a directory"),
                             ({"worker": {"sandbox": True}}, "optional environment, timeout_seconds"),
                             ({"worker": {"environment": "SERPER_API_KEY"}}, "list environment variable names"),
                             ({"worker": {"environment": ["SERPER API KEY"]}}, "list environment variable names"),
                             ({"worker": {"timeout_seconds": 0}}, "timeout_seconds must be a positive number"),
                             ({"worker": {"memory_gib": float("inf")}}, "memory_gib must be a positive number"),
                             ({"worker": {"write_files": "yes"}}, "write_files must be true or false")):
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
            app = create_app(config(catalog=directory), implementations={"serper_image_search": Lens()})
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

    def test_generated_tools_cannot_shadow_router_tools(self):
        pytest.importorskip("fastapi", reason="router tests need the server extra")
        from openvlharness.services.server import create_app

        with tempfile.TemporaryDirectory() as directory:
            write_tool(directory, "Image_Search_Tool", "shadow", cls="Shadow")
            with self.assertRaisesRegex(ValueError, "shadow base tools"):
                create_app(config(catalog=directory), implementations={"serper_image_search": object()})
