"""ToolSmith catalog loading, generated-tool calls, per-query retrieval and router wiring, offline."""
import copy
import json
import math
import os
import subprocess
import sys
import tempfile
import time
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
from openvlharness.toolsmith.catalog import json_schema, load_catalog, tool_problem
from openvlharness.toolsmith.errors import CatalogLoadError
from openvlharness.toolsmith.isolation import available

ISOLATION = available()[0]  # router tests confine the worker where this host allows it

READ_ONLY = """from openvlharness.toolsmith import GeneratedTool


class ReadOnly(GeneratedTool):
    \"\"\"A hand-written tool whose schema is a property without a setter.\"\"\"

    input_schema = property(lambda self: {"type": "object", "properties": {"word": {"type": "string"}},
                                          "required": ("word",)})

    def __init__(self):
        self.tool_name, self.tool_description = "read_only", "Echo a word."

    def execute(self, word=""):
        return word
"""

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
import subprocess

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


MARKER_TOOL = '''
import sys

from openvlharness.toolsmith import GeneratedTool

sys.toolsmith_marker_imported = True


class Marker(GeneratedTool):
    def __init__(self):
        super().__init__(tool_name="marker", tool_description="bird egg marker", input_schema={"type": "object"})

    def execute(self):
        return "marked"
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

    def test_tools_the_agent_cannot_offer_are_skipped_or_rejected_in_strict_mode(self):
        cases = {"listed": ('tool_description="listed"', 'tool_description=["listed"]',
                            "tool_description must be strings and input_schema a mapping"),
                 "typed": ('{"type": "string"}', '{"type": "str"}', "not a valid JSON Schema"),
                 "linked": ('{"type": "string"}', '{"$ref": "file:///etc/hosts"}',
                            r"uses \$ref, which generated tools may not use: such keywords can make argument "
                            r"validation fetch documents or take exponential time"),
                 # A pattern that takes exponential time on an argument ending in "!": seconds at 25 characters.
                 "patterned": ('{"type": "string"}', '{"type": "string", "pattern": "^([A-Za-z]+\\\\s?)+$"}',
                               "uses pattern, which generated tools may not use"),
                 "keyed": ('{"type": "string"}', '{"type": "object", "patternProperties": {"^(a+)+$": {}}}',
                           "uses patternProperties, which generated tools may not use")}
        for name, (old, new, message) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                write_tool(directory, "good_lookup", "kept")
                write_tool(directory, name, name)
                path = Path(directory, f"{name}.py")
                path.write_text(path.read_text().replace(old, new, 1))
                with self.assertLogs("openvlharness.toolsmith.catalog", "WARNING") as logs:
                    self.assertEqual(list(load_catalog(directory)), ["good_lookup"])
                self.assertRegex(logs.output[0], message)
                with self.assertRaisesRegex(ValueError, message):
                    load_catalog(directory, strict=True)

    def test_a_schema_the_validator_cannot_apply_fails_the_call_without_loading_anything(self):
        import http.server
        import threading

        requests_seen = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                requests_seen.append(self.path)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"type": "string"}')

            def log_message(self, *args):
                pass
        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        with tempfile.TemporaryDirectory() as directory:
            secret = Path(directory, "credentials.json")
            secret.write_text('{"client_secret": "not-for-the-model"}')
            for schema in ({"type": "object", "properties": {"x": {"type": "str"}}},
                           {"type": "object", "properties": {"x": {"$ref": secret.as_uri()}}},
                           {"type": "object", "properties": {"x": {"$ref": f"http://127.0.0.1:{server.server_port}/s"}}}):
                tool = types.SimpleNamespace(tool_name="city_lookup", tool_description="d", input_schema=schema)
                capability = Capability(tool, client_config={"base_url": "http://router.invalid"}, client=FakeClient())
                with self.subTest(schema=schema):
                    result = capability.invoke(VirtualCall("call-1", "city_lookup", {"x": "Paris"}), MemoryStore())
                    self.assertEqual(result.error["code"], "invalid_tool_schema")
                    self.assertNotIn("not-for-the-model", json.dumps(result.error))
        self.assertEqual(requests_seen, [])

    def test_refused_keywords_count_only_where_they_are_schema_keywords(self):
        accepted = {
            "properties named like keywords": {"type": "object", "properties": {
                "$id": {"type": "string"}, "$ref": {"type": "string"}, "pattern": {"type": "string"},
                "patternProperties": {"type": "string"}}},
            "definitions named like keywords": {"type": "object", "definitions": {"$id": {"type": "string"},
                                                                                  "pattern": {"type": "string"}}},
            "tuples, as JSON carries them as arrays": {
                "type": "object", "properties": {"kind": {"type": ("string", "null"), "enum": ("a", "b", None)}},
                "required": ("kind",)},
            "data that looks like a keyword": {"type": "object", "properties": {"x": {
                "enum": [{"$ref": "http://example.com/s"}], "default": {"$id": "http://example.com/"},
                "examples": [{"pattern": "^(a+)+$"}]}}},
        }
        rejected = {
            "a reference to a URL": ({"type": "object", "properties": {"x": {"$ref": "http://example.com/s"}}},
                                     r"uses \$ref, which generated tools may not use"),
            # Local references too: two per level make a small schema take exponential time to validate.
            "a reference within the schema": ({"type": "object", "definitions": {"city": {"type": "string"}},
                                               "properties": {"city": {"$ref": "#/definitions/city"}}}, r"\$ref"),
            "a reference to the whole schema": ({"type": "object", "properties": {"child": {"$ref": "#"}}}, r"\$ref"),
            # Any draft: a subschema in Draft 3 or 4 has keywords (extends, id) the scan does not model.
            "a known draft": ({"$schema": "http://json-schema.org/draft-07/schema#", "type": "object"}, r"\$schema"),
            "a nested Draft 3": ({"type": "object", "properties": {"x": {
                "$schema": "http://json-schema.org/draft-03/schema#", "extends": {"$ref": "http://192.0.2.1/s"}}}},
                r"\$schema"),
            "a nested Draft 4 id": ({"type": "object", "properties": {"x": {
                "$schema": "http://json-schema.org/draft-04/schema#", "id": "http://192.0.2.1/base.json"}}},
                r"\$schema"),
            "a malformed $schema URL": ({"type": "object", "properties": {"x": {"$schema": "http://[::1"}}},
                                        r"\$schema"),
            "a reference to a file in a list of schemas": ({"anyOf": [{"$ref": "file:///etc/hosts"}]}, r"\$ref"),
            "a reference in a tuple of schemas": ({"anyOf": ({"$ref": "file:///etc/hosts"},)}, r"\$ref"),
            "a relative reference in a dependency": ({"type": "object", "dependencies": {"x": {"$ref": "s.json"}}},
                                                     r"\$ref"),
            "an $id keyword": ({"type": "object", "properties": {"x": {"$id": "http://example.com/"}}}, r"\$id"),
            "an anchor": ({"type": "object", "$anchor": "here"}, r"\$anchor"),
            "a pattern": ({"type": "object", "properties": {"x": {"type": "string", "pattern": "^([A-Za-z]+\\s?)+$"}}},
                          "uses pattern, which generated tools may not use"),
            "a pattern on property names": ({"type": "object", "propertyNames": {"pattern": "^(a+)+$"}}, "uses pattern"),
            "patternProperties": ({"type": "object", "patternProperties": {"^(a+)+$": {}}}, "uses patternProperties"),
            "an invalid type": ({"type": "object", "properties": {"n": {"type": "int"}}}, "not a valid JSON Schema"),
            # The router calls execute(**args): the root must be an object of named arguments.
            "an array root": ({"type": "array", "items": {"type": "string"}}, r'must have "type": "object".*"array"'),
            "no root type": ({"properties": {"city": {"type": "string"}}}, r'must have "type": "object".*has none'),
            "a nullable root": ({"type": ["object", "null"]}, r'must have "type": "object"'),
            "a root union": ({"anyOf": [{"type": "object"}, {"type": "string"}]}, r'must have "type": "object"'),
        }
        for name, schema in accepted.items():
            with self.subTest(name=name):
                self.assertIsNone(tool_problem({"tool_name": "t", "tool_description": "d", "input_schema": schema}))
        for name, (schema, message) in rejected.items():
            with self.subTest(name=name):
                self.assertRegex(tool_problem({"tool_name": "t", "tool_description": "d", "input_schema": schema}),
                                 message)
        with tempfile.TemporaryDirectory() as directory:
            write_tool(directory, "city_lookup", "kept")
            path = Path(directory, "city_lookup.py")
            path.write_text(path.read_text().replace('{"type": "string"}', '{"type": ("string", "null")}', 1)
                            .replace('["image_id"]', '("image_id",)', 1))
            self.assertEqual(list(load_catalog(directory, strict=True)), ["city_lookup"])

    def test_the_schema_the_agent_validates_with_is_the_one_the_catalog_checked(self):
        # Tuples pass the check as the arrays JSON carries; the validator gets lists too.
        schema = {"type": "object", "properties": {"city": {"type": ("string", "null")}}, "required": ("city",)}
        tool = types.SimpleNamespace(tool_name="city_lookup", tool_description="d", input_schema=schema)
        self.assertIsNone(tool_problem(tool))
        capability = Capability(tool, client_config={"base_url": "http://router.invalid"}, client=FakeClient("Paris"))
        self.assertEqual(capability.schema["function"]["parameters"]["required"], ["city"])
        refused = capability.invoke(VirtualCall("call-1", "city_lookup", {"city": 3}), MemoryStore())
        self.assertEqual(refused.error["code"], "invalid_tool_arguments")
        self.assertIn("is not of type 'string', 'null'", refused.error["message"])
        missing = capability.invoke(VirtualCall("call-2", "city_lookup", {}), MemoryStore())
        self.assertEqual(missing.error["code"], "invalid_tool_arguments")
        self.assertTrue(capability.invoke(VirtualCall("call-3", "city_lookup", {"city": "Paris"}), MemoryStore()).ok)
        with tempfile.TemporaryDirectory() as directory:
            write_tool(directory, "city_lookup", "kept")
            path = Path(directory, "city_lookup.py")
            path.write_text(path.read_text().replace('["image_id"]', '("image_id",)', 1))
            self.assertEqual(json_schema(load_catalog(directory)["city_lookup"].input_schema)["required"], ["image_id"])

    def test_a_catalog_directory_that_cannot_be_listed_is_refused_not_read_as_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            write_tool(directory, "city_lookup", "kept")
            Path(directory).chmod(0)
            try:
                if os.access(directory, os.R_OK):
                    self.skipTest("this user reads any directory")
                with self.assertRaisesRegex(CatalogLoadError, "cannot be read"):
                    load_catalog(directory)
            finally:
                Path(directory).chmod(0o755)

    def test_a_tool_whose_arguments_are_no_object_is_skipped_or_refused(self):
        # Its execute() could pass a dummy run on defaults, yet the router's execute(**args) would fail every call.
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "listing.py").write_text(READ_ONLY.replace(
                '{"type": "object", "properties": {"word": {"type": "string"}},\n                                          "required": ("word",)}',
                '{"type": "array", "items": {"type": "string"}}').replace('"read_only"', '"listing"'))
            write_tool(directory, "city_lookup", "kept")
            self.assertEqual(list(load_catalog(directory)), ["city_lookup"])
            with self.assertRaisesRegex(CatalogLoadError, r'listing\.py: input_schema must have "type": "object"'):
                load_catalog(directory, strict=True)

    def test_a_tool_may_keep_its_schema_in_a_read_only_property(self):
        # The loader leaves the schema on the tool; its readers take the checked form through json_schema.
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "read_only.py").write_text(READ_ONLY)
            tool = load_catalog(directory, client="router", strict=True)["read_only"]
            self.assertEqual((tool.client, tool.input_schema["required"]), ("router", ("word",)))
            capability = Capability(tool, client_config={"base_url": "http://router.invalid"}, client=FakeClient("hi"))
            self.assertEqual(capability.schema["function"]["parameters"]["required"], ["word"])
            # A tool that cannot take its client fails as a module that cannot build its tool, by name.
            Path(directory, "read_only.py").write_text(READ_ONLY.replace("    def __init__", "    client = property()\n\n"
                                                                                         "    def __init__", 1))
            with self.assertRaisesRegex(CatalogLoadError, r"cannot load read_only\.py: AttributeError"):
                load_catalog(directory)

    def test_a_schema_built_to_make_validation_exponential_is_refused_at_once(self):
        definitions = {"d0": {"type": "string"}}
        for level in range(1, 41):  # each level refers twice to the one below: 2**40 checks of the leaf
            below = {"$ref": f"#/definitions/d{level - 1}"}
            definitions[f"d{level}"] = {"allOf": [below, below]}
        schema = {"type": "object", "definitions": definitions, "properties": {"city": {"$ref": "#/definitions/d40"}}}
        start = time.monotonic()
        self.assertRegex(tool_problem({"tool_name": "t", "tool_description": "d", "input_schema": schema}), r"\$ref")
        self.assertLess(time.monotonic() - start, 1)

    def test_a_schema_that_breaks_the_checker_skips_only_its_tool(self):
        from jsonschema import Draft7Validator

        real = Draft7Validator.check_schema

        def check_schema(schema):
            if schema.get("title") == "breaks the checker":
                raise OverflowError("the repetition number is too large")
            return real(schema)
        with tempfile.TemporaryDirectory() as directory:
            write_tool(directory, "city_lookup", "kept")
            write_tool(directory, "zz_broken", "skipped")
            path = Path(directory, "zz_broken.py")
            text = path.read_text()
            self.assertIn('"type": "object", "properties"', text)
            path.write_text(text.replace('"type": "object", "properties"',
                                         '"title": "breaks the checker", "type": "object", "properties"', 1))
            with patch.object(Draft7Validator, "check_schema", staticmethod(check_schema)), \
                    self.assertLogs("openvlharness.toolsmith.catalog", "WARNING") as logs:
                self.assertEqual(list(load_catalog(directory)), ["city_lookup"])
        self.assertIn("zz_broken.py: input_schema cannot be checked (OverflowError: the repetition number", logs.output[0])

    def test_nominatim_and_musicbrainz_get_one_request_a_second_from_each_process(self):
        from openvlharness.toolsmith.catalog import DEFAULT_CATALOG

        tools = load_catalog(DEFAULT_CATALOG)

        class Clock:  # stands in for the time module of each tool
            def __init__(self):
                self.now, self.slept = 1000.0, []

            def monotonic(self):
                return self.now

            def sleep(self, seconds):
                self.slept.append(round(seconds, 3))
                self.now += seconds

        class Response:
            status_code = 200

            def __init__(self, body):
                self.body = body

            def raise_for_status(self):
                pass

            def json(self):
                return self.body

        def get(url, params=None, headers=None, timeout=None):
            if "nominatim" in url:
                return Response([{"display_name": "Paris", "lat": "48.85", "lon": "2.35"}])
            if url.endswith("/artist"):
                return Response({"artists": [{"id": "a1", "name": "Queen"}]})
            if "musicbrainz" in url:
                return Response({"id": "a1", "name": "Queen"})
            return Response({"results": [{"id": 1, "title": "Heat", "original_title": "Heat"}]})
        for name, call, requests_made in (
                ("gazetteer_lookup", lambda tool: tool.execute(query="Paris"), 1),
                ("media_catalog", lambda tool: tool.execute(title="Queen", media_type="music"), 2)):
            namespace = type(tools[name]).execute.__globals__  # the tool module's, which load_catalog keeps
            clock = Clock()
            with self.subTest(tool=name), patch.dict(namespace, {"time": clock}), \
                    patch.object(namespace["requests"], "get", side_effect=get):
                for _ in range(2):
                    self.assertTrue(call(tools[name])["found"])
                # The first request waits for nothing; each later one for the rest of its second.
                self.assertEqual(clock.slept, [0.0] + [1.0] * (2 * requests_made - 1))
        namespace = type(tools["media_catalog"]).execute.__globals__
        clock = Clock()
        with patch.dict(os.environ, {"TMDB_API_KEY": "fixture"}), patch.dict(namespace, {"time": clock}), \
                patch.object(namespace["requests"], "get", side_effect=get):
            tools["media_catalog"].execute(title="Heat", media_type="film")
        self.assertEqual(clock.slept, [])  # TMDB is not throttled

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

    def test_unit_conversions_keep_full_precision_and_format_only_for_display(self):
        tool = self.tools["unit_compute"]
        tiny = tool.execute(op="convert", value=1, from_unit="mg", to_unit="t")
        self.assertEqual((tiny["display"], tiny["family"]), ("1e-09 t", "mass"))
        self.assertTrue(math.isclose(tiny["result"], 1e-9, rel_tol=1e-12))  # not 0.0
        self.assertTrue(math.isclose(tool.execute(op="convert", value=1, from_unit="t", to_unit="mg")["result"], 1e9))
        hot = tool.execute(op="convert", value=100, from_unit="C", to_unit="F")
        self.assertAlmostEqual(hot["result"], 212.0)
        self.assertEqual((hot["display"], hot["family"]), ("212 f", "temperature"))
        self.assertEqual(tool.execute(op="convert", value=1, from_unit="mg", to_unit="C"),
                         {"error": "cannot convert mg to c"})

    def test_wikidata_gives_the_unit_of_a_numeric_property(self):
        tool = self.tools["wikidata_query"]
        feet = {"amount": {"value": "1000"}, "unit": {"value": "http://www.wikidata.org/entity/Q3710"},
                "unitLabel": {"value": "foot"}, "rank": {"value": "http://wikiba.se/ontology#NormalRank"}}

        def get_json(url, params):
            if params.get("action") == "wbsearchentities":
                return {"search": [{"id": "Q1", "label": "Hill", "description": "a hill"}]}
            if "quantityAmount" in params["query"]:
                self.assertIn('rdfs:label "elevation above sea level"@en', params["query"])
                return {"results": {"bindings": [feet]}}
            return {"results": {"bindings": [{"propertyLabel": {"value": "elevation above sea level"},
                                              "valueLabel": {"value": "1000"}}]}}
        with patch.dict(tool.execute.__globals__, {"get_json": get_json}):
            matched = tool.execute(entity="Hill", property="elevation above sea level")["matched"]
        self.assertEqual(matched["quantities"], [{"amount": 1000.0, "unit": "foot", "unit_qid": "Q3710",
                                                  "rank": "normal"}])
        # A units lookup that fails, or a response without results, keeps the match and says so.
        for failure in (RuntimeError("https://query.wikidata.org/sparql did not answer"), {"head": {}},
                        {"results": "junk"}, {"results": {"bindings": "junk"}}, KeyError("results"), ["junk"]):
            def flaky(url, params, failure=failure):
                if "quantityAmount" in params.get("query", ""):
                    if isinstance(failure, Exception):
                        raise failure
                    return failure
                return get_json(url, params)
            with self.subTest(failure=failure), patch.dict(tool.execute.__globals__, {"get_json": flaky}):
                matched = tool.execute(entity="Hill", property="elevation above sea level")["matched"]
                self.assertEqual(matched["values"], ["1000"])
                self.assertNotIn("quantities", matched)
                self.assertTrue(matched["quantities_error"].startswith("the units could not be read: "))
        def malformed_rows(url, params):
            if "quantityAmount" in params.get("query", ""):
                return {"results": {"bindings": ["junk", {"amount": {"value": "x"}}, {"amount": {"value": "5"}, "unit": 3},
                                                 {"amount": {"value": "7"}, "unit": {"value": "http://e/Q11573"}}]}}
            return get_json(url, params)
        with patch.dict(tool.execute.__globals__, {"get_json": malformed_rows}):  # malformed rows are skipped
            self.assertEqual(tool.execute(entity="Hill", property="elevation above sea level")["matched"]["quantities"],
                             [{"amount": 7.0, "unit": "http://e/Q11573", "unit_qid": "Q11573", "rank": ""}])
        with patch.dict(tool.execute.__globals__, {"get_json": lambda url, params: {
                "search": [{"id": "Q1", "label": "Hill"}]} if params.get("action") else {"head": {}}}):
            self.assertEqual(tool.execute(entity="Hill"),
                             {"found": False, "error": "Wikidata's query service returned no results"})

    def test_elevation_is_converted_to_metres_or_refused(self):
        tool = self.tools["place_elevation_from_image"]
        self.addCleanup(setattr, tool, "client", None)

        def answer(*quantities):
            record = {"found": True, "qid": "Q1", "matched": {"property": "elevation above sea level", "values": ["1000"],
                                                            "confidence": 1.0, "quantities": list(quantities)}}
            tool.client = types.SimpleNamespace(execute=lambda name, args, tool_call_id: {"ok": True, "result": {
                "visual_matches": [{"title": "Hill - Wikipedia", "link": "https://en.wikipedia.org/wiki/Hill",
                                    "source": "Wikipedia"}], "wikidata_query": record}[name]})
            return tool.execute(image_base64="AAAA")
        foot = {"amount": 1000.0, "unit": "foot", "unit_qid": "Q3710", "rank": "normal"}
        metre = {"amount": 8848.86, "unit": "metre", "unit_qid": "Q11573", "rank": "preferred"}
        self.assertEqual(answer(foot), "304.8 m")  # not "1000 m"
        self.assertEqual(answer({**metre, "amount": 8848.0, "rank": "normal"}, metre), "8848.86 m")  # preferred first
        self.assertEqual(answer({"amount": 1.0, "unit": "kilometre", "unit_qid": "Q828224", "rank": "normal"}),
                         {"error": "Wikidata gives the elevation of Hill in kilometre, not in metres or feet",
                          "candidate": "Hill"})
        self.assertEqual(answer(), {"error": "Wikidata gives no elevation with a unit for Hill", "candidate": "Hill"})

    def test_wikidata_reports_a_missing_qid_as_not_found(self):
        tool = self.tools["wikidata_query"]

        def get_json(url, params):
            if params.get("action") != "wbgetentities":
                raise AssertionError(f"no claims are read for a missing item ({url})")
            return {"entities": {"Q999999999": {"id": "Q999999999", "missing": ""}}}
        with patch.dict(tool.execute.__globals__, {"get_json": get_json}):
            self.assertEqual(tool.execute(entity="Q999999999"),
                             {"found": False, "error": "no Wikidata entity matches 'Q999999999'"})

    def test_page_readers_take_the_routers_summary_not_the_pages_own(self):
        from openvlharness.services.tools.serper_webpage_visit.tool import Webpage_Visit_Tool as Visit
        from openvlharness.toolsmith.pipeline.examples import actress_by_character, esports_player_stats

        url = "https://en.wikipedia.org/wiki/Cheetah"
        evidence = "Summary: The cheetah is a large cat.\n\nSummary: \nA heading\n=======\nAfrica"
        page = Visit._format_success(url, "size", evidence=evidence, summary="Adults weigh 21-72 kg.")
        tool = self.tools["animal_typical_size_by_image"]
        tool.client = types.SimpleNamespace(execute=lambda name, args, tool_call_id: {"ok": True, "result": {
            "visual_matches": [{"title": "Cheetah - Wikipedia", "link": url, "source": "Wikipedia"}],
            "Webpage_Visit_Tool": page}[name]})
        self.addCleanup(setattr, tool, "client", None)
        self.assertEqual(tool.execute(image_base64="AAAA"), "Adults weigh 21-72 kg.")
        pages = "\n\n=======\n\n".join([page, Visit._format_failure(url, "size", "HTTP 404"),
                                           Visit._format_success(url, "size", evidence, "Sprints at 100 km/h.")])
        for module in (actress_by_character, esports_player_stats):
            self.assertEqual(list(module.page_answers(pages)), ["Adults weigh 21-72 kg.", "Sprints at 100 km/h."])

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

    def test_media_catalog_prefers_an_exact_tmdb_title_to_the_first_hit(self):
        tool = self.tools["media_catalog"]
        hits = [{"id": 1, "title": "Dune: Part Two"}, {"id": 2, "title": "Dune"}]
        fetched = []

        def get_json(url, params):
            if "/search/" in url:
                return {"results": hits}
            fetched.append(url.rsplit("/", 1)[1])
            return {"id": int(fetched[-1]), "title": {"1": "Dune: Part Two", "2": "Dune"}[fetched[-1]]}
        with patch.dict(os.environ, {"TMDB_API_KEY": "key"}), patch.dict(tool.execute.__globals__, {"get_json": get_json}):
            self.assertEqual(tool.execute(title="dune", media_type="film")["canonical_title"], "Dune")
            self.assertEqual(tool.execute(title="Dune 2", media_type="film")["canonical_title"], "Dune: Part Two")
        self.assertEqual(fetched, ["2", "1"])

    def test_media_catalog_prefers_an_exact_release_to_a_fuzzy_artist(self):
        tool = self.tools["media_catalog"]
        searches = {"artist": [{"name": "Thriller Kids", "id": "a1"}], "release": [{"title": "Thriller", "id": "r1"}]}
        records = {"a1": {"id": "a1", "name": "Thriller Kids", "type": "Group"},
                   "r1": {"id": "r1", "title": "Thriller", "date": "1982-11-30",
                          "artist-credit": [{"name": "Michael Jackson"}]}}
        requested, includes = [], {}

        def get_json(url, params):
            requested.append(url.rsplit("/ws/2/", 1)[1])
            includes[requested[-1]] = params.get("inc")
            entity = url.rsplit("/", 2)
            return records[entity[-1]] if entity[-1] in records else {f"{entity[-1]}s": searches[entity[-1]]}
        with patch.dict(tool.execute.__globals__, {"get_json": get_json}):
            release = tool.execute(title="thriller", media_type="music")
            self.assertEqual((release["canonical_title"], release["artist"]), ("Thriller", "Michael Jackson"))
            self.assertEqual(includes["release/r1"], "aliases+artist-credits")  # MusicBrainz omits them otherwise
            requested.clear()
            self.assertEqual(tool.execute(title="Thriller Kids", media_type="music")["canonical_title"], "Thriller Kids")
            self.assertEqual(requested, ["artist", "artist/a1"])
            self.assertEqual(includes["artist/a1"], "aliases")
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
        with fake_encoder(), patch("openvlharness.toolsmith.agent.router_catalog", return_value=None):  # a local read
            agent = ToolSmithAgent(model, {}, MemoryStore, "fixture system", "Question: {user_query}",
                                   toolsmith={"catalog": directory, "top_k": top_k, "worker": {"isolation": ISOLATION}},
                                   client_config={"base_url": "http://router.invalid", "retry": 0})
        return agent, requests

    def router_agent(self, response, **toolsmith):
        """A ToolSmithAgent whose router answers GET /toolsmith/catalog with response; no local worker may start."""
        model = ModelAdapter({"alias": "qwen3-vl-8b-instruct", "base": "http://fixture.invalid/v1"})
        get = patch("openvlharness.toolsmith.agent.requests.get", side_effect=lambda url, timeout: response(url))
        local = patch("openvlharness.toolsmith.agent.catalog_snapshot", side_effect=AssertionError("a local worker"))
        with fake_encoder(), get as fetched, local:
            agent = ToolSmithAgent(model, {}, MemoryStore, "s", "{user_query}", toolsmith={"top_k": 1, **toolsmith},
                                   client_config={"base_url": "http://router.example:18101/", "retry": 0})
        return agent, [call.args[0] for call in fetched.call_args_list]

    @staticmethod
    def reply(body, status=200):
        response = __import__("requests").Response()
        response.status_code, response._content = status, json.dumps(body).encode()
        return lambda url: response

    def test_the_agent_takes_its_generated_tools_from_the_router(self):
        # An evaluator on a host without user namespaces needs no local worker, nor the catalog's files.
        schema = {"type": "object", "properties": {"image_id": {"type": "string"}}, "required": ["image_id"]}
        served = {"catalog_sha256": "a" * 64, "tools": [
            {"tool_name": "bird_egg_lookup", "tool_description": "bird egg incubation days", "input_schema": schema}]}
        with patch("openvlharness.toolsmith.isolation.available", return_value=(False, "no user namespaces")):
            agent, urls = self.router_agent(self.reply(served), catalog="/nowhere/on/this/host")
        self.assertEqual(urls, ["http://router.example:18101/toolsmith/catalog"])
        self.assertEqual(agent.catalog_sha256, "a" * 64)
        self.assertEqual(agent.generated["bird_egg_lookup"].public_schema()["function"]["parameters"], schema)
        self.assertEqual(agent.tools_for([{"type": "text", "value": "bird egg"}])["bird_egg_lookup"],
                         agent.generated["bird_egg_lookup"])

    def test_the_agent_refuses_a_catalog_from_the_router_that_it_cannot_offer(self):
        good = {"tool_name": "t", "tool_description": "d", "input_schema": {"type": "object"}}
        for served, message in (({"tools": [good]}, "serves no valid ToolSmith catalog"),
                                ({"catalog_sha256": "a" * 64, "tools": {"t": good}}, "serves no valid ToolSmith catalog"),
                                ({"catalog_sha256": "a" * 64, "tools": [{**good, "input_schema": {"type": "array"}}]},
                                 r"cannot offer \('t'\): input_schema must have \"type\": \"object\""),
                                ({"catalog_sha256": "a" * 64, "tools": [{**good, "input_schema": {
                                    "type": "object", "properties": {"x": {"$ref": "http://192.0.2.1/s"}}}}]},
                                 r"uses \$ref"),
                                ({"catalog_sha256": "a" * 64, "tools": [good, good]}, "listed twice")):
            with self.subTest(message=message), self.assertRaisesRegex(CatalogLoadError, message):
                self.router_agent(self.reply(served))
        with self.assertRaisesRegex(CatalogLoadError, r"cannot read the ToolSmith catalog .*\(HTTPError\)"):
            self.router_agent(self.reply({}, status=500))

        def refused(url):
            raise __import__("requests").ConnectionError("refused")
        with self.assertRaisesRegex(CatalogLoadError, r"\(ConnectionError\); start the router"):
            self.router_agent(refused)

    def test_only_a_router_without_the_list_makes_the_agent_read_the_catalog_itself(self):
        with tempfile.TemporaryDirectory() as directory:
            write_tool(directory, "bird_egg_lookup", "bird egg incubation days")
            model = ModelAdapter({"alias": "qwen3-vl-8b-instruct", "base": "http://fixture.invalid/v1"})
            with fake_encoder(), patch("openvlharness.toolsmith.agent.requests.get",
                                       side_effect=lambda url, timeout: self.reply({"detail": "Not Found"}, 404)(url)), \
                    self.assertLogs("openvlharness.toolsmith.agent", "WARNING") as logs:
                agent = ToolSmithAgent(model, {}, MemoryStore, "s", "{user_query}",
                                       toolsmith={"catalog": directory, "worker": {"isolation": ISOLATION}},
                                       client_config={"base_url": "http://router.example:18101", "retry": 0})
            from openvlharness.toolsmith.catalog import catalog_fingerprint
            self.assertEqual((list(agent.generated), agent.catalog_sha256), (["bird_egg_lookup"], catalog_fingerprint(directory)))
        self.assertIn("serves no ToolSmith catalog list; reading", logs.output[0])

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

    def test_an_encoder_that_cannot_load_says_what_to_change(self):
        from openvlharness.toolsmith import retrieval
        from openvlharness.toolsmith.retrieval import EncoderError, ToolRetriever

        refusal = ("Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`, we now "
                   "require users to upgrade torch to at least v2.6 (CVE-2025-32434).")

        def refusing(error):
            def build(name):
                raise error
            return types.SimpleNamespace(SentenceTransformer=build)
        with patch.dict(sys.modules, {"sentence_transformers": refusing(ValueError(refusal))}):
            blamed = r"PyTorch 2\.6 or newer, and this is PyTorch 2\.5.*toolsmith\.encoder.*\(ValueError: Due to a"
            with patch.object(retrieval, "_torch_version", lambda: (2, 5)), \
                    self.assertRaisesRegex(EncoderError, blamed) as caught:
                ToolRetriever({"t": "d"}, encoder="BAAI/bge-m3")
            self.assertIsInstance(caught.exception.__cause__, ValueError)
            with patch.object(retrieval, "_torch_version", lambda: (2, 6)), \
                    self.assertRaisesRegex(EncoderError, r"^Cannot load the sentence encoder 'BAAI/bge-m3': "
                                                         r"ValueError: Due to a serious"):
                ToolRetriever({"t": "d"}, encoder="BAAI/bge-m3")  # PyTorch 2.6 refused it: not a version problem
        # PyTorch 2.6's own weights-only refusal of a checkpoint is not blamed on an old PyTorch.
        later = RuntimeError("Weights only load failed. In PyTorch 2.6, we changed the default value of the "
                             "`weights_only` argument in `torch.load` from `False` to `True`.")
        with patch.dict(sys.modules, {"sentence_transformers": refusing(later)}), \
                patch.object(retrieval, "_torch_version", lambda: (2, 5)), \
                self.assertRaisesRegex(EncoderError, r"^Cannot load the sentence encoder 'mine': RuntimeError: "
                                                     r"Weights only load failed"):
            ToolRetriever({"t": "d"}, encoder="mine")
        with patch.dict(sys.modules, {"sentence_transformers": refusing(OSError("no such model"))}):
            with self.assertRaisesRegex(EncoderError, "Cannot load the sentence encoder 'nope': OSError: no such model"):
                ToolRetriever({"t": "d"}, encoder="nope")
        with patch.dict(sys.modules, {"sentence_transformers": None}), \
                self.assertRaisesRegex(EncoderError, "needs sentence-transformers, which the toolsmith extra"):
            ToolRetriever({"t": "d"}, encoder="nope")  # the toolsmith extra is missing

    def test_the_agent_reads_tool_metadata_without_importing_the_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "marker.py").write_text(MARKER_TOOL)
            write_tool(directory, "bird_egg_lookup", "bird egg incubation days")
            agent, _ = self.make_agent(directory, [], top_k=2)
            self.assertFalse(hasattr(sys, "toolsmith_marker_imported"))  # the module ran only in the worker
            self.assertEqual(set(agent.generated), {"bird_egg_lookup", "marker"})
            self.assertEqual(agent.generated["marker"].public_schema()["function"],
                             {"name": "marker", "description": "bird egg marker", "parameters": {"type": "object"}})

    def test_main_agent_offers_every_configured_tool(self):
        model = ModelAdapter({"alias": "qwen3-vl-8b-instruct", "base": "http://fixture.invalid/v1"})
        agent = MainAgent(model, {}, MemoryStore, "fixture system", "Question: {user_query}")
        self.assertIs(agent.tools_for([{"type": "text", "value": "How many?"}]), agent.capabilities)

    def test_generated_tools_cannot_shadow_base_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            write_tool(directory, "Image_Search_Tool", "shadow", cls="Shadow")
            model = ModelAdapter({"alias": "qwen3-vl-8b-instruct", "base": "http://fixture.invalid/v1"})
            base = Capability(load_catalog(directory)["Image_Search_Tool"], client_config={"base_url": "http://router.invalid"})
            with fake_encoder(), patch("openvlharness.toolsmith.agent.router_catalog", return_value=None), \
                    self.assertRaisesRegex(
                        CatalogLoadError, f"^the ToolSmith catalog {directory} has tools that shadow base tools: "
                                          "Image_Search_Tool; rename them$"):
                ToolSmithAgent(model, {"Image_Search_Tool": base}, MemoryStore, "s", "{user_query}",
                               toolsmith={"catalog": directory, "worker": {"isolation": ISOLATION}},
                               client_config={"base_url": "http://router.invalid"})


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
                             ({"router": "tfidf"}, "optional catalog, top_k, encoder, worker and trust_router_catalog"),
                             ({"trust_router_catalog": "yes"}, "trust_router_catalog must be true or false"),
                             ({"trust_router_catalog": None}, "trust_router_catalog must be true or false"),
                             ({"catalog": ""}, "not a directory path"),
                             ({"catalog": None}, "not a directory path"), ({"catalog": 3}, "not a directory path"),
                             ({"encoder": None}, "encoder must name a sentence-transformers model"),
                             ({"encoder": " "}, "encoder must name a sentence-transformers model"),
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
        # A catalog not on this host is a valid setting for an evaluator, which reads it from the router; the
        # router itself refuses it, in one line.
        self.assertEqual(load_agent_config(config(catalog="/nonexistent/catalog"))["toolsmith"]["catalog"],
                         "/nonexistent/catalog")

    def test_the_router_refuses_a_catalog_that_is_not_on_its_host(self):
        pytest.importorskip("fastapi", reason="router tests need the server extra")
        from openvlharness.services.server import create_app
        with self.assertRaisesRegex(CatalogLoadError, "'/nonexistent/catalog' is not a directory"):
            create_app(config(catalog="/nonexistent/catalog"), implementations={"serper_image_search": object()})

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

    def test_a_router_without_toolsmith_does_not_load_the_worker(self):
        pytest.importorskip("fastapi", reason="router tests need the server extra")
        # The worker needs the POSIX resource module; other routers must import where it is missing.
        probe = ("import sys; import openvlharness.services.server; "
                 "print('openvlharness.toolsmith.worker' in sys.modules)")
        done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=120,
                              cwd=Path(__file__).resolve().parents[1])
        self.assertEqual(done.stdout.strip(), "False", done.stderr)

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
            client = TestClient(app)
            reported = client.get("/env").json()["toolsmith"]
            listed = client.get("/toolsmith/catalog").json()
        self.assertEqual(reported, {"catalog_sha256": started})
        # The tools the worker validated, with the digest of the same snapshot: what evaluators read.
        self.assertEqual(listed["catalog_sha256"], started)
        self.assertEqual([(tool["tool_name"], tool["tool_description"]) for tool in listed["tools"]],
                         [("bird_egg_lookup", "Bird egg facts")])
        self.assertEqual(listed["tools"][0]["input_schema"]["type"], "object")
        plain = create_app({key: value for key, value in config().items() if key != "toolsmith"},
                           implementations={"serper_image_search": object()})
        self.assertEqual(TestClient(plain).get("/toolsmith/catalog").status_code, 404)  # only ToolSmith routers

    def test_other_loader_rules_make_another_catalog_fingerprint(self):
        from openvlharness.toolsmith import catalog as catalog_module

        with tempfile.TemporaryDirectory() as directory:
            write_tool(directory, "bird_egg_lookup", "Bird egg facts")
            started = catalog_module.catalog_fingerprint(directory)
            with patch.object(catalog_module, "LOADER_RULES", catalog_module.LOADER_RULES + 1):
                self.assertNotEqual(catalog_module.catalog_fingerprint(directory), started)
            self.assertEqual(catalog_module.catalog_fingerprint(directory), started)

    def test_generated_tools_cannot_shadow_router_tools(self):
        pytest.importorskip("fastapi", reason="router tests need the server extra")
        from openvlharness.services.server import create_app

        with tempfile.TemporaryDirectory() as directory:
            write_tool(directory, "Image_Search_Tool", "shadow", cls="Shadow")
            with self.assertRaisesRegex(
                    CatalogLoadError, f"^the ToolSmith catalog {directory} has tools that shadow base tools: "
                                      "Image_Search_Tool; rename them$"):
                create_app(config(catalog=directory, worker={"isolation": ISOLATION}),
                           implementations={"serper_image_search": object()})
