"""Generated tools run in a separate, resource-limited worker process, offline."""
import copy
import json
import os
import re
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from openvlharness.services.envelope import _build_error_payload, _build_success_payload, _normalize_result_payload
from openvlharness.toolsmith import worker as worker_module
from openvlharness.toolsmith.catalog import load_catalog
from openvlharness.toolsmith.worker import CatalogWorker

TOOL = '''
import os
import time

from openvlharness.services.tools.base import ToolArgumentError
from openvlharness.toolsmith import GeneratedTool, ToolCallError


class {cls}(GeneratedTool):
    def __init__(self):
        super().__init__(tool_name="{name}", tool_description="{name} tool", input_schema={{
            "type": "object", "properties": {{"value": {{"type": "string"}}}}}})

    def execute(self, value=""):
{body}
'''

BODIES = {
    "probe": """        print("tool output on stdout")
        names = ("SERVICE_API_KEY", "GRANTED_TOKEN", "TOOLSMITH_LEGACY_MAY_INPUTS", "PLAIN_SETTING")
        return {"pid": os.getpid(), "cwd": os.getcwd(), "env": {name: os.environ.get(name) for name in names}}""",
    "mixed": """        return {"text": "naïve ✓ \\u2028", "float": 0.1 + 0.2, "big": 10 ** 30, "neg_zero": -0.0,
                5: "int key", "set": {3, 1, 2}, "bytes": b"raw", "nested": [(1, 2.5), None, True],
                "nan": float("nan")}""",
    "fails": """        raise ValueError("bad input: " + value)""",
    "bad_args": """        raise ToolArgumentError("value must be a river")""",
    "crash": """        os._exit(3)""",
    "sleepy": """        time.sleep(float(value))
        return {"slept": float(value), "pid": os.getpid()}""",
    "writer": """        with open("written.txt", "w") as stream:
            stream.write("content")
            stream.flush()
        return "wrote\"""",
    "hog": """        return len(bytearray(int(value)))""",
    "caller": """        try:
            return {"result": self.call_tool(value, query="q")}
        except ToolCallError as exc:
            return {"tool_error": str(exc)}
        except Exception as exc:
            return {"exception": f"{type(exc).__name__}: {exc}"}""",
    "talker": """        return "x" * int(value)""",
}


def write_tool(directory, name, body=None):
    cls = "".join(part.title() for part in name.split("_"))
    Path(directory, f"{name}.py").write_text(TOOL.format(cls=cls, name=name, body=body or BODIES[name]))


class FakeClient:
    """Stands in for the router's ToolClient: answers every call_tool and records it."""

    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, []

    def execute(self, tool_name, args, *, tool_call_id):
        self.calls.append((tool_name, copy.deepcopy(args), tool_call_id))
        if self.error:
            raise self.error
        return {"ok": True, "tool_name": tool_name, "result": copy.deepcopy(self.result), "error": None,
                "tool_call_id": tool_call_id, "meta": {"http_status": 200}}


def in_process(directory, name, args, call_id="call-1"):
    """The envelope the router built when it ran the tool in its own process."""
    tool = load_catalog(directory, client=FakeClient())[name]
    try:
        result, meta = _normalize_result_payload(tool.execute(**args))
    except Exception as exc:
        return _build_error_payload(name, call_id, "tool_execution_error", str(exc),
                                    {"exception_type": type(exc).__name__})
    return _build_success_payload(name, call_id, result, meta)


def wire(payload):
    """The bytes the router sends for an envelope (Starlette's JSONResponse rendering)."""
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


class WorkerTests(unittest.TestCase):
    def catalog(self, *names):
        directory = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(directory, ignore_errors=True))
        for name in names:
            write_tool(directory, name)
        return directory

    def start(self, directory, **options):
        options.setdefault("client", FakeClient())
        worker = CatalogWorker(directory, **options)
        self.addCleanup(worker.close)
        return worker

    def test_tools_run_outside_the_router_and_return_the_in_process_envelopes(self):
        directory = self.catalog("probe", "mixed", "fails", "bad_args")
        worker = self.start(directory)
        self.assertEqual(list(worker.tools), ["bad_args", "fails", "mixed", "probe"])
        self.assertEqual(worker.tools["probe"]["input_schema"]["properties"], {"value": {"type": "string"}})
        payload, status = worker.execute("probe", {}, tool_call_id="call-1")
        self.assertEqual(status, 200)
        self.assertNotEqual(payload["result"]["pid"], os.getpid())
        self.assertNotIn(payload["result"]["cwd"], (os.getcwd(), directory))
        for name, args, expected_status in (("mixed", {}, 200), ("fails", {"value": "x"}, 500),
                                            ("bad_args", {}, 400)):
            with self.subTest(tool=name):
                payload, status = worker.execute(name, args, tool_call_id="call-1")
                self.assertEqual(status, expected_status)
                self.assertEqual(wire(payload), wire(in_process(directory, name, args)))
        self.assertIn("non-finite float converted to string", worker.execute("mixed", {}, tool_call_id="c")[0]
                      ["meta"]["normalization_warnings"][-1])

    def test_the_packaged_catalog_gives_the_same_results_as_in_process(self):
        worker = self.start(None)
        self.assertEqual(list(worker.tools), list(load_catalog()))
        for args in ({"op": "arith", "operator": "add", "operands": [0.1, 0.2]},
                     {"op": "convert", "value": 26.2, "from_unit": "miles", "to_unit": "km"},
                     {"op": "date_diff", "date1": "2024-02-28", "date2": "March 1, 2025"},
                     {"op": "arith", "operator": "divide", "operands": [1, 0]}):
            with self.subTest(args=args):
                payload, status = worker.execute("unit_compute", args, tool_call_id="call-1")
                self.assertEqual(status, 200)
                self.assertEqual(wire(payload), wire(in_process(None, "unit_compute", args)))

    def test_the_environment_drops_credentials_unless_granted(self):
        directory = self.catalog("probe")
        settings = {"SERVICE_API_KEY": "secret", "GRANTED_TOKEN": "granted", "TOOLSMITH_LEGACY_MAY_INPUTS": "1",
                    "PLAIN_SETTING": "kept"}
        with patch.dict(os.environ, settings):
            environment = self.start(directory, environment=["GRANTED_TOKEN"]).execute(
                "probe", {}, tool_call_id="call-1")[0]["result"]["env"]
        self.assertEqual(environment, {"SERVICE_API_KEY": None, "GRANTED_TOKEN": "granted",
                                       "TOOLSMITH_LEGACY_MAY_INPUTS": "1", "PLAIN_SETTING": "kept"})

    def test_file_writes_and_memory_are_limited(self):
        directory = self.catalog("writer", "hog", "probe")
        worker = self.start(directory, memory_gib=1)
        payload, status = worker.execute("writer", {}, tool_call_id="call-1")
        self.assertEqual(status, 500)
        self.assertIn("File too large", payload["error"]["message"])
        pid = worker.execute("probe", {}, tool_call_id="call-2")[0]["result"]["pid"]
        payload, _ = worker.execute("hog", {"value": str(2 << 30)}, tool_call_id="call-3")
        self.assertEqual(payload["error"]["detail"]["exception_type"], "MemoryError")
        self.assertEqual(worker.execute("hog", {"value": "1024"}, tool_call_id="call-4")[0]["result"], 1024)
        self.assertEqual(worker.execute("probe", {}, tool_call_id="call-5")[0]["result"]["pid"], pid)
        writable = self.start(directory, write_files=True)
        self.assertEqual(writable.execute("writer", {}, tool_call_id="call-6")[0]["result"], "wrote")

    def test_a_crash_returns_an_error_and_later_calls_get_a_new_worker(self):
        worker = self.start(self.catalog("crash", "probe"))
        before = worker.execute("probe", {}, tool_call_id="call-1")[0]["result"]["pid"]
        payload, status = worker.execute("crash", {}, tool_call_id="call-2")
        self.assertEqual((status, payload["ok"], payload["tool_call_id"]), (500, False, "call-2"))
        self.assertIn("exited with status 3", payload["error"]["message"])
        payload, status = worker.execute("probe", {}, tool_call_id="call-3")
        self.assertEqual(status, 200)
        self.assertNotEqual(payload["result"]["pid"], before)

    def test_a_call_past_the_time_limit_fails_while_other_calls_finish(self):
        worker = self.start(self.catalog("sleepy", "probe"), timeout_seconds=3)
        old = worker.execute("probe", {}, tool_call_id="call-0")[0]["result"]["pid"]
        results = {}

        def call(key, seconds):
            started = time.monotonic()
            results[key] = (worker.execute("sleepy", {"value": str(seconds)}, tool_call_id=key),
                            time.monotonic() - started)
        hung = threading.Thread(target=call, args=("hung", 60))
        hung.start()
        time.sleep(1.5)
        other = threading.Thread(target=call, args=("other", 2.25))
        other.start()
        hung.join(10)
        other.join(10)
        (payload, status), elapsed = results["hung"]
        self.assertEqual((status, payload["error"]["detail"]["exception_type"]), (500, "TimeoutError"))
        self.assertIn("did not finish within 3 seconds", payload["error"]["message"])
        self.assertLess(elapsed, 10)
        (payload, status), _ = results["other"]
        self.assertEqual((status, payload["result"]["slept"], payload["result"]["pid"]), (200, 2.25, old))
        payload, status = worker.execute("probe", {}, tool_call_id="call-3")
        self.assertEqual(status, 200)
        self.assertNotEqual(payload["result"]["pid"], old)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                os.kill(old, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            self.fail("the worker with the hung call was not stopped")

    def test_tools_reach_only_the_allowed_router_tools(self):
        client = FakeClient(result={"text": "summary"})
        worker = self.start(self.catalog("caller", "probe"), client=client,
                            base_tools={"Text_Search_Tool", "Python_Coding_Agent_Tool", "Visual_Grounding_Tool"})
        for name in ("Text_Search_Tool", "probe", "Unregistered_Tool"):
            with self.subTest(tool=name):
                payload, _ = worker.execute("caller", {"value": name}, tool_call_id="call-1")
                self.assertEqual(payload["result"], {"result": {"text": "summary"}})
        self.assertEqual([(name, args) for name, args, _ in client.calls],
                         [(name, {"query": "q"}) for name in ("Text_Search_Tool", "probe", "Unregistered_Tool")])
        self.assertTrue(all(call_id.startswith("caller-") for _, _, call_id in client.calls))
        for name in ("Python_Coding_Agent_Tool", "Visual_Grounding_Tool"):
            with self.subTest(tool=name):
                payload, _ = worker.execute("caller", {"value": name}, tool_call_id="call-2")
                self.assertEqual(payload["result"], {"tool_error": f"{name}: Generated tools cannot call {name}"})
        self.assertEqual(len(client.calls), 3)
        client.error = requests.exceptions.ReadTimeout("router timed out")
        payload, _ = worker.execute("caller", {"value": "Text_Search_Tool"}, tool_call_id="call-3")
        self.assertEqual(payload["result"], {"exception": "ReadTimeout: router timed out"})

    def test_an_oversized_or_malformed_message_stops_only_that_worker(self):
        worker = self.start(self.catalog("talker", "probe"))
        before = worker.execute("probe", {}, tool_call_id="call-1")[0]["result"]["pid"]
        with patch.object(worker_module, "MAX_MESSAGE_BYTES", 4096):
            worker._retire(worker.current)
            small = worker.execute("probe", {}, tool_call_id="call-2")[0]["result"]["pid"]
        payload, status = worker.execute("talker", {"value": "100000"}, tool_call_id="call-3")
        self.assertEqual(status, 500)
        self.assertIn("invalid message", payload["error"]["message"])
        payload, status = worker.execute("probe", {}, tool_call_id="call-4")
        self.assertEqual(status, 200)
        self.assertNotIn(payload["result"]["pid"], (before, small))

    def test_a_worker_whose_router_is_gone_exits_and_removes_its_copy(self):
        worker = self.start(self.catalog("probe"))
        process, home = worker.current.process, worker.current.home
        self.assertTrue((home / "catalog" / "probe.py").exists())
        process.stdin.close()  # what the worker sees when the router process is killed
        self.assertEqual(process.wait(timeout=10), 0)
        self.assertFalse(home.exists())

    def test_the_worker_keeps_the_catalog_it_started_with(self):
        directory = self.catalog("crash", "probe")
        worker = self.start(directory)
        write_tool(directory, "probe", '        return "changed on disk"')
        write_tool(directory, "added", '        return "new"')
        worker.execute("crash", {}, tool_call_id="call-1")
        payload, status = worker.execute("probe", {}, tool_call_id="call-2")
        self.assertEqual(status, 200)
        self.assertIn("pid", payload["result"])
        self.assertNotIn("added", worker.tools)

    def test_catalog_errors_are_raised_when_the_worker_starts(self):
        with self.assertRaisesRegex(ValueError, "not a directory"):
            CatalogWorker("/nonexistent/toolsmith/catalog", client=FakeClient())
        directory = self.catalog()
        Path(directory, "empty.py").write_text("VALUE = 1\n")
        expected = "^" + re.escape(f"{Path(directory, 'empty.py')} must define exactly one")
        with self.assertRaisesRegex(ValueError, expected):
            CatalogWorker(directory, client=FakeClient())
        directory = self.catalog("probe")
        Path(directory, "copy.py").write_text(Path(directory, "probe.py").read_text())
        with self.assertRaisesRegex(ValueError, "Duplicate generated tool name 'probe'"):
            CatalogWorker(directory, client=FakeClient())
        directory = self.catalog()
        Path(directory, "broken.py").write_text("import missing_module_for_the_test\n")
        with self.assertRaisesRegex(RuntimeError, "No module named 'missing_module_for_the_test'"):
            CatalogWorker(directory, client=FakeClient())

    def test_configuration_settings_reach_the_worker(self):
        directory = self.catalog("probe")
        config = {"toolsmith": {"catalog": directory, "worker": {"environment": ["GRANTED_TOKEN"], "memory_gib": 2,
                                                                 "write_files": True}},
                  "services": {"client": {"base_url": "http://router.invalid", "timeout_seconds": 42}}}
        worker = CatalogWorker.from_config(config, client=FakeClient(), base_tools={"OCR_Tool", "Depth_Estimation_Tool"})
        self.addCleanup(worker.close)
        self.assertEqual((worker.timeout, worker.limits, worker.refused),
                         (42, {"memory_bytes": 2 << 30, "file_bytes": None}, {"Depth_Estimation_Tool"}))
        del config["toolsmith"]["worker"]
        worker = CatalogWorker.from_config(config, client=FakeClient(), base_tools=set())
        self.addCleanup(worker.close)
        self.assertEqual((worker.timeout, worker.limits), (42, {"memory_bytes": 128 << 30, "file_bytes": 0}))
