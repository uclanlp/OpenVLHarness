"""Generated tools run in a separate, resource-limited worker process, offline."""
import copy
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import unittest.mock
from pathlib import Path
from unittest.mock import patch

import requests

from openvlharness.services.envelope import _build_error_payload, _build_success_payload, _normalize_result_payload
from openvlharness.toolsmith import worker as worker_module
from openvlharness.toolsmith.catalog import load_catalog
from openvlharness.toolsmith.isolation import available
from openvlharness.toolsmith.worker import CatalogWorker

# Workers run confined where this host allows it, as the router runs them; elsewhere they run without isolation.
ISOLATION = available()[0]

TOOL = '''
import os
import time

from openvlharness.services.tools.base import ToolArgumentError
from openvlharness.toolsmith import GeneratedTool, ToolCallError

import sys

# Tells worker processes apart (in a PID namespace each one is PID 1); shared by the modules of one process.
INSTANCE = sys.__dict__.setdefault("toolsmith_test_instance", os.urandom(8).hex())


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
        return {"pid": os.getpid(), "instance": INSTANCE, "cwd": os.getcwd(),
                "env": {name: os.environ.get(name) for name in names}}""",
    "mixed": """        return {"text": "naïve ✓ \\u2028", "float": 0.1 + 0.2, "big": 10 ** 30, "neg_zero": -0.0,
                5: "int key", "set": {3, 1, 2}, "bytes": b"raw", "nested": [(1, 2.5), None, True],
                "nan": float("nan")}""",
    "fails": """        raise ValueError("bad input: " + value)""",
    "bad_args": """        raise ToolArgumentError("value must be a river")""",
    "crash": """        os._exit(3)""",
    "sleepy": """        time.sleep(float(value))
        return {"slept": float(value), "instance": INSTANCE}""",
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
    "fanout": """        import threading
        results = [None] * int(value)

        def call(index):
            try:
                results[index] = ["ok", self.call_tool("Text_Search_Tool", query=str(index))]
            except ToolCallError as exc:
                results[index] = ["refused", str(exc)]
        threads = [threading.Thread(target=call, args=(index,)) for index in range(int(value))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return results""",
    "spinner": """        import re
        return bool(re.match(r"(a+)+$", "a" * 60 + "b"))  # backtracks for ages, holding the GIL""",
    "echo": """        return len(value)""",
    "chatty": """        print("chatty " * 20000, flush=True)
        import warnings
        warnings.warn("chatty warning")
        return "printed\"""",
}


def write_tool(directory, name, body=None):
    cls = "".join(part.title() for part in name.split("_"))
    Path(directory, f"{name}.py").write_text(TOOL.format(cls=cls, name=name, body=body or BODIES[name]))


def _alive(pid):
    """Whether a process exists and is not a zombie."""
    try:
        with open(f"/proc/{pid}/stat") as stream:
            return stream.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


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
        options.setdefault("isolation", ISOLATION)
        worker = CatalogWorker(directory, **options)
        self.addCleanup(worker.close)
        return worker

    def test_the_router_runs_a_bounded_number_of_calls_from_generated_tools_at_once(self):
        class SlowClient(FakeClient):
            def execute(self, tool_name, args, *, tool_call_id):
                time.sleep(2)  # the three calls overlap well within this
                return super().execute(tool_name, args, tool_call_id=tool_call_id)

        worker = self.start(self.catalog("fanout"), client=SlowClient(result="found"))
        with patch.object(worker_module, "MAX_FORWARDED", 2):
            payload, status = worker.execute("fanout", {"value": "3"}, tool_call_id="call-1")
        self.assertEqual(status, 200)
        self.assertEqual(sorted(outcome for outcome, _ in payload["result"]), ["ok", "ok", "refused"])
        self.assertIn(["refused", "Text_Search_Tool: more than 2 calls from generated tools are running"],
                      payload["result"])
        self.assertEqual(worker.current.forwarding, 0)  # every admitted call released its place

    def test_a_tool_list_with_malformed_metadata_is_refused(self):
        directory = self.catalog("probe")
        with patch.object(worker_module, "tool_problem", lambda tool: "input_schema is not a valid JSON Schema"):
            with self.assertRaisesRegex(RuntimeError, "the worker reported no valid tool list"):
                self.start(directory)

    def test_a_call_to_a_worker_that_stopped_reading_still_times_out(self):
        def call(worker, name, value, results):
            started = time.monotonic()
            payload, status = worker.execute(name, {"value": value}, tool_call_id=name)
            results[name] = (status, payload["error"]["message"], time.monotonic() - started)

        # The spinner holds the GIL, so the worker reads nothing; the echo call is larger than the pipe buffer.
        for grace, delay, expected in ((60, 1, "echo did not finish within 3 seconds"),
                                       (0.5, 2, "the generated-tool worker stopped while echo was running")):
            with self.subTest(grace=grace), patch.object(worker_module, "RETIRE_SECONDS", grace):
                worker = self.start(self.catalog("spinner", "echo"), timeout_seconds=3)
                first, results = worker.current, {}
                spinning = threading.Thread(target=call, args=(worker, "spinner", "", results))
                spinning.start()
                time.sleep(delay)
                large = threading.Thread(target=call, args=(worker, "echo", "A" * (2 << 20), results))
                large.start()
                spinning.join(20)
                large.join(20)
                self.assertEqual(results["spinner"][:2], (500, "spinner did not finish within 3 seconds"))
                self.assertEqual(results["echo"][0], 500)
                self.assertTrue(results["echo"][1].startswith(expected), results["echo"][1])
                self.assertLess(results["echo"][2], 3.5)  # its own deadline, or the retired worker's end
                deadline = time.monotonic() + 15
                while first.process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.2)
                self.assertIsNotNone(first.process.poll())  # the retired worker was killed
                self.assertEqual(worker.execute("echo", {"value": "ab"}, tool_call_id="call-9"),
                                 (_build_success_payload("echo", "call-9", 2, {}), 200))

    def test_a_worker_started_by_a_thread_that_ended_keeps_running(self):
        # Linux sends the death signal when the starting thread ends, so workers start from a lasting thread.
        started = []
        thread = threading.Thread(target=lambda: started.append(self.start(self.catalog("echo"))))
        thread.start()
        thread.join()
        time.sleep(1)
        self.assertEqual(started[0].execute("echo", {"value": "abc"}, tool_call_id="call-1")[1], 200)

    def test_workers_die_with_the_router(self):
        script = ("import sys, threading, time\nfrom openvlharness.toolsmith.worker import CatalogWorker\n"
                  "worker = CatalogWorker(sys.argv[1], client=None, isolation=sys.argv[2] == '1', "
                  "report_environment=False)\nprint(worker.current.process.pid, flush=True)\n"
                  "threading.Thread(target=worker.execute, args=('spinner', {}), kwargs={'tool_call_id': 'c'}, "
                  "daemon=True).start()\ntime.sleep(600)\n")
        directory = self.catalog("spinner")
        for confined in sorted({False, ISOLATION}):
            with self.subTest(isolation=confined):
                router = subprocess.Popen([sys.executable, "-c", script, directory, "1" if confined else "0"],
                                          stdout=subprocess.PIPE, text=True,
                                          env={**os.environ, "PYTHONPATH": str(worker_module.PACKAGE_ROOT)})
                child = int(router.stdout.readline())
                time.sleep(2)
                router.kill()
                router.wait()
                deadline = time.monotonic() + 10
                while _alive(child) and time.monotonic() < deadline:
                    time.sleep(0.2)
                self.assertFalse(_alive(child))

    def test_tool_prints_reach_a_router_log_file_without_file_writes(self):
        # The router's stderr is often a log file; the worker, which cannot write files, still prints there.
        directory = self.catalog("chatty")
        log = Path(directory, "router.log")
        log.write_text("router started\n")
        script = ("import json, sys\nfrom openvlharness.toolsmith.worker import CatalogWorker\n"
                  "worker = CatalogWorker(sys.argv[1], client=None, isolation=sys.argv[2] == '1', "
                  "report_environment=False)\n"
                  "print(json.dumps(worker.execute('chatty', {}, tool_call_id='call-1')), file=sys.__stdout__)\n"
                  "worker.close()\n")
        with open(log, "a") as stream:
            done = subprocess.run([sys.executable, "-c", script, directory, "1" if ISOLATION else "0"],
                                  stdout=subprocess.PIPE, stderr=stream, text=True, timeout=120,
                                  env={**os.environ, "PYTHONPATH": str(worker_module.PACKAGE_ROOT)})
        payload, status = json.loads(done.stdout)
        self.assertEqual((payload["result"], status), ("printed", 200), log.read_text()[-2000:])
        self.assertIn("chatty " * 20000, log.read_text())
        self.assertIn("chatty warning", log.read_text())

    def test_what_a_worker_prints_is_bounded_and_without_control_characters(self):
        directory = self.catalog("chatty")
        log = Path(directory, "router.log")
        script = ("import json, sys\nfrom openvlharness.toolsmith import worker as module\n"
                  "module.STDERR_BYTES = 1 << 16\n"
                  "worker = module.CatalogWorker(sys.argv[1], client=None, isolation=sys.argv[2] == '1', "
                  "report_environment=False)\n"
                  "print(json.dumps(worker.execute('chatty', {}, tool_call_id='call-1')), file=sys.__stdout__)\n"
                  "worker.close()\n")
        tool = Path(directory, "chatty.py")
        tool.write_text(tool.read_text().replace('print("chatty " * 20000, flush=True)',
                                                 'print("\\x1b]0;title\\x07\\x1b[2J" + "chatty " * 20000, flush=True)'))
        with open(log, "w") as stream:
            done = subprocess.run([sys.executable, "-c", script, directory, "1" if ISOLATION else "0"],
                                  stdout=subprocess.PIPE, stderr=stream, text=True, timeout=120,
                                  env={**os.environ, "PYTHONPATH": str(worker_module.PACKAGE_ROOT)})
        self.assertEqual(json.loads(done.stdout)[0]["result"], "printed")
        text = log.read_text()
        self.assertNotIn("\x1b", text)
        self.assertIn("]0;title[2Jchatty chatty", text)
        self.assertIn("output past 65536 bytes dropped]", text)  # the limit this test sets
        self.assertLess(len(text.encode()), (1 << 16) + 4096)

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

    def test_relative_path_settings_resolve_against_the_router_directory(self):
        router = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(router, ignore_errors=True))
        Path(router, "keys").mkdir()
        Path(router, "keys", "serper.txt").write_text("a key")
        Path(router, "legacy").mkdir()
        Path(router, "legacy", "legacy_helper.py").write_text("VALUE = 'imported from a relative PYTHONPATH'\n")
        directory = self.catalog()
        write_tool(directory, "paths", '''        import legacy_helper
        names = ("SERPER_API_KEY_FILE", "IMAGE_UPLOAD_LOCAL_DIR", "HF_HOME")
        return {"env": {name: os.environ.get(name) for name in names}, "helper": legacy_helper.VALUE,
                "key": open(os.environ["SERPER_API_KEY_FILE"]).read()}''')
        here = os.getcwd()
        os.chdir(router)
        self.addCleanup(os.chdir, here)
        settings = {"SERPER_API_KEY_FILE": "keys/serper.txt", "IMAGE_UPLOAD_LOCAL_DIR": "uploads",
                    "HF_HOME": "hf_home", "PYTHONPATH": "legacy"}
        with patch.dict(os.environ, settings):
            worker = self.start(directory, environment=["SERPER_API_KEY_FILE"])
        os.chdir(here)
        result = worker.execute("paths", {}, tool_call_id="call-1")[0]["result"]
        self.assertEqual(result, {"env": {"SERPER_API_KEY_FILE": os.path.join(router, "keys", "serper.txt"),
                                          "IMAGE_UPLOAD_LOCAL_DIR": os.path.join(router, "uploads"),
                                          "HF_HOME": os.path.join(router, "hf_home")},
                                  "helper": "imported from a relative PYTHONPATH", "key": "a key"})

    def test_file_writes_and_memory_are_limited(self):
        directory = self.catalog("writer", "hog", "probe")
        worker = self.start(directory, memory_gib=1)
        payload, status = worker.execute("writer", {}, tool_call_id="call-1")
        self.assertEqual(status, 500)
        self.assertIn("File too large", payload["error"]["message"])
        instance = worker.execute("probe", {}, tool_call_id="call-2")[0]["result"]["instance"]
        payload, _ = worker.execute("hog", {"value": str(2 << 30)}, tool_call_id="call-3")
        self.assertEqual(payload["error"]["detail"]["exception_type"], "MemoryError")
        self.assertEqual(worker.execute("hog", {"value": "1024"}, tool_call_id="call-4")[0]["result"], 1024)
        self.assertEqual(worker.execute("probe", {}, tool_call_id="call-5")[0]["result"]["instance"], instance)
        writable = self.start(directory, write_files=True)
        self.assertEqual(writable.execute("writer", {}, tool_call_id="call-6")[0]["result"], "wrote")

    def test_a_crash_returns_an_error_and_later_calls_get_a_new_worker(self):
        worker = self.start(self.catalog("crash", "probe"))
        before = worker.execute("probe", {}, tool_call_id="call-1")[0]["result"]["instance"]
        payload, status = worker.execute("crash", {}, tool_call_id="call-2")
        self.assertEqual((status, payload["ok"], payload["tool_call_id"]), (500, False, "call-2"))
        self.assertIn("exited with status 3", payload["error"]["message"])
        payload, status = worker.execute("probe", {}, tool_call_id="call-3")
        self.assertEqual(status, 200)
        self.assertNotEqual(payload["result"]["instance"], before)

    def test_a_call_past_the_time_limit_fails_while_other_calls_finish(self):
        worker = self.start(self.catalog("sleepy", "probe"), timeout_seconds=3)
        old = worker.execute("probe", {}, tool_call_id="call-0")[0]["result"]["instance"]
        process = worker.current.process
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
        self.assertEqual((status, payload["result"]["slept"], payload["result"]["instance"]), (200, 2.25, old))
        payload, status = worker.execute("probe", {}, tool_call_id="call-3")
        self.assertEqual(status, 200)
        self.assertNotEqual(payload["result"]["instance"], old)
        deadline = time.monotonic() + 10
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        self.assertIsNotNone(process.poll(), "the worker with the hung call was not stopped")

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
        before = worker.execute("probe", {}, tool_call_id="call-1")[0]["result"]["instance"]
        with patch.object(worker_module, "MAX_MESSAGE_BYTES", 4096):
            worker._retire(worker.current)
            small = worker.execute("probe", {}, tool_call_id="call-2")[0]["result"]["instance"]
        payload, status = worker.execute("talker", {"value": "100000"}, tool_call_id="call-3")
        self.assertEqual(status, 500)
        self.assertIn("invalid message", payload["error"]["message"])
        payload, status = worker.execute("probe", {}, tool_call_id="call-4")
        self.assertEqual(status, 200)
        self.assertNotIn(payload["result"]["instance"], (before, small))

    def test_an_oversized_result_fails_only_its_call(self):
        worker = self.start(self.catalog("talker", "sleepy"))
        first, results = worker.current, {}

        def call(name, value):
            results[name] = worker.execute(name, {"value": value}, tool_call_id=name)
        threads = [threading.Thread(target=call, args=("sleepy", "2")),
                   threading.Thread(target=call, args=("talker", str(worker_module.MAX_MESSAGE_BYTES + 1)))]
        for thread in threads:
            thread.start()
            time.sleep(0.5)
        for thread in threads:
            thread.join(60)
        payload, status = results["talker"]
        self.assertEqual((status, payload["error"]["message"]),
                         (500, "the result of talker is larger than the 32 MiB limit"))
        self.assertEqual((results["sleepy"][1], results["sleepy"][0]["result"]["slept"]), (200, 2.0))
        self.assertIs(worker.current, first)  # the same worker goes on

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
            CatalogWorker("/nonexistent/toolsmith/catalog", client=FakeClient(), isolation=ISOLATION)
        directory = self.catalog()
        Path(directory, "empty.py").write_text("VALUE = 1\n")
        expected = "^" + re.escape(f"{Path(directory, 'empty.py')} must define exactly one")
        with self.assertRaisesRegex(ValueError, expected):
            CatalogWorker(directory, client=FakeClient(), isolation=ISOLATION)
        directory = self.catalog("probe")
        Path(directory, "copy.py").write_text(Path(directory, "probe.py").read_text())
        with self.assertRaisesRegex(ValueError, "Duplicate generated tool name 'probe'"):
            CatalogWorker(directory, client=FakeClient(), isolation=ISOLATION)
        directory = self.catalog()
        Path(directory, "broken.py").write_text("import missing_module_for_the_test\n")
        with self.assertRaisesRegex(RuntimeError, "No module named 'missing_module_for_the_test'"):
            CatalogWorker(directory, client=FakeClient(), isolation=ISOLATION)

    def test_platforms_without_address_space_limits_run_the_worker_without_one(self):
        with patch.object(worker_module.sys, "platform", "darwin"), patch.object(worker_module.resource,
                                                                                "setrlimit") as setrlimit:
            worker_module.limit_resources(memory_bytes=2 << 30, file_bytes=0)
            self.assertEqual(setrlimit.call_args_list, [unittest.mock.call(worker_module.resource.RLIMIT_FSIZE, (0, 0))])
            with self.assertLogs("openvlharness.toolsmith.worker", "WARNING") as logs:
                worker = self.start(self.catalog("probe"), isolation=False)
        self.assertIsNone(worker.limits["memory_bytes"])
        self.assertTrue(any("does not enforce address-space limits" in line for line in logs.output))
        self.assertEqual(worker.execute("probe", {}, tool_call_id="call-1")[1], 200)

    def test_caches_follow_xdg_cache_home(self):
        with tempfile.TemporaryDirectory() as xdg:
            for name in ("huggingface", "torch"):
                Path(xdg, name).mkdir()
            Path(xdg, "huggingface", "token").write_text("hf_secret")
            clean = {name: value for name, value in os.environ.items()
                     if name not in {"HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
                                     "TORCH_HOME"}}
            with patch.dict(os.environ, {**clean, "XDG_CACHE_HOME": xdg}, clear=True):
                worker = self.start(self.catalog("probe"), write_files=True)
                spec = worker.confinement(Path(tempfile.mkdtemp()))
            self.assertEqual(spec["overlay"], [os.path.join(xdg, "huggingface"), os.path.join(xdg, "torch")])
            self.assertIn(os.path.join(xdg, "huggingface", "token"), spec["mask"])

    def test_configuration_settings_reach_the_worker(self):
        directory = self.catalog("probe")
        config = {"toolsmith": {"catalog": directory, "worker": {"environment": ["GRANTED_TOKEN"], "memory_gib": 2,
                                                                 "write_files": True, "isolation": ISOLATION}},
                  "services": {"client": {"base_url": "http://router.invalid", "timeout_seconds": 42}}}
        worker = CatalogWorker.from_config(config, client=FakeClient(), base_tools={"OCR_Tool", "Depth_Estimation_Tool"})
        self.addCleanup(worker.close)
        parent = None if ISOLATION else os.getpid()  # an unconfined worker watches the router itself
        self.assertEqual((worker.timeout, worker.limits, worker.refused),
                         (37.8, {"memory_bytes": 2 << 30, "file_bytes": None, "parent": parent},
                          {"Depth_Estimation_Tool"}))
        config["toolsmith"]["worker"] = {"isolation": ISOLATION, "timeout_seconds": 42}
        with self.assertLogs("openvlharness.toolsmith.worker", "WARNING") as logs:
            worker = CatalogWorker.from_config(config, client=FakeClient(), base_tools=set())
        self.addCleanup(worker.close)
        self.assertEqual((worker.timeout, worker.limits),
                         (42, {"memory_bytes": 128 << 30, "file_bytes": 0, "parent": parent}))
        self.assertTrue(any("may time out and retry" in line for line in logs.output))
