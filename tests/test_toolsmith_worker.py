"""Generated tools run in a separate, resource-limited worker process, offline."""
import copy
import ctypes
import json
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests

from openvlharness.services.envelope import _build_error_payload, _build_success_payload, _normalize_result_payload
from openvlharness.toolsmith import worker as worker_module
from openvlharness.toolsmith.catalog import load_catalog
from openvlharness.toolsmith.errors import CatalogLoadError
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
        names = ("SERVICE_API_KEY", "GRANTED_TOKEN", "TOOL_OPTION", "PLAIN_SETTING")
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
    # Three children, each holding a share of the per-process cap; with 0.6 they hold 1.8 times it together.
    "forker": """        import resource
        size = int(resource.getrlimit(resource.RLIMIT_AS)[0] * float(value))
        children = []
        for _ in range(3):
            read, write = os.pipe()
            pid = os.fork()
            if pid == 0:
                block = bytearray(size)
                for offset in range(0, size, 4096):
                    block[offset] = 1
                os.write(write, b"x")
                time.sleep(60)
                os._exit(0)
            os.close(write)
            os.read(read, 1)
            children.append(pid)
        time.sleep(2)  # all three hold their memory for four of the watchdog's samples
        for pid in children:
            os.kill(pid, 9)
            os.waitpid(pid, 0)
        return "held\"""",
    "burner": """        import threading

        def spin():
            while True:
                pass
        threading.Thread(target=spin, daemon=True).start()  # left running after the call returns
        return "started\"""",
    "shouter": """        sys.stderr.write("q" * 102400)
        sys.stderr.flush()
        os._exit(3)""",
    "flooder": """        import gc
        channel = next(item for item in gc.get_objects() if type(item).__name__ == "_Channel")
        channel.send({"type": "result", "id": "q" * (1 << 20), "ok": True, "result": None, "meta": {}})
        time.sleep(5)
        return "sent\"""",
    "napper": """        time.sleep(float(value))
        return "rested\"""",
    "sneaky": """        import threading

        def spin():
            while True:
                pass

        def keep_busy():  # a call to another generated tool always in flight
            while True:
                try:
                    self.call_tool("napper", value="1")
                except Exception:
                    time.sleep(0.1)
        threading.Thread(target=spin, daemon=True).start()
        threading.Thread(target=keep_busy, daemon=True).start()
        return "started\"""",
    "filler": """        for index in range(int(value)):  # files no process maps, in the sandbox's tmpfs
            with open(f"/tmp/fill-{index}", "wb") as stream:
                stream.write(b"\\1" * (200 << 20))
        time.sleep(5)
        return "filled\"""",
    "memfd_writer": """        fd = os.memfd_create("held")
        chunk, written = b"\\1" * (16 << 20), 0
        try:
            while written < int(value):
                written += os.write(fd, chunk)
        except OSError as exc:
            return f"{type(exc).__name__}: {exc.strerror}"
        finally:
            os.close(fd)
        return "written\"""",
    # A file in the sandbox's /tmp that the tool maps (value MiB), and a child holding more (extra MiB).
    "mapper": """        import mmap
        size, extra = (int(part) << 20 for part in value.split(":"))
        with open("/tmp/mapped.bin", "wb") as stream:
            for _ in range(size >> 24):
                stream.write(b"\\1" * (16 << 20))
        with open("/tmp/mapped.bin", "rb") as stream:
            view = mmap.mmap(stream.fileno(), 0, prot=mmap.PROT_READ)
        touched = sum(view[offset] for offset in range(0, size, 4096))  # every page is mapped
        child = 0
        if extra:
            read, write = os.pipe()
            child = os.fork()
            if child == 0:
                view.close()  # the child's own address space holds only what it allocates
                block = bytearray(extra)
                for offset in range(0, extra, 4096):
                    block[offset] = 1
                os.write(write, b"x")
                time.sleep(60)
                os._exit(0)
            os.close(write)
            os.read(read, 1)
        time.sleep(3)
        if child:
            os.kill(child, 9)
            os.waitpid(child, 0)
        view.close()
        os.unlink("/tmp/mapped.bin")
        return f"mapped {touched}\"""",
    # Holds value GiB and runs a short command over and over: each is vforked, sharing that memory until it execs.
    "spawner": """        import subprocess
        block = bytearray(int(float(value) * (1 << 30)))
        for offset in range(0, len(block), 4096):
            block[offset] = 1
        start, runs = time.monotonic(), 0
        while time.monotonic() - start < 4:
            subprocess.run(["true"], check=True)
            runs += 1
        return runs""",
}


def write_tool(directory, name, body=None):
    cls = "".join(part.title() for part in name.split("_"))
    Path(directory, f"{name}.py").write_text(TOOL.format(cls=cls, name=name, body=body or BODIES[name]))


def _alive(pid):
    """Whether a process exists and has not died.

    A dead process shows Z (zombie) until it is reaped, then X (dead) for the moment its
    parent reaps it; neither runs. Counting X as alive failed test_workers_die_with_the_router
    about once in 60 runs, for a worker that had died at once.
    """
    try:
        with open(f"/proc/{pid}/stat") as stream:
            return stream.read().rsplit(")", 1)[1].split()[0] not in ("Z", "X", "x")
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
    """The envelope the router would build running the tool in its own process."""
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
        worker = self.start(self.catalog("spinner", "echo"), timeout_seconds=3)
        first, results = worker.current, {}

        def call(name, value):
            started = time.monotonic()
            payload, status = worker.execute(name, {"value": value}, tool_call_id=name)
            results[name] = (status, payload["error"]["message"], time.monotonic() - started)
        # The spinner holds the GIL, so the worker reads nothing; the echo call is larger than the pipe buffer.
        spinning = threading.Thread(target=call, args=("spinner", ""))
        spinning.start()
        time.sleep(1)
        large = threading.Thread(target=call, args=("echo", "A" * (2 << 20)))
        large.start()
        spinning.join(20)
        large.join(20)
        self.assertEqual({name: outcome[:2] for name, outcome in results.items()},
                         {"spinner": (500, "spinner did not finish within 3 seconds"),
                          "echo": (500, "echo did not finish within 3 seconds")})
        self.assertLess(results["echo"][2], 4.5)  # its own deadline, not a blocked send
        deadline = time.monotonic() + 10
        while first.process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.2)
        self.assertIsNotNone(first.process.poll())  # retired and, with no call left, killed
        self.assertEqual(worker.execute("echo", {"value": "ab"}, tool_call_id="call-9"),
                         (_build_success_payload("echo", "call-9", 2, {}), 200))

    def test_a_timeout_does_not_cut_short_the_other_calls_of_its_worker(self):
        # B times out and retires the worker; A, started later, still finishes within its own limit.
        worker = self.start(self.catalog("sleepy"), timeout_seconds=3)
        first, results = worker.current, {}

        def call(name, seconds):
            results[name] = worker.execute("sleepy", {"value": str(seconds)}, tool_call_id=name)
        with patch.object(worker_module, "RETIRE_GRACE_SECONDS", 0.5):
            late = threading.Thread(target=call, args=("B", 30))
            late.start()
            time.sleep(2)
            other = threading.Thread(target=call, args=("A", 2.5))  # finishes about 1.5 s after B's timeout
            other.start()
            late.join(20)
            other.join(20)
            self.assertEqual(results["B"][1], 500)
            self.assertEqual((results["A"][1], results["A"][0]["result"]["slept"]), (200, 2.5))
            deadline = time.monotonic() + 10
            while first.process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.2)
        self.assertIsNotNone(first.process.poll())  # stopped once A had finished
        self.assertIsNot(worker.current, first)

    def test_a_worker_started_by_a_thread_that_ended_keeps_running(self):
        # Linux sends the death signal when the starting thread ends, so workers start from a lasting thread.
        started = []
        thread = threading.Thread(target=lambda: started.append(self.start(self.catalog("echo"))))
        thread.start()
        thread.join()
        time.sleep(1)
        self.assertEqual(started[0].execute("echo", {"value": "abc"}, tool_call_id="call-1")[1], 200)

    def test_workers_die_with_the_router(self):
        # A killed router cannot remove its worker's private directory or its outbound proxy's socket directory:
        # here both are made in one the test removes.
        script = ("import sys, tempfile, threading, time\nfrom openvlharness.toolsmith import proxy\n"
                  "from openvlharness.toolsmith.worker import CatalogWorker\n"
                  "tempfile.tempdir = proxy.SOCKET_BASE = sys.argv[3]\n"
                  "worker = CatalogWorker(sys.argv[1], client=None, isolation=sys.argv[2] == '1', "
                  "report_environment=False)\nprint(worker.current.process.pid, flush=True)\n"
                  "threading.Thread(target=worker.execute, args=('spinner', {}), kwargs={'tool_call_id': 'c'}, "
                  "daemon=True).start()\ntime.sleep(600)\n")
        directory = self.catalog("spinner")
        leftovers = tempfile.TemporaryDirectory(dir="/tmp")  # short enough for the socket's path
        self.addCleanup(leftovers.cleanup)
        for confined in sorted({False, ISOLATION}):
            with self.subTest(isolation=confined):
                router = subprocess.Popen([sys.executable, "-c", script, directory, "1" if confined else "0",
                                           leftovers.name], stdout=subprocess.PIPE, text=True,
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
        settings = {"SERVICE_API_KEY": "secret", "GRANTED_TOKEN": "granted", "TOOL_OPTION": "1",
                    "PLAIN_SETTING": "kept"}
        with patch.dict(os.environ, settings):
            environment = self.start(directory, environment=["GRANTED_TOKEN"]).execute(
                "probe", {}, tool_call_id="call-1")[0]["result"]["env"]
        self.assertEqual(environment, {"SERVICE_API_KEY": None, "GRANTED_TOKEN": "granted",
                                       "TOOL_OPTION": "1", "PLAIN_SETTING": "kept"})

    def test_relative_path_settings_resolve_against_the_router_directory(self):
        router = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(router, ignore_errors=True))
        Path(router, "keys").mkdir()
        Path(router, "keys", "serper.txt").write_text("a key")
        Path(router, "helpers").mkdir()
        Path(router, "helpers", "shared_helper.py").write_text("VALUE = 'imported from a relative PYTHONPATH'\n")
        directory = self.catalog()
        write_tool(directory, "paths", '''        import shared_helper
        names = ("SERPER_API_KEY_FILE", "IMAGE_UPLOAD_LOCAL_DIR", "HF_HOME")
        return {"env": {name: os.environ.get(name) for name in names}, "helper": shared_helper.VALUE,
                "key": open(os.environ["SERPER_API_KEY_FILE"]).read()}''')
        here = os.getcwd()
        os.chdir(router)
        self.addCleanup(os.chdir, here)
        settings = {"SERPER_API_KEY_FILE": "keys/serper.txt", "IMAGE_UPLOAD_LOCAL_DIR": "uploads",
                    "HF_HOME": "hf_home", "PYTHONPATH": "helpers"}
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

    def test_a_message_over_the_routers_limit_stops_only_that_worker(self):
        # The router's own check, for a worker that does not keep to the limit it was given.
        worker = self.start(self.catalog("talker", "probe"))
        before = worker.execute("probe", {}, tool_call_id="call-1")[0]["result"]["instance"]
        worker.current.channel.limit = 4096  # from the read after the one already waiting
        self.assertEqual(worker.execute("probe", {}, tool_call_id="call-2")[1], 200)
        payload, status = worker.execute("talker", {"value": "100000"}, tool_call_id="call-3")
        self.assertEqual(status, 500)
        self.assertIn("invalid message", payload["error"]["message"])
        payload, status = worker.execute("probe", {}, tool_call_id="call-4")
        self.assertEqual(status, 200)
        self.assertNotEqual(payload["result"]["instance"], before)

    def test_an_oversized_result_fails_only_its_call(self):
        with patch.object(worker_module, "MAX_MESSAGE_BYTES", 1 << 20):  # the router passes its limit on
            worker = self.start(self.catalog("talker", "sleepy"))
        first, results = worker.current, {}

        def call(name, value):
            results[name] = worker.execute(name, {"value": value}, tool_call_id=name)
        threads = [threading.Thread(target=call, args=("sleepy", "2")),
                   threading.Thread(target=call, args=("talker", str(2 << 20)))]
        for thread in threads:
            thread.start()
            time.sleep(0.5)
        for thread in threads:
            thread.join(60)
        payload, status = results["talker"]
        self.assertEqual((status, payload["error"]["message"]),
                         (500, "the result of talker is larger than the 1 MiB limit"))
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
        odd = ("class Odd(Exception):\n    {}\n"
               "raise Odd('the real reason: missing_dependency')\n")
        for source, error in (("import missing_module_for_the_test\n",
                               "ModuleNotFoundError: No module named 'missing_module_for_the_test'"),
                              ("class Broken(GeneratedTool)\n    pass\n", "SyntaxError: expected ':' (broken.py, line 1)"),
                              ("import sys\nsys.exit(3)\n", "SystemExit: 3"),
                              # Exceptions that refuse attributes or break when asked for one keep their reason.
                              (odd.format("def __setattr__(self, name, value):\n        raise RuntimeError('no')"),
                               "Odd: the real reason: missing_dependency"),
                              (odd.format("catalog_module = property(lambda self: 'x')"),
                               "Odd: the real reason: missing_dependency"),
                              (odd.format("def __getattribute__(self, name):\n"
                                          "        if name == 'catalog_module':\n            raise KeyError(name)\n"
                                          "        return super().__getattribute__(name)"),
                               "Odd: the real reason: missing_dependency"),
                              ("class Unprintable(Exception):\n    def __str__(self):\n        raise RuntimeError\n"
                               "raise Unprintable()\n", "Unprintable: <unprintable message>")):
            directory = self.catalog()
            Path(directory, "broken.py").write_text(source)
            with self.subTest(error=error), self.assertRaisesRegex(CatalogLoadError, "^" + re.escape(
                    f"the ToolSmith catalog {directory} cannot load broken.py: {error}")):
                CatalogWorker(directory, client=FakeClient(), isolation=ISOLATION)

    def test_a_tool_list_check_that_raises_stops_the_worker_it_started(self):
        created = []
        original = worker_module._Worker

        class Recording(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                created.append(self)
        with patch.object(worker_module, "_Worker", Recording), \
                patch.object(worker_module, "tool_problem", side_effect=ValueError("Invalid IPv6 URL")), \
                self.assertRaisesRegex(ValueError, "Invalid IPv6 URL"):
            CatalogWorker(self.catalog("echo"), client=FakeClient(), isolation=ISOLATION)
        self.assertEqual(len(created), 1)
        self.assertIsNotNone(created[0].process.poll())  # discarded, not left running

    def test_a_closed_worker_router_is_not_kept_alive_by_its_exit_hook(self):
        import gc
        import weakref

        router = CatalogWorker(self.catalog("echo"), client=FakeClient(), isolation=ISOLATION)
        router.close()
        reference = weakref.ref(router)
        del router
        gc.collect()
        self.assertIsNone(reference())

    def test_metadata_is_read_without_network_and_hugging_face_goes_offline_without_one(self):
        directory = self.catalog()
        Path(directory, "seen.py").write_text(
            "import os\n\nfrom openvlharness.toolsmith import GeneratedTool\n\n\nclass Seen(GeneratedTool):\n"
            "    def __init__(self):\n        names = ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')\n"
            "        super().__init__(tool_name='seen', input_schema={'type': 'object'},\n"
            "                         tool_description=' '.join(str(os.environ.get(name)) for name in names))\n\n"
            "    def execute(self):\n        return self.tool_description\n")
        built = []

        class Recording(CatalogWorker):
            def __init__(self, *args, **kwargs):
                built.append(kwargs["network"])
                super().__init__(*args, **kwargs)
        from openvlharness.toolsmith import proxy

        with patch.object(worker_module, "CatalogWorker", Recording), \
                patch.object(proxy, "OutboundProxy", side_effect=AssertionError("a metadata worker started a proxy")):
            tools = worker_module.catalog_tools(directory, settings={"isolation": ISOLATION, "network": True})
        self.assertEqual(built, [False])  # the agent's and the pipeline's reads of tool descriptions, with no proxy
        # Only a confined worker is really without network; then cached models load without retrying the Hub.
        offline = "1 1" if ISOLATION else "None None"
        self.assertEqual(tools["seen"].tool_description, offline)
        self.assertEqual(self.start(directory, network=False).execute("seen", {}, tool_call_id="call-1")[0]["result"],
                         offline)
        self.assertEqual(self.start(directory, network=True).execute("seen", {}, tool_call_id="call-2")[0]["result"],
                         "None None")

    def test_a_schema_in_a_read_only_property_is_listed_as_json_carries_it(self):
        directory = self.catalog()
        Path(directory, "read_only.py").write_text(
            "from openvlharness.toolsmith import GeneratedTool\n\n\nclass ReadOnly(GeneratedTool):\n"
            "    input_schema = property(lambda self: {'type': 'object', 'required': ('word',)})\n\n"
            "    def __init__(self):\n        self.tool_name, self.tool_description = 'read_only', 'Echo a word.'\n\n"
            "    def execute(self, word=''):\n        return word\n")
        tools = worker_module.catalog_tools(directory, settings={"isolation": ISOLATION})
        self.assertEqual(tools["read_only"].input_schema["required"], ["word"])

    @pytest.mark.skipif(not ISOLATION, reason="only a confined worker has an outbound proxy")
    def test_an_outbound_proxy_that_cannot_start_is_a_worker_start_error(self):
        from openvlharness.toolsmith import proxy
        from openvlharness.toolsmith.errors import WorkerStartError

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(proxy, "SOCKET_BASE", os.path.join(directory, "missing")), \
                self.assertRaisesRegex(WorkerStartError, r"^the ToolSmith outbound proxy cannot start: .*No such file"):
            CatalogWorker(self.catalog("echo"), client=FakeClient(), isolation=True, network=True)

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

    def test_a_catalog_that_fails_to_load_leaves_no_private_directory(self):
        directory = self.catalog("probe")
        Path(directory, "probe_copy.py").write_text(Path(directory, "probe.py").read_text())  # a duplicate name
        temporary = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(temporary, ignore_errors=True))
        with patch.object(tempfile, "tempdir", temporary), self.assertRaisesRegex(ValueError, "Duplicate"):
            self.start(directory)
        self.assertEqual([name for name in os.listdir(temporary) if name.startswith(worker_module.HOME_PREFIX)], [])

    def test_the_memory_limit_holds_for_all_the_processes_of_a_worker(self):
        # Each child stays under the per-process address-space cap; together they pass memory_gib.
        worker = self.start(self.catalog("forker", "echo"), memory_gib=1)
        self.assertEqual(worker.execute("forker", {"value": "0.1"}, tool_call_id="call-1")[0]["result"], "held")
        first, started = worker.current, time.monotonic()
        with self.assertLogs("openvlharness.toolsmith.worker", "WARNING") as logs:
            payload, status = worker.execute("forker", {"value": "0.6"}, tool_call_id="call-2")
        self.assertLess(time.monotonic() - started, 20)
        self.assertEqual((status, payload["error"]["detail"]["exception_type"]), (500, "MemoryLimitExceeded"), payload)
        self.assertRegex(payload["error"]["message"],
                         r"^the generated-tool worker stopped while forker was running: its processes and in-memory "
                         r"files used [\d.]+ GiB of memory together, more than toolsmith\.worker\.memory_gib allows \(1\.00 GiB\)$")
        self.assertTrue(any("stopping it" in line for line in logs.output))
        self.assertEqual(worker.execute("echo", {"value": "ab"}, tool_call_id="call-3"),
                         (_build_success_payload("echo", "call-3", 2, {}), 200))
        self.assertIsNot(worker.current, first)
        self.assertIsNotNone(first.process.poll())

    def test_a_protocol_error_repeats_little_of_what_the_worker_sent(self):
        worker = self.start(self.catalog("flooder", "echo"))
        with self.assertLogs("openvlharness.toolsmith.worker", "ERROR") as logs:
            payload, status = worker.execute("flooder", {}, tool_call_id="call-1")
        self.assertEqual(status, 500)
        self.assertLess(len(payload["error"]["message"]), 600, payload["error"]["message"][:300])
        self.assertIn("more characters", payload["error"]["message"])
        self.assertLess(max(len(line) for line in logs.output), 600)
        self.assertEqual(worker.execute("echo", {"value": "ab"}, tool_call_id="call-2")[1], 200)

    def test_the_idle_cpu_rule_does_not_stop_a_worker_whose_call_came_in_meanwhile(self):
        original, deciding = worker_module._Worker._exceed, threading.Event()

        def delayed(self, reason, kind):  # the router's other threads run between the check and the kill
            deciding.set()
            time.sleep(1)
            return original(self, reason, kind)
        with patch.multiple(worker_module, IDLE_CPU_WINDOW_SECONDS=2, IDLE_CPU_SECONDS=0.5, IDLE_SETTLE_SECONDS=0.5), \
                patch.object(worker_module._Worker, "_exceed", delayed):
            worker = self.start(self.catalog("burner", "sleepy"))
            self.assertEqual(worker.execute("burner", {}, tool_call_id="call-1")[1], 200)
            self.assertTrue(deciding.wait(15))
            payload, status = worker.execute("sleepy", {"value": "2"}, tool_call_id="call-2")
        self.assertEqual(status, 200, payload)

    def test_calls_a_tool_makes_to_generated_tools_do_not_keep_its_worker_busy(self):
        class Loopback(FakeClient):  # the router runs a call_tool to a generated tool on the same CatalogWorker
            worker = None

            def execute(self, tool_name, args, *, tool_call_id):
                return self.worker.execute(tool_name, args, tool_call_id=tool_call_id)[0]
        client = Loopback()
        with patch.multiple(worker_module, IDLE_CPU_WINDOW_SECONDS=2, IDLE_CPU_SECONDS=0.5, IDLE_SETTLE_SECONDS=0.5):
            worker = self.start(self.catalog("sneaky", "napper"), client=client)
            client.worker = worker
            first = worker.current
            with self.assertLogs("openvlharness.toolsmith.worker", "WARNING") as logs:
                self.assertEqual(worker.execute("sneaky", {}, tool_call_id="call-1")[1], 200)
                returned = time.monotonic()
                while first.process.poll() is None and time.monotonic() - returned < 20:
                    time.sleep(0.1)
        self.assertIsNotNone(first.process.poll())
        self.assertTrue(any("with no call in flight" in line for line in logs.output), logs.output)

    def test_the_watchdog_samples_a_slow_tree_less_often(self):
        samples = []
        real = worker_module.tree_usage

        def slow(root, parent, **options):
            samples.append(time.monotonic())
            time.sleep(0.2)  # as a tree of thousands of processes takes
            return real(root, parent, **options)
        with patch.object(worker_module, "tree_usage", slow):
            worker = self.start(self.catalog("echo"))
            time.sleep(5)
            worker.close()
        gaps = [later - earlier for earlier, later in zip(samples, samples[1:])]
        self.assertGreater(min(gaps[1:]), 0.2 * worker_module.WATCH_BACKOFF)

    def test_the_watchdogs_wait_after_a_slow_sample_is_capped(self):
        # A sample reported as 100 seconds long (a tree of any size) must not leave the worker unmeasured for 1000.
        waits, clock = [], iter(range(0, 10000, 100))

        def wait(seconds):
            waits.append(seconds)
            return len(waits) > 3
        worker = SimpleNamespace(ended=SimpleNamespace(wait=wait), meter=SimpleNamespace(measure=lambda: (0, 0.0)),
                                 owner=SimpleNamespace(memory_limit=None), lock=threading.Lock(), started=0,
                                 ready=SimpleNamespace(done=lambda: False))
        with patch.object(worker_module, "time", SimpleNamespace(monotonic=lambda: next(clock))):
            worker_module._Worker._watch(worker)
        self.assertEqual(waits, [worker_module.WATCH_SECONDS] + [worker_module.WATCH_MAX_SECONDS] * 3)
        self.assertLessEqual(worker_module.WATCH_MAX_SECONDS, 5)

    @pytest.mark.skipif(not ISOLATION, reason="the sandbox's own file systems exist only with isolation")
    def test_files_in_the_sandboxs_memory_backed_file_systems_count_toward_the_memory_limit(self):
        # Three files of 200 MiB in /tmp: no process maps them, and each is under the file-size cap.
        worker = self.start(self.catalog("filler", "echo"), memory_gib=0.5, write_files=True)
        with self.assertLogs("openvlharness.toolsmith.worker", "WARNING"):
            payload, status = worker.execute("filler", {"value": "3"}, tool_call_id="call-1")
        self.assertEqual((status, payload["error"]["detail"]["exception_type"]), (500, "MemoryLimitExceeded"), payload)
        self.assertEqual(worker.execute("echo", {"value": "ab"}, tool_call_id="call-2")[1], 200)

    @pytest.mark.skipif(not ISOLATION, reason="the sandbox's own file systems exist only with isolation")
    def test_a_tmpfs_file_a_tool_maps_counts_once(self):
        # 240 MiB in the sandbox's /tmp, mapped: in the file system's use and in the tool's Pss, counted once.
        worker = self.start(self.catalog("mapper"), memory_gib=0.5, write_files=True)
        payload, status = worker.execute("mapper", {"value": "240:0"}, tool_call_id="call-1")
        self.assertEqual((status, payload["result"]), (200, "mapped 61440"), payload)
        # It still counts: with a child holding 300 MiB more, the worker is past 0.5 GiB and stopped.
        with self.assertLogs("openvlharness.toolsmith.worker", "WARNING"):
            payload, status = worker.execute("mapper", {"value": "240:300"}, tool_call_id="call-2")
        self.assertEqual((status, payload["error"]["detail"]["exception_type"]), (500, "MemoryLimitExceeded"), payload)

    def test_a_tool_that_runs_commands_while_holding_most_of_its_memory_is_not_stopped(self):
        # Each command is vforked: until it execs, the child uses the tool's memory, which counts once.
        worker = self.start(self.catalog("spawner"), memory_gib=0.5)
        payload, status = worker.execute("spawner", {"value": "0.28"}, tool_call_id="call-1")
        self.assertEqual(status, 200, payload)
        self.assertGreater(payload["result"], 100)

    @pytest.mark.skipif(platform.machine() not in ("x86_64", "aarch64") or not os.path.isdir("/proc/self/task"),
                        reason="clone() with a C callback, as glibc on Linux has it")
    def test_a_child_that_shares_its_parents_memory_counts_once(self):
        # A thread waits in clone(CLONE_VM | CLONE_VFORK) while the child sleeps in this process's memory, as
        # vfork() and posix_spawn() leave a child until it execs.
        libc = ctypes.CDLL(None, use_errno=True)
        libc.clone.restype = ctypes.c_int
        libc.clone.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        stack = ctypes.create_string_buffer(1 << 16)
        me, before = os.getpid(), set(worker_module._child_pids(os.getpid()))
        alone = worker_module.tree_usage(me, os.getppid())[0]
        started = {}

        def spawn():
            started["pid"] = libc.clone(ctypes.cast(libc.sleep, ctypes.c_void_p), ctypes.addressof(stack) + len(stack)
                                        - 256, 0x100 | 0x4000 | signal.SIGCHLD, 3)  # CLONE_VM | CLONE_VFORK
        thread = threading.Thread(target=spawn, daemon=True)
        thread.start()
        deadline = time.monotonic() + 2
        while not set(worker_module._child_pids(me)) - before and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(set(worker_module._child_pids(me)) - before, "the child did not start")
        measured = {"kcmp": worker_module.tree_usage(me, os.getppid())[0]}
        with patch.object(worker_module, "_same_memory", return_value=None):  # where kcmp is refused
            measured["heuristic"] = worker_module.tree_usage(me, os.getppid())[0]
        with patch.object(worker_module, "_shares_parent_memory", return_value=False):  # not told apart
            measured["unchecked"] = worker_module.tree_usage(me, os.getppid())[0]
        thread.join(10)
        os.waitpid(started["pid"], 0)
        self.assertLess(measured["kcmp"], 1.5 * alone, measured)
        self.assertLess(measured["heuristic"], 1.5 * alone, measured)
        self.assertGreater(measured["unchecked"], 1.5 * alone, measured)  # the child does share this memory

    @pytest.mark.skipif(platform.machine() not in ("x86_64", "aarch64") or not os.path.isdir("/proc/self/task"),
                        reason="clone() with a C callback, as glibc on Linux has it")
    def test_where_kcmp_cannot_tell_a_forked_child_is_not_taken_for_a_vforked_one(self):
        # A thread waits in vfork while a forked child, with the same code and stack addresses, holds memory
        # of its own: past VFORK_SECONDS, that child counts.
        libc = ctypes.CDLL(None, use_errno=True)
        libc.clone.restype = ctypes.c_int
        libc.clone.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        stack = ctypes.create_string_buffer(1 << 16)
        me, started = os.getpid(), {}

        def spawn():
            started["pid"] = libc.clone(ctypes.cast(libc.sleep, ctypes.c_void_p), ctypes.addressof(stack) + len(stack)
                                        - 256, 0x100 | 0x4000 | signal.SIGCHLD, 4)  # CLONE_VM | CLONE_VFORK
        thread = threading.Thread(target=spawn, daemon=True)
        thread.start()
        read, write = os.pipe()
        child = os.fork()
        if child == 0:
            block = bytearray(64 << 20)
            for offset in range(0, len(block), 4096):
                block[offset] = 1
            os.write(write, b"x")
            time.sleep(30)
            os._exit(0)
        self.addCleanup(lambda: (os.kill(child, signal.SIGKILL), os.waitpid(child, 0)))
        os.close(write)
        os.read(read, 1)
        time.sleep(worker_module.VFORK_SECONDS + 0.5)
        with patch.object(worker_module, "_same_memory", return_value=None):  # where kcmp is refused
            shares = worker_module._shares_parent_memory(me, child)
            counted = worker_module.tree_usage(child, me)[0] if not shares else 0
        thread.join(10)
        os.waitpid(started["pid"], 0)
        self.assertFalse(shares)
        self.assertGreater(counted, 60 << 20)

    @pytest.mark.skipif(platform.machine() != "x86_64" or not os.path.isdir("/proc/self/task"),
                        reason="SYS_exit's number, on Linux")
    def test_a_process_whose_main_thread_has_exited_still_counts(self):
        holder = ("import ctypes, mmap, threading, time\n"
                  "def hold():\n"
                  "    block = mmap.mmap(-1, 96 << 20, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)\n"
                  "    for offset in range(0, 96 << 20, 4096):\n        block[offset] = 1\n"
                  "    shared = mmap.mmap(-1, 32 << 20)\n"
                  "    for offset in range(0, 32 << 20, 4096):\n        shared[offset] = 1\n"
                  "    print('held', flush=True)\n    time.sleep(60)\n"
                  "threading.Thread(target=hold).start()\nctypes.CDLL(None).syscall(60, 0)\n")
        process = subprocess.Popen([sys.executable, "-c", holder], stdout=subprocess.PIPE, text=True)
        self.addCleanup(lambda: (process.kill(), process.wait(), process.stdout.close()))
        self.assertEqual(process.stdout.readline(), "held\n")
        with open(f"/proc/{process.pid}/stat") as stream:
            self.assertEqual(stream.read().rsplit(")", 1)[1].split()[0], "Z")
        self.assertGreater(worker_module.tree_usage(process.pid, os.getpid())[0], 120 << 20)
        self.assertNotEqual(worker_module._live_task(process.pid), process.pid)
        shmem = worker_module._process_usage(process.pid)[3]
        self.assertGreaterEqual(shmem, 32 << 20)  # shared anonymous memory is no file of a sandbox
        self.assertLess(worker_module._file_shmem(process.pid, shmem, worker_module._not_sandbox_shmem()), 8 << 20)
        # Two processes without a main thread do not pass for one address space.
        other = subprocess.Popen([sys.executable, "-c", holder], stdout=subprocess.PIPE, text=True)
        self.addCleanup(lambda: (other.kill(), other.wait(), other.stdout.close()))
        self.assertEqual(other.stdout.readline(), "held\n")
        self.assertFalse(worker_module._shares_parent_memory(process.pid, other.pid))
        real_open = open

        def without_rollup(path, *args, **kwargs):  # as before Linux 4.14
            if str(path).endswith("smaps_rollup"):
                raise FileNotFoundError(2, "No such file or directory")
            return real_open(path, *args, **kwargs)
        with patch.object(worker_module, "open", without_rollup, create=True), \
                patch.object(worker_module.os.path, "exists", lambda path: False):
            self.assertGreater(worker_module.tree_usage(process.pid, os.getpid())[0], 120 << 20)

    @pytest.mark.skipif(platform.machine() != "x86_64" or not os.path.isdir("/proc/self/task"),
                        reason="SYS_exit's number, on Linux")
    def test_a_process_whose_threads_each_live_briefly_still_counts(self):
        # Its main thread has exited, and each thread starts the next and ends at once.
        churn = ("import _thread, ctypes, mmap, sys, threading, time\n"
                 "size = 64 << 20\nblock = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)\n"
                 "for offset in range(0, size, 4096):\n    block[offset] = 1\n"
                 "def step():\n    _thread.start_new_thread(step, ())\n    until = time.time() + float(sys.argv[1])\n"
                 "    while time.time() < until:\n        pass\n"
                 "threading.Thread(target=step).start()\nprint('held', flush=True)\nctypes.CDLL(None).syscall(60, 0)\n")
        for life in ("0", "0.0005", "0.005"):
            with self.subTest(life=life):
                process = subprocess.Popen([sys.executable, "-c", churn, life], stdout=subprocess.PIPE, text=True)
                try:
                    self.assertEqual(process.stdout.readline(), "held\n")
                    time.sleep(0.3)
                    meter = worker_module.TreeMeter(process.pid)
                    samples = [meter.measure()[0] >> 20 for _ in range(50)]
                finally:
                    process.kill()
                    process.wait()
                    process.stdout.close()
                self.assertGreater(min(samples), 60, samples)

    def test_the_last_memory_read_stands_in_for_a_process_that_shows_none(self):
        me, last = os.getpid(), {}
        read = worker_module._process_usage(me, last=last)[1]
        with patch.object(worker_module, "_read_memory", side_effect=worker_module._NoTask("churn")):
            self.assertEqual(worker_module._process_usage(me, last=last)[1], read)
            last[me] = (b"1", *last[me][1:])  # another process that had this PID
            self.assertEqual(worker_module._process_usage(me, last=last)[1], 0)
        last[1 << 30] = (b"1", 5, None)
        worker_module.tree_usage(me, os.getppid(), last=last)
        self.assertNotIn(1 << 30, last)

    def test_the_sandboxs_own_file_systems_are_decided_once(self):
        meter, asked = worker_module.TreeMeter(os.getpid(), [["/"]]), []
        meter.init = os.getpid()
        real = os.stat
        with patch.object(worker_module.os, "stat", lambda path, **kw: "host" if path == "/" else real(path, **kw)), \
                patch.object(meter, "_own", lambda group: asked.append(group) or []):
            self.assertEqual([meter.private_bytes(), meter.private_bytes()], [0, 0])
        self.assertEqual(asked, [["/"]])

    def test_where_proc_hides_their_addresses_a_child_is_not_taken_for_a_vforked_one(self):
        # For a process the router may not trace, /proc shows the code and stack addresses as 1, 1 and 0.
        boot = int(time.clock_gettime(time.CLOCK_BOOTTIME) * os.sysconf("SC_CLK_TCK"))
        fields = ["S"] + ["0"] * 49
        fields[19], fields[23:26] = str(boot), ["1", "1", "0"]
        hidden, waiting = f"7 (tool) {' '.join(fields)}\n".encode(), f"7 (tool) D {' '.join(fields[1:])}\n".encode()
        real_open, real_listdir = open, os.listdir

        def fake_open(path, *args, **kwargs):
            if str(path).startswith(("/proc/70001/", "/proc/70002/")):
                return __import__("io").BytesIO(waiting if "/task/" in str(path) else hidden)
            return real_open(path, *args, **kwargs)

        def fake_listdir(path):
            return ["70001"] if str(path) == "/proc/70001/task" else real_listdir(path)
        with patch.object(worker_module, "open", fake_open, create=True), \
                patch.object(worker_module.os, "listdir", fake_listdir):
            self.assertFalse(worker_module._waiting_on_vfork(70001, 70002))
            fields[23:26] = ["4194304", "4202496", "140735000000000"]  # addresses it may read: the heuristic holds
            hidden = f"7 (tool) {' '.join(fields)}\n".encode()
            self.assertTrue(worker_module._waiting_on_vfork(70001, 70002))

    @pytest.mark.skipif(not os.access("/dev/shm", os.W_OK), reason="no writable /dev/shm here")
    def test_a_host_tmpfs_file_a_process_maps_is_not_taken_for_a_sandbox_file(self):
        # A sandbox maps the host's files read-only; one on a host tmpfs is not in the sandbox's file systems,
        # which the meter counts itself, so it must not come off the processes' memory.
        import mmap

        with tempfile.NamedTemporaryFile(dir="/dev/shm", prefix="openvlharness-test-") as stream:
            stream.write(b"\1" * (32 << 20))
            stream.flush()
            view = mmap.mmap(stream.fileno(), 0, prot=mmap.PROT_READ)
            try:
                self.assertEqual(sum(view[offset] for offset in range(0, 32 << 20, 4096)), 8192)
                memory, _, mapped = worker_module.tree_usage(os.getpid(), os.getppid(), mapped_files=True)
            finally:
                view.close()
        self.assertGreater(memory, 32 << 20)
        self.assertLess(mapped, 8 << 20)

    def test_with_file_writes_no_file_outgrows_the_memory_limit(self):
        # A memfd holds memory no process's address space shows; the file-size cap bounds each one.
        worker = self.start(self.catalog("memfd_writer"), memory_gib=0.25, write_files=True)
        self.assertEqual(worker.limits["file_bytes"], 1 << 28)
        small = worker.execute("memfd_writer", {"value": str(1 << 20)}, tool_call_id="call-1")[0]["result"]
        large = worker.execute("memfd_writer", {"value": str(300 << 20)}, tool_call_id="call-2")[0]["result"]
        self.assertEqual((small, large), ("written", "OSError: File too large"))

    def test_tree_usage_counts_a_process_and_its_descendants(self):
        holder = ("import time\nblock = bytearray(64 << 20)\nfor offset in range(0, len(block), 4096):\n"
                  "    block[offset] = 1\nprint('held', flush=True)\ntime.sleep(60)\n")
        starter = ("import subprocess, sys, time\nchild = subprocess.Popen([sys.executable, '-c', sys.argv[1]])\n"
                   "print(child.pid, flush=True)\ntime.sleep(60)\n")
        parent = subprocess.Popen([sys.executable, "-c", starter, holder], stdout=subprocess.PIPE, text=True)
        self.addCleanup(parent.wait)
        self.addCleanup(parent.kill)
        child = int(parent.stdout.readline())
        self.addCleanup(lambda: subprocess.run(["kill", "-9", str(child)], stderr=subprocess.DEVNULL))
        self.assertEqual(parent.stdout.readline().strip(), "held")  # the child's 64 MiB are touched
        for files in (True, False):  # each thread's children file, or a scan of /proc where the kernel has none
            with self.subTest(children_files=files), patch.object(worker_module, "_has_children_files", files):
                memory, cpu, _ = worker_module.tree_usage(parent.pid, os.getpid())
                self.assertGreater(memory, 64 << 20)
                self.assertGreater(cpu, 0)
        self.assertIsNone(worker_module.tree_usage(parent.pid, parent.pid))  # its parent is not the one given

    def test_work_left_running_after_a_call_stops_the_worker_within_about_one_window(self):
        # The thresholds are module constants; small ones keep the test short.
        worker = self.start(self.catalog("burner", "echo"))
        with patch.multiple(worker_module, IDLE_CPU_WINDOW_SECONDS=2, IDLE_CPU_SECONDS=0.5, IDLE_SETTLE_SECONDS=0.5):
            first = worker.current
            self.assertEqual(worker.execute("echo", {"value": "a"}, tool_call_id="call-1")[1], 200)
            time.sleep(4)  # two quiet windows: an idle worker stays
            self.assertIsNone(first.process.poll())
            self.assertIs(worker.current, first)
            with self.assertLogs("openvlharness.toolsmith.worker", "WARNING") as logs:
                self.assertEqual(worker.execute("burner", {}, tool_call_id="call-2")[1], 200)
                returned = time.monotonic()
                while first.process.poll() is None and time.monotonic() - returned < 15:
                    time.sleep(0.1)
            self.assertIsNotNone(first.process.poll())
            self.assertLess(time.monotonic() - returned, 0.5 + 2 + 1.5)  # settling, one window, two samples
            self.assertTrue(any("CPU-seconds in 2 seconds with no call in flight" in line for line in logs.output),
                            logs.output)
            self.assertEqual(worker.execute("echo", {"value": "ab"}, tool_call_id="call-3"),
                             (_build_success_payload("echo", "call-3", 2, {}), 200))
            self.assertIsNot(worker.current, first)

    def test_tool_output_is_bounded_across_the_workers_of_a_router(self):
        # Each worker prints 100 KiB and crashes, so every call has a new worker with its own per-worker budget.
        directory = self.catalog("shouter")
        log = Path(directory, "router.log")
        script = ("import json, sys, time\nfrom openvlharness.toolsmith import worker as module\n"
                  "module.STDERR_BYTES, module.STDERR_TOTAL_BYTES = 64 << 10, 160 << 10\n"
                  "worker = module.CatalogWorker(sys.argv[1], client=None, isolation=sys.argv[2] == '1', "
                  "report_environment=False)\n"
                  "statuses = [worker.execute('shouter', {}, tool_call_id=f'call-{index}')[1] for index in range(4)]\n"
                  "time.sleep(2)\nprint(json.dumps(statuses), file=sys.__stdout__)\nworker.close()\n")
        with open(log, "w") as stream:
            done = subprocess.run([sys.executable, "-c", script, directory, "1" if ISOLATION else "0"],
                                  stdout=subprocess.PIPE, stderr=stream, text=True, timeout=180,
                                  env={**os.environ, "PYTHONPATH": str(worker_module.PACKAGE_ROOT)})
        self.assertEqual(json.loads(done.stdout), [500] * 4)
        text = log.read_text()
        self.assertEqual(text.count("q"), 160 << 10)
        self.assertEqual(text.count("output past 65536 bytes dropped]"), 2)  # the first two workers
        self.assertEqual(text.count(f"generated tools printed {160 << 10} bytes; what they print from now on is "
                                    "dropped]"), 1)

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
                spec = worker.confinement(Path(xdg, "home"))  # the spec only names the home directory
            self.assertEqual(spec["overlay"], [os.path.join(xdg, "huggingface"), os.path.join(xdg, "torch")])
            self.assertIn(os.path.join(xdg, "huggingface", "token"), spec["mask"])

    def test_configuration_settings_reach_the_worker(self):
        directory = self.catalog("probe")
        config = {"toolsmith": {"catalog": directory, "worker": {"environment": ["GRANTED_TOKEN"], "memory_gib": 2,
                                                                 "write_files": True, "isolation": ISOLATION,
                                                                 "cache_layer_gib": 0.25,
                                                                 "network_ports": [443, 8443],
                                                                 "deny_networks": ["203.0.113.0/24"]}},
                  "services": {"client": {"base_url": "http://router.invalid", "timeout_seconds": 42}}}
        worker = CatalogWorker.from_config(config, client=FakeClient(), base_tools={"OCR_Tool", "Depth_Estimation_Tool"})
        self.addCleanup(worker.close)
        parent = None if ISOLATION else os.getpid()  # an unconfined worker watches the router itself
        # A confined, networked worker also gets the socket of the router's outbound proxy.
        proxy = {"proxy": worker.proxy.path} if ISOLATION else {}
        self.assertEqual((worker.timeout, worker.limits, worker.refused, worker.memory_limit),
                         (37.8, {"memory_bytes": 2 << 30, "file_bytes": 2 << 30, "message_bytes": 32 << 20,
                                 "parent": parent, **proxy},
                          {"Depth_Estimation_Tool"}, 2 << 30))
        self.assertEqual(worker.confinement(Path(directory, "home"))["layer_bytes"], 1 << 28)
        if ISOLATION:
            self.assertEqual((worker.proxy.ports, [str(network) for network in worker.proxy.deny]),
                             ({443, 8443}, ["203.0.113.0/24"]))
        config["toolsmith"]["worker"] = {"isolation": ISOLATION, "timeout_seconds": 42, "network": False}
        with self.assertLogs("openvlharness.toolsmith.worker", "WARNING") as logs:
            worker = CatalogWorker.from_config(config, client=FakeClient(), base_tools=set())
        self.addCleanup(worker.close)
        self.assertEqual((worker.timeout, worker.limits, worker.memory_limit, worker.proxy),
                         (42, {"memory_bytes": 128 << 30, "file_bytes": 0, "message_bytes": 32 << 20,
                               "parent": parent}, 128 << 30, None))
        self.assertEqual(worker.confinement(Path(directory, "home"))["layer_bytes"], 1 << 30)
        self.assertTrue(any("may time out and retry" in line for line in logs.output))
