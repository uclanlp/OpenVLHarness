"""Linux-namespace confinement of Gate 1 runs and of the generated-tool worker, offline."""
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from openvlharness.toolsmith import isolation
from openvlharness.toolsmith.worker import PACKAGE_ROOT, CatalogWorker

CONFINABLE, REASON = isolation.available()
needs_isolation = pytest.mark.skipif(not CONFINABLE, reason=f"this host cannot confine processes: {REASON}")

PROBE = r'''
import json, os, socket, sys
seen = {}
def attempt(name, action):
    try:
        seen[name] = ["ok", action()]
    except Exception as exc:
        seen[name] = ["blocked", type(exc).__name__]
attempt("secret", lambda: open(sys.argv[1]).read())
attempt("processes", lambda: sorted(int(entry) for entry in os.listdir("/proc") if entry.isdigit()))
attempt("caller environ", lambda: open(f"/proc/{sys.argv[2]}/environ").read()[:4])
attempt("pid", os.getpid)
attempt("write prefix", lambda: open(os.path.join(sys.prefix, "isolation-test"), "w").write("x"))
attempt("connect", lambda: socket.create_connection(("127.0.0.1", int(sys.argv[3])), timeout=5).close())
attempt("uid", os.getuid)
attempt("capabilities", lambda: [line.split()[1] for line in open("/proc/self/status") if line.startswith("CapEff")][0])
attempt("etc", lambda: sorted(os.listdir("/etc")))
attempt("opt", lambda: os.listdir("/opt"))
print(json.dumps(seen))
'''

TOOL = '''
import os
import socket

from openvlharness.toolsmith import GeneratedTool


class Snoop(GeneratedTool):
    def __init__(self):
        super().__init__(tool_name="snoop", tool_description="snoop", input_schema={"type": "object"})

    def execute(self, secret="", port=0, token="", router=0):
        seen = {}
        for name, action in (("secret", lambda: open(secret).read()),
                             ("granted file", lambda: open(os.environ["SERVICE_KEY_FILE"]).read()),
                             ("router environ", lambda: open(f"/proc/{router}/environ").read()[:4]),
                             ("processes", lambda: sorted(int(p) for p in os.listdir("/proc") if p.isdigit())),
                             ("token", lambda: open(token).read()),
                             ("connect", lambda: socket.create_connection(("127.0.0.1", port), timeout=5).close())):
            try:
                seen[name] = ["ok", action()]
            except Exception as exc:
                seen[name] = ["blocked", type(exc).__name__]
        return seen
'''


def listener():
    """Accept connections on a loopback port until the test ends; return the port."""
    server = socket.create_server(("127.0.0.1", 0))

    def accept():
        while True:
            try:
                server.accept()[0].close()
            except OSError:
                return
    threading.Thread(target=accept, daemon=True).start()
    return server


class FakeClient:
    def execute(self, tool_name, args, *, tool_call_id):
        return {"ok": False, "error": {"code": "unavailable", "message": "no tools in this test"}}


@needs_isolation
class LauncherTests(unittest.TestCase):
    def confined(self, *, network):
        server = listener()
        self.addCleanup(server.close)
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory(prefix="openvlharness-") as root:
            secret, script = Path(home, "secret.txt"), Path(home, "probe.py")
            secret.write_text("a key")
            script.write_text(PROBE)
            spec = {"root": root, "read": isolation.system_paths() + [str(script)], "tmp": ["/tmp"],
                    "cwd": "/tmp", "network": network}
            argv = [sys.executable, str(script), str(secret), str(os.getpid()), str(server.getsockname()[1])]
            done = subprocess.run(isolation.command(spec, argv), capture_output=True, text=True, timeout=60,
                                  env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(PACKAGE_ROOT)})
        self.assertEqual(done.returncode, 0, done.stderr)
        return json.loads(done.stdout)

    def test_a_confined_process_sees_only_what_it_is_given(self):
        seen = self.confined(network=False)
        self.assertEqual(seen["secret"], ["blocked", "FileNotFoundError"])
        self.assertEqual((seen["processes"], seen["pid"]), (["ok", [1, 2]], ["ok", 2]))  # the init and the command
        self.assertEqual(seen["caller environ"][0], "blocked")
        self.assertEqual(seen["write prefix"], ["blocked", "OSError"])
        self.assertEqual(seen["connect"][0], "blocked")
        self.assertEqual(seen["uid"], ["ok", os.getuid()])
        self.assertEqual(seen["capabilities"], ["ok", "0000000000000000"])
        self.assertEqual(seen["etc"], ["ok", sorted(name for name in isolation.ETC if os.path.lexists(f"/etc/{name}"))])
        self.assertEqual(seen["opt"], ["blocked", "FileNotFoundError"])

    def test_network_access_is_opt_in(self):
        self.assertEqual(self.confined(network=True)["connect"], ["ok", None])

    def test_the_exit_status_and_signals_of_the_command_come_back(self):
        with tempfile.TemporaryDirectory(prefix="openvlharness-") as root:
            spec = {"root": root, "read": isolation.system_paths(), "tmp": ["/tmp"]}
            environment = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(PACKAGE_ROOT)}
            for code, expected in (("raise SystemExit(7)", 7), ("import os, signal; os.kill(os.getpid(), signal.SIGTERM)",
                                                                  -15), ("import os; os.abort()", -6)):
                with self.subTest(code=code):
                    done = subprocess.run(isolation.command(spec, [sys.executable, "-c", code]), env=environment,
                                          capture_output=True, timeout=60)
                    self.assertEqual(done.returncode, expected)


class GateOneIsolationTests(unittest.TestCase):
    SOURCE = '''import re

from openvlharness.toolsmith import GeneratedTool


class Snoop(GeneratedTool):
    def __init__(self):
        super().__init__(tool_name="snoop", tool_description="d", input_schema={{"type": "object", "properties": {{}}}})

    def execute(self):
        {body}
'''

    @needs_isolation
    def test_gate_one_runs_without_host_files_or_network_even_when_the_scan_is_bypassed(self):
        from openvlharness.toolsmith.pipeline.verification import run_once

        server = listener()
        self.addCleanup(server.close)
        with tempfile.TemporaryDirectory() as home:
            secret = Path(home, "secret.txt")
            secret.write_text("a key")
            read = self.SOURCE.format(body=f'return __builtins__["open"]({str(secret)!r}).read()')
            self.assertEqual(run_once(read, 30, isolation=False)[0]["tool_name"], "snoop")
            self.assertIn("No such file or directory", run_once(read, 30)[1])
            connect = self.SOURCE.format(body='return __builtins__["__import__"]("socket").create_connection('
                                              f'("127.0.0.1", {server.getsockname()[1]}), timeout=5) and "connected"')
            self.assertIsNone(run_once(connect, 30, isolation=False)[1])
            self.assertIn("Network is unreachable", run_once(connect, 30)[1])


@needs_isolation
class WorkerIsolationTests(unittest.TestCase):
    def test_generated_tools_see_neither_host_files_nor_the_router_but_keep_the_network(self):
        server = listener()
        self.addCleanup(server.close)
        with tempfile.TemporaryDirectory() as home:
            catalog, cache = Path(home, "catalog"), Path(home, "hf_home")
            catalog.mkdir()
            (catalog / "snoop.py").write_text(TOOL)
            cache.mkdir()
            (cache / "token").write_text("hf_secret")
            secret, granted = Path(home, "secret.txt"), Path(home, "service.key")
            secret.write_text("a key")
            granted.write_text("granted key")
            arguments = {"secret": str(secret), "port": server.getsockname()[1], "token": str(cache / "token"),
                         "router": os.getpid()}
            for network in (True, False):
                with self.subTest(network=network), patch.dict(os.environ, {"HF_HOME": str(cache),
                                                                            "SERVICE_KEY_FILE": str(granted)}):
                    worker = CatalogWorker(catalog, client=FakeClient(), network=network,
                                           environment=["SERVICE_KEY_FILE"])
                    try:
                        seen = worker.execute("snoop", arguments, tool_call_id="call-1")[0]["result"]
                    finally:
                        worker.close()
                    self.assertEqual(seen["secret"], ["blocked", "FileNotFoundError"])
                    self.assertEqual(seen["granted file"], ["ok", "granted key"])
                    self.assertEqual(seen["router environ"][0], "blocked")
                    self.assertEqual(seen["processes"], ["ok", [1, 2]])
                    self.assertEqual(seen["token"], ["ok", ""])
                    self.assertEqual(seen["connect"][0], "ok" if network else "blocked")


class FailClosedTests(unittest.TestCase):
    def test_root_callers_are_refused(self):
        with patch.object(isolation, "_probe", None), patch.object(isolation.os, "getuid", return_value=0):
            confinable, reason = isolation.available()
        self.assertEqual((confinable, reason), (False, isolation.ROOT_REFUSED))
        code = ("import os, sys; os.getuid = lambda: 0; from openvlharness.toolsmith.isolation import launch; "
                "launch({'root': '/nonexistent'}, [sys.executable, '-c', 'print(1)'])")
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                              env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(PACKAGE_ROOT)})
        self.assertEqual((done.returncode, done.stdout), (isolation.SETUP_FAILED, ""))
        self.assertIn("refuses to run as root", done.stderr)

    def test_without_namespaces_the_worker_runs_only_when_isolation_is_turned_off(self):
        with tempfile.TemporaryDirectory() as catalog:
            with patch.object(isolation, "_probe", (False, "user namespaces are disabled")):
                with self.assertRaisesRegex(RuntimeError, r"\(user namespaces are disabled\)\. Set toolsmith\.worker\."
                                                          r"isolation: false"):
                    CatalogWorker(catalog, client=FakeClient())
                with self.assertLogs("openvlharness.toolsmith.worker", "WARNING") as logs:
                    CatalogWorker(catalog, client=FakeClient(), isolation=False).close()
            self.assertTrue(any("isolation is off" in line for line in logs.output))

    def test_the_probe_reports_why_confinement_fails(self):
        with patch.object(isolation, "_probe", None), patch.object(isolation.sys, "platform", "darwin"):
            confinable, reason = isolation.available()
        self.assertFalse(confinable)
        self.assertIn("needs Linux", reason)
        self.assertEqual(isolation.available(), (CONFINABLE, REASON))  # the cached result is back
