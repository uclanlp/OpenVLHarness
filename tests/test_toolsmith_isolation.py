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
def nested():
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.unshare(0x10000000) != 0:  # CLONE_NEWUSER
        raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
attempt("nested user namespace", nested)
attempt("process cap", lambda: __import__("resource").getrlimit(__import__("resource").RLIMIT_NPROC)[0])
attempt("resolver", lambda: os.path.exists("/run/systemd/resolve"))
attempt("tmpfs", lambda: {path: [os.statvfs(path).f_blocks * os.statvfs(path).f_frsize, os.statvfs(path).f_files]
                          for path in ("/tmp", "/dev/shm", "/dev", "/")})
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
            spec = {"root": root, "read": isolation.system_paths(network=network) + [str(script)], "tmp": ["/tmp"],
                    "cwd": "/tmp", "network": network}
            argv = [sys.executable, str(script), str(secret), str(os.getpid()), str(server.getsockname()[1])]
            done = subprocess.run(isolation.command(spec, argv), capture_output=True, text=True, timeout=60,
                                  env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(PACKAGE_ROOT)})
        self.assertEqual(done.returncode, 0, done.stderr)
        return json.loads(done.stdout)

    def test_an_overlay_path_takes_writes_in_a_private_layer(self):
        script = ("import json, os, sys\ncache = sys.argv[1]\nseen = [open(os.path.join(cache, 'config.json')).read()]\n"
                  "open(os.path.join(cache, 'config.json'), 'w').write('changed')\n"
                  "open(os.path.join(cache, 'planted.py'), 'w').write('print(1)')\n"
                  "seen += [open(os.path.join(cache, 'config.json')).read(), sorted(os.listdir(cache)),"
                  " os.path.exists('/.layers')]\nprint(json.dumps(seen))\n")
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory(prefix="openvlharness-") as root:
            cache = Path(home, "hf_home")
            cache.mkdir()
            (cache / "config.json").write_text("host")
            spec = {"root": root, "read": isolation.system_paths(), "overlay": [str(cache)], "tmp": ["/tmp"],
                    "cwd": "/tmp", "network": False}
            done = subprocess.run(isolation.command(spec, [sys.executable, "-c", script, str(cache)]),
                                  capture_output=True, text=True, timeout=60,
                                  env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(PACKAGE_ROOT)})
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertEqual(json.loads(done.stdout), ["host", "changed", ["config.json", "planted.py"], False])
            self.assertEqual(((cache / "config.json").read_text(), sorted(os.listdir(cache))), ("host", ["config.json"]))

    def test_the_overlay_layer_has_the_size_the_spec_gives(self):
        script = ("import json, os, sys\ncache = sys.argv[1]\nstat = os.statvfs(cache)\noutcome = []\n"
                  "for megabytes in (8, 24):\n    try:\n"
                  "        with open(os.path.join(cache, f'{megabytes}.bin'), 'wb') as stream:\n"
                  "            stream.write(b'0' * (megabytes << 20))\n        outcome.append('written')\n"
                  "    except OSError as exc:\n        outcome.append(exc.strerror)\n"
                  "print(json.dumps([stat.f_blocks * stat.f_frsize, outcome]))\n")
        for layer_bytes, expected in ((16 << 20, [16 << 20, ["written", "No space left on device"]]),
                                      (None, [isolation.LAYER_BYTES, ["written", "written"]])):
            with self.subTest(layer_bytes=layer_bytes), tempfile.TemporaryDirectory() as home, \
                    tempfile.TemporaryDirectory(prefix="openvlharness-") as root:
                cache = Path(home, "hf_home")
                cache.mkdir()
                spec = {"root": root, "read": isolation.system_paths(), "overlay": [str(cache)], "tmp": ["/tmp"],
                        "cwd": "/tmp", "network": False, **({"layer_bytes": layer_bytes} if layer_bytes else {})}
                done = subprocess.run(isolation.command(spec, [sys.executable, "-c", script, str(cache)]),
                                      capture_output=True, text=True, timeout=60,
                                      env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(PACKAGE_ROOT)})
                self.assertEqual(done.returncode, 0, done.stderr)
                self.assertEqual(json.loads(done.stdout), expected)
                self.assertEqual(os.listdir(cache), [])

    def test_without_network_the_resolver_socket_is_not_mounted(self):
        # A file-system socket reaches systemd-resolved whatever the network namespace.
        host = os.path.exists("/run/systemd/resolve")
        self.assertEqual((self.confined(network=True)["resolver"], self.confined(network=False)["resolver"]),
                         (["ok", host], ["ok", False]))
        self.assertNotIn("/run/systemd/resolve", isolation.system_paths(network=False))

    def test_nested_namespaces_are_refused_and_processes_capped(self):
        # A nested user namespace would bring back capabilities: to mount an uncapped tmpfs, or look under a mask.
        seen = self.confined(network=False)
        self.assertEqual(seen["nested user namespace"], ["blocked", "OSError"])
        release = tuple(int(part) for part in __import__("re").findall(r"\d+", __import__("platform").release())[:2])
        if release >= (5, 14):
            self.assertEqual(seen["process cap"], ["ok", isolation.PROCESSES])

    def test_temporary_file_systems_are_capped(self):
        # tmpfs memory is outside the address-space limit, so each mount has its own size and file cap.
        self.assertEqual(self.confined(network=False)["tmpfs"], ["ok", {
            "/tmp": [1 << 30, 65536], "/dev/shm": [1 << 30, 65536], "/dev": [16 << 20, 65536], "/": [16 << 20, 65536]}])

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


    def test_tools_never_write_the_host_model_caches(self):
        writer = TOOL.replace("class Snoop", "class Writer").replace('"snoop"', '"cache_writer"').replace(
            "    def execute(self, secret=\"\", port=0, token=\"\", router=0):",
            "    def execute(self, path=\"\"):\n"
            "        try:\n            open(path, \"w\").write(\"poisoned\")\n        except OSError as exc:\n"
            "            return type(exc).__name__\n        return open(path).read()\n\n"
            "    def unused(self, secret=\"\", port=0, token=\"\", router=0):")
        with tempfile.TemporaryDirectory() as home:
            catalog, cache = Path(home, "catalog"), Path(home, "hf_home")
            catalog.mkdir()
            (catalog / "cache_writer.py").write_text(writer)
            (cache / "hub").mkdir(parents=True)
            (cache / "hub" / "config.json").write_text("host")
            for write_files, expected in ((True, "poisoned"), (False, "OSError")):
                with self.subTest(write_files=write_files), patch.dict(os.environ, {"HF_HOME": str(cache)}):
                    worker = CatalogWorker(catalog, client=FakeClient(), write_files=write_files)
                    try:
                        result = worker.execute("cache_writer", {"path": str(cache / "hub" / "config.json")},
                                                tool_call_id="call-1")[0]["result"]
                    finally:
                        worker.close()
                    self.assertIn(expected, result)
                    self.assertEqual((cache / "hub" / "config.json").read_text(), "host")


class FailClosedTests(unittest.TestCase):
    @needs_isolation
    def test_a_namespace_that_cannot_refuse_nested_ones_is_not_used(self):
        # As if the kernel kept user.max_user_namespaces read-only inside the namespace. launch() ends the
        # process with os._exit, so the launcher, not the with, removes the root (cleanup).
        code = ("import builtins, sys\nopen_ = builtins.open\n"
                "def refusing(path, *args, **kwargs):\n"
                "    if str(path).endswith('max_user_namespaces'):\n"
                "        raise PermissionError(13, 'Permission denied')\n"
                "    return open_(path, *args, **kwargs)\n"
                "builtins.open = refusing\nfrom openvlharness.toolsmith import isolation\n"
                "with __import__('tempfile').TemporaryDirectory(prefix='openvlharness-') as root:\n"
                "    isolation.launch({'root': root, 'read': isolation.system_paths(network=False), 'tmp': ['/tmp'],"
                " 'network': False, 'cleanup': root}, [sys.executable, '-c', 'print(\"confined\")'])\n")
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                              env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(PACKAGE_ROOT)})
        self.assertEqual((done.returncode, done.stdout), (isolation.SETUP_FAILED, ""))
        self.assertIn("cannot refuse nested user namespaces", done.stderr)

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


class OverlayFallbackTests(unittest.TestCase):
    """The meter counts the sandbox's own file systems, never the host's behind a bind of a cache."""

    def private_bytes(self, caches, patch_overlay=""):
        from openvlharness.toolsmith.worker import TreeMeter

        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory(prefix="openvlharness-") as root:
            overlay = [os.path.join(home, name) for name in caches]
            for path in overlay:
                os.mkdir(path)
                # What the host's file system holds at least, however empty it was.
                Path(path, "weights.bin").write_bytes(b"\1" * (64 << 20))
            spec = {"root": root, "read": isolation.system_paths(), "tmp": ["/tmp"], "overlay": overlay, "cwd": "/tmp",
                    "parent": os.getpid()}
            script = ("import os, sys, time\n"
                      f"for path in {overlay!r}:\n"
                      "    try:\n        open(os.path.join(path, 'blob'), 'wb').write(b'0' * (8 << 20))\n"
                      "    except OSError:\n        pass\n"
                      "print('ready', flush=True)\ntime.sleep(30)\n")
            code = ("import json, sys\nfrom openvlharness.toolsmith import isolation\n" + patch_overlay
                    + "isolation.launch(json.loads(sys.argv[1]), sys.argv[2:])\n")
            argv = [sys.executable, "-c", code, json.dumps(spec), sys.executable, "-c", script]
            process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                       env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(PACKAGE_ROOT)})
            try:
                self.assertEqual(process.stdout.readline(), "ready\n")
                return TreeMeter(process.pid, isolation.private_mounts(spec)).private_bytes()
            finally:
                process.kill()
                process.wait()
                process.stdout.close()

    FAIL = ("real = isolation._mount\n"
            "def refuse(libc, source, target, fstype, *rest):\n"
            "    if fstype == 'overlay' and FAILS(str(target)):\n"
            "        raise OSError(22, 'Invalid argument')\n"
            "    return real(libc, source, target, fstype, *rest)\n"
            "isolation._mount = refuse\n")

    @needs_isolation
    def test_a_cache_that_is_bound_read_only_counts_nothing(self):
        for name, caches, code in (("a comma in its path", ["hf,home"], ""),
                                   ("a failed mount", ["hf_home"], self.FAIL.replace("FAILS", "bool"))):
            with self.subTest(name):
                self.assertLess(self.private_bytes(caches, code), 8 << 20)

    @needs_isolation
    def test_the_layer_counts_once_even_when_the_first_overlay_failed(self):
        both = self.private_bytes(["hf_home", "torch"])
        self.assertTrue(16 << 20 <= both < 24 << 20, both)
        second = self.private_bytes(["hf_home", "torch"], self.FAIL.replace("FAILS(", "(lambda path: 'hf_home' in path)("))
        self.assertTrue(8 << 20 <= second < 16 << 20, second)


class SysVTests(unittest.TestCase):
    @needs_isolation
    @unittest.skipIf(isolation._release() < isolation.IPC_SYSCTL_KERNEL if sys.platform == "linux" else True,
                     "a user namespace cannot set SysV limits before Linux 5.19")
    @unittest.skipUnless(os.path.exists("/proc/sys/kernel/shmmni"), "a kernel without SysV IPC")
    def test_the_sandbox_gets_no_shared_memory_segment(self):
        code = ("import ctypes\nlibc = ctypes.CDLL(None, use_errno=True)\n"
                "print(libc.shmget(0, 4096, 0o1600), ctypes.get_errno())\n")
        with tempfile.TemporaryDirectory(prefix="openvlharness-") as root:
            spec = {"root": root, "read": isolation.system_paths(), "tmp": ["/tmp"], "cwd": "/tmp"}
            done = subprocess.run(isolation.command(spec, [sys.executable, "-c", code]), capture_output=True, text=True,
                                  timeout=60, check=False,
                                  env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(PACKAGE_ROOT)})
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.split()[0], "-1", done.stdout)


class SysVLimitTests(unittest.TestCase):
    """Which sysctls the launcher writes, and which failures end the launch (the kernel is mocked)."""

    def run_limits(self, release, refused):
        written = {}

        def fake_open(path, mode="r"):
            if str(path).startswith("/proc/sys/kernel/") and refused:
                raise PermissionError(13, "Permission denied") if refused is True else refused

            class Sink:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                def write(self, value):
                    written[str(path)] = value
            return Sink()
        with patch("builtins.open", fake_open), patch.object(isolation, "_release", lambda: release), \
                patch.object(isolation.resource, "setrlimit"):
            isolation._limit_namespace()
        return written

    def test_every_sysv_limit_is_written_to_its_sysctl(self):
        written = self.run_limits((6, 8), refused=False)
        self.assertEqual({name: written[f"/proc/sys/{name}"] for name in isolation.IPC_LIMITS}, isolation.IPC_LIMITS)
        self.assertEqual(written["/proc/sys/user/max_user_namespaces"], "0")

    def test_a_kernel_that_allows_the_limits_but_refuses_them_fails_the_launch(self):
        with self.assertRaisesRegex(OSError, "cannot refuse SysV IPC"):
            self.run_limits((6, 8), refused=True)

    def test_a_kernel_without_sysv_ipc_has_nothing_to_refuse(self):
        missing = FileNotFoundError(2, "No such file or directory")  # CONFIG_SYSVIPC=n: no kernel/shm* sysctls
        self.assertEqual(self.run_limits((6, 8), refused=missing), {"/proc/sys/user/max_user_namespaces": "0"})

    def test_older_kernels_keep_the_hosts_limits(self):
        self.assertEqual(self.run_limits((5, 15), refused=True), {"/proc/sys/user/max_user_namespaces": "0"})
