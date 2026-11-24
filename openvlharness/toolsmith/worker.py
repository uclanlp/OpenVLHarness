"""Run the generated tools of a catalog in a separate, resource-limited worker process.

The tool router serves generated tools through a CatalogWorker and never imports
them. The worker process loads a private copy of the catalog, taken when the router
started, and runs each call in its own thread. It inherits the router's environment
without credential-like variables, unless toolsmith.worker.environment grants them,
starts in a private directory, and runs with its address space capped and, unless
toolsmith.worker.write_files is set, without file writes. With toolsmith.worker.isolation
(the default) it runs confined by Linux namespaces (see isolation): it sees only the
interpreter, the packages, the catalog copy and the model caches, read-only, and none of
the router's other files or processes; toolsmith.worker.network: false also removes its
network access.

A generated tool reaches other tools only through call_tool, which the worker passes
to the router process: calls to the other generated tools and to the base tools in
CALLABLE_BASE_TOOLS go through the router as before, and calls to the other router
tools are refused. A call that fails, crashes the worker or runs past the time limit
returns an error envelope, and the router starts a new worker for later calls; a
worker with a call past the limit takes no new calls and is stopped once its other
calls finish.

The router starts the worker as: python -m openvlharness.toolsmith.worker CATALOG LIMITS
"""
import asyncio
import atexit
import concurrent.futures
import itertools
import json
import logging
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from ..contracts import ServiceResult
from ..services.envelope import _build_error_payload, _build_success_payload, _normalize_result_payload
from ..services.tools.base import ToolArgumentError
from .catalog import DEFAULT_CATALOG, catalog_modules, load_catalog

logger = logging.getLogger(__name__)

# The base tools the generator is told it may call (generation.BASE_RESULTS).
CALLABLE_BASE_TOOLS = ("Text_Search_Tool", "Image_Search_Tool", "Webpage_Visit_Tool", "OCR_Tool", "Zoom_In_Tool")
MEMORY_GIB = 128
LOAD_SECONDS = 120
CLIENT_SHARE = 0.9  # default worker deadline, as a share of the agent's HTTP timeout
EXIT_SECONDS = 5
MAX_MESSAGE_BYTES = 256 << 20
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
HOME_PREFIX = "openvlharness-toolsmith-"


def memory_limit_enforced():
    """Whether this platform enforces an address-space limit; macOS rejects RLIMIT_AS outright."""
    return sys.platform.startswith("linux")


def limit_resources(*, memory_bytes=None, file_bytes=None):
    """Cap the address space of the calling process and the size of the files it writes.

    None leaves a limit unchanged; a cap never exceeds the current hard limit, and
    the process cannot raise it again. The address-space cap applies only where
    memory_limit_enforced().
    """
    if not memory_limit_enforced():
        memory_bytes = None
    for kind, value in ((resource.RLIMIT_AS, memory_bytes), (resource.RLIMIT_FSIZE, file_bytes)):
        if value is not None:
            hard = resource.getrlimit(kind)[1]
            value = value if hard == resource.RLIM_INFINITY else min(value, hard)
            resource.setrlimit(kind, (value, value))


def worker_environment(granted=()):
    """Return the router's environment without credential-like variables, except granted names.

    A name looks like a credential under the local Python executor's rule (KEY, TOKEN,
    SECRET, PASSWORD or CREDENTIAL, or an AWS_ or GOOGLE_APPLICATION prefix). The
    package the router runs comes first on PYTHONPATH, so the worker imports the same code.
    """
    from ..services.tools.python_executor_local.executor import _SECRET_ENV

    granted = set(granted)
    environment = {name: value for name, value in os.environ.items()
                   if name in granted or not _SECRET_ENV.search(name)}
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (str(PACKAGE_ROOT), os.environ.get("PYTHONPATH"))))
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


class _Channel:
    """Newline-delimited JSON messages over a pair of pipes."""

    def __init__(self, reader, writer, *, limit=None):
        self.reader, self.writer, self.limit = reader, writer, limit
        self.lock = threading.Lock()

    def send(self, message, **options):
        """Write one message; options go to json.dumps."""
        line = json.dumps(message, **options).encode("ascii") + b"\n"
        with self.lock:
            self.writer.write(line)
            self.writer.flush()

    def receive(self):
        """Return the next message, or None when the other side has closed the pipe.

        Raises:
            ValueError: The message is longer than the limit, cut off or not JSON.
        """
        line = self.reader.readline(self.limit + 1) if self.limit else self.reader.readline()
        if not line:
            return None
        if not line.endswith(b"\n"):
            raise ValueError("message longer than the limit or cut off")
        return json.loads(line)


def _text(exc):
    """Return str(exc), or a placeholder when the exception cannot be printed."""
    try:
        return str(exc)
    except Exception:
        return f"<unprintable {type(exc).__name__}>"


def _valid_result(message):
    """Return whether a result message from the worker has the fields its outcome needs."""
    if message.get("ok") is True:
        return "result" in message and isinstance(message.get("meta"), dict)
    return (message.get("ok") is False and isinstance(message.get("message"), str)
            and isinstance(message.get("exception_type"), str) and isinstance(message.get("argument_error"), bool))


def _valid_call(message):
    """Return whether a call_tool message from the worker names a tool, arguments and a call ID."""
    return (type(message.get("id")) is int and isinstance(message.get("tool"), str)
            and isinstance(message.get("args"), dict) and isinstance(message.get("tool_call_id"), str))


def _failure(tool_name, tool_call_id, message, kind):
    """Return the 500 envelope for a call the worker could not complete."""
    return _build_error_payload(tool_name, tool_call_id, "tool_execution_error", message, {"exception_type": kind}), 500


def _exit_reason(returncode):
    """Describe how a worker process ended."""
    if returncode < 0:
        try:
            return f"the worker was stopped by {signal.Signals(-returncode).name}"
        except ValueError:
            return f"the worker was stopped by signal {-returncode}"
    return f"the worker exited with status {returncode}"


class _WorkerGone(RuntimeError):
    """The worker process has stopped and takes no calls."""


class _Worker:
    """One worker process: its pipes, the calls it is running and the thread that reads its messages."""

    def __init__(self, owner, files, environment, limits):
        """Write the catalog copy to a private directory and start the process."""
        self.owner = owner
        self.home = Path(tempfile.mkdtemp(prefix=HOME_PREFIX))
        self.catalog = self.home / "catalog"
        try:
            self.catalog.mkdir()
            (self.home / "root").mkdir()
            for name, data in files.items():
                (self.catalog / name).write_bytes(data)
            argv = [sys.executable, "-m", "openvlharness.toolsmith.worker", str(self.catalog), json.dumps(limits)]
            if owner.isolation:
                from .isolation import command

                argv = command(owner.confinement(self.home), argv)
            self.process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=self.home,
                                            env=environment, start_new_session=True)
        except BaseException:
            shutil.rmtree(self.home, ignore_errors=True)
            raise
        self.channel = _Channel(self.process.stdout, self.process.stdin, limit=MAX_MESSAGE_BYTES)
        self.ready = concurrent.futures.Future()
        self.calls = {}
        self.abandoned = set()
        self.ids = itertools.count(1)
        self.lock = threading.Lock()
        self.process_lock = threading.Lock()
        self.retiring = self.stopped = False
        self.reason = None
        self.tools = []
        self.reader = threading.Thread(target=self._read, name=f"toolsmith-worker-{self.process.pid}", daemon=True)
        self.reader.start()

    def submit(self, tool_name, args):
        """Send one call and return its ID and the future of the worker's answer.

        Raises:
            _WorkerGone: The worker has stopped.
        """
        with self.lock:
            if self.stopped:
                raise _WorkerGone(self.reason)
            request = next(self.ids)
            answer = self.calls[request] = concurrent.futures.Future()
        try:
            self.channel.send({"type": "execute", "id": request, "tool": tool_name, "args": args})
        except OSError as exc:
            with self.lock:
                self.calls.pop(request, None)
            raise _WorkerGone(self.reason or "the worker closed its input") from exc
        return request, answer

    def send(self, message):
        """Send a message unless the worker has stopped."""
        try:
            self.channel.send(message)
        except OSError:
            pass

    def abandon(self, request):
        """Stop waiting for a call past its time limit; return False when its answer has already arrived."""
        with self.lock:
            if self.calls.pop(request, None) is None:
                return False
            self.abandoned.add(request)
            return True

    def retire(self):
        """Take no new calls and stop the process once the calls still running have finished."""
        with self.lock:
            self.retiring = True
            idle = not self.calls
        if idle:
            self.kill()

    def kill(self):
        """Kill the process and anything it started, unless it has already been reaped."""
        with self.process_lock:
            if self.process.returncode is None:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except OSError:
                    pass
                try:
                    self.process.kill()
                except OSError:
                    pass

    def _read(self):
        """Handle the worker's messages until it stops, then fail the calls still waiting."""
        reason = None
        try:
            while True:
                message = self.channel.receive()
                if message is None:
                    break
                self._handle(message)
        except Exception as exc:
            reason = f"the worker sent an invalid message ({type(exc).__name__}: {_text(exc)})"
            logger.error("ToolSmith worker %s: %s", self.process.pid, reason)
        if reason is None:
            # The isolation launcher outlives the worker briefly while it cleans up; keep its exit status.
            try:
                self.process.wait(timeout=EXIT_SECONDS)
            except subprocess.TimeoutExpired:
                pass
        self.kill()
        with self.process_lock:
            returncode = self.process.wait()
        with self.lock:
            self.stopped = True
            self.reason = reason or _exit_reason(returncode)
            calls, self.calls = self.calls, {}
            self.abandoned.clear()
        for answer in calls.values():
            answer.set_result({"failure": self.reason})
        if not self.ready.done():
            self.ready.set_result({"type": "ready", "error": {"type": "RuntimeError", "message": self.reason}})
        shutil.rmtree(self.home, ignore_errors=True)
        self.owner._stopped(self)

    def _handle(self, message):
        """Dispatch one message from the worker; a message that breaks the protocol raises ValueError."""
        kind = message.get("type") if isinstance(message, dict) else None
        if kind == "ready" and not self.ready.done():
            self.ready.set_result(message)
        elif kind == "result":
            if not _valid_result(message):
                raise ValueError("malformed result")
            with self.lock:
                answer = self.calls.pop(message.get("id"), None)
                late = answer is None and message.get("id") in self.abandoned
                self.abandoned.discard(message.get("id"))
                idle = self.retiring and not self.calls
            if answer is None and not late:
                raise ValueError(f"result for unknown call {message.get('id')!r}")
            if answer is not None:
                answer.set_result(message)
            if idle:
                self.kill()
        elif kind == "call":
            if not _valid_call(message):
                raise ValueError("malformed call")
            threading.Thread(target=self.owner._forward, args=(self, message), daemon=True).start()
        else:
            raise ValueError(f"unexpected message type {kind!r}")


class CatalogWorker:
    """Serve the generated tools of a catalog from a worker process; the router's handle on them."""

    def __init__(self, directory=None, *, client, base_tools=(), environment=(), timeout_seconds=600,
                 memory_gib=MEMORY_GIB, write_files=False, isolation=True, network=True):
        """Copy the catalog, start the first worker and read the tools it loaded.

        Args:
            directory: Catalog directory; defaults to the packaged catalog.
            client: ToolClient for calls from generated tools; it reaches the router itself.
            base_tools: Names of the router's base tools.
            environment: Credential-like variables the worker may inherit.
            timeout_seconds: Wall-clock limit of one call.
            memory_gib: Address-space limit of the worker process, in GiB.
            write_files: Let the worker write files, for example a local image upload.
            isolation: Confine the worker with Linux namespaces; without it, generated
                tools can read every file the router can.
            network: Let a confined worker use the network.

        Raises:
            ValueError: The catalog cannot be loaded.
            RuntimeError: The worker does not start, or isolation is requested on a host
                that does not allow it.
        """
        self.directory = Path(directory or DEFAULT_CATALOG)
        self.files = {path.name: path.read_bytes() for path in catalog_modules(self.directory)}
        self.client = client
        self.refused = set(base_tools) - set(CALLABLE_BASE_TOOLS)
        self.timeout = timeout_seconds
        self.limits = {"memory_bytes": int(memory_gib * (1 << 30)), "file_bytes": None if write_files else 0}
        if not memory_limit_enforced():
            self.limits["memory_bytes"] = None
            logger.warning("%s does not enforce address-space limits: the generated-tool worker runs without "
                           "toolsmith.worker.memory_gib.", sys.platform)
        self.environment = worker_environment(environment)
        self.granted = set(environment)
        self.isolation, self.network = isolation, network
        if isolation:
            from .isolation import available

            confinable, reason = available()
            if not confinable:
                raise RuntimeError(f"Generated tools run confined by Linux user namespaces, which this host does not "
                                   f"allow ({reason}). Set toolsmith.worker.isolation: false to run them without it.")
        else:
            logger.warning("toolsmith.worker.isolation is off: generated tools can read every file the router can "
                           "read and reach the network.")
        withheld = sorted(set(os.environ) - set(self.environment))
        if withheld:
            logger.warning("Generated tools run without these router environment variables: %s. Grant any they "
                           "read with toolsmith.worker.environment.", ", ".join(withheld))
        self.lock = threading.Lock()
        self.start_lock = threading.Lock()
        self.workers = set()
        self.closed = False
        self.tools = self.current = None
        self.current = self._start()
        self.tools = {tool["tool_name"]: tool for tool in self.current.tools}
        atexit.register(self.close)

    @classmethod
    def from_config(cls, config, *, client, base_tools):
        """Build the worker for an agent-system configuration with a toolsmith section."""
        toolsmith = config["toolsmith"]
        settings = toolsmith.get("worker") or {}
        client_timeout = config["services"]["client"].get("timeout_seconds", 600)
        # The worker deadline starts after dispatch; ending it earlier than the agent's HTTP read timeout
        # lets the structured timeout arrive before the client gives up and retries the call.
        timeout = settings.get("timeout_seconds", round(CLIENT_SHARE * client_timeout, 3))
        if timeout >= client_timeout:
            logger.warning("toolsmith.worker.timeout_seconds (%s) is not below services.client.timeout_seconds (%s): "
                           "the agent may time out and retry a generated tool before its error arrives.",
                           timeout, client_timeout)
        return cls(toolsmith.get("catalog"), client=client, base_tools=base_tools,
                   environment=settings.get("environment", ()), timeout_seconds=timeout,
                   memory_gib=settings.get("memory_gib", MEMORY_GIB), write_files=settings.get("write_files", False),
                   isolation=settings.get("isolation", True), network=settings.get("network", True))

    def execute(self, tool_name, args, *, tool_call_id):
        """Run one call in the worker and return the router envelope and its HTTP status."""
        try:
            worker, request, answer = self._submit(tool_name, args)
        except (ValueError, RuntimeError, OSError) as exc:
            return _failure(tool_name, tool_call_id, f"the generated-tool worker is unavailable: {exc}", "WorkerError")
        try:
            outcome = answer.result(timeout=self.timeout)
        except concurrent.futures.TimeoutError:
            outcome = None
        return self._finish(worker, request, answer, outcome, tool_name, tool_call_id)

    async def execute_async(self, tool_name, args, *, tool_call_id):
        """Run one call like execute, awaiting the answer on the event loop instead of in a thread.

        The router uses this, so a generated tool waiting for the calls it makes back
        into the router holds none of the thread-pool workers those calls need.
        """
        loop = asyncio.get_running_loop()
        try:
            worker, request, answer = await loop.run_in_executor(None, self._submit, tool_name, args)
        except (ValueError, RuntimeError, OSError) as exc:
            return _failure(tool_name, tool_call_id, f"the generated-tool worker is unavailable: {exc}", "WorkerError")
        # asyncio.wait, unlike wait_for, does not cancel the future the worker's reader resolves.
        done, _ = await asyncio.wait({asyncio.wrap_future(answer)}, timeout=self.timeout)
        return self._finish(worker, request, answer, answer.result() if done else None, tool_name, tool_call_id)

    def close(self):
        """Stop every worker; later calls fail."""
        with self.lock:
            self.closed = True
            workers, self.current = list(self.workers), None
        for worker in workers:
            worker.kill()
        for worker in workers:
            worker.reader.join(timeout=5)

    def confinement(self, home):
        """The isolation spec of a worker whose private directory is home."""
        from .isolation import python_masks, system_paths

        environment, expand = self.environment, os.path.expanduser
        caches = [environment.get("HF_HOME") or expand("~/.cache/huggingface"), environment.get("HF_HUB_CACHE"),
                  environment.get("HUGGINGFACE_HUB_CACHE"), environment.get("TRANSFORMERS_CACHE"),
                  environment.get("TORCH_HOME") or expand("~/.cache/torch")]
        caches = list(dict.fromkeys(path for path in caches if path and os.path.isdir(path)))
        package_root = os.path.realpath(PACKAGE_ROOT)
        read = system_paths() + [str(home / "catalog")]
        read += [entry for entry in environment.get("PYTHONPATH", "").split(os.pathsep)
                 if entry and os.path.isdir(entry) and os.path.realpath(entry) != package_root]
        # Certificate bundles, and files that granted variables name (a key file, service-account credentials).
        read += [environment[name] for name in ("SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
                                                *sorted(self.granted))
                 if environment.get(name) and os.path.isabs(environment[name]) and os.path.exists(environment[name])]
        write = []
        if self.limits["file_bytes"] is None:
            write += caches
            if environment.get("IMAGE_UPLOAD_PROVIDER", "").lower() == "local":
                # The local image host serves this directory, so uploads must land in the host's copy.
                uploads = environment.get("IMAGE_UPLOAD_LOCAL_DIR") or os.path.join(tempfile.gettempdir(),
                                                                                     "openvlharness-uploads")
                os.makedirs(uploads, mode=0o700, exist_ok=True)
                write.append(uploads)
        else:
            read += caches
        temporary = ["/tmp", "/var/tmp", expand("~")] + ([environment["TMPDIR"]] if environment.get("TMPDIR") else [])
        return {"root": str(home / "root"), "read": read, "write": write, "tmp": temporary,
                "mask": [os.path.join(cache, name) for cache in caches for name in ("token", "stored_tokens")]
                + python_masks(), "cwd": str(home), "network": self.network, "cleanup": str(home)}

    def _submit(self, tool_name, args):
        """Send one call to the worker taking new calls; return the worker, the call ID and its answer."""
        worker = self._worker()
        request, answer = worker.submit(tool_name, args)
        return worker, request, answer

    def _finish(self, worker, request, answer, outcome, tool_name, tool_call_id):
        """Turn the worker's answer into the router envelope; outcome None means the time limit passed."""
        if outcome is None:
            if worker.abandon(request):
                self._retire(worker)
                return _failure(tool_name, tool_call_id, f"{tool_name} did not finish within {self.timeout:g} seconds",
                                "TimeoutError")
            outcome = answer.result()
        if "failure" in outcome:
            return _failure(tool_name, tool_call_id, "the generated-tool worker stopped while "
                            f"{tool_name} was running: {outcome['failure']}", "WorkerError")
        if outcome["ok"]:
            return _build_success_payload(tool_name, tool_call_id, outcome["result"], outcome["meta"]), 200
        return (_build_error_payload(tool_name, tool_call_id, "tool_execution_error", outcome["message"],
                                     {"exception_type": outcome["exception_type"]}),
                400 if outcome["argument_error"] else 500)

    def _start(self):
        """Start a worker and wait until it has loaded the catalog."""
        worker = _Worker(self, self.files, self.environment, self.limits)
        with self.lock:
            self.workers.add(worker)
        try:
            ready = worker.ready.result(timeout=LOAD_SECONDS)
        except concurrent.futures.TimeoutError:
            worker.kill()
            raise RuntimeError(f"the ToolSmith worker did not load the catalog within {LOAD_SECONDS} seconds")
        error = ready.get("error")
        tools = ready.get("tools")
        if error is None and not (isinstance(tools, list) and all(
                isinstance(tool, dict) and isinstance(tool.get("tool_name"), str) for tool in tools)):
            error = {"type": "RuntimeError", "message": "the worker reported no tool list"}
        if error is None and self.tools is not None and [tool["tool_name"] for tool in tools] != list(self.tools):
            error = {"type": "RuntimeError", "message": "a restarted worker loaded different tools"}
        if error is not None:
            worker.kill()
            message = str(error.get("message")).replace(str(worker.catalog), str(self.directory))
            raise (ValueError if error.get("type") == "ValueError" else RuntimeError)(message)
        worker.tools = tools
        return worker

    def _worker(self):
        """Return the worker that takes new calls, starting one if the last has stopped or been retired."""
        with self.start_lock:
            with self.lock:
                if self.closed:
                    raise RuntimeError("the tool router is shutting down")
                worker = self.current
            if worker is not None and not (worker.stopped or worker.retiring):
                return worker
            worker = self._start()
            with self.lock:
                self.current = worker
            return worker

    def _retire(self, worker):
        """Send no new calls to a worker whose call ran past the limit; it stops once its other calls finish."""
        with self.lock:
            if self.current is worker:
                self.current = None
        worker.retire()

    def _stopped(self, worker):
        """Forget a worker whose process has ended."""
        with self.lock:
            self.workers.discard(worker)
            if self.current is worker:
                self.current = None

    def _forward(self, worker, message):
        """Answer one call_tool from the worker: run it through the router, unless the tool is refused."""
        name, call_id = message["tool"], message["tool_call_id"]
        if name in self.refused:
            reply = {"payload": _build_error_payload(name, call_id, "tool_not_allowed",
                                                     f"Generated tools cannot call {name}")}
        else:
            try:
                reply = {"payload": self.client.execute(name, message["args"], tool_call_id=call_id)}
            except Exception as exc:
                reply = {"error": {"module": type(exc).__module__, "type": type(exc).__name__, "message": _text(exc),
                                   "status_code": getattr(exc, "status_code", None)}}
        try:
            worker.send({"type": "reply", "id": message["id"], **reply})
        except (TypeError, ValueError) as exc:
            worker.send({"type": "reply", "id": message["id"], "error": {
                "module": "builtins", "type": "RuntimeError", "message": f"{name}: the reply is not JSON: {exc}"}})


def _rebuild(error):
    """Recreate an exception the router's client raised, so call_tool fails as it does in the router."""
    message = str(error.get("message", ""))
    module, name = error.get("module"), error.get("type")
    if module == "openvlharness.services.http" and name == "HTTPStatusError":
        from ..services.http import HTTPStatusError

        return HTTPStatusError(message, error.get("status_code"))
    if module == "requests.exceptions":
        import requests

        kind = getattr(requests.exceptions, str(name), None)
        if isinstance(kind, type) and issubclass(kind, requests.exceptions.RequestException):
            return kind(message)
    return RuntimeError(message)


class _Router:
    """The client generated tools get in the worker: each call goes to the router process."""

    def __init__(self, channel):
        self.channel = channel
        self.ids = itertools.count(1)
        self.waiting = {}
        self.lock = threading.Lock()

    def execute(self, tool_name, args, *, tool_call_id):
        """Return the router's execute envelope for one call, as ToolClient.execute does.

        Raises:
            TypeError, ValueError: The call cannot be sent as JSON.
            Exception: The router's client failed; the same exception class for
                requests and HTTP status errors, otherwise RuntimeError.
        """
        if not (isinstance(tool_name, str) and isinstance(args, dict) and isinstance(tool_call_id, str)):
            raise TypeError("call_tool needs a tool name and keyword arguments")
        with self.lock:
            call = next(self.ids)
            answer = self.waiting[call] = concurrent.futures.Future()
        try:
            self.channel.send({"type": "call", "id": call, "tool": tool_name, "args": args,
                               "tool_call_id": tool_call_id}, allow_nan=False)
        except BaseException:
            with self.lock:
                self.waiting.pop(call, None)
            raise
        reply = answer.result()
        if "error" in reply:
            raise _rebuild(reply["error"])
        return reply["payload"]

    def answer(self, message):
        """Deliver the router's reply to the waiting call."""
        with self.lock:
            answer = self.waiting.pop(message.get("id"), None)
        if answer is not None:
            answer.set_result(message)


def _run(tools, message, channel):
    """Run one call and send its result; the fields mirror the router's envelope for an in-process tool."""
    reply = {"type": "result", "id": message["id"]}
    try:
        result = tools[message["tool"]].execute(**message["args"])
        extra = {}
        if isinstance(result, ServiceResult):
            extra, result = result.metadata, result.value
        result, meta = _normalize_result_payload(result)
        meta.update(extra)
        reply.update(ok=True, result=result, meta=meta)
    except BaseException as exc:
        reply.update(ok=False, message=_text(exc), exception_type=type(exc).__name__,
                     argument_error=isinstance(exc, ToolArgumentError))
    try:
        channel.send(reply)
    except (TypeError, ValueError) as exc:
        channel.send({"type": "result", "id": message["id"], "ok": False, "message": f"the result is not JSON: {exc}",
                      "exception_type": type(exc).__name__, "argument_error": False})


def _prefer_oom_kill():
    """Ask Linux to stop the worker before other processes when memory runs out; elsewhere do nothing."""
    try:
        with open("/proc/self/oom_score_adj", "w") as stream:
            stream.write("1000")
    except OSError:
        pass


def main(directory, limits):
    """Load the catalog, report its tools, then run calls until the router closes the pipe."""
    channel = _Channel(os.fdopen(os.dup(0), "rb"), os.fdopen(os.dup(1), "wb"))
    # Messages use private copies of the pipes; prints from tools go to the router's stderr.
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.close(null)
    os.dup2(2, 1)
    _prefer_oom_kill()
    limit_resources(**json.loads(limits))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    router = _Router(channel)
    try:
        tools = load_catalog(directory, client=router)
    except Exception as exc:
        channel.send({"type": "ready", "error": {"type": type(exc).__name__, "message": _text(exc)}})
        os._exit(1)
    channel.send({"type": "ready", "tools": [{"tool_name": tool.tool_name, "tool_description": tool.tool_description,
                                              "input_schema": tool.input_schema} for tool in tools.values()]})
    while (message := channel.receive()) is not None:
        if message["type"] == "execute":
            threading.Thread(target=_run, args=(tools, message, channel), daemon=True).start()
        elif message["type"] == "reply":
            router.answer(message)
    # The router has gone without stopping the worker (it was killed): remove the private copy it made.
    home = Path(directory).parent
    if home.name.startswith(HOME_PREFIX):
        shutil.rmtree(home, ignore_errors=True)
    os._exit(0)


if __name__ == "__main__":
    main(*sys.argv[1:])
