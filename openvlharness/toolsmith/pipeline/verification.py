"""Phase 4 gates: a static scan and a sandboxed run (Gate 1), then an LLM code review (Gate 2)."""
import ast
import importlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from ..worker import WATCH_SECONDS, TreeMeter
from .prompts import prompt
from .sandbox import MEMORY_BYTES

MODULES = {"re", "json", "datetime"}
# A candidate reaches an allowed module's public API only, not the modules it imports (json.codecs, re.enum).
API = {name: set(importlib.import_module(name).__all__) for name in MODULES}
FORBIDDEN = {"eval", "exec", "compile", "__import__", "open", "input", "breakpoint", "help",
             "globals", "locals", "vars", "getattr", "setattr", "delattr"}
# Attributes that reach frames, code and their globals without a dunder name.
INTROSPECTION = {"gi_frame", "gi_code", "gi_yieldfrom", "cr_frame", "cr_code", "cr_await", "cr_origin", "ag_frame",
                 "ag_code", "ag_await", "f_globals", "f_locals", "f_builtins", "f_back", "f_code", "tb_frame",
                 "tb_next", "co_code", "co_consts"}
# Any double underscore in a string (dunder names are built from pieces), or a str.format field that walks
# into frames.
HIDDEN_ACCESS = re.compile(r"__|\{[^{}]*\.(?:gi|cr|ag|f|tb|co)_[^{}]*\}")
AXES = ("parametric_usage", "description_abstractness", "edge_case_handling", "task_alignment")
# How much of each output stream Gate 1 keeps: the end, which holds the error or the metadata line.
OUTPUT_BYTES = 1 << 20


def _tail(stream, kept, limit):
    """Read a pipe to its end, keeping only its last limit bytes in kept."""
    with stream:
        while chunk := stream.read1(1 << 16):
            kept += chunk
            del kept[:-limit]


def scan(source):
    """Gate 1 static check: list dynamic-execution, I/O and structure problems in a candidate.

    The scan narrows what a candidate can express and is defense in depth only:
    the sandboxed run (run_once) is the boundary that keeps it from the host.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    issues = []
    modules = {alias.asname or alias.name: alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
               for alias in node.names if alias.name in MODULES}
    attribute_values = {id(node.value) for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            issues += [f"import {alias.name} is not allowed" for alias in node.names if alias.name not in MODULES]
        elif isinstance(node, ast.ImportFrom) and node.module == "openvlharness.toolsmith":
            issues += [f"only GeneratedTool and ToolCallError may be imported, not {alias.name}"
                       for alias in node.names if alias.name not in {"GeneratedTool", "ToolCallError"}]
        elif isinstance(node, ast.ImportFrom) and node.module not in MODULES:
            issues.append(f"import from {node.module} is not allowed")
        elif isinstance(node, ast.ImportFrom):
            issues += [f"{node.module} does not export {alias.name}" for alias in node.names
                       if alias.name not in API[node.module]]
        elif isinstance(node, ast.Name) and (node.id in FORBIDDEN or node.id.startswith("__")):
            issues.append(f"{node.id} is not allowed")
        elif isinstance(node, ast.Name) and node.id in modules and id(node) not in attribute_values:
            issues.append(f"module {node.id} may only be used as {node.id}.<name>")
        elif isinstance(node, ast.Attribute) and ((node.attr.startswith("__") and node.attr != "__init__")
                                                   or node.attr in INTROSPECTION):
            issues.append(f"attribute {node.attr} is not allowed")
        elif (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in modules
              and node.attr not in API[modules[node.value.id]]):
            issues.append(f"{node.value.id}.{node.attr} is not allowed")
        elif isinstance(node, ast.Match):
            issues.append("match statements are not allowed")  # class patterns read attributes, dunders included
        elif (isinstance(node, ast.Attribute) and node.attr in {"format", "format_map"}
              and not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))):
            # A format string built at run time can walk into attributes.
            issues.append(f"{node.attr} is allowed only on a string literal")
        elif isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
            text = node.value.decode("latin-1") if isinstance(node.value, bytes) else node.value
            if HIDDEN_ACCESS.search(text):
                issues.append(f"the string {text[:40]!r} may name a hidden attribute")
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and any(isinstance(base, ast.Name) and base.id == "GeneratedTool" for base in node.bases)]
    if len(classes) != 1:
        return issues + [f"expected one GeneratedTool subclass, found {len(classes)}"]
    execute = next((node for node in classes[0].body if isinstance(node, ast.FunctionDef) and node.name == "execute"), None)
    if execute is None:
        return issues + ["the tool does not define execute"]
    names = {node.value.value for node in ast.walk(classes[0]) if isinstance(node, ast.keyword)
             and node.arg == "tool_name" and isinstance(node.value, ast.Constant)}
    if any(isinstance(node, ast.Constant) and node.value in names for node in ast.walk(execute)):
        issues.append("execute calls the tool itself")
    return issues


def declared_name(source):
    """Return the string literal a candidate's tool class passes as tool_name, or None if it passes no single one."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    names = {node.value.value for node in ast.walk(tree) if isinstance(node, ast.keyword) and node.arg == "tool_name"
             and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)}
    computed = [node for node in ast.walk(tree) if isinstance(node, ast.keyword) and node.arg == "tool_name"
                and not isinstance(node.value, ast.Constant)]
    return next(iter(names)) if len(names) == 1 and not computed else None


def run_once(source, timeout, *, isolation=True):
    """Gate 1 runtime check: run a candidate once on dummy input in a separate, resource-limited process.

    Every tool the candidate calls answers with a failure, so the check is cheap
    and reaches no service. With isolation (see openvlharness.toolsmith.isolation)
    the process has no network and sees only the interpreter, the packages and the
    candidate, read-only. Each of its processes has MEMORY_BYTES of address space, and
    on Linux the run is measured every WATCH_SECONDS like a router's worker (see
    worker.TreeMeter) and stopped once its processes, and with isolation its private
    file systems, hold more than MEMORY_BYTES together.

    Args:
        source: Candidate module source.
        timeout: Wall-clock budget in seconds.
        isolation: Confine the run with Linux namespaces.

    Returns:
        (metadata, None) with the tool's name, description and input schema, or
        (None, issue) when the run fails, times out, uses too much memory or returns
        non-JSON data.
    """
    package = Path(__file__).resolve().parents[3]
    # One BLAS thread: the thread pools numpy and scikit-learn start on import would otherwise
    # reserve more than the 2 GiB address-space cap on many-core hosts.
    environment = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(package), "PYTHONDONTWRITEBYTECODE": "1",
                   "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory(prefix="openvlharness-") as root:
        Path(directory, "candidate.py").write_text(source)
        argv = [sys.executable, "-m", "openvlharness.toolsmith.pipeline.sandbox", directory]
        mounts = []
        if isolation:
            from ..isolation import command, private_mounts, python_masks, system_paths

            spec = {"root": root, "read": system_paths() + [directory], "mask": python_masks(),
                    "tmp": ["/tmp", "/var/tmp"], "cwd": "/tmp", "parent": os.getpid()}
            argv, mounts = command(spec, argv), private_mounts(spec)
        # Output is drained as it comes and only its end is kept, so a chatty candidate cannot fill memory.
        # A session of its own, so stopping the run stops what it started too.
        process = subprocess.Popen(argv, cwd=directory, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, start_new_session=True)
        stdout, stderr = bytearray(), bytearray()
        readers = [threading.Thread(target=_tail, args=(stream, kept, OUTPUT_BYTES), daemon=True)
                   for stream, kept in ((process.stdout, stdout), (process.stderr, stderr))]
        for reader in readers:
            reader.start()
        meter = TreeMeter(process.pid, mounts) if os.path.isdir(f"/proc/{process.pid}") else None
        try:
            returncode, used = _wait(process, timeout, meter)
        except subprocess.TimeoutExpired:
            _kill(process)
            process.wait()
            return None, f"execution exceeded {timeout} seconds"
        except BaseException:  # Ctrl-C, say, which no longer reaches the run in its own session
            _kill(process)
            process.wait()
            raise
        finally:
            for reader in readers:
                reader.join(5)
    if used is not None:
        return None, (f"execution used {used / (1 << 30):.2f} GiB of memory in all its processes, more than "
                      f"{MEMORY_BYTES / (1 << 30):g} GiB")
    if returncode:
        lines = stderr.decode("utf-8", "replace").strip().splitlines()
        return None, f"execution failed: {lines[-1] if lines else f'exit status {returncode}'}"
    try:
        # The last line feed ends the metadata line; JSON text may hold other line separators (U+2028).
        return json.loads(stdout.decode("utf-8", "replace").strip().split("\n")[-1]), None
    except (IndexError, ValueError):
        return None, f"execution printed no metadata line within its last {OUTPUT_BYTES} bytes of output"


def _wait(process, timeout, meter):
    """Wait for process for timeout seconds, measuring it with meter every WATCH_SECONDS.

    Returns:
        (exit status, None), or (exit status, bytes used) when the run's memory passed
        MEMORY_BYTES and the process was killed.

    Raises:
        subprocess.TimeoutExpired: The process did not end in time; it still runs.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            return process.wait(timeout=max(0, min(WATCH_SECONDS, deadline - time.monotonic()))), None
        except subprocess.TimeoutExpired:
            if time.monotonic() >= deadline:
                raise
        usage = meter.measure() if meter else None
        if usage and usage[0] > MEMORY_BYTES:
            _kill(process)
            return process.wait(), usage[0]


def _kill(process):
    """Kill a Gate 1 run and the processes of its session."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        pass
    try:
        process.kill()
    except OSError:
        pass


def review(critic, question, metadata, source, *, mean, minimum, max_tokens=12800):
    """Gate 2: score a candidate on four 0-5 axes and admit it when both thresholds hold.

    Returns:
        (passed, feedback, scores).

    Raises:
        IncompleteReply: A reply ran out of tokens or holds no text.
        RuntimeError: Two replies in a row held no valid scores: each axis an integer
            from 0 to 5, the critic prompt's scale. Either way the critic
            failed, not the candidate: the gap stays undecided for a later run.
    """
    request = prompt("critic.user").format(question=question, name=metadata["tool_name"],
                                           description=metadata["tool_description"],
                                           schema=json.dumps(metadata["input_schema"]), code=source)
    for _ in range(2):
        reply = critic.generate(request, system_prompt=prompt("critic.system"), max_tokens=max_tokens,
                                require_complete=True)
        found = re.search(r"\{.*\}", reply or "", re.DOTALL)
        try:
            verdict = json.loads(found.group(0))
            scores = {axis: _score(verdict[axis]) for axis in AXES}
        except (AttributeError, KeyError, TypeError, ValueError):
            continue  # no verdict, or a score off the scale ([10, 1, 1, 1] would pass on its mean): ask again
        passed = sum(scores.values()) / len(scores) >= mean and min(scores.values()) >= minimum
        return passed, f"scores {json.dumps(scores)}; {verdict.get('critique', '')}", scores
    raise RuntimeError("the critic's replies held no valid scores (integers from 0 to 5)")


def _score(value):
    """A critic score as a float, if it is an integer from 0 to 5.

    Raises:
        ValueError: It is not.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 5 or value != int(value):
        raise ValueError(f"a critic score must be an integer from 0 to 5, not {value!r}")
    return float(value)
