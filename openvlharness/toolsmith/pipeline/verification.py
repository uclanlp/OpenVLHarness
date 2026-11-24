"""Phase 4 gates: a static scan and a sandboxed run (Gate 1), then an LLM code review (Gate 2)."""
import ast
import importlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from .prompts import prompt

MODULES = {"re", "json", "datetime"}
# A candidate reaches an allowed module's public API only, not the modules it imports (json.codecs, re.enum).
API = {name: set(importlib.import_module(name).__all__) for name in MODULES}
FORBIDDEN = {"eval", "exec", "compile", "__import__", "open", "input", "breakpoint", "help",
             "globals", "locals", "vars", "getattr", "setattr", "delattr"}
# Attributes that reach frames, code and their globals without a dunder name.
INTROSPECTION = {"gi_frame", "gi_code", "gi_yieldfrom", "cr_frame", "cr_code", "cr_await", "cr_origin", "ag_frame",
                 "ag_code", "ag_await", "f_globals", "f_locals", "f_builtins", "f_back", "f_code", "tb_frame",
                 "tb_next", "co_code", "co_consts"}
# A dunder name as a string, or a str.format field that walks into dunders or frames.
HIDDEN_ACCESS = re.compile(r"^__\w+__$|\{[^{}]*(?:__|\.(?:gi|cr|ag|f|tb|co)_)[^{}]*\}")
AXES = ("parametric_usage", "description_abstractness", "edge_case_handling", "task_alignment")


def scan(source):
    """Gate 1 static check: list dynamic-execution, I/O and structure problems in a candidate.

    The scan narrows what a candidate can express; the sandboxed run (run_once)
    is the boundary that keeps it from the host.
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
        elif isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
            text = node.value.decode("latin-1") if isinstance(node.value, bytes) else node.value
            if HIDDEN_ACCESS.search(text):
                issues.append(f"the string {text[:40]!r} reaches hidden attributes")
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


def run_once(source, timeout, *, isolation=True):
    """Gate 1 runtime check: run a candidate once on dummy input in a separate, resource-limited process.

    Every tool the candidate calls answers with a failure, so the check is cheap
    and reaches no service. With isolation (see openvlharness.toolsmith.isolation)
    the process has no network and sees only the interpreter, the packages and the
    candidate, read-only.

    Args:
        source: Candidate module source.
        timeout: Wall-clock budget in seconds.
        isolation: Confine the run with Linux namespaces.

    Returns:
        (metadata, None) with the tool's name, description and input schema, or
        (None, issue) when the run fails, times out or returns non-JSON data.
    """
    package = Path(__file__).resolve().parents[3]
    # One BLAS thread: the thread pools numpy and scikit-learn start on import would otherwise
    # reserve more than the 2 GiB address-space cap on many-core hosts.
    environment = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(package), "PYTHONDONTWRITEBYTECODE": "1",
                   "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory(prefix="openvlharness-") as root:
        Path(directory, "candidate.py").write_text(source)
        argv = [sys.executable, "-m", "openvlharness.toolsmith.pipeline.sandbox", directory]
        if isolation:
            from ..isolation import command, python_masks, system_paths

            argv = command({"root": root, "read": system_paths() + [directory], "mask": python_masks(),
                            "tmp": ["/tmp", "/var/tmp"], "cwd": "/tmp", "network": False}, argv)
        try:
            done = subprocess.run(argv, cwd=directory, env=environment, capture_output=True, text=True,
                                  timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return None, f"execution exceeded {timeout} seconds"
    if done.returncode:
        lines = done.stderr.strip().splitlines()
        return None, f"execution failed: {lines[-1] if lines else f'exit status {done.returncode}'}"
    return json.loads(done.stdout.strip().splitlines()[-1]), None


def review(critic, question, metadata, source, *, mean, minimum):
    """Gate 2: score a candidate on four 0-5 axes and admit it when both thresholds hold.

    Returns:
        (passed, feedback, scores). An unparsable critic reply is asked once more,
        then counts as a rejection.
    """
    request = prompt("critic.user").format(question=question, name=metadata["tool_name"],
                                           description=metadata["tool_description"],
                                           schema=json.dumps(metadata["input_schema"]), code=source)
    for _ in range(2):
        reply = critic.generate(request, system_prompt=prompt("critic.system"), max_tokens=4096)
        found = re.search(r"\{.*\}", reply or "", re.DOTALL)
        try:
            verdict = json.loads(found.group(0))
            scores = {axis: float(verdict[axis]) for axis in AXES}
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        passed = sum(scores.values()) / len(scores) >= mean and min(scores.values()) >= minimum
        return passed, f"scores {json.dumps(scores)}; {verdict.get('critique', '')}", scores
    return False, "the critic reply could not be parsed", {}
